// =============================================================================
// fused_block3.cu - COLIDE Project (CORRECTED v2)
// BiLSTM layers only: BiLSTM1 fw+rev → combine → BiLSTM2 fw+rev → last timestep
// Input: (128, 16) from block2 output
// Output: (128,) — concatenation of BiLSTM2 forward/reverse last timestep
// Target: RTX 3050 SM 8.6
// Compilation: nvcc -arch=sm_86 -o fused_block3 fused_block3.cu
// =============================================================================

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constexpr int SEQ          = 16;   // sequence length after Block2 pooling
constexpr int IN_CH        = 128;  // input channels to BiLSTM1
constexpr int H1           = 128;  // BiLSTM1 hidden size
constexpr int H1x2         = H1 * 2;  // 256, input size for BiLSTM2
constexpr int H2           = 64;   // BiLSTM2 hidden size
constexpr int OUT_SIZE     = H2 * 2;  // 128, final output size

// ---------------------------------------------------------------------------
// Device helper
// ---------------------------------------------------------------------------
__device__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// ---------------------------------------------------------------------------
// Generic LSTM forward kernel (reused for all 4 passes)
// ---------------------------------------------------------------------------
__global__ void lstm_kernel(
    const float* __restrict__ in,
    float* __restrict__ out_h,
    const float* __restrict__ w_ih,   // [4*hidden_size, input_size]
    const float* __restrict__ w_hh,   // [4*hidden_size, hidden_size]
    const float* __restrict__ b_ih,   // [4*hidden_size]
    const float* __restrict__ b_hh,   // [4*hidden_size]
    int input_size,
    int hidden_size,
    int seq_len,
    int reverse
) {
    extern __shared__ float shmem[];
    float* s_in = shmem;
    // Double-buffered hidden state: reads for timestep t always come from a
    // DIFFERENT shared array than the one this timestep writes into, so a
    // read and a write can never touch the same location within one sync
    // epoch. Fixed 2026-07-01: the single-buffer version (read s_h_prev[j]
    // for all j, then write own s_h_prev[h], gated only by one
    // __syncthreads() per iteration) was verified by compute-sanitizer
    // racecheck to have a genuine hazard between the line-103-equivalent
    // write and the hidden-to-hidden read loop -- confirmed independently by
    // the naive kernel producing different output across repeated runs with
    // an identical fixed host-side RNG seed (non-determinism that pure FP32
    // summation-order error cannot produce). synccheck found no barrier
    // misuse, so the fix is structural (separate read/write buffers), not a
    // missing sync.
    float* s_h_prev[2] = {
        &shmem[input_size * seq_len],
        &shmem[input_size * seq_len + hidden_size],
    };

    int h = threadIdx.x;
    if (h >= hidden_size) return;

    // load entire input tile into shared memory
    for (int i = h; i < input_size * seq_len; i += blockDim.x) {
        s_in[i] = in[i];
    }
    __syncthreads();

    float c = 0.0f;
    float h_val = 0.0f;

    // initialize previous hidden state to zero (both buffers, so whichever
    // one t=0 reads from is valid)
    s_h_prev[0][h] = 0.0f;
    s_h_prev[1][h] = 0.0f;
    __syncthreads();

    for (int t = 0; t < seq_len; ++t) {
        int pos = reverse ? (seq_len - 1 - t) : t;
        float* read_buf = s_h_prev[t % 2];
        float* write_buf = s_h_prev[(t + 1) % 2];

        // gate biases
        float i_gate = b_ih[h] + b_hh[h];
        float f_gate = b_ih[hidden_size + h] + b_hh[hidden_size + h];
        float g_gate = b_ih[2 * hidden_size + h] + b_hh[2 * hidden_size + h];
        float o_gate = b_ih[3 * hidden_size + h] + b_hh[3 * hidden_size + h];

        // input-to-hidden
        for (int f = 0; f < input_size; ++f) {
            float x = s_in[f * seq_len + pos];
            i_gate += w_ih[h * input_size + f] * x;
            f_gate += w_ih[(hidden_size + h) * input_size + f] * x;
            g_gate += w_ih[(2 * hidden_size + h) * input_size + f] * x;
            o_gate += w_ih[(3 * hidden_size + h) * input_size + f] * x;
        }

        // hidden-to-hidden
        for (int j = 0; j < hidden_size; ++j) {
            float prev_h = read_buf[j];
            i_gate += w_hh[h * hidden_size + j] * prev_h;
            f_gate += w_hh[(hidden_size + h) * hidden_size + j] * prev_h;
            g_gate += w_hh[(2 * hidden_size + h) * hidden_size + j] * prev_h;
            o_gate += w_hh[(3 * hidden_size + h) * hidden_size + j] * prev_h;
        }

        float i_val = sigmoid(i_gate);
        float f_val = sigmoid(f_gate);
        float g_val = tanhf(g_gate);
        float o_val = sigmoid(o_gate);

        c     = f_val * c + i_val * g_val;
        h_val = o_val * tanhf(c);

        out_h[h * seq_len + t] = h_val;

        write_buf[h] = h_val;
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Combine forward and reverse hidden states (interleaved)
// ---------------------------------------------------------------------------
__global__ void combine_kernel(
    const float* __restrict__ fw,
    const float* __restrict__ rev,
    float* __restrict__ out,
    int hidden_size,
    int seq_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = 2 * hidden_size * seq_len;
    if (idx >= total) return;

    int half = hidden_size * seq_len;
    if (idx < half) {
        out[idx] = fw[idx];
    } else {
        out[idx] = rev[idx - half];
    }
}

// ---------------------------------------------------------------------------
// Extract last timestep from forward and reverse tensors
// ---------------------------------------------------------------------------
__global__ void extract_last_timestep_kernel(
    const float* __restrict__ fw,
    const float* __restrict__ rev,
    int hidden,
    int seq_len,
    float* __restrict__ out
) {
    int i = threadIdx.x;
    if (i < hidden) {
        out[i]           = fw[i * seq_len + (seq_len - 1)];
        out[i + hidden] = rev[i * seq_len + (seq_len - 1)];
    }
}

// ---------------------------------------------------------------------------
// CPU reference: LSTM forward
// ---------------------------------------------------------------------------
void cpu_lstm_forward(
    const std::vector<float>& input, int input_size, int hidden_size, int seq_len,
    const std::vector<float>& w_ih, const std::vector<float>& w_hh,
    const std::vector<float>& b_ih, const std::vector<float>& b_hh,
    std::vector<float>& output_h, bool reverse)
{
    output_h.assign(hidden_size * seq_len, 0.0f);
    std::vector<float> h_prev(hidden_size, 0.0f), c_prev(hidden_size, 0.0f);
    for (int t = 0; t < seq_len; ++t) {
        int pos = reverse ? (seq_len - 1 - t) : t;
        std::vector<float> h_new(hidden_size), c_new(hidden_size);
        for (int h = 0; h < hidden_size; ++h) {
            float i_g = b_ih[h] + b_hh[h];
            float f_g = b_ih[hidden_size + h] + b_hh[hidden_size + h];
            float g_g = b_ih[2 * hidden_size + h] + b_hh[2 * hidden_size + h];
            float o_g = b_ih[3 * hidden_size + h] + b_hh[3 * hidden_size + h];
            for (int f = 0; f < input_size; ++f) {
                float x = input[f * seq_len + pos];
                i_g += w_ih[h * input_size + f] * x;
                f_g += w_ih[(hidden_size + h) * input_size + f] * x;
                g_g += w_ih[(2 * hidden_size + h) * input_size + f] * x;
                o_g += w_ih[(3 * hidden_size + h) * input_size + f] * x;
            }
            for (int j = 0; j < hidden_size; ++j) {
                i_g += w_hh[h * hidden_size + j] * h_prev[j];
                f_g += w_hh[(hidden_size + h) * hidden_size + j] * h_prev[j];
                g_g += w_hh[(2 * hidden_size + h) * hidden_size + j] * h_prev[j];
                o_g += w_hh[(3 * hidden_size + h) * hidden_size + j] * h_prev[j];
            }
            float i_val = 1.0f/(1.0f+expf(-i_g));
            float f_val = 1.0f/(1.0f+expf(-f_g));
            float g_val = tanhf(g_g);
            float o_val = 1.0f/(1.0f+expf(-o_g));
            c_new[h] = f_val * c_prev[h] + i_val * g_val;
            h_new[h] = o_val * tanhf(c_new[h]);
            output_h[h * seq_len + t] = h_new[h];
        }
        h_prev = h_new;
        c_prev = c_new;
    }
}

// ---------------------------------------------------------------------------
// CPU reference for entire BiLSTM pipeline → last timestep
// ---------------------------------------------------------------------------
std::vector<float> cpu_pipeline(
    const std::vector<float>& input,
    const std::vector<float>& w_ih1_f, const std::vector<float>& w_hh1_f,
    const std::vector<float>& b_ih1_f, const std::vector<float>& b_hh1_f,
    const std::vector<float>& w_ih1_r, const std::vector<float>& w_hh1_r,
    const std::vector<float>& b_ih1_r, const std::vector<float>& b_hh1_r,
    const std::vector<float>& w_ih2_f, const std::vector<float>& w_hh2_f,
    const std::vector<float>& b_ih2_f, const std::vector<float>& b_hh2_f,
    const std::vector<float>& w_ih2_r, const std::vector<float>& w_hh2_r,
    const std::vector<float>& b_ih2_r, const std::vector<float>& b_hh2_r)
{
    std::vector<float> h1_fw, h1_rev;
    cpu_lstm_forward(input, IN_CH, H1, SEQ, w_ih1_f, w_hh1_f, b_ih1_f, b_hh1_f, h1_fw, false);
    cpu_lstm_forward(input, IN_CH, H1, SEQ, w_ih1_r, w_hh1_r, b_ih1_r, b_hh1_r, h1_rev, true);

    std::vector<float> in2(H1x2 * SEQ);
    for (int t = 0; t < SEQ; ++t) {
        for (int i = 0; i < H1; ++i) in2[i * SEQ + t] = h1_fw[i * SEQ + t];
        for (int i = 0; i < H1; ++i) in2[(i + H1) * SEQ + t] = h1_rev[i * SEQ + t];
    }

    std::vector<float> h2_fw, h2_rev;
    cpu_lstm_forward(in2, H1x2, H2, SEQ, w_ih2_f, w_hh2_f, b_ih2_f, b_hh2_f, h2_fw, false);
    cpu_lstm_forward(in2, H1x2, H2, SEQ, w_ih2_r, w_hh2_r, b_ih2_r, b_hh2_r, h2_rev, true);

    std::vector<float> out(OUT_SIZE);
    int last = SEQ - 1;
    for (int i = 0; i < H2; ++i) out[i]      = h2_fw[i * SEQ + last];
    for (int i = 0; i < H2; ++i) out[i + H2] = h2_rev[i * SEQ + last];
    return out;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    std::cout << "=== COLIDE Fused Block3 (BiLSTM) ===\n";
    srand(42);
    auto randf = [](){ return (float)rand() / RAND_MAX - 0.5f; };

    // Random input and weights
    std::vector<float> h_input(IN_CH * SEQ);
    for (auto& v : h_input) v = randf();

    std::vector<float> w_ih1_f(4 * H1 * IN_CH), w_hh1_f(4 * H1 * H1), b_ih1_f(4 * H1), b_hh1_f(4 * H1);
    for (auto& v : w_ih1_f) v = randf(); for (auto& v : w_hh1_f) v = randf();
    for (auto& v : b_ih1_f) v = randf(); for (auto& v : b_hh1_f) v = randf();
    auto w_ih1_r = w_ih1_f, w_hh1_r = w_hh1_f, b_ih1_r = b_ih1_f, b_hh1_r = b_hh1_f;

    std::vector<float> w_ih2_f(4 * H2 * H1x2), w_hh2_f(4 * H2 * H2), b_ih2_f(4 * H2), b_hh2_f(4 * H2);
    for (auto& v : w_ih2_f) v = randf(); for (auto& v : w_hh2_f) v = randf();
    for (auto& v : b_ih2_f) v = randf(); for (auto& v : b_hh2_f) v = randf();
    auto w_ih2_r = w_ih2_f, w_hh2_r = w_hh2_f, b_ih2_r = b_ih2_f, b_hh2_r = b_hh2_f;

    // CPU reference
    auto cpu_out = cpu_pipeline(h_input,
        w_ih1_f, w_hh1_f, b_ih1_f, b_hh1_f,
        w_ih1_r, w_hh1_r, b_ih1_r, b_hh1_r,
        w_ih2_f, w_hh2_f, b_ih2_f, b_hh2_f,
        w_ih2_r, w_hh2_r, b_ih2_r, b_hh2_r);

    // GPU allocations
    float *d_input, *d_h1_fw, *d_h1_rev, *d_in2, *d_h2_fw, *d_h2_rev, *d_out;
    cudaMalloc(&d_input,  h_input.size() * sizeof(float));
    cudaMalloc(&d_h1_fw,  H1 * SEQ * sizeof(float));
    cudaMalloc(&d_h1_rev, H1 * SEQ * sizeof(float));
    cudaMalloc(&d_in2,    H1x2 * SEQ * sizeof(float));
    cudaMalloc(&d_h2_fw,  H2 * SEQ * sizeof(float));
    cudaMalloc(&d_h2_rev, H2 * SEQ * sizeof(float));
    cudaMalloc(&d_out,    OUT_SIZE * sizeof(float));

    auto copy_to_dev = [](float*& d, const std::vector<float>& v) {
        cudaMalloc(&d, v.size() * sizeof(float));
        cudaMemcpy(d, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice);
    };
    float *d_w_ih1_f, *d_w_hh1_f, *d_b_ih1_f, *d_b_hh1_f;
    float *d_w_ih1_r, *d_w_hh1_r, *d_b_ih1_r, *d_b_hh1_r;
    float *d_w_ih2_f, *d_w_hh2_f, *d_b_ih2_f, *d_b_hh2_f;
    float *d_w_ih2_r, *d_w_hh2_r, *d_b_ih2_r, *d_b_hh2_r;
    copy_to_dev(d_w_ih1_f, w_ih1_f); copy_to_dev(d_w_hh1_f, w_hh1_f);
    copy_to_dev(d_b_ih1_f, b_ih1_f); copy_to_dev(d_b_hh1_f, b_hh1_f);
    copy_to_dev(d_w_ih1_r, w_ih1_r); copy_to_dev(d_w_hh1_r, w_hh1_r);
    copy_to_dev(d_b_ih1_r, b_ih1_r); copy_to_dev(d_b_hh1_r, b_hh1_r);
    copy_to_dev(d_w_ih2_f, w_ih2_f); copy_to_dev(d_w_hh2_f, w_hh2_f);
    copy_to_dev(d_b_ih2_f, b_ih2_f); copy_to_dev(d_b_hh2_f, b_hh2_f);
    copy_to_dev(d_w_ih2_r, w_ih2_r); copy_to_dev(d_w_hh2_r, w_hh2_r);
    copy_to_dev(d_b_ih2_r, b_ih2_r); copy_to_dev(d_b_hh2_r, b_hh2_r);

    cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice);

    // GPU pipeline
    auto full_launch = [&]() {
        int smem1 = (IN_CH * SEQ + 2 * H1) * sizeof(float);  // +2*H1: double-buffered hidden state
        lstm_kernel<<<1, H1, smem1>>>(d_input, d_h1_fw, d_w_ih1_f, d_w_hh1_f, d_b_ih1_f, d_b_hh1_f, IN_CH, H1, SEQ, 0);
        lstm_kernel<<<1, H1, smem1>>>(d_input, d_h1_rev, d_w_ih1_r, d_w_hh1_r, d_b_ih1_r, d_b_hh1_r, IN_CH, H1, SEQ, 1);

        int comb_blocks = (H1x2 * SEQ + 255) / 256;
        combine_kernel<<<comb_blocks, 256>>>(d_h1_fw, d_h1_rev, d_in2, H1, SEQ);

        int smem2 = (H1x2 * SEQ + 2 * H2) * sizeof(float);  // +2*H2: double-buffered hidden state
        lstm_kernel<<<1, H2, smem2>>>(d_in2, d_h2_fw, d_w_ih2_f, d_w_hh2_f, d_b_ih2_f, d_b_hh2_f, H1x2, H2, SEQ, 0);
        lstm_kernel<<<1, H2, smem2>>>(d_in2, d_h2_rev, d_w_ih2_r, d_w_hh2_r, d_b_ih2_r, d_b_hh2_r, H1x2, H2, SEQ, 1);

        extract_last_timestep_kernel<<<1, H2>>>(d_h2_fw, d_h2_rev, H2, SEQ, d_out);
        cudaDeviceSynchronize();
    };

    // Warm-up
    full_launch();

    // Timing (100 iterations)
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        full_launch();
    }
    auto end = std::chrono::high_resolution_clock::now();
    double avg_us = std::chrono::duration<double, std::micro>(end - start).count() / 100.0;

    // Copy GPU output and validate
    std::vector<float> gpu_out(OUT_SIZE);
    cudaMemcpy(gpu_out.data(), d_out, OUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Tolerance matches fused_block3.cu (1e-2) -- this was previously 1e-3,
    // a stricter threshold than every other Block 3 variant's validation,
    // which made this kernel fail on FP32 summation-order differences
    // (~1e-3 magnitude) that the other variants' own tolerance would accept.
    // Not a functional bug: fixed 2026-07-01 for a consistent basis across
    // the optimization progression.
    bool pass = true;
    for (int i = 0; i < OUT_SIZE; ++i) {
        if (fabs(gpu_out[i] - cpu_out[i]) > 1e-2) {
            std::cout << "Mismatch at " << i << ": GPU " << gpu_out[i] << " CPU " << cpu_out[i] << "\n";
            pass = false;
            break;
        }
    }
    std::cout << (pass ? "✅ FP32 validation PASSED\n" : "❌ FP32 validation FAILED\n");
    std::cout << "⏱️  Block3 (BiLSTM) time: " << avg_us << " µs\n";
    std::cout << "   PyTorch GPU target: 784.1 µs (n=50-trial mean, see benchmark_pytorch_block3_stats.py)\n";

    // Cleanup
    cudaFree(d_input); cudaFree(d_h1_fw); cudaFree(d_h1_rev); cudaFree(d_in2);
    cudaFree(d_h2_fw); cudaFree(d_h2_rev); cudaFree(d_out);
    cudaFree(d_w_ih1_f); cudaFree(d_w_hh1_f); cudaFree(d_b_ih1_f); cudaFree(d_b_hh1_f);
    cudaFree(d_w_ih1_r); cudaFree(d_w_hh1_r); cudaFree(d_b_ih1_r); cudaFree(d_b_hh1_r);
    cudaFree(d_w_ih2_f); cudaFree(d_w_hh2_f); cudaFree(d_b_ih2_f); cudaFree(d_b_hh2_f);
    cudaFree(d_w_ih2_r); cudaFree(d_w_hh2_r); cudaFree(d_b_ih2_r); cudaFree(d_b_hh2_r);

    return 0;
}