// =============================================================================
// fused_block3_fp16.cu - COLIDE Project
// BiLSTM with native FP16 half2 compute for W_hh dot products
// Compile: nvcc -arch=sm_86 -o fused_block3_fp16 fused_block3_fp16.cu
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

constexpr int SEQ      = 16;
constexpr int IN_CH    = 128;
constexpr int H1       = 128;
constexpr int H1x2     = 256;
constexpr int H2       = 64;
constexpr int OUT_SIZE  = 128;

// ---------------------------------------------------------------------------
// Repack transposed W_hh from (H, 4H) FP32 to (H, 2H) half2 paired layout
// Input:  w_t[j * 4H + k] where k=0..H-1 is i_gate, k=H..2H-1 is f_gate, etc.
// Output: w_h2[j * 2H + h] = half2(i_weight, f_weight) for h=0..H-1
//         w_h2[j * 2H + H + h] = half2(g_weight, o_weight) for h=0..H-1
// ---------------------------------------------------------------------------
__global__ void repack_whh_to_half2(
    const float* __restrict__ w_t,    // (H, 4H) transposed FP32
    __half2* __restrict__ w_h2,       // (H, 2H) half2 paired
    int hidden_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = hidden_size * hidden_size;  // H * H pairs to create
    if (idx >= total) return;

    int j = idx / hidden_size;  // row (input hidden unit)
    int h = idx % hidden_size;  // column (output hidden unit)
    int four_h = 4 * hidden_size;
    int two_h  = 2 * hidden_size;

    // Read 4 gate weights from transposed FP32
    float i_w = w_t[j * four_h + h];
    float f_w = w_t[j * four_h + hidden_size + h];
    float g_w = w_t[j * four_h + 2 * hidden_size + h];
    float o_w = w_t[j * four_h + 3 * hidden_size + h];

    // Pack into half2 pairs
    w_h2[j * two_h + h]               = __halves2half2(__float2half(i_w), __float2half(f_w));
    w_h2[j * two_h + hidden_size + h] = __halves2half2(__float2half(g_w), __float2half(o_w));
}

// ---------------------------------------------------------------------------
// Transpose kernel (FP32, for creating transposed W_hh first)
// ---------------------------------------------------------------------------
__global__ void transpose_kernel(
    const float* __restrict__ in, float* __restrict__ out,
    int rows, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    int r = idx / cols, c = idx % cols;
    out[c * rows + r] = in[r * cols + c];
}

// ---------------------------------------------------------------------------
// FP16 W_ih conversion
// ---------------------------------------------------------------------------
__global__ void fp32_to_fp16_kernel(const float* in, __half* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = __float2half(in[idx]);
}

// ---------------------------------------------------------------------------
// Linear projection with FP16 weights
// ---------------------------------------------------------------------------
__global__ void linear_proj_fp16_kernel(
    const float* __restrict__ in,
    const __half* __restrict__ w,
    float* __restrict__ out,
    int input_size, int out_rows, int seq_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_rows * seq_len) return;
    int r = idx / seq_len, t = idx % seq_len;
    float sum = 0.0f;
    for (int c = 0; c < input_size; ++c)
        sum += __half2float(w[r * input_size + c]) * in[c * seq_len + t];
    out[idx] = sum;
}

// ---------------------------------------------------------------------------
// LSTM recurrent kernel with native half2 FMA
// ---------------------------------------------------------------------------
__global__ void lstm_recurrent_fp16_kernel(
    const float* __restrict__ gate_ih_all,
    const __half2* __restrict__ w_hh_h2,  // (H, 2H) half2 paired
    const float* __restrict__ bias_ih,
    const float* __restrict__ bias_hh,
    float* __restrict__ output_hidden,
    int hidden_size, int seq_len, bool reverse
) {
    extern __shared__ __half s_h_prev[];
    int h = threadIdx.x;
    if (h >= hidden_size) return;

    s_h_prev[h] = __float2half(0.0f);
    __syncthreads();

    float c = 0.0f;
    int two_h = 2 * hidden_size;

    for (int t = 0; t < seq_len; ++t) {
        int pos = reverse ? (seq_len - 1 - t) : t;

        float i_gate = gate_ih_all[h*seq_len+pos] + bias_ih[h] + bias_hh[h];
        float f_gate = gate_ih_all[(hidden_size+h)*seq_len+pos] + bias_ih[hidden_size+h] + bias_hh[hidden_size+h];
        float g_gate = gate_ih_all[(2*hidden_size+h)*seq_len+pos] + bias_ih[2*hidden_size+h] + bias_hh[2*hidden_size+h];
        float o_gate = gate_ih_all[(3*hidden_size+h)*seq_len+pos] + bias_ih[3*hidden_size+h] + bias_hh[3*hidden_size+h];

        // Native FP16 half2 accumulation
        __half2 acc_if = __float2half2_rn(0.0f);
        __half2 acc_go = __float2half2_rn(0.0f);

        for (int j = 0; j < hidden_size; ++j) {
            __half prev_h = s_h_prev[j];
            __half2 prev_h2 = __half2half2(prev_h);

            int base = j * two_h;
            __half2 if_pair = w_hh_h2[base + h];
            __half2 go_pair = w_hh_h2[base + hidden_size + h];

            acc_if = __hfma2(if_pair, prev_h2, acc_if);
            acc_go = __hfma2(go_pair, prev_h2, acc_go);
        }

        // Extract FP16 results to FP32
        i_gate += __low2float(acc_if);
        f_gate += __high2float(acc_if);
        g_gate += __low2float(acc_go);
        o_gate += __high2float(acc_go);

        float i_val = 1.0f/(1.0f+expf(-i_gate));
        float f_val = 1.0f/(1.0f+expf(-f_gate));
        float g_val = tanhf(g_gate);
        float o_val = 1.0f/(1.0f+expf(-o_gate));

        c = f_val*c + i_val*g_val;
        float h_new = o_val * tanhf(c);
        output_hidden[h*seq_len + t] = h_new;
        s_h_prev[h] = __float2half(h_new);
        __syncthreads();
    }
}

__global__ void combine_kernel(
    const float* __restrict__ fw, const float* __restrict__ rev,
    float* __restrict__ out, int hidden_size, int seq_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = 2 * hidden_size * seq_len;
    if (idx >= total) return;
    int half_n = hidden_size * seq_len;
    out[idx] = (idx < half_n) ? fw[idx] : rev[idx - half_n];
}

__global__ void extract_last_timestep_kernel(
    const float* __restrict__ fw, const float* __restrict__ rev,
    int hidden, int seq_len, float* __restrict__ out
) {
    int i = threadIdx.x;
    if (i < hidden) {
        out[i] = fw[i*seq_len + (seq_len-1)];
        out[i+hidden] = rev[i*seq_len + (seq_len-1)];
    }
}

void lstm_direction_fp16(
    cudaStream_t stream,
    const float* d_input, float* d_output_hidden,
    const __half* d_w_ih_fp16, const __half2* d_w_hh_h2,
    const float* d_bias_ih, const float* d_bias_hh,
    int input_size, int hidden_size, int seq_len, bool reverse,
    float* d_gate_ih_all
) {
    int out_rows = 4*hidden_size;
    int total = out_rows*seq_len;
    int threads = 256;
    int blocks = (total+threads-1)/threads;
    linear_proj_fp16_kernel<<<blocks, threads, 0, stream>>>(
        d_input, d_w_ih_fp16, d_gate_ih_all, input_size, out_rows, seq_len);
    int smem = hidden_size * sizeof(__half);
    lstm_recurrent_fp16_kernel<<<1, hidden_size, smem, stream>>>(
        d_gate_ih_all, d_w_hh_h2, d_bias_ih, d_bias_hh,
        d_output_hidden, hidden_size, seq_len, reverse);
}

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------
void cpu_lstm_forward(
    const std::vector<float>& input, int input_size, int hidden_size, int seq_len,
    const std::vector<float>& w_ih, const std::vector<float>& w_hh,
    const std::vector<float>& b_ih, const std::vector<float>& b_hh,
    std::vector<float>& output_h, bool reverse)
{
    output_h.assign(hidden_size*seq_len, 0.0f);
    std::vector<float> h_prev(hidden_size, 0.0f), c_prev(hidden_size, 0.0f);
    for (int t = 0; t < seq_len; ++t) {
        int pos = reverse ? (seq_len-1-t) : t;
        std::vector<float> h_new(hidden_size), c_new(hidden_size);
        for (int h = 0; h < hidden_size; ++h) {
            float i_g=b_ih[h]+b_hh[h], f_g=b_ih[hidden_size+h]+b_hh[hidden_size+h],
                  g_g=b_ih[2*hidden_size+h]+b_hh[2*hidden_size+h], o_g=b_ih[3*hidden_size+h]+b_hh[3*hidden_size+h];
            for (int f=0; f<input_size; ++f) {
                float x=input[f*seq_len+pos];
                i_g+=w_ih[h*input_size+f]*x; f_g+=w_ih[(hidden_size+h)*input_size+f]*x;
                g_g+=w_ih[(2*hidden_size+h)*input_size+f]*x; o_g+=w_ih[(3*hidden_size+h)*input_size+f]*x;
            }
            for (int j=0; j<hidden_size; ++j) {
                i_g+=w_hh[h*hidden_size+j]*h_prev[j]; f_g+=w_hh[(hidden_size+h)*hidden_size+j]*h_prev[j];
                g_g+=w_hh[(2*hidden_size+h)*hidden_size+j]*h_prev[j]; o_g+=w_hh[(3*hidden_size+h)*hidden_size+j]*h_prev[j];
            }
            float i_val=1.0f/(1.0f+expf(-i_g)), f_val=1.0f/(1.0f+expf(-f_g)),
                  g_val=tanhf(g_g), o_val=1.0f/(1.0f+expf(-o_g));
            c_new[h]=f_val*c_prev[h]+i_val*g_val;
            h_new[h]=o_val*tanhf(c_new[h]);
            output_h[h*seq_len+t]=h_new[h];
        }
        h_prev=h_new; c_prev=c_new;
    }
}

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
    std::vector<float> in2(H1x2*SEQ);
    for (int t=0;t<SEQ;++t) {
        for (int i=0;i<H1;++i) in2[i*SEQ+t]=h1_fw[i*SEQ+t];
        for (int i=0;i<H1;++i) in2[(i+H1)*SEQ+t]=h1_rev[i*SEQ+t];
    }
    std::vector<float> h2_fw, h2_rev;
    cpu_lstm_forward(in2, H1x2, H2, SEQ, w_ih2_f, w_hh2_f, b_ih2_f, b_hh2_f, h2_fw, false);
    cpu_lstm_forward(in2, H1x2, H2, SEQ, w_ih2_r, w_hh2_r, b_ih2_r, b_hh2_r, h2_rev, true);
    std::vector<float> out(OUT_SIZE);
    for (int i=0;i<H2;++i) out[i]=h2_fw[i*SEQ+SEQ-1];
    for (int i=0;i<H2;++i) out[i+H2]=h2_rev[i*SEQ+SEQ-1];
    return out;
}

int main() {
    std::cout << "=== COLIDE Block3 FP16 Native half2 ===\n";
    srand(42);
    auto randf = [](){ return (float)rand()/RAND_MAX - 0.5f; };

    std::vector<float> h_input(IN_CH*SEQ);
    for (auto& v : h_input) v = randf();

    std::vector<float> w_ih1_f(4*H1*IN_CH), w_hh1_f(4*H1*H1), b_ih1_f(4*H1), b_hh1_f(4*H1);
    for (auto& v:w_ih1_f) v=randf(); for (auto& v:w_hh1_f) v=randf();
    for (auto& v:b_ih1_f) v=randf(); for (auto& v:b_hh1_f) v=randf();
    auto w_ih1_r=w_ih1_f, w_hh1_r=w_hh1_f, b_ih1_r=b_ih1_f, b_hh1_r=b_hh1_f;

    std::vector<float> w_ih2_f(4*H2*H1x2), w_hh2_f(4*H2*H2), b_ih2_f(4*H2), b_hh2_f(4*H2);
    for (auto& v:w_ih2_f) v=randf(); for (auto& v:w_hh2_f) v=randf();
    for (auto& v:b_ih2_f) v=randf(); for (auto& v:b_hh2_f) v=randf();
    auto w_ih2_r=w_ih2_f, w_hh2_r=w_hh2_f, b_ih2_r=b_ih2_f, b_hh2_r=b_hh2_f;

    auto cpu_out = cpu_pipeline(h_input,
        w_ih1_f, w_hh1_f, b_ih1_f, b_hh1_f, w_ih1_r, w_hh1_r, b_ih1_r, b_hh1_r,
        w_ih2_f, w_hh2_f, b_ih2_f, b_hh2_f, w_ih2_r, w_hh2_r, b_ih2_r, b_hh2_r);

    // Upload FP32 weights
    float *d_input;
    cudaMalloc(&d_input, h_input.size()*sizeof(float));
    cudaMemcpy(d_input, h_input.data(), h_input.size()*sizeof(float), cudaMemcpyHostToDevice);

    auto copy_vec = [](float*& d, const std::vector<float>& v) {
        cudaMalloc(&d, v.size()*sizeof(float));
        cudaMemcpy(d, v.data(), v.size()*sizeof(float), cudaMemcpyHostToDevice);
    };
    float *d_w_ih1_f,*d_w_hh1_f,*d_b_ih1_f,*d_b_hh1_f;
    float *d_w_ih1_r,*d_w_hh1_r,*d_b_ih1_r,*d_b_hh1_r;
    float *d_w_ih2_f,*d_w_hh2_f,*d_b_ih2_f,*d_b_hh2_f;
    float *d_w_ih2_r,*d_w_hh2_r,*d_b_ih2_r,*d_b_hh2_r;
    copy_vec(d_w_ih1_f,w_ih1_f); copy_vec(d_w_hh1_f,w_hh1_f);
    copy_vec(d_b_ih1_f,b_ih1_f); copy_vec(d_b_hh1_f,b_hh1_f);
    copy_vec(d_w_ih1_r,w_ih1_r); copy_vec(d_w_hh1_r,w_hh1_r);
    copy_vec(d_b_ih1_r,b_ih1_r); copy_vec(d_b_hh1_r,b_hh1_r);
    copy_vec(d_w_ih2_f,w_ih2_f); copy_vec(d_w_hh2_f,w_hh2_f);
    copy_vec(d_b_ih2_f,b_ih2_f); copy_vec(d_b_hh2_f,b_hh2_f);
    copy_vec(d_w_ih2_r,w_ih2_r); copy_vec(d_w_hh2_r,w_hh2_r);
    copy_vec(d_b_ih2_r,b_ih2_r); copy_vec(d_b_hh2_r,b_hh2_r);

    int thr = 256;

    // Step 1: Transpose W_hh to (H, 4H)
    float *d_w_hh1_f_t, *d_w_hh1_r_t, *d_w_hh2_f_t, *d_w_hh2_r_t;
    int n1=4*H1*H1, n2=4*H2*H2;
    cudaMalloc(&d_w_hh1_f_t, n1*sizeof(float));
    cudaMalloc(&d_w_hh1_r_t, n1*sizeof(float));
    cudaMalloc(&d_w_hh2_f_t, n2*sizeof(float));
    cudaMalloc(&d_w_hh2_r_t, n2*sizeof(float));
    transpose_kernel<<<(n1+thr-1)/thr,thr>>>(d_w_hh1_f, d_w_hh1_f_t, 4*H1, H1);
    transpose_kernel<<<(n1+thr-1)/thr,thr>>>(d_w_hh1_r, d_w_hh1_r_t, 4*H1, H1);
    transpose_kernel<<<(n2+thr-1)/thr,thr>>>(d_w_hh2_f, d_w_hh2_f_t, 4*H2, H2);
    transpose_kernel<<<(n2+thr-1)/thr,thr>>>(d_w_hh2_r, d_w_hh2_r_t, 4*H2, H2);

    // Step 2: Repack transposed W_hh to half2 paired layout
    __half2 *d_w_hh1_f_h2, *d_w_hh1_r_h2, *d_w_hh2_f_h2, *d_w_hh2_r_h2;
    cudaMalloc(&d_w_hh1_f_h2, H1*2*H1*sizeof(__half2));
    cudaMalloc(&d_w_hh1_r_h2, H1*2*H1*sizeof(__half2));
    cudaMalloc(&d_w_hh2_f_h2, H2*2*H2*sizeof(__half2));
    cudaMalloc(&d_w_hh2_r_h2, H2*2*H2*sizeof(__half2));
    repack_whh_to_half2<<<(H1*H1+thr-1)/thr,thr>>>(d_w_hh1_f_t, d_w_hh1_f_h2, H1);
    repack_whh_to_half2<<<(H1*H1+thr-1)/thr,thr>>>(d_w_hh1_r_t, d_w_hh1_r_h2, H1);
    repack_whh_to_half2<<<(H2*H2+thr-1)/thr,thr>>>(d_w_hh2_f_t, d_w_hh2_f_h2, H2);
    repack_whh_to_half2<<<(H2*H2+thr-1)/thr,thr>>>(d_w_hh2_r_t, d_w_hh2_r_h2, H2);

    // Step 3: Convert W_ih to FP16
    __half *d_w_ih1_f_fp16, *d_w_ih1_r_fp16, *d_w_ih2_f_fp16, *d_w_ih2_r_fp16;
    cudaMalloc(&d_w_ih1_f_fp16, 4*H1*IN_CH*sizeof(__half));
    cudaMalloc(&d_w_ih1_r_fp16, 4*H1*IN_CH*sizeof(__half));
    cudaMalloc(&d_w_ih2_f_fp16, 4*H2*H1x2*sizeof(__half));
    cudaMalloc(&d_w_ih2_r_fp16, 4*H2*H1x2*sizeof(__half));
    fp32_to_fp16_kernel<<<(4*H1*IN_CH+thr-1)/thr,thr>>>(d_w_ih1_f, d_w_ih1_f_fp16, 4*H1*IN_CH);
    fp32_to_fp16_kernel<<<(4*H1*IN_CH+thr-1)/thr,thr>>>(d_w_ih1_r, d_w_ih1_r_fp16, 4*H1*IN_CH);
    fp32_to_fp16_kernel<<<(4*H2*H1x2+thr-1)/thr,thr>>>(d_w_ih2_f, d_w_ih2_f_fp16, 4*H2*H1x2);
    fp32_to_fp16_kernel<<<(4*H2*H1x2+thr-1)/thr,thr>>>(d_w_ih2_r, d_w_ih2_r_fp16, 4*H2*H1x2);
    cudaDeviceSynchronize();

    // Output and scratch buffers
    float *d_h1_fw, *d_h1_rev, *d_in2, *d_h2_fw, *d_h2_rev, *d_out;
    cudaMalloc(&d_h1_fw, H1*SEQ*sizeof(float));
    cudaMalloc(&d_h1_rev, H1*SEQ*sizeof(float));
    cudaMalloc(&d_in2, H1x2*SEQ*sizeof(float));
    cudaMalloc(&d_h2_fw, H2*SEQ*sizeof(float));
    cudaMalloc(&d_h2_rev, H2*SEQ*sizeof(float));
    cudaMalloc(&d_out, OUT_SIZE*sizeof(float));

    float *d_gate1_fw, *d_gate1_rev, *d_gate2_fw, *d_gate2_rev;
    cudaMalloc(&d_gate1_fw, 4*H1*SEQ*sizeof(float));
    cudaMalloc(&d_gate1_rev, 4*H1*SEQ*sizeof(float));
    cudaMalloc(&d_gate2_fw, 4*H2*SEQ*sizeof(float));
    cudaMalloc(&d_gate2_rev, 4*H2*SEQ*sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto launch_pipeline = [&](cudaStream_t s) {
        lstm_direction_fp16(s, d_input, d_h1_fw, d_w_ih1_f_fp16, d_w_hh1_f_h2, d_b_ih1_f, d_b_hh1_f, IN_CH, H1, SEQ, false, d_gate1_fw);
        lstm_direction_fp16(s, d_input, d_h1_rev, d_w_ih1_r_fp16, d_w_hh1_r_h2, d_b_ih1_r, d_b_hh1_r, IN_CH, H1, SEQ, true, d_gate1_rev);
        int cb=(H1x2*SEQ+255)/256;
        combine_kernel<<<cb,256,0,s>>>(d_h1_fw, d_h1_rev, d_in2, H1, SEQ);
        lstm_direction_fp16(s, d_in2, d_h2_fw, d_w_ih2_f_fp16, d_w_hh2_f_h2, d_b_ih2_f, d_b_hh2_f, H1x2, H2, SEQ, false, d_gate2_fw);
        lstm_direction_fp16(s, d_in2, d_h2_rev, d_w_ih2_r_fp16, d_w_hh2_r_h2, d_b_ih2_r, d_b_hh2_r, H1x2, H2, SEQ, true, d_gate2_rev);
        extract_last_timestep_kernel<<<1,H2,0,s>>>(d_h2_fw, d_h2_rev, H2, SEQ, d_out);
    };

    // Warmup
    launch_pipeline(stream);
    cudaStreamSynchronize(stream);

    // Validate
    std::vector<float> gpu_out(OUT_SIZE);
    cudaMemcpy(gpu_out.data(), d_out, OUT_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
    bool pass = true;
    for (int i = 0; i < OUT_SIZE; ++i) {
        if (fabs(gpu_out[i] - cpu_out[i]) > 5e-2) {
            std::cout << "Mismatch at " << i << ": GPU " << gpu_out[i] << " CPU " << cpu_out[i] << "\n";
            pass = false; break;
        }
    }
    std::cout << (pass ? "✅ FP16 half2 validation PASSED\n" : "❌ FP16 validation FAILED\n");

    // Timing
    const int iters = 100;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        launch_pipeline(stream);
        cudaStreamSynchronize(stream);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double fp16_us = std::chrono::duration<double, std::micro>(end-start).count()/iters;

    std::cout << "⏱️  Block3 FP16 half2:    " << fp16_us << " µs\n";
    std::cout << "   Block3 FP32:          1007.3 µs\n";
    std::cout << "   Speedup FP16/FP32:    " << 1007.3/fp16_us << "x\n";
    std::cout << "   PyTorch GPU target:   784.1 µs (n=50-trial mean, see benchmark_pytorch_block3_stats.py)\n";

    // Cleanup
    cudaStreamDestroy(stream);
    cudaFree(d_input);
    cudaFree(d_h1_fw); cudaFree(d_h1_rev); cudaFree(d_in2);
    cudaFree(d_h2_fw); cudaFree(d_h2_rev); cudaFree(d_out);
    cudaFree(d_gate1_fw); cudaFree(d_gate1_rev);
    cudaFree(d_gate2_fw); cudaFree(d_gate2_rev);
    cudaFree(d_w_ih1_f); cudaFree(d_w_hh1_f); cudaFree(d_b_ih1_f); cudaFree(d_b_hh1_f);
    cudaFree(d_w_ih1_r); cudaFree(d_w_hh1_r); cudaFree(d_b_ih1_r); cudaFree(d_b_hh1_r);
    cudaFree(d_w_ih2_f); cudaFree(d_w_hh2_f); cudaFree(d_b_ih2_f); cudaFree(d_b_hh2_f);
    cudaFree(d_w_ih2_r); cudaFree(d_w_hh2_r); cudaFree(d_b_ih2_r); cudaFree(d_b_hh2_r);
    cudaFree(d_w_hh1_f_t); cudaFree(d_w_hh1_r_t);
    cudaFree(d_w_hh2_f_t); cudaFree(d_w_hh2_r_t);
    cudaFree(d_w_hh1_f_h2); cudaFree(d_w_hh1_r_h2);
    cudaFree(d_w_hh2_f_h2); cudaFree(d_w_hh2_r_h2);
    cudaFree(d_w_ih1_f_fp16); cudaFree(d_w_ih1_r_fp16);
    cudaFree(d_w_ih2_f_fp16); cudaFree(d_w_ih2_r_fp16);

    return 0;
}
