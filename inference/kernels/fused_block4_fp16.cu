// =============================================================================
// fused_block4_fp16.cu - COLIDE Project
// Dense classification head with FP16 half2: 128 -> 64 ReLU -> 5 logits
// Tests whether FP16 benefits launch-overhead-bound small kernels
// Compile: nvcc -arch=sm_86 -o fused_block4_fp16 fused_block4_fp16.cu
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

constexpr int INPUT_DIM  = 128;
constexpr int HIDDEN_DIM = 64;
constexpr int NUM_CLASSES = 5;

// ---------------------------------------------------------------------------
// FP16 half2 fused dense kernel
// ---------------------------------------------------------------------------
__global__ void fused_dense_fp16_kernel(
    const half* __restrict__ input,        // [INPUT_DIM]
    const half2* __restrict__ fc1_weight,  // (HIDDEN_DIM, INPUT_DIM/2) packed
    const half* __restrict__ fc1_bias,     // [HIDDEN_DIM]
    const half2* __restrict__ fc2_weight,  // (NUM_CLASSES, HIDDEN_DIM/2) packed
    const half* __restrict__ fc2_bias,     // [NUM_CLASSES]
    float* __restrict__ output             // [NUM_CLASSES] logits in FP32
) {
    __shared__ half s_input[INPUT_DIM];
    __shared__ half s_hidden[HIDDEN_DIM];

    // 1. Load input into shared memory
    for (int i = threadIdx.x; i < INPUT_DIM; i += blockDim.x) {
        s_input[i] = input[i];
    }
    __syncthreads();

    // 2. First dense layer: 128 -> 64 with ReLU (half2 dot product)
    if (threadIdx.x < HIDDEN_DIM) {
        half2 sum = __float2half2_rn(0.0f);
        const half2* input_h2 = (const half2*)s_input;
        const int half2_per_row = INPUT_DIM / 2;  // 64

        for (int j = 0; j < half2_per_row; ++j) {
            half2 w = fc1_weight[threadIdx.x * half2_per_row + j];
            half2 x = input_h2[j];
            sum = __hfma2(w, x, sum);
        }

        // Reduce half2 to float, add bias, ReLU
        float val = __half2float(sum.x) + __half2float(sum.y) + __half2float(fc1_bias[threadIdx.x]);
        val = fmaxf(val, 0.0f);
        s_hidden[threadIdx.x] = __float2half(val);
    }
    __syncthreads();

    // 3. Second dense layer: 64 -> 5
    if (threadIdx.x < NUM_CLASSES) {
        half2 sum = __float2half2_rn(0.0f);
        const half2* hidden_h2 = (const half2*)s_hidden;
        const int half2_per_row = HIDDEN_DIM / 2;  // 32

        for (int j = 0; j < half2_per_row; ++j) {
            half2 w = fc2_weight[threadIdx.x * half2_per_row + j];
            half2 h = hidden_h2[j];
            sum = __hfma2(w, h, sum);
        }

        float val = __half2float(sum.x) + __half2float(sum.y) + __half2float(fc2_bias[threadIdx.x]);
        output[threadIdx.x] = val;
    }
}

// ---------------------------------------------------------------------------
// FP32 reference kernel (same as original)
// ---------------------------------------------------------------------------
__global__ void fused_dense_fp32_kernel(
    const float* __restrict__ input,
    const float* __restrict__ fc1_weight,
    const float* __restrict__ fc1_bias,
    const float* __restrict__ fc2_weight,
    const float* __restrict__ fc2_bias,
    float* __restrict__ output
) {
    __shared__ float s_input[INPUT_DIM];
    __shared__ float s_hidden[HIDDEN_DIM];

    for (int i = threadIdx.x; i < INPUT_DIM; i += blockDim.x) {
        s_input[i] = input[i];
    }
    __syncthreads();

    if (threadIdx.x < HIDDEN_DIM) {
        float sum = fc1_bias[threadIdx.x];
        for (int j = 0; j < INPUT_DIM; ++j) {
            sum += fc1_weight[threadIdx.x * INPUT_DIM + j] * s_input[j];
        }
        s_hidden[threadIdx.x] = fmaxf(sum, 0.0f);
    }
    __syncthreads();

    if (threadIdx.x < NUM_CLASSES) {
        float sum = fc2_bias[threadIdx.x];
        for (int j = 0; j < HIDDEN_DIM; ++j) {
            sum += fc2_weight[threadIdx.x * HIDDEN_DIM + j] * s_hidden[j];
        }
        output[threadIdx.x] = sum;
    }
}

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------
void cpu_dense(const std::vector<float>& input,
    const std::vector<float>& fc1_w, const std::vector<float>& fc1_b,
    const std::vector<float>& fc2_w, const std::vector<float>& fc2_b,
    std::vector<float>& output) {
    std::vector<float> hidden(HIDDEN_DIM);
    for (int i = 0; i < HIDDEN_DIM; ++i) {
        float sum = fc1_b[i];
        for (int j = 0; j < INPUT_DIM; ++j)
            sum += fc1_w[i * INPUT_DIM + j] * input[j];
        hidden[i] = fmaxf(sum, 0.0f);
    }
    output.assign(NUM_CLASSES, 0.0f);
    for (int i = 0; i < NUM_CLASSES; ++i) {
        float sum = fc2_b[i];
        for (int j = 0; j < HIDDEN_DIM; ++j)
            sum += fc2_w[i * HIDDEN_DIM + j] * hidden[j];
        output[i] = sum;
    }
}

int main() {
    std::cout << "=== COLIDE Block4 FP16 vs FP32 Comparison ===\n";
    srand(42);
    auto randf = [](){ return (float)rand() / RAND_MAX - 0.5f; };

    // Generate random data
    std::vector<float> h_input(INPUT_DIM);
    std::vector<float> fc1_w(HIDDEN_DIM * INPUT_DIM);
    std::vector<float> fc1_b(HIDDEN_DIM);
    std::vector<float> fc2_w(NUM_CLASSES * HIDDEN_DIM);
    std::vector<float> fc2_b(NUM_CLASSES);
    for (auto& v : h_input) v = randf();
    for (auto& v : fc1_w) v = randf();
    for (auto& v : fc1_b) v = randf();
    for (auto& v : fc2_w) v = randf();
    for (auto& v : fc2_b) v = randf();

    // CPU reference
    std::vector<float> cpu_out;
    cpu_dense(h_input, fc1_w, fc1_b, fc2_w, fc2_b, cpu_out);

    // Convert to half for FP16
    std::vector<half> h_input_h(INPUT_DIM);
    std::vector<half> fc1_b_h(HIDDEN_DIM);
    std::vector<half> fc2_b_h(NUM_CLASSES);
    for (int i = 0; i < INPUT_DIM; ++i) h_input_h[i] = __float2half(h_input[i]);
    for (int i = 0; i < HIDDEN_DIM; ++i) fc1_b_h[i] = __float2half(fc1_b[i]);
    for (int i = 0; i < NUM_CLASSES; ++i) fc2_b_h[i] = __float2half(fc2_b[i]);

    // Pack weights into half2
    std::vector<half2> fc1_w_h2(HIDDEN_DIM * INPUT_DIM / 2);
    for (int i = 0; i < HIDDEN_DIM; ++i)
        for (int j = 0; j < INPUT_DIM / 2; ++j)
            fc1_w_h2[i * (INPUT_DIM/2) + j] = __floats2half2_rn(
                fc1_w[i * INPUT_DIM + 2*j], fc1_w[i * INPUT_DIM + 2*j + 1]);

    std::vector<half2> fc2_w_h2(NUM_CLASSES * HIDDEN_DIM / 2);
    for (int i = 0; i < NUM_CLASSES; ++i)
        for (int j = 0; j < HIDDEN_DIM / 2; ++j)
            fc2_w_h2[i * (HIDDEN_DIM/2) + j] = __floats2half2_rn(
                fc2_w[i * HIDDEN_DIM + 2*j], fc2_w[i * HIDDEN_DIM + 2*j + 1]);

    // ===== FP32 GPU =====
    float *d_input, *d_fc1_w, *d_fc1_b, *d_fc2_w, *d_fc2_b, *d_output;
    cudaMalloc(&d_input, INPUT_DIM * sizeof(float));
    cudaMalloc(&d_fc1_w, fc1_w.size() * sizeof(float));
    cudaMalloc(&d_fc1_b, fc1_b.size() * sizeof(float));
    cudaMalloc(&d_fc2_w, fc2_w.size() * sizeof(float));
    cudaMalloc(&d_fc2_b, fc2_b.size() * sizeof(float));
    cudaMalloc(&d_output, NUM_CLASSES * sizeof(float));

    cudaMemcpy(d_input, h_input.data(), INPUT_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc1_w, fc1_w.data(), fc1_w.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc1_b, fc1_b.data(), fc1_b.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc2_w, fc2_w.data(), fc2_w.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc2_b, fc2_b.data(), fc2_b.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Validate FP32
    fused_dense_fp32_kernel<<<1, 64>>>(d_input, d_fc1_w, d_fc1_b, d_fc2_w, d_fc2_b, d_output);
    cudaDeviceSynchronize();
    std::vector<float> gpu_fp32(NUM_CLASSES);
    cudaMemcpy(gpu_fp32.data(), d_output, NUM_CLASSES * sizeof(float), cudaMemcpyDeviceToHost);

    bool fp32_pass = true;
    for (int i = 0; i < NUM_CLASSES; ++i)
        if (fabs(gpu_fp32[i] - cpu_out[i]) > 1e-3) { fp32_pass = false; break; }
    std::cout << (fp32_pass ? "✅ FP32 validation PASSED\n" : "❌ FP32 validation FAILED\n");

    // Timing FP32
    for (int i = 0; i < 100; ++i)
        fused_dense_fp32_kernel<<<1, 64>>>(d_input, d_fc1_w, d_fc1_b, d_fc2_w, d_fc2_b, d_output);
    cudaDeviceSynchronize();

    const int iters = 10000;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i)
        fused_dense_fp32_kernel<<<1, 64>>>(d_input, d_fc1_w, d_fc1_b, d_fc2_w, d_fc2_b, d_output);
    cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();
    double fp32_us = std::chrono::duration<double, std::micro>(t2 - t1).count() / iters;

    // ===== FP16 GPU =====
    half *d_input_h, *d_fc1_b_h, *d_fc2_b_h;
    half2 *d_fc1_w_h2, *d_fc2_w_h2;
    float *d_output_fp16;

    cudaMalloc(&d_input_h, INPUT_DIM * sizeof(half));
    cudaMalloc(&d_fc1_w_h2, fc1_w_h2.size() * sizeof(half2));
    cudaMalloc(&d_fc1_b_h, HIDDEN_DIM * sizeof(half));
    cudaMalloc(&d_fc2_w_h2, fc2_w_h2.size() * sizeof(half2));
    cudaMalloc(&d_fc2_b_h, NUM_CLASSES * sizeof(half));
    cudaMalloc(&d_output_fp16, NUM_CLASSES * sizeof(float));

    cudaMemcpy(d_input_h, h_input_h.data(), INPUT_DIM * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc1_w_h2, fc1_w_h2.data(), fc1_w_h2.size() * sizeof(half2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc1_b_h, fc1_b_h.data(), HIDDEN_DIM * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc2_w_h2, fc2_w_h2.data(), fc2_w_h2.size() * sizeof(half2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc2_b_h, fc2_b_h.data(), NUM_CLASSES * sizeof(half), cudaMemcpyHostToDevice);

    // Validate FP16
    fused_dense_fp16_kernel<<<1, 64>>>(d_input_h, d_fc1_w_h2, d_fc1_b_h, d_fc2_w_h2, d_fc2_b_h, d_output_fp16);
    cudaDeviceSynchronize();
    std::vector<float> gpu_fp16(NUM_CLASSES);
    cudaMemcpy(gpu_fp16.data(), d_output_fp16, NUM_CLASSES * sizeof(float), cudaMemcpyDeviceToHost);

    bool fp16_pass = true;
    for (int i = 0; i < NUM_CLASSES; ++i)
        if (fabs(gpu_fp16[i] - cpu_out[i]) > 5e-2) { fp16_pass = false; break; }
    std::cout << (fp16_pass ? "✅ FP16 validation PASSED" : "❌ FP16 validation FAILED");
    std::cout << " (tolerance 5e-2)\n";

    // Timing FP16
    for (int i = 0; i < 100; ++i)
        fused_dense_fp16_kernel<<<1, 64>>>(d_input_h, d_fc1_w_h2, d_fc1_b_h, d_fc2_w_h2, d_fc2_b_h, d_output_fp16);
    cudaDeviceSynchronize();

    auto t3 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i)
        fused_dense_fp16_kernel<<<1, 64>>>(d_input_h, d_fc1_w_h2, d_fc1_b_h, d_fc2_w_h2, d_fc2_b_h, d_output_fp16);
    cudaDeviceSynchronize();
    auto t4 = std::chrono::high_resolution_clock::now();
    double fp16_us = std::chrono::duration<double, std::micro>(t4 - t3).count() / iters;

    // Results
    std::cout << "\n⏱️  Block4 FP32: " << fp32_us << " µs\n";
    std::cout << "⏱️  Block4 FP16: " << fp16_us << " µs\n";
    std::cout << "   Speedup FP16/FP32: " << fp32_us / fp16_us << "x\n";
    std::cout << "   PyTorch GPU target: 122.1 µs\n";

    if (fp16_us >= fp32_us * 0.95) {
        std::cout << "\n📊 Finding: FP16 does NOT improve Block 4.\n";
        std::cout << "   Block 4 is launch-overhead-bound, not compute-bound.\n";
        std::cout << "   FP16 benefits only compute-bound kernels (Block 3 BiLSTM).\n";
    } else {
        std::cout << "\n📊 Finding: FP16 improves Block 4 by " << fp32_us / fp16_us << "x\n";
    }

    // Cleanup
    cudaFree(d_input); cudaFree(d_fc1_w); cudaFree(d_fc1_b);
    cudaFree(d_fc2_w); cudaFree(d_fc2_b); cudaFree(d_output);
    cudaFree(d_input_h); cudaFree(d_fc1_w_h2); cudaFree(d_fc1_b_h);
    cudaFree(d_fc2_w_h2); cudaFree(d_fc2_b_h); cudaFree(d_output_fp16);

    return 0;
}
