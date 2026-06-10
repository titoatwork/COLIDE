// =============================================================================
// fused_block1.cu - COLIDE Project
// Fused kernel: Linear Projection + Reshape + Conv1D + BatchNorm + ReLU
// Target: NVIDIA GeForce RTX 3050 (Ampere SM 8.6)
// Compilation: nvcc -arch=sm_86 -o fused_block1 fused_block1.cu
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
constexpr int INPUT_FEATURES = 10;
constexpr int PROJ_OUT = 64;              // output size of linear projection
constexpr int IN_CHANNELS = 2;            // after reshape
constexpr int SEQ_LEN = 32;               // after reshape
constexpr int OUT_CHANNELS = 64;
constexpr int KERNEL_SIZE = 3;
constexpr int PADDING = 1;
constexpr float BN_EPSILON = 1e-5f;

// ---------------------------------------------------------------------------
// FP32 fused kernel
// ---------------------------------------------------------------------------
__global__ void fused_block1_fp32(
    const float* __restrict__ input,          // [10]
    const float* __restrict__ proj_weight,    // [64][10]
    const float* __restrict__ proj_bias,      // [64]
    const float* __restrict__ conv_weight,    // [64][2][3]
    const float* __restrict__ conv_bias,      // [64]
    const float* __restrict__ bn_weight,      // [64] gamma
    const float* __restrict__ bn_bias,        // [64] beta
    const float* __restrict__ bn_mean,        // [64] running_mean
    const float* __restrict__ bn_var,         // [64] running_var
    float* __restrict__ output                // [64 * 32]
) {
    // Grid: 2 blocks of 1024 threads → total 2048 threads
    // Each thread computes one (channel, position) output element.
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= OUT_CHANNELS * SEQ_LEN) return;

    // --- Step 1: Linear projection (64 threads cooperatively) ---
    __shared__ float proj_shared[PROJ_OUT];
    if (threadIdx.x < PROJ_OUT) {
        float sum = proj_bias[threadIdx.x];
        for (int i = 0; i < INPUT_FEATURES; ++i) {
            sum += proj_weight[threadIdx.x * INPUT_FEATURES + i] * input[i];
        }
        proj_shared[threadIdx.x] = sum;
    }
    __syncthreads();

    // --- Step 2: 1D convolution + BatchNorm + ReLU ---
    int c = tid / SEQ_LEN;          // output channel (0..63)
    int p = tid % SEQ_LEN;          // spatial position (0..31)

    float accum = conv_bias[c];
    for (int ic = 0; ic < IN_CHANNELS; ++ic) {
        for (int k = 0; k < KERNEL_SIZE; ++k) {
            int pos = p + k - PADDING;
            float val = 0.0f;
            if (pos >= 0 && pos < SEQ_LEN) {
                val = proj_shared[ic * SEQ_LEN + pos];
            }
            accum += conv_weight[c * (IN_CHANNELS * KERNEL_SIZE) + ic * KERNEL_SIZE + k] * val;
        }
    }

    // BatchNorm
    float inv_std = rsqrtf(bn_var[c] + BN_EPSILON);
    float bn_out = bn_weight[c] * (accum - bn_mean[c]) * inv_std + bn_bias[c];

    // ReLU
    output[tid] = fmaxf(bn_out, 0.0f);
}

// ---------------------------------------------------------------------------
// FP16 fused kernel (uses half2 for weights to reduce memory traffic)
// ---------------------------------------------------------------------------
__global__ void fused_block1_fp16(
    const float* __restrict__ input_f32,      // still float input
    const __half* __restrict__ proj_weight,   // [64][10] half
    const __half* __restrict__ proj_bias,     // [64] half
    const __half* __restrict__ conv_weight,   // [64][2][3] half
    const __half* __restrict__ conv_bias,     // [64] half
    const __half* __restrict__ bn_weight,     // [64] half
    const __half* __restrict__ bn_bias,       // [64] half
    const __half* __restrict__ bn_mean,       // [64] half
    const __half* __restrict__ bn_var,        // [64] half
    __half* __restrict__ output               // [64*32] half
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= OUT_CHANNELS * SEQ_LEN) return;

    // --- Linear projection in half ---
    __shared__ __half proj_shared[PROJ_OUT];
    if (threadIdx.x < PROJ_OUT) {
        half sum = proj_bias[threadIdx.x];
        for (int i = 0; i < INPUT_FEATURES; ++i) {
            sum = __hfma(proj_weight[threadIdx.x * INPUT_FEATURES + i],
                         __float2half(input_f32[i]), sum);
        }
        proj_shared[threadIdx.x] = sum;
    }
    __syncthreads();

    // --- Convolution ---
    int c = tid / SEQ_LEN;
    int p = tid % SEQ_LEN;
    half accum = conv_bias[c];
    for (int ic = 0; ic < IN_CHANNELS; ++ic) {
        for (int k = 0; k < KERNEL_SIZE; ++k) {
            int pos = p + k - PADDING;
            half val = 0.0f;
            if (pos >= 0 && pos < SEQ_LEN) {
                val = proj_shared[ic * SEQ_LEN + pos];
            }
            accum = __hfma(
                conv_weight[c * (IN_CHANNELS * KERNEL_SIZE) + ic * KERNEL_SIZE + k],
                val, accum);
        }
    }

    // BatchNorm in half
    half inv_std = hrsqrt(bn_var[c] + __float2half(BN_EPSILON));
    half bn_out = __hfma(__hmul(bn_weight[c], __hsub(accum, bn_mean[c])), inv_std, bn_bias[c]);

    // ReLU
    output[tid] = __hgt(bn_out, __float2half(0.0f)) ? bn_out : __float2half(0.0f);
}

// ---------------------------------------------------------------------------
// CPU reference (for validation) - same operations, FP32
// ---------------------------------------------------------------------------
void cpu_reference(
    const std::vector<float>& input,
    const std::vector<float>& proj_weight,
    const std::vector<float>& proj_bias,
    const std::vector<float>& conv_weight,
    const std::vector<float>& conv_bias,
    const std::vector<float>& bn_weight,
    const std::vector<float>& bn_bias,
    const std::vector<float>& bn_mean,
    const std::vector<float>& bn_var,
    std::vector<float>& output)
{
    // Linear projection
    std::vector<float> proj(PROJ_OUT);
    for (int i = 0; i < PROJ_OUT; ++i) {
        float s = proj_bias[i];
        for (int j = 0; j < INPUT_FEATURES; ++j)
            s += proj_weight[i * INPUT_FEATURES + j] * input[j];
        proj[i] = s;
    }

    // Conv1D + BN + ReLU
    for (int c = 0; c < OUT_CHANNELS; ++c) {
        for (int p = 0; p < SEQ_LEN; ++p) {
            float sum = conv_bias[c];
            for (int ic = 0; ic < IN_CHANNELS; ++ic) {
                for (int k = 0; k < KERNEL_SIZE; ++k) {
                    int pos = p + k - PADDING;
                    float val = (pos >= 0 && pos < SEQ_LEN) ? proj[ic * SEQ_LEN + pos] : 0.0f;
                    sum += conv_weight[c * (IN_CHANNELS * KERNEL_SIZE) + ic * KERNEL_SIZE + k] * val;
                }
            }
            float bn = bn_weight[c] * (sum - bn_mean[c]) / std::sqrt(bn_var[c] + BN_EPSILON) + bn_bias[c];
            output[c * SEQ_LEN + p] = std::max(0.0f, bn);
        }
    }
}

// ---------------------------------------------------------------------------
// Timing helper
// ---------------------------------------------------------------------------
template<typename Func>
double time_ms(Func f, int iterations = 1000) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) f();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;
    return ms.count() / iterations;
}

// ---------------------------------------------------------------------------
// Main test harness
// ---------------------------------------------------------------------------
int main() {
    std::cout << "=== COLIDE Fused Block1 Kernel ===\n";

    // --- Allocate and initialize random weights (FP32) ---
    std::vector<float> input(INPUT_FEATURES);
    std::vector<float> proj_weight(PROJ_OUT * INPUT_FEATURES);
    std::vector<float> proj_bias(PROJ_OUT);
    std::vector<float> conv_weight(OUT_CHANNELS * IN_CHANNELS * KERNEL_SIZE);
    std::vector<float> conv_bias(OUT_CHANNELS);
    std::vector<float> bn_weight(OUT_CHANNELS);
    std::vector<float> bn_bias(OUT_CHANNELS);
    std::vector<float> bn_mean(OUT_CHANNELS);
    std::vector<float> bn_var(OUT_CHANNELS);
    // Fill with random but reproduceable values
    srand(42);
    auto randf = [](){ return (float)rand() / RAND_MAX - 0.5f; };
    for (auto& x : input) x = randf();
    for (auto& x : proj_weight) x = randf();
    for (auto& x : proj_bias) x = randf();
    for (auto& x : conv_weight) x = randf();
    for (auto& x : conv_bias) x = randf();
    for (auto& x : bn_weight) x = randf() + 1.0f; // gamma > 0
    for (auto& x : bn_bias) x = randf();
    for (auto& x : bn_mean) x = randf();
    for (auto& x : bn_var) x = fabs(randf()) + 1.0f; // variance positive

    // --- CPU reference ---
    std::vector<float> cpu_out(OUT_CHANNELS * SEQ_LEN);
    cpu_reference(input, proj_weight, proj_bias, conv_weight, conv_bias,
                  bn_weight, bn_bias, bn_mean, bn_var, cpu_out);

    // --- GPU FP32 execution ---
    // Allocate device memory
    float *d_input, *d_proj_weight, *d_proj_bias, *d_conv_weight, *d_conv_bias;
    float *d_bn_weight, *d_bn_bias, *d_bn_mean, *d_bn_var, *d_output;
    cudaMalloc(&d_input, INPUT_FEATURES * sizeof(float));
    cudaMalloc(&d_proj_weight, PROJ_OUT * INPUT_FEATURES * sizeof(float));
    cudaMalloc(&d_proj_bias, PROJ_OUT * sizeof(float));
    cudaMalloc(&d_conv_weight, OUT_CHANNELS * IN_CHANNELS * KERNEL_SIZE * sizeof(float));
    cudaMalloc(&d_conv_bias, OUT_CHANNELS * sizeof(float));
    cudaMalloc(&d_bn_weight, OUT_CHANNELS * sizeof(float));
    cudaMalloc(&d_bn_bias, OUT_CHANNELS * sizeof(float));
    cudaMalloc(&d_bn_mean, OUT_CHANNELS * sizeof(float));
    cudaMalloc(&d_bn_var, OUT_CHANNELS * sizeof(float));
    cudaMalloc(&d_output, OUT_CHANNELS * SEQ_LEN * sizeof(float));

    // Copy weights to device
    cudaMemcpy(d_input, input.data(), INPUT_FEATURES * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_proj_weight, proj_weight.data(), PROJ_OUT * INPUT_FEATURES * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_proj_bias, proj_bias.data(), PROJ_OUT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv_weight, conv_weight.data(), OUT_CHANNELS * IN_CHANNELS * KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv_bias, conv_bias.data(), OUT_CHANNELS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn_weight, bn_weight.data(), OUT_CHANNELS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn_bias, bn_bias.data(), OUT_CHANNELS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn_mean, bn_mean.data(), OUT_CHANNELS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn_var, bn_var.data(), OUT_CHANNELS * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel (2 blocks of 1024 threads)
    dim3 grid(2, 1, 1);
    dim3 block(1024, 1, 1);
    fused_block1_fp32<<<grid, block>>>(d_input, d_proj_weight, d_proj_bias,
        d_conv_weight, d_conv_bias, d_bn_weight, d_bn_bias, d_bn_mean, d_bn_var, d_output);
    cudaDeviceSynchronize();

    // Read back output
    std::vector<float> gpu_out(OUT_CHANNELS * SEQ_LEN);
    cudaMemcpy(gpu_out.data(), d_output, OUT_CHANNELS * SEQ_LEN * sizeof(float), cudaMemcpyDeviceToHost);

    // Validate against CPU
    bool pass = true;
    for (int i = 0; i < OUT_CHANNELS * SEQ_LEN; ++i) {
        if (fabs(gpu_out[i] - cpu_out[i]) > 1e-3) {
            std::cout << "Mismatch at " << i << ": GPU " << gpu_out[i] << " CPU " << cpu_out[i] << "\n";
            pass = false;
            break;
        }
    }
    std::cout << (pass ? "✅ FP32 validation PASSED\n" : "❌ FP32 validation FAILED\n");

    // Timing (warmup and measure)
    fused_block1_fp32<<<grid, block>>>(d_input, d_proj_weight, d_proj_bias,
        d_conv_weight, d_conv_bias, d_bn_weight, d_bn_bias, d_bn_mean, d_bn_var, d_output);
    cudaDeviceSynchronize();
    auto kernel_time = time_ms([&](){
        fused_block1_fp32<<<grid, block>>>(d_input, d_proj_weight, d_proj_bias,
            d_conv_weight, d_conv_bias, d_bn_weight, d_bn_bias, d_bn_mean, d_bn_var, d_output);
        cudaDeviceSynchronize();
    }, 1000);
    std::cout << "⏱️  Fused kernel (FP32) time: " << kernel_time * 1000.0 << " µs\n";
    std::cout << "   PyTorch GPU (cuDNN) block1 p50: 300.1 µs (target to beat)\n";

    // --- FP16 variant (if desired) ---
    // Convert weights to half and repeat the same measurement
    // (omitted for brevity, can be added on request)

    // Cleanup
    cudaFree(d_input); cudaFree(d_proj_weight); cudaFree(d_proj_bias);
    cudaFree(d_conv_weight); cudaFree(d_conv_bias);
    cudaFree(d_bn_weight); cudaFree(d_bn_bias);
    cudaFree(d_bn_mean); cudaFree(d_bn_var);
    cudaFree(d_output);

    return 0;
}