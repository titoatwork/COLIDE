// =============================================================================
// fused_block2.cu - COLIDE Project (CORRECTED)
// Fused kernel: Conv1D(64→128,k=3,pad=1) + BatchNorm + ReLU + MaxPool1D(2)
// Target: NVIDIA GeForce RTX 3050 (Ampere SM 8.6)
// Compilation: nvcc -arch=sm_86 -o fused_block2 fused_block2.cu
// =============================================================================

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
constexpr int BLOCK2_IN_CHANNELS  = 64;
constexpr int BLOCK2_SEQ_LEN      = 32;
constexpr int BLOCK2_OUT_CHANNELS = 128;
constexpr int BLOCK2_KERNEL_SIZE  = 3;
constexpr int BLOCK2_POOL_SIZE    = 2;
constexpr int BLOCK2_OUT_SEQ_LEN  = BLOCK2_SEQ_LEN / BLOCK2_POOL_SIZE;  // 16
constexpr float BN2_EPSILON = 1e-5f;

// ---------------------------------------------------------------------------
// FP32 fused kernel (CORRECTED)
// ---------------------------------------------------------------------------
__global__ void fused_block2_fp32(
    const float* __restrict__ input,          // [64 * 32]
    const float* __restrict__ conv_weight,    // [128 * 64 * 3]
    const float* __restrict__ conv_bias,      // [128]
    const float* __restrict__ bn_weight,      // [128] gamma
    const float* __restrict__ bn_bias,        // [128] beta
    const float* __restrict__ bn_mean,        // [128] running_mean
    const float* __restrict__ bn_var,         // [128] running_var
    float* __restrict__ output                // [128 * 16]
) {
    // -------------------------------------------------------------
    // 1. Load full (64, 32) input into shared memory (per-block)
    //    Using grid-stride loop: every thread in the block loads
    //    multiple elements until the entire tile is in shared memory.
    // -------------------------------------------------------------
    __shared__ float in_shared[BLOCK2_IN_CHANNELS][BLOCK2_SEQ_LEN];

    for (int idx = threadIdx.x; idx < BLOCK2_IN_CHANNELS * BLOCK2_SEQ_LEN; idx += blockDim.x) {
        int ic = idx / BLOCK2_SEQ_LEN;
        int p  = idx % BLOCK2_SEQ_LEN;
        in_shared[ic][p] = input[ic * BLOCK2_SEQ_LEN + p];
    }
    __syncthreads();

    // -------------------------------------------------------------
    // 2. Compute output (128, 16) – each thread handles one (c_out, j)
    // -------------------------------------------------------------
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= BLOCK2_OUT_CHANNELS * BLOCK2_OUT_SEQ_LEN) return;

    int c_out = tid / BLOCK2_OUT_SEQ_LEN;   // 0..127
    int j     = tid % BLOCK2_OUT_SEQ_LEN;   // 0..15

    int p0 = j * BLOCK2_POOL_SIZE;          // left position (0,2,4,...)
    int p1 = p0 + 1;                        // right position (1,3,5,...)

    float accum0 = conv_bias[c_out];
    float accum1 = conv_bias[c_out];

    // Convolution over input channels and kernel positions
    for (int ic = 0; ic < BLOCK2_IN_CHANNELS; ++ic) {
        for (int k = -1; k <= 1; ++k) {
            int pos0 = p0 + k;
            float val0 = (pos0 >= 0 && pos0 < BLOCK2_SEQ_LEN) ? in_shared[ic][pos0] : 0.0f;

            int pos1 = p1 + k;
            float val1 = (pos1 >= 0 && pos1 < BLOCK2_SEQ_LEN) ? in_shared[ic][pos1] : 0.0f;

            int w_idx = c_out * (BLOCK2_IN_CHANNELS * BLOCK2_KERNEL_SIZE) +
                        ic * BLOCK2_KERNEL_SIZE + (k + 1);
            float w = conv_weight[w_idx];

            accum0 += w * val0;
            accum1 += w * val1;
        }
    }

    // -------------------------------------------------------------
    // 3. BatchNorm + ReLU
    // -------------------------------------------------------------
    float inv_std = rsqrtf(bn_var[c_out] + BN2_EPSILON);
    float bn0 = bn_weight[c_out] * (accum0 - bn_mean[c_out]) * inv_std + bn_bias[c_out];
    float bn1 = bn_weight[c_out] * (accum1 - bn_mean[c_out]) * inv_std + bn_bias[c_out];

    float relu0 = fmaxf(bn0, 0.0f);
    float relu1 = fmaxf(bn1, 0.0f);

    // -------------------------------------------------------------
    // 4. MaxPool – take the max of the two adjacent elements
    // -------------------------------------------------------------
    output[tid] = fmaxf(relu0, relu1);
}

// ---------------------------------------------------------------------------
// CPU reference (unchanged)
// ---------------------------------------------------------------------------
void cpu_reference_block2(
    const std::vector<float>& input,
    const std::vector<float>& conv_weight,
    const std::vector<float>& conv_bias,
    const std::vector<float>& bn_weight,
    const std::vector<float>& bn_bias,
    const std::vector<float>& bn_mean,
    const std::vector<float>& bn_var,
    std::vector<float>& output)
{
    output.assign(BLOCK2_OUT_CHANNELS * BLOCK2_OUT_SEQ_LEN, 0.0f);
    for (int c_out = 0; c_out < BLOCK2_OUT_CHANNELS; ++c_out) {
        for (int j = 0; j < BLOCK2_OUT_SEQ_LEN; ++j) {
            int p0 = j * 2;
            int p1 = p0 + 1;
            float acc0 = conv_bias[c_out];
            float acc1 = conv_bias[c_out];
            for (int ic = 0; ic < BLOCK2_IN_CHANNELS; ++ic) {
                for (int k = -1; k <= 1; ++k) {
                    int pos0 = p0 + k;
                    float val0 = (pos0 >= 0 && pos0 < BLOCK2_SEQ_LEN) ? input[ic * BLOCK2_SEQ_LEN + pos0] : 0.0f;
                    int pos1 = p1 + k;
                    float val1 = (pos1 >= 0 && pos1 < BLOCK2_SEQ_LEN) ? input[ic * BLOCK2_SEQ_LEN + pos1] : 0.0f;
                    int w_idx = c_out * (BLOCK2_IN_CHANNELS * BLOCK2_KERNEL_SIZE) + ic * BLOCK2_KERNEL_SIZE + (k+1);
                    float w = conv_weight[w_idx];
                    acc0 += w * val0;
                    acc1 += w * val1;
                }
            }
            float inv_std = 1.0f / sqrtf(bn_var[c_out] + BN2_EPSILON);
            float bn0 = bn_weight[c_out] * (acc0 - bn_mean[c_out]) * inv_std + bn_bias[c_out];
            float bn1 = bn_weight[c_out] * (acc1 - bn_mean[c_out]) * inv_std + bn_bias[c_out];
            float relu0 = std::max(0.0f, bn0);
            float relu1 = std::max(0.0f, bn1);
            output[c_out * BLOCK2_OUT_SEQ_LEN + j] = std::max(relu0, relu1);
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
    std::cout << "=== COLIDE Fused Block2 Kernel (corrected) ===\n";

    srand(42);
    auto randf = [](){ return (float)rand() / RAND_MAX - 0.5f; };

    std::vector<float> input(BLOCK2_IN_CHANNELS * BLOCK2_SEQ_LEN);
    for (auto& x : input) x = randf();

    std::vector<float> conv_weight(BLOCK2_OUT_CHANNELS * BLOCK2_IN_CHANNELS * BLOCK2_KERNEL_SIZE);
    for (auto& x : conv_weight) x = randf();
    std::vector<float> conv_bias(BLOCK2_OUT_CHANNELS);
    for (auto& x : conv_bias) x = randf();
    std::vector<float> bn_weight(BLOCK2_OUT_CHANNELS);
    for (auto& x : bn_weight) x = randf() + 1.0f;
    std::vector<float> bn_bias(BLOCK2_OUT_CHANNELS);
    for (auto& x : bn_bias) x = randf();
    std::vector<float> bn_mean(BLOCK2_OUT_CHANNELS);
    for (auto& x : bn_mean) x = randf();
    std::vector<float> bn_var(BLOCK2_OUT_CHANNELS);
    for (auto& x : bn_var) x = fabs(randf()) + 1.0f;

    std::vector<float> cpu_out(BLOCK2_OUT_CHANNELS * BLOCK2_OUT_SEQ_LEN);
    cpu_reference_block2(input, conv_weight, conv_bias, bn_weight, bn_bias,
                         bn_mean, bn_var, cpu_out);

    float *d_input, *d_conv_weight, *d_conv_bias;
    float *d_bn_weight, *d_bn_bias, *d_bn_mean, *d_bn_var, *d_output;
    cudaMalloc(&d_input, input.size() * sizeof(float));
    cudaMalloc(&d_conv_weight, conv_weight.size() * sizeof(float));
    cudaMalloc(&d_conv_bias, conv_bias.size() * sizeof(float));
    cudaMalloc(&d_bn_weight, bn_weight.size() * sizeof(float));
    cudaMalloc(&d_bn_bias, bn_bias.size() * sizeof(float));
    cudaMalloc(&d_bn_mean, bn_mean.size() * sizeof(float));
    cudaMalloc(&d_bn_var, bn_var.size() * sizeof(float));
    cudaMalloc(&d_output, BLOCK2_OUT_CHANNELS * BLOCK2_OUT_SEQ_LEN * sizeof(float));

    cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv_weight, conv_weight.data(), conv_weight.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv_bias, conv_bias.data(), conv_bias.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn_weight, bn_weight.data(), bn_weight.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn_bias, bn_bias.data(), bn_bias.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn_mean, bn_mean.data(), bn_mean.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn_var, bn_var.data(), bn_var.size() * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid(2, 1, 1);
    dim3 block(1024, 1, 1);
    fused_block2_fp32<<<grid, block>>>(d_input, d_conv_weight, d_conv_bias,
        d_bn_weight, d_bn_bias, d_bn_mean, d_bn_var, d_output);
    cudaDeviceSynchronize();

    std::vector<float> gpu_out(BLOCK2_OUT_CHANNELS * BLOCK2_OUT_SEQ_LEN);
    cudaMemcpy(gpu_out.data(), d_output, BLOCK2_OUT_CHANNELS * BLOCK2_OUT_SEQ_LEN * sizeof(float), cudaMemcpyDeviceToHost);

    bool pass = true;
    for (int i = 0; i < BLOCK2_OUT_CHANNELS * BLOCK2_OUT_SEQ_LEN; ++i) {
        if (fabs(gpu_out[i] - cpu_out[i]) > 1e-3) {
            std::cout << "Mismatch at " << i << ": GPU " << gpu_out[i] << " CPU " << cpu_out[i] << "\n";
            pass = false;
            break;
        }
    }
    std::cout << (pass ? "✅ FP32 validation PASSED\n" : "❌ FP32 validation FAILED\n");

    fused_block2_fp32<<<grid, block>>>(d_input, d_conv_weight, d_conv_bias,
        d_bn_weight, d_bn_bias, d_bn_mean, d_bn_var, d_output);
    cudaDeviceSynchronize();

    auto kernel_time = time_ms([&](){
        fused_block2_fp32<<<grid, block>>>(d_input, d_conv_weight, d_conv_bias,
            d_bn_weight, d_bn_bias, d_bn_mean, d_bn_var, d_output);
        cudaDeviceSynchronize();
    }, 1000);

    std::cout << "⏱️  Fused kernel (FP32) time: " << kernel_time * 1000.0 << " µs\n";
    std::cout << "   PyTorch GPU target: 333.8 µs\n";

    cudaFree(d_input);
    cudaFree(d_conv_weight);
    cudaFree(d_conv_bias);
    cudaFree(d_bn_weight);
    cudaFree(d_bn_bias);
    cudaFree(d_bn_mean);
    cudaFree(d_bn_var);
    cudaFree(d_output);

    return 0;
}