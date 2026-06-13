// =============================================================================
// fused_block4.cu - COLIDE Project
// Dense classification head: 128 -> 64 ReLU -> 5 logits
// Input: (128,) vector from Block3
// Output: (5,) raw logits (no softmax)
// Compile: nvcc -arch=sm_86 -o fused_block4 fused_block4.cu
// =============================================================================

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

constexpr int INPUT_DIM  = 128;
constexpr int HIDDEN_DIM = 64;
constexpr int NUM_CLASSES = 5;

// ---------------------------------------------------------------------------
// Fused dense kernel: 128 -> 64 ReLU -> 5 (single block)
// ---------------------------------------------------------------------------
__global__ void fused_dense_kernel(
    const float* __restrict__ input,        // [INPUT_DIM]
    const float* __restrict__ fc1_weight,   // (HIDDEN_DIM, INPUT_DIM) row-major
    const float* __restrict__ fc1_bias,     // [HIDDEN_DIM]
    const float* __restrict__ fc2_weight,   // (NUM_CLASSES, HIDDEN_DIM) row-major
    const float* __restrict__ fc2_bias,     // [NUM_CLASSES]
    float* __restrict__ output              // [NUM_CLASSES] logits
) {
    __shared__ float s_input[INPUT_DIM];      // cached input
    __shared__ float s_hidden[HIDDEN_DIM];    // intermediate 64 values

    // 1. Load input vector into shared memory cooperatively
    for (int i = threadIdx.x; i < INPUT_DIM; i += blockDim.x) {
        s_input[i] = input[i];
    }
    __syncthreads();

    // 2. First dense layer: 128 -> 64 with ReLU
    // Each thread computes one output neuron if threadIdx.x < HIDDEN_DIM
    if (threadIdx.x < HIDDEN_DIM) {
        float sum = fc1_bias[threadIdx.x];
        for (int j = 0; j < INPUT_DIM; ++j) {
            sum += fc1_weight[threadIdx.x * INPUT_DIM + j] * s_input[j];
        }
        // ReLU
        s_hidden[threadIdx.x] = fmaxf(sum, 0.0f);
    }
    __syncthreads();

    // 3. Second dense layer: 64 -> 5
    // Only the first 5 threads compute final logits
    if (threadIdx.x < NUM_CLASSES) {
        float sum = fc2_bias[threadIdx.x];
        for (int j = 0; j < HIDDEN_DIM; ++j) {
            sum += fc2_weight[threadIdx.x * HIDDEN_DIM + j] * s_hidden[j];
        }
        output[threadIdx.x] = sum;  // raw logits
    }
}

// ---------------------------------------------------------------------------
// CPU reference (same computation)
// ---------------------------------------------------------------------------
void cpu_dense(
    const std::vector<float>& input,
    const std::vector<float>& fc1_weight,
    const std::vector<float>& fc1_bias,
    const std::vector<float>& fc2_weight,
    const std::vector<float>& fc2_bias,
    std::vector<float>& output
) {
    std::vector<float> hidden(HIDDEN_DIM, 0.0f);
    // Layer 1: 128 -> 64 ReLU
    for (int i = 0; i < HIDDEN_DIM; ++i) {
        float sum = fc1_bias[i];
        for (int j = 0; j < INPUT_DIM; ++j) {
            sum += fc1_weight[i * INPUT_DIM + j] * input[j];
        }
        hidden[i] = fmaxf(sum, 0.0f);
    }
    // Layer 2: 64 -> 5
    output.assign(NUM_CLASSES, 0.0f);
    for (int i = 0; i < NUM_CLASSES; ++i) {
        float sum = fc2_bias[i];
        for (int j = 0; j < HIDDEN_DIM; ++j) {
            sum += fc2_weight[i * HIDDEN_DIM + j] * hidden[j];
        }
        output[i] = sum;
    }
}

// ---------------------------------------------------------------------------
// Main test harness
// ---------------------------------------------------------------------------
int main() {
    std::cout << "=== COLIDE Fused Block4 (Dense head) ===\n";
    srand(42);
    auto randf = [](){ return (float)rand() / RAND_MAX - 0.5f; };

    // Random input and weights
    std::vector<float> h_input(INPUT_DIM);
    for (auto& v : h_input) v = randf();

    std::vector<float> fc1_w(HIDDEN_DIM * INPUT_DIM);
    std::vector<float> fc1_b(HIDDEN_DIM);
    std::vector<float> fc2_w(NUM_CLASSES * HIDDEN_DIM);
    std::vector<float> fc2_b(NUM_CLASSES);
    for (auto& v : fc1_w) v = randf();
    for (auto& v : fc1_b) v = randf();
    for (auto& v : fc2_w) v = randf();
    for (auto& v : fc2_b) v = randf();

    // CPU reference
    std::vector<float> cpu_out;
    cpu_dense(h_input, fc1_w, fc1_b, fc2_w, fc2_b, cpu_out);

    // GPU allocations
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

    // Launch kernel: 1 block of 64 threads (covers both layers, only 5 active for final)
    fused_dense_kernel<<<1, 64>>>(d_input, d_fc1_w, d_fc1_b, d_fc2_w, d_fc2_b, d_output);
    cudaDeviceSynchronize();

    // Validate
    std::vector<float> gpu_out(NUM_CLASSES);
    cudaMemcpy(gpu_out.data(), d_output, NUM_CLASSES * sizeof(float), cudaMemcpyDeviceToHost);

    bool pass = true;
    for (int i = 0; i < NUM_CLASSES; ++i) {
        if (fabs(gpu_out[i] - cpu_out[i]) > 1e-3) {
            std::cout << "Mismatch at " << i << ": GPU " << gpu_out[i] << " CPU " << cpu_out[i] << "\n";
            pass = false;
            break;
        }
    }
    std::cout << (pass ? "✅ FP32 validation PASSED\n" : "❌ FP32 validation FAILED\n");

    // Timing (1000 iterations for small kernel)
    // Warmup
    fused_dense_kernel<<<1, 64>>>(d_input, d_fc1_w, d_fc1_b, d_fc2_w, d_fc2_b, d_output);
    cudaDeviceSynchronize();

    const int iters = 10000;  // small kernel, many iterations
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        fused_dense_kernel<<<1, 64>>>(d_input, d_fc1_w, d_fc1_b, d_fc2_w, d_fc2_b, d_output);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    double avg_us = std::chrono::duration<double, std::micro>(end - start).count() / iters;

    std::cout << "⏱️  Block4 (Dense) time: " << avg_us << " µs\n";
    std::cout << "   PyTorch GPU target: 186.5 µs\n";

    // Cleanup
    cudaFree(d_input); cudaFree(d_fc1_w); cudaFree(d_fc1_b);
    cudaFree(d_fc2_w); cudaFree(d_fc2_b); cudaFree(d_output);

    return 0;
}