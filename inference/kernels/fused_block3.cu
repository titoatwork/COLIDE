// =============================================================================
// fused_block3.cu - COLIDE Project
// BiLSTM layers with CUDA Graphs optimization
// Compile: nvcc -arch=sm_86 -o fused_block3 fused_block3.cu
// =============================================================================

#include <cuda_runtime.h>
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

__global__ void transpose_kernel(
    const float* __restrict__ in, float* __restrict__ out,
    int rows, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    int r = idx / cols, c = idx % cols;
    out[c * rows + r] = in[r * cols + c];
}

__global__ void linear_proj_kernel(
    const float* __restrict__ in, const float* __restrict__ w,
    float* __restrict__ out,
    int input_size, int out_rows, int seq_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_rows * seq_len) return;
    int r = idx / seq_len, t = idx % seq_len;
    float sum = 0.0f;
    for (int c = 0; c < input_size; ++c)
        sum += w[r * input_size + c] * in[c * seq_len + t];
    out[idx] = sum;
}

__global__ void lstm_recurrent_kernel(
    const float* __restrict__ gate_ih_all,
    const float* __restrict__ w_hh_t,
    const float* __restrict__ bias_ih,
    const float* __restrict__ bias_hh,
    float* __restrict__ output_hidden,
    int hidden_size, int seq_len, bool reverse
) {
    extern __shared__ float s_h_prev[];
    int h = threadIdx.x;
    if (h >= hidden_size) return;
    s_h_prev[h] = 0.0f;
    __syncthreads();

    float c = 0.0f;
    int four_h = 4 * hidden_size;

    for (int t = 0; t < seq_len; ++t) {
        int pos = reverse ? (seq_len - 1 - t) : t;
        float i_gate = gate_ih_all[h*seq_len+pos] + bias_ih[h] + bias_hh[h];
        float f_gate = gate_ih_all[(hidden_size+h)*seq_len+pos] + bias_ih[hidden_size+h] + bias_hh[hidden_size+h];
        float g_gate = gate_ih_all[(2*hidden_size+h)*seq_len+pos] + bias_ih[2*hidden_size+h] + bias_hh[2*hidden_size+h];
        float o_gate = gate_ih_all[(3*hidden_size+h)*seq_len+pos] + bias_ih[3*hidden_size+h] + bias_hh[3*hidden_size+h];

        for (int j = 0; j < hidden_size; ++j) {
            float prev_h = s_h_prev[j];
            int base = j * four_h;
            i_gate += w_hh_t[base + h] * prev_h;
            f_gate += w_hh_t[base + hidden_size + h] * prev_h;
            g_gate += w_hh_t[base + 2*hidden_size + h] * prev_h;
            o_gate += w_hh_t[base + 3*hidden_size + h] * prev_h;
        }

        float i_val = 1.0f/(1.0f+expf(-i_gate));
        float f_val = 1.0f/(1.0f+expf(-f_gate));
        float g_val = tanhf(g_gate);
        float o_val = 1.0f/(1.0f+expf(-o_gate));
        c = f_val*c + i_val*g_val;
        float h_new = o_val * tanhf(c);
        output_hidden[h*seq_len + t] = h_new;
        s_h_prev[h] = h_new;
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
    int half = hidden_size * seq_len;
    out[idx] = (idx < half) ? fw[idx] : rev[idx - half];
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

void lstm_direction(
    cudaStream_t stream,
    const float* d_input, float* d_output_hidden,
    const float* d_w_ih, const float* d_w_hh_t,
    const float* d_bias_ih, const float* d_bias_hh,
    int input_size, int hidden_size, int seq_len, bool reverse,
    float* d_gate_ih_all
) {
    int out_rows = 4*hidden_size;
    int total = out_rows*seq_len;
    int threads = 256;
    int blocks = (total+threads-1)/threads;
    linear_proj_kernel<<<blocks, threads, 0, stream>>>(
        d_input, d_w_ih, d_gate_ih_all, input_size, out_rows, seq_len);
    int smem = hidden_size * sizeof(float);
    lstm_recurrent_kernel<<<1, hidden_size, smem, stream>>>(
        d_gate_ih_all, d_w_hh_t, d_bias_ih, d_bias_hh,
        d_output_hidden, hidden_size, seq_len, reverse);
}

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
    std::cout << "=== COLIDE Block3 (transposed W_hh + CUDA Graphs) ===\n";
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

    float *d_input, *d_h1_fw, *d_h1_rev, *d_in2, *d_h2_fw, *d_h2_rev, *d_out;
    cudaMalloc(&d_input, h_input.size()*sizeof(float));
    cudaMalloc(&d_h1_fw, H1*SEQ*sizeof(float));
    cudaMalloc(&d_h1_rev, H1*SEQ*sizeof(float));
    cudaMalloc(&d_in2, H1x2*SEQ*sizeof(float));
    cudaMalloc(&d_h2_fw, H2*SEQ*sizeof(float));
    cudaMalloc(&d_h2_rev, H2*SEQ*sizeof(float));
    cudaMalloc(&d_out, OUT_SIZE*sizeof(float));

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
    cudaMemcpy(d_input, h_input.data(), h_input.size()*sizeof(float), cudaMemcpyHostToDevice);

    float *d_w_hh1_f_t,*d_w_hh1_r_t,*d_w_hh2_f_t,*d_w_hh2_r_t;
    cudaMalloc(&d_w_hh1_f_t, 4*H1*H1*sizeof(float));
    cudaMalloc(&d_w_hh1_r_t, 4*H1*H1*sizeof(float));
    cudaMalloc(&d_w_hh2_f_t, 4*H2*H2*sizeof(float));
    cudaMalloc(&d_w_hh2_r_t, 4*H2*H2*sizeof(float));
    {
        int n1=4*H1*H1, n2=4*H2*H2, thr=256;
        transpose_kernel<<<(n1+thr-1)/thr,thr>>>(d_w_hh1_f,d_w_hh1_f_t,4*H1,H1);
        transpose_kernel<<<(n1+thr-1)/thr,thr>>>(d_w_hh1_r,d_w_hh1_r_t,4*H1,H1);
        transpose_kernel<<<(n2+thr-1)/thr,thr>>>(d_w_hh2_f,d_w_hh2_f_t,4*H2,H2);
        transpose_kernel<<<(n2+thr-1)/thr,thr>>>(d_w_hh2_r,d_w_hh2_r_t,4*H2,H2);
        cudaDeviceSynchronize();
    }

    float *d_gate1_fw,*d_gate1_rev,*d_gate2_fw,*d_gate2_rev;
    cudaMalloc(&d_gate1_fw, 4*H1*SEQ*sizeof(float));
    cudaMalloc(&d_gate1_rev, 4*H1*SEQ*sizeof(float));
    cudaMalloc(&d_gate2_fw, 4*H2*SEQ*sizeof(float));
    cudaMalloc(&d_gate2_rev, 4*H2*SEQ*sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto launch_pipeline = [&](cudaStream_t s) {
        lstm_direction(s, d_input, d_h1_fw, d_w_ih1_f, d_w_hh1_f_t, d_b_ih1_f, d_b_hh1_f, IN_CH, H1, SEQ, false, d_gate1_fw);
        lstm_direction(s, d_input, d_h1_rev, d_w_ih1_r, d_w_hh1_r_t, d_b_ih1_r, d_b_hh1_r, IN_CH, H1, SEQ, true, d_gate1_rev);
        int cb=(H1x2*SEQ+255)/256;
        combine_kernel<<<cb,256,0,s>>>(d_h1_fw, d_h1_rev, d_in2, H1, SEQ);
        lstm_direction(s, d_in2, d_h2_fw, d_w_ih2_f, d_w_hh2_f_t, d_b_ih2_f, d_b_hh2_f, H1x2, H2, SEQ, false, d_gate2_fw);
        lstm_direction(s, d_in2, d_h2_rev, d_w_ih2_r, d_w_hh2_r_t, d_b_ih2_r, d_b_hh2_r, H1x2, H2, SEQ, true, d_gate2_rev);
        extract_last_timestep_kernel<<<1,H2,0,s>>>(d_h2_fw, d_h2_rev, H2, SEQ, d_out);
    };

    // === Benchmark 1: Without CUDA Graphs ===
    launch_pipeline(stream);
    cudaStreamSynchronize(stream);

    const int iters = 100;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i=0; i<iters; ++i) { launch_pipeline(stream); cudaStreamSynchronize(stream); }
    auto end = std::chrono::high_resolution_clock::now();
    double no_graph_us = std::chrono::duration<double, std::micro>(end-start).count()/iters;

    std::vector<float> gpu_out(OUT_SIZE);
    cudaMemcpy(gpu_out.data(), d_out, OUT_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
    bool pass=true;
    for (int i=0; i<OUT_SIZE; ++i) {
        if (fabs(gpu_out[i]-cpu_out[i]) > 1e-2) {
            std::cout<<"Mismatch at "<<i<<": GPU "<<gpu_out[i]<<" CPU "<<cpu_out[i]<<"\n";
            pass=false; break;
        }
    }
    std::cout<<(pass?"✅ FP32 validation PASSED\n":"❌ FP32 validation FAILED\n");
    std::cout<<"⏱️  Without CUDA Graphs: "<<no_graph_us<<" µs\n";

    // === Benchmark 2: With CUDA Graphs ===
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    launch_pipeline(stream);
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

    cudaGraphLaunch(graphExec, stream);
    cudaStreamSynchronize(stream);

    cudaMemcpy(gpu_out.data(), d_out, OUT_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
    bool gpass=true;
    for (int i=0; i<OUT_SIZE; ++i) {
        if (fabs(gpu_out[i]-cpu_out[i]) > 1e-2) {
            std::cout<<"Graph mismatch at "<<i<<": GPU "<<gpu_out[i]<<" CPU "<<cpu_out[i]<<"\n";
            gpass=false; break;
        }
    }
    std::cout<<(gpass?"✅ CUDA Graph validation PASSED\n":"❌ CUDA Graph validation FAILED\n");

    start = std::chrono::high_resolution_clock::now();
    for (int i=0; i<iters; ++i) { cudaGraphLaunch(graphExec, stream); cudaStreamSynchronize(stream); }
    end = std::chrono::high_resolution_clock::now();
    double graph_us = std::chrono::duration<double, std::micro>(end-start).count()/iters;

    std::cout<<"⏱️  With CUDA Graphs:    "<<graph_us<<" µs\n";
    std::cout<<"   Speedup from graphs:  "<<no_graph_us/graph_us<<"x\n";
    std::cout<<"   PyTorch GPU target:   740.7 µs\n";

    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(d_input); cudaFree(d_h1_fw); cudaFree(d_h1_rev); cudaFree(d_in2);
    cudaFree(d_h2_fw); cudaFree(d_h2_rev); cudaFree(d_out);
    cudaFree(d_gate1_fw); cudaFree(d_gate1_rev);
    cudaFree(d_gate2_fw); cudaFree(d_gate2_rev);
    cudaFree(d_w_ih1_f); cudaFree(d_w_hh1_f); cudaFree(d_b_ih1_f); cudaFree(d_b_hh1_f);
    cudaFree(d_w_ih1_r); cudaFree(d_w_hh1_r); cudaFree(d_b_ih1_r); cudaFree(d_b_hh1_r);
    cudaFree(d_w_ih2_f); cudaFree(d_w_hh2_f); cudaFree(d_b_ih2_f); cudaFree(d_b_hh2_f);
    cudaFree(d_w_ih2_r); cudaFree(d_w_hh2_r); cudaFree(d_b_ih2_r); cudaFree(d_b_hh2_r);
    cudaFree(d_w_hh1_f_t); cudaFree(d_w_hh1_r_t);
    cudaFree(d_w_hh2_f_t); cudaFree(d_w_hh2_r_t);
    return 0;
}
