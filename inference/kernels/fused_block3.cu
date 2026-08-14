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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fstream>

constexpr int SEQ      = 16;
constexpr int IN_CH    = 128;
constexpr int H1       = 128;
constexpr int H1x2     = 256;
constexpr int H2       = 64;
constexpr int OUT_SIZE  = 128;  // last-timestep BiLSTM channels = 2*H2
constexpr int FULL_SIZE = 2 * H2 * SEQ;  // full sequence [SEQ, 2*H2] row-major

#define CUDA_CHECK(call) do { \
  cudaError_t err__ = (call); \
  if (err__ != cudaSuccess) { \
    std::cerr << "CUDA error " << cudaGetErrorString(err__) \
              << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
    std::exit(1); \
  } \
} while(0)
#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

// Deterministic RNG fill (seeded caller); values in [-0.5, 0.5).
static void fill_rand(std::vector<float>& v) {
    for (auto& x : v) x = (float)rand() / RAND_MAX - 0.5f;
}

// Little-endian raw float32 dump (numpy tofile compatible).
static bool load_f32_bin(const std::string& path, std::vector<float>& v, size_t expected) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "Failed to open for read: " << path << "\n";
        return false;
    }
    v.resize(expected);
    f.read(reinterpret_cast<char*>(v.data()), expected * sizeof(float));
    if (!f || static_cast<size_t>(f.gcount()) != expected * sizeof(float)) {
        std::cerr << "Short read: " << path << " (expected " << expected << " floats)\n";
        return false;
    }
    return true;
}

static bool write_f32_bin(const std::string& path, const std::vector<float>& v) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "Failed to open for write: " << path << "\n";
        return false;
    }
    f.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(float));
    return static_cast<bool>(f);
}

// Interleave layer-2 fw/rev into time-major channel-last [SEQ, 2*H2]:
// for t: fw[0:H2] at t, then rev[0:H2] at t.
static std::vector<float> pack_full_seq(
    const std::vector<float>& h2_fw, const std::vector<float>& h2_rev)
{
    std::vector<float> full(FULL_SIZE);
    for (int t = 0; t < SEQ; ++t) {
        for (int h = 0; h < H2; ++h)
            full[t * OUT_SIZE + h] = h2_fw[h * SEQ + t];
        for (int h = 0; h < H2; ++h)
            full[t * OUT_SIZE + H2 + h] = h2_rev[h * SEQ + t];
    }
    return full;
}

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

// Double-buffer invariant: within one timestep, every thread reads previous
// hidden only from read_buf and writes its new hidden only to write_buf
// (disjoint shared arrays). __syncthreads() after all writes; buffer roles
// swap each timestep via t%2. Never read and overwrite the same shared
// hidden array in one sync epoch (race fixed; see fused_block3_naive.cu).
//
// Reverse alignment: when reverse=true, pos = seq_len-1-t is the original
// sequence index. Gates read input at pos and h_new is stored at pos so
// output_hidden[h, k] always corresponds to input position k for both
// directions. Forward has pos==t. After layer-1 reverse is position-aligned,
// combine_kernel concatenates fw|rev into a time-aligned BiLSTM sequence
// for layer 2.
__global__ void lstm_recurrent_kernel(
    const float* __restrict__ gate_ih_all,
    const float* __restrict__ w_hh_t,
    const float* __restrict__ bias_ih,
    const float* __restrict__ bias_hh,
    float* __restrict__ output_hidden,
    int hidden_size, int seq_len, bool reverse
) {
    extern __shared__ float shmem[];
    float* s_h[2] = { &shmem[0], &shmem[hidden_size] };
    int h = threadIdx.x;
    if (h >= hidden_size) return;
    s_h[0][h] = 0.0f;
    s_h[1][h] = 0.0f;
    __syncthreads();

    float c = 0.0f;
    int four_h = 4 * hidden_size;

    for (int t = 0; t < seq_len; ++t) {
        int pos = reverse ? (seq_len - 1 - t) : t;
        float* read_buf  = s_h[t % 2];
        float* write_buf = s_h[(t + 1) % 2];

        float i_gate = gate_ih_all[h*seq_len+pos] + bias_ih[h] + bias_hh[h];
        float f_gate = gate_ih_all[(hidden_size+h)*seq_len+pos] + bias_ih[hidden_size+h] + bias_hh[hidden_size+h];
        float g_gate = gate_ih_all[(2*hidden_size+h)*seq_len+pos] + bias_ih[2*hidden_size+h] + bias_hh[2*hidden_size+h];
        float o_gate = gate_ih_all[(3*hidden_size+h)*seq_len+pos] + bias_ih[3*hidden_size+h] + bias_hh[3*hidden_size+h];

        for (int j = 0; j < hidden_size; ++j) {
            float prev_h = read_buf[j];
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
        // Store at original sequence position (pos), not recurrence index t.
        output_hidden[h*seq_len + pos] = h_new;
        write_buf[h] = h_new;
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

// After reverse alignment, fw/rev sequences are time-aligned to input
// positions. Index (seq_len-1) matches PyTorch output[:, -1, :] semantics
// (not reverse recurrence's final state at original pos 0). Full sequence
// is preferred for V3 attention; last-timestep is the current harness
// contract for parity with existing Block-3 benchmarks.
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
    CUDA_CHECK_LAST();
    // 2 * hidden_size: double-buffered previous/next hidden state
    int smem = 2 * hidden_size * sizeof(float);
    lstm_recurrent_kernel<<<1, hidden_size, smem, stream>>>(
        d_gate_ih_all, d_w_hh_t, d_bias_ih, d_bias_hh,
        d_output_hidden, hidden_size, seq_len, reverse);
    CUDA_CHECK_LAST();
}

// CPU reference: same reverse alignment as GPU (store at original pos).
// Gate order: i, f, g, o (matches PyTorch LSTM; see docs/CUDA_WEIGHT_MAPPING.md).
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
            // Align to original sequence position (same as GPU).
            output_h[h*seq_len+pos]=h_new[h];
        }
        h_prev=h_new; c_prev=c_new;
    }
}

// Primary CPU contract: full sequence [SEQ, 2*H2] time-major channel-last.
// Also fills optional last-timestep vector (legacy_last_state auxiliary check).
void cpu_pipeline_full(
    const std::vector<float>& input,
    const std::vector<float>& w_ih1_f, const std::vector<float>& w_hh1_f,
    const std::vector<float>& b_ih1_f, const std::vector<float>& b_hh1_f,
    const std::vector<float>& w_ih1_r, const std::vector<float>& w_hh1_r,
    const std::vector<float>& b_ih1_r, const std::vector<float>& b_hh1_r,
    const std::vector<float>& w_ih2_f, const std::vector<float>& w_hh2_f,
    const std::vector<float>& b_ih2_f, const std::vector<float>& b_hh2_f,
    const std::vector<float>& w_ih2_r, const std::vector<float>& w_hh2_r,
    const std::vector<float>& b_ih2_r, const std::vector<float>& b_hh2_r,
    std::vector<float>& out_full,
    std::vector<float>* out_last = nullptr)
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
    out_full = pack_full_seq(h2_fw, h2_rev);
    if (out_last) {
        out_last->resize(OUT_SIZE);
        for (int i=0;i<H2;++i) (*out_last)[i]=h2_fw[i*SEQ+SEQ-1];
        for (int i=0;i<H2;++i) (*out_last)[i+H2]=h2_rev[i*SEQ+SEQ-1];
    }
}

// legacy_last_state: last-timestep only (auxiliary; primary check is full sequence).
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
    std::vector<float> full, last;
    cpu_pipeline_full(input,
        w_ih1_f, w_hh1_f, b_ih1_f, b_hh1_f, w_ih1_r, w_hh1_r, b_ih1_r, b_hh1_r,
        w_ih2_f, w_hh2_f, b_ih2_f, b_hh2_f, w_ih2_r, w_hh2_r, b_ih2_r, b_hh2_r,
        full, &last);
    return last;
}

int main(int argc, char** argv) {
    std::cout << "=== COLIDE Block3 (transposed W_hh + CUDA Graphs) ===\n";

    // Optional weight-file inject: argv[1] directory or env COLIDE_B3_WEIGHTS
    std::string weight_dir;
    if (argc >= 2 && argv[1] && argv[1][0] != '\0')
        weight_dir = argv[1];
    else if (const char* env = std::getenv("COLIDE_B3_WEIGHTS"))
        weight_dir = env;
    const bool inject_mode = !weight_dir.empty();
    if (inject_mode)
        std::cout << "WEIGHT_INJECT_MODE dir=" << weight_dir << "\n";

    std::vector<float> h_input(IN_CH*SEQ);
    std::vector<float> w_ih1_f(4*H1*IN_CH), w_hh1_f(4*H1*H1), b_ih1_f(4*H1), b_hh1_f(4*H1);
    std::vector<float> w_ih1_r(4*H1*IN_CH), w_hh1_r(4*H1*H1), b_ih1_r(4*H1), b_hh1_r(4*H1);
    std::vector<float> w_ih2_f(4*H2*H1x2), w_hh2_f(4*H2*H2), b_ih2_f(4*H2), b_hh2_f(4*H2);
    std::vector<float> w_ih2_r(4*H2*H1x2), w_hh2_r(4*H2*H2), b_ih2_r(4*H2), b_hh2_r(4*H2);

    if (inject_mode) {
        auto p = [&](const char* name) { return weight_dir + "/" + name; };
        bool ok = true;
        ok &= load_f32_bin(p("input.bin"), h_input, IN_CH*SEQ);
        ok &= load_f32_bin(p("w_ih1_f.bin"), w_ih1_f, w_ih1_f.size());
        ok &= load_f32_bin(p("w_hh1_f.bin"), w_hh1_f, w_hh1_f.size());
        ok &= load_f32_bin(p("b_ih1_f.bin"), b_ih1_f, b_ih1_f.size());
        ok &= load_f32_bin(p("b_hh1_f.bin"), b_hh1_f, b_hh1_f.size());
        ok &= load_f32_bin(p("w_ih1_r.bin"), w_ih1_r, w_ih1_r.size());
        ok &= load_f32_bin(p("w_hh1_r.bin"), w_hh1_r, w_hh1_r.size());
        ok &= load_f32_bin(p("b_ih1_r.bin"), b_ih1_r, b_ih1_r.size());
        ok &= load_f32_bin(p("b_hh1_r.bin"), b_hh1_r, b_hh1_r.size());
        ok &= load_f32_bin(p("w_ih2_f.bin"), w_ih2_f, w_ih2_f.size());
        ok &= load_f32_bin(p("w_hh2_f.bin"), w_hh2_f, w_hh2_f.size());
        ok &= load_f32_bin(p("b_ih2_f.bin"), b_ih2_f, b_ih2_f.size());
        ok &= load_f32_bin(p("b_hh2_f.bin"), b_hh2_f, b_hh2_f.size());
        ok &= load_f32_bin(p("w_ih2_r.bin"), w_ih2_r, w_ih2_r.size());
        ok &= load_f32_bin(p("w_hh2_r.bin"), w_hh2_r, w_hh2_r.size());
        ok &= load_f32_bin(p("b_ih2_r.bin"), b_ih2_r, b_ih2_r.size());
        ok &= load_f32_bin(p("b_hh2_r.bin"), b_hh2_r, b_hh2_r.size());
        if (!ok) {
            std::cerr << "WEIGHT_INJECT_MODE: failed to load one or more weight files\n";
            return 1;
        }
    } else {
        // Independent deterministic weights: seed 42 input+L1fw, 43 L1rev, 44 L2fw, 45 L2rev
        srand(42);
        fill_rand(h_input);
        fill_rand(w_ih1_f); fill_rand(w_hh1_f); fill_rand(b_ih1_f); fill_rand(b_hh1_f);
        srand(43);
        fill_rand(w_ih1_r); fill_rand(w_hh1_r); fill_rand(b_ih1_r); fill_rand(b_hh1_r);
        srand(44);
        fill_rand(w_ih2_f); fill_rand(w_hh2_f); fill_rand(b_ih2_f); fill_rand(b_hh2_f);
        srand(45);
        fill_rand(w_ih2_r); fill_rand(w_hh2_r); fill_rand(b_ih2_r); fill_rand(b_hh2_r);
    }

    std::vector<float> cpu_full, cpu_last;
    cpu_pipeline_full(h_input,
        w_ih1_f, w_hh1_f, b_ih1_f, b_hh1_f, w_ih1_r, w_hh1_r, b_ih1_r, b_hh1_r,
        w_ih2_f, w_hh2_f, b_ih2_f, b_hh2_f, w_ih2_r, w_hh2_r, b_ih2_r, b_hh2_r,
        cpu_full, &cpu_last);

    float *d_input, *d_h1_fw, *d_h1_rev, *d_in2, *d_h2_fw, *d_h2_rev, *d_out;
    CUDA_CHECK(cudaMalloc(&d_input, h_input.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_h1_fw, H1*SEQ*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_h1_rev, H1*SEQ*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_in2, H1x2*SEQ*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_h2_fw, H2*SEQ*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_h2_rev, H2*SEQ*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, OUT_SIZE*sizeof(float)));

    auto copy_vec = [](float*& d, const std::vector<float>& v) {
        CUDA_CHECK(cudaMalloc(&d, v.size()*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d, v.data(), v.size()*sizeof(float), cudaMemcpyHostToDevice));
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
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size()*sizeof(float), cudaMemcpyHostToDevice));

    float *d_w_hh1_f_t,*d_w_hh1_r_t,*d_w_hh2_f_t,*d_w_hh2_r_t;
    CUDA_CHECK(cudaMalloc(&d_w_hh1_f_t, 4*H1*H1*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w_hh1_r_t, 4*H1*H1*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w_hh2_f_t, 4*H2*H2*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w_hh2_r_t, 4*H2*H2*sizeof(float)));
    {
        int n1=4*H1*H1, n2=4*H2*H2, thr=256;
        transpose_kernel<<<(n1+thr-1)/thr,thr>>>(d_w_hh1_f,d_w_hh1_f_t,4*H1,H1);
        CUDA_CHECK_LAST();
        transpose_kernel<<<(n1+thr-1)/thr,thr>>>(d_w_hh1_r,d_w_hh1_r_t,4*H1,H1);
        CUDA_CHECK_LAST();
        transpose_kernel<<<(n2+thr-1)/thr,thr>>>(d_w_hh2_f,d_w_hh2_f_t,4*H2,H2);
        CUDA_CHECK_LAST();
        transpose_kernel<<<(n2+thr-1)/thr,thr>>>(d_w_hh2_r,d_w_hh2_r_t,4*H2,H2);
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    float *d_gate1_fw,*d_gate1_rev,*d_gate2_fw,*d_gate2_rev;
    CUDA_CHECK(cudaMalloc(&d_gate1_fw, 4*H1*SEQ*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gate1_rev, 4*H1*SEQ*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gate2_fw, 4*H2*SEQ*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gate2_rev, 4*H2*SEQ*sizeof(float)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    auto launch_pipeline = [&](cudaStream_t s) {
        lstm_direction(s, d_input, d_h1_fw, d_w_ih1_f, d_w_hh1_f_t, d_b_ih1_f, d_b_hh1_f, IN_CH, H1, SEQ, false, d_gate1_fw);
        lstm_direction(s, d_input, d_h1_rev, d_w_ih1_r, d_w_hh1_r_t, d_b_ih1_r, d_b_hh1_r, IN_CH, H1, SEQ, true, d_gate1_rev);
        int cb=(H1x2*SEQ+255)/256;
        combine_kernel<<<cb,256,0,s>>>(d_h1_fw, d_h1_rev, d_in2, H1, SEQ);
        CUDA_CHECK_LAST();
        lstm_direction(s, d_in2, d_h2_fw, d_w_ih2_f, d_w_hh2_f_t, d_b_ih2_f, d_b_hh2_f, H1x2, H2, SEQ, false, d_gate2_fw);
        lstm_direction(s, d_in2, d_h2_rev, d_w_ih2_r, d_w_hh2_r_t, d_b_ih2_r, d_b_hh2_r, H1x2, H2, SEQ, true, d_gate2_rev);
        // legacy_last_state: extract last timestep for auxiliary check / out_last.bin
        extract_last_timestep_kernel<<<1,H2,0,s>>>(d_h2_fw, d_h2_rev, H2, SEQ, d_out);
        CUDA_CHECK_LAST();
    };

    // === Benchmark 1: Without CUDA Graphs ===
    launch_pipeline(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const int iters = 100;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i=0; i<iters; ++i) {
        launch_pipeline(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    auto end = std::chrono::high_resolution_clock::now();
    double no_graph_us = std::chrono::duration<double, std::micro>(end-start).count()/iters;

    // Full-sequence GPU output: channel-major d_h2_fw/rev -> [SEQ, 2*H2]
    std::vector<float> h_h2_fw(H2*SEQ), h_h2_rev(H2*SEQ);
    CUDA_CHECK(cudaMemcpy(h_h2_fw.data(), d_h2_fw, H2*SEQ*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_h2_rev.data(), d_h2_rev, H2*SEQ*sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<float> gpu_full = pack_full_seq(h_h2_fw, h_h2_rev);

    // legacy_last_state auxiliary check
    std::vector<float> gpu_out(OUT_SIZE);
    CUDA_CHECK(cudaMemcpy(gpu_out.data(), d_out, OUT_SIZE*sizeof(float), cudaMemcpyDeviceToHost));

    const float tol = 1e-2f;
    float max_err_full = 0.0f, max_err_last = 0.0f;
    bool pass = true;
    for (int i = 0; i < FULL_SIZE; ++i) {
        float e = fabsf(gpu_full[i] - cpu_full[i]);
        if (e > max_err_full) max_err_full = e;
        if (e > tol) {
            if (pass)
                std::cout << "Full-seq mismatch at " << i << ": GPU " << gpu_full[i]
                          << " CPU " << cpu_full[i] << "\n";
            pass = false;
        }
    }
    for (int i = 0; i < OUT_SIZE; ++i) {
        float e = fabsf(gpu_out[i] - cpu_last[i]);
        if (e > max_err_last) max_err_last = e;
        if (e > tol) {
            if (pass)
                std::cout << "legacy_last_state mismatch at " << i << ": GPU " << gpu_out[i]
                          << " CPU " << cpu_last[i] << "\n";
            pass = false;
        }
    }
    std::cout << "max abs error full_seq: " << max_err_full
              << "  legacy_last_state: " << max_err_last << "\n";
    std::cout << (pass ? "✅ FP32 validation PASSED\n" : "❌ FP32 validation FAILED\n");
    std::cout << "⏱️  Without CUDA Graphs: " << no_graph_us << " µs\n";

    if (inject_mode) {
        write_f32_bin(weight_dir + "/out_last.bin", gpu_out);
        write_f32_bin(weight_dir + "/out_full.bin", gpu_full);
        std::cout << "Wrote out_last.bin (" << OUT_SIZE << " floats) and out_full.bin ("
                  << FULL_SIZE << " floats) to " << weight_dir << "\n";
    }

    // === Benchmark 2: With CUDA Graphs ===
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    launch_pipeline(stream);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaMemcpy(h_h2_fw.data(), d_h2_fw, H2*SEQ*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_h2_rev.data(), d_h2_rev, H2*SEQ*sizeof(float), cudaMemcpyDeviceToHost));
    gpu_full = pack_full_seq(h_h2_fw, h_h2_rev);
    CUDA_CHECK(cudaMemcpy(gpu_out.data(), d_out, OUT_SIZE*sizeof(float), cudaMemcpyDeviceToHost));

    float max_err_full_g = 0.0f, max_err_last_g = 0.0f;
    bool gpass = true;
    for (int i = 0; i < FULL_SIZE; ++i) {
        float e = fabsf(gpu_full[i] - cpu_full[i]);
        if (e > max_err_full_g) max_err_full_g = e;
        if (e > tol) {
            if (gpass)
                std::cout << "Graph full-seq mismatch at " << i << ": GPU " << gpu_full[i]
                          << " CPU " << cpu_full[i] << "\n";
            gpass = false;
        }
    }
    for (int i = 0; i < OUT_SIZE; ++i) {
        float e = fabsf(gpu_out[i] - cpu_last[i]);
        if (e > max_err_last_g) max_err_last_g = e;
        if (e > tol) {
            if (gpass)
                std::cout << "Graph legacy_last_state mismatch at " << i << ": GPU " << gpu_out[i]
                          << " CPU " << cpu_last[i] << "\n";
            gpass = false;
        }
    }
    std::cout << "graph max abs error full_seq: " << max_err_full_g
              << "  legacy_last_state: " << max_err_last_g << "\n";
    std::cout << (gpass ? "✅ CUDA Graph validation PASSED\n" : "❌ CUDA Graph validation FAILED\n");

    start = std::chrono::high_resolution_clock::now();
    for (int i=0; i<iters; ++i) {
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    end = std::chrono::high_resolution_clock::now();
    double graph_us = std::chrono::duration<double, std::micro>(end-start).count()/iters;

    std::cout<<"⏱️  With CUDA Graphs:    "<<graph_us<<" µs\n";
    std::cout<<"   Speedup from graphs:  "<<no_graph_us/graph_us<<"x\n";
    std::cout<<"   PyTorch GPU target:   784.1 µs (n=50-trial mean, see benchmark_pytorch_block3_stats.py)\n";

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

    bool all_pass = pass && gpass;
    return all_pass ? 0 : 1;
}
