#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <vector>
#include "common.cuh"


// C[m x n] = scale * A[m x k] * B[k x n]   (B_TRANS: B stored as [n x k])
template <bool B_TRANS>
__global__ void matmul_naive(const float* a, const float* b, float* c,
                             int m, int n, int k, float scale) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < m && col < n) {
    float acc = 0.f;
    for (int t = 0; t < k; ++t) {
      float bv = B_TRANS ? b[col * k + t] : b[t * n + col];
      acc += a[row * k + t] * bv;
    }
    c[row * n + col] = acc * scale;
  }
}

__global__ void softmax(const float* in, float* out, int cols) {
    const float* x = in + (size_t)blockIdx.x * cols;
    float* y = out + (size_t)blockIdx.x * cols;
    float m = -INFINITY;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) m = fmaxf(m, x[j]);
    m = block_reduce<MaxOp>(m);
    float s = 0.f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) s += expf(x[j] - m);
    s = block_reduce<SumOp>(s);
    float inv = 1.f / s;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) y[j] = expf(x[j] - m) * inv;
}


int main() {
    const int s = 10;
    const int d = 256;
    const size_t bytes_sd = (size_t)s * d * sizeof(float);
    const size_t bytes_ss = (size_t)s * s * sizeof(float);

    // Host (CPU) data
    std::vector<float> h_q((size_t)s * d), h_k((size_t)s * d), h_v((size_t)s * d);
    std::vector<float> h_qk((size_t)s * s), h_soft((size_t)s * s), h_out((size_t)s * d);
    srand(0);
    auto rnd = [] { return 2.f * rand() / (float)RAND_MAX - 1.f; };
    for (auto& x : h_q) x = rnd();
    for (auto& x : h_k) x = rnd();
    for (auto& x : h_v) x = rnd();
    
    // Device (GPU) data
    float *d_q, *d_k, *d_v, *d_qk, *d_soft, *d_out;
    CUDA_CHECK(cudaMalloc(&d_q, bytes_sd));
    CUDA_CHECK(cudaMalloc(&d_k, bytes_sd));
    CUDA_CHECK(cudaMalloc(&d_v, bytes_sd));
    CUDA_CHECK(cudaMalloc(&d_qk, bytes_ss));
    CUDA_CHECK(cudaMalloc(&d_soft, bytes_ss));
    CUDA_CHECK(cudaMalloc(&d_out, bytes_sd));

    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), bytes_sd, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), bytes_sd, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), bytes_sd, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_qk, 0xFF, bytes_ss));
    CUDA_CHECK(cudaMemset(d_soft, 0xFF, bytes_ss));
    CUDA_CHECK(cudaMemset(d_out, 0xFF, bytes_sd));

    dim3 block(16, 16);
    dim3 grid_qk((s + 15) / 16, (s + 15) / 16);
    dim3 grid_pv((d + 15) / 16, (s + 15) / 16);
    int threads = std::min(1024, std::max(32, (d + 31) / 32 * 32));

    matmul_naive<true><<<grid_qk, block>>>(d_q, d_k, d_qk, s, s, d, 1 / sqrtf((float)d));
    softmax<<<s, threads>>>(d_qk, d_soft, s);
    matmul_naive<false><<<grid_pv, block>>>(d_soft, d_v, d_out, s, d, s, 1.f);
    CUDA_CHECK(cudaGetLastError());                    // catches launch errors
    CUDA_CHECK(cudaDeviceSynchronize());               // catches execution errors

    CUDA_CHECK(cudaMemcpy(h_qk.data(), d_qk, bytes_ss, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_soft.data(), d_soft, bytes_ss, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes_sd, cudaMemcpyDeviceToHost));
    
    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v);
    cudaFree(d_qk); cudaFree(d_soft); cudaFree(d_out);

}