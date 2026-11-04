#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <type_traits>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include "common.cuh"
using namespace nvcuda;

// per batch: C[M x N] = scale * A[M x K] * B[K x N]   (B_TRANS: B stored as [N x K])
// M, N, K must be multiples of 16. 1D blocks; one warp per 16x16 output tile.
template <bool B_TRANS>
__global__ void wmma_matmul(const __half* A, const __half* B, __half* C,
                            int M, int N, int K, float scale) {
    using BLayout = std::conditional_t<B_TRANS, wmma::col_major, wmma::row_major>;

    const size_t bz = blockIdx.z;
    A += bz * M * K;
    B += bz * K * N;
    C += bz * M * N;

    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int tilesN = N / 16;
    const int tileRow = warp_id / tilesN, tileCol = warp_id % tilesN;
    if (tileRow >= M / 16) return;               // whole warp exits together

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, BLayout> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.f);

    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(a, A + (size_t)tileRow * 16 * K + k, K);
        const __half* bptr = B_TRANS ? B + (size_t)tileCol * 16 * K + k
                                     : B + (size_t)k * N + tileCol * 16;
        wmma::load_matrix_sync(b, bptr, B_TRANS ? K : N);
        wmma::mma_sync(acc, a, b, acc);
    }

    // convert float acc -> half tile via shared memory staging
    __shared__ float stage[4][16 * 16];          // 4 warps per block
    const int lw = threadIdx.x >> 5, lane = threadIdx.x & 31;
    wmma::store_matrix_sync(stage[lw], acc, 16, wmma::mem_row_major);
    __syncwarp();
    __half* ctile = C + (size_t)tileRow * 16 * N + tileCol * 16;
    for (int i = lane; i < 256; i += 32)
        ctile[(i / 16) * N + (i % 16)] = __float2half(stage[lw][i] * scale);
}

// one block per row; compute in float; columns >= valid are masked (prob = 0)
__global__ void softmax(const __half* in, __half* out, int cols, int valid) {
    const __half* x = in + (size_t)blockIdx.x * cols;
    __half* y = out + (size_t)blockIdx.x * cols;
    float m = -INFINITY;
    for (int j = threadIdx.x; j < valid; j += blockDim.x) m = fmaxf(m, __half2float(x[j]));
    m = block_reduce<MaxOp>(m);
    float s = 0.f;
    for (int j = threadIdx.x; j < valid; j += blockDim.x) s += expf(__half2float(x[j]) - m);
    s = block_reduce<SumOp>(s);
    float inv = 1.f / s;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        y[j] = __float2half(j < valid ? expf(__half2float(x[j]) - m) * inv : 0.f);
}


int main() {
    const int B = 4, s = 10, S = 16, d = 256;    // s = real length, S = padded
    const size_t bytes_sd = (size_t)B * S * d * sizeof(__half);
    const size_t bytes_ss = (size_t)B * S * S * sizeof(__half);

    std::vector<__half> h_q(B * S * d, __float2half(0.f)), h_k = h_q, h_v = h_q;
    std::vector<__half> h_out(B * S * d);
    srand(0);
    for (int b = 0; b < B; ++b)
        for (int i = 0; i < s; ++i)              // only real rows; padding stays 0
            for (int t = 0; t < d; ++t) {
                size_t idx = ((size_t)b * S + i) * d + t;
                h_q[idx] = __float2half(2.f * rand() / (float)RAND_MAX - 1.f);
                h_k[idx] = __float2half(2.f * rand() / (float)RAND_MAX - 1.f);
                h_v[idx] = __float2half(2.f * rand() / (float)RAND_MAX - 1.f);
            }

    __half *d_q, *d_k, *d_v, *d_qk, *d_soft, *d_out;
    CUDA_CHECK(cudaMalloc(&d_q, bytes_sd));  CUDA_CHECK(cudaMalloc(&d_k, bytes_sd));
    CUDA_CHECK(cudaMalloc(&d_v, bytes_sd));  CUDA_CHECK(cudaMalloc(&d_qk, bytes_ss));
    CUDA_CHECK(cudaMalloc(&d_soft, bytes_ss)); CUDA_CHECK(cudaMalloc(&d_out, bytes_sd));
    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), bytes_sd, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), bytes_sd, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), bytes_sd, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_qk, 0xFF, bytes_ss));
    CUDA_CHECK(cudaMemset(d_soft, 0xFF, bytes_ss));
    CUDA_CHECK(cudaMemset(d_out, 0xFF, bytes_sd));

    const int TPB = 128;                         // 4 warps per block
    auto grid_for = [&](int M, int N) {
        int tiles = (M / 16) * (N / 16);
        return dim3((tiles * 32 + TPB - 1) / TPB, 1, B);
    };
    int sm_threads = 32;                         // S=16 columns -> one warp is plenty

    wmma_matmul<true><<<grid_for(S, S), TPB>>>(d_q, d_k, d_qk, S, S, d, 1.f / sqrtf((float)d));
    softmax<<<B * S, sm_threads>>>(d_qk, d_soft, S, s);
    wmma_matmul<false><<<grid_for(S, d), TPB>>>(d_soft, d_v, d_out, S, d, S, 1.f);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes_sd, cudaMemcpyDeviceToHost));

    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v);
    cudaFree(d_qk); cudaFree(d_soft); cudaFree(d_out);
}

