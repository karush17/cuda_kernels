#include <mma.h>
#include <vector>
#include <cuda_fp16.h>
#include "common.cuh"
using namespace nvcuda;

__global__ void wmma_matmul(const __half* A, const __half* B, float* C, int M, int N, int K) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int tilesN = N / 16;
    const int tileRow = warp_id / tilesN, tileCol = warp_id % tilesN;
    if (tileRow >= M / 16) return;                       // whole warp exits together: safe

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.f);

    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(a, A + (size_t)tileRow * 16 * K + k, K);
        wmma::load_matrix_sync(b, B + (size_t)k * N + tileCol * 16, N);
        wmma::mma_sync(acc, a, b, acc);                  // 16x16x16 multiply-accumulate in one warp-wide op
    }
    wmma::store_matrix_sync(C + (size_t)tileRow * 16 * N + tileCol * 16, acc, N, wmma::mem_row_major);
}

int main() {
    const int m = 128;
    const int n = 32;
    const int k = 32;
    const int threads = 128;
    const int warps = (m / 16) * (n / 16);
    const int blocks = (warps * 32 + threads - 1) / 128;
    const size_t bytes_a = (size_t)m * k * sizeof(__half);
    const size_t bytes_b = (size_t)k * n * sizeof(__half);
    const size_t bytes_c = (size_t)m * n * sizeof(float);

    // Host (CPU) data
    std::vector<__half> h_a((size_t)m * k), h_b((size_t)k * n);
    std::vector<float> h_c((size_t)m * n), a_f((size_t)m * k), b_f((size_t)k * n), c_f((size_t)m * n);
    srand(0);
    for (size_t i = 0; i < h_a.size(); ++i) {
        h_a[i] = __float2half(2.f * rand() / (float)RAND_MAX - 1.f);
        a_f[i] = __half2float(h_a[i]);
    }
    for (size_t i = 0; i < h_b.size(); ++i) {
        h_b[i] = __float2half(2.f * rand() / (float)RAND_MAX - 1.f);
        b_f[i] = __half2float(h_b[i]);
    }
    for (int r = 0; r < m; ++r) {
        for (int c = 0; c < n; ++c) {
            double acc = 0.0;
            for (int t = 0; t < k; ++t) acc += (double)a_f[(size_t)r * k + t] * b_f[(size_t)t * n + c];
            c_f[(size_t)r * n + c] = (float)acc;
        }
    }

    // Device (GPU) data
    __half *d_a, *d_b;
    float *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes_a));
    CUDA_CHECK(cudaMalloc(&d_b, bytes_b));
    CUDA_CHECK(cudaMalloc(&d_c, bytes_c));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes_a, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes_b, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_c, 0xFF, bytes_c));

    wmma_matmul<<<blocks, threads>>>(d_a, d_b, d_c, m, n, k);
    CUDA_CHECK(cudaGetLastError());                    // catches launch errors
    CUDA_CHECK(cudaDeviceSynchronize());               // catches execution errors

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes_c, cudaMemcpyDeviceToHost));
    size_t bad = 0;
    for (size_t i = 0; i < h_c.size(); ++i)
        if (!(fabs((double)h_c[i] - c_f[i]) <= 1e-3 + 1e-3 * fabs((double)c_f[i]))) ++bad;
    printf("bad elements: %zu / %zu \n", bad, h_c.size());
    printf(bad == 0 ? "OK\n" : "MISMATCH\n");

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

// launch: threads = 128; warps_total = (M/16)*(N/16); blocks = (warps_total*32 + 127) / 128;