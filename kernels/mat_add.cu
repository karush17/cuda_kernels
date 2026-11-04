#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_fp16.h>
#include "common.cuh"

__global__ void mat_add(const float* a, const float* b, float* c, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        for (int t = 0; t < n; ++t) {c[row * n + t] = a[row * n + t] + b[row * n + t]; }
    }
}


int main() {
    const int m = 128;
    const int n = 256;
    const int threads = 16;
    const size_t bytes = (size_t)m * n * sizeof(float);

    // Host (CPU) data
    std::vector<float> h_a((size_t)m * n), h_b((size_t)m * n), h_c((size_t)m * n);
    for (size_t i = 0; i < h_a.size(); ++i) {
        h_a[i] = 1.0f;
    }
    for (size_t i = 0; i < h_b.size(); ++i) {
        h_b[i] = 2.0f;
    }

    // Device (GPU) data
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_c, 0xFF, bytes));

    // launch
    dim3 block(threads, threads);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    mat_add<<<grid, block>>>(d_a, d_b, d_c, m, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < h_c.size(); ++i) {
        if (h_c[i] != 3.0f) {printf("mismatch"); return 1; }
    }
    printf("OK");

    cudaFree(d_a); cudaFree(d_b);cudaFree(d_c);

}


