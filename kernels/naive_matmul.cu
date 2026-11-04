#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)


__global__ void matmul_naive(const float* a, const float* b, float* c, int m, int n, int k) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m && col < n) {
    float acc = 0.0f;
    for (int t = 0; t < k; ++t) acc += a[row * k + t] * b[t * n + col];
    c[row * n + col] = acc;
  }
}


int main() {
    const int m = 1000;
    const int n = 70;
    const int k = 50;
    const size_t bytes_a = (size_t)m * k * sizeof(float);
    const size_t bytes_b = (size_t)n * k * sizeof(float);
    const size_t bytes_c = (size_t)m * n * sizeof(float);

    // Host (CPU) data
    float* h_a = (float*)malloc(bytes_a);
    float* h_b = (float*)malloc(bytes_b);
    float* h_c = (float*)malloc(bytes_c);
    for (int i = 0; i < m * k; ++i) h_a[i] = 1.0f;
    for (int i = 0; i < n * k; ++i) h_b[i] = 2.0f;

    
    // Device (GPU) data
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes_a));
    CUDA_CHECK(cudaMalloc(&d_b, bytes_b));
    CUDA_CHECK(cudaMalloc(&d_c, bytes_c));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes_a, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes_b, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_c, 0xFF, bytes_c));

    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    matmul_naive<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
    CUDA_CHECK(cudaGetLastError());                    // catches launch errors
    CUDA_CHECK(cudaDeviceSynchronize());               // catches execution errors

    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes_c, cudaMemcpyDeviceToHost));
    
    if (h_c[0] != 100.0f) { printf("Mismatch"); return 1; }
    printf("OK\n");

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
}