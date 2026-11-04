#include <cstdio>
#include <cstdlib>
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

__global__ void sum_reduce(const float* in, float* out, int n) {
  extern __shared__ float s[];
  int tid = threadIdx.x;
  
  float v = 0.f;
  for (int i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x)
    v += in[i];
  s[tid] = v;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) s[tid] += s[tid + stride];
    __syncthreads();
  }

  if (tid == 0) atomicAdd(out, s[0]);
}


int main() {
    const int n = 1 << 20;
    const size_t bytes = n * sizeof(float);

    // Host (CPU) data
    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    for (int i = 0; i < n; ++i) { h_a[i] = 1.0f; h_b[i] = 0.0f; }

    // Device (GPU) data
    float *d_a, *d_b;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // Launch: <<<number of blocks, threads per block>>>
    int threads = 256;
    int blocks = (n + threads - 1) / threads;         // ceiling division
    sum_reduce<<<blocks, threads, threads * sizeof(float)>>>(d_a, d_b, n);
    CUDA_CHECK(cudaGetLastError());                    // catches launch errors
    CUDA_CHECK(cudaDeviceSynchronize());               // catches execution errors

    CUDA_CHECK(cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost));

    if (h_b[0] == 0.0f) { printf("Mismatch"); return 1; }
    printf("OK\n");

    cudaFree(d_a); cudaFree(d_b);
    free(h_a); free(h_b);
}