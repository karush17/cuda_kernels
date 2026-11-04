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

__global__ void sum_reduce_warp(const float* in, float* out, int n) {
  extern __shared__ float s[];
  int tid = threadIdx.x;
  int lane = tid % 32;
  int warp = tid / 32;
  float v = (tid < n) ? in[tid]: 0.f;

  // reudce values within warp
  for (int offset = 16; offset > 0; offset /= 2) {
    float other = __shfl_down_sync(
      0xffffffff,
      v,
      offset
    );
    v += other;
  }
  __syncthreads();
  
  // reduce across warps
  if (lane == 0) atomicAdd(out, v);
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
    CUDA_CHECK(cudaMemset(d_b, 0, bytes));

    // Launch: <<<number of blocks, threads per block>>>
    int threads = 256;
    int blocks = 32;
    sum_reduce_warp<<<blocks, threads>>>(d_a, d_b, n);
    CUDA_CHECK(cudaGetLastError());                    // catches launch errors
    CUDA_CHECK(cudaDeviceSynchronize());               // catches execution errors

    CUDA_CHECK(cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost));

    if (h_b[0] == 0.0f) { printf("Mismatch"); return 1; }
    printf("OK\n");

    cudaFree(d_a); cudaFree(d_b);
}