#include "common.cuh"

__global__ void fma_peak(float* out, int iters, float a, float b) {
    float x0 = threadIdx.x, x1 = x0 + 1, x2 = x0 + 2, x3 = x0 + 3,
          x4 = x0 + 4,      x5 = x0 + 5, x6 = x0 + 6, x7 = x0 + 7;
    for (int i = 0; i < iters; ++i) {                  // 8 independent dependency chains = instruction-level parallelism
        x0 = fmaf(x0, a, b); x1 = fmaf(x1, a, b); x2 = fmaf(x2, a, b); x3 = fmaf(x3, a, b);
        x4 = fmaf(x4, a, b); x5 = fmaf(x5, a, b); x6 = fmaf(x6, a, b); x7 = fmaf(x7, a, b);
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;  // keeps the compiler from deleting the work
}

int main() {
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    int threads = 256, blocks = p.multiProcessorCount * 8, iters = 100000;
    float* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)blocks * threads * sizeof(float)));

    float ms = time_ms([&] { fma_peak<<<blocks, threads>>>(d_out, iters, 0.999f, 0.001f); }, 10, 3);
    double flops = (double)blocks * threads * iters * 8 * 2;     // 8 FMAs per iteration, 2 FLOPs per FMA
    printf("%.3f ms, %.1f GFLOPS (%.2f TFLOPS)\n", ms, gflops(flops, ms), gflops(flops, ms) / 1e3);
}