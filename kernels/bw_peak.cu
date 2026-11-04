#include "common.cuh"

__global__ void copy_f4(const float4* in, float4* out, size_t n4) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride)
        out[i] = in[i];
}

int main() {
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    const size_t n = (size_t)1 << 27;                   // 128M floats = 512 MB per buffer (reduce if your GPU is small)
    const size_t bytes = n * sizeof(float);
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemset(d_in, 1, bytes));

    int threads = 256;
    int blocks = p.multiProcessorCount * 8;             // enough blocks to fill the GPU; grid-stride does the rest
    float ms = time_ms([&] { copy_f4<<<blocks, threads>>>((float4*)d_in, (float4*)d_out, n / 4); });
    printf("copy_f4:            %.3f ms, %.1f GB/s\n", ms, gbps(2.0 * bytes, ms));

    float ms2 = time_ms([&] { cudaMemcpyAsync(d_out, d_in, bytes, cudaMemcpyDeviceToDevice); });
    printf("cudaMemcpy D2D:     %.3f ms, %.1f GB/s\n", ms2, gbps(2.0 * bytes, ms2));
    printf("L2 size: %.1f MB\n", p.l2CacheSize / 1048576.0);
}