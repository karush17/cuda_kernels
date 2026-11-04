#include "common.cuh"
#include <vector>
#include <algorithm>

__global__ void rmsnorm(const float* in, const float* gamma, float* out, int cols, float eps) {
    __shared__ float s_sq;
    
    int tid = threadIdx.x;
    const float* x = in + (size_t)blockIdx.x * cols;
    float* y = out + (size_t)blockIdx.x * cols;

    // 1. load input
    if (tid == 0) {
        s_sq = 0.0f;
    }
    __syncthreads();
    float v = (tid < cols) ? x[tid] : 0.0f;

    // 2. reduce contributions
    atomicAdd(&s_sq, v * v);
    __syncthreads();

    // 3. calculate statistics
    float mean_sq = s_sq / cols;
    float inv_rms = rsqrtf(mean_sq + eps);

    // 4. normalize and apply transformations
    if (tid < cols) {
        float norm = (v * inv_rms);
        y[tid] = norm * gamma[tid];
    }

}

int main() {
    const int rows = 4096, cols = 1000;
    float eps = 1e-6;
    size_t count = (size_t)rows * cols, bytes = count * sizeof(float);
    size_t col_count = (size_t)cols, col_bytes = col_count * sizeof(float);
    std::vector<float> h_in(count), h_out(count);
    std::vector<float> gamma(col_count);
    for (auto& v : h_in) v = (rand() / float(RAND_MAX) - 0.5f) * 20.f;
    for (auto& v : gamma) v = (rand() / float(RAND_MAX) - 0.5f) * 20.f;

    float *d_in, *d_out, *d_gamma, *d_beta;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_gamma, col_bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, gamma.data(), col_bytes, cudaMemcpyHostToDevice));

    int threads = std::min(1024, std::max(32, (cols + 31) / 32 * 32));
    CUDA_CHECK(cudaMemset(d_out, 0xFF, bytes));
    rmsnorm<<<rows, threads>>>(d_in, d_gamma, d_out, cols, eps);
    CUDA_CHECK(cudaGetLastError());                    // catches launch errors
    CUDA_CHECK(cudaDeviceSynchronize());               // catches execution errors
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    float ms = time_ms([&] {rmsnorm<<<rows, threads>>>(d_in, d_gamma, d_out, cols, eps); });
    printf("time - %.3f ms\n", ms);

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_gamma);
}
