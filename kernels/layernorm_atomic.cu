#include "common.cuh"
#include <vector>
#include <algorithm>

__global__ void layernorm(const float* in, const float* gamma, const float* beta, float* out, int cols, float eps) {
    __shared__ float s_sum;
    __shared__ float s_sq;

    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* x = in + row * cols;
    float* y = out + row * cols;

    // 1. load input
    if (tid == 0) {
        s_sum = 0.0f;
        s_sq = 0.0f;
    }
    __syncthreads();
    float v = (tid < cols) ? x[tid] : 0.0f;

    // 2. reduce contributions
    atomicAdd(&s_sum, v);
    atomicAdd(&s_sq, v*v);
    __syncthreads();

    // 3. calculate statistics
    float mean = s_sum / cols;
    float mean_sq = s_sq / cols;
    float var = fmaxf(mean_sq - mean * mean, 0.0f);
    float inv_std = rsqrtf(var + eps);

    // 4. normalize and apply transformation
    if (tid < cols) {
        float norm = (v - mean) * inv_std;
        y[tid] = norm * gamma[tid] + beta[tid];
    }
}

int main() {
    const int cols = 256;
    float eps = 1e-6;
    size_t count = (size_t)cols, bytes = count * sizeof(float);
    size_t col_count = (size_t)cols, col_bytes = col_count * sizeof(float);
    std::vector<float> h_in(count), h_out(count), h_ref(count);
    std::vector<float> gamma(col_count), beta(col_count);
    for (auto& v : h_in) v = (rand() / float(RAND_MAX) - 0.5f) * 20.f;
    for (auto& v : gamma) v = (rand() / float(RAND_MAX) - 0.5f) * 20.f;
    for (auto& v : beta) v = (rand() / float(RAND_MAX) - 0.5f) * 20.f;

    float *d_in, *d_out, *d_gamma, *d_beta;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_gamma, col_bytes));
    CUDA_CHECK(cudaMalloc(&d_beta, col_bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, gamma.data(), col_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, beta.data(), col_bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    CUDA_CHECK(cudaMemset(d_out, 0xFF, bytes));
    layernorm<<<1, threads>>>(d_in, d_gamma, d_beta, d_out, cols, eps);
    CUDA_CHECK(cudaGetLastError());                    // catches launch errors
    CUDA_CHECK(cudaDeviceSynchronize());               // catches execution errors
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    float ms = time_ms([&] {layernorm<<<1, threads>>>(d_in, d_gamma, d_beta, d_out, cols, eps); });
    printf("time - %.3f ms\n", ms);
}
