#include "common.cuh"
#include <vector>
#include <algorithm>

__global__ void layernorm(const float* in, const float* gamma, const float* beta, float* out, int cols, float eps) {
    const float* x = in + (size_t)blockIdx.x * cols;
    float* y = out + (size_t)blockIdx.x * cols;

    float s = 0.f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) s += x[j];
    float mean = block_reduce<SumOp>(s) / cols;

    float sq= 0.f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float d = x[j] - mean;
        sq += d * d;
    }
    float var = block_reduce<SumOp>(sq) / cols;
    float rstd = rsqrtf(var + eps);

    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        y[j] = (x[j] - mean) * rstd * gamma[j] + beta[j];
    }
}

void layernorm_cpu(const float* in, const float* gamma, const float* beta, float* out, int rows, int cols, float eps) {
    for (int r = 0; r < rows; ++r) {
        const float* x = in + (size_t)r * cols;
        double mean = 0; for (int j = 0; j < cols; ++j) mean += x[j]; mean /= cols;
        double var = 0; for (int j = 0; j < cols; ++j) var += (x[j] - mean) * (x[j] - mean); var /= cols;
        double rstd = 1.0 / sqrt(var + eps);
        for (int j = 0; j < cols; ++j) out[(size_t)r * cols + j] = (float)(x[j] - mean)* rstd * gamma[j] + beta[j];
    }
}

int main() {
    const int rows = 4096, cols = 1000;
    float eps = 1e-6;
    size_t count = (size_t)rows * cols, bytes = count * sizeof(float);
    size_t col_count = (size_t)cols, col_bytes = col_count * sizeof(float);
    std::vector<float> h_in(count), h_out(count), h_ref(count);
    std::vector<float> gamma(col_count), beta(col_count);
    for (auto& v : h_in) v = (rand() / float(RAND_MAX) - 0.5f) * 20.f;
    for (auto& v : gamma) v = (rand() / float(RAND_MAX) - 0.5f) * 20.f;
    for (auto& v : beta) v = (rand() / float(RAND_MAX) - 0.5f) * 20.f;
    layernorm_cpu(h_in.data(), gamma.data(), beta.data(), h_ref.data(), rows, cols, eps);

    float *d_in, *d_out, *d_gamma, *d_beta;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_gamma, col_bytes));
    CUDA_CHECK(cudaMalloc(&d_beta, col_bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, gamma.data(), col_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, beta.data(), col_bytes, cudaMemcpyHostToDevice));

    int threads = std::min(1024, std::max(32, (cols + 31) / 32 * 32));
    CUDA_CHECK(cudaMemset(d_out, 0xFF, bytes));
    layernorm<<<rows, threads>>>(d_in, d_gamma, d_beta, d_out, cols, eps);
    CUDA_CHECK(cudaGetLastError());                    // catches launch errors
    CUDA_CHECK(cudaDeviceSynchronize());               // catches execution errors
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    float max_err = 0.f; int bad = 0;
    for (size_t i = 0; i < count; ++i) {
        float d = fabsf(h_out[i] - h_ref[i]);
        if (!(d <= 1e-6f)) ++bad; else max_err = fmaxf(max_err, d);
    }
    printf("bad=%d max_err=%2.f\n", bad, max_err);

    float ms = time_ms([&] {layernorm<<<rows, threads>>>(d_in, d_gamma, d_beta, d_out, cols, eps); });
    printf("time - %.3f ms\n", ms);
}
