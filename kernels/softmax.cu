#include "common.cuh"
#include <vector>
#include <algorithm>

__global__ void softmax(const float* in, float* out, int cols) {
    const float* x = in + (size_t)blockIdx.x * cols;
    float* y = out + (size_t)blockIdx.x * cols;

    float m = -INFINITY;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) m = fmaxf(m, x[j]);
    m = block_reduce<MaxOp>(m);

    float s = 0.f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) s += expf(x[j] - m);
    s = block_reduce<SumOp>(s);

    float inv = 1.f / s;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) y[j] = expf(x[j] - m) * inv;
}

void softmax_cpu(const float* in, float* out, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const float* x = in + (size_t)r * cols;
        float* y = out + (size_t)r * cols;
        double m = -INFINITY;
        double s = 0;
        for (int j = 0; j < cols; ++j) m = std::max(m, (double)x[j]);
        for (int j = 0; j < cols; ++j) s += exp((double)x[j] - m);
        for (int j = 0; j < cols; ++j) y[j] = (float)(exp((double)x[j] - m) / s);
    }
}


int main() {
    const int rows = 4096, cols = 1000;
    size_t count = (size_t)rows * cols, bytes = count * sizeof(float);
    std::vector<float> h_in(count), h_out(count), h_ref(count);
    for (auto& v : h_in) v = (rand() / float(RAND_MAX) - 0.5f) * 20.f;
    softmax_cpu(h_in.data(), h_ref.data(), rows, cols);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    int threads = std::min(1024, std::max(32, (cols + 31) / 32 * 32));
    CUDA_CHECK(cudaMemset(d_out, 0xFF, bytes));
    softmax<<<rows, threads>>>(d_in, d_out, cols);
    CUDA_CHECK(cudaGetLastError());                    // catches launch errors
    CUDA_CHECK(cudaDeviceSynchronize());               // catches execution errors
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    float max_err = 0.f; int bad = 0;
    for (size_t i = 0; i < count; ++i) {
        float d = fabsf(h_out[i] - h_ref[i]);
        if (!(d <= 1e-6f)) ++bad; else max_err = fmaxf(max_err, d);
    }
    printf("bad=%d max_err=%2.f\n", bad, max_err);

    float ms = time_ms([&] {softmax<<<rows, threads>>>(d_in, d_out, cols); });
    printf("time - %.3f ms\n", ms);
}
