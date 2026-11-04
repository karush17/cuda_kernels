#include "common.cuh"
#include <vector>
#include <algorithm>

__global__ void softmax_vector(const float* in, float* out, int n) {
    __shared__ float shared[1024];
    int tid = threadIdx.x;

    // 1. load input data
    float v = (tid < n) ? in[tid]: -INFINITY;
    shared[tid] = v;
    __syncthreads();

    // 2. max reduction
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }
    float max_val = shared[0];

    // 3. exponent
    float e = (tid < n) ? expf(in[tid] - max_val) : 0.0f;

    // 4. sum reduction
    shared[tid] = e;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    float denominator = shared[0];

    // 5. normalize
    if (tid < n) {
        out[tid] = e / denominator;
    }
}


int main() {
    const int cols = 1024;
    size_t count = (size_t)cols, bytes = count * sizeof(float);
    std::vector<float> h_in(count), h_out(count), h_ref(count);
    for (auto& v : h_in) v = (rand() / float(RAND_MAX) - 0.5f) * 20.f;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    int threads = 1024;
    CUDA_CHECK(cudaMemset(d_out, 0xFF, bytes));
    softmax_vector<<<1, threads>>>(d_in, d_out, cols);
    CUDA_CHECK(cudaGetLastError());                    // catches launch errors
    CUDA_CHECK(cudaDeviceSynchronize());               // catches execution errors
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    float ms = time_ms([&] {softmax_vector<<<1, threads>>>(d_in, d_out, cols); });
    printf("time - %.3f ms\n", ms);
}
