#include "common.cuh"
#include <vector>


#define TILE 32

__global__ void naive_transpose(const float* in, float* out, int m, int n) {
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < n && y < m) out[x * m + y] = in[y * n + x];
}


// launch:
// dim3 block(TILE, TILE);
// dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
// matmul_tiled<<<grid, block>>>(dA, dB, dC, M, N, K);


int main() {
    const int m = 4096;
    const int n = 2048;
    const size_t count = (size_t)m * n;
    const size_t bytes_a = count * sizeof(float);

    // Host (CPU) data
    std::vector<float> h_in(bytes_a), h_out(bytes_a);
    for (size_t i = 0; i < count; ++i) h_in[i] = (float)i;

    
    // Device (GPU) data
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes_a));
    CUDA_CHECK(cudaMalloc(&d_out, bytes_a));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes_a, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    CUDA_CHECK(cudaMemset(d_out, 0xFF, bytes_a));
    naive_transpose<<<grid, block>>>(d_in, d_out, m, n);
    CUDA_CHECK(cudaGetLastError());                    // catches launch errors
    CUDA_CHECK(cudaDeviceSynchronize());               // catches execution errors
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes_a, cudaMemcpyDeviceToHost));

    float ms = time_ms([&] {naive_transpose<<<grid, block>>>(d_in, d_out, m, n); });
    printf("time taken: %.3f ms\n", ms);

    cudaFree(d_in); cudaFree(d_out);
}