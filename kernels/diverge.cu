#include "common.cuh"

__global__ void diverge(float* out, int iters, int mode) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    float x = (float)t;
    // mode 0: even/odd lanes split INSIDE every warp.  mode 1: whole warps take the same side.
    bool cond = (mode == 0) ? (threadIdx.x & 1) : ((threadIdx.x >> 5) & 1);
    if (cond) { for (int i = 0; i < iters; ++i) x = fmaf(x, 1.0001f, 0.5f); }
    else      { for (int i = 0; i < iters; ++i) x = fmaf(x, 0.9999f, 0.25f); }
    out[t] = x;
}

int main() {
    const int blocks = 1024, threads = 256, iters = 20000;
    float* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)blocks * threads * sizeof(float)));
    for (int mode = 0; mode < 2; ++mode) {
        float ms = time_ms([&] { diverge<<<blocks, threads>>>(d_out, iters, mode); }, 10, 3);
        printf("mode %d (%s): %.3f ms\n", mode, mode == 0 ? "divergent" : "uniform per warp", ms);
    }
}