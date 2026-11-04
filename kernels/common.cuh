#pragma once
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)


inline double gbps(double bytes, float ms)   { return bytes / (ms * 1e-3) / 1e9; }
inline double gflops(double flops, float ms) { return flops / (ms * 1e-3) / 1e9; }

// Cold-cache timing: flush L2 before every iteration, report the MEDIAN of per-iteration times.
// flush_bytes should be at least 2x your GPU's L2 size (prop.l2CacheSize).
template <typename F>
float time_ms_cold(F&& launch, void* flush_buf, size_t flush_bytes,
                   int iters = 50, int warmup = 5) {
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaGetLastError());
    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));
    std::vector<float> times(iters);
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaMemsetAsync(flush_buf, i & 0xFF, flush_bytes));   // evicts previous data from L2
        CUDA_CHECK(cudaEventRecord(s));
        launch();
        CUDA_CHECK(cudaEventRecord(e));
        CUDA_CHECK(cudaEventSynchronize(e));
        CUDA_CHECK(cudaEventElapsedTime(&times[i], s, e));
    }
    CUDA_CHECK(cudaEventDestroy(s));
    CUDA_CHECK(cudaEventDestroy(e));
    std::sort(times.begin(), times.end());
    return times[iters / 2];
}

// ---- timing utility ----
template <typename F>
float time_ms(F&& launch, int iters = 10, int warmup = 1){
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaGetLastError());
    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));
    CUDA_CHECK(cudaEventRecord(s));
    for (int i = 0; i < iters; ++i) launch();
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
    CUDA_CHECK(cudaEventDestroy(s));
    CUDA_CHECK(cudaEventDestroy(e));
    return ms / iters;
}

// ---- reduciton operators ----
struct SumOp {
    __device__ __forceinline__ float operator()(float a, float b) const {return a + b; }
    __device__ __forceinline__ static float identity() {return 0.f; }
};

struct MaxOp {
    __device__ __forceinline__ float operator()(float a, float b) const {return fmaxf(a, b); }
    __device__ __forceinline__ static float identity() {return -INFINITY; }
};


// ---- warp-level reduce: result valid in lane 0 ----
template <typename Op>
__device__ __forceinline__ float warp_reduce(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v = Op()(v, __shfl_down_sync(0xffffffffu, v, offset));
    return v;
}

// ---- block-level reduce: result returned to ALL threads ----
// Requirements: blockDim.x is a multiple of 32 and <= 1024; every thread calls it.
template <typename Op>
__device__ float block_reduce(float v) {
    __shared__ float warp_vals[32];
    __shared__ float result;
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;

    v = warp_reduce<Op>(v);                       // 1) reduce within each warp
    if (lane == 0) warp_vals[wid] = v;            // 2) one value per warp -> shared
    __syncthreads();

    if (wid == 0) {                               // 3) first warp reduces the warp results
        v = (lane < nwarps) ? warp_vals[lane] : Op::identity();
        v = warp_reduce<Op>(v);
        if (lane == 0) result = v;
    }
    __syncthreads();
    float r = result;                             // 4) broadcast to everyone
    __syncthreads();                              // 5) safe to reuse shared vars on the next call
    return r;
}