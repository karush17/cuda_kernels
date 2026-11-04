#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)


#define TILE 16
__global__ void matmul_tiled(const float* A, const float* B, float* C,
                             int M, int N, int K) {     // C[MxN] = A[MxK] * B[KxN]
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.f;
        __syncthreads();                                // tile fully loaded

        for (int k = 0; k < TILE; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();                                // done reading before overwrite
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

// ---- launcher for torch ----
torch::Tensor matmul(torch::Tensor A, torch::Tensor B) {
    // 1. validate
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "inputs must be tensors");
    TORCH_CHECK(A.device() == B.device(), "inputs must be on the same device");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32, "inputs must have same dtype");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "inputs must have 2 dims");
    TORCH_CHECK(A.size(1) == B.size(0), "inner dimensions must agree");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "inputs must be continuous");
    const int64_t kMax = std::numeric_limits<int>::max();
    TORCH_CHECK(A.size(0) < kMax && A.size(1) < kMax && B.size(1) < kMax, "dimensions too large for indexing");

    // 2. device + stream
    const at::cuda::OptionalCUDAGuard guard(A.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const int m = A.size(0), k = A.size(1), n = B.size(1);

    // 3. allocate the output
    auto c = torch::empty({m, n}, A.options());

    // 4. degenerate shapes
    if (m == 0 || n == 0) return c;
    if (k == 0) return c.zero_();

    // 5. launch
    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);
    TORCH_CHECK(grid.y <= 65535, "m too large for grid limit");
    matmul_tiled<<<grid, block, 0, stream>>>(A.data_ptr<float>(), B.data_ptr<float>(), c.data_ptr<float>(), m, n, k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return c;
}