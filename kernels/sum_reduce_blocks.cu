#include "common.cuh"
#include <vector>
#include <numeric>

__global__ void sum_reduce_shfl(const float* in, float* out, int n) {
    float v = 0.f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) v += in[i];
    v = block_reduce<SumOp>(v);
    if (threadIdx.x == 0) atomicAdd(out, v);
}


int main() {
    const int n = 1 << 26;
   std::vector<float> h(n);
   for (int i = 0; i < n; ++i) h[i] = rand() / (float)RAND_MAX;
   double ref = std::accumulate(h.begin(), h.end(), 0.0);

   float *d_in, *d_out;
   CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
   CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
   CUDA_CHECK(cudaMemcpy(d_in, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));

   const int threads = 256;
   const int blocks = 1024;

   auto run = [&] {
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
    sum_reduce_shfl<<<blocks, threads>>>(d_in, d_out, n);
   };
   run();
   float got;
   CUDA_CHECK(cudaMemcpy(&got, d_out, sizeof(float), cudaMemcpyDeviceToHost));
   printf("GPU=%.1f ref=%.1f error=%.1f\n", got, ref, fabs(got - ref) / ref);

   float ms = time_ms(run);
   printf("time taken: %.3f\n", ms);

    cudaFree(d_in); cudaFree(d_out);

}
