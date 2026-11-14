// Ground-truth workload for the bench end-to-end test (tests/test_bench.py).
// Inside the capture range: 1000 kernel launches, 3 H2D + 2 D2H memcpys, 4 cudaStreamSynchronize.
#include <cstdio>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

__global__ void step(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] * 0.999f + 1.0f;
}

int main() {
    const int n = 1 << 16;
    float *d, h[16] = {0};
    cudaMalloc(&d, n * sizeof(float));
    cudaStream_t s;
    cudaStreamCreate(&s);
    cudaProfilerStart();
    for (int k = 0; k < 3; k++) cudaMemcpyAsync(d, h, sizeof(h), cudaMemcpyHostToDevice, s);
    for (int k = 0; k < 1000; k++) step<<<n / 256, 256, 0, s>>>(d, n);
    for (int k = 0; k < 2; k++) cudaMemcpyAsync(h, d, sizeof(h), cudaMemcpyDeviceToHost, s);
    for (int k = 0; k < 4; k++) cudaStreamSynchronize(s);
    cudaProfilerStop();
    size_t free_b, total_b;
    cudaMemGetInfo(&free_b, &total_b);
    printf("BENCH_RESULT {\"steps\": 1000, \"seconds\": 1.0, \"peak_vram_bytes\": %zu}\n", total_b - free_b);
    return 0;
}
