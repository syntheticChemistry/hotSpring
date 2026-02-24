// SPDX-License-Identifier: AGPL-3.0-only
//
// CUDA FP64:FP32 ratio micro-benchmark — identical workload to the WGSL version.
// Measures raw ALU throughput using a dependent FMA chain.
//
// Build:  nvcc -O3 -arch=sm_86 bench_fp64_ratio.cu -o bench_fp64_ratio_cuda
//         (sm_86 for GA102/RTX 3090, sm_70 for GV100/Titan V)
//
// Run:    ./bench_fp64_ratio_cuda
//         CUDA_VISIBLE_DEVICES=1 ./bench_fp64_ratio_cuda  (for 2nd GPU)

#include <stdio.h>
#include <cuda_runtime.h>

#define CHAIN_LENGTH 4096
#define N_THREADS    4194304
#define BLOCK_SIZE   256
#define WARMUP       3
#define MEASURE      10

#define CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

__global__ void fma_chain_f32(float *output) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_THREADS) return;

    float acc = (float)i * 1.0000001f;
    float m = 1.0000001f;
    float a = 0.0000001f;

    for (int j = 0; j < CHAIN_LENGTH; j++) {
        acc = acc * m + a;
    }

    output[i] = acc;
}

__global__ void fma_chain_f64(double *output) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_THREADS) return;

    double acc = (double)i * 1.0000001;
    double m = 1.0000001;
    double a = 0.0000001;

    for (int j = 0; j < CHAIN_LENGTH; j++) {
        acc = acc * m + a;
    }

    output[i] = acc;
}

int main() {
    int dev;
    cudaDeviceProp prop;
    CHECK(cudaGetDevice(&dev));
    CHECK(cudaGetDeviceProperties(&prop, dev));

    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  CUDA FP64:FP32 Ratio — Pure FMA Chain Micro-Benchmark     ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    printf("  GPU: %s\n", prop.name);
    printf("  SM count: %d\n", prop.multiProcessorCount);
    printf("  Clock: %d MHz\n", prop.clockRate / 1000);
    printf("  Memory: %zu MB\n", prop.totalGlobalMem / (1024*1024));
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("\n");
    printf("  Threads:      %12d\n", N_THREADS);
    printf("  FMA chain:    %12d ops/thread\n", CHAIN_LENGTH);
    printf("  Total FLOPs:  %12llu (2 FLOP per FMA x chain x threads)\n",
           2ULL * CHAIN_LENGTH * N_THREADS);
    printf("  Warmup:       %12d rounds\n", WARMUP);
    printf("  Measure:      %12d rounds\n", MEASURE);
    printf("\n");

    double total_flops = 2.0 * (double)CHAIN_LENGTH * (double)N_THREADS;
    int grid = (N_THREADS + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // FP32 benchmark
    float *d_f32;
    CHECK(cudaMalloc(&d_f32, (size_t)N_THREADS * sizeof(float)));

    for (int i = 0; i < WARMUP; i++)
        fma_chain_f32<<<grid, BLOCK_SIZE>>>(d_f32);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaEventRecord(start));
    for (int i = 0; i < MEASURE; i++)
        fma_chain_f32<<<grid, BLOCK_SIZE>>>(d_f32);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float f32_ms = 0;
    CHECK(cudaEventElapsedTime(&f32_ms, start, stop));
    f32_ms /= MEASURE;
    double f32_tflops = total_flops / (f32_ms * 1e-3) / 1e12;

    printf("── FP32 FMA chain ──\n");
    printf("  Time:       %.3f ms\n", f32_ms);
    printf("  Throughput: %.2f TFLOPS (fp32)\n\n", f32_tflops);

    CHECK(cudaFree(d_f32));

    // FP64 benchmark
    double *d_f64;
    CHECK(cudaMalloc(&d_f64, (size_t)N_THREADS * sizeof(double)));

    for (int i = 0; i < WARMUP; i++)
        fma_chain_f64<<<grid, BLOCK_SIZE>>>(d_f64);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaEventRecord(start));
    for (int i = 0; i < MEASURE; i++)
        fma_chain_f64<<<grid, BLOCK_SIZE>>>(d_f64);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float f64_ms = 0;
    CHECK(cudaEventElapsedTime(&f64_ms, start, stop));
    f64_ms /= MEASURE;
    double f64_tflops = total_flops / (f64_ms * 1e-3) / 1e12;

    printf("── FP64 FMA chain ──\n");
    printf("  Time:       %.3f ms\n", f64_ms);
    printf("  Throughput: %.2f TFLOPS (fp64)\n\n", f64_tflops);

    CHECK(cudaFree(d_f64));

    double ratio = f32_tflops / f64_tflops;
    printf("══════════════════════════════════════════════════════════════\n");
    printf("  RESULT: fp64:fp32 = 1:%.1f\n\n", ratio);
    printf("  FP32: %.2f TFLOPS\n", f32_tflops);
    printf("  FP64: %.2f TFLOPS\n", f64_tflops);
    printf("  Ratio: 1:%.1f (fp64 is %.1fx slower than fp32)\n", ratio, ratio);
    printf("══════════════════════════════════════════════════════════════\n");

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    return 0;
}
