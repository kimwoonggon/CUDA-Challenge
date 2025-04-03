#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define N (1<<24)
#define THREADS 1024

__global__ void fp64(const double* input, int8_t* output, float scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) 
    {
        double x = input[idx];
        int q = __float2int_rn((float)(x / scale));
        output[idx] = max(-128, min(127, q));
    }
}

__global__ void fp32(const float* input, int8_t* output, float scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) 
    {
        float x = input[idx];
        int q = __float2int_rn(x / scale);
        output[idx] = max(-128, min(127, q));
    }
}

__global__ void fp16(const half* input, int8_t* output, float scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) 
    {
        float x = __half2float(input[idx]);
        int q = __float2int_rn(x / scale);
        output[idx] = max(-128, min(127, q));
    }
}

void init(double* d64, float* d32, half* d16)
{
    for (int i = 0; i < N; ++i) 
    {
        float val = sinf(i * 0.001f) * 10.0f;
        d64[i] = (double)val;
        d32[i] = val;
        d16[i] = __float2half(val);
    }
}

int main()
{
    double* d_fp64;
    float*  d_fp32;
    half*   d_fp16;
    int8_t *d_out64, *d_out32, *d_out16;
    cudaMallocManaged(&d_fp64, N * sizeof(double));
    cudaMallocManaged(&d_fp32, N * sizeof(float));
    cudaMallocManaged(&d_fp16, N * sizeof(half));
    cudaMallocManaged(&d_out64, N * sizeof(int8_t));
    cudaMallocManaged(&d_out32, N * sizeof(int8_t));
    cudaMallocManaged(&d_out16, N * sizeof(int8_t));

    init(d_fp64, d_fp32, d_fp16);

    dim3 threads(THREADS);
    dim3 blocks((N + THREADS - 1) / THREADS);
    float scale = 0.1f;

    cudaEvent_t start, stop;
    float time_fp64 = 0, time_fp32 = 0, time_fp16 = 0;

    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    fp64<<<blocks, threads>>>(d_fp64, d_out64, scale);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_fp64, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    fp32<<<blocks, threads>>>(d_fp32, d_out32, scale);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_fp32, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    fp16<<<blocks, threads>>>(d_fp16, d_out16, scale);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_fp16, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    printf("FP64 -> int8: %.3f ms\n", time_fp64);
    printf("FP32 -> int8: %.3f ms\n", time_fp32);
    printf("FP16 -> int8: %.3f ms\n", time_fp16);

    cudaFree(d_fp64); 
    cudaFree(d_fp32); 
    cudaFree(d_fp16);
    cudaFree(d_out64); 
    cudaFree(d_out32); 
    cudaFree(d_out16);

    return 0;
}
