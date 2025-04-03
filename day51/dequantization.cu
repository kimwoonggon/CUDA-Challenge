#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define N (1 << 24)
#define THREADS 1024

__global__ void to_fp64(const int8_t* input, double* output, float scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) 
    {
        output[idx] = (double)(input[idx]) * scale;
    }
}

__global__ void to_fp32(const int8_t* input, float* output, float scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) 
    {
        output[idx] = (float)(input[idx]) * scale;
    }
}

__global__ void to_fp16(const int8_t* input, half* output, float scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) 
    {
        float val = (float)(input[idx]) * scale;
        output[idx] = __float2half(val);
    }
}

void init(int8_t* data)
{
    for (int i = 0; i < N; ++i) 
    {
        float val = sinf(i * 0.001f) * 10.0f;
        int q = (int)roundf(val / 0.1f);
        data[i] = max(-128, min(127, q));
    }
}

int main()
{
    int8_t* d_in;
    double* d_out64;
    float*  d_out32;
    half*   d_out16;

    cudaMallocManaged(&d_in, N * sizeof(int8_t));
    cudaMallocManaged(&d_out64, N * sizeof(double));
    cudaMallocManaged(&d_out32, N * sizeof(float));
    cudaMallocManaged(&d_out16, N * sizeof(half));

    init(d_in);

    dim3 threads(THREADS);
    dim3 blocks((N + THREADS - 1) / THREADS);
    float scale = 0.1f;

    cudaEvent_t start, stop;
    float time_fp64 = 0, time_fp32 = 0, time_fp16 = 0;

    cudaEventCreate(&start); 
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    to_fp64<<<blocks, threads>>>(d_in, d_out64, scale);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_fp64, start, stop);
    cudaEventDestroy(start); 
    cudaEventDestroy(stop);

    cudaEventCreate(&start); 
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    to_fp32<<<blocks, threads>>>(d_in, d_out32, scale);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_fp32, start, stop);
    cudaEventDestroy(start); 
    cudaEventDestroy(stop);

    cudaEventCreate(&start); 
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    to_fp16<<<blocks, threads>>>(d_in, d_out16, scale);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_fp16, start, stop);
    cudaEventDestroy(start); 
    cudaEventDestroy(stop);

    printf("int8 -> FP64: %.3f ms\n", time_fp64);
    printf("int8 -> FP32: %.3f ms\n", time_fp32);
    printf("int8 -> FP16: %.3f ms\n", time_fp16);

    cudaFree(d_in);
    cudaFree(d_out64);
    cudaFree(d_out32);
    cudaFree(d_out16);

    return 0;
}
