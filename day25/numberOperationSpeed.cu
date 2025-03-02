#include <curand_kernel.h>
#include <stdint.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <time.h>

#define N (1<<26)
#define THREADS 1024
#define BLOCKS ((N + THREADS - 1) / THREADS)

__global__ void init(void *vect, int type, unsigned long seed) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) 
    {
        curandState state;
        curand_init(seed, tid, 0, &state);

        if (type == 0)
        { 
            double rand_val = curand_uniform_double(&state);
            ((double*)vect)[tid] = rand_val * 2000.0 - 1000.0;
        } 
        else if (type == 1)
        { 
            float rand_val = curand_uniform(&state);
            ((float*)vect)[tid] = rand_val * 2000.0f - 1000.0f;
        } 
        else if (type == 2)
        { 
            float rand_val = curand_uniform(&state);
            float scaled = rand_val * 2000.0f - 1000.0f;
            ((half*)vect)[tid] = __float2half_rn(scaled);
        } 
    }
}

__global__ void init_int8(void *vect, unsigned long seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) 
    {
        curandState state;
        curand_init(seed, tid, 0, &state);
        int rand_int = curand(&state);
        ((int8_t*)vect)[tid] = (int8_t)((rand_int % 256) - 128);
    }
}

__global__ void ReLU_float(float *vect)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N)
    {
        vect[tid] = (vect[tid] >= 0) ? vect[tid] : 0;
    }
}

__global__ void ReLU_double(double *vect)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N)
    {
        vect[tid] = (vect[tid] >= 0) ? vect[tid] : 0;
    }
}

__global__ void ReLU_half(half *vect)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N)
    {
        vect[tid] = __hge(vect[tid], __float2half_rn(0.0f)) ? vect[tid] : __float2half_rn(0.0f);
    }
}

__global__ void ReLU_int(int8_t *vect)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N)
    {
        vect[tid] = (vect[tid] >= 0) ? vect[tid] : 0;
    }
}

int main()
{
    double *vect_fp64;
    float *vect_fp32;
    half *vect_fp16;
    int8_t *vect_int8;

    cudaMallocManaged(&vect_fp64, N * sizeof(double));
    cudaMallocManaged(&vect_fp32, N * sizeof(float));
    cudaMallocManaged(&vect_fp16, N * sizeof(half));
    cudaMallocManaged(&vect_int8, N * sizeof(int8_t));

    init<<<BLOCKS, THREADS>>>((void*)vect_fp64, 0, time(NULL));
    cudaDeviceSynchronize();

    init<<<BLOCKS, THREADS>>>((void*)vect_fp32, 1, time(NULL));
    cudaDeviceSynchronize();

    init<<<BLOCKS, THREADS>>>((void*)vect_fp16, 2, time(NULL));
    cudaDeviceSynchronize();

    init_int8<<<BLOCKS, THREADS>>>((void*)vect_int8, time(NULL));
    cudaDeviceSynchronize();

    float time_fp64, time_fp32, time_fp16, time_int8;
    float flops_fp64, flops_fp32, flops_fp16, flops_int8;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    ReLU_double<<<BLOCKS, THREADS>>>(vect_fp64);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_fp64, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    flops_fp64 = ((float)N / time_fp64) * 1e-6f;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    ReLU_float<<<BLOCKS, THREADS>>>(vect_fp32);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_fp32, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    flops_fp32 = ((float)N / time_fp32) * 1e-6f;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    ReLU_half<<<BLOCKS, THREADS>>>(vect_fp16);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_fp16, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    flops_fp16 = ((float)N / time_fp16) * 1e-6f;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    ReLU_int<<<BLOCKS, THREADS>>>(vect_int8);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_int8, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    flops_int8 = ((float)N / time_int8) * 1e-6f;

    printf("ReLU FP64: Time = %f ms, GFLOPS = %f\n", time_fp64, flops_fp64);
    printf("ReLU FP32: Time = %f ms, GFLOPS = %f\n", time_fp32, flops_fp32);
    printf("ReLU FP16: Time = %f ms, GFLOPS = %f\n", time_fp16, flops_fp16);
    printf("ReLU INT8: Time = %f ms, GFLOPS = %f\n", time_int8, flops_int8);

    cudaFree(vect_fp64);
    cudaFree(vect_fp32);
    cudaFree(vect_fp16);
    cudaFree(vect_int8);

    return 0;
}