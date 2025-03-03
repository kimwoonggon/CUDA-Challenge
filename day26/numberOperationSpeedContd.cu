#include <curand_kernel.h>
#include <stdint.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#define N (1<<26)
#define THREADS 1024
#define BLOCKS ((N + THREADS - 1) / THREADS)

//compute range
void compute_range_double(double* arr, int n, double *minVal, double *maxVal) 
{
    *minVal = arr[0];
    *maxVal = arr[0];
    for (int i = 1; i < n; i++) 
    {
        if(arr[i] < *minVal) *minVal = arr[i];
        if(arr[i] > *maxVal) *maxVal = arr[i];
    }
}

void compute_range_float(float* arr, int n, float *minVal, float *maxVal) 
{
    *minVal = arr[0];
    *maxVal = arr[0];
    for (int i = 1; i < n; i++) 
    {
        if(arr[i] < *minVal) *minVal = arr[i];
        if(arr[i] > *maxVal) *maxVal = arr[i];
    }
}

void compute_range_half(half* arr, int n, float *minVal, float *maxVal) 
{
    *minVal = __half2float(arr[0]);
    *maxVal = __half2float(arr[0]);
    for (int i = 1; i < n; i++) {
        float val = __half2float(arr[i]);
        if(val < *minVal) *minVal = val;
        if(val > *maxVal) *maxVal = val;
    }
}

void compute_range_int8(int8_t* arr, int n, int *minVal, int *maxVal) 
{
    *minVal = arr[0];
    *maxVal = arr[0];
    for (int i = 1; i < n; i++) {
        if(arr[i] < *minVal) *minVal = arr[i];
        if(arr[i] > *maxVal) *maxVal = arr[i];
    }
}

// Init
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

// ReLU
__global__ void ReLU_double(double *vect)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N)
        vect[tid] = (vect[tid] >= 0) ? vect[tid] : 0;
}

__global__ void ReLU_float(float *vect)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N)
        vect[tid] = (vect[tid] >= 0) ? vect[tid] : 0;
}

__global__ void ReLU_half(half *vect)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N)
        vect[tid] = __hge(vect[tid], __float2half_rn(0.0f)) ? vect[tid] : __float2half_rn(0.0f);
}

__global__ void ReLU_int(int8_t *vect)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N)
        vect[tid] = (vect[tid] >= 0) ? vect[tid] : 0;
}

// Sigmoid
__global__ void sigmoid_double(double *vect)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N)
    {
        double x = vect[tid];
        vect[tid] = 1.0 / (1.0 + exp(-x));
    }
}

__global__ void sigmoid_float(float *vect)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N)
    {
        float x = vect[tid];
        vect[tid] = 1.0f / (1.0f + expf(-x));
    }
}

__global__ void sigmoid_half(half *vect)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N)
    {
        float x = __half2float(vect[tid]);
        float y = 1.0f / (1.0f + expf(-x));
        vect[tid] = __float2half_rn(y);
    }
}

__global__ void sigmoid_int(int8_t *vect)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N)
    {
        float x = (float)vect[tid];
        float y = 1.0f / (1.0f + expf(-x));
        vect[tid] = (int8_t)(y * 127.0f);
    }
}

// tanh
__global__ void tanh_double(double *vect)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N)
        vect[tid] = tanh(vect[tid]);
}

__global__ void tanh_float(float *vect)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N)
        vect[tid] = tanhf(vect[tid]);
}

__global__ void tanh_half(half *vect)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N)
    {
        float x = __half2float(vect[tid]);
        vect[tid] = __float2half_rn(tanhf(x));
    }
}

__global__ void tanh_int(int8_t *vect)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N)
    {
        float x = (float)vect[tid];
        float y = tanhf(x);
        vect[tid] = (int8_t)(y * 127.0f);
    }
}

// Linear
__global__ void linear_double(double *vect)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N)
        vect[tid] = vect[tid];
}

__global__ void linear_float(float *vect)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N)
        vect[tid] = vect[tid];
}

__global__ void linear_half(half *vect)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N)
        vect[tid] = vect[tid];
}

__global__ void linear_int(int8_t *vect)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N)
        vect[tid] = vect[tid];
}

// LeakyReLU
__global__ void leakyrelu_double(double *vect)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N)
    {
        double x = vect[tid];
        vect[tid] = (x >= 0.0) ? x : 0.01 * x;
    }
}

__global__ void leakyrelu_float(float *vect)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N)
    {
        float x = vect[tid];
        vect[tid] = (x >= 0.0f) ? x : 0.01f * x;
    }
}

__global__ void leakyrelu_half(half *vect)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N)
    {
        float x = __half2float(vect[tid]);
        float y = (x >= 0.0f) ? x : 0.01f * x;
        vect[tid] = __float2half_rn(y);
    }
}

__global__ void leakyrelu_int(int8_t *vect)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N)
    {
        int x = vect[tid];
        int y = (x >= 0) ? x : (int)(0.01f * x);
        vect[tid] = (int8_t)y;
    }
}

// GELU
__global__ void gelu_double(double *vect)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N)
    {
        double x = vect[tid];
        double c = sqrt(2.0 / 3.14159265358979323846);
        double inner = c * (x + 0.044715 * x * x * x);
        vect[tid] = 0.5 * x * (1.0 + tanh(inner));
    }
}

__global__ void gelu_float(float *vect)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N)
    {
        float x = vect[tid];
        float c = sqrtf(2.0f / 3.14159265358979323846f);
        float inner = c * (x + 0.044715f * x * x * x);
        vect[tid] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

__global__ void gelu_half(half *vect)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N)
    {
        float x = __half2float(vect[tid]);
        float c = sqrtf(2.0f / 3.14159265358979323846f);
        float inner = c * (x + 0.044715f * x * x * x);
        float y = 0.5f * x * (1.0f + tanhf(inner));
        vect[tid] = __float2half_rn(y);
    }
}

__global__ void gelu_int(int8_t *vect)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < N)
    {
        float x = (float)vect[tid];
        float c = sqrtf(2.0f / 3.14159265358979323846f);
        float inner = c * (x + 0.044715f * x * x * x);
        float y = 0.5f * x * (1.0f + tanhf(inner));
        vect[tid] = (int8_t)(y * 127.0f);
    }
}


int main()
{
    double *vect_fp64;
    float  *vect_fp32;
    half   *vect_fp16;
    int8_t *vect_int8;

    cudaMallocManaged(&vect_fp64, N * sizeof(double));
    cudaMallocManaged(&vect_fp32, N * sizeof(float));
    cudaMallocManaged(&vect_fp16, N * sizeof(half));
    cudaMallocManaged(&vect_int8, N * sizeof(int8_t));

    unsigned long seed = time(NULL);
    float time_ms, flops;

    cudaEvent_t start, stop;

    double dmin, dmax;
    float fmin, fmax, hmin, hmax;
    int imin, imax;

    //ReLU
    init<<<BLOCKS, THREADS>>>((void*)vect_fp64, 0, seed);
    cudaDeviceSynchronize();
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    ReLU_double<<<BLOCKS, THREADS>>>(vect_fp64);
    cudaDeviceSynchronize();
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    flops = ((float)N / time_ms) * 1e-6f;
    compute_range_double(vect_fp64, N, &dmin, &dmax);
    printf("ReLU FP64: Time = %f ms, GFLOPS = %f, Range = [%f, %f]\n", time_ms, flops, dmin, dmax);
    FILE *fp0 = fopen("/mnt/d/CUDA/day26/relu_fp64.bin", "wb"); 
    fwrite(vect_fp64, sizeof(double), N, fp0); 
    fclose(fp0);

    init<<<BLOCKS, THREADS>>>((void*)vect_fp32, 1, seed);
    cudaDeviceSynchronize();
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    ReLU_float<<<BLOCKS, THREADS>>>(vect_fp32);
    cudaDeviceSynchronize();
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    flops = ((float)N / time_ms) * 1e-6f;
    compute_range_float(vect_fp32, N, &fmin, &fmax);
    printf("ReLU FP32: Time = %f ms, GFLOPS = %f, Range = [%f, %f]\n", time_ms, flops, fmin, fmax);

    init<<<BLOCKS, THREADS>>>((void*)vect_fp16, 2, seed);
    cudaDeviceSynchronize();
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    ReLU_half<<<BLOCKS, THREADS>>>(vect_fp16);
    cudaDeviceSynchronize();
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    flops = ((float)N / time_ms) * 1e-6f;
    compute_range_half(vect_fp16, N, &hmin, &hmax);
    printf("ReLU FP16: Time = %f ms, GFLOPS = %f, Range = [%f, %f]\n", time_ms, flops, hmin, hmax);

    init_int8<<<BLOCKS, THREADS>>>((void*)vect_int8, seed);
    cudaDeviceSynchronize();
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    ReLU_int<<<BLOCKS, THREADS>>>(vect_int8);
    cudaDeviceSynchronize();
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    flops = ((float)N / time_ms) * 1e-6f;
    compute_range_int8(vect_int8, N, &imin, &imax);
    printf("ReLU INT8: Time = %f ms, GFLOPS = %f, Range = [%d, %d]\n\n", time_ms, flops, imin, imax);
    FILE *fp1 = fopen("/mnt/d/CUDA/day26/relu_int8.bin", "wb"); 
    fwrite(vect_int8, sizeof(int8_t), N, fp1); 
    fclose(fp1);


    //sigmoid
    init<<<BLOCKS, THREADS>>>((void*)vect_fp64, 0, seed);
    cudaDeviceSynchronize();
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    sigmoid_double<<<BLOCKS, THREADS>>>(vect_fp64);
    cudaDeviceSynchronize();
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    flops = ((float)N / time_ms) * 1e-6f;
    compute_range_double(vect_fp64, N, &dmin, &dmax);
    printf("Sigmoid FP64: Time = %f ms, GFLOPS = %f, Range = [%f, %f]\n", time_ms, flops, dmin, dmax);
    FILE *fp2 = fopen("/mnt/d/CUDA/day26/sigmoid_fp64.bin", "wb"); 
    fwrite(vect_fp64, sizeof(double), N, fp2); 
    fclose(fp2);

    init<<<BLOCKS, THREADS>>>((void*)vect_fp32, 1, seed);
    cudaDeviceSynchronize();
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    sigmoid_float<<<BLOCKS, THREADS>>>(vect_fp32);
    cudaDeviceSynchronize();
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    flops = ((float)N / time_ms) * 1e-6f;
    compute_range_float(vect_fp32, N, &fmin, &fmax);
    printf("Sigmoid FP32: Time = %f ms, GFLOPS = %f, Range = [%f, %f]\n", time_ms, flops, fmin, fmax);

    init<<<BLOCKS, THREADS>>>((void*)vect_fp16, 2, seed);
    cudaDeviceSynchronize();
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    sigmoid_half<<<BLOCKS, THREADS>>>(vect_fp16);
    cudaDeviceSynchronize();
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    flops = ((float)N / time_ms) * 1e-6f;
    compute_range_half(vect_fp16, N, &hmin, &hmax);
    printf("Sigmoid FP16: Time = %f ms, GFLOPS = %f, Range = [%f, %f]\n", time_ms, flops, hmin, hmax);

    init_int8<<<BLOCKS, THREADS>>>((void*)vect_int8, seed);
    cudaDeviceSynchronize();
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    sigmoid_int<<<BLOCKS, THREADS>>>(vect_int8);
    cudaDeviceSynchronize();
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    flops = ((float)N / time_ms) * 1e-6f;
    compute_range_int8(vect_int8, N, &imin, &imax);
    printf("Sigmoid INT8: Time = %f ms, GFLOPS = %f, Range = [%d, %d]\n\n", time_ms, flops, imin, imax);
    FILE *fp3 = fopen("/mnt/d/CUDA/day26/sigmoid_int8.bin", "wb"); 
    fwrite(vect_int8, sizeof(int8_t), N, fp3); 
    fclose(fp3);


    //tanh
    init<<<BLOCKS, THREADS>>>((void*)vect_fp64, 0, seed);
    cudaDeviceSynchronize();
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    tanh_double<<<BLOCKS, THREADS>>>(vect_fp64);
    cudaDeviceSynchronize();
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    flops = ((float)N / time_ms) * 1e-6f;
    compute_range_double(vect_fp64, N, &dmin, &dmax);
    printf("Tanh FP64: Time = %f ms, GFLOPS = %f, Range = [%f, %f]\n", time_ms, flops, dmin, dmax);
    FILE *fp4 = fopen("/mnt/d/CUDA/day26/tanh_fp64.bin", "wb"); 
    fwrite(vect_fp64, sizeof(double), N, fp4); 
    fclose(fp4);

    init<<<BLOCKS, THREADS>>>((void*)vect_fp32, 1, seed);
    cudaDeviceSynchronize();
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    tanh_float<<<BLOCKS, THREADS>>>(vect_fp32);
    cudaDeviceSynchronize();
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    flops = ((float)N / time_ms) * 1e-6f;
    compute_range_float(vect_fp32, N, &fmin, &fmax);
    printf("Tanh FP32: Time = %f ms, GFLOPS = %f, Range = [%f, %f]\n", time_ms, flops, fmin, fmax);

    init<<<BLOCKS, THREADS>>>((void*)vect_fp16, 2, seed);
    cudaDeviceSynchronize();
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    tanh_half<<<BLOCKS, THREADS>>>(vect_fp16);
    cudaDeviceSynchronize();
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    flops = ((float)N / time_ms) * 1e-6f;
    compute_range_half(vect_fp16, N, &hmin, &hmax);
    printf("Tanh FP16: Time = %f ms, GFLOPS = %f, Range = [%f, %f]\n", time_ms, flops, hmin, hmax);

    init_int8<<<BLOCKS, THREADS>>>((void*)vect_int8, seed);
    cudaDeviceSynchronize();
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    tanh_int<<<BLOCKS, THREADS>>>(vect_int8);
    cudaDeviceSynchronize();
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    flops = ((float)N / time_ms) * 1e-6f;
    compute_range_int8(vect_int8, N, &imin, &imax);
    printf("Tanh INT8: Time = %f ms, GFLOPS = %f, Range = [%d, %d]\n\n", time_ms, flops, imin, imax);
    FILE *fp5 = fopen("/mnt/d/CUDA/day26/tanh_int8.bin", "wb"); 
    fwrite(vect_int8, sizeof(int8_t), N, fp5); 
    fclose(fp5);


    //linear
    init<<<BLOCKS, THREADS>>>((void*)vect_fp64, 0, seed);
    cudaDeviceSynchronize();
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    linear_double<<<BLOCKS, THREADS>>>(vect_fp64);
    cudaDeviceSynchronize();
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    flops = ((float)N / time_ms) * 1e-6f;
    compute_range_double(vect_fp64, N, &dmin, &dmax);
    printf("Linear FP64: Time = %f ms, GFLOPS = %f, Range = [%f, %f]\n", time_ms, flops, dmin, dmax);
    FILE *fp6 = fopen("/mnt/d/CUDA/day26/linear_fp64.bin", "wb"); 
    fwrite(vect_fp64, sizeof(double), N, fp6); 
    fclose(fp6);

    init<<<BLOCKS, THREADS>>>((void*)vect_fp32, 1, seed);
    cudaDeviceSynchronize();
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    linear_float<<<BLOCKS, THREADS>>>(vect_fp32);
    cudaDeviceSynchronize();
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    flops = ((float)N / time_ms) * 1e-6f;
    compute_range_float(vect_fp32, N, &fmin, &fmax);
    printf("Linear FP32: Time = %f ms, GFLOPS = %f, Range = [%f, %f]\n", time_ms, flops, fmin, fmax);

    init<<<BLOCKS, THREADS>>>((void*)vect_fp16, 2, seed);
    cudaDeviceSynchronize();
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    linear_half<<<BLOCKS, THREADS>>>(vect_fp16);
    cudaDeviceSynchronize();
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    flops = ((float)N / time_ms) * 1e-6f;
    compute_range_half(vect_fp16, N, &hmin, &hmax);
    printf("Linear FP16: Time = %f ms, GFLOPS = %f, Range = [%f, %f]\n", time_ms, flops, hmin, hmax);

    init_int8<<<BLOCKS, THREADS>>>((void*)vect_int8, seed);
    cudaDeviceSynchronize();
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    linear_int<<<BLOCKS, THREADS>>>(vect_int8);
    cudaDeviceSynchronize();
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    flops = ((float)N / time_ms) * 1e-6f;
    compute_range_int8(vect_int8, N, &imin, &imax);
    printf("Linear INT8: Time = %f ms, GFLOPS = %f, Range = [%d, %d]\n\n", time_ms, flops, imin, imax);
    FILE *fp7 = fopen("/mnt/d/CUDA/day26/linear_int8.bin", "wb"); 
    fwrite(vect_int8, sizeof(int8_t), N, fp7); 
    fclose(fp7);


    //LeakyReLU
    init<<<BLOCKS, THREADS>>>((void*)vect_fp64, 0, seed);
    cudaDeviceSynchronize();
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    leakyrelu_double<<<BLOCKS, THREADS>>>(vect_fp64);
    cudaDeviceSynchronize();
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    flops = ((float)N / time_ms) * 1e-6f;
    compute_range_double(vect_fp64, N, &dmin, &dmax);
    printf("LeakyReLU FP64: Time = %f ms, GFLOPS = %f, Range = [%f, %f]\n", time_ms, flops, dmin, dmax);
    FILE *fp8 = fopen("/mnt/d/CUDA/day26/leakyrelu_fp64.bin", "wb"); 
    fwrite(vect_fp64, sizeof(double), N, fp8); 
    fclose(fp8);

    init<<<BLOCKS, THREADS>>>((void*)vect_fp32, 1, seed);
    cudaDeviceSynchronize();
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    leakyrelu_float<<<BLOCKS, THREADS>>>(vect_fp32);
    cudaDeviceSynchronize();
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    flops = ((float)N / time_ms) * 1e-6f;
    compute_range_float(vect_fp32, N, &fmin, &fmax);
    printf("LeakyReLU FP32: Time = %f ms, GFLOPS = %f, Range = [%f, %f]\n", time_ms, flops, fmin, fmax);

    init<<<BLOCKS, THREADS>>>((void*)vect_fp16, 2, seed);
    cudaDeviceSynchronize();
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    leakyrelu_half<<<BLOCKS, THREADS>>>(vect_fp16);
    cudaDeviceSynchronize();
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    flops = ((float)N / time_ms) * 1e-6f;
    compute_range_half(vect_fp16, N, &hmin, &hmax);
    printf("LeakyReLU FP16: Time = %f ms, GFLOPS = %f, Range = [%f, %f]\n", time_ms, flops, hmin, hmax);

    init_int8<<<BLOCKS, THREADS>>>((void*)vect_int8, seed);
    cudaDeviceSynchronize();
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    leakyrelu_int<<<BLOCKS, THREADS>>>(vect_int8);
    cudaDeviceSynchronize();
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    flops = ((float)N / time_ms) * 1e-6f;
    compute_range_int8(vect_int8, N, &imin, &imax);
    printf("LeakyReLU INT8: Time = %f ms, GFLOPS = %f, Range = [%d, %d]\n\n", time_ms, flops, imin, imax);
    FILE *fp9 = fopen("/mnt/d/CUDA/day26/leakyrelu_int8.bin", "wb"); 
    fwrite(vect_int8, sizeof(int8_t), N, fp9); 
    fclose(fp9);


    //GeLU
    init<<<BLOCKS, THREADS>>>((void*)vect_fp64, 0, seed);
    cudaDeviceSynchronize();
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    gelu_double<<<BLOCKS, THREADS>>>(vect_fp64);
    cudaDeviceSynchronize();
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    flops = ((float)N / time_ms) * 1e-6f;
    compute_range_double(vect_fp64, N, &dmin, &dmax);
    printf("GELU FP64: Time = %f ms, GFLOPS = %f, Range = [%f, %f]\n", time_ms, flops, dmin, dmax);
    FILE *fp10 = fopen("/mnt/d/CUDA/day26/gelu_fp64.bin", "wb"); 
    fwrite(vect_fp64, sizeof(double), N, fp10); 
    fclose(fp10);

    init<<<BLOCKS, THREADS>>>((void*)vect_fp32, 1, seed);
    cudaDeviceSynchronize();
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    gelu_float<<<BLOCKS, THREADS>>>(vect_fp32);
    cudaDeviceSynchronize();
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    flops = ((float)N / time_ms) * 1e-6f;
    compute_range_float(vect_fp32, N, &fmin, &fmax);
    printf("GELU FP32: Time = %f ms, GFLOPS = %f, Range = [%f, %f]\n", time_ms, flops, fmin, fmax);

    init<<<BLOCKS, THREADS>>>((void*)vect_fp16, 2, seed);
    cudaDeviceSynchronize();
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    gelu_half<<<BLOCKS, THREADS>>>(vect_fp16);
    cudaDeviceSynchronize();
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    flops = ((float)N / time_ms) * 1e-6f;
    compute_range_half(vect_fp16, N, &hmin, &hmax);
    printf("GELU FP16: Time = %f ms, GFLOPS = %f, Range = [%f, %f]\n", time_ms, flops, hmin, hmax);

    init_int8<<<BLOCKS, THREADS>>>((void*)vect_int8, seed);
    cudaDeviceSynchronize();
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    gelu_int<<<BLOCKS, THREADS>>>(vect_int8);
    cudaDeviceSynchronize();
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    flops = ((float)N / time_ms) * 1e-6f;
    compute_range_int8(vect_int8, N, &imin, &imax);
    printf("GELU INT8: Time = %f ms, GFLOPS = %f, Range = [%d, %d]\n", time_ms, flops, imin, imax);
    FILE *fp11 = fopen("/mnt/d/CUDA/day26/gelu_int8.bin", "wb"); 
    fwrite(vect_int8, sizeof(int8_t), N, fp11); 
    fclose(fp11);

    cudaFree(vect_fp64);
    cudaFree(vect_fp32);
    cudaFree(vect_fp16);
    cudaFree(vect_int8);

    return 0;
}
