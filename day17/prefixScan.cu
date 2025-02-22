#include <stdio.h>
#include <curand_kernel.h>
#include <time.h>

#define N (1<<26)
#define THREADS 1024
#define BLOCKS ((N + THREADS - 1) / THREADS)

void koggeCPU(int *x, int *y) 
{
    y[0] = x[0];
    for (int i = 1; i < N; i++) 
    {
        y[i] = y[i - 1] + x[i];
    }
}

__global__ void koggeGPU(int *x, int *y, int *partialSums) 
{
    int t_i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int x_s[THREADS];

    if (t_i < N)
    {
        x_s[threadIdx.x] = x[t_i];
    }
    else
    {
        x_s[threadIdx.x] = 0;
    }
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) 
    {
        __syncthreads();
        int temp = 0;
        if (threadIdx.x >= stride) 
        {
            temp = x_s[threadIdx.x] + x_s[threadIdx.x - stride];
        }
        __syncthreads();
        if (threadIdx.x >= stride) 
        {
            x_s[threadIdx.x] = temp;
        }
    }

    if (threadIdx.x == blockDim.x - 1)
    {
        partialSums[blockIdx.x] = x_s[threadIdx.x];
    }

    if (t_i < N)
    {
        y[t_i] = x_s[threadIdx.x];
    }

}

__global__ void koggeDoubleBufferGPU(int *x, int *y, int *partialSums) 
{
    int t_i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int shared1[THREADS];
    __shared__ int shared2[THREADS];

    int *inShared = shared1;
    int *outShared = shared2;

    if (t_i < N)
    {
        shared1[threadIdx.x] = x[t_i];
    }
    else
    {
        shared1[threadIdx.x] = 0;
    }
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) 
    {
        if (threadIdx.x >= stride) 
        {
            outShared[threadIdx.x] = inShared[threadIdx.x] + inShared[threadIdx.x - stride];
        }
        else
        {
            outShared[threadIdx.x] = inShared[threadIdx.x];
        }
        __syncthreads();

        int *t = inShared;
        inShared = outShared;
        outShared = t;
    }

    if (threadIdx.x == blockDim.x - 1)
    {
        partialSums[blockIdx.x] = inShared[threadIdx.x];
    }

    if (t_i < N)
    {
        y[t_i] = inShared[threadIdx.x];
    }
}

__global__ void add(int *y, int *partialSums) 
{
    int t_i = blockIdx.x * blockDim.x + threadIdx.x;

    if (blockIdx.x > 0 && t_i < N)
    {
        y[t_i] += partialSums[blockIdx.x - 1];
    }
}

int main() 
{
    int *input, *output_cpu, *output_gpu, *output_gpu_double_shared, *partialSums1, *partialSums2;
    size_t size = N * sizeof(int);

    cudaMallocManaged(&input, size);
    output_cpu = (int*)malloc(size);
    cudaMallocManaged(&output_gpu, size);
    cudaMallocManaged(&output_gpu_double_shared, size);
    cudaMallocManaged(&partialSums1, BLOCKS * sizeof(int));
    cudaMallocManaged(&partialSums2, BLOCKS * sizeof(int));

    for (int i = 0; i < N; i++) 
    {
        input[i] = 1;
    }
    cudaMemset(output_gpu, 0, size);
    cudaMemset(output_gpu_double_shared, 0, size);
    cudaMemset(partialSums1, 0, BLOCKS * sizeof(int));
    cudaMemset(partialSums2, 0, BLOCKS * sizeof(int));

    float cpu_time, gpu_time, gpu_time_double_shared;
    clock_t cpu_start = clock();
    koggeCPU(input, output_cpu);
    cpu_time = ((double)(clock() - cpu_start) / CLOCKS_PER_SEC) * 1000;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    koggeGPU<<<BLOCKS, THREADS>>>(input, output_gpu, partialSums1);
    cudaDeviceSynchronize();

    for (int i = 1; i < BLOCKS; i++) 
    {
        partialSums1[i] += partialSums1[i - 1];
    }

    add<<<BLOCKS, THREADS>>>(output_gpu, partialSums1);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2);
    koggeDoubleBufferGPU<<<BLOCKS, THREADS>>>(input, output_gpu_double_shared, partialSums2);
    cudaDeviceSynchronize();

    for (int i = 1; i < BLOCKS; i++) 
    {
        partialSums2[i] += partialSums2[i - 1];
    }

    add<<<BLOCKS, THREADS>>>(output_gpu_double_shared, partialSums2);
    cudaDeviceSynchronize();

    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&gpu_time_double_shared, start2, stop2);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);

    if (output_cpu[N - 1] != output_gpu[N - 1] || output_gpu[N - 1] != output_gpu_double_shared[N - 1])
    {
        printf("Failure\n");
    }
    else
    {
        printf("Success\n");
    }

    printf("\nCPU execution time: %.4f ms\n", cpu_time);
    printf("GPU execution time: %.4f ms\n", gpu_time);
    printf("GPU execution time (Double Buffer): %.4f ms\n", gpu_time_double_shared);

    cudaFree(input);
    free(output_cpu);
    cudaFree(output_gpu);
    cudaFree(output_gpu_double_shared);
    cudaFree(partialSums1);
    cudaFree(partialSums2);
    
    return 0;
}
