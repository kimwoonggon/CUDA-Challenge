#include <stdio.h>
#include <curand_kernel.h>
#include <time.h>

#define N (1<<26)
#define THREADS 1024
#define BLOCKS ((N + 2*THREADS - 1) / (2*THREADS))

void brentCPU(int *x, int *y) 
{
    y[0] = x[0];
    for (int i = 1; i < N; i++) 
    {
        y[i] = y[i - 1] + x[i];
    }
}

__global__ void brentGPU(int *x, int *y, int *partialSums) 
{   
    int t_i = blockIdx.x * blockDim.x + threadIdx.x;
    int chunk = 2 * blockIdx.x * blockDim.x;
    __shared__ int x_s[2*THREADS];

    if (t_i < N)
    {
        x_s[threadIdx.x] = x[threadIdx.x + chunk];
        x_s[threadIdx.x + blockDim.x] = x[threadIdx.x + chunk + blockDim.x];
    }
    else
    {
        x_s[threadIdx.x] = 0;
        x_s[threadIdx.x + blockDim.x] = 0;
    }
    __syncthreads();

    for (int stride = 1; stride <= blockDim.x; stride *= 2) 
    {
        int index = (threadIdx.x + 1) * 2 * stride - 1;
        if (index < 2 * blockDim.x) 
        {
            x_s[index] += x_s[index - stride];
        }
        __syncthreads();
    }

    for (int stride = blockDim.x/2; stride >= 1; stride /= 2)
    {
        int index = (threadIdx.x + 1) * 2 * stride - 1;
        if((index + stride) < 2 * blockDim.x)
        {
            x_s[index + stride] += x_s[index];
        }
        __syncthreads();
    }

    if (threadIdx.x == blockDim.x - 1)
    {
        partialSums[blockIdx.x] = x_s[2 * blockDim.x - 1];
    }

    if (t_i < N)
    {
        y[chunk + threadIdx.x] = x_s[threadIdx.x];
        y[chunk + threadIdx.x + blockDim.x] = x_s[threadIdx.x + blockDim.x];
    }

}

__global__ void add(int *y, int *partialSums) 
{
    int t_i = blockIdx.x * blockDim.x + threadIdx.x;
    int chunk = 2 * blockIdx.x * blockDim.x;

    if (blockIdx.x > 0 && t_i < N)
    {
        y[chunk + threadIdx.x] += partialSums[blockIdx.x];
        y[chunk + threadIdx.x + blockDim.x] += partialSums[blockIdx.x - 1];
    }
}

int main() 
{
    int *input, *output_cpu, *output_gpu, *partialSums1;
    size_t size = N * sizeof(int);

    cudaMallocManaged(&input, size);
    output_cpu = (int*)malloc(size);
    cudaMallocManaged(&output_gpu, size);
    cudaMallocManaged(&partialSums1, BLOCKS * sizeof(int));

    for (int i = 0; i < N; i++) 
    {
        input[i] = 1;
    }
    cudaMemset(output_gpu, 0, size);
    cudaMemset(partialSums1, 0, BLOCKS * sizeof(int));

    float cpu_time, gpu_time;
    clock_t cpu_start = clock();
    brentCPU(input, output_cpu);
    cpu_time = ((double)(clock() - cpu_start) / CLOCKS_PER_SEC) * 1000;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    brentGPU<<<BLOCKS, THREADS>>>(input, output_gpu, partialSums1);
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

    if (output_cpu[N - 1] != output_gpu[N - 1])
    {
        printf("Failure\n");
    }
    else
    {
        printf("Success\n");
    }

    printf("\nCPU execution time: %.4f ms\n", cpu_time);
    printf("GPU execution time: %.4f ms\n", gpu_time);

    cudaFree(input);
    free(output_cpu);
    cudaFree(output_gpu);
    cudaFree(partialSums1);
    
    return 0;
}
