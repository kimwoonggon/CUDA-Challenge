#include <curand_kernel.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define N 10 * 1000000

#define THREADS 1024
#define SAMPLES_PER_THREAD (N / THREADS + 1)
#define TOTAL_THREADS ((N + SAMPLES_PER_THREAD - 1) / SAMPLES_PER_THREAD)
#define BLOCKS ((TOTAL_THREADS + THREADS - 1) / THREADS)

__global__ void monte_carlo_warp(int *res, int samples_per_thread)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int lane = threadIdx.x % warpSize;

    curandState state;
    curand_init(idx, 0, 0, &state);

    int count = 0;
    for (int i = 0; i < samples_per_thread; ++i)
    {
        float x = curand_uniform(&state);
        float y = curand_uniform(&state);
        if (x * x + y * y <= 1.0f)
        {
            count++;
        }
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        count += __shfl_down_sync(0xFFFFFFFF, count, offset);
    }

    if (lane == 0)
    {
        atomicAdd(res, count);
    }
}


__global__ void monte_carlo_thread(int *res, int samples_per_thread) 
{
    __shared__ int shared_counts[THREADS];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    curandState state;
    curand_init(idx, 0, 0, &state);

    int local_count = 0;
    for (int i = 0; i < samples_per_thread; ++i) 
    {
        float x = curand_uniform(&state);
        float y = curand_uniform(&state);
        if (x * x + y * y <= 1.0f) local_count++;
    }

    shared_counts[threadIdx.x] = local_count;
    __syncthreads();

    if (threadIdx.x == 0) 
    {
        int block_sum = 0;
        for (int i = 0; i < blockDim.x; i++) 
        {
            block_sum += shared_counts[i];
        }
        atomicAdd(res, block_sum);
    }
}

int main() 
{
    int *res_warp, *res_thread;
    cudaMallocManaged(&res_warp, sizeof(int));
    cudaMallocManaged(&res_thread, sizeof(int));

    cudaEvent_t start, stop;
    float time_warp, time_thread;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemset(res_warp, 0, sizeof(int));
    cudaEventRecord(start);
    monte_carlo_warp<<<BLOCKS, THREADS>>>(res_warp, SAMPLES_PER_THREAD);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_warp, start, stop);
    float pi_warp = (4.0f * (*res_warp)) / N;

    cudaMemset(res_thread, 0, sizeof(int));
    cudaEventRecord(start);
    monte_carlo_thread<<<BLOCKS, THREADS>>>(res_thread, SAMPLES_PER_THREAD);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_thread, start, stop);
    float pi_thread = (4.0f * (*res_thread)) / N;

    cudaFree(res_thread);
    cudaFree(res_warp);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Warp Sync:- Pi %f, Time %f ms\n", pi_warp, time_warp);
    printf("Thread Sync:- Pi %f, Time %f ms\n", pi_thread, time_thread);

    return 0;
}
