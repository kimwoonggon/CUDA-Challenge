// Limit Testing RTX A400
// No. of SMs = 48
// Max. threads per SM = 2048
// Max. threads per block = 1024
// Max. blocks per SM = 2 (lauching 1024 threads each)
// Max. parallelism at a time = 2048 * 48 = 98,304

// Max. shared memory per SM = 96KB usable
// Max. shared memory per block = 48KB (lauching 1024 threads each)
// So, a block's shared memory can store 48,000 / 4 = 12,000 integers (4 bytes each)
// So, to attain maximum efficiency, each thread should coarse over floor(12,000 / 1024) = 11 elements

#include <stdio.h>
#include <curand_kernel.h>
#include <time.h>

#define N (1<<26)

#define THREADS 1024
#define BLOCKS ((N + THREADS - 1) / THREADS)

#define SHARED_BLOCKS ((N + (2*THREADS) - 1) / (2*THREADS))

#define ELEMENTS 11

#define MEM_SIZE (ELEMENTS * THREADS)

#define COARSE_BLOCKS ((N + (ELEMENTS * THREADS) - 1) / (ELEMENTS * THREADS))

__global__ void sumReduceGPU(int *arr, unsigned long long *sum)
{
    int t_i = blockIdx.x * blockDim.x + threadIdx.x;

    for (int stride = blockDim.x / 2; stride >= 1; stride /= 2)
    {
        __syncthreads();
        if (threadIdx.x < stride && (t_i + stride) < N)
        {
            arr[t_i] += arr[t_i + stride];
        }
    }
    if (threadIdx.x == 0)
    {
        atomicAdd(sum, (unsigned long long) arr[blockIdx.x * blockDim.x]);
    }
}

__global__ void sumReduceSharedGPU(int *arr, unsigned long long *sum)
{
    int arr_chunk = 2 * blockIdx.x * blockDim.x;    
    int t_i = threadIdx.x + arr_chunk;

    __shared__ int arr_s[THREADS];
    if (t_i < N) {
        int second = (t_i + blockDim.x < N) ? arr[t_i + blockDim.x] : 0;
        arr_s[threadIdx.x] = arr[t_i] + second;
    } else {
        arr_s[threadIdx.x] = 0;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride >= 1; stride /= 2)
    {
        __syncthreads();
        if (threadIdx.x < stride)
        {
            arr_s[threadIdx.x] += arr_s[threadIdx.x + stride];
        }
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(sum, (unsigned long long) arr_s[0]);
    }
}

__global__ void sumReduceSharedCoarseGPU(int *arr, unsigned long long *sum)
{
    int base = blockIdx.x * blockDim.x * ELEMENTS;
    int t_i = base + threadIdx.x;

    __shared__ int arr_s[THREADS];

    int sum_coarse = 0;
    for (int i = 0; i < ELEMENTS; i++)
    {
        int index = t_i + i * blockDim.x;
        if (index < N)
            sum_coarse += arr[index];
    }

    arr_s[threadIdx.x] = sum_coarse;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride >= 1; stride /= 2)
    {
        __syncthreads();
        if (threadIdx.x < stride)
        {
            arr_s[threadIdx.x] += arr_s[threadIdx.x + stride];
        }
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(sum, (unsigned long long) arr_s[0]);
    }
}

void sumReduceCPU(int *arr, unsigned long long *sum)
{   
    unsigned long long temp = 0;
    for (int i = 0; i < N; i++)
    {
        temp += arr[i];
    }
    *sum = temp;
}

__global__ void init(int *arr1, int *arr2, int *arr3, unsigned int seed) 
{
    int t_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_i < N) {
        curandState state;
        curand_init(seed, t_i, 0, &state);
        arr1[t_i] = curand(&state) % 1000;
        arr2[t_i] = arr1[t_i];
        arr3[t_i] = arr1[t_i];
    }
}

int main() 
{
    int *arr1, *arr2, *arr3;
    unsigned long long *sum_cpu, *sum_gpu, *sum_gpu_shared, *sum_gpu_shared_coarsed;

    size_t size = N * sizeof(int);

    cudaMallocManaged(&arr1, size);
    cudaMallocManaged(&arr2, size);
    cudaMallocManaged(&arr3, size);
    cudaMallocManaged(&sum_cpu, sizeof(unsigned long long));
    cudaMallocManaged(&sum_gpu, sizeof(unsigned long long));
    cudaMallocManaged(&sum_gpu_shared, sizeof(unsigned long long));
    cudaMallocManaged(&sum_gpu_shared_coarsed, sizeof(unsigned long long));

    cudaMemset(sum_cpu, 0, sizeof(unsigned long long));
    cudaMemset(sum_gpu, 0, sizeof(unsigned long long));
    cudaMemset(sum_gpu_shared, 0, sizeof(unsigned long long));
    cudaMemset(sum_gpu_shared_coarsed, 0, sizeof(unsigned long long));

    init<<<BLOCKS, THREADS>>>(arr1, arr2, arr3, time(NULL));
    cudaDeviceSynchronize();

    float cpu_time, gpu_time, gpu_time_shared, gpu_time_shared_coarse;

    clock_t cpu_start = clock();
    sumReduceCPU(arr1, sum_cpu);
    cpu_time = ((double)(clock() - cpu_start) / CLOCKS_PER_SEC) * 1000;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    sumReduceGPU<<<BLOCKS, THREADS>>>(arr1, sum_gpu);
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
    sumReduceSharedGPU<<<SHARED_BLOCKS, THREADS>>>(arr2, sum_gpu_shared);
    cudaDeviceSynchronize();
    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&gpu_time_shared, start2, stop2);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);

    cudaEvent_t start3, stop3;
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);
    cudaEventRecord(start3);
    sumReduceSharedCoarseGPU<<<COARSE_BLOCKS, THREADS>>>(arr3, sum_gpu_shared_coarsed);
    cudaDeviceSynchronize();
    cudaEventRecord(stop3);
    cudaEventSynchronize(stop3);
    cudaEventElapsedTime(&gpu_time_shared_coarse, start3, stop3);
    cudaEventDestroy(start3);
    cudaEventDestroy(stop3);

    if(*sum_cpu != *sum_gpu || *sum_gpu != *sum_gpu_shared || *sum_gpu_shared != *sum_gpu_shared_coarsed)
    {
        printf("Failure\n");
    }
    else
    {
        printf("Success\n");
    }

    printf("\nCPU execution time: %.4f ms\n", cpu_time);
    printf("GPU execution time (Naive): %.4f ms\n", gpu_time);
    printf("GPU execution time (Shared Memory): %.4f ms\n", gpu_time_shared);
    printf("GPU execution time (Shared Memory + Thread Coarsening): %.4f ms\n", gpu_time_shared_coarse);

    cudaFree(arr1);
    cudaFree(arr2);
    cudaFree(arr3);
    cudaFree(sum_cpu);
    cudaFree(sum_gpu);
    cudaFree(sum_gpu_shared);
    cudaFree(sum_gpu_shared_coarsed);

    return 0;
}
