#include <stdio.h>
#include <math.h>
#include <time.h>

#define N (1<<25)
#define THREADS 256
#define BLOCKS ((THREADS + N - 1) / THREADS)

#define BINS 5
#define TOTAL_BINS (BLOCKS * BINS)

#define COARSE 32
#define BLOCKS_COARSE ((THREADS * COARSE + N - 1) / (THREADS * COARSE))

void init(unsigned int *data)
{
    srand(time(NULL));
    
    for (unsigned int i = 0; i < N; i++)
    {
        data[i] = (rand() % 100) + 1;
    }
}

void histCPU(unsigned int *data, unsigned int *hist)
{
    for(unsigned int i = 0; i < N; i++)
    {
        if(data[i] > 0 && data[i] <= 100)
        {
            hist[(data[i] - 1)/20] += 1;
        }
    }
}

__global__ void histGPU(unsigned int *data, unsigned int *hist)
{
    unsigned int t_i = blockIdx.x * blockDim.x + threadIdx.x;
    if(t_i < N)
    {
        if(data[t_i] > 0 && data[t_i] <= 100)
        {
            atomicAdd(&hist[(data[t_i] - 1)/20], 1);
        }
    }
}

__global__ void histPvtGPU(unsigned int *data, unsigned int *hist)
{
    unsigned int t_i = blockIdx.x * blockDim.x + threadIdx.x;
    if(t_i < N)
    {
        if(data[t_i] > 0 && data[t_i] <= 100)
        {
            atomicAdd(&hist[(blockIdx.x * BINS + (data[t_i] - 1)/20)], 1);
        }
    }

    if(blockIdx.x > 0)
    {
        __syncthreads();
        for(unsigned bin = threadIdx.x; bin < BINS; bin += blockDim.x)
        {
            unsigned int val = hist[blockIdx.x * BINS + bin];
            if(val > 0)
            {
                atomicAdd(&hist[bin], val);
            }
        }
    }
}

__global__ void histPvtSharedGPU(unsigned int *data, unsigned int *hist)
{
    unsigned int t_i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ unsigned int hist_s[BINS];
    if (threadIdx.x < BINS)
    {
        hist_s[threadIdx.x] = 0u;
    }
    __syncthreads();

    if(t_i < N)
    {
        if(data[t_i] > 0 && data[t_i] <= 100)
        {
            atomicAdd(&hist_s[(data[t_i] - 1)/20], 1);
        }
    }
    __syncthreads();

    for(unsigned bin = threadIdx.x; bin < BINS; bin += blockDim.x)
    {   
        if(hist_s[bin] > 0)
        {
            atomicAdd(&hist[bin], hist_s[bin]);
        }
    }
}

__global__ void histPvtSharedContiguousPartitionGPU(unsigned int *data, unsigned int *hist)
{
    unsigned int t_i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ unsigned int hist_s[BINS];
    if (threadIdx.x < BINS)
    {
        hist_s[threadIdx.x] = 0u;
    }
    __syncthreads();

    for(unsigned int thread = t_i * COARSE; thread < min((t_i + 1) * COARSE, N); thread++)
    {
        if(data[thread] > 0 && data[thread] <= 100)
        {
            atomicAdd(&hist_s[(data[thread] - 1)/20], 1);
        }
    }
    __syncthreads();

    for(unsigned bin = threadIdx.x; bin < BINS; bin += blockDim.x)
    {   
        if(hist_s[bin] > 0)
        {
            atomicAdd(&hist[bin], hist_s[bin]);
        }
    }
}

__global__ void histPvtSharedStridedGPU(unsigned int *data, unsigned int *hist)
{
    unsigned int t_i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ unsigned int hist_s[BINS];
    if (threadIdx.x < BINS)
    {
        hist_s[threadIdx.x] = 0u;
    }
    __syncthreads();

    for(unsigned int thread = t_i; thread < N; thread += blockDim.x * gridDim.x)
    {
        if(data[thread] > 0 && data[thread] <= 100)
        {
            atomicAdd(&hist_s[(data[thread] - 1)/20], 1);
        }
    }
    __syncthreads();

    for(unsigned bin = threadIdx.x; bin < BINS; bin += blockDim.x)
    {   
        if(hist_s[bin] > 0)
        {
            atomicAdd(&hist[bin], hist_s[bin]);
        }
    }
}

__global__ void verify(unsigned int *hist_cpu, unsigned int *hist_gpu, unsigned int *hist_gpu_pvt, unsigned int *hist_gpu_pvt_shared, unsigned int *hist_gpu_pvt_shared_contiguous, unsigned int *hist_gpu_pvt_shared_strided, unsigned int *errors)
{
    unsigned int t_i = blockIdx.x * blockDim.x + threadIdx.x;

    if (t_i < BINS)
    {
        if (hist_cpu[t_i] != hist_gpu[t_i] || hist_gpu[t_i] != hist_gpu_pvt[t_i] || hist_gpu_pvt[t_i] != hist_gpu_pvt_shared[t_i] || hist_gpu_pvt_shared[t_i] != hist_gpu_pvt_shared_contiguous[t_i] || hist_gpu_pvt_shared_contiguous[t_i] != hist_gpu_pvt_shared_strided[t_i])
        {
            atomicAdd(errors, 1);
        }
    }
}

int main()
{
    unsigned int *data, *hist_cpu, *hist_gpu, *hist_gpu_pvt, *hist_gpu_pvt_shared, *hist_gpu_pvt_shared_contiguous, *hist_gpu_pvt_shared_strided;

    size_t size = N * sizeof(unsigned int);

    cudaMallocManaged(&data, size);
    cudaMallocManaged(&hist_cpu, BINS * sizeof(unsigned int));
    cudaMallocManaged(&hist_gpu, BINS * sizeof(unsigned int));
    cudaMallocManaged(&hist_gpu_pvt, TOTAL_BINS * sizeof(unsigned int));
    cudaMallocManaged(&hist_gpu_pvt_shared, BINS * sizeof(unsigned int));
    cudaMallocManaged(&hist_gpu_pvt_shared_contiguous, BINS * sizeof(unsigned int));
    cudaMallocManaged(&hist_gpu_pvt_shared_strided, BINS * sizeof(unsigned int));

    cudaMemset(hist_cpu, 0, BINS * sizeof(unsigned int));
    cudaMemset(hist_gpu, 0, BINS * sizeof(unsigned int));
    cudaMemset(hist_gpu_pvt, 0, TOTAL_BINS * sizeof(unsigned int));
    cudaMemset(hist_gpu_pvt_shared, 0, BINS * sizeof(unsigned int));
    cudaMemset(hist_gpu_pvt_shared_contiguous, 0, BINS * sizeof(unsigned int));
    cudaMemset(hist_gpu_pvt_shared_strided, 0, BINS * sizeof(unsigned int));

    init(data);

    float cpu_time, gpu_time, gpu_time_pvt, gpu_time_pvt_shared, gpu_time_pvt_shared_contiguous, gpu_time_pvt_shared_strided;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    histGPU<<<BLOCKS, THREADS>>>(data, hist_gpu);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);

    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2);
    histPvtGPU<<<BLOCKS, THREADS>>>(data, hist_gpu_pvt);
    cudaDeviceSynchronize();
    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&gpu_time_pvt, start2, stop2);

    cudaEvent_t start3, stop3;
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);
    cudaEventRecord(start3);
    histPvtSharedGPU<<<BLOCKS, THREADS>>>(data, hist_gpu_pvt_shared);
    cudaDeviceSynchronize();
    cudaEventRecord(stop3);
    cudaEventSynchronize(stop3);
    cudaEventElapsedTime(&gpu_time_pvt_shared, start3, stop3);

    cudaEvent_t start4, stop4;
    cudaEventCreate(&start4);
    cudaEventCreate(&stop4);
    cudaEventRecord(start4);
    histPvtSharedContiguousPartitionGPU<<<BLOCKS_COARSE, THREADS>>>(data, hist_gpu_pvt_shared_contiguous);
    cudaDeviceSynchronize();
    cudaEventRecord(stop4);
    cudaEventSynchronize(stop4);
    cudaEventElapsedTime(&gpu_time_pvt_shared_contiguous, start4, stop4);

    cudaEvent_t start5, stop5;
    cudaEventCreate(&start5);
    cudaEventCreate(&stop5);
    cudaEventRecord(start5);
    histPvtSharedStridedGPU<<<BLOCKS_COARSE, THREADS>>>(data, hist_gpu_pvt_shared_strided);
    cudaDeviceSynchronize();
    cudaEventRecord(stop5);
    cudaEventSynchronize(stop5);
    cudaEventElapsedTime(&gpu_time_pvt_shared_strided, start5, stop5);


    clock_t cpu_start = clock();
    histCPU(data, hist_cpu);
    cpu_time = ((double)(clock() - cpu_start) / CLOCKS_PER_SEC) * 1000;

    unsigned int *errors;
    cudaMallocManaged(&errors, sizeof(unsigned int));
    verify<<<1, 5>>>(hist_cpu, hist_gpu, hist_gpu_pvt, hist_gpu_pvt_shared, hist_gpu_pvt_shared_contiguous, hist_gpu_pvt_shared_strided, errors);
    cudaDeviceSynchronize();

    printf("\nNumber of mismatches: %d\n", *errors);
    printf("\nCPU execution time: %.4f ms\n", cpu_time);
    printf("GPU execution time (Naive): %.4f ms\n", gpu_time);
    printf("GPU execution time (Privatization): %.4f ms\n", gpu_time_pvt);
    printf("GPU execution time (Privatization + Shared Memory): %.4f ms\n", gpu_time_pvt_shared);
    printf("GPU execution time (Privatization + Shared Memory + Contiguous Coarsening): %.4f ms\n", gpu_time_pvt_shared_contiguous);
    printf("GPU execution time (Privatization + Shared Memory + Strided Coarsening): %.4f ms\n", gpu_time_pvt_shared_strided);

    printf("\nHistogram:\n");
    for (int i = 0; i < BINS; i++)
    {
        printf("Bin %d has %u numbers\n", i, hist_cpu[i]);
    }

    cudaFree(data);
    cudaFree(hist_cpu);
    cudaFree(hist_gpu);
    cudaFree(hist_gpu_pvt);
    cudaFree(hist_gpu_pvt_shared);
    cudaFree(hist_gpu_pvt_shared_contiguous);
    cudaFree(hist_gpu_pvt_shared_strided);
    cudaFree(errors);

    return 0;
}