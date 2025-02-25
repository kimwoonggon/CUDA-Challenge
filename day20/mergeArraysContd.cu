#include <stdio.h>
#include <curand_kernel.h>
#include <time.h>

#define N (1<<10)
#define M (1<<11)
#define THREADS 512
#define BLOCKS ((M + N + THREADS * ELEMENTS - 1) / (THREADS * ELEMENTS))
#define ELEMENTS 4
#define ELEMENTS_PER_BLOCK (THREADS * ELEMENTS)
#define TILED_BLOCKS ((N + M + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK)

void init(int *arr, int size) 
{
    for (int i = 0; i < size; i++) 
    {
        arr[i] = rand();
    }

    qsort(arr, size, sizeof(int), (int (*)(const void *, const void *)) (int (*)(const int *, const int *)) [](const int *a, const int *b)
    {
        return *a - *b;
    });
}

__host__ __device__ void mergeCPU(int *a, int *b, int *c, int m, int n) 
{
    int i = 0, j = 0, k = 0;
    
    while(i < m && j < n) 
    {
        if(a[i] <= b[j]) 
        {
            c[k++] = a[i++];
        } 
        else 
        {
            c[k++] = b[j++];
        }
    }
    
    while(j < n) 
    {
        c[k++] = b[j++];
    }
    
    while(i < m) 
    {
        c[k++] = a[i++];
    }
}


__device__ int coRank(int *a, int *b, int m, int n, int k)
{
    int iLow, iHigh;
    if(k > n)
    {
        iLow = k - n;
    }
    else
    {
        iLow = 0;
    }

    if(k < m)
    {
        iHigh = k;
    }
    else
    {
        iHigh = m;
    }

    while (iLow < iHigh)
    {
        int i = (iLow + iHigh) / 2;
        int j = k - i;

        if (i > 0 && j < n && a[i - 1] > b[j])
        {
            iHigh = i;
        }
        else if (j > 0 && i < m && b[j - 1] > a[i])
        {
            iLow = i;
        }
        else
        {
            return i;
        }
    }
    return iLow;
}

__global__ void mergeGPU(int *a, int *b, int *c, int m, int n)
{
    int k = (blockIdx.x * blockDim.x + threadIdx.x) * ELEMENTS;

    if(k < (m + n))
    {
        int i = coRank(a, b, m, n, k);
        int j = k - i;
        int kNext;

        if((k + ELEMENTS) < ( m + n))
        {
            kNext = k + ELEMENTS;
        }
        else
        {
            kNext = m + n;    
        }

        int iNext = coRank(a, b, m, n, kNext);
        int jNext = kNext - iNext;

        mergeCPU(&a[i], &b[j], & c[k], (iNext - i), (jNext - j));
    }
}

__global__ void mergeTiledGPU(int *a, int *b, int *c, int m, int n)
{
    int kBlock = blockIdx.x * ELEMENTS_PER_BLOCK;
    int kNextBlock;

    if(blockIdx.x < gridDim.x - 1)
    {
        kNextBlock = kBlock + ELEMENTS_PER_BLOCK;
    }
    else
    {
        kNextBlock = m + n;
    }

    __shared__ int iBlock;
    __shared__ int iNextBlock;

    if(threadIdx.x == 0)
    {
        iBlock = coRank(a, b, m, n, kBlock);
    }
    if(threadIdx.x == blockDim.x - 1)
    {
        iNextBlock = coRank(a, b, m, n, kNextBlock);
    }
    __syncthreads();

    int jBlock = kBlock - iBlock;
    int jNextBlock = kNextBlock - iNextBlock;

    __shared__ int a_s[ELEMENTS_PER_BLOCK];
    int mBlock = iNextBlock - iBlock;

    for(int x = threadIdx.x; x < mBlock; x += blockDim.x)
    {
        a_s[x] = a[iBlock + x];
    }

    __shared__ int b_s[ELEMENTS_PER_BLOCK];
    int nBlock = jNextBlock - jBlock;

    for(int y = threadIdx.x; y < nBlock; y += blockDim.x)
    {
        b_s[y] = b[jBlock + y];
    }
    __syncthreads();

    __shared__ int c_s[ELEMENTS_PER_BLOCK];
    int k = threadIdx.x * ELEMENTS;

    if(k < mBlock + nBlock)
    {
        int i = coRank(a_s, b_s, mBlock, nBlock, k);
        int j = k - i;
        int kNext;

        if((k + ELEMENTS) < (mBlock + nBlock))
        {
            kNext = k + ELEMENTS;
        }
        else
        {
            kNext = mBlock + nBlock;
        }

        int iNext = coRank(a_s, b_s, mBlock, nBlock, kNext);
        int jNext = kNext - iNext;
        mergeCPU(&a_s[i], &b_s[j], &c_s[k], iNext - i, jNext - j);
    }
    __syncthreads();

    for(int z = threadIdx.x; z < mBlock + nBlock; z += blockDim.x)
    {
        c[kBlock + z] = c_s[z];
    }
}

int main() 
{
    int *a, *b, *output_cpu, *output_gpu, *output_gpu_tiled;
    size_t size_a = N * sizeof(int);
    size_t size_b = M * sizeof(int);
    size_t size_merge = (N + M) * sizeof(int);

    cudaMallocManaged(&a, size_a);
    cudaMallocManaged(&b, size_b);
    cudaMallocManaged(&output_gpu, size_merge);
    cudaMallocManaged(&output_gpu_tiled, size_merge);

    output_cpu = (int*)malloc(size_merge);

    init(a, N);
    init(b, M);

    memset(output_cpu, 0, size_merge);
    cudaMemset(output_gpu, 0, size_merge);
    cudaMemset(output_gpu_tiled, 0, size_merge);

    float cpu_time, gpu_time, gpu_tiled_time;
    clock_t cpu_start = clock();
    mergeCPU(a, b, output_cpu, N, M);
    cpu_time = ((double)(clock() - cpu_start) / CLOCKS_PER_SEC) * 1000;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    mergeGPU<<<BLOCKS, THREADS>>>(a, b, output_gpu, N, M);
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

    mergeTiledGPU<<<TILED_BLOCKS, THREADS>>>(a, b, output_gpu_tiled, N, M);
    cudaDeviceSynchronize();

    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&gpu_tiled_time, start2, stop2);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);

    for (int i = 0; i < (N + M); i++) 
    {
        if (output_cpu[i] != output_gpu[i] || output_gpu[i] != output_gpu_tiled[i]) 
        {
            printf("Mismatch at index %d: CPU = %d, GPU = %d, Tiled GPU - %d\n", i, output_cpu[i], output_gpu[i], output_gpu_tiled[i]);
            break;
        }
    }

    printf("\nCPU execution time: %.4f ms\n", cpu_time);
    printf("GPU execution time: %.4f ms\n", gpu_time);
    printf("GPU execution time (Tiled): %.4f ms\n", gpu_tiled_time);

    cudaFree(a);
    cudaFree(b);
    free(output_cpu);
    cudaFree(output_gpu);
    cudaFree(output_gpu_tiled);
    
    return 0;
}