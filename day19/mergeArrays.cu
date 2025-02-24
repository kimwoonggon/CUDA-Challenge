#include <stdio.h>
#include <curand_kernel.h>
#include <time.h>

#define N (1<<20)
#define M (1<<21)
#define THREADS 1024
#define BLOCKS ((M + N + THREADS * ELEMENTS - 1) / (THREADS * ELEMENTS))
#define ELEMENTS 8

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

    while (iLow <= iHigh)
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

int main() 
{
    int *a, *b, *output_cpu, *output_gpu;
    size_t size_a = N * sizeof(int);
    size_t size_b = M * sizeof(int);
    size_t size_merge = (N + M) * sizeof(int);

    cudaMallocManaged(&a, size_a);
    cudaMallocManaged(&b, size_b);
    cudaMallocManaged(&output_gpu, size_merge);

    output_cpu = (int*)malloc(size_merge);

    init(a, N);
    init(b, M);

    memset(output_cpu, 0, size_merge);
    cudaMemset(output_gpu, 0, size_merge);

    float cpu_time, gpu_time;
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

    for (int i = 0; i < (N + M); i++) 
    {
        if (output_cpu[i] != output_gpu[i]) 
        {
            printf("Mismatch at index %d: CPU = %d, GPU = %d\n", i, output_cpu[i], output_gpu[i]);
            break;
        }
    }

    printf("\nCPU execution time: %.4f ms\n", cpu_time);
    printf("GPU execution time: %.4f ms\n", gpu_time);

    cudaFree(a);
    cudaFree(b);
    free(output_cpu);
    cudaFree(output_gpu);
    
    return 0;
}