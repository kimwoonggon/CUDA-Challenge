#include <stdio.h>
#include <curand_kernel.h>
#include <time.h>

#define N (1<<24)
#define THREADS 1024
#define BLOCKS ((N + 2*THREADS - 1) / (2*THREADS))

__device__ void brentGPU(int *x) 
{   
    int chunk = 2 * blockIdx.x * blockDim.x;
    __shared__ int x_s[2*THREADS];

    if (chunk + threadIdx.x < N)
    {
        x_s[threadIdx.x] = x[chunk + threadIdx.x];
    }
    else
    {
        x_s[threadIdx.x] = 0;
    }
    
    if (chunk + threadIdx.x + blockDim.x < N)
    {
        x_s[threadIdx.x + blockDim.x] = x[chunk + threadIdx.x + blockDim.x];
    }
    else
    {
        x_s[threadIdx.x + blockDim.x] = 0;
    }
    __syncthreads();

    int n = 2 * blockDim.x;

    for (int stride = 1; stride < n; stride *= 2) 
    {
        int index = (threadIdx.x + 1) * 2 * stride - 1;
        if (index < n)
        {
            x_s[index] += x_s[index - stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        x_s[n - 1] = 0;
    }
    __syncthreads();

    for (int stride = n/2; stride >= 1; stride /= 2)
    {
        int index = (threadIdx.x + 1) * 2 * stride - 1;
        if(index < n)
        {
            int t = x_s[index - stride];
            x_s[index - stride] = x_s[index];
            x_s[index] += t;
        }
        __syncthreads();
    }

    if (chunk + threadIdx.x < N)
    {
        x[chunk + threadIdx.x] = x_s[threadIdx.x];
    }

    if (chunk + threadIdx.x + blockDim.x < N)
    {
        x[chunk + threadIdx.x + blockDim.x] = x_s[threadIdx.x + blockDim.x];
    }
}

__global__ void radixSort(int *inp, int *out, int *bits, int iter)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i < N)
    {
        int key = inp[i];
        int bit = (key >> iter) & 1;
        bits[i] = bit;
    }
    __syncthreads();

    brentGPU(bits);
    __syncthreads();
    
    if(i < N)
    {
        int key = inp[i];
        int bit = (key >> iter) & 1;
        int oneBefore = bits[i];
        
        int lastBit = (inp[N-1] >> iter) & 1;
        int oneTotal = bits[N-1] + lastBit;
        
        int dst;
        if(bit == 0)
        {
            dst = i - oneBefore;
        }
        else
        {
            dst = (N - oneTotal) + oneBefore;
        }

        out[dst] = key;
    }
}

void init(int *arr, int n) 
{
    srand(time(NULL));
    for (int i = 0; i < n; i++) 
    {
        arr[i] = rand() % 100;
    }
}

void radixSortCPU(const int *a, int *result, int iter, int n) 
{
    int *cpu_bits = (int*)malloc(n * sizeof(int));
    int *prefix = (int*)malloc(n * sizeof(int));
    
    for (int i = 0; i < n; i++) 
    {
        cpu_bits[i] = (a[i] >> iter) & 1;
    }
    
    prefix[0] = 0;
    for (int i = 1; i < n; i++) 
    {
        prefix[i] = prefix[i - 1] + cpu_bits[i - 1];
    }
    
    int oneTotal = prefix[n - 1] + cpu_bits[n - 1];
    
    for (int i = 0; i < n; i++) 
    {
        int bit = cpu_bits[i];
        int oneBefore = prefix[i];
        int dst;
        if (bit == 0) 
        {
            dst = i - oneBefore;
        } 
        else 
        {
            dst = (n - oneTotal) + oneBefore;
        }
        result[dst] = a[i];
    }
    
    free(cpu_bits);
    free(prefix);
}

int main() 
{
    int *a, *output_cpu, *output_gpu, *bits;
    size_t size = N * sizeof(int);

    cudaMallocManaged(&a, size);
    cudaMallocManaged(&output_gpu, size);
    cudaMallocManaged(&bits, size);
    output_cpu = (int*)malloc(size);

    init(a, N);

    memset(output_cpu, 0, size);
    cudaMemset(output_gpu, 0, size);
    cudaMemset(bits, 0, size);

    int iter = 0;

    clock_t cpu_start = clock();
    radixSortCPU(a, output_cpu, iter, N);
    float cpu_time = ((double)(clock() - cpu_start) / CLOCKS_PER_SEC) * 1000;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    radixSort<<<BLOCKS, THREADS>>>(a, output_gpu, bits, iter);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("CPU execution time: %.4f ms\n", cpu_time);
    printf("GPU execution time: %.4f ms\n", gpu_time);

    cudaFree(a);
    cudaFree(output_gpu);
    cudaFree(bits);
    free(output_cpu);

    return 0;
}