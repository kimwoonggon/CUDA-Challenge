#include <stdio.h>
#include <curand_kernel.h>
#include <time.h>

__global__ void init(int *arr, int N, unsigned int seed) {
    int t_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_i < N) {
        curandState state;
        curand_init(seed, t_i, 0, &state);
        arr[t_i] = curand(&state) % 1000;
    }
}

__global__ void calcSum(int *arr_in, int *arr_out, int N)
{   
    __shared__ int mem[2048];

    int next_i = blockDim.x * blockIdx.x * 2 + threadIdx.x;

    if (next_i < N) 
    {
        if (next_i + blockDim.x < N) 
        {
            mem[threadIdx.x] = arr_in[next_i] + arr_in[next_i + blockDim.x];
        } 
        else 
        {
            mem[threadIdx.x] = arr_in[next_i];
        }
    } 
    else 
    {
        mem[threadIdx.x] = 0;
    }
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if(threadIdx.x < stride)
        {
            mem[threadIdx.x] += mem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        arr_out[blockIdx.x] = mem[0];
    }
}

int addition(int *arr_in, int N)
{
    int sum = 0;
    for(int i = 0; i < N; i++)
    {
        sum += arr_in[i];
    }

    return sum;
}

int main() {
    int N = 1 << 20 + 1;
    int *arr_in, *arr_out;
    size_t size = N * sizeof(int);

    cudaMallocManaged(&arr_in, size);
    cudaMallocManaged(&arr_out, size);

    size_t n_threads = 1024;
    size_t max_blocks = 32 * 48;
    size_t n_blocks = (N + n_threads - 1) / n_threads;
    n_blocks = (n_blocks > max_blocks) ? max_blocks : n_blocks;

    init<<<n_blocks, n_threads>>>(arr_in, N, time(NULL));
    cudaDeviceSynchronize();

    cudaMemset(arr_out, 0, size);

    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    cudaEventRecord(start_gpu);

    calcSum<<<n_blocks, n_threads>>>(arr_in, arr_out, N);
    cudaDeviceSynchronize();

    calcSum<<<1, n_threads>>>(arr_out, arr_out, N);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    float gpu_time = 0.0f;
    cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);

    int gpu_sum = arr_out[0];

    clock_t start_cpu = clock();
    int cpu_sum = addition(arr_in, N);
    clock_t end_cpu = clock();
    double cpu_time = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC * 1000;

    if (gpu_sum == cpu_sum) {
        printf("Success\n");
    } else {
        printf("Failure: GPU sum = %d, CPU sum = %d\n", gpu_sum, cpu_sum);
    }

    printf("Summation of array CPU time: %f ms\n", cpu_time);
    printf("Summation of array GPU time: %f ms\n", gpu_time);

    cudaFree(arr_in);
    cudaFree(arr_out);

    return 0;
}
