#include <stdio.h>
#include <math.h>
#include <curand_kernel.h>
#include <time.h>

#define N 1951
#define MEM 16

__global__ void init(int *x, unsigned int seed) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
    {
        curandState state;
        curand_init(seed, (row * N + col), 0, &state);
        x[row * N + col] =  curand(&state) % 1000;
    }
}

__global__ void matMulGPU(int *a, int *b, int *c)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
    {
        int val = 0;
        for(int k = 0; k < N; k++)
        {
            val += a[row * N + k] * b[k * N + col];
        }
        c[row * N + col] = val;
    }
}

__global__ void matMulTiled(int *a, int *b, int *c)
{
    __shared__ int mem_a[MEM][MEM];
    __shared__ int mem_b[MEM][MEM];

    int t_x = threadIdx.x;
    int t_y = threadIdx.y;

    int b_x = blockIdx.x;
    int b_y = blockIdx.y;

    int row = b_y * MEM + t_y;
    int col = b_x * MEM + t_x;

    int val = 0;


    for(int stride = 0; stride < ceil((float)N/MEM) ; stride++)
    {
        if ((row < N) && ((stride * MEM + t_x) < N))
        {
            mem_a[t_y][t_x] = a[row * N + stride * MEM + t_x];
        }
        else
        {
            mem_a[t_y][t_x] = 0.0f;
        }
        if ((col < N) && ((stride * MEM + t_y) < N))
        {
            mem_b[t_y][t_x] = b[(stride * MEM + t_y) * N + col];
        }
        else
        {
            mem_b[t_y][t_x] = 0.0f;
        }

        __syncthreads();

        for(int k = 0; k < MEM; k++)
        {
            val += mem_a[t_y][k] * mem_b[k][t_x];
        }

        __syncthreads();
    }
    c[row * N + col] = val;

}

void matMulCPU(int *a, int *b, int *c)
{
    int val = 0;

    for (int  i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            for ( int k = 0; k < N; k++)
            {
                val += a[i * N + k] * b[k * N + j];
            }
            c[i * N + j] = val;
            val = 0;
        }
    }
}

bool verify(int *c_cpu, int *c_gpu, int *c_gpu_tiled) 
{   
    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < N; j++) 
        {
            int c = i * N + j;
            if ((c_cpu[c] != c_gpu[c]) && (c_gpu_tiled[c] != c_gpu[c])) 
            {
                printf("Mismatch at (%d, %d): CPU=%d, GPU=%d, Tiled=%d\n", i, j, c_cpu[c], c_gpu[c], c_gpu_tiled[c]);
                return false;
            }
        }
    }
    return true;
}

int main()
{
    int *a, *b, *c_gpu, *c_gpu_tiled, *c_cpu;
    int size = N * N * sizeof(int);

    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c_gpu, size);
    cudaMallocManaged(&c_gpu_tiled, size);
    cudaMallocManaged(&c_cpu, size);

    dim3 n_threads(MEM, MEM); //RTX A4000 config
    dim3 n_blocks(N / MEM, N / MEM);

    cudaStream_t strm1, strm2;

    cudaStreamCreate(&strm1);
    cudaStreamCreate(&strm2);

    init<<<n_blocks, n_threads, 0, strm1>>>(a, time(NULL));
    init<<<n_blocks, n_threads, 0, strm2>>>(b, time(NULL));

    cudaStreamSynchronize(strm1);
    cudaStreamSynchronize(strm2);

    cudaMemset(c_gpu, 0, size);
    cudaMemset(c_gpu_tiled, 0, size);
    cudaMemset(c_cpu, 0, size);

    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    cudaEventRecord(start_gpu);

    matMulGPU<<<n_blocks, n_threads>>>(a, b, c_gpu);

    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    float gpu_time = 0.0f;
    cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);

    cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess)
    {
      printf("Synchronous Error1: %s\n", cudaGetErrorString(err1));
    }

    cudaError_t err2 = cudaDeviceSynchronize();
    if (err2 != cudaSuccess)
    {
      printf("Asynchronous Error1: %s\n", cudaGetErrorString(err2));
    }

    cudaEvent_t start_gpu_tiled, stop_gpu_tiled;
    cudaEventCreate(&start_gpu_tiled);
    cudaEventCreate(&stop_gpu_tiled);
    cudaEventRecord(start_gpu_tiled);

    matMulTiled<<<n_blocks, n_threads>>>(a, b, c_gpu_tiled);

    cudaEventRecord(stop_gpu_tiled);
    cudaEventSynchronize(stop_gpu_tiled);
    float gpu_time_tiled = 0.0f;
    cudaEventElapsedTime(&gpu_time_tiled, start_gpu_tiled, stop_gpu_tiled);

    cudaError_t err3 = cudaGetLastError();
    if (err3 != cudaSuccess)
    {
      printf("Synchronous Error1: %s\n", cudaGetErrorString(err3));
    }

    cudaError_t err4 = cudaDeviceSynchronize();
    if (err4 != cudaSuccess)
    {
      printf("Asynchronous Error1: %s\n", cudaGetErrorString(err4));
    }

    clock_t start_cpu = clock();
    matMulCPU(a, b, c_cpu);
    clock_t end_cpu = clock();
    double cpu_time = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC * 1000;

    bool flag = verify(c_cpu, c_gpu, c_gpu_tiled);
    if(flag)
    {
        printf("Success\n");
    }
    else
    {
        printf("Error\n");
    }

    printf("Matrix Multiplication CPU time: %f ms\n", cpu_time);
    printf("Matrix Multiplication GPU time: %f ms\n", gpu_time);
    printf("Tiled Matrix Multiplication GPU time: %f ms\n", gpu_time_tiled);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c_cpu);
    cudaFree(c_gpu);
    cudaFree(c_gpu_tiled);
    cudaStreamDestroy(strm1);
    cudaStreamDestroy(strm2);

    return 0;
}