#include <stdio.h>
#include <math.h>

#define N 1000

__global__ void init(int *x) 
{
    int t_x = blockIdx.x * blockDim.x + threadIdx.x;
    int t_y = blockIdx.y * blockDim.y + threadIdx.y;

    int strideX = gridDim.x * blockDim.x;
    int strideY = gridDim.y * blockDim.y;

    for (int i = t_x; i < N; i += strideX)
    {
        for (int j = t_y; j < N; j += strideY)
        {
            x[i * N + j] = i * j;
        }
    }
}


__global__ void matMulGPU(int *a, int *b, int *c)
{
    int t_x = blockIdx.x * blockDim.x + threadIdx.x;
    int t_y = blockIdx.y * blockDim.y + threadIdx.y;

    int strideX = gridDim.x * blockDim.x;
    int strideY = gridDim.y * blockDim.y;

    int val = 0;

    for (int  i = t_x; i < N; i += strideX)
    {
        for (int j = t_y; j < N; j += strideY)
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

void matMulCPU(int *a, int *b, int *c)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            a[i * N + j] = i * j;
            b[i * N + j] = i * j;
        }
    }

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

int main()
{
    int *a, *b, *c;
    int size = N * N * sizeof(int);

    cudaMalloc(&a, size);
    cudaMalloc(&b, size);
    cudaMalloc(&c, size);

    dim3 n_threads(32, 32); //RTX A4000 config
    dim3 n_blocks(((N + n_threads.x - 1)/n_threads.x), ((N + n_threads.y - 1)/n_threads.y));

    cudaStream_t strm1, strm2;

    cudaStreamCreate(&strm1);
    cudaStreamCreate(&strm2);

    init<<<n_blocks, n_threads, 0, strm1>>>(a);
    init<<<n_blocks, n_threads, 0, strm2>>>(b);

    cudaStreamSynchronize(strm1);
    cudaStreamSynchronize(strm2);

    cudaMemset(c, 0, size);
    matMulGPU<<<n_blocks, n_threads>>>(a, b, c);

    cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess)
    {
      printf("Synchronous Error: %s\n", cudaGetErrorString(err1));
    }

    cudaError_t err2 = cudaDeviceSynchronize();
    if (err2 != cudaSuccess)
    {
      printf("Asynchronous Error: %s\n", cudaGetErrorString(err2));
    }

    int *a_cpu, *b_cpu, *c_cpu;
    a_cpu = (int*)malloc(size);
    b_cpu = (int*)malloc(size);
    c_cpu = (int*)malloc(size);

    memset(c_cpu, 0, size);
    matMulCPU(a_cpu, b_cpu, c_cpu);

    int *c_result = (int*)malloc(size);
    cudaMemcpy(c_result, c, size, cudaMemcpyDeviceToHost);

    bool flag = false;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if(c_result[i * N + j] != c_cpu[i * N + j])
            {
                printf("Error in c[%d][%d]: CPU=%d, GPU=%d\n", i, j, c_cpu[i * N + j], c_result[i * N + j]);
                flag = true;
                break;
            }
        }
    }
    if(!flag) printf("Success\n");

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaStreamDestroy(strm1);
    cudaStreamDestroy(strm2);
    free(a_cpu);
    free(b_cpu);
    free(c_cpu);

    return 0;
}