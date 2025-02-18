#include <stdio.h>
#include <math.h>

#define N 100000000

__global__ void init(int *x, int val) 
{
    int t_ID = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    if(t_ID < N)
    {
        for (int i = t_ID; i < N; i += stride) 
        {
            x[i] = val;
        }
    }
}


__global__ void addVector(int *a, int *b, int *c)
{
    int t_ID = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    if(t_ID < N)
    {
        for (int i = t_ID; i < N; i += stride)
        {
            c[i] = a[i] + b[i];
        }
    }
}

int main()
{
    int *a, *b, *c;

    size_t size = N * sizeof(int);

    cudaMalloc(&a, size);
    cudaMalloc(&b, size);
    cudaMalloc(&c, size);

    size_t n_threads = 1024; // RTX A4000 specifications
    size_t max_blocks = 32 * 48; // RTX A4000 specifications

    size_t n_blocks = (N + n_threads - 1)/n_threads;

    n_blocks = (n_blocks > max_blocks) ? max_blocks : n_blocks;

    cudaStream_t strm1, strm2, strm3;
    cudaStreamCreate(&strm1);
    cudaStreamCreate(&strm2);
    cudaStreamCreate(&strm3);

    init<<<n_blocks, n_threads, 0, strm1>>>(a, 1);
    init<<<n_blocks, n_threads, 0, strm2>>>(b, 2);
    init<<<n_blocks, n_threads, 0, strm3>>>(c, 0);

    cudaStreamSynchronize(strm1);
    cudaStreamSynchronize(strm2);
    cudaStreamSynchronize(strm3);

    addVector<<<n_blocks, n_threads>>>(a, b, c);

    cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess) {
    printf("Kernel launch error: %s\n", cudaGetErrorString(err1));
    }

    cudaError_t err2 = cudaDeviceSynchronize();
    if (err2 != cudaSuccess)
    {
      printf("Asynchronous Error: %s\n", cudaGetErrorString(err2));
    }

    int *c_host = (int*)malloc(size);

    cudaMemcpy(c_host, c, size, cudaMemcpyDeviceToHost);

    bool error = false;
    for(int i=0; i < N; i++)
    {
        if (c_host[i] != 3)
        {
            printf("Error in:- c[%d]\n", i+1);
            error = true;
            break;
        }
    }
    if(!error) printf("Success\n");

    
    free(c_host);
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    cudaStreamDestroy(strm1);
    cudaStreamDestroy(strm2);
    cudaStreamDestroy(strm3);

    return 0;
}