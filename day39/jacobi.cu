#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define OFFSET(j, i, m) ((j) * (m) + (i))
#define THREADS 32

__device__ double atomicMaxDouble(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    
    return __longlong_as_double(old);
}

__global__ void calcNextKernel(double *A, double *Anew, int m, int n, double *error)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    __shared__ double block_error[THREADS][THREADS];
    
    double local_error = 0.0;
    
    if (i < m-1 && j < n-1)
    {
        Anew[OFFSET(j, i, m)] = 0.25 * (A[OFFSET(j, i+1, m)] + A[OFFSET(j, i-1, m)] + A[OFFSET(j-1, i, m)] + A[OFFSET(j+1, i, m)]);
        local_error = fabs(Anew[OFFSET(j, i, m)] - A[OFFSET(j, i, m)]);
    }
    
    block_error[threadIdx.y][threadIdx.x] = local_error;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s && threadIdx.y < blockDim.y)
        {
            block_error[threadIdx.y][threadIdx.x] = fmax(block_error[threadIdx.y][threadIdx.x], block_error[threadIdx.y][threadIdx.x + s]);
        }
        __syncthreads();
    }
    
    for (int s = blockDim.y / 2; s > 0; s >>= 1)
    {
        if (threadIdx.y < s && threadIdx.x == 0)
        {
            block_error[threadIdx.y][0] = fmax(block_error[threadIdx.y][0], block_error[threadIdx.y + s][0]);
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        atomicMaxDouble(error, block_error[0][0]);
    }
}

__global__ void swapKernel(double *A, double *Anew, int m, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    if (i < m-1 && j < n-1)
    {
        A[OFFSET(j, i, m)] = Anew[OFFSET(j, i, m)];
    }
}

void initialize(double *A, double *Anew, int m, int n)
{
    memset(A, 0, n * m * sizeof(double));
    memset(Anew, 0, n * m * sizeof(double));
    
    for (int i = 0; i < m; i++)
    {
        A[i] = 1.0;
        Anew[i] = 1.0;
    }
}

int main(int argc, char** argv)
{
    const int n = 1 << 13;
    const int m = 1 << 13;
    const int iter_max = 1000;
    const double tol = 1.0e-6;
    double error = 1.0;
    
    double *A, *Anew, *d_error;
    
    cudaMallocManaged(&A, sizeof(double) * n * m);
    cudaMallocManaged(&Anew, sizeof(double) * n * m);
    cudaMallocManaged(&d_error, sizeof(double));
    
    initialize(A, Anew, m, n);
    
    dim3 thread(THREADS, THREADS);
    dim3 block((m - 2 + thread.x - 1) / thread.x, (n - 2 + thread.y - 1) / thread.y);
    
    printf("Jacobi relaxation Calculation: %d x %d mesh\n", n, m);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    int iter = 0;
    while (error > tol && iter < iter_max)
    {
        *d_error = 0.0;
        
        calcNextKernel<<<block, thread>>>(A, Anew, m, n, d_error);
        cudaDeviceSynchronize();
        
        error = *d_error;
        
        swapKernel<<<block, thread>>>(A, Anew, m, n);
        cudaDeviceSynchronize();
        
        if (iter % 100 == 0)
        {
            printf("Iteration:- %d, Error:- %0.6f\n", iter, error);
        }
        iter++;
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float runtime_ms = 0.0f;
    cudaEventElapsedTime(&runtime_ms, start, stop);
    
    printf(" total: %f s\n", runtime_ms / 1000.0f);
    
    cudaFree(A);
    cudaFree(Anew);
    cudaFree(d_error);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
