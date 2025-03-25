#include <stdio.h>
#include <math.h>
#include <curand_kernel.h>
#include <time.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define N 1951
#define MEM 16

__global__ void init(float *x, unsigned int seed) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N)
    {
        curandState state;
        curand_init(seed, (row * N + col), 0, &state);
        x[row * N + col] = curand_uniform(&state) * 1000.0f;
    }
}

__global__ void matMulGPU(float *a, float *b, float *c)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N)
    {
        float val = 0.0f;
        for (int k = 0; k < N; k++)
        {
            val += a[row * N + k] * b[k * N + col];
        }
        c[row * N + col] = val;
    }
}

__global__ void matMulTiled(float *a, float *b, float *c)
{
    __shared__ float mem_a[MEM][MEM];
    __shared__ float mem_b[MEM][MEM];
    int t_x = threadIdx.x;
    int t_y = threadIdx.y;
    int b_x = blockIdx.x;
    int b_y = blockIdx.y;
    int row = b_y * MEM + t_y;
    int col = b_x * MEM + t_x;
    float val = 0.0f;
    for (int stride = 0; stride < ceilf((float)N / MEM); stride++)
    {
        if ((row < N) && ((stride * MEM + t_x) < N))
            mem_a[t_y][t_x] = a[row * N + stride * MEM + t_x];
        else
            mem_a[t_y][t_x] = 0.0f;
        if ((col < N) && ((stride * MEM + t_y) < N))
            mem_b[t_y][t_x] = b[(stride * MEM + t_y) * N + col];
        else
            mem_b[t_y][t_x] = 0.0f;
        __syncthreads();
        for (int k = 0; k < MEM; k++)
            val += mem_a[t_y][k] * mem_b[k][t_x];
        __syncthreads();
    }
    if (row < N && col < N)
        c[row * N + col] = val;
}

void matMulCPU(float *a, float *b, float *c)
{
    float val = 0.0f;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            val = 0.0f;
            for (int k = 0; k < N; k++)
                val += a[i * N + k] * b[k * N + j];
            c[i * N + j] = val;
        }
    }
}

bool verify(float *ref, float *gpu, float *tiled, float *cublas)
{
    for (int i = 0; i < N * N; i++)
    {
        float r = ref[i];
        float d1 = fabsf(r - gpu[i]);
        float d2 = fabsf(r - tiled[i]);
        float d3 = fabsf(r - cublas[i]);
        if (d1 > 1e-2f || d2 > 1e-2f || d3 > 1e-2f)
        {
            printf("Mismatch at %d: CPU=%f, GPU=%f, Tiled=%f, cuBLAS=%f\n", i, r, gpu[i], tiled[i], cublas[i]);
            return false;
        }
    }
    return true;
}

int main()
{
    float *a, *b, *c_gpu, *c_gpu_tiled, *c_cpu, *c_cublas;
    int size = N * N * sizeof(float);

    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c_gpu, size);
    cudaMallocManaged(&c_gpu_tiled, size);
    cudaMallocManaged(&c_cpu, size);
    cudaMallocManaged(&c_cublas, size);

    dim3 n_threads(MEM, MEM);
    dim3 n_blocks((N + MEM - 1) / MEM, (N + MEM - 1) / MEM);

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
    cudaMemset(c_cublas, 0, size);

    cudaEvent_t start, stop;
    float gpu_time, tiled_time, cublas_time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matMulGPU<<<n_blocks, n_threads>>>(a, b, c_gpu);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    matMulTiled<<<n_blocks, n_threads>>>(a, b, c_gpu_tiled);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&tiled_time, start, stop);
    cudaDeviceSynchronize();

    clock_t start_cpu = clock();
    matMulCPU(a, b, c_cpu);
    clock_t end_cpu = clock();
    double cpu_time = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC * 1000;

    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;
    cudaEventRecord(start);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                &alpha, b, N, a, N, &beta, c_cublas, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cublas_time, start, stop);
    cublasDestroy(handle);

    bool flag = verify(c_cpu, c_gpu, c_gpu_tiled, c_cublas);
    if (flag)
        printf("All outputs match!\n");
    else
        printf("Mismatch found!\n");

    printf("CPU time: %.2f ms\n", cpu_time);
    printf("GPU time: %.2f ms\n", gpu_time);
    printf("Tiled GPU time: %.2f ms\n", tiled_time);
    printf("cuBLAS time: %.2f ms\n", cublas_time);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c_cpu);
    cudaFree(c_gpu);
    cudaFree(c_gpu_tiled);
    cudaFree(c_cublas);
    cudaStreamDestroy(strm1);
    cudaStreamDestroy(strm2);

    return 0;
}
