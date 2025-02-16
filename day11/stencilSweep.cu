#include <stdio.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>

__constant__ float val[7];

__global__ void stencilGPU(float *input, float *output, float *val, int N)
{
    int depth = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(depth >=1 && depth < N - 1 && row >=1 && row < N - 1 && col >=1 && col < N - 1)
    {   
        output[depth * N * N + row * N + col] = val[0] * input[depth * N * N + row * N + col] +
                                                val[1] * input[depth * N * N + row * N + col - 1] +
                                                val[2] * input[depth * N * N + row * N + col + 1] +
                                                val[3] * input[depth * N * N + (row - 1) * N + col] +
                                                val[4] * input[depth * N * N + (row + 1) * N + col] +
                                                val[5] * input[(depth - 1) * N * N + row * N + col] +
                                                val[6] * input[(depth + 1) * N * N + row * N + col];
    }    
}

__global__ void init(float *input, int N, unsigned int seed)
{
    int depth = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(depth < N && row < N && col < N)
    {
        curandState state;
        curand_init(seed, depth * N * N + row * N + col, 0, &state);
        input[depth * N * N + row * N + col] = curand_uniform(&state);
    }
}

void stencilCPU(float *input, float *output, float *val, int N)
{
    for (int depth = 1; depth < N - 1; depth++)
    {
        for (int row = 1; row < N - 1; row++)
        {
            for (int col = 1; col < N - 1; col++)
            {
                output[depth * N * N + row * N + col] = val[0] * input[depth * N * N + row * N + col] +
                                                        val[1] * input[depth * N * N + row * N + col - 1] +
                                                        val[2] * input[depth * N * N + row * N + col + 1] +
                                                        val[3] * input[depth * N * N + (row - 1) * N + col] +
                                                        val[4] * input[depth * N * N + (row + 1) * N + col] +
                                                        val[5] * input[(depth - 1) * N * N + row * N + col] +
                                                        val[6] * input[(depth + 1) * N * N + row * N + col];
            }
        }
    }
}

__global__ void verify(float *output_cpu, float *output_gpu, int N, int *errors)
{
    int depth = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (depth >= 1 && depth < N - 1 && row >= 1 && row < N - 1 && col >= 1 && col < N - 1)
    {
        if (fabs(output_cpu[depth * N * N + row * N + col] - output_gpu[depth * N * N + row * N + col]) > 1e-4)
        {
            atomicAdd(errors, 1);
        }
    }
}

int main()
{
    int N = 128;
    float *input, *output_gpu, *output_cpu;

    size_t size = N * N * N * sizeof(float);
    cudaMallocManaged(&input, size);
    cudaMallocManaged(&output_gpu, size);
    cudaMallocManaged(&output_cpu, size);

    cudaMemset(output_gpu, 0, size);
    cudaMemset(output_cpu, 0, size);

    dim3 threads(8, 8, 8);
    dim3 blocks((N + threads.x - 1)/threads.x, (N + threads.y - 1)/threads.y, (N + threads.z - 1)/threads.z);

    init<<<blocks, threads>>>(input, N, time(NULL));
    cudaDeviceSynchronize();

    float *val;
    cudaMallocManaged(&val, 7 * sizeof(float));
    val[0] = 1.0f;
    val[1] = 2.0f;
    val[2] = 3.0f;
    val[3] = 4.0f;
    val[4] = 5.0f;
    val[5] = 6.0f;
    val[6] = 7.0f;

    float gpu_time, cpu_time;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    stencilGPU<<<blocks, threads>>>(input, output_gpu, val, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);

    clock_t cpu_start = clock();
    stencilCPU(input, output_cpu, val, N);
    cpu_time = ((double)(clock() - cpu_start) / CLOCKS_PER_SEC) * 1000;

    int *errors;
    cudaMallocManaged(&errors, sizeof(int));
    verify<<<blocks, threads>>>(output_cpu, output_gpu, N, errors);
    cudaDeviceSynchronize();

    printf("Number of mismatches: %d\n", *errors);
    printf("GPU time: %f ms\n", gpu_time);
    printf("CPU time: %f ms\n", cpu_time);

    cudaFree(input);
    cudaFree(output_gpu);
    cudaFree(output_cpu);
    cudaFree(val);
    cudaFree(errors);

    return 0;
}