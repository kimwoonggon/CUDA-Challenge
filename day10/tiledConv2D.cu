#include <stdio.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>

#define RADIUS 1
#define EPSILON 1e-3
#define IN_TILE_DIM 32
#define OUT_TILE_DIM (IN_TILE_DIM - 2 * RADIUS)

__constant__ float filter[2 * RADIUS + 1][2 * RADIUS + 1];

__global__ void conv2D(float *input, float *output, int N, int M) 
{
    int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - RADIUS;
    int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - RADIUS;

    __shared__ float N_shared[IN_TILE_DIM][IN_TILE_DIM];

    if (row >= 0 && row < N && col >= 0 && col < M)
        N_shared[threadIdx.y][threadIdx.x] = input[row * M + col];
    else
        N_shared[threadIdx.y][threadIdx.x] = 0.0f;
    __syncthreads();

    int tile_row = threadIdx.y - RADIUS;
    int tile_col = threadIdx.x - RADIUS;

    if (tile_row >= 0 && tile_row < OUT_TILE_DIM && tile_col >= 0 && tile_col < OUT_TILE_DIM) 
    {
        int out_row = blockIdx.y * OUT_TILE_DIM + tile_row;
        int out_col = blockIdx.x * OUT_TILE_DIM + tile_col;
        if (out_row < N && out_col < M) 
        {
            float val = 0.0f;
            int f_size = 2 * RADIUS + 1;
            for (int row_f = 0; row_f < f_size; row_f++) 
            {
                for (int col_f = 0; col_f < f_size; col_f++) 
                {
                    val += filter[row_f][col_f] * 
                           N_shared[threadIdx.y + row_f - RADIUS][threadIdx.x + col_f - RADIUS];
                }
            }
            output[out_row * M + out_col] = val;
        }
    }
}

void conv2D_cpu(float *input, float *output, float *host_filter, int N, int M) 
{
    int f_size = 2 * RADIUS + 1;
    for (int row = 0; row < N; row++) 
    {
        for (int col = 0; col < M; col++) 
        {
            float val = 0.0f;
            for (int row_f = 0; row_f < f_size; row_f++) 
            {
                for (int col_f = 0; col_f < f_size; col_f++) 
                {
                    int cur_row = row - RADIUS + row_f;
                    int cur_col = col - RADIUS + col_f;
                    if (cur_row >= 0 && cur_row < N && cur_col >= 0 && cur_col < M) 
                    {
                        val += host_filter[row_f * f_size + col_f] * input[cur_row * M + cur_col];
                    }
                }
            }
            output[row * M + col] = val;
        }
    }
}

__global__ void verify(float *gpu_res, float *cpu_res, int *errors, int N, int M) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < M) 
    {
        int idx = row * M + col;
        if (fabsf(gpu_res[idx] - cpu_res[idx]) > EPSILON) 
        {
            atomicAdd(errors, 1);
        }
    }
}

__global__ void init(float *input, int N, int M, unsigned int seed) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < M) 
    {
        curandState state;
        curand_init(seed, row * M + col, 0, &state);
        input[row * M + col] = curand_uniform(&state);
    }
}

__global__ void heNormal_kernel(float *filter_temp, unsigned int seed) 
{
    int t_x = blockIdx.x * blockDim.x + threadIdx.x;
    int f_size = 2 * RADIUS + 1;
    int total = f_size * f_size;

    if (t_x < total) 
    {
        float fan_in = (float)total;
        float std_dev = sqrtf(2.0f / fan_in);
        
        curandState state;
        curand_init(seed, t_x, 0, &state);
        filter_temp[t_x] = curand_normal(&state) * std_dev;
    }
}

int main() 
{
    int N = 128, M = 64;
    float gpu_time, cpu_time;

    float *in_gpu, *out_gpu, *cpu_res_gpu;
    int *d_errors;

    cudaMalloc(&in_gpu, N * M * sizeof(float));
    cudaMalloc(&out_gpu, N * M * sizeof(float));
    cudaMalloc(&cpu_res_gpu, N * M * sizeof(float));
    cudaMalloc(&d_errors, sizeof(int));

    dim3 threads(32, 32);
    dim3 blocks((M + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);
    init<<<blocks, threads>>>(in_gpu, N, M, time(NULL));
    cudaDeviceSynchronize();
    
    int f_size = (2 * RADIUS + 1) * (2 * RADIUS + 1);
    int numThreads = 256;
    int numBlocks = (f_size + numThreads - 1) / numThreads;

    float *temp_filter;
    cudaMalloc(&temp_filter, f_size * sizeof(float));
    heNormal_kernel<<<numBlocks, numThreads>>>(temp_filter, time(NULL));
    cudaDeviceSynchronize();

    float *host_filter = (float*)malloc(f_size * sizeof(float));
    cudaMemcpy(host_filter, temp_filter, f_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaMemcpyToSymbol(filter, host_filter, f_size * sizeof(float));
    
    cudaFree(temp_filter);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    dim3 conv_threads(IN_TILE_DIM, IN_TILE_DIM);
    dim3 conv_blocks((M + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM);
    
    cudaEventRecord(start);
    conv2D<<<conv_blocks, conv_threads>>>(in_gpu, out_gpu, N, M);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);

    float *input_cpu = (float*)malloc(N * M * sizeof(float));
    float *output_cpu = (float*)malloc(N * M * sizeof(float));
    
    cudaMemcpy(input_cpu, in_gpu, N * M * sizeof(float), cudaMemcpyDeviceToHost);

    clock_t cpu_start = clock();
    conv2D_cpu(input_cpu, output_cpu, host_filter, N, M);
    cpu_time = ((double)(clock() - cpu_start) / CLOCKS_PER_SEC) * 1000;

    cudaMemcpy(cpu_res_gpu, output_cpu, N * M * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(d_errors, 0, sizeof(int));
    verify<<<conv_blocks, conv_threads>>>(out_gpu, cpu_res_gpu, d_errors, N, M);
    cudaDeviceSynchronize();
    
    int errors;
    cudaMemcpy(&errors, d_errors, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Mismatches: %d\n", errors);
    printf("GPU Convolution Execution Time: %.2f ms\n", gpu_time);
    printf("CPU Convolution Execution Time: %.2f ms\n", cpu_time);

    free(input_cpu);
    free(output_cpu);
    free(host_filter);
    cudaFree(in_gpu);
    cudaFree(out_gpu);
    cudaFree(cpu_res_gpu);
    cudaFree(d_errors);

    return 0;
}
