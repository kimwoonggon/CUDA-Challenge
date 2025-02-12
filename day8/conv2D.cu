#include <stdio.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>

#define EPSILON 1e-3

__global__ void conv2D(float *input, float *output, float *filter, int radius, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < N && col < N) 
    {
        float val = 0.0f;
        int f_size = 2 * radius + 1;

        for(int row_f = 0; row_f < f_size; row_f++) 
        {
            for(int col_f = 0; col_f < f_size; col_f++) 
            {
                int cur_row = row - radius + row_f;
                int cur_col = col - radius + col_f;

                if(cur_row >=0 && cur_row < N && cur_col >= 0 && cur_col < N) 
                {
                    val += filter[row_f * f_size + col_f] * input[cur_row * N + cur_col]; 
                }
            }
        }
        output[row * N + col] = val;
    }
}

void conv2D_cpu(float *input, float *output, float *filter, int radius, int N) 
{
    int f_size = 2 * radius + 1;

    for(int row = 0; row < N; row++) 
    {
        for(int col = 0; col < N; col++) 
        {
            float val = 0.0f;

            for(int row_f = 0; row_f < f_size; row_f++) 
            {
                for(int col_f = 0; col_f < f_size; col_f++) 
                {
                    int cur_row = row - radius + row_f;
                    int cur_col = col - radius + col_f;
                    
                    if(cur_row >= 0 && cur_row < N && cur_col >= 0 && cur_col < N) 
                    {
                        val += filter[row_f * f_size + col_f] * input[cur_row * N + cur_col];
                    }
                }
            }
            output[row * N + col] = val;
        }
    }
}

__global__ void heNormal(float *filter, int radius, unsigned int seed) {
    int t_x = blockIdx.x * blockDim.x + threadIdx.x;
    int f_size = 2 * radius + 1;
    int total = f_size * f_size;

    if(t_x < total) 
    {
        float fan_in = (float)total;
        float std_dev = sqrtf(2.0f / fan_in);
        
        curandState state;
        curand_init(seed, t_x, 0, &state);
        filter[t_x] = curand_normal(&state) * std_dev;
    }
}

__global__ void verify(float *gpu_res, float *cpu_res, int *errors, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(row < N && col < N) 
    {
        int idx = row * N + col;
        if(fabsf(gpu_res[idx] - cpu_res[idx]) > EPSILON) 
        {
            atomicAdd(errors, 1);
        }
    }
}

__global__ void init(float *input, int N, unsigned int seed) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < N && col < N) 
    {
        curandState state;
        curand_init(seed, row * N + col, 0, &state);
        input[row * N + col] = curand_uniform(&state);
    }
}

int main() {
    int N = 128;
    int radius = 1;
    float gpu_time, cpu_time;

    float *in_gpu, *out_gpu, *filter, *cpu_res_gpu;
    int *d_errors;

    cudaMalloc(&in_gpu, N*N*sizeof(float));
    cudaMalloc(&out_gpu, N*N*sizeof(float));
    cudaMalloc(&filter, (2*radius+1)*(2*radius+1)*sizeof(float));
    cudaMalloc(&cpu_res_gpu, N*N*sizeof(float));
    cudaMalloc(&d_errors, sizeof(int));

    dim3 init_threads(16, 16);
    dim3 init_blocks((N+15)/16, (N+15)/16);
    init<<<init_blocks, init_threads>>>(in_gpu, N, time(NULL));
    
    size_t f_size = (2*radius+1)*(2*radius+1);
    heNormal<<<(f_size+255)/256, 256>>>(filter, radius, time(NULL));
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    dim3 conv_threads(16, 16);
    dim3 conv_blocks((N+15)/16, (N+15)/16);
    
    cudaEventRecord(start);
    conv2D<<<conv_blocks, conv_threads>>>(in_gpu, out_gpu, filter, radius, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);

    float *input_cpu = (float*)malloc(N*N*sizeof(float));
    float *filter_cpu = (float*)malloc((2*radius+1)*(2*radius+1)*sizeof(float));
    float *output_cpu = (float*)malloc(N*N*sizeof(float));
    
    cudaMemcpy(input_cpu, in_gpu, N*N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(filter_cpu, filter, (2*radius+1)*(2*radius+1)*sizeof(float), cudaMemcpyDeviceToHost);

    clock_t cpu_start = clock();
    conv2D_cpu(input_cpu, output_cpu, filter_cpu, radius, N);
    cpu_time = ((double)(clock() - cpu_start) / CLOCKS_PER_SEC) * 1000;

    cudaMemcpy(cpu_res_gpu, output_cpu, N*N*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(d_errors, 0, sizeof(int));
    verify<<<conv_blocks, conv_threads>>>(out_gpu, cpu_res_gpu, d_errors, N);
    
    int errors;
    cudaMemcpy(&errors, d_errors, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Mismatches: %d\n", errors);
    printf("GPU Convolution Execution Time: %.2f ms\n", gpu_time);
    printf("CPU Convolution Execution Time: %.2f ms\n", cpu_time);

    free(input_cpu);
    free(filter_cpu);
    free(output_cpu);
    cudaFree(in_gpu);
    cudaFree(out_gpu);
    cudaFree(filter);
    cudaFree(cpu_res_gpu);
    cudaFree(d_errors);

    return 0;
}