#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define THREADS 32

__global__ void softmaxKernel(float* d_input, float* d_output, int batch, int seq_len, int dim)
{
    int b = blockIdx.z;
    int s = blockIdx.y;
    int t = threadIdx.x;

    if (b >= batch || s >= seq_len || t >= dim) return;

    int base = (b * seq_len + s) * dim;

    __shared__ float max_val;
    __shared__ float sum_exp;

    float local_max = -INFINITY;

    for (int i = t; i < dim; i += blockDim.x) 
    {
        float val = d_input[base + i];
        if (val > local_max) local_max = val;
    }

    __shared__ float buffer_max[THREADS];
    buffer_max[t] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) 
    {
        if (t < stride) 
        {
            buffer_max[t] = fmaxf(buffer_max[t], buffer_max[t + stride]);
        }
        __syncthreads();
    }

    if (t == 0) max_val = buffer_max[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = t; i < dim; i += blockDim.x) 
    {
        float exp_val = expf(d_input[base + i] - max_val);
        d_output[base + i] = exp_val;  // Store temporarily
        local_sum += exp_val;
    }

    __shared__ float buffer_sum[THREADS];
    buffer_sum[t] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) 
    {
        if (t < stride) 
        {
            buffer_sum[t] += buffer_sum[t + stride];
        }
        __syncthreads();
    }

    if (t == 0) sum_exp = buffer_sum[0];
    __syncthreads();

    for (int i = t; i < dim; i += blockDim.x) 
    {
        d_output[base + i] /= sum_exp;
    }
}

void initSoftmax(float* input, int batch, int seq_len, int dim)
{
    for (int b = 0; b < batch; ++b) 
    {
        for (int s = 0; s < seq_len; ++s) 
        {
            for (int d = 0; d < dim; ++d) 
            {
                int idx = (b * seq_len + s) * dim + d;
                input[idx] = sinf(idx * 0.02f);
            }
        }
    }
}

int main()
{
    int batch = 32, seq_len = 128, dim = 256;
    float softmax_time;

    size_t total_size = batch * seq_len * dim * sizeof(float);

    float *d_input, *d_output;
    cudaMallocManaged(&d_input, total_size);
    cudaMallocManaged(&d_output, total_size);

    initSoftmax(d_input, batch, seq_len, dim);

    dim3 threads(THREADS);
    dim3 blocks(1, seq_len, batch);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    softmaxKernel<<<blocks, threads>>>(d_input, d_output, batch, seq_len, dim);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&softmax_time, start, stop);

    printf("Softmax execution time: %.2f ms\n", softmax_time);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
