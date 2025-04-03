#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define THREADS 32

__global__ void layerNormKernel(float* d_input, float* d_output, float* d_gamma, float* d_beta, int batch, int seq_len, int dim, float epsilon)
{
    int b = blockIdx.z;
    int s = blockIdx.y;
    int d = threadIdx.x;

    if (b >= batch || s >= seq_len || d >= dim) return;

    int base = (b * seq_len + s) * dim;

    __shared__ float mean;
    __shared__ float variance;

    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;

    for (int i = d; i < dim; i += blockDim.x) 
    {
        float val = d_input[base + i];
        local_sum += val;
        local_sq_sum += val * val;
    }

    __shared__ float buffer[THREADS * 2];
    buffer[d] = local_sum;
    buffer[d + blockDim.x] = local_sq_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) 
    {
        if (d < stride) 
        {
            buffer[d] += buffer[d + stride];
            buffer[d + blockDim.x] += buffer[d + blockDim.x + stride];
        }
        __syncthreads();
    }

    if (d == 0) 
    {
        mean = buffer[0] / dim;
        variance = buffer[blockDim.x] / dim - mean * mean;
    }
    __syncthreads();

    for (int i = d; i < dim; i += blockDim.x) 
    {
        float x = d_input[base + i];
        float norm = (x - mean) / sqrtf(variance + epsilon);
        d_output[base + i] = norm * d_gamma[i] + d_beta[i];
    }
}

void initLayerNorm(float* input, float* gamma, float* beta, int batch, int seq_len, int dim)
{
    for (int b = 0; b < batch; ++b) 
    {
        for (int s = 0; s < seq_len; ++s) 
        {
            for (int d = 0; d < dim; ++d) 
            {
                int idx = (b * seq_len + s) * dim + d;
                input[idx] = sinf(idx * 0.01f);
            }
        }
    }

    for (int d = 0; d < dim; ++d) 
    {
        gamma[d] = 1.0f;
        beta[d]  = 0.0f;
    }
}

int main()
{
    int batch = 32, seq_len = 128, dim = 256;
    float epsilon = 1e-5f;
    float norm_time;

    size_t total_size = batch * seq_len * dim * sizeof(float);
    size_t param_size = dim * sizeof(float);

    float *d_input, *d_output, *d_gamma, *d_beta;
    cudaMallocManaged(&d_input, total_size);
    cudaMallocManaged(&d_output, total_size);
    cudaMallocManaged(&d_gamma, param_size);
    cudaMallocManaged(&d_beta, param_size);

    initLayerNorm(d_input, d_gamma, d_beta, batch, seq_len, dim);

    dim3 threads(THREADS);
    dim3 blocks(1, seq_len, batch);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    layerNormKernel<<<blocks, threads>>>(d_input, d_output, d_gamma, d_beta, batch, seq_len, dim, epsilon);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&norm_time, start, stop);

    printf("LayerNorm execution time: %.2f ms\n", norm_time);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
