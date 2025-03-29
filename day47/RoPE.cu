#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define THREADS 32
#define OFFSET3D(b, s, d, seq_len, dim) (((b) * (seq_len) + (s)) * (dim) + (d))

__global__ void ropeKernel(float*  d_input, float* d_output, float* d_cos_table, float* d_sin_table, int batch, int seq_len, int dim)
{
    int b = blockIdx.z;
    int s = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < batch && s < seq_len && d < dim / 2)
    {
        int d_even = d * 2;
        int d_odd  = d_even + 1;

        int idx_even = OFFSET3D(b, s, d_even, seq_len, dim);
        int idx_odd  = OFFSET3D(b, s, d_odd, seq_len, dim);
        int idx_theta = s * dim + d_even;

        float x0 = d_input[idx_even];
        float x1 = d_input[idx_odd];
        float cos_theta = d_cos_table[idx_theta];
        float sin_theta = d_sin_table[idx_theta];

        d_output[idx_even] = x0 * cos_theta - x1 * sin_theta;
        d_output[idx_odd]  = x0 * sin_theta + x1 * cos_theta;
    }
}

void initRoPE(float* input, float* cos_table, float* sin_table, int batch, int seq_len, int dim)
{
    for (int b = 0; b < batch; ++b)
    {
        for (int s = 0; s < seq_len; ++s)
        {
            for (int d = 0; d < dim; ++d)
            {
                int idx = OFFSET3D(b, s, d, seq_len, dim);
                input[idx] = sinf(idx * 0.001f);
            }
        }
    }

    for (int s = 0; s < seq_len; ++s)
    {
        float theta = 0.01f * s;
        for (int d = 0; d < dim; ++d)
        {
            int idx = s * dim + d;
            cos_table[idx] = cosf(theta);
            sin_table[idx] = sinf(theta);
        }
    }
}

int main()
{
    int batch = 32, seq_len = 128, dim = 256;
    float rope_time;

    size_t total_size = batch * seq_len * dim * sizeof(float);
    size_t table_size = seq_len * dim * sizeof(float);

    float *d_input, *d_output, *d_cos_table, *d_sin_table;
    cudaMallocManaged(&d_input, total_size);
    cudaMallocManaged(&d_output, total_size);
    cudaMallocManaged(&d_cos_table, table_size);
    cudaMallocManaged(&d_sin_table, table_size);

    initRoPE(d_input, d_cos_table, d_sin_table, batch, seq_len, dim);

    dim3 threads(THREADS, THREADS);
    dim3 blocks((dim / 2 + threads.x - 1) / threads.x, (seq_len + threads.y - 1) / threads.y, batch);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    cudaEventRecord(start);
    ropeKernel<<<blocks, threads>>>(d_input, d_output, d_cos_table, d_sin_table, batch, seq_len, dim);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&rope_time, start, stop);

    printf("RoPE execution time: %.2f ms\n", rope_time);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_cos_table);
    cudaFree(d_sin_table);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
