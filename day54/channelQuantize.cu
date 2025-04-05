#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include <cmath>
#include <algorithm>


#define N 1024
#define TILE 16

__global__ void init(float *x, unsigned int seed)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N)
    {
        curandState state;
        curand_init(seed, row * N + col, 0, &state);
        x[row * N + col] = curand_uniform(&state) * 2.0f - 1.0f;
    }
}

__global__ void quantize(const float* input, int8_t* output, const float* scale, const int8_t* zp, bool per_row)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N)
    {
        int channel = per_row ? row : col;
        float s = scale[channel];
        int8_t z = zp[channel];
        int idx = row * N + col;

        int q = __float2int_rn(input[idx] / s) + z;
        output[idx] = max(-128, min(127, q));
    }
}

__global__ void dequantize(const int8_t* a, const int8_t* b, float* c, const float* scale_a, const float* scale_b, const int8_t* zp_a, const int8_t* zp_b)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
    {
        int32_t acc = 0;
        for (int k = 0; k < N; ++k)
        {
            int8_t qa = a[row * N + k];
            int8_t qb = b[k * N + col];
            acc += (int32_t)(qa - zp_a[row]) * (int32_t)(qb - zp_b[col]);
        }
        float result = acc * scale_a[row] * scale_b[col];
        c[row * N + col] = result;
    }
}

void qparams(const float* input, float* scale, int8_t* zp, bool per_row)
{
    for (int i = 0; i < N; ++i)
    {
        float min_val = 1e30f, max_val = -1e30f;
        for (int j = 0; j < N; ++j)
        {
            float val = per_row ? input[i * N + j] : input[j * N + i];
            min_val = fminf(min_val, val);
            max_val = fmaxf(max_val, val);
        }
        float range = max_val - min_val;
        float s = range / 255.0f;
        int8_t z = static_cast<int8_t>(std::round(-min_val / s));

        scale[i] = (s == 0.0f) ? 1.0f : s;
        zp[i] = z;
    }
}

int main()
{
    float *a_fp32, *b_fp32, *c_fp32;
    int8_t *a_int8, *b_int8;
    float *scale_a, *scale_b;
    int8_t *zp_a, *zp_b;

    size_t size_matrix = N * N;
    size_t size_bytes = size_matrix * sizeof(float);
    size_t size_int8 = size_matrix * sizeof(int8_t);

    cudaMallocManaged(&a_fp32, size_bytes);
    cudaMallocManaged(&b_fp32, size_bytes);
    cudaMallocManaged(&c_fp32, size_bytes);
    cudaMallocManaged(&a_int8, size_int8);
    cudaMallocManaged(&b_int8, size_int8);

    cudaMallocManaged(&scale_a, N * sizeof(float));
    cudaMallocManaged(&scale_b, N * sizeof(float));
    cudaMallocManaged(&zp_a, N * sizeof(int8_t));
    cudaMallocManaged(&zp_b, N * sizeof(int8_t));

    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    init<<<blocks, threads>>>(a_fp32, time(NULL));
    init<<<blocks, threads>>>(b_fp32, time(NULL));
    cudaDeviceSynchronize();

    qparams(a_fp32, scale_a, zp_a, true);
    qparams(b_fp32, scale_b, zp_b, false);

    quantize<<<blocks, threads>>>(a_fp32, a_int8, scale_a, zp_a, true);
    quantize<<<blocks, threads>>>(b_fp32, b_int8, scale_b, zp_b, false);
    cudaDeviceSynchronize();

    cudaMemset(c_fp32, 0, size_bytes);

    cudaEvent_t start, stop;
    float elapsed_time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    dequantize<<<blocks, threads>>>(a_int8, b_int8, c_fp32, scale_a, scale_b, zp_a, zp_b);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);

    printf("Per-channel Quantized MatMul Time: %.2f ms\n", elapsed_time);

    cudaFree(a_fp32); cudaFree(b_fp32); cudaFree(c_fp32);
    cudaFree(a_int8); cudaFree(b_int8);
    cudaFree(scale_a); cudaFree(scale_b);
    cudaFree(zp_a); cudaFree(zp_b);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
