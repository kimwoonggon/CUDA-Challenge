#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublasLt.h>
#include <curand_kernel.h>
#include <time.h>
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

__global__ void matMulFusedTiledQuant(float *a, float *b, half *c, float scale_a, float scale_b)
{
    __shared__ int8_t tile_a[TILE][TILE];
    __shared__ int8_t tile_b[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int32_t acc = 0;

    for (int m = 0; m < N / TILE; m++)
    {
        if (row < N && (m * TILE + tx) < N)
        {
            float val_a = a[row * N + m * TILE + tx];
            tile_a[ty][tx] = max(-128, min(127, __float2int_rn(val_a / scale_a)));
        }
        else
            tile_a[ty][tx] = 0;

        if (col < N && (m * TILE + ty) < N)
        {
            float val_b = b[(m * TILE + ty) * N + col];
            tile_b[ty][tx] = max(-128, min(127, __float2int_rn(val_b / scale_b)));
        }
        else
            tile_b[ty][tx] = 0;

        __syncthreads();

        for (int k = 0; k < TILE; k++)
        {
            acc += (int32_t)tile_a[ty][k] * (int32_t)tile_b[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N)
    {
        float val = acc * scale_a * scale_b;
        c[row * N + col] = __float2half(val);
    }
}

void runCuBLASLtINT8(const int8_t *a_i8, const int8_t *b_i8, half *c_fp16)
{
    cublasLtHandle_t ltHandle;
    cublasLtMatmulDesc_t opDesc;
    cublasLtMatrixLayout_t aLayout, bLayout, cLayout;
    cublasLtMatmulPreference_t preference;
    size_t workspaceSize = 4 * 1024 * 1024;
    void* workspace;
    cudaMalloc(&workspace, workspaceSize);

    cublasLtCreate(&ltHandle);

    cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32I, CUDA_R_32I);
    cublasLtMatrixLayoutCreate(&aLayout, CUDA_R_8I, N, N, N);
    cublasLtMatrixLayoutCreate(&bLayout, CUDA_R_8I, N, N, N);
    cublasLtMatrixLayoutCreate(&cLayout, CUDA_R_16F, N, N, N);

    cublasLtMatmulPreferenceCreate(&preference);
    cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));

    int32_t alpha = 1;
    int32_t beta = 0;

    cublasLtMatmul(ltHandle, opDesc, &alpha, a_i8, aLayout, b_i8, bLayout, &beta, c_fp16, cLayout, c_fp16, cLayout, NULL, workspace, workspaceSize, 0);

    cublasLtDestroy(ltHandle);
    cudaFree(workspace);
}

int main()
{
    float *a_fp32, *b_fp32;
    int8_t *a_int8, *b_int8;
    half *c_fused, *c_cublas;

    size_t size_fp32 = N * N * sizeof(float);
    size_t size_int8 = N * N * sizeof(int8_t);
    size_t size_fp16 = N * N * sizeof(half);

    cudaMallocManaged(&a_fp32, size_fp32);
    cudaMallocManaged(&b_fp32, size_fp32);
    cudaMallocManaged(&a_int8, size_int8);
    cudaMallocManaged(&b_int8, size_int8);
    cudaMallocManaged(&c_fused, size_fp16);
    cudaMallocManaged(&c_cublas, size_fp16);

    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    init<<<blocks, threads>>>(a_fp32, time(NULL));
    init<<<blocks, threads>>>(b_fp32, time(NULL));
    cudaDeviceSynchronize();

    float scale_a = 0.02f;
    float scale_b = 0.02f;

    for (int i = 0; i < N * N; i++)
    {
        a_int8[i] = static_cast<int8_t>(std::max(-128, std::min(127, static_cast<int>(std::round(a_fp32[i] / scale_a)))));
        b_int8[i] = static_cast<int8_t>(std::max(-128, std::min(127, static_cast<int>(std::round(b_fp32[i] / scale_b)))));
    }

    float fused_time = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    matMulFusedTiledQuant<<<blocks, threads>>>(a_fp32, b_fp32, c_fused, scale_a, scale_b);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&fused_time, start, stop);

    float cublas_time = 0.0f;
    cudaEventRecord(start);
    runCuBLASLtINT8(a_int8, b_int8, c_cublas);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cublas_time, start, stop);

    printf("Fused Tiled (int8→fp16): %.2f ms\n", fused_time);
    printf("cuBLASLt INT8 GEMM: %.2f ms\n", cublas_time);

    cudaFree(a_fp32);
    cudaFree(b_fp32);
    cudaFree(a_int8);
    cudaFree(b_int8);
    cudaFree(c_fused);
    cudaFree(c_cublas);

    return 0;
}
