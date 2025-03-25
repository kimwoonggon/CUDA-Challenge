#include <stdio.h>
#include <math.h>
#include <curand_kernel.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define N 64
#define MEM 16
#define BATCH 128

__global__ void init(float *x, int pitch, unsigned int seed) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.z;

    if (row < N && col < N && batch < BATCH) {
        curandState state;
        curand_init(seed, batch * N * N + row * N + col, 0, &state);
        x[batch * pitch + row * N + col] = curand_uniform(&state) * 10.0f;
    }
}

void matMulCPU(float *a, float *b, float *c, int pitch) {
    float val;
    for (int batch = 0; batch < BATCH; batch++) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                val = 0.0f;
                for (int k = 0; k < N; k++) {
                    val += a[batch * pitch + i * N + k] * b[batch * pitch + k * N + j];
                }
                c[batch * pitch + i * N + j] = val;
            }
        }
    }
}

bool verify(float *c_cpu, float *c_gpu, int pitch) {
    for (int batch = 0; batch < BATCH; batch++) {
        for (int i = 0; i < N * N; i++) {
            float diff = fabsf(c_cpu[batch * pitch + i] - c_gpu[batch * pitch + i]);
            if (diff > 1e-2f) {
                printf("Mismatch at batch %d, index %d: CPU=%f, GPU=%f\n",
                       batch, i, c_cpu[batch * pitch + i], c_gpu[batch * pitch + i]);
                return false;
            }
        }
    }
    return true;
}

int main() {
    int size = N * N * BATCH * sizeof(float);
    int pitch = N * N;

    float *a, *b, *c_cpu, *c_gpu;
    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c_cpu, size);
    cudaMallocManaged(&c_gpu, size);

    dim3 threads(MEM, MEM);
    dim3 blocks((N + MEM - 1) / MEM, (N + MEM - 1) / MEM, BATCH);

    init<<<blocks, threads>>>(a, pitch, time(NULL));
    init<<<blocks, threads>>>(b, pitch, time(NULL) + 111);
    cudaDeviceSynchronize();

    cudaMemset(c_cpu, 0, size);
    cudaMemset(c_gpu, 0, size);

    clock_t start_cpu = clock();
    matMulCPU(a, b, c_cpu, pitch);
    clock_t end_cpu = clock();
    double cpu_time = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC * 1000;

    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;

    cudaEvent_t start, stop;
    float gpu_time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, N, N,
        &alpha,
        b, N, pitch,
        a, N, pitch,
        &beta,
        c_gpu, N, pitch,
        BATCH
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    cublasDestroy(handle);

    bool match = verify(c_cpu, c_gpu, pitch);
    if (match)
        printf("Batched matmul success\n");
    else
        printf("Batched matmul mismatch\n");

    printf("CPU time: %.2f ms\n", cpu_time);
    printf("cuBLAS batched time: %.2f ms\n", gpu_time);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c_cpu);
    cudaFree(c_gpu);

    return 0;
}
