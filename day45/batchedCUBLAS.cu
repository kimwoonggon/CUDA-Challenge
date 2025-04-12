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

__global__ void batchedmatMulGPU(float *a, float *b, float *c, int pitch) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.z;
    
    if (row < N && col < N && batch < BATCH) {
        float val = 0.0f;
        for (int k = 0; k < N; k++) {
            val += a[batch * pitch + row * N + k] * b[batch * pitch + k * N + col];
        }
        c[batch * pitch + row * N + col] = val;
    }
}

__global__ void batchedmatMulTiled(float *a, float *b, float *c, int pitch) {
    __shared__ float mem_a[MEM][MEM];
    __shared__ float mem_b[MEM][MEM];
    
    int t_x = threadIdx.x;
    int t_y = threadIdx.y;
    int b_x = blockIdx.x;
    int b_y = blockIdx.y;
    int batch = blockIdx.z;
    
    int row = b_y * MEM + t_y;
    int col = b_x * MEM + t_x;
    float val = 0.0f;
    
    if (row < N && col < N && batch < BATCH) {
        for (int stride = 0; stride < ceilf((float)N / MEM); stride++) {
            if ((row < N) && ((stride * MEM + t_x) < N))
                mem_a[t_y][t_x] = a[batch * pitch + row * N + stride * MEM + t_x];
            else
                mem_a[t_y][t_x] = 0.0f;
                
            if ((col < N) && ((stride * MEM + t_y) < N))
                mem_b[t_y][t_x] = b[batch * pitch + (stride * MEM + t_y) * N + col];
            else
                mem_b[t_y][t_x] = 0.0f;
                
            __syncthreads();
            
            for (int k = 0; k < MEM; k++)
                val += mem_a[t_y][k] * mem_b[k][t_x];
                
            __syncthreads();
        }
        c[batch * pitch + row * N + col] = val;
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

bool verify(float *c_cpu, float *c_gpu, float *c_gpu_tiled, float *c_cublas, int pitch) {
    for (int batch = 0; batch < BATCH; batch++) {
        for (int i = 0; i < N * N; i++) {
            float ref = c_cpu[batch * pitch + i];
            float d1 = fabsf(ref - c_gpu[batch * pitch + i]);
            float d2 = fabsf(ref - c_gpu_tiled[batch * pitch + i]);
            float d3 = fabsf(ref - c_cublas[batch * pitch + i]);
            
            if (d1 > 1e-2f || d2 > 1e-2f || d3 > 1e-2f) {
                printf("Mismatch at batch %d, index %d: CPU=%f, GPU=%f, Tiled=%f, cuBLAS=%f\n",
                       batch, i, ref, c_gpu[batch * pitch + i], c_gpu_tiled[batch * pitch + i], c_cublas[batch * pitch + i]);
                return false;
            }
        }
    }
    return true;
}

int main() {
    int size = N * N * BATCH * sizeof(float);
    int pitch = N * N;

    float *a, *b, *c_cpu, *c_gpu, *c_gpu_tiled, *c_cublas;
    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c_cpu, size);
    cudaMallocManaged(&c_gpu, size);
    cudaMallocManaged(&c_gpu_tiled, size);
    cudaMallocManaged(&c_cublas, size);

    dim3 threads(MEM, MEM);
    dim3 blocks((N + MEM - 1) / MEM, (N + MEM - 1) / MEM, BATCH);

    init<<<blocks, threads>>>(a, pitch, time(NULL));
    init<<<blocks, threads>>>(b, pitch, time(NULL) + 111);
    cudaDeviceSynchronize();

    cudaMemset(c_cpu, 0, size);
    cudaMemset(c_gpu, 0, size);
    cudaMemset(c_gpu_tiled, 0, size);
    cudaMemset(c_cublas, 0, size);

    cudaEvent_t start, stop;
    float gpu_time, tiled_time, cublas_time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Timing for regular GPU implementation
    cudaEventRecord(start);
    batchedmatMulGPU<<<blocks, threads>>>(a, b, c_gpu, pitch);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    cudaDeviceSynchronize();

    // Timing for tiled GPU implementation
    cudaEventRecord(start);
    batchedmatMulTiled<<<blocks, threads>>>(a, b, c_gpu_tiled, pitch);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&tiled_time, start, stop);
    cudaDeviceSynchronize();
    
    clock_t start_cpu = clock();
    matMulCPU(a, b, c_cpu, pitch);
    clock_t end_cpu = clock();
    double cpu_time = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC * 1000;

    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;

    // Timing for cuBLAS implementation
    cudaEventRecord(start);
    cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, N, N,
        &alpha,
        b, N, pitch,
        a, N, pitch,
        &beta,
        c_cublas, N, pitch,
        BATCH
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cublas_time, start, stop);
    cublasDestroy(handle);

    bool match = verify(c_cpu, c_gpu, c_gpu_tiled, c_cublas, pitch);
    if (match)
        printf("All outputs match!\n");
    else
        printf("Mismatch found!\n");

    printf("CPU time: %.2f ms\n", cpu_time);
    printf("GPU time: %.2f ms\n", gpu_time);
    printf("Tiled GPU time: %.2f ms\n", tiled_time);
    printf("cuBLAS batched time: %.2f ms\n", cublas_time);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c_cpu);
    cudaFree(c_gpu);
    cudaFree(c_gpu_tiled);
    cudaFree(c_cublas);

    return 0;
}
