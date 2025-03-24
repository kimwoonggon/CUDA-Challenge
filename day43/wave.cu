#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <math.h>

#define N (1<<12)
#define THREADS 1024
#define BLOCKS ((N * N + THREADS - 1) / THREADS)
#define STEPS 500
#define C 0.5f

__global__ void wave_step(float *u_prev, float *u_curr, float *u_next, int n, float dt, float dx2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n)
    {
        int i = idx / n;
        int j = idx % n;
        if (i > 0 && i < n - 1 && j > 0 && j < n - 1)
        {
            float up    = u_curr[(i - 1) * n + j];
            float down  = u_curr[(i + 1) * n + j];
            float left  = u_curr[i * n + j - 1];
            float right = u_curr[i * n + j + 1];
            float center = u_curr[idx];
            float laplacian = up + down + left + right - 4.0f * center;
            u_next[idx] = 2.0f * center - u_prev[idx] + (C * C * dt * dt / dx2) * laplacian;
        }
    }
}

void init(float *u_prev, float *u_curr, int n, float dt)
{
    for (int i = 0; i < n * n; i++)
    {
        u_prev[i] = 0.0f;
        u_curr[i] = 0.0f;
    }

    int cx = n / 2;
    int cy = n / 2;
    float amplitude = 1.0f;
    float velocity = -0.2f;

    u_curr[cy * n + cx] = amplitude;
    u_prev[cy * n + cx] = amplitude - velocity * dt;
}

int main()
{
    int size = N * N * sizeof(float);
    float *u_prev, *u_curr, *u_next;

    cudaMallocManaged(&u_prev, size);
    cudaMallocManaged(&u_curr, size);
    cudaMallocManaged(&u_next, size);

    float dt = 0.1f;
    float dx = 1.0f;
    float dx2 = dx * dx;

    init(u_prev, u_curr, N, dt);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int step = 0; step < STEPS; step++) 
    {
        wave_step<<<BLOCKS, THREADS>>>(u_prev, u_curr, u_next, N, dt, dx2);
        cudaDeviceSynchronize();
        float *temp = u_prev;
        u_prev = u_curr;
        u_curr = u_next;
        u_next = temp;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("\nFinished %d steps in %.3f ms (%.3f ms/step)\n", STEPS, ms, ms / STEPS);

    printf("\nSample displacement values:\n");
    for (int i = N / 2 - 2; i <= N / 2 + 2; i++) 
    {
        for (int j = N / 2 - 2; j <= N / 2 + 2; j++) 
        {
            printf("%.4f ", u_curr[i * N + j]);
        }
        printf("\n");
    }

    cudaFree(u_prev);
    cudaFree(u_curr);
    cudaFree(u_next);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
