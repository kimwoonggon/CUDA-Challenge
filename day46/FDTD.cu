#include <cstdio>
#include <cuda_runtime.h>

#define IDX(x, y, z, nx, ny) ((z)*(nx)*(ny) + (y)*(nx) + (x))

#define NX 128
#define NY 128
#define NZ 128
#define STEPS 200
#define DT 0.01f
#define DX 1.0f

__global__ void update_h(float *Hx, float *Hy, float *Hz, const float *Ex, const float *Ey, const float *Ez, int nx, int ny, int nz, float dt, float dx) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx - 1 || y >= ny - 1 || z >= nz - 1) 
        return;

    int idx = IDX(x, y, z, nx, ny);

    Hx[idx] -= (dt / dx) * (Ez[IDX(x, y + 1, z, nx, ny)] - Ez[idx] - Ey[IDX(x, y, z + 1, nx, ny)] + Ey[idx]);
    Hy[idx] -= (dt / dx) * (Ex[IDX(x, y, z + 1, nx, ny)] - Ex[idx] - Ez[IDX(x + 1, y, z, nx, ny)] + Ez[idx]);
    Hz[idx] -= (dt / dx) * (Ey[IDX(x + 1, y, z, nx, ny)] - Ey[idx] - Ex[IDX(x, y + 1, z, nx, ny)] + Ex[idx]);
}

__global__ void update_e(float *Ex, float *Ey, float *Ez, const float *Hx, const float *Hy, const float *Hz, int nx, int ny, int nz, float dt, float dx) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x == 0 || y == 0 || z == 0 || x >= nx || y >= ny || z >= nz) 
        return;

    int idx = IDX(x, y, z, nx, ny);

    Ex[idx] += (dt / dx) * (Hz[idx] - Hz[IDX(x, y - 1, z, nx, ny)] - Hy[idx] + Hy[IDX(x, y, z - 1, nx, ny)]);
    Ey[idx] += (dt / dx) * (Hx[idx] - Hx[IDX(x, y, z - 1, nx, ny)] - Hz[idx] + Hz[IDX(x - 1, y, z, nx, ny)]);
    Ez[idx] += (dt / dx) * (Hy[idx] - Hy[IDX(x - 1, y, z, nx, ny)] - Hx[idx] + Hx[IDX(x, y - 1, z, nx, ny)]);
}

__global__ void inject_source(float *Ez, int nx, int ny, int nz, float t) 
{
    int cx = nx / 2;
    int cy = ny / 2;
    int cz = nz / 2;
    int idx = IDX(cx, cy, cz, nx, ny);
    Ez[idx] += sinf(0.1f * t);
}

int main() 
{
    int total = NX * NY * NZ;
    float *Ex, *Ey, *Ez;
    float *Hx, *Hy, *Hz;

    size_t size = total * sizeof(float);
    cudaMallocManaged(&Ex, size);
    cudaMallocManaged(&Ey, size);
    cudaMallocManaged(&Ez, size);
    cudaMallocManaged(&Hx, size);
    cudaMallocManaged(&Hy, size);
    cudaMallocManaged(&Hz, size);

    cudaMemset(Ex, 0, size);
    cudaMemset(Ey, 0, size);
    cudaMemset(Ez, 0, size);
    cudaMemset(Hx, 0, size);
    cudaMemset(Hy, 0, size);
    cudaMemset(Hz, 0, size);

    dim3 threads(8, 8, 8);
    dim3 blocks((NX + 7) / 8, (NY + 7) / 8, (NZ + 7) / 8);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int step = 0; step < STEPS; step++) 
    {
        update_h<<<blocks, threads>>>(Hx, Hy, Hz, Ex, Ey, Ez, NX, NY, NZ, DT, DX);
        cudaDeviceSynchronize();

        update_e<<<blocks, threads>>>(Ex, Ey, Ez, Hx, Hy, Hz, NX, NY, NZ, DT, DX);
        cudaDeviceSynchronize();

        inject_source<<<1, 1>>>(Ez, NX, NY, NZ, step * DT);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("FDTD completed in %.3f ms\n", ms);

    int idx = IDX(NX / 2, NY / 2, NZ / 2, NX, NY);
    printf("Final Ez at center: %.6f\n", Ez[idx]);

    cudaFree(Ex); cudaFree(Ey); cudaFree(Ez);
    cudaFree(Hx); cudaFree(Hy); cudaFree(Hz);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
