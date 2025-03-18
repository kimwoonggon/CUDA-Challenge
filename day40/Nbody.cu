#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <math.h>
#include <time.h>
 
#define M_PI 3.14159265358979323846
#define THREADS 1024
#define N (1<<16)
 
#define G 6.67408e-11f

__global__ void nbody(float *x,  float *y,  float *z, float *m, float *vx, float *vy, float *vz, float dt)
{
    extern __shared__ float sdata[];
    int tileSize = blockDim.x * 4;
    float *tileA = sdata;
    float *tileB = sdata + tileSize;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N)
    {
        float px = x[i];
        float py = y[i];
        float pz = z[i];
    
        float vx_i = vx[i];
        float vy_i = vy[i];
        float vz_i = vz[i];
    
        float ax_old = 0.f;
        float ay_old = 0.f;
        float az_old = 0.f;
    
        int numTiles = (N + blockDim.x - 1) / blockDim.x;
        int idxGlobal = threadIdx.x;
        if (idxGlobal < blockDim.x) 
        {
            int globalTileIdx = idxGlobal;
            if (globalTileIdx < N) 
            {
                tileB[idxGlobal * 4 + 0] = x[globalTileIdx];
                tileB[idxGlobal * 4 + 1] = y[globalTileIdx];
                tileB[idxGlobal * 4 + 2] = z[globalTileIdx];
                tileB[idxGlobal * 4 + 3] = m[globalTileIdx];
            }
        }
        __syncthreads();

        for (int tile = 0; tile < numTiles; tile++) 
        {
            float *oldTile = (tile % 2 == 0) ? tileB : tileA;
            float *newTile = (tile % 2 == 0) ? tileA : tileB;
    
            for (int j = 0; j < blockDim.x; j++) 
            {
                int jGlobal = tile * blockDim.x + j;
                if (jGlobal >= N || jGlobal == i) 
                {
                    continue;
                }
                float ox = oldTile[j * 4 + 0];
                float oy = oldTile[j * 4 + 1];
                float oz = oldTile[j * 4 + 2];
                float om = oldTile[j * 4 + 3];
    
                float dx = ox - px;
                float dy = oy - py;
                float dz = oz - pz;
                float distSqr = dx*dx + dy*dy + dz*dz + 1e-10f;
                float invDist = rsqrtf(distSqr);
                float invDist3 = invDist * invDist * invDist;
                float f = om * invDist3;
    
                ax_old += dx * f;
                ay_old += dy * f;
                az_old += dz * f;
            }
            __syncthreads();
    
            if (tile + 1 < numTiles) 
            {
                int nextIdx = (tile + 1) * blockDim.x + threadIdx.x;
                if (nextIdx < N) 
                {
                    newTile[threadIdx.x * 4 + 0] = x[nextIdx];
                    newTile[threadIdx.x * 4 + 1] = y[nextIdx];
                    newTile[threadIdx.x * 4 + 2] = z[nextIdx];
                    newTile[threadIdx.x * 4 + 3] = m[nextIdx];
                }
            }
            __syncthreads();
        }

        ax_old *= G;
        ay_old *= G;
        az_old *= G;
    
        px += vx_i * dt + 0.5f * ax_old * dt * dt;
        py += vy_i * dt + 0.5f * ay_old * dt * dt;
        pz += vz_i * dt + 0.5f * az_old * dt * dt;
    
        if (threadIdx.x < blockDim.x) 
        {
            int globalTileIdx = threadIdx.x;
            if (globalTileIdx < N) 
            {
                tileB[threadIdx.x * 4 + 0] = x[globalTileIdx];
                tileB[threadIdx.x * 4 + 1] = y[globalTileIdx];
                tileB[threadIdx.x * 4 + 2] = z[globalTileIdx];
                tileB[threadIdx.x * 4 + 3] = m[globalTileIdx];
            }
        }
        __syncthreads();
    
        float ax_new = 0.f;
        float ay_new = 0.f;
        float az_new = 0.f;
    
        for (int tile = 0; tile < numTiles; tile++) 
        {
            float *oldTile = (tile % 2 == 0) ? tileB : tileA;
            float *newTile = (tile % 2 == 0) ? tileA : tileB;
    
            for (int j = 0; j < blockDim.x; j++) 
            {
                int jGlobal = tile * blockDim.x + j;
                if (jGlobal >= N || jGlobal == i) 
                {
                    continue;
                }
                float ox = oldTile[j * 4 + 0];
                float oy = oldTile[j * 4 + 1];
                float oz = oldTile[j * 4 + 2];
                float om = oldTile[j * 4 + 3];
    
                float dx = ox - px;
                float dy = oy - py;
                float dz = oz - pz;
                float distSqr = dx*dx + dy*dy + dz*dz + 1e-10f;
                float invDist = rsqrtf(distSqr);
                float invDist3 = invDist * invDist * invDist;
                float f = om * invDist3;
    
                ax_new += dx * f;
                ay_new += dy * f;
                az_new += dz * f;
            }
            __syncthreads();

            if (tile + 1 < numTiles) 
            {
                int nextIdx = (tile + 1) * blockDim.x + threadIdx.x;
                if (nextIdx < N) 
                {
                    newTile[threadIdx.x * 4 + 0] = x[nextIdx];
                    newTile[threadIdx.x * 4 + 1] = y[nextIdx];
                    newTile[threadIdx.x * 4 + 2] = z[nextIdx];
                    newTile[threadIdx.x * 4 + 3] = m[nextIdx];
                }
            }
            __syncthreads();
        }
    
        ax_new *= G;
        ay_new *= G;
        az_new *= G;
    
        vx_i += 0.5f * (ax_old + ax_new) * dt;
        vy_i += 0.5f * (ay_old + ay_new) * dt;
        vz_i += 0.5f * (az_old + az_new) * dt;
    
        x[i]  = px;
        y[i]  = py;
        z[i]  = pz;
        vx[i] = vx_i;
        vy[i] = vy_i;
        vz[i] = vz_i;
    }
}
 
void init(float *x, float *y, float *z, float *m, float *vx, float *vy, float *vz)
{
    srand((unsigned)time(NULL));
    for (int i = 0; i < N; i++) 
    {

        x[i] = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
        y[i] = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
        z[i] = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;

        m[i] = 1.0f + 999.0f * ((float)rand() / RAND_MAX);

        vx[i] = 0.001f * (2.0f * ((float)rand() / RAND_MAX) - 1.0f);
        vy[i] = 0.001f * (2.0f * ((float)rand() / RAND_MAX) - 1.0f);
        vz[i] = 0.001f * (2.0f * ((float)rand() / RAND_MAX) - 1.0f);
    }

    if (N > 5) 
    {
        m[0] = 100000.0f;
        x[0] = y[0] = z[0] = 0.0f;
        vx[0] = vy[0] = vz[0] = 0.0f;

        for (int i = 1; i < 5; i++) 
        {
            m[i] = 10000.0f + 20000.0f * ((float)rand() / RAND_MAX);
            float angle = 2.0f * (float)M_PI * i / 4.0f;
            float radius = 0.5f;
            x[i] = radius * cosf(angle);
            y[i] = radius * sinf(angle);
            z[i] = 0.1f * ((float)rand() / RAND_MAX - 0.5f);

            float speed = 0.005f;
            vx[i] = -speed * sinf(angle);
            vy[i] =  speed * cosf(angle);
            vz[i] = 0.0f;
        }
    }
}

int main()
{
    size_t size = N * sizeof(float);

    float *x, *y, *z, *m;
    float *vx, *vy, *vz;

    cudaMallocManaged(&x, size);
    cudaMallocManaged(&y, size);
    cudaMallocManaged(&z, size);
    cudaMallocManaged(&m, size);
    cudaMallocManaged(&vx, size);
    cudaMallocManaged(&vy, size);
    cudaMallocManaged(&vz, size);

    init(x, y, z, m, vx, vy, vz);

    printf("\nInitial sample:\n");
    for (int i = 0; i < 5; i++) 
    {
        printf("Body %d: pos=(%.4f,%.4f,%.4f), vel=(%.6f,%.6f,%.6f), mass=%.2f\n", i, x[i], y[i], z[i], vx[i], vy[i], vz[i], m[i]);
    }

    float dt = 0.01f;
    int steps = 100;

    size_t sharedMem = 2 * THREADS * 4 * sizeof(float);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int step = 0; step < steps; step++) 
    {
        nbody<<<(N + THREADS - 1) / THREADS, THREADS, sharedMem>>>(x, y, z, m, vx, vy, vz, dt);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("\nFinished %d steps in %.3f ms (%.3f ms/step)\n", steps, ms, ms/steps);

    printf("\nFinal sample:\n");
    for (int i = 0; i < 5; i++) 
    {
        printf("Body %d: pos=(%.4f,%.4f,%.4f), vel=(%.6f,%.6f,%.6f), mass=%.2f\n", i, x[i], y[i], z[i], vx[i], vy[i], vz[i], m[i]);
    }

    cudaFree(x);
    cudaFree(y);
    cudaFree(z);
    cudaFree(m);
    cudaFree(vx);
    cudaFree(vy);
    cudaFree(vz);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
 }
 