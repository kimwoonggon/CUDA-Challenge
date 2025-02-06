#include <stdio.h>

#define N  10000

__global__ void matrixMulGPU( int * a, int * b, int * c )
{
  int threadX = blockIdx.x * blockDim.x + threadIdx.x;
  int threadY = blockIdx.y * blockDim.y + threadIdx.y;

  int strideX = gridDim.x * blockDim.x;
  int strideY = gridDim.y * blockDim.y;

  int val;

  for (int i = threadX; i < N; i += strideX)
  {
    for (int j = threadY; j < N; j+= strideY)
    {
      val = 0;
      for ( int k = 0; k < N; ++k )
        val += a[i * N + k] * b[k * N + j];
      c[i * N + j] = val;
    }
  }
}

void matrixMulCPU( int * a, int * b, int * c )
{
  int val = 0;

  for( int row = 0; row < N; ++row )
    for( int col = 0; col < N; ++col )
    {
      val = 0;
      for ( int k = 0; k < N; ++k )
        val += a[row * N + k] * b[k * N + col];
      c[row * N + col] = val;
    }
}

int main()
{
  int *a, *b, *c_cpu, *c_gpu;

  int size = N * N * sizeof (int);

  cudaMallocManaged (&a, size);
  cudaMallocManaged (&b, size);
  cudaMallocManaged (&c_cpu, size);
  cudaMallocManaged (&c_gpu, size);

  for( int row = 0; row < N; ++row )
    for( int col = 0; col < N; ++col )
    {
      a[row*N + col] = row;
      b[row*N + col] = col+2;
      c_cpu[row*N + col] = 0;
      c_gpu[row*N + col] = 0;
    }

  dim3 threads_per_block(16, 16);
  dim3 number_of_blocks(((N + threads_per_block.x - 1)/threads_per_block.x), ((N + threads_per_block.y - 1)/threads_per_block.y));

  matrixMulGPU <<< number_of_blocks, threads_per_block >>> ( a, b, c_gpu );

  cudaError_t err1 = cudaDeviceSynchronize();
  if (err1 != cudaSuccess)
  {
    printf("Asynchronous Error: %s\n", cudaGetErrorString(err1));
  }

  cudaError_t err2 = cudaGetLastError();
  if (err2 != cudaSuccess)
  {
    printf("Synchronous Error: %s\n", cudaGetErrorString(err2));
  }

  matrixMulCPU( a, b, c_cpu );

  bool error = false;
  for( int row = 0; row < N && !error; ++row )
    for( int col = 0; col < N && !error; ++col )
      if (c_cpu[row * N + col] != c_gpu[row * N + col])
      {
        printf("FOUND ERROR at c[%d][%d]\n", row, col);
        error = true;
        break;
      }
  if (!error)
    printf("Success!\n");

  cudaFree(a); cudaFree(b);
  cudaFree( c_cpu ); cudaFree( c_gpu );
}