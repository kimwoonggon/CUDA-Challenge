# Day 10: Tiled 2D Convolution for Enhanced Performance

**Objective:**
- **Tiled 2D Convolution Kernel:** Optimize the 2D convolution kernel from Day 8 using tiling to enhance performance.
- **Shared Memory Utilization:** Implement tiling by leveraging shared memory to reduce global memory accesses.
- **Constant Memory for Filter:** Store the convolution filter in constant memory for faster access within the kernel.
- **Performance Benchmarking:** Compare the execution time of the tiled GPU convolution against a CPU implementation to quantify performance gains.

**Key Learnings:**
- **Tiled Convolution:** Learned and implemented tiled convolution, a technique that breaks down the input data into smaller blocks (tiles) to improve data locality and cache utilization.
- **Shared Memory Optimization:** Utilized shared memory to create on-chip tiles of the input matrix, significantly reducing redundant fetches from global memory for each pixel's neighborhood.
- **Constant Memory Usage:**  Explored the use of constant memory to store the convolution filter. Constant memory is cached and offers faster read access compared to global memory, especially when the same filter is repeatedly accessed by all threads.
- **Performance Improvement:** Achieved substantial performance improvement over the naive convolution by minimizing global memory bandwidth usage through tiling and shared memory.

**Tiled 2D Convolution Kernel:**
```c
__constant__ float filter[2 * RADIUS + 1][2 * RADIUS + 1];

__global__ void conv2D(float *input, float *output, int N, int M) 
{
    int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - RADIUS;
    int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - RADIUS;

    __shared__ float N_shared[IN_TILE_DIM][IN_TILE_DIM];

    if (row >= 0 && row < N && col >= 0 && col < M)
        N_shared[threadIdx.y][threadIdx.x] = input[row * M + col];
    else
        N_shared[threadIdx.y][threadIdx.x] = 0.0f;
    __syncthreads();

    int tile_row = threadIdx.y - RADIUS;
    int tile_col = threadIdx.x - RADIUS;

    if (tile_row >= 0 && tile_row < OUT_TILE_DIM && tile_col >= 0 && tile_col < OUT_TILE_DIM) 
    {
        int out_row = blockIdx.y * OUT_TILE_DIM + tile_row;
        int out_col = blockIdx.x * OUT_TILE_DIM + tile_col;
        if (out_row < N && out_col < M) 
        {
            float val = 0.0f;
            int f_size = 2 * RADIUS + 1;
            for (int row_f = 0; row_f < f_size; row_f++) 
            {
                for (int col_f = 0; col_f < f_size; col_f++) 
                {
                    val += filter[row_f][col_f] * 
                           N_shared[threadIdx.y + row_f - RADIUS][threadIdx.x + col_f - RADIUS];
                }
            }
            output[out_row * M + out_col] = val;
        }
    }
}
```

**Results:**
  - **Tiled Convolution Execution Time on GPU:** 0.07 ms
  - **Convolution Execution Time on CPU:** 0.66 ms