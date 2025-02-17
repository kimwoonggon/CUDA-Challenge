# Day 13: Thread Coarsening for 3D Stencil Operation

**Objective:**
- **Thread Coarsening Optimization:** Implement thread coarsening in the 3D tiled stencil kernel to further optimize performance.
- **Reduced Thread Granularity:** Explore thread coarsening as a method to reduce thread overhead and improve efficiency, particularly in memory-bound operations.
- **Performance Benchmarking:** Evaluate and compare the performance of the thread-coarsened GPU stencil kernel against the naive GPU, tiled GPU, and CPU implementations from previous days.

**Key Learnings:**
- **Thread Coarsening Technique:** Learned and applied thread coarsening, a strategy to reduce the number of threads launched by having each thread do more work. In the context of stencil operations, this means assigning each thread to compute the stencil for multiple points in the depth dimension.
- **Implementation of Coarsened Tiling:**  Implemented thread coarsening within the tiled 3D stencil kernel (`threadCoarsedStencilGPU`). This kernel reduces the granularity of thread assignment along the depth dimension, allowing each thread to process a "coarser" chunk of data, specifically along the Z-axis.
- **Benefits of Thread Coarsening:** Observed that thread coarsening can further improve performance by reducing thread management overhead and potentially increasing instruction throughput per thread. By reducing the number of threads, we also reduce synchronization points and can better utilize shared memory for larger tiles in the XY planes within each coarser thread block.
- **Enhanced Performance over Tiled Approach:** Achieved additional performance gains compared to the tiled GPU implementation from Day 12, demonstrating that thread coarsening is an effective optimization, especially when combined with tiling for stencil operations.

**Thread Coarsened 3D Stencil Kernel:**
```c
__global__ void threadCoarsedStencilGPU(float *input, float *output, float *val, int N)
{
    int depth = blockIdx.z * OUT_TILE_COARSED;
    int row = blockIdx.y * OUT_TILE_COARSED + threadIdx.y - ORDER;
    int col = blockIdx.x * OUT_TILE_COARSED + threadIdx.x - ORDER;

    __shared__ float prev[IN_TILE_COARSED][IN_TILE_COARSED];
    __shared__ float curr[IN_TILE_COARSED][IN_TILE_COARSED];
    __shared__ float next[IN_TILE_COARSED][IN_TILE_COARSED];

    if(depth >= 1 && depth < N - 1 && row >= 0 && row < N && col >= 0 && col < N)
    {
        prev[threadIdx.y][threadIdx.x] = input[depth * N * N + row * N + col];
    }

    if(depth >= 0 && depth < N && row >= 0 && row < N && col >= 0 && col < N)
    {
        curr[threadIdx.y][threadIdx.x] = input[depth * N * N + row * N + col];
    }

    for (int n_layer = depth; n_layer < depth + OUT_TILE_COARSED; n_layer++)
    {
        if(n_layer >= 1 && n_layer < N - 1 && row >= 0 && row < N && col >= 0 && col < N)
        {
            next[threadIdx.y][threadIdx.x] = input[(n_layer + 1) * N * N + row * N + col];
        }
        __syncthreads();

        if(n_layer >=1 && depth < N - 1 && row >=1 && row < N - 1 && col >=1 && col < N - 1)
        {
            if(threadIdx.y >=1 && threadIdx.y < IN_TILE_DIM - 1 && threadIdx.x >=1 && threadIdx.x < IN_TILE_DIM - 1)
            {
                output[n_layer * N * N + row * N + col] = val[0] * curr[threadIdx.y][threadIdx.x] +
                                                val[1] * curr[threadIdx.y][threadIdx.x - 1] +
                                                val[2] * curr[threadIdx.y][threadIdx.x + 1] +
                                                val[3] * curr[threadIdx.y - 1][threadIdx.x] +
                                                val[4] * curr[threadIdx.y + 1][threadIdx.x] +
                                                val[5] * prev[threadIdx.y][threadIdx.x] +
                                                val[6] * next[threadIdx.y][threadIdx.x];
            }
        }
        __syncthreads();
        prev[threadIdx.y][threadIdx.x] = curr[threadIdx.y][threadIdx.x];
        curr[threadIdx.y][threadIdx.x] = next[threadIdx.y][threadIdx.x];
    }
}
```

**Results (for N = 256):**
   - **GPU Execution time for tiled and thread coarsed approach:** 7.199744 ms
   - **GPU Execution time for tiled approach:** 18.025248 ms
   - **GPU Execution time:** 36.248322 ms
   - **CPU Execution time:** 201.901993 ms
