# Day 12: Optimized 3D Stencil Operation with Tiling

**Objective:**
- **Tiled 3D Stencil Kernel:** Optimize the 3D stencil operation from Day 11 by implementing tiling.
- **Shared Memory Optimization:**  Enhance data locality and reduce global memory access by utilizing shared memory in the 3D stencil kernel.
- **Performance Comparison:**  Compare the performance of the tiled 3D stencil GPU kernel against the naive GPU and CPU implementations from Day 11.

**Key Learnings:**
- **Tiled 3D Stencil Implementation:** Learned to apply tiling techniques to 3D stencil operations. This involves dividing the 3D data into smaller tiles and processing each tile in parallel, which improves cache utilization and reduces global memory traffic.
- **Shared Memory for 3D Tiling:**  Effectively used shared memory to create local 3D tiles within each thread block. This allows threads within a block to quickly access neighboring data points required for the stencil calculation, stored in fast shared memory instead of slower global memory.
- **Performance Optimization through Tiling:**  Observed a significant performance improvement by using the tiled approach compared to the naive GPU implementation from Day 11, as well as a much larger speedup compared to the CPU. Tiling and shared memory are crucial for optimizing memory-bound computations like stencil operations.

**Tiled 3D Stencil Kernel:**
```c
__global__ void tiledStencilGPU(float *input, float *output, float *val, int N)
{
    int depth = blockIdx.z * OUT_TILE_DIM + threadIdx.z - ORDER;
    int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - ORDER;
    int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - ORDER;

    __shared__ float inp_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];

    if(depth >=1 && depth < N && row >=1 && row < N && col >=1 && col < N)
    {
        inp_s[threadIdx.z][threadIdx.y][threadIdx.x] = input[depth * N * N + row * N + col];
    }
    __syncthreads();

    if(depth >=1 && depth < N - 1 && row >=1 && row < N - 1 && col >=1 && col < N - 1)
    {
        if(threadIdx.z >=1 && threadIdx.z < IN_TILE_DIM - 1 && threadIdx.y >=1 && threadIdx.y < IN_TILE_DIM - 1 && threadIdx.x >=1 && threadIdx.x < IN_TILE_DIM - 1)
        {
            output[depth * N * N + row * N + col] = val[0] * inp_s[threadIdx.z][threadIdx.y][threadIdx.x] +
                                                val[1] * inp_s[threadIdx.z][threadIdx.y][threadIdx.x - 1] +
                                                val[2] * inp_s[threadIdx.z][threadIdx.y][threadIdx.x + 1] +
                                                val[3] * inp_s[threadIdx.z][threadIdx.y - 1][threadIdx.x] +
                                                val[4] * inp_s[threadIdx.z][threadIdx.y + 1][threadIdx.x] +
                                                val[5] * inp_s[threadIdx.z - 1][threadIdx.y][threadIdx.x] +
                                                val[6] * inp_s[threadIdx.z + 1][threadIdx.y][threadIdx.x];
        }
    }
}
```
**Results (for N = 256):**
   - **GPU Execution time for tiled approach:** 18.025248 ms
   - **GPU Execution time:** 36.248322 ms
   - **CPU Execution time:** 201.901993 ms