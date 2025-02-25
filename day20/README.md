# Day 20: Tiled GPU Merge - Enhancing Performance with Shared Memory Tiling

**Objective:**
- **Implement Tiled GPU Merge:** Implement a tiled version of the GPU merge operation from Day 19 (`mergeGPU`). This day focuses on using shared memory tiling to further optimize the parallel merge process, especially for scenarios where data locality and shared memory utilization can be improved.
- **Shared Memory Tiling:** Explore how to divide the merge problem into smaller tiles that fit into shared memory and process these tiles efficiently to reduce global memory access overhead.
- **Performance Comparison:** Compare the performance of the tiled GPU merge (`mergeTiledGPU`) against the previous GPU merge implementation (`mergeGPU`) and a CPU-based merge, analyzing the impact of tiling on performance, particularly for smaller datasets where CPU performance might be competitive.

**Key Learnings:**
- **Tiled Algorithms for GPUs:** Learned about the concept of tiling in GPU algorithms. Tiling involves dividing a large problem into smaller blocks (tiles) that can be processed more efficiently, often by fitting them into shared memory. Tiling can improve data locality and reduce redundant global memory accesses.
- **Tiled Merge Implementation:** Implemented a tiled GPU merge kernel (`mergeTiledGPU`). This kernel divides the overall merge task into tiles. For each tile, it loads relevant sub-arrays from global memory into shared memory, performs the merge operation within shared memory using a similar co-ranking and CPU merge sub-routine approach as in `mergeGPU`, and then writes the merged tile back to global memory.
- **Shared Memory for Tiling Benefits:** Understood that shared memory is crucial for realizing the benefits of tiling. By loading tiles into shared memory, the kernel can perform the core merge operations on fast on-chip memory, significantly reducing the need to access slower global memory for each element comparison and merge step within a tile.
- **Performance Trade-offs for Small Datasets:** Observed that for smaller datasets, while tiling can improve GPU performance compared to a non-tiled GPU version, the overhead of managing tiles, transferring data to shared memory, and kernel launch might make CPU-based merge operations competitive or even faster. The benefits of tiling and GPU acceleration become more pronounced as problem sizes increase and global memory bandwidth becomes a more significant bottleneck.

**Tiled GPU Merge Kernel (`mergeTiledGPU`):**
```c
__global__ void mergeTiledGPU(int *a, int *b, int *c, int m, int n)
{
    int kBlock = blockIdx.x * ELEMENTS_PER_BLOCK;
    int kNextBlock;

    if(blockIdx.x < gridDim.x - 1)
    {
        kNextBlock = kBlock + ELEMENTS_PER_BLOCK;
    }
    else
    {
        kNextBlock = m + n;
    }

    __shared__ int iBlock;
    __shared__ int iNextBlock;

    if(threadIdx.x == 0)
    {
        iBlock = coRank(a, b, m, n, kBlock);
    }
    if(threadIdx.x == blockDim.x - 1)
    {
        iNextBlock = coRank(a, b, m, n, kNextBlock);
    }
    __syncthreads();

    int jBlock = kBlock - iBlock;
    int jNextBlock = kNextBlock - iNextBlock;

    __shared__ int a_s[ELEMENTS_PER_BLOCK];
    int mBlock = iNextBlock - iBlock;

    for(int x = threadIdx.x; x < mBlock; x += blockDim.x)
    {
        a_s[x] = a[iBlock + x];
    }

    __shared__ int b_s[ELEMENTS_PER_BLOCK];
    int nBlock = jNextBlock - jBlock;

    for(int y = threadIdx.x; y < nBlock; y += blockDim.x)
    {
        b_s[y] = b[jBlock + y];
    }
    __syncthreads();

    __shared__ int c_s[ELEMENTS_PER_BLOCK];
    int k = threadIdx.x * ELEMENTS;

    if(k < mBlock + nBlock)
    {
        int i = coRank(a_s, b_s, mBlock, nBlock, k);
        int j = k - i;
        int kNext;

        if((k + ELEMENTS) < (mBlock + nBlock))
        {
            kNext = k + ELEMENTS;
        }
        else
        {
            kNext = mBlock + nBlock;
        }

        int iNext = coRank(a_s, b_s, mBlock, nBlock, kNext);
        int jNext = kNext - iNext;
        mergeCPU(&a_s[i], &b_s[j], &c_s[k], iNext - i, jNext - j);
    }
    __syncthreads();

    for(int z = threadIdx.x; z < mBlock + nBlock; z += blockDim.x)
    {
        c[kBlock + z] = c_s[z];
    }
}
```

**Results (for N=2^10, M=2^11):**
- **CPU execution time:** 0.018 ms
- **GPU execution time:** 0.203 ms
- **GPU execution time (Tiled):** 0.069 ms

**Results Analysis:**
- For these smaller array sizes (N=2^10, M=2^11), the CPU merge is actually the fastest. This illustrates that for small problems, the overhead associated with GPU kernel launch, data transfer (even with managed memory), and thread management can outweigh the benefits of GPU parallelism.
- The naive GPU merge (`mergeGPU`) is significantly slower than the CPU in this scenario. This is likely due to the overheads mentioned above and potentially less-than-optimal memory access patterns for smaller problem sizes on the GPU.
- The tiled GPU merge (`mergeTiledGPU`) shows a considerable performance improvement over the naive mergeGPU, even for these small arrays, and comes closer to the CPU performance. Tiling helps improve data locality and shared memory utilization, which becomes beneficial even for moderately sized problems.
- These results emphasize that the choice between CPU and GPU and the optimal GPU algorithm strategy depends heavily on the scale of the problem. For very small datasets, CPU execution might be preferable due to lower overhead. As dataset sizes grow, tiled and optimized GPU implementations become increasingly advantageous to leverage the GPU's parallel processing power and memory bandwidth effectively.