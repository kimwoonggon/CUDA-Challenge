# Day 14: Histogram Calculation Optimization with Privatization and Shared Memory

**Objective:**
- **Optimize Histogram Calculation:** Implement and optimize a histogram calculation on the GPU, focusing on reducing the overhead of atomic operations and global memory access.
- **Privatization Technique:** Utilize privatization by having each thread block compute a private histogram to minimize atomic contention in global memory.
- **Shared Memory for Accumulation:** Enhance performance further by using shared memory to accumulate private histograms within each block before reducing to a global histogram.
- **Performance Benchmarking:** Compare the execution times of different GPU histogram implementations (naive, privatized, and privatized with shared memory) against a CPU implementation to analyze optimization benefits.

**Key Learnings:**
- **GPU Histogram Challenges:** Understood the challenges of implementing histograms on GPUs, particularly the performance bottleneck caused by atomic operations when multiple threads increment the same histogram bins in global memory concurrently.
- **Atomic Operations Bottleneck:** Learned that frequent atomic operations in global memory can significantly degrade GPU performance due to serialization and high latency of global memory accesses.
- **Privatization for Reduced Atomicity:** Implemented privatization by assigning each thread block to calculate its own private histogram. This drastically reduces the number of atomic operations required in global memory during the initial histogram computation phase.
- **Shared Memory Histogram Accumulation:** Explored the use of shared memory to further optimize the privatized approach. By using shared memory to store and accumulate the private histograms within each block, we minimize global memory transactions and leverage the much faster on-chip shared memory.
- **Reduction Step:** Implemented a reduction step where the private histograms computed by each block are combined (using atomic addition in global memory, but now significantly less frequent) to produce the final global histogram.
- **Performance Improvement Analysis:** Observed a progressive performance improvement as optimizations were applied:
    - **Naive GPU:** Suffers from high overhead due to numerous atomic operations directly to global memory.
    - **Privatized GPU:**  Significant improvement by reducing global atomic operations, but still uses global memory for private histograms.
    - **Privatized GPU with Shared Memory:**  Achieves the best performance by using shared memory for private histogram accumulation, thus minimizing global memory access and atomic operations, resulting in a substantial speedup. But, still not faster than CPU.

**Shared Memory Optimized Histogram Kernel:**
```c
__global__ void histPvtSharedGPU(unsigned int *data, unsigned int *hist)
{
    unsigned int t_i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ unsigned int hist_s[TOTAL_BINS];
    for(unsigned bin = threadIdx.x; bin < BINS; bin += blockDim.x)
    {
        hist_s[bin] = 0u;
    }
    __syncthreads();

    if(t_i < N)
    {
        if(data[t_i] > 0 && data[t_i] <= 100)
        {
            atomicAdd(&hist_s[(blockIdx.x * BINS + (data_t_i] - 1)/20)], 1);
        }
    }
    __syncthreads();

    for(unsigned bin = threadIdx.x; bin < BINS; bin += blockDim.x)
    {
        if(hist_s[bin] > 0)
        {
            atomicAdd(&hist[bin], hist_s[bin]);
        }
    }
}
```

**Results:**
- **CPU execution time:** 0.1090 ms
- **GPU execution time:** 5.7651 ms
- **GPU execution time with privatization of input:** 0.7229 ms
- **GPU execution time with privatization of input in shared memory:** 0.1362 ms

**Histogram:**
- Bin 0 has 6458 numbers
- Bin 1 has 6514 numbers
- Bin 2 has 6513 numbers
- Bin 3 has 6557 numbers
- Bin 4 has 6726 numbers