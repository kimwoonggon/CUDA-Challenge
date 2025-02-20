# Day 15: Thread Coarsening Strategies for Histogram Optimization and Scalability

**Objective:**
- **Explore Thread Coarsening:** Investigate thread coarsening techniques to further optimize the shared memory based histogram calculation from Day 14, particularly for very large datasets.
- **Contiguous vs. Strided Coarsening:** Implement and compare two thread coarsening approaches: contiguous partitioning and strided thread reuse, to evaluate their impact on performance.
- **Scalability Analysis:** Analyze the performance of CPU and different GPU histogram implementations as the data size (N) increases, identifying scenarios where GPU acceleration becomes significantly advantageous.

**Key Learnings:**
- **Thread Coarsening for Large Datasets:** Extended the optimization strategies by applying thread coarsening to handle very large datasets more efficiently.  Thread coarsening aims to reduce thread launch overhead and improve resource utilization when processing massive arrays.
- **Contiguous Partitioning Coarsening:** Implemented contiguous partitioning where each thread is responsible for processing a contiguous block of input data elements. This can improve data locality for each thread within its assigned block.
- **Strided Thread Re-use Coarsening:** Implemented strided thread re-use where threads process data elements with a strided pattern across the entire input array. This approach can potentially lead to better load balancing and memory access patterns in certain scenarios.
- **CPU vs. GPU Performance Scaling:** Realized that for small datasets, the overhead of GPU kernel launch and data transfer can outweigh the benefits of parallelism, making CPU execution faster. However, as the data size (N) increases significantly, optimized GPU implementations, especially those with thread coarsening and shared memory, demonstrate superior scalability and performance compared to CPU.
- **Strided Coarsening Performance:** Observed that in this particular histogram implementation, the strided thread re-use approach to coarsening outperformed contiguous partitioning. This may be due to more balanced workload distribution across threads or more favorable memory access patterns for strided access in this specific workload.
- **Importance of Algorithm Choice based on Data Size:** Emphasized that choosing the right algorithm and optimization strategy depends heavily on the problem size. For smaller problems, simple CPU implementations might suffice, while for large-scale data processing, optimized GPU kernels with techniques like shared memory and thread coarsening are essential to achieve significant speedups.

**Strided Thread Re-use Coarsened Histogram Kernel:**
```c
__global__ void histPvtSharedStridedGPU(unsigned int *data, unsigned int *hist)
{
    unsigned int t_i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ unsigned int hist_s[BINS];
    if (threadIdx.x < BINS)
    {
        hist_s[threadIdx.x] = 0u;
    }
    __syncthreads();

    for(unsigned int thread = t_i; thread < N; thread += blockDim.x * gridDim.x)
    {
        if(data[thread] > 0 && data[thread] <= 100)
        {
            atomicAdd(&hist_s[(data[thread] - 1)/20], 1);
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

**Results (N = 2^25):**
- **CPU execution time:** 96.4690 ms
- **GPU execution time (Naive):** 5269.8564 ms
- **GPU execution time (Privatization):** 266.9722 ms
- **GPU execution time (Privatization + Shared Memory):** 70.9786 ms
- **GPU execution time (Privatization + Shared Memory + Contiguous Coarsening):** 23.6373 ms
- **GPU execution time (Privatization + Shared Memory + Strided Coarsening):** 11.4761 ms

**Histogram:**
- Bin 0 has 6709317 numbers
- Bin 1 has 6709132 numbers
- Bin 2 has 6709941 numbers
- Bin 3 has 6713852 numbers
- Bin 4 has 6712190 numbers