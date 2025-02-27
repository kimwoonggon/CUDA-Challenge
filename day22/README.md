# Day 22: Radix Sort Optimization - Shared Memory for Enhanced Bitwise Sorting

**Objective:**
- **Optimize Bitwise Radix Sort with Shared Memory:** Further optimize the GPU-based bitwise radix sort step from Day 21 by incorporating shared memory to reduce global memory accesses and improve data locality.
- **Shared Memory for Bit Extraction and Prefix Sum Context:** Utilize shared memory to stage data for bit extraction and to provide a local, faster context for the Brent-Kung prefix sum algorithm within the radix sort kernel.
- **Compare Performance with and without Shared Memory Optimization:** Benchmark and compare the performance of the original `radixSort` kernel (Day 21) against the new `radixSortShared` kernel that utilizes shared memory, and against the CPU implementation.

**Key Learnings:**
- **Shared Memory Optimization in Radix Sort:** Learned how shared memory can be effectively used to optimize a bitwise radix sort step. By loading relevant data into shared memory, we reduce the latency and bandwidth demands on global memory during the bit extraction and prefix sum stages of the algorithm.
- **Improved Data Locality:** Realized that using shared memory enhances data locality. Threads within a block can access the shared memory region much faster than global memory, which is particularly beneficial for algorithms like radix sort that involve repeated accesses to the input data and intermediate bit arrays.
- **Reduced Global Memory Traffic:** Observed that the `radixSortShared` kernel reduces global memory traffic. By performing bit extractions and potentially providing a faster input to the Brent-Kung prefix sum within shared memory, the number of global memory reads and writes is lessened.
- **Performance Gains from Shared Memory:**  Benchmarked the performance and observed that the shared memory optimized `radixSortShared` kernel outperforms the original `radixSort` kernel, demonstrating the effectiveness of shared memory in improving the performance of the bitwise radix sort step. This optimization builds upon the Brent-Kung prefix sum from previous days, showing how different optimization techniques can be layered for cumulative gains.

**GPU Radix Sort Kernels:**

**Optimized Radix Sort Kernel:**
```c
__global__ void radixSortShared(int *inp, int *out, int *bits, int iter) 
{
    int chunk = 2 * blockIdx.x * blockDim.x;
    int tid   = threadIdx.x;
    int index1 = chunk + tid;
    int index2 = chunk + tid + blockDim.x;

    __shared__ int s_keys[2*THREADS];
    __shared__ int s_bits[2*THREADS];

    int key1 = (index1 < N) ? inp[index1] : 0;
    int key2 = (index2 < N) ? inp[index2] : 0;
    s_keys[tid] = key1;
    s_keys[tid + blockDim.x]  = key2;

    int bit1 = (index1 < N) ? ((key1 >> iter) & 1) : 0;
    int bit2 = (index2 < N) ? ((key2 >> iter) & 1) : 0;
    s_bits[tid] = bit1;
    s_bits[tid + blockDim.x] = bit2;
    __syncthreads();

    if(index1 < N)
    {
        bits[index1] = s_bits[tid];
    }
    if(index2 < N)
    {
        bits[index2] = s_bits[tid + blockDim.x];
    }
    __syncthreads();

    brentGPU(bits);
    __syncthreads();

    int scanned_bit1 = (index1 < N) ? bits[index1] : 0;
    int scanned_bit2 = (index2 < N) ? bits[index2] : 0;

    __shared__ int block_total;
    if(tid == 0) 
    {
        int lastIndex = chunk + 2 * blockDim.x - 1;
        int lastGlobal = (lastIndex < N) ? bits[lastIndex] : 0;
        int lastBit = (lastIndex < N) ? ((inp[lastIndex] >> iter) & 1) : 0;
        block_total = lastGlobal + lastBit;
    }
    __syncthreads();

    if(index1 < N) 
    {
        int dst;
        if (bit1 == 0)
        {
            dst = index1 - scanned_bit1;
        }
        else
        {
            dst = (N - block_total) + scanned_bit1;
        }
        out[dst] = key1;
    }
    if(index2 < N) 
    {
        int dst;
        if (bit2 == 0)
        {
            dst = index2 - scanned_bit2;
        }
        else
        {
            dst = (N - block_total) + scanned_bit2;
        }
        out[dst] = key2;
    }
}
```

**Results (for N=2^24):**
- **CPU execution time:** 215.93 ms
- **GPU execution time:** 18.81 ms
- **GPU execution time (Shared):** 16.21 ms