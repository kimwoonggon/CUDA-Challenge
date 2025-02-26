# Day 21: Radix Sort - Bitwise Sorting with GPU Acceleration using Brent-Kung Scan

**Objective:**
- **Implement Bitwise Radix Sort on GPU:** Implement a single-pass (one bit position) of the radix sort algorithm on the GPU. This day focuses on sorting based on a single bit and leveraging GPU parallelism for this bitwise sorting step.
- **Utilize Brent-Kung Prefix Sum:** Integrate the Brent-Kung prefix sum algorithm (developed in Day 18) as a key subroutine within the radix sort kernel to efficiently calculate prefix sums necessary for determining element output positions.
- **Performance Benchmarking:** Compare the execution time of this GPU-accelerated bitwise radix sort step against a CPU implementation to evaluate the performance benefits of the GPU approach.

**Key Learnings:**
- **Radix Sort Fundamentals:** Learned the basics of radix sort, a non-comparative integer sorting algorithm that sorts elements by processing individual digits (or bits) from least significant to most significant. This day focuses on the core bitwise sorting step.
- **Bitwise Radix Sort:** Understood how radix sort can be implemented bit by bit. For each bit position, elements are grouped based on whether that bit is 0 or 1. This bitwise sorting is a fundamental component of the full radix sort algorithm.
- **Prefix Sum for Radix Sort:**  Discovered the crucial role of prefix sum (scan) in radix sort. Prefix sums are used to determine the correct output positions for elements after they have been partitioned based on the current bit being sorted.  Specifically, prefix sums help determine the starting positions for 0s and 1s in the sorted output array.
- **Brent-Kung Algorithm Re-use:**  Successfully reused the Brent-Kung prefix sum algorithm kernel developed on Day 18 as a building block for the radix sort implementation. This demonstrates the modularity and reusability of optimized GPU kernels. By using Brent-Kung, we achieve efficient parallel prefix sum computation within the radix sort step.
- **GPU Acceleration of Sorting Subroutines:** Demonstrated how even a single step of a complex algorithm like radix sort can be accelerated on the GPU by parallelizing key subroutines like prefix sum and bit partitioning.

**GPU Radix Sort Kernel (`radixSort`):**
```c
__global__ void radixSort(int *inp, int *out, int *bits, int iter)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N)
    {
        int key = inp[i];
        int bit = (key >> iter) & 1;
        bits[i] = bit;
    }
    __syncthreads();

    brentGPU(bits);
    __syncthreads();

    if(i < N)
    {
        int key = inp[i];
        int bit = (key >> iter) & 1;
        int oneBefore = bits[i];

        int lastBit = (inp[N-1] >> iter) & 1;
        int oneTotal = bits[N-1] + lastBit;

        int dst;
        if(bit == 0)
        {
            dst = i - oneBefore;
        }
        else
        {
            dst = (N - oneTotal) + oneBefore;
        }

        out[dst] = key;
    }
}
```

**Results (for N=2^24):**
- **CPU execution time:** 215.93 ms
- **GPU execution time:** 18.81 ms