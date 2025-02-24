# Day 19: GPU Accelerated Merge Operation - Hybrid CPU/GPU Approach

**Objective:**
- **Implement GPU Merge:** Implement a parallel merge operation on the GPU to efficiently merge two pre-sorted arrays into a single sorted array.
- **Co-ranking for Parallelism:** Explore the concept of co-ranking to partition the merge task into parallel subproblems suitable for GPU execution.
- **Hybrid CPU/GPU Strategy:** Utilize a hybrid approach where the GPU kernel performs the parallel merge at a high level, but delegates the actual merging of smaller sub-arrays to a CPU merge function, demonstrating a combined approach to leverage both CPU and GPU strengths.
- **Performance Benchmarking:** Compare the performance of the GPU merge implementation against a purely CPU-based merge to assess the benefits of GPU acceleration for this operation.

**Key Learnings:**
- **Parallel Merge Concept:** Learned the fundamental concept of parallel merge, which aims to divide the merging of two large sorted arrays into smaller, independent merge tasks that can be executed concurrently on a GPU.
- **Co-ranking Technique:** Understood the co-ranking technique. Co-ranking, in this context, is used to determine how to partition the two input arrays (`a` and `b`) for parallel merging. The `coRank` function essentially performs a binary search to find the correct split points in arrays `a` and `b` for a given rank `k` in the merged output. This partitioning is crucial for ensuring the correctness of the parallel merge.
- **Hybrid CPU/GPU Kernel Design:** Implemented a hybrid approach. The `mergeGPU` kernel uses GPU threads to calculate the co-ranks and partition the problem. It then efficiently dispatches the merging of these smaller, partitioned sub-arrays to the `mergeCPU` function. This demonstrates a strategy where computationally intensive high-level parallel orchestration is done on the GPU, while potentially simpler or less parallelizable sub-tasks can be delegated to CPU code within the kernel.
- **Trade-offs in Hybrid Approach:**  Considered the trade-offs of a hybrid approach. While leveraging the CPU for sub-merges can simplify kernel code and potentially reuse existing optimized CPU merge routines, it also introduces CPU-GPU synchronization points within the kernel execution and might limit the overall parallelism if the CPU merge parts become bottlenecks.
- **Performance of GPU Merge:** Benchmarked the GPU merge implementation and observed a performance improvement compared to a purely CPU-based merge for the given problem sizes.

**Co-ranking Function (`coRank`):**
```c
__device__ int coRank(int *a, int *b, int m, int n, int k)
{
    int iLow, iHigh;
    if(k > n)
    {
        iLow = k - n;
    }
    else
    {
        iLow = 0;
    }

    if(k < m)
    {
        iHigh = k;
    }
    else
    {
        iHigh = m;
    }

    while (iLow <= iHigh)
    {
        int i = (iLow + iHigh) / 2;
        int j = k - i;

        if (i > 0 && j < n && a[i - 1] > b[j])
        {
            iHigh = i;
        }
        else if (j > 0 && i < m && b[j - 1] > a[i])
        {
            iLow = i;
        }
        else
        {
            return i;
        }
    }
}
```

**GPU Merge Kernel (`mergeGPU`)**
```c
__global__ void mergeGPU(int *a, int *b, int *c, int m, int n)
{
    int k = (blockIdx.x * blockDim.x + threadIdx.x) * ELEMENTS;

    if(k < (m + n))
    {
        int i = coRank(a, b, m, n, k);
        int j = k - i;
        int kNext;

        if((k + ELEMENTS) < ( m + n))
        {
            kNext = k + ELEMENTS;
        }
        else
        {
            kNext = m + n;    
        }

        int iNext = coRank(a, b, m, n, kNext);
        int jNext = kNext - iNext;

        mergeCPU(&a[i], &b[j], & c[k], (iNext - i), (jNext - j));
    }
}
```

**Results (for N=2^20, M=2^21):**
- **CPU execution time:** 17.96 ms
- **GPU execution time:** 15.97 ms