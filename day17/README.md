# Day 17: Kogge-Stone Prefix Sum Algorithm - Shared Memory and Double Buffering Optimizations

**Objective:**
- **Implement Kogge-Stone Prefix Sum:** Implement the Kogge-Stone parallel prefix sum (scan) algorithm on the GPU, exploring its efficient parallel structure for prefix sum computation.
- **Shared Memory Optimization:** Utilize shared memory to implement the Kogge-Stone algorithm, reducing global memory accesses during the iterative summation process.
- **Double Buffering for Performance:** Explore double buffering within the shared memory Kogge-Stone kernel to potentially improve performance by overlapping memory operations and computation.
- **Performance Benchmarking:** Compare the performance of a naive shared memory Kogge-Stone GPU kernel and a double-buffered shared memory version against a sequential CPU implementation.

**Key Learnings:**
- **Kogge-Stone Algorithm:** Learned about the Kogge-Stone algorithm, a fast parallel prefix sum algorithm known for its logarithmic time complexity and efficient parallel structure, particularly suitable for GPU architectures.
- **Parallel Prefix Sum:** Understood the concept of prefix sum and its importance in various parallel algorithms, including sorting, stream compaction, and more complex computations.
- **Shared Memory Kogge-Stone Implementation:** Implemented the Kogge-Stone algorithm in CUDA using shared memory. This significantly reduces global memory traffic by performing intermediate calculations in fast on-chip shared memory.
- **Double Buffering Technique:** Explored the double buffering optimization technique within the shared memory Kogge-Stone kernel. Double buffering, using two shared memory arrays, aims to hide latency by allowing computation to proceed in one buffer while data is being prepared in the other. In this context, it might help overlap memory transfers within shared memory between Kogge-Stone algorithm stages.
- **Performance Comparison and Algorithm Choice:** Compared the performance of the CPU and different GPU Kogge-Stone implementations, observing the performance benefits of the parallel GPU algorithms and the potential impact of double buffering.

**Kogge-Stone Kernels:**

**1. Basic Shared Memory Kogge-Stone Kernel (`koggeGPU`):**
```c
__global__ void koggeGPU(int *x, int *y, int *partialSums)
{
    int t_i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int x_s[THREADS];

    if (t_i < N)
    {
        x_s[threadIdx.x] = x[t_i];
    }
    else
    {
        x_s[threadIdx.x] = 0;
    }
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        __syncthreads();
        int temp = 0;
        if (threadIdx.x >= stride)
        {
            temp = x_s[threadIdx.x] + x_s[threadIdx.x - stride];
        }
        __syncthreads();
        if (threadIdx.x >= stride)
        {
            x_s[threadIdx.x] = temp;
        }
    }

    if (threadIdx.x == blockDim.x - 1)
    {
        partialSums[blockIdx.x] = x_s[threadIdx.x];
    }

    if (t_i < N)
    {
        y[t_i] = x_s[threadIdx.x];
    }
}
```
**1. Double-Buffered Shared Memory Kogge-Stone Kernel (`koggeDoubleBufferGPU`):**
```c
__global__ void koggeDoubleBufferGPU(int *x, int *y, int *partialSums)
{
    int t_i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int shared1[THREADS];
    __shared__ int shared2[THREADS];

    int *inShared = shared1;
    int *outShared = shared2;

    if (t_i < N)
    {
        shared1[threadIdx.x] = x[t_i];
    }
    else
    {
        shared1[threadIdx.x] = 0;
    }
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if (threadIdx.x >= stride)
        {
            outShared[threadIdx.x] = inShared[threadIdx.x] + inShared[threadIdx.x - stride];
        }
        else
        {
            outShared[threadIdx.x] = inShared[threadIdx.x];
        }
        __syncthreads();

        int *t = inShared;
        inShared = outShared;
        outShared = t;
    }

    if (threadIdx.x == blockDim.x - 1)
    {
        partialSums[blockIdx.x] = inShared[threadIdx.x];
    }

    if (t_i < N)
    {
        y[t_i] = inShared[threadIdx.x];
    }
}
```

**Results (for N=2^26):**
- **CPU execution time:** 185.6950 ms
- **GPU execution time:** 54.8239 ms
- **GPU execution time (Double Buffer):** 52.4045 ms