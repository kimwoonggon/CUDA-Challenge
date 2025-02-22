# Day 18: Brent-Kung Prefix Sum Algorithm - Exploring Trade-offs in Parallel Prefix Sum

**Objective:**
- **Implement Brent-Kung Prefix Sum:** Implement the Brent-Kung parallel prefix sum (scan) algorithm on the GPU. This day explores an alternative parallel prefix sum algorithm to the Kogge-Stone algorithm studied on Day 17.
- **Compare Brent-Kung and Kogge-Stone:** Understand the structural and performance differences between the Brent-Kung and Kogge-Stone algorithms, both of which achieve parallel prefix sum computation.
- **Performance Benchmarking:** Compare the performance of the Brent-Kung GPU kernel against a sequential CPU implementation and consider its performance relative to the Kogge-Stone implementations from Day 17.

**Key Learnings:**
- **Brent-Kung Algorithm:** Learned about the Brent-Kung algorithm, another efficient parallel prefix sum algorithm. Unlike Kogge-Stone which is strictly a parallel prefix sum, Brent-Kung is often described as a parallel algorithm for *list ranking* but can be adapted for prefix sum. In the context of prefix sum, Brent-Kung utilizes a different parallel structure compared to Kogge-Stone, often involving more arithmetic operations but potentially different memory access patterns.
- **Algorithm Trade-offs:** Understood that different parallel algorithms, like Kogge-Stone and Brent-Kung, can have different performance characteristics. Kogge-Stone is known for its lower latency and fewer arithmetic operations, while Brent-Kung might offer advantages in other aspects, such as memory access patterns or implementation complexity in certain scenarios.
- **Work-Efficiency vs. Span:**  While both algorithms are work-efficient (O(N) work for N elements), they can differ in their span (longest path of dependencies, affecting latency). Kogge-Stone typically has a smaller span (O(log N)) than Brent-Kung, potentially leading to different performance scaling under various conditions.
- **Practical Algorithm Selection:**  Realized that the "best" prefix sum algorithm for a given GPU application may depend on factors like problem size, hardware architecture, and other workload characteristics. Benchmarking and understanding the trade-offs between algorithms like Kogge-Stone and Brent-Kung is important for optimal performance.

**Brent-Kung GPU Kernel (`brentGPU`):**
```c
__global__ void brentGPU(int *x, int *y, int *partialSums)
{
    int t_i = blockIdx.x * blockDim.x + threadIdx.x;
    int chunk = 2 * blockIdx.x * blockDim.x;
    __shared__ int x_s[2*THREADS];

    if (t_i < N)
    {
        x_s[threadIdx.x] = x[threadIdx.x + chunk];
        x_s[threadIdx.x + blockDim.x] = x[threadIdx.x + chunk + blockDim.x];
    }
    else
    {
        x_s[threadIdx.x] = 0;
        x_s[threadIdx.x + blockDim.x] = 0;
    }
    __syncthreads();

    for (int stride = 1; stride <= blockDim.x; stride *= 2)
    {
        int index = (threadIdx.x + 1) * 2 * stride - 1;
        if (index < 2 * blockDim.x)
        {
            x_s[index] += x_s[index - stride];
        }
        __syncthreads();
    }

    for (int stride = blockDim.x/2; stride >= 1; stride /= 2)
    {
        int index = (threadIdx.x + 1) * 2 * stride - 1;
        if((index + stride) < 2 * blockDim.x)
        {
            x_s[index + stride] += x_s[index];
        }
        __syncthreads();
    }

    if (threadIdx.x == blockDim.x - 1)
    {
        partialSums[blockIdx.x] = x_s[2 * blockDim.x - 1];
    }

    if (t_i < N)
    {
        y[chunk + threadIdx.x] = x_s[threadIdx.x];
        y[chunk + threadIdx.x + blockDim.x] = x_s[threadIdx.x + blockDim.x];
    }
}
```

**Results (for N=2^26):**
- **CPU execution time:** 191.0370 ms
- **GPU execution time:** 52.3859 ms