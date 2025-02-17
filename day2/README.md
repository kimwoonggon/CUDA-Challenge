# Day 2: Vector Addition Acceleration

**Objective:**  
- Accelerate vector addition on a very large vector (size = \(10^8\)) using CUDA.
- Utilize multi-stream initialization for efficient data loading.
- Employ a strided approach to optimize memory access.
- Compare GPU performance with CPU performance.

**Key Learnings:**  
- **Memory Allocation:** How to allocate large arrays on the GPU using `cudaMalloc`.
- **Strided Memory Access:** Implementing kernels that process elements in a strided loop so that all threads contribute evenly.
- **CUDA Streams:** Overlapping initialization of multiple arrays concurrently using separate CUDA streams.
- **Performance Boost:** Observing a significant speed-up on the RTX A4000 compared to CPU execution.

**Vector Addition Kernel:**  
```c
__global__ void addVector(int *a, int *b, int *c)
{
    int t_ID = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    if(t_ID < N)
    {
        for (int i = t_ID; i < N; i += stride)
        {
            c[i] = a[i] + b[i];
        }
    }
}
```