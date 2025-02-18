# Day 6: GPU Accelerated Matrix Multiplication with Tiling

**Objective:**
- **Random Matrix Initialization:** Use CUDA's `curand` library to populate two matrices with random integers.
- **Matrix Multiplication:** Multiply matrices in parallel on the GPU using:
  - A straightforward (naive) GPU kernel.
  - An optimized tiled GPU kernel that leverages shared memory.
- **CPU Multiplication:** Perform matrix multiplication on the CPU for performance comparison.
- **Performance Benchmarking:** Compare CPU execution time against GPU and tiled GPU execution times.

Managed memory is used to simplify data handling between the host and device, and CUDA streams are employed for concurrent matrix initialization.

**Key Learnings:**
- **CUDA Random Number Generation:** Learned how to initialize random states with `curand_init` and generate random numbers using `curand`.
- **Matrix Multiplication on GPU:** Implemented both naive and tiled matrix multiplication kernels.
- **Tiling Optimization:** Utilized shared memory to reduce global memory access and enhance performance.
- **Performance Benchmarking:** Measured and compared CPU vs GPU performance for large-scale matrix multiplication.
- **Memory Management:** Leveraged CUDA Unified Memory for streamlined host-device data management.

**Tiled Matrix Multiplication Kernel:**
```c
__global__ void matMulTiled(int *a, int *b, int *c)
{
    __shared__ int mem_a[MEM][MEM];
    __shared__ int mem_b[MEM][MEM];

    int t_x = threadIdx.x;
    int t_y = threadIdx.y;

    int b_x = blockIdx.x;
    int b_y = blockIdx.y;

    int row = b_y * MEM + t_y;
    int col = b_x * MEM + t_x;

    int val = 0;

    for(int stride = 0; stride < ceil((float)N/MEM); stride++)
    {
        if ((row < N) && ((stride * MEM + t_x) < N))
        {
            mem_a[t_y][t_x] = a[row * N + stride * MEM + t_x];
        }
        else
        {
            mem_a[t_y][t_x] = 0;
        }
        if ((col < N) && ((stride * MEM + t_y) < N))
        {
            mem_b[t_y][t_x] = b[(stride * MEM + t_y) * N + col];
        }
        else
        {
            mem_b[t_y][t_x] = 0;
        }
        __syncthreads();

        for(int k = 0; k < MEM; k++)
        {
            val += mem_a[t_y][k] * mem_b[k][t_x];
        }
        __syncthreads();
    }
    c[row * N + col] = val;
}
```

**Results:**
- **Execution Time on CPU -** 51694.223000 ms
- **Execution Time on GPU -** 675.109863 ms
- **Execution Time on GPU for a Tiled approach -** 277.286774 ms