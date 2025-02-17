# Day 11: 3D Stencil Operation Acceleration

**Objective:**
- **3D Stencil Kernel:** Develop a CUDA kernel to perform a 3D stencil operation.
- **3D Grid and Block Configuration:** Utilize a 3D grid and block configuration to process 3D data efficiently.
- **Performance Benchmarking:** Compare the performance of the GPU-accelerated 3D stencil operation with a CPU-based implementation.

**Key Learnings:**
- **3D Kernel Development:** Extended kernel development skills to 3D operations. Learned how to write a CUDA kernel (`stencilGPU`) to perform stencil calculations in three dimensions, processing volumetric data.
- **3D Grid & Block Configuration:**  Configured 3D grids and blocks to map effectively to the 3D data space. This allows for parallel processing across all three dimensions of the data, which is essential for efficient 3D stencil computations.
- **3D Stencil Operation Implementation:** Implemented a 7-point 3D stencil operation where each point in the 3D grid is updated based on its current value and the values of its 6 immediate neighbors in 3D space (left, right, up, down, front, and back).
- **Performance Gain in 3D Processing:** Observed a significant performance speedup on the GPU compared to the CPU for the 3D stencil operation, highlighting the benefits of GPU acceleration for computationally intensive 3D data processing.

**3D Stencil Kernel:**
```c
__constant__ float val[7];

__global__ void stencilGPU(float *input, float *output, float *val, int N)
{
    int depth = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(depth >=1 && depth < N - 1 && row >=1 && row < N - 1 && col >=1 && col < N - 1)
    {
        output[depth * N * N + row * N + col] = val[0] * input[depth * N * N + row * N + col] +
                                                val[1] * input[depth * N * N + row * N + col - 1] +
                                                val[2] * input[depth * N * N + row * N + col + 1] +
                                                val[3] * input[depth * N * N + (row - 1) * N + col] +
                                                val[4] * input[depth * N * N + (row + 1) * N + col] +
                                                val[5] * input[(depth - 1) * N * N + row * N + col] +
                                                val[6] * input[(depth + 1) * N * N + row * N + col];
    }
}
```

**Results:**
   - **GPU Execution time:** 36.248322 ms
   - **CPU Execution time:** 201.901993 ms