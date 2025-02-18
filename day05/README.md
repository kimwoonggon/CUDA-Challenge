# Day 5: GPU Accelerated Array Summation

**Objective:**
- **Random Array Initialization:** Using CUDA's `curand` library to populate an array with random integers.
- **Partial Reduction:** Summing array elements efficiently in parallel using shared memory.
- **Performance Benchmarking:** Comparing GPU performance against a CPU summation.

Managed memory is used to simplify data handling between the host and device, and CUDA events are utilized for precise timing of the GPU kernel execution.

**Key Learnings:**
- **CUDA Random Number Generation:** Learnt how to initialize random states with `curand_init` and generate random numbers using `curand`.

- **Partial Reduction:** Implemented a reduction algorithm using shared memory and synchronization to sum large arrays efficiently.

- **Performance Benchmarking:** Measured and compared GPU and CPU execution times to highlight the benefits of parallel processing.

- **Memory Management:** Utilized CUDA Unified Memory for easier host-device data management.

**Partial Sum Kernel:**
```c
__global__ void calcSum(int *arr_in, int *arr_out, int N)
{   
    __shared__ int mem[2048];

    int next_i = blockDim.x * blockIdx.x * 2 + threadIdx.x;

    if (next_i < N) 
    {
        if (next_i + blockDim.x < N) 
        {
            mem[threadIdx.x] = arr_in[next_i] + arr_in[next_i + blockDim.x];
        } 
        else 
        {
            mem[threadIdx.x] = arr_in[next_i];
        }
    } 
    else 
    {
        mem[threadIdx.x] = 0;
    }
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if(threadIdx.x < stride)
        {
            mem[threadIdx.x] += mem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        arr_out[blockIdx.x] = mem[0];
    }
}
```

**Results:**
- **Execution Time on CPU -** 3.859000 ms
- **Execution Time on GPU -** 0.831488 ms