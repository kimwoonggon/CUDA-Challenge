# Day 8: GPU Accelerated 2D Convolution with He Normal Initialization

**Objective:**
- **2D Convolution Kernel:** Developed a CUDA kernel to perform 2D convolution on an input matrix with a given filter.
- **He Normal Initialization:** Initialized filter weights using He Normal Initialization for better weight distribution.
- **Performance Benchmarking:** Measured and compared GPU and CPU execution times for convolution.

The 2D convolution kernel applies a filter over the input data with boundary checking, while the filter weights are initialized using a He Normal distribution. Additionally, a verification kernel is used to compare the GPU results with the CPU implementation.

**Key Learnings:**
- **2D Convolution in CUDA:** Implemented a convolution kernel with nested loops and boundary conditions.
- **He Normal Initialization:** Utilized CUDA's random number generation to initialize filter weights based on the He Normal method.
- **Performance Measurement:** Employed CUDA events to time the GPU kernel and compared it to a CPU version.
- **Result Verification:** Counted mismatches between GPU and CPU results using atomic operations.

---

**GPU Convolution Kernel:**
```c
__global__ void conv2D(float *input, float *output, float *filter, int radius, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float val = 0.0f;
        int f_size = 2 * radius + 1;

        for (int row_f = 0; row_f < f_size; row_f++) {
            for (int col_f = 0; col_f < f_size; col_f++) {
                int cur_row = row - radius + row_f;
                int cur_col = col - radius + col_f;

                if (cur_row >= 0 && cur_row < N && cur_col >= 0 && cur_col < N) {
                    val += filter[row_f * f_size + col_f] * input[cur_row * N + cur_col]; 
                }
            }
        }
        output[row * N + col] = val;
    }
}
```

**Results:**
- **Execution Time on CPU -** 0.68 ms
- **Execution Time on GPU -** 0.01 ms