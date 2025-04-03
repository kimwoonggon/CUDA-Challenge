# Day 52: Fused Quantized Matrix Multiplication

**Objective:**
- **Implement Fused Quantized Matrix Multiplication on GPU:** Develop a CUDA C++ program to perform matrix multiplication where the input matrices are quantized to 8-bit integers on-the-fly, the multiplication is performed using integer arithmetic, and the result is then dequantized back to a floating-point representation. This "fused" approach aims to improve performance by operating on lower-precision data.
- **Understand Quantization and Dequantization within Matrix Multiplication:** Learn how quantization and dequantization steps can be integrated directly into a matrix multiplication kernel on the GPU.
- **Explore Performance Benefits of Mixed-Precision Computation:** Observe the potential speedup achieved by using lower-precision integer arithmetic for the core computation compared to standard floating-point matrix multiplication on the CPU.
- **Analyze Accuracy Trade-offs with Quantization:** Recognize that quantization introduces a loss of precision and understand how to compare the results of the quantized operation with a full-precision floating-point calculation.

**Key Learnings:**
- **Fused Quantized Matrix Multiplication:**
    - Understood the concept of performing matrix multiplication using quantized integer representations of the input matrices to potentially improve computational throughput and reduce memory bandwidth.
    - Learned that the "fused" aspect refers to performing the quantization and dequantization steps within the same kernel as the matrix multiplication itself.
- **Integer Arithmetic for Matrix Multiplication:**
    - Experienced how to perform the core matrix multiplication using 8-bit integer values and accumulate the result in a higher-precision integer type (32-bit) to mitigate overflow.
- **Scaling Factors for Quantization and Dequantization:**
    - Understood the role of scaling factors (`scale_a`, `scale_b`) in mapping the floating-point input values to the integer range and then back to an approximate floating-point result. The choice of these scaling factors is crucial for the accuracy of the quantized operation.
- **CUDA Streams for Concurrent Operations:**
    - Utilized CUDA streams to potentially overlap the initialization of the two input matrices (`a` and `b`), which can lead to improved overall execution time.
- **Accuracy vs. Performance Trade-off:**
    - Observed that while the fused quantized matrix multiplication can be significantly faster than the standard floating-point version, it comes at the cost of reduced accuracy due to the quantization step. A larger tolerance is needed when comparing the results.

**Code Implementation Details:**

- **Includes:**
    - `stdio.h`: Standard input/output library.
    - `math.h`: Mathematical functions like `fabsf`.
    - `curand_kernel.h`: CUDA random number generation library.
    - `time.h`: For seeding the random number generator.
    - `cuda_runtime.h`: CUDA runtime API for interacting with the GPU.
- **Defines:**
    - `N`: Dimension of the square matrices (N x N), set to 1024.
    - `MEM`: Tile size used for thread block dimensions, set to 16.
- **`init` Global Function:**
    - Initializes a matrix on the GPU with random floating-point values between -1.0 and 1.0 using the `curand` library. It takes a CUDA stream as an argument to allow for asynchronous execution.
- **`matMulFusedQuant` Global Function:**
    - This CUDA kernel performs the fused quantized matrix multiplication.
    - **Thread and Block Mapping:** Standard 2D grid and thread block configuration for matrix multiplication.
    - **Quantization:** Inside the inner loop (over `k`), each element of matrices `a` and `b` is quantized to an 8-bit integer (`int8_t`) using the provided scaling factors `scale_a` and `scale_b`. The values are clamped to the range [-128, 127].
    - **Integer Multiplication and Accumulation:** The quantized integer values are multiplied, and the result is accumulated in a 32-bit integer variable `acc`.
    - **Dequantization:** After the inner loop completes, the accumulated integer result is dequantized back to a floating-point number by multiplying it with the product of the scaling factors (`scale_a * scale_b`).
    - The final floating-point result is stored in the output matrix `c`.
- **`main` Function:**
    - Allocates managed memory on the GPU for the input matrices `a` and `b`, the output matrix from the fused quantized operation `c_fused`, and the output matrix from the standard floating-point CPU multiplication `c_cpu`.
    - Creates two CUDA streams (`strm1`, `strm2`) for the asynchronous initialization of matrices `a` and `b`.
    - Launches the `init` kernel on the respective streams to initialize `a` and `b` concurrently.
    - Synchronizes both streams to ensure that the initialization is complete before proceeding.
    - Performs a standard floating-point matrix multiplication on the CPU for comparison.
    - Launches the `matMulFusedQuant` kernel on the GPU.
    - Measures the execution time of both the CPU and the fused quantized GPU operations using `clock()` and CUDA events, respectively.
    - Compares the results of the fused quantized matrix multiplication with the CPU result. A larger tolerance (`5.0f`) is used in the comparison due to the expected loss of precision from quantization.
    - Prints the number of mismatches found (if any) and whether the fused quantized matmul was "verified" (meaning the difference was within the tolerance).
    - Prints the execution times for both the CPU and the fused quantized GPU operations.
    - Frees the allocated memory and destroys the CUDA streams and events.


**Results (N = 1024)**
- **CPU time:** 4606.42 ms
- **Fused Quantized GPU time:** 35.83 ms