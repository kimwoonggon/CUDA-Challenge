# Day 53: Tiled and Fused Quantized Matrix Multiplication with cuBLASLt Comparison

**Objective:**
- **Implement Tiled and Fused Quantized Matrix Multiplication:** Develop a CUDA C++ program to perform matrix multiplication using a tiled approach with on-the-fly quantization of inputs to 8-bit integers and outputting the result in half-precision (FP16). The tiling strategy aims to improve data locality by utilizing shared memory.
- **Compare Performance with cuBLASLt:** Evaluate the performance of the custom tiled and fused quantized matrix multiplication against the highly optimized INT8 General Matrix Multiplication (GEMM) function provided by the NVIDIA cuBLASLt library.
- **Understand the Benefits of Tiling and Optimized Libraries:** Learn about the advantages of using tiling techniques in matrix multiplication for better memory access patterns and the significant performance gains offered by well-optimized libraries like cuBLASLt.

**Key Learnings:**
- **Tiled Matrix Multiplication:**
    - Understood the concept of dividing the matrices into smaller blocks (tiles) and processing these tiles in shared memory to reduce the number of accesses to slower global memory, thereby improving performance.
- **Fused Quantization and Half-Precision Output:**
    - Implemented the quantization of input matrices (FP32) to 8-bit integers (INT8) directly within the matrix multiplication kernel.
    - Learned how to convert the final dequantized result to half-precision (FP16) for potential memory savings and compatibility with other FP16 operations.
- **cuBLASLt Library:**
    - Gained experience in using the NVIDIA cuBLASLt library, which provides highly optimized routines for tensor operations, including matrix multiplication.
    - Learned how to set up and call the INT8 GEMM function from cuBLASLt.
- **Performance Comparison and Library Optimization:**
    - Observed a direct performance comparison between a custom CUDA implementation (even with tiling and quantization) and a highly optimized library function from cuBLASLt.
    - Understood the significant performance advantages offered by well-tuned libraries that leverage the full capabilities of the GPU hardware.

**Code Implementation Details:**

- **Includes:**
    - `stdio.h`: Standard input/output library.
    - `cuda_runtime.h`: CUDA runtime API.
    - `cuda_fp16.h`: Support for half-precision floating-point numbers.
    - `cublasLt.h`: NVIDIA cuBLASLt library header.
    - `curand_kernel.h`: CUDA random number generation.
    - `time.h`: For seeding random numbers.
    - `cmath`: Math functions like `round`.
    - `algorithm`: For `std::max` and `std::min`.
- **Defines:**
    - `N`: Dimension of the square matrices (N x N), set to 1024.
    - `TILE`: Size of the tiles used in the tiled matrix multiplication, set to 16.
- **`init` Global Function:**
    - Initializes a matrix on the GPU with random floating-point values between -1.0 and 1.0 using `curand`.
- **`matMulFusedTiledQuant` Global Function:**
    - This CUDA kernel performs the tiled and fused quantized matrix multiplication.
    - **Shared Memory:** Uses `__shared__` memory to create tiles (`tile_a`, `tile_b`) for the sub-blocks of matrices `a` and `b`.
    - **Tiling:** The outer loop iterates through the tiles of matrix `b`. Inside, it loads a tile of `a` and a corresponding tile of `b` into shared memory.
    - **Quantization:** As elements are loaded into shared memory, they are quantized to `int8_t` using the scaling factors `scale_a` and `scale_b`.
    - **Matrix Multiplication:** The inner loop performs the multiplication of the tiles stored in shared memory. Accumulation is done in `int32_t`.
    - **Dequantization and Output:** After processing all the tiles, the accumulated result is dequantized and converted to `half` (FP16) before being written to the output matrix `c`.
- **`runCuBLASLtINT8` Function:**
    - This function demonstrates how to use the cuBLASLt library to perform an INT8 matrix multiplication.
    - **Initialization:** It initializes the cuBLASLt handle, matrix operation descriptor, matrix layout descriptors for the input (INT8) and output (FP16) matrices, and the preference settings for the workspace size.
    - **Matrix Multiplication:** It calls the `cublasLtMatmul` function with the appropriate parameters to perform the INT8 GEMM.
    - **Cleanup:** It destroys the cuBLASLt handle and frees the allocated workspace memory.
- **`main` Function:**
    - Allocates managed memory for the input FP32 matrices (`a_fp32`, `b_fp32`), intermediate INT8 matrices (`a_int8`, `b_int8`), and the output FP16 matrices (`c_fused`, `c_cublas`).
    - Initializes the input FP32 matrices using the `init` kernel.
    - Manually quantizes the FP32 input matrices to INT8 on the GPU for use with the `runCuBLASLtINT8` function.
    - Launches the custom `matMulFusedTiledQuant` kernel and measures its execution time using CUDA events.
    - Calls the `runCuBLASLtINT8` function to perform the matrix multiplication using the cuBLASLt library and measures its execution time.
    - Prints the execution times for both the custom implementation and the cuBLASLt function.
    - Frees all allocated memory.

**Output Analysis:**

The output shows a significant performance difference between the custom tiled and fused quantized implementation and the cuBLASLt INT8 GEMM function:


**Results (N = 1024)**
- **Fused Tiled (int8→fp16):** 32.19 ms
- **cuBLASLt INT8 GEMM:** 1.11 ms
