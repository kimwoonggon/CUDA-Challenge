# Day 51: Dequantization of int8 to Floating-Point Numbers

**Objective:**
- **Implement Dequantization on GPU:** Develop CUDA C++ programs to perform dequantization of 8-bit integers back to floating-point numbers (double-precision, single-precision, and half-precision) on the GPU.
- **Understand Reverse Data Type Conversion and Scaling:** Learn how to convert quantized integer data back to its original or a higher-precision floating-point representation using a scaling factor.
- **Explore Performance Differences Across Precisions (Reverse Conversion):** Observe and compare the execution times for dequantizing int8 to different floating-point precisions (FP64, FP32, FP16) on the GPU.

**Key Learnings:**
- **Dequantization:**
    - Understood dequantization as the inverse process of quantization, where integer values are mapped back to a continuous range of floating-point numbers.
    - Learned that the same scaling factor used during quantization is typically used during dequantization to approximate the original values.
- **CUDA Implementation for Different Precisions (Reverse):**
    - Developed separate CUDA kernels (`to_fp64`, `to_fp32`, `to_fp16`) to handle dequantization from 8-bit integers (`int8_t`) to double-precision (`double`), single-precision (`float`), and half-precision (`half`).
    - Understood how to access and process elements of arrays with different data types within CUDA kernels during the reverse conversion.
- **Half-Precision Floating Point (Reverse):**
    - Further practiced using the `half` data type (FP16) and converting between `float` and `half` using `__float2half()`.
- **Parallel Data Processing on GPU (Reverse):**
    - Continued to utilize CUDA's parallel processing capabilities to efficiently perform the dequantization operation on a large array of integers concurrently.
- **Performance Measurement (Reverse):**
    - Used CUDA events to accurately measure the execution time of each dequantization kernel for different target floating-point precisions.

**Code Implementation Details:**

- **Includes:**
    - `stdio.h`: Standard input/output library.
    - `stdlib.h`: Standard general utilities library.
    - `cuda_runtime.h`: CUDA runtime API for interacting with the GPU.
    - `cuda_fp16.h`: Header file providing support for half-precision floating-point numbers.
- **Defines:**
    - `N`: The number of elements in the arrays being processed (2<sup>24</sup>).
    - `THREADS`: The number of threads per block used for the CUDA kernels (1024).
- **`to_fp64` Global Function:**
    - This CUDA kernel takes an array of `int8_t` as input and dequantizes it to an array of `double` (FP64).
    - It performs the dequantization by casting the integer input value to a `double` and then multiplying it by the `scale` factor.
- **`to_fp32` Global Function:**
    - This CUDA kernel performs the same dequantization process as `to_fp64`, but it outputs an array of `float` (FP32). It casts the `int8_t` input to `float` before multiplying by the `scale`.
- **`to_fp16` Global Function:**
    - This CUDA kernel takes an array of `int8_t` as input and dequantizes it to an array of `half` (FP16).
    - It first casts the `int8_t` input to a `float`, multiplies it by the `scale`, and then converts the `float` value to `half` using `__float2half()`.
- **`init` Function:**
    - This function runs on the CPU to initialize the input array of `int8_t`. It simulates the result of a quantization process by generating floating-point values, scaling and rounding them to the nearest integer within the int8 range.
- **`main` Function:**
    - Allocates managed memory on the GPU for the input array (`d_in`) of `int8_t` and the output arrays (`d_out64`, `d_out32`, `d_out16`) for each target floating-point precision.
    - Initializes the input array using the `init` function.
    - Defines the thread block dimension and calculates the grid dimension to cover all `N` elements.
    - Sets the `scale` factor used for dequantization (which should be the same as the one used for quantization).
    - Uses CUDA events to measure the execution time of each of the three dequantization kernels (`to_fp64`, `to_fp32`, `to_fp16`).
    - Launches each kernel with the calculated grid and block dimensions.
    - Prints the measured execution times for each dequantization operation.
    - Frees the allocated memory on the GPU.

**Results (N = 2<sup>24</sup>)**
- **int8 -> FP64:** 11.158 ms
- **int8 -> FP32:** 5.371 ms
- **int8 -> FP16:** 2.915 ms