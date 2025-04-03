# Day 50: Quantization of Floating-Point Numbers to int8

**Objective:**
- **Implement Quantization on GPU:** Develop CUDA C++ programs to perform quantization of floating-point numbers (double-precision, single-precision, and half-precision) to 8-bit integers on the GPU.
- **Understand Data Type Conversion and Scaling:** Learn how to convert floating-point data to a lower-precision integer format using a scaling factor.
- **Explore Performance Differences Across Precisions:** Observe and compare the execution times for quantizing different floating-point precisions (FP64, FP32, FP16) to int8 on the GPU.

**Key Learnings:**
- **Quantization:**
    - Understood the concept of quantization, which is the process of mapping a continuous range of values (or a large set of discrete values) to a smaller set of discrete values. In this case, we are quantizing floating-point numbers to the range of an 8-bit integer (-128 to 127).
    - Learned about the use of a scaling factor to map the floating-point range to the integer range. The choice of scale factor is crucial for preserving the information from the original data.
- **CUDA Implementation for Different Precisions:**
    - Developed separate CUDA kernels (`fp64`, `fp32`, `fp16`) to handle quantization from double-precision (`double`), single-precision (`float`), and half-precision (`half`) to 8-bit integers (`int8_t`).
    - Understood how to access and process elements of arrays with different data types within CUDA kernels.
- **Half-Precision Floating Point:**
    - Learned about the `half` data type (FP16) and how to convert between `half` and `float` using the `cuda_fp16.h` header and the `__half2float()` and `__float2half()` functions.
- **Parallel Data Processing on GPU:**
    - Utilized CUDA's parallel processing capabilities by launching kernels with a large number of threads to perform the quantization operation on a large array of numbers concurrently.
- **Performance Measurement:**
    - Used CUDA events to accurately measure the execution time of each quantization kernel for different floating-point precisions.

**Code Implementation Details:**

- **Includes:**
    - `stdio.h`: Standard input/output library.
    - `stdlib.h`: Standard general utilities library.
    - `cuda_runtime.h`: CUDA runtime API for interacting with the GPU.
    - `cuda_fp16.h`: Header file providing support for half-precision floating-point numbers.
- **Defines:**
    - `N`: The number of elements in the arrays being processed (2<sup>24</sup>).
    - `THREADS`: The number of threads per block used for the CUDA kernels (1024).
- **`fp64` Global Function:**
    - This CUDA kernel takes an array of `double` (FP64) as input and quantizes it to an array of `int8_t`.
    - It calculates the quantized value `q` by dividing the input value `x` by the `scale` factor, casting it to a `float` for the division and then converting it to the nearest integer using `__float2int_rn()`.
    - The result is clamped to the range of an 8-bit signed integer (-128 to 127) using `max()` and `min()` before being stored in the output array.
- **`fp32` Global Function:**
    - This CUDA kernel performs the same quantization process as `fp64`, but it takes an array of `float` (FP32) as input.
- **`fp16` Global Function:**
    - This CUDA kernel takes an array of `half` (FP16) as input.
    - It first converts the `half` value to a `float` using `__half2float()` and then performs the same quantization steps as in the `fp32` kernel.
- **`init` Function:**
    - This function runs on the CPU to initialize the input arrays for FP64, FP32, and FP16 with sample floating-point values generated using `sinf`. The `float` value is cast to `double` and converted to `half` accordingly.
- **`main` Function:**
    - Allocates managed memory on the GPU for the input arrays (`d_fp64`, `d_fp32`, `d_fp16`) and the output arrays (`d_out64`, `d_out32`, `d_out16`) for each floating-point precision.
    - Initializes the input arrays using the `init` function.
    - Defines the thread block dimension and calculates the grid dimension to cover all `N` elements.
    - Sets the `scale` factor used for quantization.
    - Uses CUDA events to measure the execution time of each of the three quantization kernels (`fp64`, `fp32`, `fp16`).
    - Launches each kernel with the calculated grid and block dimensions.
    - Prints the measured execution times for each quantization operation.
    - Frees the allocated memory on the GPU.

**Results (N = 2<sup>24</sup>)**
- **FP64 -> int8:** 11.764 ms
- **FP32 -> int8:** 5.453 ms
- **FP16 -> int8:** 2.785 ms