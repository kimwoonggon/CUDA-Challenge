# Day 54: Per-Channel Quantized Matrix Multiplication

**Objective:**
- **Implement Per-Channel Quantized Matrix Multiplication:** Develop a CUDA C++ program to perform matrix multiplication using per-channel quantization. This involves calculating separate quantization parameters (scale and zero-point) for each row of the first matrix and each column of the second matrix, quantizing the matrices accordingly, performing the multiplication in the integer domain, and then dequantizing the result back to floating-point.
- **Understand Per-Channel Quantization:** Learn the concept of per-channel quantization as a method to improve the accuracy of quantized neural networks by using different quantization parameters for different channels (in this case, rows and columns).
- **Utilize CUDA for Parallel Quantization and Multiplication:** Gain experience in using CUDA to parallelize the steps of calculating quantization parameters, quantizing the matrices, and performing the dequantized matrix multiplication.

**Key Learnings:**
- **Per-Channel Quantization:**
    - Understood that per-channel quantization can lead to better accuracy than per-tensor quantization, as it accounts for variations in the dynamic range of values across different channels (rows or columns).
    - Learned the process of determining the minimum and maximum values for each channel to calculate the scale and zero-point.
- **Quantization Parameters (Scale and Zero-Point):**
    - Understood the role of the scale factor in mapping the floating-point range to the integer range (-128 to 127 for int8).
    - Learned how the zero-point is used to represent the floating-point value 0 in the quantized integer range.
- **Quantization and Dequantization Formulas:**
    - Quantization: `q = round(input / scale) + zero_point`
    - Dequantization (implicitly in the `dequantize` kernel): `result = (qa - zp_a) * (qb - zp_b) * scale_a * scale_b`
- **Matrix Multiplication in Quantized Domain:**
    - Performed the core matrix multiplication using the quantized 8-bit integer values, taking into account the zero-points during the accumulation.
- **CUDA Implementation for Per-Channel Quantization:**
    - Developed CUDA kernels (`quantize` and `dequantize`) to perform the quantization and the dequantized matrix multiplication in parallel on the GPU.
    - Implemented a CPU function (`qparams`) to calculate the per-channel quantization parameters.

**Code Implementation Details:**

- **Includes:**
    - `stdio.h`: Standard input/output library.
    - `stdlib.h`: Standard general utilities library.
    - `cuda_runtime.h`: CUDA runtime API.
    - `curand_kernel.h`: CUDA random number generation.
    - `math.h`: Mathematical functions like `roundf`, `fminf`, `fmaxf`.
    - `cmath`: C++ math functions like `std::round`.
    - `algorithm`: For `std::max` and `std::min`.
- **Defines:**
    - `N`: Dimension of the square matrices (N x N), set to 1024.
    - `TILE`: Tile size used for thread block dimensions, set to 16.
- **`init` Global Function:**
    - Initializes a matrix on the GPU with random floating-point values between -1.0 and 1.0 using `curand`.
- **`quantize` Global Function:**
    - This CUDA kernel quantizes a floating-point input matrix to an 8-bit integer output matrix.
    - It takes per-channel scale and zero-point values as input.
    - The `per_row` flag determines whether the quantization is done per row or per column.
    - It calculates the quantized value using the formula and clamps it to the range [-128, 127].
- **`dequantize` Global Function:**
    - This CUDA kernel performs the matrix multiplication on the quantized input matrices (`a` and `b`) and then dequantizes the result.
    - It iterates through the elements of the output matrix.
    - For each element, it performs an inner product over the corresponding row of `a` and column of `b`.
    - During the inner product, it subtracts the zero-points for the respective row of `a` and column of `b` from the quantized values.
    - Finally, it multiplies the accumulated integer result by the scale factors for the corresponding row of `a` and column of `b` to get the dequantized floating-point value.
- **`qparams` Function:**
    - This CPU function calculates the per-channel quantization parameters (scale and zero-point) for a given input matrix.
    - It iterates through each row (if `per_row` is true) or each column (if `per_row` is false) of the input matrix.
    - For each channel, it finds the minimum and maximum values.
    - It calculates the range, scale (range / 255), and zero-point (rounded value of -min_val / scale).
- **`main` Function:**
    - Allocates managed memory on the GPU for the floating-point input and output matrices, the quantized integer matrices, and the per-channel scale and zero-point arrays.
    - Initializes the input matrices using the `init` kernel.
    - Calls the `qparams` function on the host to calculate the per-row quantization parameters for matrix `a` and per-column parameters for matrix `b`. These parameters are then implicitly available on the GPU due to `cudaMallocManaged`.
    - Launches the `quantize` kernel to quantize matrices `a` and `b` to 8-bit integers.
    - Launches the `dequantize` kernel to perform the matrix multiplication on the quantized matrices and dequantize the result.
    - Measures the execution time of the `dequantize` kernel using CUDA events.
    - Prints the elapsed time.
    - Frees all allocated memory and destroys the CUDA events.

**Results (N = 1024)**
- **Per-channel Quantized MatMul Time:** 21.38 ms
