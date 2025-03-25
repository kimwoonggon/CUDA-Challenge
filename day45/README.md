# Day 45: Batched Matrix Multiplication on the GPU with cuBLAS

**Objective:**
- **Implement Batched Matrix Multiplication on GPU:** Develop a CUDA C++ program to perform batched matrix multiplication of multiple independent matrices in parallel using the cuBLAS library.
- **Understand Batched Operations in Linear Algebra:** Learn about the concept of batched operations, where the same operation is performed on a collection of independent data (in this case, matrices).
- **Utilize cuBLAS for High-Performance Batched Operations:** Gain experience in using the `cublasSgemmStridedBatched` function from the cuBLAS library to efficiently execute a batch of matrix multiplications on the GPU.
- **Compare Performance with CPU Batch Processing:** Compare the execution time of the batched matrix multiplication on the GPU using cuBLAS with a sequential CPU implementation.

**Key Learnings:**
- **Batched Matrix Multiplication:**
    - Understood the concept of performing the same matrix multiplication operation on a batch of independent matrices. This is a common operation in various fields, including deep learning, where the same computations might need to be performed for multiple data samples simultaneously.
- **cuBLAS for Batched Operations:**
    - Learned about the `cublasSgemmStridedBatched` function in the cuBLAS library, which is specifically designed to handle batched matrix multiplications efficiently on the GPU.
    - Understood the importance of specifying the strides between consecutive matrices in the batch within the input and output arrays. This allows cuBLAS to process each matrix in the batch in parallel.
- **Memory Layout for Batched Operations:**
    - Recognized how the input and output matrices for the batch are organized in memory, with a consistent stride between the starting addresses of each matrix.
- **Performance Benefits of Batched GPU Operations:**
    - Observed the significant performance advantage of using the GPU with cuBLAS for batched matrix multiplication compared to a sequential CPU implementation. The GPU's parallel architecture allows it to process multiple matrix multiplications concurrently, leading to substantial speedups.

**Code Implementation Details:**

- **Includes:** Standard C/CUDA libraries along with `curand_kernel.h` for random initialization and `cublas_v2.h` for using the cuBLAS library.
- **Defines:**
    - `N`: Dimension of the square matrices (N x N).
    - `MEM`: Tile size (used in the `threads` dimension for kernel launch, though not directly used in the cuBLAS call).
    - `BATCH`: Number of independent matrix multiplications to perform in the batch.
- **`init` Global Function:** Initializes a batch of `BATCH` matrices `a` and `b` on the GPU with random floating-point values using the `curand` library. The batch index is incorporated into the random seed to ensure different random values for each matrix in the batch. The `pitch` parameter is used to correctly calculate the memory offset for each matrix in the batched array.
- **`matMulCPU` Function:** Implements the batched matrix multiplication on the CPU. It iterates through each batch and performs a standard matrix multiplication for each pair of matrices. The `pitch` parameter is used to access the correct matrices within the batched arrays.
- **`verify` Function:** Compares the results of the CPU and GPU batched matrix multiplications for correctness. It iterates through each batch and each element of the resulting matrices, checking if the absolute difference is within a small tolerance.
- **`main` Function:**
    - Defines the size of the batched arrays and the pitch (stride) between consecutive matrices.
    - Allocates managed memory on the GPU for the batch of input matrices `a` and `b`, and the output matrices `c_cpu` and `c_gpu`.
    - Defines the thread block and grid dimensions for the initialization kernel.
    - Launches the `init` kernel to initialize the batch of input matrices on the GPU.
    - Performs batched matrix multiplication on the CPU and measures the execution time using `clock()`.
    - **Uses the cuBLAS library:**
        - Creates a cuBLAS handle using `cublasCreate(&handle)`.
        - Sets the `alpha` and `beta` parameters for the matrix multiplication (in this case, `alpha = 1.0f` and `beta = 0.0f`, corresponding to `C = alpha * op(B) * op(A) + beta * C`).
        - Calls the `cublasSgemmStridedBatched` function with the following parameters:
            - `handle`: The cuBLAS handle.
            - `CUBLAS_OP_N`: Specifies that neither matrix `B` nor `A` should be transposed.
            - `N, N, N`: Dimensions of the matrices (M, N, K where C(M,N) = op(B(M,K)) * op(A(K,N))).
            - `&alpha`: Pointer to the scalar alpha.
            - `b`: Pointer to the start of the batched matrix `B` on the GPU.
            - `N`: Leading dimension of matrix `B`.
            - `pitch`: Stride between the starting addresses of consecutive matrices `B` in the batch.
            - `a`: Pointer to the start of the batched matrix `A` on the GPU.
            - `N`: Leading dimension of matrix `A`.
            - `pitch`: Stride between the starting addresses of consecutive matrices `A` in the batch.
            - `&beta`: Pointer to the scalar beta.
            - `c_gpu`: Pointer to the start of the batched result matrix `C` on the GPU.
            - `N`: Leading dimension of matrix `C`.
            - `pitch`: Stride between the starting addresses of consecutive matrices `C` in the batch.
            - `BATCH`: The number of matrices in the batch.
        - Measures the execution time of the `cublasSgemmStridedBatched` function using CUDA events.
        - Destroys the cuBLAS handle using `cublasDestroy(handle)`.
    - Calls the `verify` function to compare the CPU and GPU results.
    - Prints whether the batched matrix multiplication was successful and the execution times for the CPU and cuBLAS implementations.
    - Frees the allocated memory on the GPU.

**Results:**

- CPU time: 73.92 ms
- cuBLAS batched time: 1.20 ms