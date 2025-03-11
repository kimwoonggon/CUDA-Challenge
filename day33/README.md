# Day 33: Matrix Multiplication with Triton

**Objective:**
- **Implement Matrix Multiplication in Triton:** Develop a Triton kernel to perform matrix multiplication and execute it on the GPU.
- **Extend Triton Knowledge to 2D Operations:**  Learn how to write Triton kernels for 2-dimensional operations like matrix multiplication, expanding on the 1D vector addition from Day 32.
- **Performance Comparison for Matrix Multiplication:**  Benchmark and compare the performance of the Triton-implemented matrix multiplication against CPU (Python loop) and NumPy matrix multiplication.

**Key Learnings:**
- **2D Triton Kernels:**
    - **2D Program IDs (`tl.program_id(axis=0)`, `tl.program_id(axis=1)`):**  Learned how to use 2D program IDs in Triton to define kernels that operate on 2D data structures like matrices.  `axis=0` typically corresponds to rows (x-dimension) and `axis=1` to columns (y-dimension) in the thread grid.
    - **2D Thread Grid (`grid = (triton.cdiv(N, 32), triton.cdiv(N, 32))`)**:  Understood how to launch Triton kernels with a 2D grid of thread blocks, appropriate for matrix operations.  The grid dimensions are calculated based on the matrix size `N` and a chosen block size (32x32 in this case).
    - **2D Indexing and Offsets:** Implemented 2D indexing within Triton kernels to access matrix elements correctly.  Using `pid_x`, `pid_y`, and `tt.arange(0, 32)` to calculate thread-local ranges (`x`, `y`) and then combine them to generate global indices for matrix elements.
- **Matrix Multiplication Algorithm in Triton (`matmul_gpu` Kernel):**
    - **Tiling/Blocking Strategy:**  Implemented a basic form of tiling (or blocking) in the `matmul_gpu` kernel by iterating in steps of `32*256` for the outer loops (`i`, `j`). This approach, while not fully optimized tiling, hints at dividing the matrix multiplication into smaller blocks to improve memory access patterns and potentially fit data closer to compute units.
    - **Accumulation in Registers:**  Used `val = tl.zeros((32, 32), dtype=tl.int32)` to initialize an accumulator in registers for each thread block and accumulated the intermediate results of the dot product within this register variable (`val += a_val * b_val`).
    - **Innermost Dot Product Loop (`k` loop):** Implemented the core matrix multiplication logic with a `for k in range(0, N)` loop, performing the dot product accumulation. In each iteration of the `k` loop, it loads elements from matrix `A` and `B`, multiplies them, and adds to the accumulator `val`.
    - **Masking for Boundary Conditions:**  Used masking (`mask = (i_idx < N) & (j_idx < N)`) to handle cases where thread blocks go beyond the matrix boundaries, ensuring correct behavior for non-power-of-two matrix sizes (although `N` is a power of two in this example, the masking is good practice).
    - **Global Memory Accesses:** Noted that the current `matmul_gpu` implementation involves repeated global memory accesses within the inner loop (`k` loop) to load elements of matrices `A` and `B`.  More advanced tiling and shared memory techniques (as will likely be explored in future days) are typically needed for significant performance optimization in matrix multiplication to reduce global memory bandwidth demands.
- **Performance Comparison for Matrix Multiplication:**
    - **Benchmarking against CPU and NumPy:** Compared the execution time of the `matmul_gpu` Triton kernel against:
        - **CPU Loop Matrix Multiplication (`matmul_cpu`):** A triply nested Python loop implementation representing a naive CPU matrix multiplication.
        - **NumPy Matrix Multiplication (`np.matmul`):** NumPy's highly optimized matrix multiplication function, leveraging BLAS (Basic Linear Algebra Subprograms) libraries and vectorized CPU instructions.
    - **Performance Metrics:** Measured execution time for GPU (Triton) and CPU (loop, NumPy) matrix multiplication.

**Code Implementation Details:**

- **Triton Kernels:**
    - **`init(x_ptr, stride_x, stride_y, N)`:**
        - Initializes matrices `A` and `B` with simple values based on row and column indices (`val = (i + x[:, None]) * (j + y[None, :])`). The `stride_x` and `stride_y` parameters are present in the function signature but not actually used in the current implementation. They might have been intended for a more complex initialization pattern or memory layout (perhaps for strided access), but in the given code, they are not utilized.
    - **`matmul_gpu(a_ptr, b_ptr, c_ptr, N)`:**
        - Implements the matrix multiplication kernel as described in "Key Learnings," with nested loops, tiling hints (though basic), accumulator registers, and masking.
- **CPU Matrix Multiplication (`matmul_cpu`):**
    - Implements a triply nested loop to perform matrix multiplication on CPU, serving as a very slow baseline for comparison.
- **Python `main()` Function:**
    - **Tensor Creation:** Creates PyTorch tensors `a`, `b`, `c` on the GPU to store matrices.
    - **Kernel Launches:** Launches the `init` and `matmul_gpu` Triton kernels with a 2D grid configuration.
    - **Timing:** Uses `time.time()` for timing the Triton kernel, CPU loop, and NumPy matrix multiplication. `torch.cuda.synchronize()` is used to ensure GPU operations are complete before timing.
    - **Result Verification:** Compares the result of Triton matrix multiplication (`c_gpu`) with the CPU loop result (`c_cpu`) to check for correctness. Also performs NumPy matrix multiplication (`c_np`) but primarily uses the CPU loop for direct comparison in the correctness check.
- **NumPy Matrix Multiplication:** Uses `np.matmul(a_np, b_np)` for highly optimized NumPy-based matrix multiplication timing.

**Results (Performance for N=256):**
- **GPU matmul time:** 0.0006 seconds
- **CPU loop time:** 5.6488 seconds
- **NumPy matmul time:** 0.0093 seconds