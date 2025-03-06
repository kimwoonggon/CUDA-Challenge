# Day 28: SpMV Performance with ELL Sparse Matrix Format in CUDA

**Objective:**
- **Implement SpMV for ELL Format:** Develop a CUDA kernel to perform Sparse Matrix-Vector Multiplication (SpMV) when the sparse matrix is stored in the Ellpack (ELL) format.
- **Evaluate ELL SpMV Performance:** Benchmark and measure the performance (execution time) of the implemented ELL SpMV kernel and compare it against CPU-based dense SpMV and GPU SpMV implementations using COO and CSR formats (from previous days).
- **Understand ELL Format in SpMV Context:** Analyze the suitability and performance characteristics of the ELL format for SpMV operations on GPUs, considering its strengths and weaknesses.

**Key Learnings:**
- **Sparse Matrix-Vector Multiplication (SpMV) with ELL Format:**
    - **ELL SpMV Algorithm:**  Learned how to perform SpMV when a matrix is in ELL format.  For each row, iterate through the `ellVal` and `ellCol` entries up to `max_nnz`. If a column index in `ellCol` is valid (not a padding value, like -1 in our case), multiply the corresponding `ellVal` with the input vector element at that column index and accumulate the result into the output vector for the current row.
    - **Parallelism in ELL SpMV:**  Understood that ELL format naturally lends itself to row-parallel SpMV execution on GPUs. Each thread (or block of threads) can be assigned to compute the SpMV for one or more rows independently. The regular structure of ELL, with fixed `max_nnz` for all rows, simplifies parallel processing.
- **CUDA Kernel Implementation for ELL SpMV (`spMV_ELL`):**
    - **Kernel Structure:** Implemented a `spMV_ELL` CUDA kernel that takes `ellCol`, `ellVal`, input vector `in_vect`, output vector `out_vect`, and `ell_max_nnz` as input.
    - **Row-based Parallelism:**  The kernel is designed for row-parallel execution. Each CUDA thread block is responsible for processing a set of rows.  Inside the kernel, for each row, a loop iterates up to `ell_max_nnz`. It checks for valid column indices (to avoid padding values) and performs the multiplication and summation to compute the output vector element for that row.
    - **Global Memory Access Pattern:**  Noted that in ELL SpMV, memory access to `ellVal`, `ellCol`, and the input vector `in_vect` is relatively regular and predictable, which is beneficial for GPU memory performance.
- **Performance Comparison:**
    - **Benchmarking against CPU and other GPU SpMV Methods:**  Compared the execution time of the `spMV_ELL` kernel with:
        - CPU-based dense SpMV (`spMV_CPU`).
        - GPU-based COO SpMV (`spMV_COO` from Day 23/24).
        - GPU-based CSR SpMV (`spMV_CSR` from Day 24).
    - **Performance Observations:** Observed the relative performance of ELL SpMV in comparison to other methods and analyzed scenarios where ELL might be perform well or less optimally.

**Code Implementation Details:**

- **Initialization Kernels (`init`, `initVect`):** Reuses the `init` and `initVect` CUDA kernels from Day 27 to generate a dense sparse matrix and an input vector on the GPU.
- **Sparse Format Conversion Functions (`COO`, `CSR`, `ELL`, `JDS`):** Reuses the conversion functions from Day 27 to convert the dense matrix to COO, CSR, ELL, and JDS formats.
- **SpMV Kernels:**
    - **`spMV_COO` and `spMV_CSR`:** Reuses the `spMV_COO` and `spMV_CSR` kernels from Day 24 for comparison.
    - **`spMV_ELL` (New Kernel):** Implements the SpMV kernel for ELL format as described in "Key Learnings".  It calculates the output vector elements by iterating through the `ellVal` and `ellCol` arrays for each row, up to `ell_max_nnz`.
- **CPU SpMV (`spMV_CPU`):**  Reuses the CPU-based dense SpMV function from Day 24 for baseline comparison.
- **Timing and Benchmarking:** Uses CUDA events to accurately measure the execution time of the GPU kernels and `clock()` for CPU timing.  Calculates creation times for different sparse formats as well as SpMV operation times.

**Results (Performance for N=2^12, SPARSITY=0.1):**
- **Dense Matrix Storage Size:** 64.00 MB
- **Sparse Matrix Storage Size (COO):** 19.19 MB
- **Sparse Matrix Storage Size (CSR):** 12.81 MB
- **Sparse Matrix Storage Size (ELL):** 15.41 MB
- **Sparse Matrix Storage Size (JDS):** 12.81 MB

- **COO Creation Time:** 61.929000 ms
- **COO Operation Time:** 870.334473 ms
- **COO Total Time:** 932.263473 ms

- **CSR Creation Time:** 50.014000 ms
- **CSR Operation Time:** 0.286400 ms
- **CSR Total Time:** 50.300400 ms

- **ELL Creation Time:** 102.755000 ms
- **ELL Operation Time:** 11.824992 ms
- **ELL Total Time:** 114.579992 ms

- **CPU spMV Time:** 30.488001 ms