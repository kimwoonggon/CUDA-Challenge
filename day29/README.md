# Day 29: Completion of SpMV Implementations and Comparative Analysis (COO, CSR, ELL, JDS)

**Objective:**
- **Implement SpMV for JDS Format:** Implement a CUDA kernel to perform Sparse Matrix-Vector Multiplication (SpMV) for matrices stored in the Jagged Diagonal Storage (JDS) format.
- **Comprehensive SpMV Performance Comparison:**  Benchmark and compare the performance (execution time for both format conversion and SpMV operation) of all four GPU-based SpMV implementations (COO, CSR, ELL, JDS) against each other and against CPU-based dense SpMV.
- **Analyze Format Suitability for SpMV:**  Provide a comparative analysis of COO, CSR, ELL, and JDS formats in the context of SpMV operations on GPUs, discussing their trade-offs in terms of performance, storage, and implementation complexity.  Identify scenarios where each format might be most appropriate.

**Key Learnings:**
- **Sparse Matrix-Vector Multiplication (SpMV) with JDS Format:**
    - **JDS SpMV Algorithm:** Learned the algorithm for performing SpMV with the JDS format. This involves understanding how to traverse the jagged diagonals and use the permutation array (`jdsPerm`) to correctly map the results back to the original row order.
    - **Challenge of Irregular Access:**  Recognized that JDS SpMV can involve more complex and potentially less regular memory access patterns compared to ELL or CSR due to the jagged diagonal structure and the need to use the permutation array to determine the original row indices.
- **CUDA Kernel Implementation for JDS SpMV (`spMV_JDS`):**
    - **Kernel Structure:** Implemented the `spMV_JDS` CUDA kernel. This kernel takes `jdsRow`, `jdsCol`, `jdsVal`, `jdsPerm`, input vector `in_vect`, output vector `out_vect`, `non_zero_jds`, and `jds_max_nnz` as inputs.
    - **Thread Mapping and Jagged Diagonal Traversal:** The kernel is designed to launch threads equal to the total number of non-zero elements (`non_zero_jds`).  Each thread is responsible for processing one non-zero element. The kernel calculates which "jagged diagonal" each thread belongs to using the `jdsRow` array. It then uses the thread's ID and the jagged diagonal information to determine the correct row index from the `jdsPerm` array and the column index and value from `jdsCol` and `jdsVal`.
    - **Atomic Accumulation:** As with COO SpMV, atomicAdd operations are used to accumulate results into the output vector because multiple threads might contribute to the same row in the output vector due to the JDS format's structure and thread distribution.
- **Comparative Performance Analysis of COO, CSR, ELL, and JDS SpMV:**
    - **Benchmarking All Four GPU SpMV Kernels:**  Benchmarked `spMV_COO`, `spMV_CSR`, `spMV_ELL`, and the newly implemented `spMV_JDS` kernels, along with CPU dense SpMV (`spMV_CPU`).
    - **Performance Metrics:** Measured format creation times, SpMV operation times, total execution times (creation + operation), and storage sizes for each sparse format.
    - **Analysis Focus:**  Focused on comparing the performance trade-offs of each format, considering both format conversion overhead and SpMV kernel execution speed, and relating performance to the structural characteristics of each format.

**Code Implementation Details:**

- **Initialization Kernels (`init`, `initVect`):** Reuses the initialization kernels for the dense sparse matrix and the input vector.
- **Sparse Format Conversion Functions (`COO`, `CSR`, `ELL`, `JDS`):** Reuses the format conversion functions from Day 27 and Day 28.
- **SpMV Kernels:**
    - **`spMV_COO`, `spMV_CSR`, `spMV_ELL`:** Reuses the SpMV kernels from Day 24 and Day 28.
    - **`spMV_JDS` (New Kernel):** Implements the `spMV_JDS` kernel as described in "Key Learnings," handling the jagged diagonal traversal and row permutation logic to perform SpMV correctly.
- **CPU SpMV (`spMV_CPU`):** Reuses the CPU dense SpMV function for baseline comparison.
- **Timing and Output:**  The `main` function now includes timing and execution for JDS SpMV and prints out comprehensive results for all four sparse formats and CPU SpMV, including storage sizes, creation times, operation times, and total times.

**Results (Comprehensive Performance for N=2^12, SPARSITY=0.1):**
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

- **JDS Creation Time:** 9086.815000 ms
- **JDS Operation Time:** 155.978745 ms
- **JDS Total Time:** 9242.793745 ms

- **CPU spMV Time:** 30.488001 ms