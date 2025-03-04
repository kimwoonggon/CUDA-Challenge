<<<<<<< HEAD
# Day 27: Exploring ELL and JDS Sparse Matrix Formats in CUDA
=======
# Day 27: Exploring ELL and JDS Sparse Matrix Formats
>>>>>>> d7408d1 (Day27 updates)

**Objective:**
- **Implement ELL and JDS Format Conversion:**  Implement CUDA-based functions to convert a dense sparse matrix into Ellpack (ELL) and Jagged Diagonal Storage (JDS) formats.
- **Understand ELL and JDS Formats:** Learn the structure and characteristics of the ELL and JDS sparse matrix formats, understanding how they store sparse matrices and in what scenarios they might be advantageous.
- **Compare Storage Characteristics (Briefly):** Observe and briefly compare the storage sizes of ELL and JDS formats against COO and CSR formats (introduced in previous days) for a given sparse matrix. The main focus is on understanding ELL and JDS, not a deep dive into storage comparison as done previously.

**Key Learnings:**
- **Ellpack (ELL) Format:**
    - **Concept:** Learned about the Ellpack (or Ellpack-Itpack) format, which is efficient when the number of non-zero elements per row is relatively uniform or when a maximum number of non-zeros per row can be established without excessive padding.
    - **Structure:**  Understood that ELL format stores two arrays: `ellVal` and `ellCol`. Both have dimensions of `N` (number of rows) x `max_nnz` (maximum non-zeros in any row).  `ellVal[i][j]` stores the value of the j-th non-zero element in row `i`, and `ellCol[i][j]` stores the column index of that element. Rows with fewer than `max_nnz` non-zeros are padded with zeros in `ellVal` and typically placeholder column indices (like -1) in `ellCol`.
    - **Pros:**  Simple and regular structure, allowing for efficient vectorized or parallel access, especially suitable for GPU processing when row lengths are nearly uniform. No need for row pointers like CSR.
    - **Cons:**  Can be inefficient in storage if the number of non-zeros per row varies widely, as it is dictated by the maximum row length, leading to significant padding and wasted storage if many rows have far fewer non-zeros than `max_nnz`.
- **Jagged Diagonal Storage (JDS) Format:**
    - **Concept:** Learned about the Jagged Diagonal Storage (JDS) format, designed to improve memory access patterns and reduce storage overhead compared to ELL when row lengths are highly variable or skewed (some rows are much denser than others).
    - **Structure:** JDS is more complex than ELL and involves three main arrays and a permutation array:
        - `jdsVal`: Stores the non-zero values.
        - `jdsCol`: Stores the column indices of the non-zero values.
        - `jdsRow`: Row pointers, but used differently from CSR. `jdsRow[d]` points to the start of the d-th jagged diagonal in `jdsVal` and `jdsCol`.
        - `jdsPerm`: An array storing row permutations. Rows are reordered (permuted) based on the number of non-zero elements in descending order. This reordering is crucial for JDS efficiency.
    - **Jagged Diagonals:** JDS is based on the concept of "jagged diagonals". After row permutation (ordering rows by non-zero count), the first jagged diagonal consists of the first non-zero element from each row (if it exists), the second jagged diagonal consists of the second non-zero element from each row that has at least two non-zeros, and so on.
    - **Pros:** Can be more storage-efficient than ELL when row lengths are highly variable. Can offer better performance than COO for certain operations by improving memory access patterns after permutation and jagged diagonal organization.
    - **Cons:** More complex to implement and understand than COO, CSR, or ELL. The permutation step adds overhead, and the irregular structure can make some operations more complicated than on regular formats.

**Code Implementation Details:**

- **Dense Sparse Matrix Initialization (`init` kernel):** Reuses the `init` CUDA kernel from Day 23 and Day 24 to generate a dense sparse matrix on the GPU.
- **COO and CSR Conversion (`COO`, `CSR` functions):** Reuses the `COO` and `CSR` conversion functions from Day 23 and Day 24, utilizing cuSPARSE to convert the dense matrix to COO and CSR formats for storage size comparison.
- **ELL Conversion (`ELL` function):**
    - Implements the conversion from a dense sparse matrix to ELL format on the CPU (as format conversion is often done on the host before kernel execution).
    - First, it calculates `row_nnz` (number of non-zeros in each row) and determines `max_nnz` (maximum non-zeros in any row).
    - Allocates memory for `ellVal` and `ellCol` arrays of size `N` x `max_nnz`.
    - Iterates through the dense matrix, and for each row, stores the non-zero values and their column indices in the `ellVal` and `ellCol` arrays respectively, padding with zeros and -1 for column indices when a row has fewer than `max_nnz` non-zero elements.
- **JDS Conversion (`JDS` function):**
    - Implements the conversion to JDS format on the CPU.
<<<<<<< HEAD
    - **Row Permutation:** First, it calculates `row_nnz` for each row and then permutes the rows in descending order of `row_nnz`. The `jdsPerm` array stores this permutation.
    - **Jagged Diagonal Construction:**  Determines `jds_max_nnz` (which is the same as `max_nnz` after permutation, i.e., the non-zero count of the densest row).
=======
    - Row Permutation: First, it calculates `row_nnz` for each row and then permutes the rows in descending order of `row_nnz`. The `jdsPerm` array stores this permutation.
    - Jagged Diagonal Construction:  Determines `jds_max_nnz` (which is the same as `max_nnz` after permutation, i.e., the non-zero count of the densest row).
>>>>>>> d7408d1 (Day27 updates)
    - Allocates memory for `jdsRow`, `jdsCol`, `jdsVal`, and `jdsPerm`.
    - Populates the `jdsRow` array which acts as pointers to the start of each jagged diagonal. `jdsRow[d]` points to the index in `jdsVal` and `jdsCol` where the d-th jagged diagonal begins.
    - Extracts non-zero values and column indices into `jdsVal` and `jdsCol` in a jagged diagonal order, respecting the row permutation.

**Results (Storage Size Comparison for N=2^12, SPARSITY=0.1):**
<<<<<<< HEAD
- **Dense Matrix Storage Size: 64.00 MB
=======
- **Dense Matrix Storage Size:** 64.00 MB
>>>>>>> d7408d1 (Day27 updates)
- **Sparse Matrix Storage Size (COO):** 19.20 MB
- **Sparse Matrix Storage Size (CSR):** 12.82 MB
- **Sparse Matrix Storage Size (ELL):** 15.09 MB
- **Sparse Matrix Storage Size (JDS):** 12.82 MB