# Day 23: Sparse Matrix Formats - COO and CSR for Efficient Storage

**Objective:**
- **Explore Sparse Matrix Formats:** Investigate and implement the conversion of a dense matrix to sparse matrix formats, specifically Coordinate (COO) and Compressed Sparse Row (CSR) formats, to understand efficient storage for sparse data.
- **COO Format Implementation:**  Use the cuSPARSE library to convert a dense matrix to the COO format and analyze its storage requirements.
- **CSR Format Implementation:** Utilize the cuSPARSE library to convert the same dense matrix to the CSR format and compare its storage efficiency with COO.
- **Storage Size Comparison:**  Compare the storage size of the original dense matrix with its COO and CSR representations, highlighting the memory savings achieved by using sparse formats for matrices with low density.

**Key Learnings:**
- **Sparse Matrix Concept:** Learned about sparse matrices, which are matrices where most of the elements are zero. Sparse matrices are common in various applications, including graph algorithms, machine learning, and scientific computing. Efficiently storing and processing sparse matrices is crucial for reducing memory usage and computational cost.
- **Coordinate (COO) Format:** Understood the Coordinate (COO) format for sparse matrix storage. COO stores a sparse matrix as a list of (row, column, value) tuples for each non-zero element. It is simple to construct and understand but can be less efficient for certain operations compared to other formats.
- **Compressed Sparse Row (CSR) Format:** Learned about the Compressed Sparse Row (CSR) format, a more sophisticated sparse matrix format. CSR stores a sparse matrix using three arrays: `csrRowPtr`, `csrCol`, and `csrVal`. `csrRowPtr` points to the start of each row in `csrCol` and `csrVal`, `csrCol` stores the column indices of non-zero elements, and `csrVal` stores the values of the non-zero elements. CSR is generally more efficient than COO for matrix-vector multiplication and other operations that involve row-wise or column-wise access.
- **Storage Efficiency of Sparse Formats:**  Observed and quantified the significant reduction in storage size when using COO and CSR formats compared to storing the matrix in dense format, especially for sparse matrices where the sparsity level is high (most elements are zero).
- **cuSPARSE Library for Sparse Matrix Conversion:** Utilized the cuSPARSE library, NVIDIA's library for sparse linear algebra, to perform efficient conversion from dense to sparse matrix formats (COO and CSR) on the GPU. This showcased the ease of use and power of specialized libraries for sparse matrix operations on GPUs.

**Code Implementation Details:**

- **Dense Sparse Matrix Initialization (`init` kernel):** A CUDA kernel `init` is used to initialize a dense matrix on the GPU where elements have a `SPARSITY` probability of being non-zero, with non-zero values being random integers.
- **COO Conversion (`COO` function):** The `COO` function performs the dense-to-COO conversion using cuSPARSE functions:
    - `cusparseCreateDnMat` creates a descriptor for the dense matrix.
    - `cusparseCreateCoo` creates a descriptor for the COO sparse matrix, allocating memory for `cooRow`, `cooCol`, and `cooVal` arrays.
    - `cusparseDenseToSparse_bufferSize`, `cusparseDenseToSparse_analysis`, and `cusparseDenseToSparse_convert` functions perform the actual conversion using an optimized algorithm from cuSPARSE.
- **CSR Conversion (`CSR` function):** The `CSR` function is analogous to the `COO` function but converts the dense matrix to CSR format using `cusparseCreateCsr` and related cuSPARSE functions, allocating memory for `csrRowPtr`, `csrCol`, and `csrVal` arrays.
- **Storage Size Calculation:** The `main` function calculates and prints the storage size of the dense matrix and the converted COO and CSR matrices in Megabytes (MB), demonstrating the storage savings.

**Results (for N=2^12, SPARSITY=0.1):**
- **Dense Matrix Storage Size:** 64.00 MB
- **Sparse Matrix Storage Size (COO):** 19.19 MB
- **Sparse Matrix Storage Size (CSR):** 12.81 MB

**Results Analysis:**
- As shown in the results, for a matrix of size 4096x4096 (N=2^12) with a sparsity of 10% (`SPARSITY=0.1`), the sparse matrix formats (COO and CSR) significantly reduce storage space compared to the dense matrix format.
- COO format reduces the storage size to approximately 19.19 MB from the original 64.00 MB of the dense matrix. This is because COO only stores the non-zero values and their coordinates.
- CSR format achieves even better storage compression, further reducing the size to approximately 12.81 MB. CSR is generally more space-efficient than COO due to its compressed row pointer structure, especially when sparsity is high and non-zero values are clustered in rows.
- These results clearly demonstrate the importance of using sparse matrix formats for memory efficiency when dealing with matrices containing a large proportion of zero elements. Choosing the appropriate sparse format (like CSR for row-wise operations or COO for simpler construction) can lead to substantial savings in memory and potentially improved performance for sparse linear algebra operations.