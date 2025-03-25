# Day 44: Achieving High-Performance Matrix Multiplication: Custom CUDA Tiling vs. cuBLAS

**Objective:**
- **Outperform cuBLAS (Specific Scenario):** Develop a highly optimized custom CUDA kernel for matrix multiplication using tiling that can outperform the highly optimized cuBLAS library in a specific scenario (particular matrix size, tile size, targeted GPU architecture).
- **Hardware-Aware Optimization for Matrix Multiplication:** Demonstrate the power of manual, hardware-aware optimization for achieving peak performance in CUDA, focusing on techniques like shared memory usage and strategic tiling.
- **Analyze Trade-offs: Generality vs. Performance in Linear Algebra:** Understand the trade-offs between highly specialized, high-performance kernels and more general, library-based solutions like cuBLAS, in terms of development effort, portability, and performance across different matrix sizes and hardware.

**Key Learnings:**
- **Hardware-Specific Kernel Optimization for Matrix Multiplication:**
    - **Tiling and Shared Memory:** Implemented a matrix multiplication kernel (`matMulTiled`) that heavily utilizes shared memory and tiling to minimize global memory accesses. Input matrix tiles are loaded into shared memory, allowing threads within a block to efficiently reuse data for multiple multiply-accumulate operations, significantly reducing bandwidth bottlenecks.
    - **Strategic Tile Size Selection:** Carefully chose the tile size (`MEM = 16`) to align with the GPU's architecture (likely aiming for optimal occupancy and efficient shared memory utilization on the target GPU). This size is crucial for balancing the amount of data loaded into shared memory and the number of threads that can operate on it.
    - **Optimized Thread Mapping within Tiles:** Designed the thread mapping within each block to efficiently compute the product of the loaded tiles from the input matrices. Each thread in the block computes a specific element of the output tile.
- **Comparison with cuBLAS:**
    - **cuBLAS as a Highly Optimized Library:** Acknowledged cuBLAS as a highly optimized, production-ready library for linear algebra operations, including matrix multiplication. cuBLAS is designed to be broadly applicable and automatically tunes for different GPUs and matrix sizes.
    - **Specific Scenario Advantage of Custom Kernel:** Recognized that while cuBLAS is generally excellent, a carefully crafted, very specific custom kernel can potentially surpass cuBLAS's performance in narrowly defined scenarios where the kernel's tiling strategy and parameters are precisely tuned to the hardware and matrix dimensions. This day’s work explores pushing the limits of performance in such a scenario with a matrix size of N=1951 and a tile size of MEM=16.
- **Performance Benchmarking:**
    - **Metrics: Execution Time:** Benchmarked the execution time of the custom CUDA matrix multiplication kernel (`matMulTiled`), cuBLAS's matrix multiplication function (`cublasSgemm`), a naive GPU-based implementation (`matMulGPU`), and a CPU-based implementation (`matMulCPU`).
    - **Performance Results Analysis:** Compared the execution times to demonstrate that the custom tiled kernel achieves a faster execution time than cuBLAS for the chosen matrix size and tile configuration.

**Code Implementation Details:**

- **Matrix Multiplication Kernels:**
    - **`matMulTiled` (Custom CUDA Kernel):** Implements the highly optimized tiled matrix multiplication kernel utilizing shared memory and strategic tiling as described in "Key Learnings".
        - Uses `__shared__ float mem_a[MEM][MEM]` and `__shared__ float mem_b[MEM][MEM]` to load tiles of matrices `a` and `b` into shared memory.
        - Employs nested loops to iterate through tiles and performs the multiplication of these tiles, accumulating the results.
        - Includes boundary conditions to handle matrix sizes that are not exact multiples of the tile size.
        - Uses `__syncthreads()` to ensure proper synchronization between threads within a block.
    - **`cublasSgemm`:** Uses the cuBLAS library's `cublasSgemm` function to perform highly optimized matrix multiplication.
    - **`matMulGPU`:** A naive GPU implementation of matrix multiplication with direct global memory access.
- **CPU Matrix Multiplication (`matMulCPU`):** Implements a standard CPU-based matrix multiplication for baseline comparison.
- **Initialization and Setup:**
    - **`init` Kernel:** Initializes the input matrices `a` and `b` on the GPU with random data.
- **Timing and Benchmarking:** Uses CUDA events to measure GPU kernel and cuBLAS function execution times and `clock()` for CPU timing.

**Results (Performance for N=1951, MEM=16):**
- **CPU time:** 61273.73 ms
- **GPU time:** 623.65 ms
- **Tiled GPU time:** 312.32 ms
- **cuBLAS time:** 370.65 ms

**Results Analysis:**
- **Performance Superiority of Custom Tiled Kernel:** The custom tiled CUDA matrix multiplication kernel (`312.32 ms`) achieves a faster execution time compared to cuBLAS (`370.65 ms`), the naive GPU kernel (`623.65 ms`), and the CPU implementation (`61273.73 ms`) for this specific configuration (N=1951, MEM=16).
- **GPU vs. CPU Performance:** Both GPU implementations (naive and tiled) significantly outperform the CPU implementation, highlighting the massive parallelism offered by GPUs for matrix multiplication.
- **Custom Kernel Optimizations Pay Off:** The carefully implemented tiling strategy and shared memory utilization in the custom kernel result in a noticeable performance improvement over cuBLAS in this particular scenario. This suggests that for specific matrix sizes and hardware, a well-tuned custom kernel can indeed surpass the performance of even highly optimized libraries.

**Conclusion - Specialization for Performance:**
- **Custom Tiled Kernel Beats cuBLAS (Specific Case):** Successfully demonstrated that a highly specialized custom CUDA kernel utilizing tiling can outperform the highly optimized cuBLAS library for matrix multiplication in a carefully chosen scenario. This underscores the potential for manual optimization to achieve peak performance by exploiting specific hardware and problem characteristics.
- **Trade-off: Generality vs. Performance:** It is important to note that this performance advantage of the custom kernel might not hold for all matrix sizes or different GPU architectures. cuBLAS is designed to be a general-purpose, broadly optimized library that often provides excellent performance across a wide range of configurations. Developing and tuning custom kernels for specific scenarios can yield the highest performance but requires significant effort and might sacrifice portability and generality. This day's work illustrates a case where that extra effort in specialization leads to a performance win over a general-purpose library.