# Day 39: Jacobi Relaxation Calculation

**Objective:**
- **Implement Jacobi Relaxation on GPU:** Develop a CUDA C program to solve the Laplace equation on a 2D grid using the Jacobi relaxation method, leveraging the parallel processing capabilities of the GPU.
- **Understand Iterative PDE Solvers:** Learn about the Jacobi method as an iterative technique for approximating the solution to partial differential equations.
- **Utilize CUDA for Numerical Computation:** Gain experience in using CUDA C to perform large-scale numerical computations on the GPU, including memory management, kernel design, and synchronization.
- **Optimize for GPU Architecture:** Explore the use of shared memory and atomic operations within CUDA kernels to improve performance.

**Key Learnings:**
- **Jacobi Relaxation Method:**
    - **Iterative Approach:** Understood the iterative nature of the Jacobi method, where an initial guess for the solution is repeatedly updated until a convergence criterion is met.
    - **Laplace Equation:** Learned how the Jacobi method can be applied to numerically solve the Laplace equation, a fundamental equation in physics and engineering.
    - **Update Rule:** Implemented the core update rule of the Jacobi method, where the value at each interior grid point is replaced by the average of its four neighbors.
- **CUDA Implementation for Numerical Methods:**
    - **Kernel Design:** Developed CUDA kernels (`calcNextKernel` and `swapKernel`) to execute the Jacobi update rule and data swapping in parallel across the grid.
    - **Thread and Block Organization:** Configured the thread and block dimensions to efficiently map the 2D grid to the GPU's parallel architecture.
    - **Global and Shared Memory:** Utilized both global memory (for storing the grid data) and shared memory (`__shared__`) for efficient communication and reduction within thread blocks.
    - **Atomic Operations:** Employed atomic operations (`atomicMaxDouble`) to safely find the maximum error across all threads and blocks.
    - **Unified Memory:** Used `cudaMallocManaged` for allocating memory that can be accessed by both the CPU and the GPU, simplifying data management.
    - **CUDA Events for Timing:** Measured the execution time of the GPU computation using CUDA events for performance analysis.
- **Optimization Techniques:**
    - **Shared Memory Reduction:** Implemented a reduction pattern using shared memory within `calcNextKernel` to efficiently find the maximum error within each block before updating the global error.
    - **Avoiding Bank Conflicts:** While not explicitly shown in this basic example, understanding thread indexing and memory access patterns is crucial for avoiding bank conflicts in shared memory for further optimization.

**Code Implementation Details:**

- **Includes and Defines:**
    - `stdio.h`, `stdlib.h`, `math.h`, `cuda_runtime.h`: Standard C/CUDA libraries for input/output, memory management, mathematical functions, and CUDA runtime functions.
    - `OFFSET(j, i, m)`: A macro to calculate the linear index of a 2D grid point in a 1D array.
    - `THREADS`: Defines the number of threads per block in each dimension (32x32 in this case).
- **`atomicMaxDouble` Device Function:** Implements an atomic maximum operation for double-precision floating-point numbers in global memory. This is used to find the maximum error across all grid points.
- **`calcNextKernel` Global Function:**
    - Each thread calculates the new value for a single interior grid point using the Jacobi update rule.
    - It calculates the local error (the absolute difference between the new and old values).
    - It uses shared memory (`block_error`) to store the local error for each thread in the block.
    - A parallel reduction is performed within each block using shared memory to find the maximum error within that block.
    - Finally, the maximum error found in the block is atomically compared with the global error stored in `d_error`, and the global error is updated if the block's maximum error is larger.
- **`swapKernel` Global Function:** This kernel simply copies the new solution (`Anew`) back to the old solution array (`A`) for the next iteration.
- **`initialize` Function:** Sets the initial and boundary conditions for the problem. In this case, the top boundary of the grid is initialized to 1.0, and all other cells are initialized to 0.0.
- **`main` Function:**
    - Defines the grid dimensions (`n`, `m`), maximum number of iterations (`iter_max`), and the tolerance for convergence (`tol`).
    - Allocates memory on the GPU using `cudaMallocManaged` for the old solution (`A`), the new solution (`Anew`), and the global error (`d_error`).
    - Initializes the grid using the `initialize` function.
    - Configures the thread and block dimensions for the CUDA kernels.
    - Records the start time using CUDA events.
    - Enters the main iteration loop:
        - Resets the global error to 0.0.
        - Launches the `calcNextKernel` to compute the next iteration of the Jacobi method and calculate the error.
        - Synchronizes the device after the kernel launch.
        - Reads the global error from the device.
        - Launches the `swapKernel` to update the old solution with the new one.
        - Synchronizes the device after the kernel launch.
        - Prints the iteration number and error every 100 iterations.
    - Records the stop time and calculates the total runtime using CUDA events.
    - Prints the total execution time.
    - Frees the allocated memory on the GPU and destroys the CUDA events.

**Output:**
Jacobi relaxation Calculation: 8192 x 8192 mesh
Iteration:- 0, Error:- 0.250000
Iteration:- 100, Error:- 0.002397
Iteration:- 200, Error:- 0.001204
Iteration:- 300, Error:- 0.000804
Iteration:- 400, Error:- 0.000603
Iteration:- 500, Error:- 0.000483
Iteration:- 600, Error:- 0.000403
Iteration:- 700, Error:- 0.000345
Iteration:- 800, Error:- 0.000302
Iteration:- 900, Error:- 0.000269
total: 153.786118 s