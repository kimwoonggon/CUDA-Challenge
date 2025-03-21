# Day 42: Exploring Warp-Level Synchronization in a GPU Monte Carlo Simulation

**Objective:**
- **Estimate Pi using Monte Carlo on GPU:** Implement a Monte Carlo simulation to estimate the value of Pi using CUDA, leveraging the parallel processing capabilities of the GPU.
- **Focus on Warp-Level Synchronization:** Specifically explore and understand the use of warp-level synchronization primitives in CUDA, such as `__shfl_down_sync`, to optimize the reduction step within the simulation.
- **Compare with Block-Level Synchronization:** Briefly contrast the performance and implementation of warp-level synchronization with traditional block-level synchronization using shared memory.

**Key Learnings:**
- **Monte Carlo Method for Pi Estimation:**
    - Understood the principle of using random sampling to estimate the value of Pi by determining the ratio of points falling inside a unit circle inscribed within a unit square.
- **CUDA Implementation of Monte Carlo:**
    - Learned how to parallelize the Monte Carlo simulation by assigning each GPU thread the task of generating a certain number of random samples.
    - Utilized the `curand` library for generating high-quality pseudorandom numbers on the GPU.
- **Warp-Level Synchronization (`__shfl_down_sync`):**
    - **Understanding Warps:** Grasped the concept of a warp as a group of 32 consecutive threads that execute in lock-step on a single core in an SIMT (Single Instruction, Multiple Thread) fashion.
    - **`__shfl_down_sync` Intrinsic:** Learned how to use the `__shfl_down_sync` intrinsic to efficiently share data between threads within the same warp without the need for explicit shared memory allocation or block-level synchronization. This intrinsic allows threads to read a value from a thread with a higher lane ID within the same warp.
    - **Warp-Based Reduction:** Implemented a parallel reduction algorithm within each warp using `__shfl_down_sync` to sum up the number of samples that fell inside the circle across all threads in the warp. This is a very efficient way to perform reductions for small numbers of elements (up to the warp size).
    - **Synchronization Mask:** Understood the role of the `0xFFFFFFFF` mask in `__shfl_down_sync`, indicating that all threads within the warp should participate in the shuffle operation.
- **Comparison with Block-Level Synchronization (`__syncthreads`):**
    - Briefly observed an alternative implementation (`monte_carlo_thread`) that uses shared memory and `__syncthreads` for a block-level reduction. This method involves more overhead but is necessary when communication or reduction needs to happen across all threads within a block, not just a warp.
- **Atomic Operations (`atomicAdd`):**
    - Understood the necessity of using atomic operations to safely update a global counter (`res`) from multiple threads without race conditions.

**Code Implementation Details:**

- **Includes:**
    - `curand_kernel.h`: Header file for the CUDA random number generation library.
    - `stdio.h`: Standard input/output library.
    - `cuda_runtime.h`: CUDA runtime API.
- **Defines:**
    - `N`: Total number of Monte Carlo samples to generate.
    - `THREADS`: Number of threads per block.
    - `SAMPLES_PER_THREAD`: Number of samples each thread will generate.
    - `TOTAL_THREADS`: Total number of threads launched.
    - `BLOCKS`: Number of thread blocks.
- **`monte_carlo_warp` Global Function:**
    - **Thread and Lane Identification:** Calculates the global thread index (`idx`) and the lane ID within the warp (`lane`).
    - **Random Number Generation:** Initializes a `curandState` for each thread using its global index as the seed.
    - **Monte Carlo Sampling:** Generates `samples_per_thread` random (x, y) pairs and checks if they fall within the unit circle (x^2 + y^2 <= 1). Increments a local counter `count` for each successful sample.
    - **Warp-Level Reduction:** Performs a parallel reduction within the warp using a series of `__shfl_down_sync` operations. In each step, threads with lower lane IDs add the `count` from threads with higher lane IDs within the same warp. This efficiently sums the counts across the warp.
    - **Atomic Update:** Only the master thread of each warp (lane ID 0) atomically adds the total count for that warp to the global result variable `res`.
- **`monte_carlo_thread` Global Function:** (For comparison)
    - Uses shared memory (`shared_counts`) to store the local count for each thread within the block.
    - Performs a block-level reduction by summing the counts from shared memory using the first thread in the block.
    - Atomically adds the block sum to the global result.
- **`main` Function:**
    - Allocates managed memory for the global result variables (`res_warp`, `res_thread`).
    - Initializes CUDA events to measure the execution time of both kernels.
    - Launches the `monte_carlo_warp` kernel with the calculated number of blocks and threads.
    - Measures the execution time.
    - Calculates the estimate of Pi using the result from the warp-synchronized kernel.
    - Repeats the process for the `monte_carlo_thread` kernel.
    - Frees the allocated memory and destroys the CUDA events.
    - Prints the estimated value of Pi and the execution time for both implementations.

**Results:**
- **Warp Sync:-** Pi 3142314622976.000000, Time 1.807520 ms
- **Thread Sync:-** Pi 3142314622976.000000, Time 1.636384 ms
