# Day 49: Softmax Activation Function

**Objective:**
- **Implement Softmax Function on GPU:** Develop a CUDA C++ program to implement the Softmax function, a crucial activation function in neural networks used for multi-class classification, and accelerate its computation using the GPU.
- **Understand the Softmax Function:** Learn the mathematical formula and the purpose of the Softmax function, which converts a vector of raw scores into a probability distribution over multiple classes.
- **Utilize CUDA for Parallel Softmax Computation:** Gain experience in using CUDA to efficiently compute the Softmax function in parallel for a batch of sequences and across the dimensions of each sequence.
- **Employ Shared Memory for Efficient Reduction:** Understand how shared memory can be used to optimize the calculation of the maximum value and the sum of exponentials, which are key steps in the Softmax computation, by performing parallel reductions.

**Key Learnings:**
- **Softmax Function:**
    - Understood the mathematical formula for the Softmax function: σ(z)ᵢ = exp(zᵢ) / Σⱼ exp(zⱼ), where z is a vector of input scores, and σ(z)ᵢ is the probability of the i-th class.
    - Learned that Softmax normalizes the output of a neural network layer into a probability distribution, where the probabilities sum up to 1.
    - Recognized the importance of numerical stability when implementing Softmax, particularly by subtracting the maximum value from the input scores before exponentiation to prevent potential overflow.
- **CUDA Implementation for Parallel Softmax:**
    - Developed a CUDA kernel (`softmaxKernel`) to perform the Softmax operation in parallel for a batch of sequences.
    - Utilized a 3D grid of thread blocks where the z-dimension corresponds to the batch, the y-dimension corresponds to the sequence length, and threads within each block handle the feature dimension (which represents the number of classes in this context).
- **Shared Memory for Reduction Operations:**
    - Employed shared memory (`max_val`, `sum_exp`, `buffer_max`, `buffer_sum`) to efficiently calculate the maximum value and the sum of exponentials across the feature dimension for each sequence.
    - Implemented a parallel reduction pattern to find the maximum value and then another parallel reduction to sum the exponentiated values across the threads in a block, which significantly reduces the number of global memory accesses.
- **Thread and Block Organization:**
    - Understood how the thread block size (`THREADS = 32`) is chosen and how the grid dimensions are calculated to cover the entire batch and sequence length. The x-dimension of the block corresponds to the dimension over which Softmax is computed.
- **Numerical Stability Implementation:**
    - Implemented the common trick of subtracting the maximum value along the dimension before exponentiation to improve the numerical stability of the Softmax function.

**Code Implementation Details:**

- **Includes:**
    - `stdio.h`: Standard input/output library.
    - `stdlib.h`: Standard general utilities library.
    - `math.h`: Mathematical functions like `expf` and `fmaxf`.
    - `cuda_runtime.h`: CUDA runtime API for interacting with the GPU.
- **Defines:**
    - `THREADS`: Number of threads per block (set to 32). This is often chosen to be a power of 2 and related to the warp size of the GPU for efficient parallel execution.
- **`softmaxKernel` Global Function:**
    - This CUDA kernel performs the Softmax operation in parallel.
    - **Thread and Block Mapping:** The kernel uses a 3D grid of blocks where `blockIdx.z` iterates over the batch, `blockIdx.y` iterates over the sequence length, and threads within each block (`threadIdx.x`) iterate over the embedding dimension (representing the classes).
    - **Boundary Check:** Ensures that the thread and block indices are within the valid bounds of the input tensor.
    - **Base Index Calculation:** Calculates the starting index (`base`) for the current sequence and batch in the flattened input tensor.
    - **Shared Memory for Maximum Value Reduction:**
        - Each thread calculates a partial maximum within its assigned chunk of the dimension.
        - These partial maximums are reduced across the threads in the block using shared memory (`buffer_max`) and a parallel reduction pattern to find the overall maximum value (`max_val`) for the current sequence.
    - **Numerical Stability:** The maximum value (`max_val`) is subtracted from each element in the sequence before exponentiation.
    - **Shared Memory for Sum of Exponentials Reduction:**
        - Each thread calculates the exponential of its shifted input value and accumulates a local sum.
        - These local sums are reduced across the threads in the block using shared memory (`buffer_sum`) and a parallel reduction pattern to find the total sum of exponentials (`sum_exp`) for the current sequence.
    - **Softmax Calculation:** Finally, each thread calculates the Softmax output by dividing the exponentiated and shifted value by the total sum of exponentials (`sum_exp`). The result is stored in the `d_output` tensor.
- **`initSoftmax` Function:**
    - This function runs on the CPU to initialize the input tensor with some sample data using `sinf`.
- **`main` Function:**
    - Defines the batch size, sequence length, and embedding dimension (number of classes).
    - Calculates the total size of the input and output tensors.
    - Allocates managed memory on the GPU for the input (`d_input`) and output (`d_output`).
    - Initializes the input using the `initSoftmax` function on the host.
    - Defines the thread block dimensions (`THREADS`) and calculates the grid dimensions to cover the entire batch and sequence length. The x-dimension of the block corresponds to the dimension over which Softmax is computed.
    - Records the start and stop times using CUDA events to measure the execution time of the `softmaxKernel`.
    - Launches the `softmaxKernel` on the GPU with the calculated grid and block dimensions.
    - Synchronizes the device to wait for the kernel to complete.
    - Calculates and prints the execution time of the Softmax kernel.
    - Frees the allocated memory on the GPU and destroys the CUDA events.

**Results:**
Softmax execution time: 0.97 ms