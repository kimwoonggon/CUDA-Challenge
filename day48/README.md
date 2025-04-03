# Day 48: Layer Normalization

**Objective:**
- **Implement Layer Normalization on GPU:** Develop a CUDA C++ program to implement the Layer Normalization algorithm, a crucial technique in modern neural networks, and accelerate its computation using the GPU.
- **Understand Layer Normalization:** Learn the steps involved in Layer Normalization, including calculating the mean and variance across the features of each sample in a batch and then normalizing the features.
- **Utilize CUDA for Parallel Normalization:** Gain experience in using CUDA to perform Layer Normalization efficiently in parallel across a batch of sequences and the features within each sequence.
- **Employ Shared Memory for Efficient Reduction:** Understand how shared memory can be used to optimize the calculation of the mean and variance within each sequence by performing a parallel reduction.

**Key Learnings:**
- **Layer Normalization:**
    - Understood the purpose of Layer Normalization in stabilizing and accelerating the training of neural networks by normalizing the activations within each layer.
    - Learned that Layer Normalization computes the mean and variance across all the features (dimensions) for each sample in a batch independently.
    - Recognized the importance of the epsilon term for numerical stability.
    - Understood the affine transformation applied after normalization using the `gamma` (scale) and `beta` (shift) parameters.
- **CUDA Implementation for Parallel Layer Normalization:**
    - Developed a CUDA kernel (`layerNormKernel`) to perform Layer Normalization in parallel for a batch of sequences.
    - Utilized a 3D grid of thread blocks where the z-dimension corresponds to the batch, the y-dimension corresponds to the sequence length, and threads within each block handle the feature dimension.
- **Shared Memory for Reduction:**
    - Employed shared memory (`mean`, `variance`, `buffer`) to efficiently calculate the mean and variance of the features for each sequence.
    - Implemented a parallel reduction pattern to sum the feature values and their squares across the threads in a block, which significantly reduces the number of global memory accesses.
- **Thread and Block Organization:**
    - Understood how the thread block size (`THREADS = 32`) is chosen and how the grid dimensions are calculated to cover the entire batch and sequence length.
- **Affine Transformation:**
    - Implemented the scaling of the normalized values by `gamma` and the shifting by `beta`, allowing the network to learn the optimal scale and shift for each feature.

**Code Implementation Details:**

- **Includes:**
    - `stdio.h`: Standard input/output library.
    - `stdlib.h`: Standard general utilities library.
    - `math.h`: Mathematical functions like `sqrtf`.
    - `cuda_runtime.h`: CUDA runtime API for interacting with the GPU.
- **Defines:**
    - `THREADS`: Number of threads per block (set to 32). This is often chosen to be a power of 2 and related to the warp size of the GPU.
- **`layerNormKernel` Global Function:**
    - This CUDA kernel performs the Layer Normalization operation in parallel.
    - **Thread and Block Mapping:** The kernel uses a 3D grid of blocks where `blockIdx.z` iterates over the batch, `blockIdx.y` iterates over the sequence length, and threads within each block (`threadIdx.x`) iterate over the embedding dimension.
    - **Boundary Check:** Ensures that the thread and block indices are within the valid bounds of the input tensor.
    - **Base Index Calculation:** Calculates the starting index (`base`) for the current sequence and batch in the flattened input tensor.
    - **Shared Memory Allocation:** Declares shared memory variables `mean` and `variance` to store the calculated mean and variance for the current sequence, and a shared memory buffer `buffer` to facilitate the parallel reduction.
    - **Parallel Reduction for Mean and Variance:**
        - Each thread calculates a partial sum of the features and a partial sum of the squared features for the current sequence.
        - These partial sums are stored in the shared memory `buffer`.
        - A parallel reduction is performed using a loop with a halving stride to sum up the partial sums across all threads in the block.
    - **Mean and Variance Calculation:** The first thread in the block (with `d == 0`) calculates the mean and variance using the total sum and squared sum obtained from the reduction.
    - **Synchronization:** `__syncthreads()` is used to ensure that all threads in the block have completed their partial sums and the reduction before the mean and variance are used for normalization.
    - **Normalization and Affine Transformation:** Each thread normalizes its corresponding features by subtracting the mean and dividing by the standard deviation (calculated from the variance and epsilon). The normalized value is then scaled by the corresponding element in `d_gamma` and shifted by the corresponding element in `d_beta`.
- **`initLayerNorm` Function:**
    - This function runs on the CPU to initialize the input tensor and the `gamma` and `beta` parameters.
    - **Input Initialization:** Initializes the `input` tensor with some sample data using `sinf`.
    - **Gamma and Beta Initialization:** Initializes `gamma` to 1.0 and `beta` to 0.0, which represents an initial identity transformation after normalization (no scaling or shifting).
- **`main` Function:**
    - Defines the batch size, sequence length, embedding dimension, and the epsilon value for numerical stability.
    - Calculates the total size of the input and output tensors and the size of the `gamma` and `beta` parameter arrays.
    - Allocates managed memory on the GPU for the input (`d_input`), output (`d_output`), scale (`d_gamma`), and shift (`d_beta`) parameters.
    - Initializes these arrays using the `initLayerNorm` function on the host.
    - Defines the thread block dimensions (`THREADS`) and calculates the grid dimensions to cover the entire batch and sequence length. Note that the x-dimension of the block corresponds to the feature dimension.
    - Records the start and stop times using CUDA events to measure the execution time of the `layerNormKernel`.
    - Launches the `layerNormKernel` on the GPU with the calculated grid and block dimensions.
    - Synchronizes the device to wait for the kernel to complete.
    - Calculates and prints the execution time of the LayerNorm kernel.
    - Frees the allocated memory on the GPU and destroys the CUDA events.

**Result:**
LayerNorm execution time: 0.72 ms