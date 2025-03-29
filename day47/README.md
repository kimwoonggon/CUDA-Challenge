# Day 47: Rotary Position Embedding

**Objective:**
- **Implement Rotary Position Embedding (RoPE) on GPU:** Develop a CUDA C++ program to implement the Rotary Position Embedding (RoPE) algorithm, which is used to encode positional information in Transformer models, and accelerate its computation using the GPU.
- **Understand the RoPE Algorithm:** Learn how RoPE encodes the position of tokens in a sequence using rotations in the embedding space.
- **Utilize CUDA for Sequence Data Processing:** Gain experience in using CUDA to efficiently process sequence data in parallel, a common task in natural language processing and other sequence-based applications.

**Key Learnings:**
- **Rotary Position Embedding (RoPE):**
    - Understood the concept of RoPE as a method to incorporate positional information into sequence embeddings by applying a rotation based on the token's position.
    - Learned that RoPE applies a different rotation to different dimensions of the embedding vector, and this rotation depends on the position of the token in the sequence.
    - Recognized the property that dot products between RoPE-encoded embeddings depend only on the relative distance between the tokens, which is beneficial for sequence modeling.
- **CUDA Implementation for Parallel RoPE:**
    - Developed a CUDA kernel (`ropeKernel`) to apply the RoPE rotation to a batch of sequences in parallel.
    - Utilized a 3D grid of thread blocks to process the batch dimension, sequence length dimension, and embedding dimension in parallel.
    - Understood how to calculate the indices for accessing the input embeddings and the precomputed cosine and sine tables.
- **Precomputation of Cosine and Sine Tables:**
    - Learned that the cosine and sine values for the rotations are precomputed based on the sequence length and embedding dimension and stored in tables on the GPU for efficient access during the kernel execution.
- **Efficient Memory Access:**
    - Used the `OFFSET3D` macro to calculate the linear index in the flattened 3D tensor representing the input embeddings, ensuring efficient access to the data on the GPU.
    - Processed the embedding dimensions in pairs (even and odd) as per the RoPE algorithm.

**Code Implementation Details:**

- **Includes:**
    - `stdio.h`: Standard input/output library.
    - `stdlib.h`: Standard general utilities library.
    - `math.h`: Mathematical functions like `sinf` and `cosf`.
    - `cuda_runtime.h`: CUDA runtime API for interacting with the GPU.
- **Defines:**
    - `THREADS`: Number of threads per block (set to 32).
    - `OFFSET3D(b, s, d, seq_len, dim)`: A macro to calculate the linear index in the flattened 3D tensor (batch, sequence length, dimension).
- **`ropeKernel` Global Function:**
    - This CUDA kernel is launched on the GPU to perform the RoPE operation in parallel.
    - **Thread and Block Mapping:** The kernel uses a 3D grid of blocks to cover the batch, sequence length, and half of the embedding dimension (since RoPE operates on pairs of dimensions). Threads within each block further parallelize the computation.
    - **Coordinate Calculation:** Calculates the batch index `b`, sequence position `s`, and the index `d` for the first of the two embedding dimensions being processed by the current thread.
    - **Boundary Checks:** Ensures that the thread indices are within the valid bounds of the input tensor.
    - **Even-Odd Dimension Processing:** Processes pairs of embedding dimensions (`d_even` and `d_odd`).
    - **Index Calculation:** Calculates the linear indices (`idx_even`, `idx_odd`) for the current embedding dimensions and the index (`idx_theta`) for the corresponding cosine and sine values in the precomputed tables.
    - **RoPE Rotation:** Applies the RoPE rotation to the even and odd dimensions of the embedding vector using the fetched cosine and sine values. The rotation formula is:
        - `output[even_dim] = input[even_dim] * cos(theta) - input[odd_dim] * sin(theta)`
        - `output[odd_dim] = input[even_dim] * sin(theta) + input[odd_dim] * cos(theta)`
- **`initRoPE` Function:**
    - This function runs on the CPU to initialize the input embeddings and precompute the cosine and sine tables.
    - **Input Initialization:** Initializes the input tensor `input` with some sample data using `sinf`.
    - **Cosine and Sine Table Initialization:** Calculates the rotation angle `theta` based on the sequence position `s`. Then, for each embedding dimension `d`, it calculates `cosf(theta)` and `sinf(theta)` and stores them in the `cos_table` and `sin_table` respectively. Note that the `theta` calculation here is a simplified example (`0.01f * s`). In practice, the frequency of rotation is often varied across the embedding dimensions.
- **`main` Function:**
    - Defines the batch size, sequence length, and embedding dimension.
    - Calculates the total size of the input and output tensors and the size of the cosine and sine tables.
    - Allocates managed memory on the GPU for the input embeddings (`d_input`), output embeddings (`d_output`), cosine table (`d_cos_table`), and sine table (`d_sin_table`).
    - Initializes these arrays using the `initRoPE` function on the host.
    - Defines the thread block dimensions (`THREADS` x `THREADS`) and calculates the grid dimensions to cover the entire input tensor.
    - Records the start and stop times using CUDA events to measure the execution time of the `ropeKernel`.
    - Launches the `ropeKernel` on the GPU with the calculated grid and block dimensions.
    - Synchronizes the device to wait for the kernel to complete.
    - Calculates and prints the execution time of the RoPE kernel.
    - Frees the allocated memory on the GPU and destroys the CUDA events.

**Results:**
- RoPE execution time: 1.12 ms
