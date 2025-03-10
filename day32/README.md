# Day 32: Introduction to Triton: Vector Addition

**Objective:**
- **Introduction to Triton:** Get started with Triton, a Python-based language and compiler for writing high-performance custom Deep Learning primitives.
- **Implement Vector Addition in Triton:** Write a basic Triton kernel to perform vector addition and execute it on the GPU.
- **Performance Comparison:** Compare the performance of the Triton-implemented vector addition against standard CPU (Python loop) and NumPy vector addition methods.

**Key Learnings:**
- **Triton Programming Model:**
    - **Just-In-Time (JIT) Compilation:** Learned that Triton uses a JIT compiler to translate Python-embedded kernels into highly optimized GPU machine code at runtime. This allows for writing performance-critical GPU code in a more accessible Python environment.
    - **Kernel Definition with `@triton.jit`:** Understood how to define Triton kernels using the `@triton.jit` decorator, which marks Python functions for Triton compilation.
    - **Triton Language (`triton.language` or `tt`):**  Introduced to the basic constructs of the Triton language (accessible via `triton.language` or the alias `tt`), including:
        - `tt.program_id(axis)`:  Retrieving the program ID (similar to CUDA thread block/grid ID) for parallel execution.
        - `tt.arange(start, end)`: Generating a range of indices.
        - `tt.load(pointer + offsets, mask)`: Loading data from memory with optional masking.
        - `tt.store(pointer + offsets, value, mask)`: Storing data to memory with optional masking.
        - Basic arithmetic operations (+, -, *, /).
    - **Grid Definition:** Learned how to specify the kernel launch grid using the `[block]` syntax in Python, where `block` is a lambda function that returns the grid shape based on metadata like `BLOCK_SIZE`.
- **GPU Tensor Management with PyTorch:**
    - **Interoperability with PyTorch:**  Demonstrated how Triton can seamlessly work with PyTorch tensors. PyTorch tensors are used to allocate memory on the GPU and pass data to Triton kernels.
    - **CUDA Synchronization:** Used `nn.cuda.synchronize()` from PyTorch to ensure that GPU operations (kernel launches) are completed before measuring time or copying data back to the CPU.
    - **CUDA Events for Timing:**  Employed PyTorch CUDA events (`torch.cuda.Event`) to accurately measure the execution time of the Triton kernel on the GPU.
- **Performance Benchmarking:**
    - **Comparison Baselines:**  Compared Triton's performance against:
        - **CPU Loop Addition:** A simple Python `for` loop performing element-wise addition on NumPy arrays, representing a basic CPU approach.
        - **NumPy Vectorized Addition:** NumPy's highly optimized vectorized addition, which leverages CPU vector instructions and is generally very efficient on the CPU.
    - **Performance Metrics:** Measured and compared the execution time of each method for a large vector addition task.

**Code Implementation Details:**

- **Triton Kernels:**
    - **`init(x, val, N, BLOCK_SIZE: tt.constexpr)`:**
        - `tid = tt.program_id(axis=0)`: Gets the program ID for the 0th axis (representing the linear thread ID in this 1D grid).
        - `offsets = tid * BLOCK_SIZE + tt.arange(0, BLOCK_SIZE)`: Calculates global memory offsets for each thread within a block. Each thread block processes a chunk of `BLOCK_SIZE` elements.
        - `mask = offsets < N`: Creates a mask to handle boundary conditions, ensuring that threads only operate on valid elements within the vector of size `N`.
        - `tt.store(x + offsets, val, mask=mask)`: Stores the `val` to the memory locations pointed to by `x + offsets`, applying the `mask` to prevent out-of-bounds writes.
    - **`add(a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE: tt.constexpr)`:**
        - Similar thread ID and offset calculation as `init`.
        - `a = tt.load(a_ptr + offsets, mask=mask)`: Loads elements from vector `a` into Triton registers, using masking.
        - `b = tt.load(b_ptr + offsets, mask=mask)`: Loads elements from vector `b`.
        - `c = a + b`: Performs element-wise addition in Triton registers.
        - `tt.store(c_ptr + offsets, c, mask=mask)`: Stores the result `c` into the output vector `c_ptr`, with masking.
- **Python `main()` Function:**
    - **Tensor Creation:** `a = nn.empty(N, dtype=nn.int32, device='cuda')`, `b = nn.empty(...)`, `c = nn.empty(...)`: Creates PyTorch tensors `a`, `b`, `c` of size `N`, integer type, and allocated on the CUDA device (GPU).
    - **Grid Definition:** `block = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)`: Defines a lambda function `block` that calculates the grid size (number of blocks) for kernel launch. `triton.cdiv` performs ceiling division.  The grid is 1-dimensional, with the number of blocks determined by dividing the vector size `N` by `BLOCK_SIZE` and rounding up.
    - **Kernel Launches:** `init[block](a, 1, N, BLOCK_SIZE=BLOCK_SIZE)`, `init[block](b, 2, N, BLOCK_SIZE=BLOCK_SIZE)`, `init[block](c, 0, N, BLOCK_SIZE=BLOCK_SIZE)`, `add[block](a, b, c, N, BLOCK_SIZE=BLOCK_SIZE)`: Launches the Triton kernels `init` and `add`. The `[block]` after the kernel name specifies the grid configuration. Keyword arguments are passed to the kernels.
    - **Timing with CUDA Events:**  Uses `torch.cuda.Event` to precisely measure the execution time of the `add` kernel.
    - **Result Verification:** `c_copy = c.cpu().numpy()`, `print("Success" if (c_copy == 3).all() else "Failure")`: Copies the result tensor `c` back to the CPU as a NumPy array and verifies if all elements are equal to 3 (1+2).
- **CPU and NumPy Baselines:**
    - Implements a CPU loop and NumPy vectorized addition for performance comparison.

**Results (Performance for N=2^24, BLOCK_SIZE=1024):**
- **Triton addition time:** 0.933 ms
- **CPU loop addition time:** 5051.784 ms
- **Numpy addition time:** 22.490 ms
