### How to start  
find /home/kimwoonggon/anaconda3/envs/llamacpp -name "cudnn.h"  
/home/kimwoonggon/anaconda3/envs/llamacpp/lib/python3.12/site-packages/nvidia/cudnn/include/cudnn.h  
nvcc -arch=sm_75 -I/path/to/cudnn/include cuDNNConv2D.cu -o cuDNNConv2D  

nvcc -arch=sm_75 -I/home/kimwoonggon/anaconda3/envs/llamacpp/lib/python3.12/site-packages/nvidia/cudnn/include -L/home/kimwoonggon/anaconda3/envs/llamacpp/lib/python3.12/site-packages/nvidia/cudnn/lib -lcudnn.9 cuDNNConv2D.cu -o cuDNNConv2D  

cd /home/kimwoonggon/anaconda3/envs/llamacpp/lib/python3.12/site-packages/nvidia/cudnn/lib  
** ln -s libcudnn.so.9 libcudnn.so **  
** nvcc -arch=sm_75 -I/home/kimwoonggon/anaconda3/envs/llamacpp/lib/python3.12/site-packages/nvidia/cudnn/include -L/home/kimwoonggon/anaconda3/envs/llamacpp/lib/python3.12/site-packages/nvidia/cudnn/lib -lcudnn cuDNNConv2D.cu -o cuDNNConv2D -Xlinker -rpath=/home/kimwoonggon/anaconda3/envs/llamacpp/lib/python3.12/site-packages/nvidia/cudnn/lib
**  


find /home/kimwoonggon/anaconda3/envs/llamacpp -name "libcudnn*.so*"
/home/kimwoonggon/anaconda3/envs/llamacpp/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn_heuristic.so.9
/home/kimwoonggon/anaconda3/envs/llamacpp/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn_engines_runtime_compiled.so.9
/home/kimwoonggon/anaconda3/envs/llamacpp/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn_cnn.so.9
/home/kimwoonggon/anaconda3/envs/llamacpp/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn_adv.so.9
/home/kimwoonggon/anaconda3/envs/llamacpp/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn_engines_precompiled.so.9
/home/kimwoonggon/anaconda3/envs/llamacpp/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn.so.9
/home/kimwoonggon/anaconda3/envs/llamacpp/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn_graph.so.9
/home/kimwoonggon/anaconda3/envs/llamacpp/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn_ops.so.9

# Day 30: Achieving Peak 2D Convolution Performance: Custom CUDA Kernel vs. cuDNN

**Objective:**
- **Outperform cuDNN (Specific Scenario):** Develop a highly optimized custom CUDA kernel for 2D convolution that can outperform the highly optimized cuDNN library in a specific scenario (small filter size, float data type, targeted GPU architecture).
- **Hardware-Aware Optimization:** Demonstrate the power of manual, hardware-aware optimization for achieving peak performance in CUDA, focusing on techniques like shared memory usage, tiling, and optimized thread mapping.
- **Analyze Trade-offs: Generality vs. Performance:** Understand the trade-offs between highly specialized, high-performance kernels and more general, library-based solutions like cuDNN, in terms of development effort, portability, and performance across different scenarios.

**Key Learnings:**
- **Hardware-Specific Kernel Optimization:**
    - **Tiling and Shared Memory:** Implemented a 2D convolution kernel that heavily utilizes shared memory and tiling to minimize global memory accesses. Input tiles are loaded into shared memory, allowing threads within a block to efficiently reuse data, significantly reducing bandwidth bottlenecks.
    - **Optimized Thread Mapping:** Designed thread mapping specifically for the tile size and filter size to maximize data reuse within shared memory and ensure efficient computation of the convolution for each output pixel. The thread block dimensions (32x32 in this case) and tile dimensions are carefully chosen to align with the GPU's architecture (likely for optimal occupancy and memory access patterns on the target GPU).
    - **Constant Memory for Filter:**  Utilized constant memory (`__constant__ float filter[FILTER_SIZE][FILTER_SIZE]`) to store the filter weights. Constant memory is cached and can be very fast for read-only data accessed by all threads, especially for small filters.
- **Comparison with cuDNN:**
    - **cuDNN as a Highly Optimized Library:** Acknowledged cuDNN as a highly optimized, production-ready library for deep learning primitives, including convolution. cuDNN is designed to be broadly applicable and automatically tunes for different GPUs and problem sizes.
    - **Specific Scenario Advantage of Custom Kernel:**  Recognized that while cuDNN is generally excellent, a carefully crafted, very specific custom kernel can potentially surpass cuDNN's performance in narrowly defined scenarios where the kernel is precisely tuned to the hardware and problem characteristics. This day’s work explores pushing the limits of performance in such a scenario.
- **Performance Benchmarking:**
    - **Metrics: Execution Time:** Benchmarked the execution time of the custom CUDA convolution kernel (`conv2D`), cuDNN's convolution implementation (`conv2D_cudnn`), and a CPU-based convolution implementation (`conv2D_cpu`).
    - **Performance Results Analysis:** Compared the execution times to demonstrate that the custom kernel achieves a faster execution time than cuDNN for the chosen problem configuration (input size, filter size, data type).

**Code Implementation Details:**

- **Convolution Kernels:**
    - **`conv2D` (Custom CUDA Kernel):**  Implements the highly optimized 2D convolution kernel utilizing shared memory, tiling, and optimized thread mapping as described in "Key Learnings".
        - Uses `__shared__ float N_shared[IN_TILE_DIM][IN_TILE_DIM]` to load input tiles into shared memory.
        - Employs a nested loop and constant memory filter to perform the convolution.
    - **`conv2D_cudnn`:** Uses the cuDNN library to perform 2D convolution. Leverages cuDNN's API to set up tensor descriptors, filter descriptors, convolution descriptors, find the best algorithm, allocate workspace, and execute the convolution.
- **CPU Convolution (`conv2D_cpu`):**  Implements a straightforward CPU-based 2D convolution for baseline performance comparison.
- **Initialization and Setup:**
    - **`init` Kernel:** Initializes the input matrix on the GPU with random data.
    - **`heNormal_kernel`:** Initializes the convolution filter weights using He Normal initialization (a common initialization technique in deep learning).
    - **Constant Memory Filter:** Loads the initialized filter weights into constant memory using `cudaMemcpyToSymbol(filter, managed_filter, f_size * sizeof(float))`.
- **Timing and Benchmarking:** Uses CUDA events to measure GPU kernel execution times and `clock()` for CPU timing.

**Results (Performance for N=2^10, M=2^11, RADIUS=1, FILTER_SIZE=3):**
- **GPU Convolution Execution Time:** 1.19 ms
- **cuDNN Convolution Execution Time:** 2.44 ms
- **CPU Convolution Execution Time:** 72.55 ms

**Results Analysis:**
- **Performance Superiority of Custom Kernel:** The custom CUDA convolution kernel (`1.19 ms`) achieves significantly faster execution time compared to cuDNN (`2.44 ms`) and CPU (`72.55 ms`) for this specific configuration.
- **GPU vs. CPU Performance:** Both GPU implementations (custom and cuDNN) dramatically outperform the CPU implementation, showcasing the power of GPUs for convolution operations.
- **Custom Kernel Optimizations Pay Off:** The carefully applied optimizations in the custom kernel (shared memory, tiling, specific thread mapping) successfully result in a performance gain over even a highly tuned library like cuDNN in this specific scenario.

**Conclusion - Specialization for Performance:**
- **Custom Kernel Beats cuDNN (Specific Case):**  Successfully demonstrated that a highly specialized custom CUDA kernel can outperform cuDNN for 2D convolution in a carefully chosen scenario. This highlights the potential for manual optimization to push performance boundaries for specific problems and hardware.
- **Trade-off: Generality vs. Performance:** It's crucial to recognize that this performance gain is achieved at the cost of generality. The custom kernel is likely highly tuned to the specific GPU architecture, input/output sizes, filter size, and data type used in this benchmark. It might not be as performant or even applicable to different scenarios. cuDNN, in contrast, is designed to be a general-purpose, broadly optimized library that works efficiently across a wider range of configurations and hardware.
