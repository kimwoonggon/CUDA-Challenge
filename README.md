# 100 Days of GPU Challenge 🚀

Thanks to [Umar Jamil](https://github.com/hkproj/100-days-of-gpu) for organizing this challenge!

## Progress Log

### Day 1: Introduction to CUDA & Basic GPU Queries

**Key Learnings:**
- **CUDA Initialization & Device Query:**  
  Learned how to initialize the CUDA runtime and extract key GPU properties (e.g., GPU name, compute capability, SM count, warp size, global memory, and thread limits).

- **Basic Kernel Execution:**  
  Wrote and launched a minimal CUDA kernel to print a message from the GPU, confirming that my CUDA environment is set up correctly.

**GPU Query Example:**  
```c
cudaGetDevice(&deviceId);
cudaGetDeviceProperties(&props, deviceId);
printf("GPU Name: %s\n", props.name);
printf("Compute Capability: %d.%d\n", props.major, props.minor);
```c

### **Day 2**  
- Optimized **Vector Addition Acceleration** for a **10⁸**-sized vector on RTX A4000.  
- Developed two CUDA kernels:  
  - **Multi-stream initialization acceleration** for efficient data loading.  
  - **Strided approach for vector addition** to optimize memory access.  

### **Day 3**
- Optimized **Matrix Multiplication Acceleration** for a NxN matrix of size N = 1000.
- Developed two CUDA kernels:
  - **Multi-stream initialization acceleration** with 2D blocks and with a strided approach.
  - **2D strided memory access** for efficient multiplication.
- Explored **dim3()** for multi-dimensional thread and block configuration.

### **Day 4**
- Developed two CUDA kernels:
  - **RGB to Grayscale conversion**.
  - **Blurring of the Grayscale image**.
- Learnt Python bindings:
  - Converted CUDA kernels into a shared object (.so) file.
  - Integrated and executed the CUDA kernels from Python.

### **Day 5**
- Developed a kernel for summation of array acceleration using a partial sum approach.
- CPU Execution Time: 3.859000 ms
- GPU Execution Time: 0.831488 ms

### **Day 6**
- Developed a kernel for tile based Matrix Multiplication.
- Debugged the code so that it works even when the width of matrix is not perfectly divisible by width of tile.
- Results:-
  - Matrix Multiplication CPU time: 51694.223000 ms
  - Matrix Multiplication GPU time: 675.109863 ms
  - Tiled Matrix Multiplication GPU time: 277.286774 ms

### **Day 7**
- Recalled learnings for Day4.
- Developed a kernel for Gaussian Blur of RGB images.
- Learnt pycuda and its implementation.
- Compared the custom kernel's results with OpenCV's built-in functions.

### **Day 8**
- Developed a 2D convolutional filter kernel.
- Initialized filter weights using He Normal Initialization.
- Performance:-
  - GPU Convolution Execution Time: 0.01 ms
  - CPU Convolution Execution Time: 0.68 ms

### **Day 9**
- Recalled Learnings from Day 4 to convert an image to grayscale.
- Applied image blurring to smooth out noise, preventing the edge detection filter from capturing unwanted noise. This step also leveraged knowledge from Day 4.. Again, recalled Day 4 image blurring knowledge.
- Developed a 2D convolutional kernel with a Laplacian edge detection filter on the blurred image.

### **Day 10**
- Implemented a tiled convolutional kernels which handles cases where the input and output sizes of matrices does not match.
- Understood the concept of halo cells and shared memory.
- Results:
  - GPU Tiled Convolution Execution Time: 0.07 ms
  - CPU Tiled Convolution Execution Time: 0.66 ms

 ### **Day 11**
 - Learned about 3D arrays and how they are stored in the memory.
 - Implemented a simple stencil sweep kernel.
 - Results:
   - GPU time: 36.248322 ms
   - CPU time: 201.901993 ms

 ### **Day 12**
 - Implemented a tiled stencil sweep kernel.
 - Results for N = 256:
   - GPU time: 36.248322 ms
   - Tiled GPU time 18.025248 ms
   - CPU time: 201.901993 ms

 ### **Day 13**
 - Implemented a thread coarsed stencil sweep kernel.
 - Results for N = 256:
   - GPU time: 36.248322 ms
   - Tiled GPU time 18.025248 ms
   - Thread Coarsed GPU time: 7.199744 ms
   - CPU time: 201.901993 ms
