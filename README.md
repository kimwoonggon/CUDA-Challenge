# **100 Days of GPU Challenge** 🚀  

Thanks to [Umar Jamil](https://github.com/hkproj/100-days-of-gpu) for organizing this!  

## **Progress Log**  

### **Day 1**  
- Implemented a basic CUDA kernel to print *"Hello from GPU"*.  
- Explored and analyzed my GPU's specifications.  

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
