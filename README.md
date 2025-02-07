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
