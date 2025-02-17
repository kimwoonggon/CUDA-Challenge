# Day 3: Matrix Multiplication Acceleration 

 **Objective:**  
 - Accelerate matrix multiplication for a square matrix of size \(N \times N\) (where \(N = 1000\)) using CUDA. 
 - Implement a 2D grid and block configuration tailored for matrix multiplication. 
 - Develop a GPU kernel to efficiently perform matrix multiplication in parallel.
 - Compare the performance of GPU-accelerated matrix multiplication against a CPU-based implementation. 

 **Key Learnings:**  
 - **2D Grid & Block Configuration for Matrices:** Understood how to design and configure 2D grids and blocks in CUDA to map effectively onto matrix dimensions. This configuration is crucial for parallelizing matrix operations, allowing different blocks and threads to work on different parts of the matrices concurrently.
 - **Efficient GPU Matrix Multiplication Kernel:** Implemented a highly parallel CUDA kernel, `matMulGPU`, specifically for matrix multiplication. This kernel utilizes a 2D thread structure to calculate elements of the output matrix in parallel, leveraging the GPU's massive computational power.
 - **Parallel Matrix Element Calculation:** Learned how to structure the `matMulGPU` kernel so that each thread computes a subset of elements in the result matrix `c`. The kernel employs nested loops and thread indexing to ensure that all necessary multiplication and addition operations are performed in a distributed and parallel manner across the GPU.
 - **Performance Speedup:** Observed a significant performance improvement when using the `matMulGPU` kernel compared to the traditional CPU-based matrix multiplication, demonstrating the effectiveness of GPU acceleration for computationally intensive matrix operations.

 **Matrix Multiplication Kernel:**  
 ```c 
 __global__ void matMulGPU(int *a, int *b, int *c)  
 {  
      int t_x = blockIdx.x * blockDim.x + threadIdx.x;  
      int t_y = blockIdx.y * blockDim.y + threadIdx.y;  

      int strideX = gridDim.x * blockDim.x;  
      int strideY = gridDim.y * blockDim.y + threadIdx.y;  

      int val = 0;  

      for (int  i = t_x; i < N; i += strideX)  
      {  
          for (int j = t_y; j < N; j += strideY)  
          {  
              for ( int k = 0; k < N; k++)  
              {  
                  val += a[i * N + k] * b[k * N + j];  
              }  
              c[i * N + j] = val;  
              val = 0;  
          }  
      }  
 }
```
