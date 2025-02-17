# 100 Days of GPU Challenge 🚀

Thanks to [Umar Jamil](https://github.com/hkproj/100-days-of-gpu) for organizing this challenge!

# Progress Log

## Day 1: Introduction to CUDA & Basic GPU Queries

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
```

## Day 2: Vector Addition Acceleration

**Objective:**  
- Accelerate vector addition on a very large vector (size = \(10^8\)) using CUDA.
- Utilize multi-stream initialization for efficient data loading.
- Employ a strided approach to optimize memory access.
- Compare GPU performance with CPU performance.

**Key Learnings:**  
- **Memory Allocation:** How to allocate large arrays on the GPU using `cudaMalloc`.
- **Strided Memory Access:** Implementing kernels that process elements in a strided loop so that all threads contribute evenly.
- **CUDA Streams:** Overlapping initialization of multiple arrays concurrently using separate CUDA streams.
- **Performance Boost:** Observing a significant speed-up on the RTX A4000 compared to CPU execution.

**Vector Addition Kernel:**  
```c
__global__ void addVector(int *a, int *b, int *c)
{
    int t_ID = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    if(t_ID < N)
    {
        for (int i = t_ID; i < N; i += stride)
        {
            c[i] = a[i] + b[i];
        }
    }
}
```


## Day 3: Matrix Multiplication Acceleration 

 **Objective:**  
 - Accelerate matrix multiplication for a square matrix of size \(N \times N\) (where \(N = 1000\)) using CUDA. 
 - Implement a 2D grid and block configuration tailored for matrix multiplication. 
 - Develop a GPU kernel to efficiently perform matrix multiplication in parallel.
 - Compare the performance of GPU-accelerated matrix multiplication against a CPU-based implementation. 

 **Key Learnings:**  
 - **2D Grid & Block Configuration for Matrices:** Understood how to design and configure 2D grids and blocks in CUDA to map effectively onto matrix dimensions. This configuration is crucial for parallelizing matrix operations, allowing different blocks and threads to work on different parts of the matrices concurrently.
 - **Efficient GPU Matrix Multiplication Kernel (`matMulGPU`):** Implemented a highly parallel CUDA kernel, `matMulGPU`, specifically for matrix multiplication. This kernel utilizes a 2D thread structure to calculate elements of the output matrix in parallel, leveraging the GPU's massive computational power.
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

## Day 4: CUDA Image Processing - RGB to Grayscale & Image Blurring

---

### Overview

1. **RGB to Grayscale Conversion:** Convert a color image to grayscale by applying a weighted sum on the RGB channels.
2. **Image Blurring:** Apply a box blur filter to a grayscale image by averaging neighboring pixel values.

Both tasks utilize 2D grid and block configurations for parallel processing and demonstrate CUDA memory management and kernel launching.

---

### Part 1: RGB to Grayscale Conversion

**Objective:**  
- Convert an RGB image to grayscale.
- Learn to manage multiple color channels and apply the weighted formula:  
  'Gray = 0.21 x R + 0.71 x G + 0.07 x B'
- Utilize CUDA kernels with 2D grid and block configurations to process each pixel in parallel.

**Key Learnings:**  
- Allocating device memory for both the RGB input and grayscale output.
- Configuring 2D grids/blocks (e.g., 32×32 threads) to cover the entire image.
- Transferring data between host and device efficiently.
- Integrating CUDA kernels with Python using shared libraries for easy invocation.

**RGB to Grayscale Kernel:**  
```c
__global__ void RGB2Grayscale(unsigned char *rgb_img_in, unsigned char *gray_img_out, int img_w, int img_h) 
{
    int t_x = blockIdx.x * blockDim.x + threadIdx.x;
    int t_y = blockIdx.y * blockDim.y + threadIdx.y;

    int strideX = gridDim.x * blockDim.x;
    int strideY = gridDim.y * blockDim.y;

    if(t_x < img_w && t_y < img_h)
    {
        for (int i = t_x; i < img_w; i += strideX)
        {
            for (int j = t_y; j < img_h; j += strideY)
            {
                int grayOffset = j * img_w + i;
                int rgbOffset = grayOffset * CHANNELS;

                unsigned char r = rgb_img_in[rgbOffset];
                unsigned char g = rgb_img_in[rgbOffset + 1];
                unsigned char b = rgb_img_in[rgbOffset + 2];

                gray_img_out[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
            }
        }
    }
}
```
### Part 2: Image Blurring

**Objective:**
- Apply a box blur filter to a grayscale image.
- Process each pixel by averaging values in a neighborhood defined by `BLUR_SIZE` (resulting in a window of size '2 x BLUR_SIZE + 1').
- Use strided loops within the kernel to cover the entire image and handle boundary conditions.

**Key Learnings:**
- Processing pixel neighborhoods (convolution) in parallel.
- Handling image boundaries by checking valid indices.
- Efficiently transferring data between host and device.

**Image Blurring Kernel:**
```c
__global__ void imageBlur(unsigned char *img_in, unsigned char *img_out, int img_w, int img_h)
{
    int t_x = blockIdx.x * blockDim.x + threadIdx.x;
    int t_y = blockIdx.y * blockDim.y + threadIdx.y;

    int strideX = gridDim.x * blockDim.x;
    int strideY = gridDim.y * blockDim.y;

    for (int i = t_x; i < img_w; i += strideX)
    {
        for (int j = t_y; j < img_h; j += strideY)
        {
            int pix_val = 0;
            int n_pixels = 0;

            for (int row = -BLUR_SIZE; row <= BLUR_SIZE; row++)
            {
                for (int col = -BLUR_SIZE; col <= BLUR_SIZE; col++)
                {
                    int cur_row = j + row;
                    int cur_col = i + col;

                    if (cur_row >= 0 && cur_row < img_h && cur_col >= 0 && cur_col < img_w)
                    {
                        pix_val += img_in[cur_row * img_w + cur_col];
                        n_pixels++;
                    }
                }
            }

            img_out[j * img_w + i] = (unsigned char)(pix_val / n_pixels);
        }
    }
}
```

## Day 5: GPU Accelerated Array Summation with CUDA Reduction

**Objective:**
- **Random Array Initialization:** Using CUDA's `curand` library to populate an array with random integers.
- **Parallel Reduction:** Summing array elements efficiently in parallel using shared memory.
- **Performance Benchmarking:** Comparing GPU performance against a CPU summation.

Managed memory is used to simplify data handling between the host and device, and CUDA events are utilized for precise timing of the GPU kernel execution.

**Key Learnings:**
- **CUDA Random Number Generation:** Learnt how to initialize random states with `curand_init` and generate random numbers using `curand`.

- **Parallel Reduction:** Implemented a reduction algorithm using shared memory and synchronization to sum large arrays efficiently.

- **Performance Benchmarking:** Measured and compared GPU and CPU execution times to highlight the benefits of parallel processing.

- **Memory Management:** Utilized CUDA Unified Memory for easier host-device data management.

**Partial Sum Kernel:**
```c
__global__ void calcSum(int *arr_in, int *arr_out, int N)
{   
    __shared__ int mem[2048];

    int next_i = blockDim.x * blockIdx.x * 2 + threadIdx.x;

    if (next_i < N) 
    {
        if (next_i + blockDim.x < N) 
        {
            mem[threadIdx.x] = arr_in[next_i] + arr_in[next_i + blockDim.x];
        } 
        else 
        {
            mem[threadIdx.x] = arr_in[next_i];
        }
    } 
    else 
    {
        mem[threadIdx.x] = 0;
    }
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if(threadIdx.x < stride)
        {
            mem[threadIdx.x] += mem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        arr_out[blockIdx.x] = mem[0];
    }
}
```


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
