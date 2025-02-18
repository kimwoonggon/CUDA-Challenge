# Day 1: Introduction to CUDA & Basic GPU Queries

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