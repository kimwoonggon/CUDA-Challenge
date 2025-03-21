# Day 41: 2D Image Upsampling

**Objective:**
- **Implement Image Upsampling on GPU:** Develop a CUDA C++ program to perform 2D image upsampling using the parallel processing capabilities of the GPU.
- **Understand Nearest-Neighbor Interpolation:** Learn about the nearest-neighbor interpolation method as a basic technique for increasing the resolution of an image.
- **Utilize CUDA for Image Processing:** Gain experience in using CUDA to perform fundamental image processing operations in parallel.
- **Map Output Pixels to Input Pixels:** Understand how to calculate the corresponding input pixel coordinates for each output pixel during the upsampling process.

**Key Learnings:**
- **Image Upsampling:**
    - **Resolution Increase:** Understood the concept of increasing the spatial resolution (width and height) of a digital image.
    - **Interpolation:** Learned that upsampling typically involves interpolation techniques to estimate the color values of the newly created pixels.
- **Nearest-Neighbor Interpolation:**
    - **Simplest Method:** Recognized nearest-neighbor interpolation as the simplest upsampling method, where each new pixel in the output image takes the color value of the closest pixel in the original input image.
    - **Potential Artifacts:** Understood that while simple and fast, nearest-neighbor interpolation can lead to blocky or pixelated results, especially for larger upsampling factors.
- **CUDA for Parallel Image Processing:**
    - **Pixel-Level Parallelism:** Leveraged the inherent parallelism in image processing by assigning each output pixel's calculation to a separate thread on the GPU.
    - **Kernel Design for Image Operations:** Developed a CUDA kernel (`upsampling2D`) to perform the upsampling operation in parallel.
    - **Thread and Block Organization for Images:** Configured the thread and block dimensions to effectively cover the output image dimensions.
- **Coordinate Mapping in Upsampling:**
    - **Output to Input Mapping:** Learned how to map the coordinates of each pixel in the higher-resolution output image back to the coordinates of the corresponding pixel in the lower-resolution input image using the upsampling factor.

**Code Implementation Details:**

- **Includes:**
    - `stdio.h`, `stdlib.h`, `math.h`, `cuda_runtime.h`: Standard C/CUDA libraries for input/output, memory management, mathematical functions, and CUDA runtime functions.
- **`upsampling2D` Global Function:**
    - **Kernel Entry Point:** Defines the CUDA kernel function `upsampling2D` that will be executed on the GPU. The `extern "C"` is often used when this kernel might be called from code compiled with a C compiler or from languages like Python using CUDA wrappers.
    - **Thread Identification:** Obtains the row (`out_row`) and column (`out_col`) index of the current thread within the grid of threads, corresponding to a pixel in the output image.
    - **Output Bounds Check:** Checks if the current thread's coordinates are within the bounds of the output image dimensions (height `img_h * updim` and width `img_w * updim`).
    - **Input Coordinate Calculation:** Calculates the corresponding row (`in_row`) and column (`in_col`) in the input image by dividing the output coordinates by the upsampling factor (`updim`). Integer division effectively finds the nearest neighbor in the input image.
    - **Pixel Color Copying:** Iterates through the color channels (assuming RGB with 3 channels, where `c` goes from 0 to 2). For each channel, it copies the color value from the corresponding pixel in the input image (`input`) to the current pixel in the output image (`output`). The memory layout assumes a flattened array where pixels are stored sequentially (row by row, and channels within each pixel).
- **Usage:**
    - To use this kernel, you would typically:
        1.  Load the input image data into a host (CPU) memory array.
        2.  Allocate memory on the GPU for both the input and output images.
        3.  Copy the input image data from the host to the GPU.
        4.  Define the thread and block dimensions for the kernel launch, ensuring they cover the dimensions of the output image.
        5.  Launch the `upsampling2D` kernel on the GPU, passing the image dimensions, upsampling factor, and the pointers to the input and output image data on the GPU.
        6.  Synchronize the GPU to wait for the kernel to complete.
        7.  Copy the upsampled image data from the GPU back to the host.
        8.  Save or display the upsampled image.
