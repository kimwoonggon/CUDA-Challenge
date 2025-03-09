# Day 31: Implementing 2D Max Pooling in CUDA

**Objective:**
- **Implement a CUDA Max Pooling Kernel:** Develop a CUDA kernel to perform 2D Max Pooling, a fundamental operation in Convolutional Neural Networks (CNNs).
- **Understand Max Pooling Operation:** Learn the concept of Max Pooling, its role in feature extraction and dimensionality reduction in CNNs, and how it operates on image data.
- **Explore Thread Mapping for Pooling:** Understand how to map CUDA threads to output pixels in a pooling operation and manage input data access within pooling regions.

**Key Learnings:**
- **2D Max Pooling Operation:**
    - **Concept:** Learned about 2D Max Pooling, a downsampling technique that reduces the spatial size of feature maps. It works by dividing the input image into a set of non-overlapping rectangular regions (pooling windows) and, for each region, outputting the maximum value from the inputs in that region.
    - **Purpose in CNNs:** Understood that Max Pooling serves several purposes in CNNs:
        - **Dimensionality Reduction:** Reduces the number of parameters and computations in the network, controlling overfitting.
        - **Translation Invariance:** Makes the features more robust to small shifts and distortions in the input, as it focuses on the presence of a feature rather than its precise location.
        - **Feature Hierarchy:** Helps to build a hierarchy of features by progressively reducing spatial resolution and increasing the receptive field of neurons deeper in the network.
- **CUDA Kernel for Max Pooling (`maxPooling2D`):**
    - **Kernel Design:** Implemented a `maxPooling2D` CUDA kernel that performs the max pooling operation in parallel on the GPU.
    - **Thread Mapping:**  Utilized a 2D grid of threads where each thread is responsible for computing one output pixel. The thread's `blockIdx.y`, `blockDim.y`, `threadIdx.y` and `blockIdx.x`, `blockDim.x`, `threadIdx.x` are used to determine the output row (`out_row`) and output column (`out_col`) of the pixel it will process.
    - **Input to Output Coordinate Mapping:**  Calculated the corresponding input coordinates (`in_row`, `in_col`) for each output pixel based on the `poolDim` (pooling dimension). For a `poolDim` of 2, each output pixel corresponds to a 2x2 region in the input.  Specifically, `in_row = out_row * poolDim` and `in_col = out_col * poolDim`.
    - **Boundary Checks:** Included a check (`if(out_row < (img_h + poolDim - 1) / poolDim && out_col < (img_w + poolDim - 1) / poolDim)`) to ensure that threads only process valid output pixels within the output image dimensions. The output dimensions are calculated as `(img_h + poolDim - 1) / poolDim` and `(img_w + poolDim - 1) / poolDim` to handle cases where input dimensions are not perfectly divisible by `poolDim` (using ceiling division).
    - **Channel Processing (Assumed 3 Channels):** The kernel is designed to process images with 3 color channels (e.g., RGB). It includes a loop `for (int c = 0; c < 3; c++)` to perform max pooling independently for each channel.
    - **Max Value Search within Pooling Window:** For each output pixel and each channel, the kernel iterates through the corresponding `poolDim` x `poolDim` region in the input image using nested loops. Inside these loops, it compares each input pixel value within the pooling window to the current `max` value, updating `max` if a larger value is found.  Initialization of `max` to `INT_MIN` ensures that the first pixel value encountered in the window becomes the initial maximum.
    - **Output Writing:** After finding the maximum value within the pooling window for a given channel, the kernel writes this `max` value to the corresponding location in the `output` array. The output array is also assumed to be in a channel-interleaved format, similar to the input.

**Code Implementation Details:**

- **Kernel Function (`maxPooling2D`):**
    - Takes parameters: `img_h` (image height), `img_w` (image width), `poolDim` (pooling dimension, e.g., 2 for 2x2 pooling), `input` (pointer to the input image data on the GPU), `output` (pointer to the output image data on the GPU).
    - Uses thread and block indices to determine output pixel coordinates.
    - Iterates through the pooling window in the input to find the maximum value for each channel.
    - Writes the maximum value to the corresponding output location.
- **Data Format:** Assumes input and output image data is stored in a linear array format, channel-interleaved. For an image of size `img_h` x `img_w` with 3 channels, the data layout is typically `[R11, G11, B11, R12, G12, B12, ..., R1M, G1M, B1M, R21, G21, B21, ..., RNM, GNM, BNM]`, where `Rij`, `Gij`, `Bij` are the Red, Green, Blue values for pixel at row `i` and column `j`.

**Usage and Potential Extensions:**

- **Setting up Input and Output:** To use this kernel, you would need to:
    1. Allocate memory on the GPU for both the `input` and `output` arrays.
    2. Copy your input image data to the `input` array on the GPU.
    3. Define the `img_h`, `img_w`, and `poolDim` parameters.
    4. Determine appropriate grid and block dimensions for launching the kernel. The provided code implicitly assumes a block size that is suitable for the thread indexing logic (e.g., block dimensions like 32x32 or similar). Grid dimensions should be calculated to cover the output image size.
    5. Launch the `maxPooling2D` kernel.
    6. Copy the `output` data back from the GPU to the host if needed for further processing or inspection.
