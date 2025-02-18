# Day 7: GPU Accelerated Gaussian Blur with PyCUDA

**Objective:**
- **Gaussian Blur Kernel:** Developed a CUDA kernel for applying Gaussian blur to RGB images.
- **PyCUDA Integration:** Learned to use PyCUDA for kernel invocation and managing GPU memory.
- **Comparison with OpenCV:** Compared the results of the custom CUDA kernel with OpenCV's built-in Gaussian blur function.

The kernel applies a Gaussian filter to each pixel of an RGB image by computing a weighted average over a neighborhood defined by the blur radius and sigma (variance). The implementation uses PyCUDA to load the compiled PTX module and execute the kernel.

**Key Learnings:**
- **CUDA Kernel Development:** Wrote a kernel to perform Gaussian blur on RGB images, handling boundary conditions and computing the Gaussian weights.
- **PyCUDA Usage:** Learned how to allocate GPU memory, copy data to/from the device, and launch kernels using PyCUDA.
- **Image Processing Comparison:** Verified the custom kernel’s performance and output by comparing it with OpenCV's Gaussian blur implementation.

**Gaussian Blur Kernel:**
```c
extern "C" __global__ void imageBlur(unsigned char *img_in, unsigned char *img_out, int img_w, int img_h, int blur_radius, int sigma)
{
    int t_x = blockIdx.x * blockDim.x + threadIdx.x;
    int t_y = blockIdx.y * blockDim.y + threadIdx.y;

    if ( t_x < img_w && t_y < img_h)
    {
        float r = 0.0f;
        float g = 0.0f;
        float b = 0.0f;
        float val = 0.0f;

        for(int row = -blur_radius; row <= blur_radius; row++)
        {
            for(int col = -blur_radius; col <= blur_radius; col++)
            {
                int cur_row = t_y + row;
                int cur_col = t_x + col;

                if(cur_row < img_h && cur_row >= 0 && cur_col < img_w && cur_col >= 0)
                {
                    float gauss = expf(-(row * row + col * col) / (2.0 * sigma * sigma));

                    int channel = (cur_row * img_w + cur_col) * 3;
                    r += gauss * img_in[channel];
                    g += gauss * img_in[channel + 1];
                    b += gauss * img_in[channel + 2];

                    val += gauss;
                }
            }
        }

        int n = (t_y * img_w + t_x) * 3;
        img_out[n]   = (unsigned char)(roundf(r / val));
        img_out[n+1] = (unsigned char)(roundf(g / val));
        img_out[n+2] = (unsigned char)(roundf(b / val));
    }
}
```

**Results:**
![Output Plot](output_plot.png)