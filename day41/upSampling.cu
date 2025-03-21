#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

extern "C" __global__ void upsampling2D(int img_h, int img_w, int updim, unsigned int *input, unsigned int *output)
{
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_row < img_h * updim && out_col < img_w * updim)
    {
        int in_row = out_row / updim;
        int in_col = out_col / updim;

        for (int c = 0; c < 3; c++) 
        {
            output[((out_row * img_w * updim) + out_col) * 3 + c] = input[((in_row * img_w) + in_col) * 3 + c];
        }
    }
}
