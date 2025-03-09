#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <limits.h>

extern "C" __global__ void maxPooling2D(int img_h, int img_w, int poolDim, unsigned int *input, unsigned int *output)
{
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;

    int in_row = out_row * poolDim;
    int in_col = out_col * poolDim;

    if(out_row < (img_h + poolDim - 1) / poolDim && out_col < (img_w + poolDim - 1) / poolDim)
    {
        for (int c = 0; c < 3; c++) 
        {
            int max = INT_MIN;
            
            for (int i = 0; i < poolDim; i++) 
            {
                for (int j = 0; j < poolDim; j++) 
                {
                    int cur_row = in_row + i;
                    int cur_col = in_col + j;
                    if (cur_row < img_h && cur_col < img_w) 
                    {
                        max = (input[((cur_row * img_w) + cur_col) * 3 + c] > max) ? input[((cur_row * img_w) + cur_col) * 3 + c] : max;
                    }
                }
            }
            int output_w = (img_w + poolDim - 1) / poolDim;
            output[((out_row * output_w) + out_col) * 3 + c] = max;
        }
    }
}
