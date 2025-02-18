#include <stdio.h>
#include <math.h>

#define CHANNELS 3
#define BLUR_SIZE 3

extern "C" __global__ void RGB2Grayscale(unsigned char *rgb_img_in, unsigned char *gray_img_out, int img_w, int img_h) 
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
                int rgbOffset = grayOffset*CHANNELS;

                unsigned char r = rgb_img_in[rgbOffset];
                unsigned char g = rgb_img_in[rgbOffset + 1];
                unsigned char b = rgb_img_in[rgbOffset + 2];

                gray_img_out[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
            }
        }
    }
}

extern "C" __global__ void intToFloat(unsigned char *input, float *output, int img_w, int img_h) {
    int t_x = blockIdx.x * blockDim.x + threadIdx.x;
    int t_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(t_x < img_w && t_y < img_h) {
        int i = t_y * img_w + t_x;
        output[i] = (float)input[i];
    }
}

extern "C" __global__ void imageBlur(unsigned char *img_in, unsigned char *img_out, int img_w, int img_h)
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

extern "C" __global__ void edgeDetectionLaplacianConv2D(float *input, float *output, float *filter, int img_w, int img_h) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < img_h && col < img_w) 
    {
        int radius = 1;
        float val = 0.0f;
        int f_size = 2 * radius + 1;

        for(int row_f = 0; row_f < f_size; row_f++) 
        {
            for(int col_f = 0; col_f < f_size; col_f++) 
            {
                int cur_row = row - radius + row_f;
                int cur_col = col - radius + col_f;

                if(cur_row >=0 && cur_row < img_h && cur_col >= 0 && cur_col < img_w) 
                {
                    val += filter[row_f * f_size + col_f] * input[cur_row * img_w + cur_col]; 
                }
            }
        }
        output[row * img_w + col] = val;
    }
}

extern "C" __global__ void normalizeFloat(float *img, int img_w, int img_h) 
{
    int t_x = blockIdx.x * blockDim.x + threadIdx.x;
    int t_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(t_x < img_w && t_y < img_h) 
    {
        img[t_y * img_w + t_x] /= 255.0f;
    }
}