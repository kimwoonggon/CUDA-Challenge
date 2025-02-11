#include <stdio.h>
#include <math.h>

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

                if(cur_row < img_h && cur_row >=0 && cur_col < img_w && cur_col >= 0)
                {
                    float gauss = expf(-(row * row + col * col)/(2.0 * sigma * sigma));

                    int channel = (cur_row * img_w + cur_col) * 3;
                    r += gauss * img_in[channel];
                    g += gauss * img_in[channel + 1];
                    b += gauss * img_in[channel + 2];

                    val += gauss;
                }
            }
        }

        int n = (t_y * img_w + t_x) * 3;
        img_out[n] = (unsigned char)(roundf(r/val));
        img_out[n+1] = (unsigned char)(roundf(g/val));
        img_out[n+2] = (unsigned char)(roundf(b/val));
    }
}