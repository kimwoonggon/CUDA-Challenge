#include <stdio.h>
#include <math.h>

#define CHANNELS 3

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
                int rgbOffset = grayOffset*CHANNELS;

                unsigned char r = rgb_img_in[rgbOffset];
                unsigned char g = rgb_img_in[rgbOffset + 1];
                unsigned char b = rgb_img_in[rgbOffset + 2];

                gray_img_out[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
            }
        }
    }
}


extern "C" void init(unsigned char *img_in, unsigned char *img_out, int img_w, int img_h, float *time)
{
    unsigned char *rgb_img_in, *gray_img_out;

    size_t rgb_size = img_w * img_h * CHANNELS * sizeof(unsigned char);
    size_t gray_size = img_w * img_h * sizeof(unsigned char);

    cudaMalloc(&rgb_img_in, rgb_size);
    cudaMalloc(&gray_img_out, gray_size);

    cudaMemset(gray_img_out, 0, gray_size);

    cudaMemcpy(rgb_img_in, img_in, rgb_size, cudaMemcpyHostToDevice);

    dim3 n_threads(32, 32);
    dim3 n_blocks((img_w + n_threads.x - 1)/n_threads.x, (img_h + n_threads.y - 1)/n_threads.y);

    RGB2Grayscale<<<n_blocks, n_threads>>>(rgb_img_in, gray_img_out, img_w, img_h);

    cudaError_t err1 = cudaDeviceSynchronize();
    if (err1 != cudaSuccess)
    {
      printf("Asynchronous Error: %s\n", cudaGetErrorString(err1));
    }

    cudaError_t err2 = cudaGetLastError();
    if (err2 != cudaSuccess)
    {
      printf("Synchronous Error: %s\n", cudaGetErrorString(err2));
    }

    cudaMemcpy(img_out, gray_img_out, gray_size, cudaMemcpyDeviceToHost);

    cudaFree(rgb_img_in);
    cudaFree(gray_img_out);
}