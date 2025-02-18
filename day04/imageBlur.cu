#include <stdio.h>
#include <math.h>

#define BLUR_SIZE 5

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

extern "C" void init(unsigned char *img_in, unsigned char *img_out, int img_w, int img_h)
{
    unsigned char *img_in_cuda, *img_out_cuda;
    
    size_t size = img_w * img_h * sizeof(unsigned char);

    cudaMalloc(&img_in_cuda, size);
    cudaMalloc(&img_out_cuda, size);

    cudaMemcpy(img_in_cuda, img_in, size, cudaMemcpyHostToDevice);
    cudaMemset(img_out_cuda, 0, size);

    dim3 n_threads(32, 32);
    dim3 n_blocks((img_w + n_threads.x - 1) / n_threads.x, (img_h + n_threads.y - 1) / n_threads.y);

    imageBlur<<<n_blocks, n_threads>>>(img_in_cuda, img_out_cuda, img_w, img_h);

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

    cudaMemcpy(img_out, img_out_cuda, size, cudaMemcpyDeviceToHost);

    cudaFree(img_in_cuda);
    cudaFree(img_out_cuda);
}
