#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Declaration of the upsampling kernel
extern "C" __global__ void upsampling2D(int img_h, int img_w, int updim, unsigned int *input, unsigned int *output);

// Helper function to check CUDA errors
#define checkCudaErrors(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// Simple function to create a sample image (checkerboard pattern)
void createSampleImage(unsigned int *image, int height, int width) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            // Create a checkerboard pattern
            int isWhite = ((row / 8) % 2 == 0) ^ ((col / 8) % 2 == 0);
            
            // Set RGB values
            if (isWhite) {
                image[(row * width + col) * 3 + 0] = 255; // R
                image[(row * width + col) * 3 + 1] = 255; // G
                image[(row * width + col) * 3 + 2] = 255; // B
            } else {
                image[(row * width + col) * 3 + 0] = 0;   // R
                image[(row * width + col) * 3 + 1] = 0;   // G
                image[(row * width + col) * 3 + 2] = 0;   // B
            }
        }
    }
}

// Function to save image data to a PPM file (simple image format)
void savePPM(const char *filename, unsigned int *image, int height, int width) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error: Unable to open file %s for writing\n", filename);
        return;
    }
    
    // Write PPM header
    fprintf(fp, "P6\n%d %d\n255\n", width, height);
    
    // Allocate buffer for pixel data
    unsigned char *buffer = (unsigned char *)malloc(3 * width * height);
    
    // Convert from unsigned int to unsigned char
    for (int i = 0; i < height * width * 3; i++) {
        buffer[i] = (unsigned char)image[i];
    }
    
    // Write pixel data
    fwrite(buffer, 3, width * height, fp);
    
    // Clean up
    free(buffer);
    fclose(fp);
    printf("Image saved to %s\n", filename);
}

int main() {
    // Image dimensions
    int img_width = 64;
    int img_height = 64;
    int upsampling_factor = 4;
    
    // Calculate output image dimensions
    int out_width = img_width * upsampling_factor;
    int out_height = img_height * upsampling_factor;
    
    // Allocate memory for input and output images on host
    size_t input_size = img_width * img_height * 3 * sizeof(unsigned int);
    size_t output_size = out_width * out_height * 3 * sizeof(unsigned int);
    
    unsigned int *h_input = (unsigned int *)malloc(input_size);
    unsigned int *h_output = (unsigned int *)malloc(output_size);
    
    // Create a sample input image
    createSampleImage(h_input, img_height, img_width);
    
    // Save the input image
    savePPM("input_image.ppm", h_input, img_height, img_width);
    
    // Allocate device memory for input and output images
    unsigned int *d_input, *d_output;
    checkCudaErrors(cudaMalloc((void **)&d_input, input_size));
    checkCudaErrors(cudaMalloc((void **)&d_output, output_size));
    
    // Copy input image from host to device
    checkCudaErrors(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));
    
    // Set up execution parameters for kernel
    dim3 threads(16, 16);
    dim3 blocks((out_width + threads.x - 1) / threads.x, 
                (out_height + threads.y - 1) / threads.y);
    
    printf("Launching upsampling kernel with grid: (%d, %d), block: (%d, %d)\n", 
           blocks.x, blocks.y, threads.x, threads.y);
    
    // Launch the upsampling kernel
    upsampling2D<<<blocks, threads>>>(img_height, img_width, upsampling_factor, d_input, d_output);
    
    // Check for kernel launch errors
    checkCudaErrors(cudaGetLastError());
    
    // Wait for kernel to complete
    checkCudaErrors(cudaDeviceSynchronize());
    
    // Copy result back to host
    checkCudaErrors(cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost));
    
    // Save the output image
    savePPM("upsampled_image.ppm", h_output, out_height, out_width);
    
    // Free memory
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    
    printf("Upsampling completed successfully.\n");
    
    return 0;
}
