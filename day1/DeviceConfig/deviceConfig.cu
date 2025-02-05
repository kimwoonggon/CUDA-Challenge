#include <stdio.h>

int main()
{
    int deviceId;
    cudaGetDevice(&deviceId);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);

    printf("GPU Name: %s\n", props.name);
    printf("Compute Capability Major: %d\n", props.major);
    printf("Compute Capability Major: %d\n", props.minor);
    printf("Number of Streaming Multiprocessors: %d\n", props.multiProcessorCount);
    printf("Warp Size: %d\n", props.warpSize);
    printf("Global Memory Size: %zu bytes\n", props.totalGlobalMem);
    printf("Maximum Threads per Block: %d\n", props.maxThreadsPerBlock);
    printf("Maximum Threads per Multiprocessor: %d\n", props.maxThreadsPerMultiProcessor);

    return 0;
}