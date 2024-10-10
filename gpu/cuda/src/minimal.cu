#include <stdio.h>

__global__ void myKernel() {}

int main() {
    cudaError_t err = cudaSuccess;
    int count = 0;

    int runtimeVersion = 0;

    // Get CUDA runtime version
    err = cudaRuntimeGetVersion(&runtimeVersion);
    if (err != cudaSuccess) {
        printf("cudaRuntimeGetVersion failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("CUDA Runtime version: %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);


    err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        printf("cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("CUDA device count: %d\n", count);

    myKernel<<<1,1>>>();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("CUDA program ran successfully\n");
    return 0;
}
