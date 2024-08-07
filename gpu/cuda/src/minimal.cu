#include <stdio.h>

__global__ void myKernel() {}

int main() {
    cudaError_t err = cudaSuccess;
    int count = 0;

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
