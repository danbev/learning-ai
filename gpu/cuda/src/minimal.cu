#include <cuda_runtime.h>
#include <stdio.h>

__global__ void myKernel() {}

int main() {
    cudaError_t err = cudaSuccess;
    int count = 0;

    int runtimeVersion = 0;

    err = cudaRuntimeGetVersion(&runtimeVersion);
    if (err != cudaSuccess) {
        printf("cudaRuntimeGetVersion failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("CUDA Runtime version: %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);

    int driverVersion = 0;
    err = cudaDriverGetVersion(&driverVersion);
    if (err != cudaSuccess) {
        printf("cudaDriverGetVersion failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("CUDA Driver version: %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10);

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
