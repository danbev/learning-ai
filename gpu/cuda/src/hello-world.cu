#include <stdio.h>

// CUDA Kernel function to print "Hello World"
__global__ void helloWorld() {
    printf("Hello World from GPU!\n");
}

int main() {
    helloWorld<<<1, 1>>>();

    // Wait for GPU to finish before accessing on host
    // which is for printf to sync the output buffer with the host
    // or else the order of the output will be messed up.
    cudaDeviceSynchronize();

    printf("Hello World from CPU!\n");

    return 0;
}

