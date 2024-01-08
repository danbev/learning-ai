#include <npp.h>
#include <stdio.h>

// CUDA Kernel function to print "Hello World"
__global__ void helloWorld() {
    printf("Hello World from GPU!\n");
}

int main() {
    helloWorld<<<1, 1>>>();

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "helloWorld launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }


    // Wait for GPU to finish before accessing on host
    // which is for printf to sync the output buffer with the host
    // or else the order of the output will be messed up.
    cudaDeviceSynchronize();

    printf("Hello World from CPU!\n");

    const char* gpu_name = nppGetGpuName();
    printf("GPU name: %s\n", gpu_name);
    int sms = nppGetGpuNumSMs();

    printf("Number of Streaming Multiprocessors: %d\n", sms);

    int threads_per_sm = nppGetMaxThreadsPerSM();
    printf("Max number of threads per SM: %d\n", threads_per_sm);
    int threads_per_block = nppGetMaxThreadsPerBlock();
    printf("Max number of threads per block: %d\n", threads_per_block);

    return 0;
}

