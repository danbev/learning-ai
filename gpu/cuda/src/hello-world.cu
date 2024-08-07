#include <npp.h>
#include <stdio.h>

// CUDA Kernel function to print "Hello World"
__global__ void helloWorld() {
    printf("Hello World from GPU!\n");
}

int main() {
    cudaError_t cudaStatus = cudaGetLastError();
    // Print CUDA version
    int runtimeVersion = 0;
    cudaStatus = cudaRuntimeGetVersion(&runtimeVersion);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaRuntimeGetVersion failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }
    printf("CUDA Runtime version: %d\n", runtimeVersion);

    // Check for CUDA device
    int deviceCount = 0;
    cudaStatus = cudaGetDeviceCount(&deviceCount);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }
    printf("Number of CUDA devices: %d\n", deviceCount);

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable devices found\n");
        return 1;
    }
	


    // Print device information
    /*
    cudaDeviceProp prop;
    cudaStatus = cudaGetDeviceProperties(&prop, 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }
    printf("Device name: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    */

    helloWorld<<<1, 1>>>();

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

