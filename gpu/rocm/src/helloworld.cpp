#include <iostream>
#include <hip/hip_runtime.h>

// GPU kernel
__global__ void gpuKernel(int *flag) {
    *flag = 1; // Set flag to indicate the GPU kernel was executed
}

int main() {
    int *d_flag; // Device pointer
    int h_flag = 0; // Host flag

    // Allocate memory on the GPU
    hipMalloc((void **)&d_flag, sizeof(int));

    // Initialize flag on GPU
    hipMemcpy(d_flag, &h_flag, sizeof(int), hipMemcpyHostToDevice);

    // Launch kernel (1 block, 1 thread)
    hipLaunchKernelGGL(gpuKernel, dim3(1), dim3(1), 0, 0, d_flag);

    // Copy back the flag to host
    hipMemcpy(&h_flag, d_flag, sizeof(int), hipMemcpyDeviceToHost);

    // Check if the GPU kernel was executed
    if (h_flag == 1) {
        std::cout << "On GPU" << std::endl;
    }

    // Print message on CPU
    std::cout << "On CPU" << std::endl;

    // Free GPU memory
    hipFree(d_flag);

    return 0;
}
