#include <iostream>
#include <hip/hip_runtime.h>

// GPU kernel
__global__ void gpuKernel(int *flag) {
    *flag = 1; // Set flag to indicate the GPU kernel was executed
}

#define hipCheckError(ans) { hipAssert((ans), __FILE__, __LINE__); }
inline void hipAssert(hipError_t code, const char *file, int line, bool abort=true) {
   if (code != hipSuccess) {
      fprintf(stderr,"HIP Error: %s %s %d\n", hipGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int main() {
    int numDevices;
    hipGetDeviceCount(&numDevices);
    std::cout << "Number of devices: " << numDevices << std::endl;
    int *d_flag; // Device pointer
    int h_flag = 0; // Host flag

    // Allocate memory on the GPU
    hipCheckError(hipMalloc((void **)&d_flag, sizeof(int)));

    // Initialize flag on GPU
    hipCheckError(hipMemcpy(d_flag, &h_flag, sizeof(int), hipMemcpyHostToDevice));

    // Launch kernel (1 block, 1 thread)
    hipLaunchKernelGGL(gpuKernel, dim3(1), dim3(1), 0, 0, d_flag);
    hipDeviceSynchronize();

    // Copy back the flag to host
    hipCheckError(hipMemcpy(&h_flag, d_flag, sizeof(int), hipMemcpyDeviceToHost));

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
