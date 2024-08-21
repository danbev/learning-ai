#include <hip/hip_runtime.h>
#include <iostream>
#include <random>

#define WIDTH 1024

__global__ void matrixMultiplication(float* A, float* B, float* C, int width) {
    int row = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int col = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if (row < width && col < width) {
        float sum = 0.0f;
        for (int i = 0; i < width; i++) {
            sum += A[row * width + i] * B[i * width + col];
        }
        C[row * width + col] = sum;
    }
}

void initMatrix(float* matrix, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < size; i++) {
        matrix[i] = dis(gen);
    }
}

int main() {
    const int size = WIDTH * WIDTH;
    size_t bytes = size * sizeof(float);

    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    // Allocate host memory
    h_A = new float[size];
    h_B = new float[size];
    h_C = new float[size];

    // Initialize host matrices
    initMatrix(h_A, size);
    initMatrix(h_B, size);

    // Allocate device memory
    hipMalloc(&d_A, bytes);
    hipMalloc(&d_B, bytes);
    hipMalloc(&d_C, bytes);

    // Copy host memory to device
    hipMemcpy(d_A, h_A, bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, bytes, hipMemcpyHostToDevice);

    // Setup execution configuration
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (WIDTH + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch kernel
    hipLaunchKernelGGL(matrixMultiplication, numBlocks, threadsPerBlock, 0, 0, d_A, d_B, d_C, WIDTH);

    // Wait for GPU to finish
    hipDeviceSynchronize();

    // Copy result back to host
    hipMemcpy(h_C, d_C, bytes, hipMemcpyDeviceToHost);

    // Verify result (check a few elements)
    std::cout << "Result verification (first few elements):" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << h_C[i] << " ";
    }
    std::cout << "..." << std::endl;

    // Free memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

    return 0;
}
