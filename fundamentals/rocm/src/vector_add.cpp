#include <hip/hip_runtime.h>
#include <iostream>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int idx = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int N = 100000;
    float *A, *B, *C;
    hipMallocManaged(&A, N * sizeof(float));
    hipMallocManaged(&B, N * sizeof(float));
    hipMallocManaged(&C, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(i);
    }

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(vector_add, dim3(numBlocks), dim3(blockSize), 0, 0, A, B, C, N);

    hipDeviceSynchronize();

    std::cout << "C[0] = " << C[0] << std::endl;
    std::cout << "C[N-1] = " << C[N-1] << std::endl;

    hipFree(A);
    hipFree(B);
    hipFree(C);

    return 0;
}
