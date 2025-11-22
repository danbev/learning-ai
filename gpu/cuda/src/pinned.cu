#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__ \
                      << " : " << cudaGetErrorString(err) << "\n";     \
            std::exit(EXIT_FAILURE);                                   \
        }                                                              \
    } while (0)

__global__ void scale_kernel(float * data, int n, float factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= factor;
    }
}

int main() {
    const int N = 1 << 20;
    const size_t bytes = N * sizeof(float);

    // Pinned, page-locked memory allocation
    float * h_data = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_data, bytes));

    for (int i = 0; i < N; ++i) {
        h_data[i] = 1.0f;
    }

    float * d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUDA_CHECK(cudaMemcpyAsync(d_data, h_data, bytes, cudaMemcpyHostToDevice, stream));

    int block = 256;
    int grid = (N + block - 1) / block;
    scale_kernel<<<grid, block, 0, stream>>>(d_data, N, 2.0f);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(h_data, d_data, bytes, cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::cout << "h_data[0] = " << h_data[0] << "\n";

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFreeHost(h_data));  // <-- key difference

    return 0;
}

