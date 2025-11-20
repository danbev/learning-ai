#include <stdio.h>

// Recall that __restrict__ is a hint to the compiler that the pointers do not
// overlap in memory.
__global__ void compact_kernel(const int * __restrict__ input,
                                     int * __restrict__ output,
                                     int * __restrict__ out_count,
                                     int n) {
    extern __shared__ int scan[];  // shared memory for flags + prefix sum

    int tid = threadIdx.x;

    int x = 0;
    int flag = 0;
    if (tid < n) {
        x = input[tid];
        flag = (x != 0);  // 1 = include, 0 = discard
    }

    // tore flags in shared memory
    scan[tid] = flag;

    // syncthread is a memory barrier, like a counter for the thread which needs
    // to be reached by all threads before any can proceed.
    __syncthreads();


    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int val = 0;
        if (tid >= offset) {
            val = scan[tid - offset];
        }

        __syncthreads();

        scan[tid] += val;

        __syncthreads();
    }

    if (tid < n && flag == 1) {
        // convert to zero based index
        int outIndex = scan[tid] - 1;
        output[outIndex] = x;
    }

    // The last prefix value contains the total number of kept elements, similar
    // to using vector.back() in C++ to get it.
    if (tid == blockDim.x - 1) {
        *out_count = scan[tid];
    }
}

int main() {
    const int N = 8;
    int h_in[N] = {3, 0, 5, 0, 2, 7, 0, 4};

    int * d_in    = nullptr;
    int * d_out   = nullptr;
    int * d_count = nullptr;

    cudaMalloc(&d_in,    N * sizeof(int));
    cudaMalloc(&d_out,   N * sizeof(int));
    cudaMalloc(&d_count, sizeof(int));

    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(N);
    dim3 grid(1);
    size_t shmemBytes = N * sizeof(int);  // shared memory size for scan[]

    compact_kernel<<<grid, block, shmemBytes>>>(d_in, d_out, d_count, N);
    cudaDeviceSynchronize();

    int h_out[N];
    int h_count = 0;
    cudaMemcpy(h_out,   d_out,   N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_count, d_count, sizeof(int),    cudaMemcpyDeviceToHost);

    printf("Kept %d elements:\n", h_count);
    for (int i = 0; i < h_count; ++i) {
        printf("%d ", h_out[i]);
    }
    printf("\n");

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_count);
    return 0;
}
