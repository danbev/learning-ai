#include <cuda_runtime.h>
#include <stdio.h>

#define N 20
#define STREAMS 4

__global__ void vector_add(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t instance;

    // Allocate host memory
    h_a = (float*)malloc(N * sizeof(float));
    h_b = (float*)malloc(N * sizeof(float));
    h_c = (float*)malloc(N * sizeof(float));

    // Allocate device memory
    cudaMalloc((void**)&d_a, N * sizeof(float));
    cudaMalloc((void**)&d_b, N * sizeof(float));
    cudaMalloc((void**)&d_c, N * sizeof(float));

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = i+1;
        h_b[i] = i+1;
    }

    // Create stream
    cudaStreamCreate(&stream);

    // Begin graph capture
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    // Add operations to the graph
    int streamSize = N / STREAMS;
    int shared_mem = 0;
    int block_dim = 256;
    int grid_dim = (streamSize + 255) / block_dim;

    for (int i = 0; i < STREAMS; i++) {
        int offset = i * streamSize;
        cudaMemcpyAsync(&d_a[offset], &h_a[offset], streamSize * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(&d_b[offset], &h_b[offset], streamSize * sizeof(float), cudaMemcpyHostToDevice, stream);

        vector_add<<<grid_dim, block_dim, shared_mem, stream>>>(&d_a[offset], &d_b[offset], &d_c[offset], streamSize);

        cudaMemcpyAsync(&h_c[offset], &d_c[offset], streamSize * sizeof(float), cudaMemcpyDeviceToHost, stream);
    }

    // End graph capture
    cudaStreamEndCapture(stream, &graph);

    // Create executable graph
    cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

    // Launch the graph
    cudaGraphLaunch(instance, stream);

    // Wait for the graph to complete
    cudaStreamSynchronize(stream);

    // Print results
    printf("Result array C:\n");
    for (int i = 0; i < N; i++) {
        printf("%.2f\n", h_c[i]);
    }

    // Clean up
    cudaGraphExecDestroy(instance);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
