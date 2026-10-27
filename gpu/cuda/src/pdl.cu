// Programmatic Dependent Launch (PDL), compute capability 9.0 (Hopper) and newer.
//
// Normally a kernel only starts when the previous kernel in the same stream has
// finished. With PDL the next kernel can start while the previous one is still
// running, do work that does not depend on it (setup, index math), and then wait
// for it with cudaGridDependencySynchronize() before reading its output.
//
// A requirement is that CUDA stream are used as this is how the producer/consumer
// relationship works. Notice that the sync functions don't take any arguments.
//
// Even with CUDA graphs there is a gap of a few microseconds between one kernel
// finishing and the next one's block running. The GPU can set up the grid and
// schedule blocks while the previous kernel is still running (with PDL that is).
//
// There are two sides:
//   producer (primary):   cudaTriggerProgrammaticLaunchCompletion()
//                         "the next kernel may launch now". Without this call the
//                         trigger happens implicitly when the producer's blocks exit.
//   consumer (secondary): launched with cudaLaunchAttributeProgrammaticStreamSerialization,
//                         calls cudaGridDependencySynchronize() before reading the
//                         producer's output. This waits until the producer has
//                         finished and its writes are visible.
//
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                                       \
    do {                                                                                       \
        cudaError_t err_ = (call);                                                             \
        if (err_ != cudaSuccess) {                                                             \
            fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                                \
        }                                                                                      \
    } while (0)

constexpr int                N        = 1024;
constexpr int                THREADS  = 256;
constexpr int                BLOCKS   = N / THREADS;
constexpr unsigned long long DELAY_NS = 2000000;

// Timestamps recorded by the kernels.
enum {
    T_PRODUCER_START,
    T_PRODUCER_END,
    T_CONSUMER_START,
    T_CONSUMER_SYNCED,
    T_COUNT
};

// Nanosecond timer that is the same on all SMs (unlike clock64()).
__device__ unsigned long long now_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__global__ void producer(int * data, unsigned long long * t, bool trigger_early) {
#if __CUDA_ARCH__ >= 900
    if (trigger_early) {
        // The consumer may launch now, even though we haven't written anything yet.
        cudaTriggerProgrammaticLaunchCompletion();
    }
#endif
    const unsigned long long start = now_ns();
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        t[T_PRODUCER_START] = start;
    }

    // Pretend to do a lot of work.
    while (now_ns() - start < DELAY_NS) {
    }

    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    data[i] = i + 1;

    __syncthreads();

    if (threadIdx.x == 0) {
        atomicMax(&t[T_PRODUCER_END], now_ns());
    }
}

__global__ void consumer(int * data, unsigned long long * t) {
    // Work that doesn't depend on the producer can go here, before the sync.
    const int i = blockIdx.x*blockDim.x + threadIdx.x;

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        t[T_CONSUMER_START] = now_ns();
    }

#if __CUDA_ARCH__ >= 900
    // Wait until the producer has finished and its writes are visible.
    // Returns immediately if this kernel was launched without PDL.
    cudaGridDependencySynchronize();
#endif

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        t[T_CONSUMER_SYNCED] = now_ns();
    }

    // Now it is safe to use the producer's output.
    data[i] = 2 * data[i];
}

static void run(const char * label, bool use_pdl, bool trigger_early,
                int * d_data, unsigned long long * d_t, cudaStream_t stream, bool print) {
    CUDA_CHECK(cudaMemsetAsync(d_data, 0, N * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(d_t,    0, T_COUNT * sizeof(unsigned long long), stream));

    producer<<<BLOCKS, THREADS, 0, stream>>>(d_data, d_t, trigger_early);
    CUDA_CHECK(cudaGetLastError());

    // The consumer opts in to starting early with this launch attribute.
    cudaLaunchAttribute attr = {};
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = use_pdl ? 1 : 0;

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim          = dim3(BLOCKS);
    cfg.blockDim         = dim3(THREADS);
    cfg.dynamicSmemBytes = 0;
    cfg.stream           = stream;
    cfg.attrs            = &attr;
    cfg.numAttrs         = 1;
    // launch the consumer kernel
    CUDA_CHECK(cudaLaunchKernelEx(&cfg, consumer, d_data, d_t));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    unsigned long long t[T_COUNT];
    CUDA_CHECK(cudaMemcpy(t, d_t, sizeof(t), cudaMemcpyDeviceToHost));

    if (!print) {
        return;
    }

    // Times in microseconds relative to the producer's start.
    const auto us = [&](int idx) { return (double) ((long long) t[idx] - (long long) t[T_PRODUCER_START]) / 1000.0; };
    printf("%s\n", label);
    printf("  producer: start %8.1f us, end       %8.1f us\n", us(T_PRODUCER_START), us(T_PRODUCER_END));
    printf("  consumer: start %8.1f us, past sync %8.1f us\n\n", us(T_CONSUMER_START), us(T_CONSUMER_SYNCED));
}

int main() {
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("%s, compute capability %d.%d\n\n", prop.name, prop.major, prop.minor);
    if (prop.major < 9) {
        printf("PDL requires compute capability 9.0 or newer\n");
        return 0;
    }

    int * d_data;
    unsigned long long * d_t;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_t, T_COUNT * sizeof(unsigned long long)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Warm up, so module loading doesn't show up in the first measurement.
    run("warm-up", true, true, d_data, d_t, stream, false);

    run("1. normal launch: consumer starts after the producer has finished",
        false, true, d_data, d_t, stream, true);
    run("2. PDL, producer never triggers: consumer starts when the producer's blocks exit",
        true, false, d_data, d_t, stream, true);
    run("3. PDL, producer triggers at its start: consumer starts early and waits in the sync",
        true, true, d_data, d_t, stream, true);

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_t));

    return 0;
}
