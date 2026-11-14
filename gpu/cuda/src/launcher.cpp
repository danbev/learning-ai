#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK_CU(call) do {                                                     \
    CUresult err = (call);                                                      \
    if (err != CUDA_SUCCESS) {                                                  \
        const char *name = nullptr, *str = nullptr;                             \
        cuGetErrorName(err, &name);                                             \
        cuGetErrorString(err, &str);                                            \
        std::fprintf(stderr, "CUDA Driver error at %s:%d: %s (%s)\n",           \
                     __FILE__, __LINE__, name ? name : "?", str ? str : "?");   \
        std::exit(1);                                                           \
    }                                                                           \
} while (0)

// The main purpose of this file is to launch the vec_add kernel, either as
// src/vec_add.ptx or as compiles SASS as an argument like vec_add.cubin.
int main(int argc, char ** argv) {
    const char * module_path = (argc > 1) ? argv[1] : "src/vec_add.ptx";

    // Initialize the driver and bind GPU 0's primary context
    CHECK_CU(cuInit(0));

    CUdevice dev;
    CHECK_CU(cuDeviceGet(&dev, 0));

    CUcontext ctx;
    CHECK_CU(cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK_CU(cuCtxSetCurrent(ctx));

    // Load the PTX/CUBIN module and resolve the .entry symbol
    CUmodule mod;
    CHECK_CU(cuModuleLoad(&mod, module_path));

    CUfunction kernel;
    CHECK_CU(cuModuleGetFunction(&kernel, mod, "vec_add"));

    const uint32_t n = 1024;
    const size_t bytes = n * sizeof(float);

    std::vector<float> h_a(n);
    std::vector<float> h_b(n);
    std::vector<float> h_c(n, 0.0f);
    for (uint32_t i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }

    CUdeviceptr d_a;
    CUdeviceptr d_b;
    CUdeviceptr d_c;
    CHECK_CU(cuMemAlloc(&d_a, bytes));
    CHECK_CU(cuMemAlloc(&d_b, bytes));
    CHECK_CU(cuMemAlloc(&d_c, bytes));

    CHECK_CU(cuMemcpyHtoD(d_a, h_a.data(), bytes));
    CHECK_CU(cuMemcpyHtoD(d_b, h_b.data(), bytes));

    void * args[] = { &d_a, &d_b, &d_c, const_cast<uint32_t *>(&n) };

    const unsigned int block_size = 256;
    const unsigned int grid_size  = (n + block_size - 1) / block_size;

    CHECK_CU(cuLaunchKernel(
        kernel,
        grid_size,  1, 1,
        block_size, 1, 1,
        0,                  // dynamic shared memory bytes
        nullptr,            // stream (nullptr = default stream 0)
        args,               // kernelParams
        nullptr             // extra (alternative raw packed buffer)
    ));

    CHECK_CU(cuCtxSynchronize());
    CHECK_CU(cuMemcpyDtoH(h_c.data(), d_c, bytes));

    std::printf("c[0] = %.1f, c[%u] = %.1f\n", h_c[0], n - 1, h_c[n - 1]);

    CHECK_CU(cuMemFree(d_a));
    CHECK_CU(cuMemFree(d_b));
    CHECK_CU(cuMemFree(d_c));
    CHECK_CU(cuModuleUnload(mod));
    CHECK_CU(cuDevicePrimaryCtxRelease(dev));
    return 0;
}
