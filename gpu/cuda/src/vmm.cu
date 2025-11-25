#include <cuda.h>
#include <cstdio>

#define CHECK(call) do {                                      \
  CUresult r = (call);                                      \
  if (r != CUDA_SUCCESS) {                                  \
      const char *s; cuGetErrorString(r, &s);               \
      printf("CUDA driver error %d: %s\n", r, s);           \
      return 1;                                             \
  }                                                         \
} while (0)

__global__ void fill_kernel(int * p, int n, int v) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
      p[i] = v;
  }
}

int main() {

    // cuInit to initialize the Driver API. This is done automatically
    // otherwise when using the Runtime API.
    CHECK(cuInit(0));

    CUdevice dev;
    CHECK(cuDeviceGet(&dev, 0));

    CUcontext ctx;
    CHECK(cuCtxCreate(&ctx, 0, dev));

    CUmemAllocationProp prop{};
    prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id   = dev;
    size_t granularity = 0;
    CHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    printf("granularity: %zu bytes\n", granularity);

    size_t bytes = 1 << 20; // 1 MiB
    bytes = ((bytes + granularity - 1) / granularity) * granularity;

    CUdeviceptr addr = 0;
    CHECK(cuMemAddressReserve(&addr, bytes, 0, 0, 0));
    printf("reserved VA at 0x%llx size %zu\n", (unsigned long long)addr, bytes);

    CUmemGenericAllocationHandle handle;
    CHECK(cuMemCreate(&handle, bytes, &prop, 0));

    CHECK(cuMemMap(addr, bytes, 0, handle, 0));

    CUmemAccessDesc access{};
    access.location = prop.location;
    access.flags    = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CHECK(cuMemSetAccess(addr, bytes, &access, 1));

    int * ptr = (int *)addr;
    int n = bytes / sizeof(int);
    fill_kernel<<<(n + 255)/256, 256>>>(ptr, n, 123);
    CHECK(cuCtxSynchronize());

    int host[4] = {0};
    CHECK(cuMemcpyDtoH(host, addr, sizeof(host)));
    printf("first four ints: %d %d %d %d\n", host[0], host[1], host[2], host[3]);

    CHECK(cuMemUnmap(addr, bytes));
    CHECK(cuMemRelease(handle));
    CHECK(cuMemAddressFree(addr, bytes));
    CHECK(cuCtxDestroy(ctx));
    return 0;
}

