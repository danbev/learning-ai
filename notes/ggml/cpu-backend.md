## GGML CPU backend notes

### Building
When building the CPU backend, depending on the CMake options provided it is
possible for the build to detect the host system that is being used and it will
include specific code for that system. For example, I'm currently on Linux
x86_64 and building with the following:

```console
$ cmake -S . -B build \
    -DGGML_CPU_REPACK=ON \
    -DGGML_BACKEND_DL=ON \
    -DCMAKE_BUILD_TYPE=Debug
$ cmake --build build
...
-- CMAKE_SYSTEM_PROCESSOR: x86_64
-- GGML_SYSTEM_ARCH: x86
-- Including CPU backend
...
-- x86 detected
-- Adding CPU backend variant ggml-cpu: -march=native
```
In `ggml/src/ggml-cpu/CMakeLists.txt` we can then find the following:

```console
    elseif (GGML_SYSTEM_ARCH STREQUAL "x86")
        message(STATUS "x86 detected")
        list(APPEND GGML_CPU_SOURCES
            ggml-cpu/arch/x86/quants.c
            ggml-cpu/arch/x86/repack.cpp
            )
```

### All CPU variantsD
It is also possible to build all CPU variants as shared objects libraries and
then the system can detect the most suitable variant at runtime based on the
current system.

### C-Only CPU backend
The cpu backend exists in ggml/src/ggml-cpu and it has a number of files and
variants for different features that an architecture may support. For testing
it would be nice to have a pure c implemenation that would just use basic c and
not system specific features or optimizations. The idea would then that this
backend could be used for testing the cpu variants against a known correct
base implementation.

So how this works is that we have the backend interface code defined in
ggml-cpu.cpp. This contains the function that ggml will call to discover the
backend and register it with the system. 
```c++
static const struct ggml_backend_i ggml_backend_cpu_i = {
    /* .get_name                = */ ggml_backend_cpu_get_name,
    /* .free                    = */ ggml_backend_cpu_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ ggml_backend_cpu_graph_plan_create,
    /* .graph_plan_free         = */ ggml_backend_cpu_graph_plan_free,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ ggml_backend_cpu_graph_plan_compute,
    /* .graph_compute           = */ ggml_backend_cpu_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .optimize_graph          = */ NULL,
};
```
And `ggml-backend.h` has functions that can be called to compute:
```c++
    GGML_API enum ggml_status     ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
    GGML_API enum ggml_status     ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
    GGML_API void                 ggml_backend_sched_synchronize(ggml_backend_sched_t sched);
```
```c++
enum ggml_status ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_compute != NULL);

    return backend->iface.graph_plan_compute(backend, plan);
}

enum ggml_status ggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    enum ggml_status err = ggml_backend_graph_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    return err;
}

enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    return backend->iface.graph_compute(backend, cgraph);
}
```
For example, this is the compute function for the cpu backend:
```c++
static enum ggml_status ggml_backend_cpu_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    struct ggml_backend_plan_cpu * cpu_plan = (struct ggml_backend_plan_cpu *)plan;

    return ggml_graph_compute(&cpu_plan->cgraph, &cpu_plan->cplan);

    GGML_UNUSED(backend);
}
```
And `ggml_graph_compute` can be found in `ggml-gpu.c`:
```c++
enum ggml_status ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan) {
    ggml_cpu_init();

    GGML_ASSERT(cplan);
    GGML_ASSERT(cplan->n_threads > 0);
    GGML_ASSERT(cplan->work_size == 0 || cplan->work_data != NULL);

    int n_threads                               = cplan->n_threads;
    struct ggml_threadpool * threadpool = cplan->threadpool;

    bool disposable_threadpool = false;
```

So we will need a cpu-c-only backend (or some better name), and we can start with
a copy of ggml-cpu.c for the actual computation code as it will me much the same
apart from any systems specific macros or optimizations. Then we could load
this as a shared object library in the same way as the cpu backend and register
it and make it available to use in test-backend-ops.cpp.

```console
#!/usr/bin/env bash

set -e

cmake -B build-c-only -DGGML_BACKEND_DL=ON \
    -DGGML_CPU_ALL_VARIANTS=ON \
    -DGGML_CPU_C_ONLY_BACKEND=ON \
    -DGGML_NATIVE=OFF \
    -DCMAKE_BUILD_TYPE=Debug

cmake --build build-c-only --target test-backend-ops -j8

./build-c-only/bin/test-backend-ops --list-cpu-variants
```
So with only the backend registered and adding a new option to `test-backend-ops.cpp`
the above generates:
```
ggml_backend_load_best: /home/danbev/work/ai/llama.cpp/build-c-only/bin/libggml-cpu-icelake.so score: 0
ggml_backend_load_best: /home/danbev/work/ai/llama.cpp/build-c-only/bin/libggml-cpu-sandybridge.so score: 21
ggml_backend_load_best: /home/danbev/work/ai/llama.cpp/build-c-only/bin/libggml-cpu-x64.so score: 1
ggml_backend_load_best: failed to find ggml_backend_score in /home/danbev/work/ai/llama.cpp/build-c-only/bin/libggml-cpu-c-only.so
ggml_backend_load_best: /home/danbev/work/ai/llama.cpp/build-c-only/bin/libggml-cpu-alderlake.so score: 128
ggml_backend_load_best: /home/danbev/work/ai/llama.cpp/build-c-only/bin/libggml-cpu-sapphirerapids.so score: 0
ggml_backend_load_best: /home/danbev/work/ai/llama.cpp/build-c-only/bin/libggml-cpu-sse42.so score: 5
ggml_backend_load_best: /home/danbev/work/ai/llama.cpp/build-c-only/bin/libggml-cpu-haswell.so score: 64
ggml_backend_load_best: /home/danbev/work/ai/llama.cpp/build-c-only/bin/libggml-cpu-skylakex.so score: 0
load_backend: loaded CPU backend from /home/danbev/work/ai/llama.cpp/build-c-only/bin/libggml-cpu-alderlake.so
register_backend: registered backend CPU (1 devices)
register_device: registered device CPU (12th Gen Intel(R) Core(TM) i7-1260P)
load_backend: loaded CPU-C-ONLY backend from /home/danbev/work/ai/llama.cpp/build-c-only/bin/libggml-cpu-c-only.so
register_backend: registered backend CPU-C-ONLY (1 devices)
register_device: registered device CPU-C-ONLY (CPU C-Only Backend (for testing))
Available CPU backend variants:
  CPU - 12th Gen Intel(R) Core(TM) i7-1260P
  CPU-C-ONLY - CPU C-Only Backend (for testing)
```


_wip_
