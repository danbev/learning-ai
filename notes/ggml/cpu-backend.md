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

In `ggml-cpu.c` the `ggml_compute_forward` contains all the operations that
are implemented for a backend:
```c++
static void ggml_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor) {
    GGML_ASSERT(params);

    if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {
        return;
    }

    // extra_buffer op?
    if (ggml_cpu_extra_compute_forward(params, tensor)) {
        return;
    }

    switch (tensor->op) {
        case GGML_OP_DUP:
            {
                ggml_compute_forward_dup(params, tensor);
            } break;
```
And the `ggml_compute_forward_dup` is declared in `ops.h`:
```c++
void ggml_compute_forward_dup(const struct ggml_compute_params * params, struct ggml_tensor * dst);
```
And the definition can be found in `opt.cpp`:
```c++
void ggml_compute_forward_dup(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
        ...
```

Now, so if we look further down in ggml-cpu.c we find functions like:
```c
void ggml_cpu_fp32_to_fp16(const float * x, ggml_fp16_t * y, int64_t n) {
    int64_t i = 0;
#if defined(__F16C__)
#if defined(__AVX512F__)
    for (; i + 15 < n; i += 16) {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        __m256i y_vec = _mm512_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256((__m256i *)(y + i), y_vec);
    }
#endif
    for (; i + 7 < n; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m128i y_vec = _mm256_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i *)(y + i), y_vec);
    }
    for (; i + 3 < n; i += 4) {
        __m128 x_vec = _mm_loadu_ps(x + i);
        __m128i y_vec = _mm_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storel_epi64((__m128i *)(y + i), y_vec);
    }
#elif defined(__riscv_zvfh)
    for (int vl; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(&x[i], vl);
        vfloat16m1_t vy = __riscv_vfncvt_f_f_w_f16m1(vx, vl);
        __riscv_vse16_v_f16m1((_Float16 *)&y[i], vy, vl);
    }
#endif
    for (; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(x[i]);
    }
}
```
These are functions that are set as function pointer in the traits
```c
    [GGML_TYPE_F16] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_fp16,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_f16,
        .vec_dot_type             = GGML_TYPE_F16,
        .nrows                    = 1,
},
```
So for ggml-cpu-ref I've removed all the arch specific code and just left the
core c implemenations.
I've been able to build this backend but when I run it I get:
```console
ggml_backend_load_best: /home/danbev/work/ai/llama.cpp/build-ref/bin/libggml-cpu-icelake.so score: 0
ggml_backend_load_best: /home/danbev/work/ai/llama.cpp/build-ref/bin/libggml-cpu-sandybridge.so score: 21
ggml_backend_load_best: /home/danbev/work/ai/llama.cpp/build-ref/bin/libggml-cpu-x64.so score: 1
ggml_backend_load_best: failed to load /home/danbev/work/ai/llama.cpp/build-ref/bin/libggml-cpu-ref.so
ggml_backend_load_best: /home/danbev/work/ai/llama.cpp/build-ref/bin/libggml-cpu-alderlake.so score: 128
ggml_backend_load_best: /home/danbev/work/ai/llama.cpp/build-ref/bin/libggml-cpu-sapphirerapids.so score: 0
ggml_backend_load_best: /home/danbev/work/ai/llama.cpp/build-ref/bin/libggml-cpu-sse42.so score: 5
ggml_backend_load_best: /home/danbev/work/ai/llama.cpp/build-ref/bin/libggml-cpu-haswell.so score: 64
ggml_backend_load_best: /home/danbev/work/ai/llama.cpp/build-ref/bin/libggml-cpu-skylakex.so score: 0
loading backend from /home/danbev/work/ai/llama.cpp/build-ref/bin/libggml-cpu-alderlake.so
load_backend: loaded CPU backend from /home/danbev/work/ai/llama.cpp/build-ref/bin/libggml-cpu-alderlake.so
register_backend: registered backend CPU (1 devices)
register_device: registered device CPU (12th Gen Intel(R) Core(TM) i7-1260P)
loading backend from /home/danbev/work/ai/llama.cpp/build-ref/bin/libggml-cpu-ref.so
load_backend: failed to load /home/danbev/work/ai/llama.cpp/build-ref/bin/libggml-cpu-ref.so
Available CPU backend variants:
  CPU - 12th Gen Intel(R) Core(TM) i7-1260P

  Only one CPU backend variant found. To enable CPU variants, rebuild with:
    cmake -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON
```
Lets check the undefined symbols:
```console
$ nm -D /home/danbev/work/ai/llama.cpp/build-ref/bin/libggml-cpu-ref.so | grep " U " | c++filt
                 U __assert_fail@GLIBC_2.2.5
                 U ceilf@GLIBC_2.2.5
                 U cosf@GLIBC_2.2.5
                 U __cxa_atexit@GLIBC_2.2.5
                 U __cxa_guard_abort@CXXABI_1.3
                 U __cxa_guard_acquire@CXXABI_1.3
                 U __cxa_guard_release@CXXABI_1.3
                 U erff@GLIBC_2.2.5
                 U expf@GLIBC_2.27
                 U expm1f@GLIBC_2.2.5
                 U fclose@GLIBC_2.2.5
                 U fgets@GLIBC_2.2.5
                 U floor@GLIBC_2.2.5
                 U floorf@GLIBC_2.2.5
                 U fmaxf@GLIBC_2.2.5
                 U fminf@GLIBC_2.2.5
                 U fopen@GLIBC_2.2.5
                 U fprintf@GLIBC_2.2.5
                 U fwrite@GLIBC_2.2.5
                 U getcpu@GLIBC_2.29
                 U getenv@GLIBC_2.2.5
                 U ggml_abort
                 U ggml_aligned_free
                 U ggml_aligned_malloc
                 U ggml_are_same_shape
                 U ggml_backend_buft_alloc_buffer
                 U ggml_backend_buft_is_host
                 U ggml_backend_cpu_buffer_type
                 U ggml_backend_cpu_reg
                 U ggml_backend_reg_dev_get
                 U ggml_blck_size
                 U ggml_can_repeat
                 U ggml_critical_section_end
                 U ggml_critical_section_start
                 U ggml_element_size
                 U ggml_free
                 U ggml_gemm_iq4_nl_8x8_q8_0
                 U ggml_gemm_q2_K_8x8_q8_K
                 U ggml_gemm_q4_0_8x8_q8_0
                 U ggml_gemm_q4_K_8x8_q8_K
                 U ggml_gemv_iq4_nl_8x8_q8_0
                 U ggml_gemv_q2_K_8x8_q8_K
                 U ggml_gemv_q4_0_8x8_q8_0
                 U ggml_gemv_q4_K_8x8_q8_K
                 U ggml_get_data_f32
                 U ggml_get_glu_op
                 U ggml_get_no_alloc
                 U ggml_get_type_traits
                 U ggml_get_unary_op
                 U ggml_init
                 U ggml_is_contiguous
                 U ggml_is_contiguous_1
                 U ggml_is_contiguous_channels
                 U ggml_is_empty
                 U ggml_is_quantized
                 U ggml_is_scalar
                 U ggml_log_internal
                 U ggml_nbytes
                 U ggml_n_dims
                 U ggml_nelements
                 U ggml_new_buffer
                 U ggml_new_tensor_1d
                 U ggml_nrows
                 U ggml_op_name
                 U ggml_quantize_mat_q8_0_4x8
                 U ggml_quantize_mat_q8_K_4x8
                 U ggml_rope_yarn_corr_dims
                 U ggml_row_size
                 U ggml_threadpool_params_default
                 U ggml_time_us
                 U ggml_type_name
                 U ggml_type_size
                 U ggml_unravel_index
                 U GOMP_barrier@GOMP_1.0
                 U GOMP_parallel@GOMP_4.0
                 U GOMP_single_start@GOMP_1.0
                 U __gxx_personality_v0@CXXABI_1.3
                 U log1pf@GLIBC_2.2.5
                 U log2@GLIBC_2.29
                 U logf@GLIBC_2.27
                 U memcpy@GLIBC_2.14
                 U memset@GLIBC_2.2.5
                 U omp_get_num_threads@OMP_1.0
                 U omp_get_thread_num@OMP_1.0
                 U powf@GLIBC_2.27
                 U pthread_getaffinity_np@GLIBC_2.32
                 U pthread_self@GLIBC_2.2.5
                 U pthread_setaffinity_np@GLIBC_2.34
                 U pthread_setschedparam@GLIBC_2.2.5
                 U putenv@GLIBC_2.2.5
                 U puts@GLIBC_2.2.5
                 U quantize_iq4_xs
                 U quantize_row_iq4_nl_ref
                 U quantize_row_mxfp4_ref
                 U quantize_row_q2_K_ref
                 U quantize_row_q3_K_ref
                 U quantize_row_q4_0_ref
                 U quantize_row_q4_1_ref
                 U quantize_row_q4_K_ref
                 U quantize_row_q5_0_ref
                 U quantize_row_q5_1_ref
                 U quantize_row_q5_K_ref
                 U quantize_row_q6_K_ref
                 U quantize_row_q8_0_ref
                 U quantize_row_q8_1_ref
                 U quantize_row_q8_K_ref
                 U quantize_row_tq1_0_ref
                 U quantize_row_tq2_0_ref
                 U roundf@GLIBC_2.2.5
                 U __sched_cpualloc@GLIBC_2.7
                 U __sched_cpufree@GLIBC_2.7
                 U sinf@GLIBC_2.2.5
                 U snprintf@GLIBC_2.2.5
                 U sqrtf@GLIBC_2.2.5
                 U __stack_chk_fail@GLIBC_2.4
                 U stat@GLIBC_2.33
                 U stderr@GLIBC_2.2.5
                 U strcmp@GLIBC_2.2.5
                 U strerror@GLIBC_2.2.5
                 U tanhf@GLIBC_2.2.5
                 U _Unwind_Resume@GCC_3.0
                 U ggml_backend_cpu_get_extra_buffer_types()
                 U operator delete(void*, unsigned long)@CXXABI_1.3.9
                 U operator new(unsigned long)@GLIBCXX_3.4
                 U std::__glibcxx_assert_fail(char const*, int, char const*, char const*)@GLIBCXX_3.4.30
                 U vtable for __cxxabiv1::__class_type_info@CXXABI_1.3
                 U vtable for __cxxabiv1::__si_class_type_info@CXXABI_1.3
                 U vtable for __cxxabiv1::__vmi_class_type_info@CXXABI_1.3
```
So ggml-quants.c defines quantize_row_iq4_nl_ref
```console
$ nm -D /home/danbev/work/ai/llama.cpp/build-ref/bin/libggml-base.so | grep quantize_row_iq4_nl_ref
00000000000a1486 T quantize_row_iq4_nl_ref
```
And we link with this library:
```console
$ ldd /home/danbev/work/ai/llama.cpp/build-ref/bin/libggml-cpu-ref.so
	linux-vdso.so.1 (0x00007d7e11190000)
	libggml-base.so => /home/danbev/work/ai/llama.cpp/build-ref/bin/libggml-base.so (0x00007d7e10f46000)
	libgomp.so.1 => /lib/x86_64-linux-gnu/libgomp.so.1 (0x00007d7e10ed9000)
	libstdc++.so.6 => /lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007d7e10c00000)
	libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007d7e10b17000)
	libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007d7e10eab000)
	libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007d7e10800000)
	/lib64/ld-linux-x86-64.so.2 (0x00007d7e11192000)
```
Lets add `LD_DEBUG=libs` so see if that can show something:
```console
loading backend from /home/danbev/work/ai/llama.cpp/build-ref/bin/libggml-cpu-ref.so
     70438:	/home/danbev/work/ai/llama.cpp/build-ref/bin/libggml-cpu-ref.so: error: symbol lookup error: undefined symbol: _Z39ggml_backend_cpu_get_extra_buffer_typesv (fatal)
load_backend: failed to load /home/danbev/work/ai/llama.cpp/build-ref/bin/libggml-cpu-ref.so
```
This turned out to be me missing to copy the contents of ggml-cpu.c to the
ggml-cpu-ref implementation (and a few other headers that I also missed)
But I got this working now and I have stripped out all the SIMD and arch specific
stuff from this backend. 

I've added a "mode" to test-backend-ops.cpp:
```console
Usage: ./build-ref/bin/test-backend-ops [mode] [-o <op,..>] [-b <backend>] [-p <params regex>] [--output <console|sql|csv>] [--list-ops] [--list-cpu-variants] [--show-coverage]
    valid modes:
      - test (default, compare with CPU backend for correctness)
      - grad (compare gradients from backpropagation with method of finite differences)
      - perf (performance evaluation)
      - support (probe backend operation support)
      - cpu-variants (test CPU variants against cpu-ref backend)
    op names for -o are as given by ggml_op_desc() (e.g. ADD, MUL_MAT, etc),
        optionally including the full test case string (e.g. "ADD(type=f16,ne=[1,1,8,1],nr=[1,1,1,1],nf=1)")
    --output specifies output format (default: console, options: console, sql, csv)
    --list-ops lists all available GGML operations
    --list-cpu-variants lists all available CPU backend variants
    --show-coverage shows test coverage
  cpu-variants mode options:
    --list lists available CPU variants on this system
    --variant <name> test specific CPU variant against cpu-ref backend
```
The idea is that it should be possible to first list the available cpu variants
and then choose on that will be tested against the cpu-ref backend.

One thing is that when ggml loads the dynamic libraries it will show the
shared libraries:
```console
ggml_backend_load_best: /home/danbev/work/ai/llama.cpp/build-ref/bin/libggml-cpu-icelake.so score: 0
ggml_backend_load_best: /home/danbev/work/ai/llama.cpp/build-ref/bin/libggml-cpu-sandybridge.so score: 21
ggml_backend_load_best: /home/danbev/work/ai/llama.cpp/build-ref/bin/libggml-cpu-x64.so score: 1
ggml_backend_load_best: /home/danbev/work/ai/llama.cpp/build-ref/bin/libggml-cpu-ref.so score: 10
ggml_backend_load_best: /home/danbev/work/ai/llama.cpp/build-ref/bin/libggml-cpu-alderlake.so score: 128
ggml_backend_load_best: /home/danbev/work/ai/llama.cpp/build-ref/bin/libggml-cpu-sapphirerapids.so score: 0
ggml_backend_load_best: /home/danbev/work/ai/llama.cpp/build-ref/bin/libggml-cpu-sse42.so score: 5
ggml_backend_load_best: /home/danbev/work/ai/llama.cpp/build-ref/bin/libggml-cpu-haswell.so score: 64
ggml_backend_load_best: /home/danbev/work/ai/llama.cpp/build-ref/bin/libggml-cpu-skylakex.so score: 0
load_backend: loaded CPU backend from /home/danbev/work/ai/llama.cpp/build-ref/bin/libggml-cpu-alderlake.so
```

A backend registry manages all the backends from the same family, for example
one might manage all CUDA devices, all OpenCL devices, or all CPU variants.

The names like 'icelake', 'sandybridge', etc are not stored anywhere as far as I
can tell. And all of these cpu variants share the same source code, and
the  name returned is simply "CPU":
```c++
static const char * ggml_backend_cpu_reg_get_name(ggml_backend_reg_t reg) {
    return "CPU";

    GGML_UNUSED(backend);
}
```
Instead of having this return "CPU" this could be a macro which is set to the
values of either just "CPU" or "CPU-icelake", "CPU-sandybridge".
These could then be printed by `test-backend-ops variant --list` when listing the
variants, and also the name could be used when selecting a specific variant to
test.

_wip_
