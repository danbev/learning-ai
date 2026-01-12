### Flash Attention batched-bench issue
Currently when running the llama-batch-bench example with flash attention enabled
(the default) it fails with the following error:
```console
export CUDA_LAUNCH_BLOCKING=1
cuda-gdb --args ./${build_dir}/bin/llama-batched-bench \
    -m models/Qwen2.5-0.5B-Instruct.gguf \
    -c 2048 -b 2048 -ub 256 -npp 128 -ntg 128 -npl 2 \
    -fa on
```
And this produces the following error when run:
```console
ain: n_kv_max = 2048, n_batch = 2048, n_ubatch = 256, flash_attn = 1, is_pp_shared = 0, is_tg_separate = 0, n_gpu_layers = -1, n_threads = 4, n_threads_batch = 4

|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|

CUDA Exception: Warp Misaligned Address
The exception was triggered at PC 0x7fff20cd8fb0  flash_attn_ext_vec<64, 1, (ggml_type)1, (ggml_type)1, false>(char const*, char const*, char const*, char const*, char const*, int const*, float*, float2*, float, float, float, float, unsigned int, float, int, uint3, int, int, int, int, int, int, int, int, int, int, int, long, int, int, long, int, int, int, int, int, long)  (common.cuh:692)

Thread 1 "llama-batched-b" received signal CUDA_EXCEPTION_6, Warp Misaligned Address.
[Switching focus to CUDA kernel 0, grid 1399, block (0,1,15), thread (0,3,0), device 0, sm 0, warp 4, lane 0]
0x00007fff20cd8fe0 in _INTERNAL_7bc1cc71_29_fattn_vec_instance_f16_f16_cu_a1d0f301_92885::ggml_cuda_memcpy_1<16, 0>
   <<<(1,3,28),(32,4,1)>>> (dst=0x7fff9bfffc88, src=0x7fff2c3a2700)
    at /home/danbev/work/ai/llama.cpp/ggml/src/ggml-cuda/template-instances/../common.cuh:682
682	    for (int i = 0; i < nbytes/nb_per_cpy; ++i) {
```
Notice that `dst` is `0x7fff9bfffc88` is not divisable by 16 (it is divisable by 8).

We can run `compute-sanitizer` to get more information:
```console
compute-sanitizer --tool memcheck --target-processes all \
    ./${build_dir}/bin/llama-batched-bench \
    -m models/Qwen2.5-0.5B-Instruct.gguf \
    -c 2048 -b 2048 -ub 256 -npp 128 -ntg 128 -npl 2 \
    -fa on
```

```console
|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
========= Invalid __local__ write of size 16 bytes
=========     at ggml_cuda_memcpy_1<16,0>+0x47b0 in /home/danbev/work/ai/llama.cpp/ggml/src/ggml-cuda/common.cuh:692
=========     by thread (0,2,0) in block (0,0,0)
=========     Address 0xfffbc8 is misaligned
=========     Device Frame:void flash_attn_ext_vec<(int)64, (int)1, (ggml_type)1, (ggml_type)1, (bool)0>(const char *, const char *, const char *, const char *, const char *, const int *, float *, float2 *, float, float, float, float, unsigned int, float, int, uint3, int, int, int, int, int, int, int, int, int, int, int, long, int, int, long, int, int, int, int, int, long)+0x4810 in /home/danbev/work/ai/llama.cpp/ggml/src/ggml-cuda/common.cuh:227
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame: [0x2dfbef]
=========                in /lib/x86_64-linux-gnu/libcuda.so.1
=========     Host Frame: [0x15aa7]
=========                in /usr/local/cuda-12.6/lib64/libcudart.so.12
=========     Host Frame:cudaLaunchKernel [0x759f0]
=========                in /usr/local/cuda-12.6/lib64/libcudart.so.12
=========     Host Frame:cudaError cudaLaunchKernel<char>(char*, dim3, dim3, void**, unsigned long, CUstream_st*) in /usr/local/cuda-12.6/targets/x86_64-linux/include/cuda_runtime.h:216 [0xc55eb8]
=========                in /home/danbev/work/ai/llama.cpp/build-cuda-89-debug/bin/libggml-cuda.so.0
=========     Host Frame:__device_stub__Z18flash_attn_ext_vecILi64ELi1EL9ggml_type1ELS0_1ELb0EEvPKcS2_S2_S2_S2_PKiPfP6float2ffffjfi5uint3iiiiiiiiiiiliiliiiiil(char const*, char const*, char const*, char const*, char const*, int const*, float*, float2*, float, float, float, float, unsigned int, float, int, uint3 const&, int, int, int, int, int, int, int, int, int, int, int, long, int, int, long, int, int, int, int, int, long) in /tmp/tmpxft_0001c607_00000000-6_fattn-vec-instance-f16-f16.cudafe1.stub.c:43 [0xc4d7bd]
=========                in /home/danbev/work/ai/llama.cpp/build-cuda-89-debug/bin/libggml-cuda.so.0
=========     Host Frame:void __wrapper__device_stub_flash_attn_ext_vec<64, 1, (ggml_type)1, (ggml_type)1, false>(char const* restrict&, char const* restrict&, char const* restrict&, char const* restrict&, char const* restrict&, int const* restrict&, float* restrict&, float2* restrict&, float const&, float const&, float const&, float const&, unsigned int const&, float const&, int const&, uint3 const&, int const&, int const&, int const&, int const&, int const&, int const&, int const&, int const&, int const&, int const&, int const&, long const&, int const&, int const&, long const&, int const&, int const&, int const&, int const&, int const&, long const&) in /tmp/tmpxft_0001c607_00000000-6_fattn-vec-instance-f16-f16.cudafe1.stub.c:44 [0xc4da3e]
=========                in /home/danbev/work/ai/llama.cpp/build-cuda-89-debug/bin/libggml-cuda.so.0
=========     Host Frame:void flash_attn_ext_vec<64, 1, (ggml_type)1, (ggml_type)1, false>(char const*, char const*, char const*, char const*, char const*, int const*, float*, float2*, float, float, float, float, unsigned int, float, int, uint3, int, int, int, int, int, int, int, int, int, int, int, long, int, int, long, int, int, int, int, int, long) in /home/danbev/work/ai/llama.cpp/ggml/src/ggml-cuda/template-instances/../fattn-vec.cuh:42 [0xc55e60]
=========                in /home/danbev/work/ai/llama.cpp/build-cuda-89-debug/bin/libggml-cuda.so.0
=========     Host Frame:void launch_fattn<64, 1, 1>(ggml_backend_cuda_context&, ggml_tensor*, void (*)(char const*, char const*, char const*, char const*, char const*, int const*, float*, float2*, float, float, float, float, unsigned int, float, int, uint3, int, int, int, int, int, int, int, int, int, int, int, long, int, int, long, int, int, int, int, int, long), int, unsigned long, int, bool, bool, bool, int) in /home/danbev/work/ai/llama.cpp/ggml/src/ggml-cuda/template-instances/../fattn-common.cuh:986 [0xc3d465]
=========                in /home/danbev/work/ai/llama.cpp/build-cuda-89-debug/bin/libggml-cuda.so.0
=========     Host Frame:void ggml_cuda_flash_attn_ext_vec_case_impl<64, 1, (ggml_type)1, (ggml_type)1, false>(ggml_backend_cuda_context&, ggml_tensor*) in /home/danbev/work/ai/llama.cpp/ggml/src/ggml-cuda/template-instances/../fattn-vec.cuh:523 [0xc572e5]
=========                in /home/danbev/work/ai/llama.cpp/build-cuda-89-debug/bin/libggml-cuda.so.0
=========     Host Frame:void ggml_cuda_flash_attn_ext_vec_case<64, (ggml_type)1, (ggml_type)1>(ggml_backend_cuda_context&, ggml_tensor*) in /home/danbev/work/ai/llama.cpp/ggml/src/ggml-cuda/template-instances/../fattn-vec.cuh:538 [0xc56fa9]
=========                in /home/danbev/work/ai/llama.cpp/build-cuda-89-debug/bin/libggml-cuda.so.0
=========     Host Frame:ggml_cuda_flash_attn_ext_vec(ggml_backend_cuda_context&, ggml_tensor*) in /home/danbev/work/ai/llama.cpp/ggml/src/ggml-cuda/fattn.cu:196 [0x220682]
=========                in /home/danbev/work/ai/llama.cpp/build-cuda-89-debug/bin/libggml-cuda.so.0
=========     Host Frame:ggml_cuda_flash_attn_ext(ggml_backend_cuda_context&, ggml_tensor*) in /home/danbev/work/ai/llama.cpp/ggml/src/ggml-cuda/fattn.cu:366 [0x221057]
=========                in /home/danbev/work/ai/llama.cpp/build-cuda-89-debug/bin/libggml-cuda.so.0
=========     Host Frame:ggml_cuda_compute_forward(ggml_backend_cuda_context&, ggml_tensor*) in /home/danbev/work/ai/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:2712 [0x24317f]
=========                in /home/danbev/work/ai/llama.cpp/build-cuda-89-debug/bin/libggml-cuda.so.0
=========     Host Frame:ggml_cuda_graph_evaluate_and_capture(ggml_backend_cuda_context*, ggml_cgraph*, bool, bool) in /home/danbev/work/ai/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:3677 [0x248431]
=========                in /home/danbev/work/ai/llama.cpp/build-cuda-89-debug/bin/libggml-cuda.so.0
=========     Host Frame:ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) in /home/danbev/work/ai/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:3773 [0x248bd8]
=========                in /home/danbev/work/ai/llama.cpp/build-cuda-89-debug/bin/libggml-cuda.so.0
=========     Host Frame:ggml_backend_graph_compute_async in /home/danbev/work/ai/llama.cpp/ggml/src/ggml-backend.cpp:364 [0x704f0]
=========                in /home/danbev/work/ai/llama.cpp/build-cuda-89-debug/bin/libggml-base.so.0
=========     Host Frame:ggml_backend_sched_compute_splits(ggml_backend_sched*) in /home/danbev/work/ai/llama.cpp/ggml/src/ggml-backend.cpp:1580 [0x7560f]
=========                in /home/danbev/work/ai/llama.cpp/build-cuda-89-debug/bin/libggml-base.so.0
=========     Host Frame:ggml_backend_sched_graph_compute_async in /home/danbev/work/ai/llama.cpp/ggml/src/ggml-backend.cpp:1803 [0x765d4]
=========                in /home/danbev/work/ai/llama.cpp/build-cuda-89-debug/bin/libggml-base.so.0
=========     Host Frame:llama_context::graph_compute(ggml_cgraph*, bool) in /home/danbev/work/ai/llama.cpp/src/llama-context.cpp:2070 [0x47d325]
=========                in /home/danbev/work/ai/llama.cpp/build-cuda-89-debug/bin/libllama.so.0
=========     Host Frame:llama_context::process_ubatch(llama_ubatch const&, llm_graph_type, llama_memory_context_i*, ggml_status&) in /home/danbev/work/ai/llama.cpp/src/llama-context.cpp:1094 [0x4786b8]
=========                in /home/danbev/work/ai/llama.cpp/build-cuda-89-debug/bin/libllama.so.0
=========     Host Frame:llama_context::decode(llama_batch const&) in /home/danbev/work/ai/llama.cpp/src/llama-context.cpp:1532 [0x47a73d]
=========                in /home/danbev/work/ai/llama.cpp/build-cuda-89-debug/bin/libllama.so.0
=========     Host Frame:llama_decode in /home/danbev/work/ai/llama.cpp/src/llama-context.cpp:3438 [0x4827f7]
=========                in /home/danbev/work/ai/llama.cpp/build-cuda-89-debug/bin/libllama.so.0
=========     Host Frame:main::{lambda(llama_context*, llama_batch&, int, bool)#1}::operator()(llama_context*, llama_batch&, int, bool) const in /home/danbev/work/ai/llama.cpp/tools/batched-bench/batched-bench.cpp:90 [0x94d4d]
=========                in /home/danbev/work/ai/llama.cpp/./build-cuda-89-debug/bin/llama-batched-bench
=========     Host Frame:main in /home/danbev/work/ai/llama.cpp/tools/batched-bench/batched-bench.cpp:210 [0x95d5d]
=========                in /home/danbev/work/ai/llama.cpp/./build-cuda-89-debug/bin/llama-batched-bench
=========     Host Frame:__libc_start_call_main in ../sysdeps/nptl/libc_start_call_main.h:58 [0x2a1c9]
=========                in /lib/x86_64-linux-gnu/libc.so.6
=========     Host Frame:__libc_start_main in ../csu/libc-start.c:360 [0x2a28a]
=========                in /lib/x86_64-linux-gnu/libc.so.6
=========     Host Frame:_start [0x94844]
=========                in /home/danbev/work/ai/llama.cpp/./build-cuda-89-debug/bin/llama-batched-bench
```
And if we step through the code we will end up in this section of `fattn-vec.cuh`:
```c++
        for (int j = 0; j < ncols; ++j) {
            const float2 * Q_j = (const float2 *) (Q + j*nb01);
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ)*cpy_ne;
                if (ncols == 1 || ic0 + j < int(ne01.z)) {
                    ggml_cuda_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ],            &Q_j[i]);
                    ggml_cuda_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ + cpy_ne/2], &Q_j[i + cpy_ne/2]);
                }
            }
```
Lets inspect the destination and source pointers (for ggml_cuda_memcpy_1):
```console
(cuda-gdb) p Q_j
$1 = (const @generic float2 * @register) 0x7fff4ac18500
(cuda-gdb) p &Q_reg
$3 = (@local float2 (*)[1][4]) 0xfffc88
```
So we can see here that `Q_reg` is not aligned on a 16-byte boundary. We could try
alignas(16) but my understanding is that doing so would only align the starting
address of Q_reg, not each element within it.

If we step into `ggml_cuda_memcpy_1`:
```c++
    ggml_cuda_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ], &Q_j[i]);
                               destination                source
                               (Stack/Local)              (Global Memory)
```
```c++
static __device__ __forceinline__ void ggml_cuda_memcpy_1(void * __restrict__ dst, const void * __restrict__ src) {
    ...
    for (int i = 0; i < nbytes/nb_per_cpy; ++i) {
        if constexpr (nb_per_cpy == 1) {
            ((char *) dst)[i] = ((const char *) src)[i];
        } else if constexpr (nb_per_cpy == 2) {
            ((short *) dst)[i] = ((const short *) src)[i];
        } else if constexpr (nb_per_cpy == 4) {
            ((int *) dst)[i] = ((const int *) src)[i];
        } else if constexpr (nb_per_cpy == 8) {
            ((int2 *) dst)[i] = ((const int2 *) src)[i];
        } else if constexpr (nb_per_cpy == 16) {
----->      ((int4 *) dst)[i] = ((const int4 *) src)[i];
        } else {
            static_assert(nbytes == 0 && nbytes == -1, "bad nbytes");
        }
    }
```
```console
(cuda-gdb) n
692	            ((int4 *) dst)[i] = ((const int4 *) src)[i];
(cuda-gdb) p src
$4 = (const @generic void * @register) 0x7fff4ac18500
(cuda-gdb) p dst
$5 = (@generic void * @register) 0x7fff9bfffc88
```
And the cast to int4 and dereference is what is causing the misaligned address
error.
```console
(cuda-gdb) n

CUDA Exception: Warp Misaligned Address
The exception was triggered at PC 0x7fff933ba520  flash_attn_ext_vec<64, 1, (ggml_type)1, (ggml_type)1, false>(char const*, char const*, char const*, char const*, char const*, int const*, float*, float2*, float, float, float, float, unsigned int, float, int, uint3, int, int, int, int, int, int, int, int, int, int, int, long, int, int, long, int, int, int, int, int, long)  (common.cuh:692)

Thread 1 "llama-batched-b" received signal CUDA_EXCEPTION_6, Warp Misaligned Address.
_INTERNAL_7bc1cc71_29_fattn_vec_instance_f16_f16_cu_a1d0f301_123442::ggml_cuda_memcpy_1<16, 0><<<(1,3,28),(32,4,1)>>> (
    dst=0x7fff9bfffc88, src=0x7fff4ac18500)
    at /home/danbev/work/ai/llama.cpp/ggml/src/ggml-cuda/template-instances/../common.cuh:682
682	    for (int i = 0; i < nbytes/nb_per_cpy; ++i) {
```

A possible fix could be to split the 16-byte copy into two 8-byte copies:
```c++
            ((int2 *) dst)[2*i]     = ((const int2 *) src)[2*i];
            ((int2 *) dst)[2*i + 1] = ((const int2 *) src)[2*i + 1];
```
So even if we are on a 8-byte boundry we could still work to copy 16 bytes which
would accomodate for float2 types (8-byte alignment).
