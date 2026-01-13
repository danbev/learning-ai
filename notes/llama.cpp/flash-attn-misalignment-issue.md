### Flash Attention batched-bench issue
Currently when running the llama-batch-bench example using a debug build and with
flash attention enabled, the default, it fails with a "CUDA Exception: Warp
Misaligned Address". 

To reproduce, build llama.cpp with CUDA and debug symbols, and then run 
llama-batched-bench with the following command:
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
...
```

If we step through the code we will end up in this section of `fattn-vec.cuh`:
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
Set the following breakpoint to step through the code:
```console
(cuda-gdb) br fattn-vec.cuh:226
```
Lets inspect the destination and source pointers (for ggml_cuda_memcpy_1):
```console
(cuda-gdb) p Q_j
$1 = (const @generic float2 * @register) 0x7fff4ac18500
(cuda-gdb) p &Q_reg
$3 = (@local float2 (*)[1][4]) 0xfffc88
```
So we can see here that `Q_reg` is not aligned on a 16-byte boundary.

We can find the definition of Q_reg in ggml/src/ggml-cuda/fattn-vec.cuh:
```c++
    float2 Q_reg[ncols][(D/2)/nthreads_KQ] = {{{0.0f, 0.0f}}}; // May be only partially initialized.
```
A float2, which is a struct that contains 2 32-bit floats (64 bits, 8 bytes). The
GPU would use LD.64 instruction to load a float2 which requires 8-byte alignment.

We could try alignas(16) for Q_reg but my understanding is that doing so would
only align the startof the array, but Q_req is a 2d array so while the first
entry would be 16-byte aligned the following row would not be. For example:
```console
int ncols   = 2;
int row_len = 3;
alignas(16) float2 Q_reg[ncols][row_len] = {{...}};

Address of Q_reg[0][0]  = 0x1000 (16-byte aligned, divisible by 16)

Address of Q_reg[1][0]  = 0x1000 + (1 * row_len * sizeof(float2))
                        = 0x1000 + (1 *   3     * 8)
                        = 0x1000 + 24
                        = 0x1018 (hex 18 (decimal 24) is not divisible by 16, 24/16=1.5)
                                 so this is misaligned.
```

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
            ((int4 *) dst)[i] = ((const int4 *) src)[i]; <-- crash happens at this line
        } else {
            static_assert(nbytes == 0 && nbytes == -1, "bad nbytes");
        }
    }
```
int4 is a struct that contains 4 32-bit integers (128 bits, 16 bytes). The GPU
would use the LD.128 instruction to load an int4 which requires 16-byte alignment
and this is where the error is triggered.

We can step through to the line that causes the error and inspect the values:
```console
(cuda-gdb) n
692	            ((int4 *) dst)[i] = ((const int4 *) src)[i];
(cuda-gdb) p src
$4 = (const @generic void * @register) 0x7fff4ac18500
(cuda-gdb) p dst
$5 = (@generic void * @register) 0x7fff9bfffc88
```
And if we step over the line we get the error:
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

The proposed fix is to split the 16-byte copy into two 8-byte copies:
```c++
            ((int2 *) dst)[2*i]     = ((const int2 *) src)[2*i];
            ((int2 *) dst)[2*i + 1] = ((const int2 *) src)[2*i + 1];
```
So even if we are on a 8-byte boundry we could still work to copy 16 bytes which
would accomodate for float2 types (8-byte alignment).
