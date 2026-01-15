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
For example, if we were to define Q_reg as:
```c++
    __align__(16) float2 Q_reg[ncols][(D/2)/nthreads_KQ] = {{{0.0f, 0.0f}}}; // May be only partially initialized.
```
For the actual sizes uses there there are:
```console
    __align__(16) float2 Q_reg[1][(32)/8] = {{{0.0f, 0.0f}}}; // May be only partially initialized.
    __align__(16) float2 Q_reg[1][4]      = {{{0.0f, 0.0f}}}; // May be only partially initialized.

```
Then the first element `Q_reg[0][0]` would be 16-byte aligned:
```console
(cuda-gdb) p &Q_reg
$2 = (@local float2 (*)[1][4]) 0xfffc80
(cuda-gdb) p &Q_reg[0][0]
$4 = (@local float2 *) 0xfffc80
```
But if we look at the rest of the elements in the array we find:
```console
(cuda-gdb) p &Q_reg[0][0]
$15 = (@local float2 *) 0xfffc80        aligned 16-bytes      (dec: 16776320)

(cuda-gdb) p &Q_reg[0][1]
$16 = (@local float2 *) 0xfffc88        not aligned           (dec: 16776328)

(cuda-gdb) p &Q_reg[0][2]
$17 = (@local float2 *) 0xfffc90        aligned               (dec: 16776336)

(cuda-gdb) p &Q_reg[0][3]
$18 = (@local float2 *) 0xfffc98        not aligned           (dec: 16776344)
```

If we step into `ggml_cuda_memcpy_1`:
```c++
    ggml_cuda_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ], &Q_j[i]);
```
```console
(cuda-gdb) p cpy_nb
$19 = 16
(cuda-gdb) p j
$20 = 0

(cuda-gdb) p &Q_reg[0][0]
$31 = (@local float2 *) 0xfffc80
```
This address is 16-byte aligned and passed to ggml_cuda_memcpy_1 as dst.
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

This is not correct as it breaks the assumption that ggml_cuda_memcpy_1<16> would
do a single 16-byte copy, but it would at least get past the misalignment issue.

My attempt with aligning Q_reg using `__align__(16)` did not work or that is
what I thought but I actually think it will as we are simply casting to int4
in ggml_cuda_memcpy_1 which requires 16-byte alignment. So as long as we start
at a 16-byte aligned address we should be fine. But there are other places that
need this same treatment and it is probably the case that I saw the same error
and assumed that it was the same root cause.

With the alignas(16) it no longer crashes with the misaligned address error at
this point. But fails later when calling vec_dot_KQ here:
```console
265	                float sum = vec_dot_KQ(K + i_KQ*nb11, Q_reg[j], Q_i32[j], Q_ds[j]);
(cuda-gdb) 

CUDA Exception: Warp Misaligned Address
The exception was triggered at PC 0x7fff20cd19b0  flash_attn_ext_vec<64, 1, (ggml_type)1, (ggml_type)1, false>(char const*, char const*, char const*, char const*, char const*, int const*, float*, float2*, float, float, float, float, unsigned int, float, int, uint3, int, int, int, int, int, int, int, int, int, int, int, long, int, int, long, int, int, int, int, int, long)  (common.cuh:696)

Thread 1 "llama-batched-b" received signal CUDA_EXCEPTION_6, Warp Misaligned Address.
_INTERNAL_7bc1cc71_29_fattn_vec_instance_f16_f16_cu_a1d0f301_47704::ggml_cuda_memcpy_1<16, 0> (dst=0x7fff9bfffc2c, 
    src=0x7fff4a600000) at /home/danbev/work/ai/llama.cpp/ggml/src/ggml-cuda/template-instances/../common.cuh:686
686	    for (int i = 0; i < nbytes/nb_per_cpy; ++i) {
```
Notice that we are using Q_reg again passing it into vec_dot_KQ, and stepping
into this function we find another ggml_cuda_memcpy_1<16> call:
```c++
template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_f16(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds_v) {

    const half2 * K_h2 = (const half2 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        half2 tmp[cpy_ne];
        ggml_cuda_memcpy_1<sizeof(tmp)>(tmp, K_h2 + k_KQ_0 + (threadIdx.x % nthreads)*cpy_ne);
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
#ifdef V_DOT2_F32_F16_AVAILABLE
            ggml_cuda_mad(sum,                tmp[k_KQ_1] , ((const half2  *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#else
            ggml_cuda_mad(sum, __half22float2(tmp[k_KQ_1]), ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    return sum;
}
```
What is passed into this function:
```console
CUDA thread hit Breakpoint 4.9, flash_attn_ext_vec<64, 1, (ggml_type)1, (ggml_type)1, false><<<(1,3,28),(32,4,1)>>> (
    Q=0x7fff2c3a1800 "", K=0x7fff4a600000 "", V=0x7fff4a680000 "", mask=0x7fff2c1e1800 "", 
    sinks=0x0 <_INTERNAL_aedbb58d_7_norm_cu_9a3dffe9::min(unsigned int, int)>, KV_max=0x302000000, dst=0x302000080, 
    dst_meta=0x302005480, scale=0.125, max_bias=0, m0=1, m1=1, n_head_log2=8, logit_softcap=0, ne00=64, ne01=..., ne02=14, ne03=2, 
    nb01=3584, nb02=256, nb03=3584, ne10=64, ne11=256, ne12=2, ne13=2, nb11=256, nb12=128, nb13=262144, nb21=256, nb22=128, 
    nb23=262144, ne31=1, ne32=1, ne33=2, nb31=512, nb32=512, nb33=512)
    at /home/danbev/work/ai/llama.cpp/ggml/src/ggml-cuda/template-instances/../fattn-vec.cuh:265
265	                float sum = vec_dot_KQ(K + i_KQ*nb11, Q_reg[j], Q_i32[j], Q_ds[j]);

(cuda-gdb) p &Q_reg[j]
$55 = (@local float2 (*)[4]) 0xfffc80

(cuda-gdb) p &tmp
$57 = (@local half2 (*)[4]) 0xfffc2c
```
In this case what is being copyied is tmp which is not alighned. Lets try to
align it:
```c++
        __align__(16) half2 tmp[cpy_ne];
```
And this worked and now more crashes with misaligned address errors.
