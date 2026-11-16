## CUDA Matrix Multiplication in llama.cpp
This document contains notes related to how matrix multiplication is implemented
using CUDA in the llama.cpp project. This will include stepping through the
relevant code paths and stopping to clear up any CUDA-specific concepts as they
arise.

### Set up
```console
$ cmake --workspace --preset cuda-89-debug
$ cuda-gdb --args ./${build_dir}/bin/llama-completion -m ${model} --prompt "What is the weather in Paris?" -n 20 --jinja -ngl 99 --no-warmup -no-cnv -bs
(gdb) br ggml-cuda.cu:2417
(gdb) r
```

## Code Walkthrough
First break will be in `ggml_cuda_compute_forward` where it we have a switch
statement to determine which operation to perform. For matrix multiplication,
```c++
static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    // why is this here instead of mul_mat?
    if (dst->src[0] != nullptr && ggml_backend_buft_is_cuda_split(dst->src[0]->buffer->buft)) {
        ggml_cuda_set_peer_access(dst->src[1]->ne[1], ctx.device);
    }

    switch (dst->op) {
    ...
        case GGML_OP_MUL_MAT:
            ggml_cuda_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            break;
}
```
And we can find ggml_cuda_mul_mat in the same file:
```c++
static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS

    if (ggml_cuda_op_mul_mat_use_fwht(dst) && ggml_cuda_op_fwht(ctx, src1, dst)) {
        return;
    }
```
The functions check if the current operation can be executed as a Fast
Walsh-Hadamard Transform (FWHT):
```c++
bool ggml_cuda_op_mul_mat_use_fwht(const struct ggml_tensor * op) {
    const struct ggml_tensor * a = op->src[0];
    const struct ggml_tensor * b = op->src[1];

    return op->op == GGML_OP_MUL_MAT && ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
           a->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
           (b->type == GGML_TYPE_F32 || b->type == GGML_TYPE_F16) && ggml_is_contiguous(b) && ggml_is_contiguous(op) &&
           ggml_are_same_shape(b, op);
}
```
Quantized buffers allocate a bit more than the tensor needs and this needs to
be cleared in that case.
```c++
    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE
```
```c++
    const int cc        = ggml_cuda_info().devices[ctx.device].cc;
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
```
```console
(gdb) p cc
$44 = 1210
(gdb) p warp_size
$45 = 32
```
In common.cuh we can see the cc is for GB10 (DGX Spark):
```c++
#define GGML_CUDA_CC_DGX_SPARK       1210
```
And my RTX-4070 would be 890 ADA_LOVELACE.

Next we check if unquantized floating point (F32, F16, and BF16) matrix-vector
can be performed with the current src0, which is the weight tensor. There is a
also the counter part which is ggml_cuda_mul_mat_vec_f.
```c++
    if (ggml_cuda_should_use_mmvf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11)) {
        if (trace_q) {
            GGML_LOG_INFO("mul_mat trace [%s]: MMVF, tokens = %lld, cc = %d\n", dst->name, (long long) ne11, cc);
        }
        // The custom F16 vector kernel can be used over batched cuBLAS GEMM.
        // But this is only faster for GPUs without tensor cores or with a thin src0 matrix (particularly KQV in attention)
        ggml_cuda_mul_mat_vec_f(ctx, src0, src1, nullptr, dst);
        return;
    }
```
Before taking a look at the check in use_mmvf, MMVF is (Mul Mat Vec Float) and
is a custom kernel for unquantized floating point weights, like F16, BF16 or
F32. 
```console
(cuda-gdb) p src0->type
$5 = GGML_TYPE_Q8_0

(cuda-gdb) p src0->ne
$9 = {4096, 18560, 1, 1}
```
```c++
bool ggml_cuda_should_use_mmvf(enum ggml_type type, int cc,
    int warp_size, const int64_t * src0_ne, const size_t * src0_nb, int64_t ne11) {
    if (src0_ne[0] % 2 != 0) {
        return false;
    }

    const size_t ts = ggml_type_size(type);
    if (src0_nb[0] != ts) {
        return false;
    }
    ...
    switch (type) {
        case GGML_TYPE_F32:
            ...
        case GGML_TYPE_F16:
            ...
        case GGML_TYPE_BF16:
            ...
        default:
            return false;
```
So we can see that for our type this will return false.

```c++
    if (ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11, /*mul_mat_id =*/ false)) {
        ggml_cuda_mul_mat_f(ctx, src0, src1, nullptr, dst);
        return;
    }
```
```c++
bool ggml_cuda_should_use_mmf(enum ggml_type type, int cc, int warp_size, const int64_t * src0_ne,
        const size_t * src0_nb, const int src1_ncols, bool mul_mat_id) {
    if (ggml_is_quantized(type)) {
        return false;
    }
```
So the type needs to unquantized for this.
```c++
    if (ggml_cuda_should_use_mmvq(src0->type, cc, ne11)) {
        ggml_cuda_mul_mat_vec_q(ctx, src0, src1, nullptr, dst);
        return;
    }
```
```c++
bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11) {
    if (!ggml_is_quantized(type)) {
        return false;
    }

    // k-quants cost more to decode and mvq redoes that per column, so MMQ wins sooner.
    // Only list quant-types MMQ supports, others would fall back to cuBLAS.
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_ADA_LOVELACE) {
        switch (type) { // tuned on RTX 4090
            case GGML_TYPE_Q2_K:
                return ne11 <= 4;
            case GGML_TYPE_Q3_K:
                return ne11 <= 6;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }

    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_BLACKWELL) {
        switch (type) { // tuned on RTX 5090
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
                return ne11 <= 5;
            case GGML_TYPE_Q5_K:
                return ne11 <= 6;
            case GGML_TYPE_Q6_K:
                return ne11 <= 7;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }

#define MMVQ_MAX_BATCH_SIZE  8

    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_DGX_SPARK) {
        switch (type) { // tuned on DGX Spark GB10
            case GGML_TYPE_Q2_K:
                return ne11 <= 6;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
```
```console
(cuda-gdb) p ne11 < 8
$16 = true
```
So this will select MMVQ:
```c++
        ggml_cuda_mul_mat_vec_q(ctx, src0, src1, nullptr, dst);
                                                   ↑
                                                 MoE ids
```
mmvq.cu
```c++
void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    const ggml_tensor * ids,
    ggml_tensor * dst,
    const ggml_cuda_mm_fusion_args_host * fusion) {

    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

```
Now, the naming of local variables is done by the macro
`GGML_TENSOR_BINARY_OP_LOCALS` and recall (documented in ../../ggml.md) that
each variable name has three parts:
```console
ne   1   2 
↑    ↑   ↑
|    |   |
|    |   + dimension: 0..3
|    + tensor : 0 = src0, 1 = src1, 2 = src2 etc.
+----- what: ne = number of elements, nb = number of bytes.
```

```
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1);
    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
    }
```
Now, this quantization had me a little confused initially but just keep in mind
that this is quantizing the activations, it is passing in src1's data pointer,
only the type of the weights tensor (src0->type) is used from the src0.
```console
(cuda-gdb)
1561	        quantize_row_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
(cuda-gdb) p src1->name
$1 = "attn_norm-0", '\000' <repeats 52 times>
(cuda-gdb) p src1->type
$2 = GGML_TYPE_F32
```
And this is done because when MMVQ computes the dot products it used dp4a (dot
product 4 elements and accumulate) which multiplies 4 int8 values by 4 int8
values and adds them into an int32. So both operanads have to be integers.
So we currently have:
```console
src0 (weights)     Q4_K   already 4-bit values + scales
src1 (activations) F32    quantize_row_q8_1_cuda -> Q8_1 (int8 + scales)
```
The activations are split into blocks of 32 values.

And recall that a Blackwell has Tensor Cores units inside each SM on the GPU
next to the normal CUDA cores. These use different instructions:
```console
GPU (GB10)
  SM
    CUDA cores        dp4a (dot product 4*int8 accumulate, mmvq)
    Tensor cores      mma (whole-wrap matrix tile, mmq)
```
So small batches run on normal CUDA cores and larger batches on Tensor Cores. The
mma instruction always computes a fixed-size tile, for example 16 weight rows *
8 token columns * 32 deep and if we have fewer than 8 tokens the empty columns
are padded and computed just the same. So there is extra work involved here in
doing the padding, tiling and setup cost of copying tiles through shared memory.
This is avoided when usin mmvq which just streams the weights once and does a
simple dp4a per thread. And this is what the ne11=8 cutoff is for and this can
vary depending on the actual GPU in use.

Just to recap what we are doing here:
```console
(cuda-gdb) p dst->name
$4 = "Qcur-0", '\000' <repeats 57 times>
(cuda-gdb) p dst->op  
$5 = GGML_OP_MUL_MAT
(cuda-gdb) p dst->ne
$8 = {4096, 2, 1, 1}

(cuda-gdb) p src0->name
$6 = "blk.0.attn_q.weight", '\000' <repeats 44 times>
(cuda-gdb) p src0->ne
$9 = {4096, 4096, 1, 1}

(cuda-gdb) p src1->name
$7 = "attn_norm-0", '\000' <repeats 52 times>
(cuda-gdb) p src1->ne
$10 = {4096, 2, 1, 1}
```
So we have the weights in src0, and the activations in src1, which is this case
is the warmup and we have only two tokens. This is the tensor that will get
quantized.
```c++
const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

const int64_t s11 = src1->nb[1] / ts_src1;
const int64_t s12 = src1->nb[2] / ts_src1;
const int64_t s13 = src1->nb[3] / ts_src1;

quantize_row_q8_1_cuda(
    src1_d,            // x: pointer to activation data, _d means device pointer
    nullptr,           // ids: ids pointer for expert routing MUL_MAT_ID unused here
    src1_q8_1.get(),   // vy: cuda output buffer (device global memory) for Q8_1 blocks
    src0->type,        // the type of the weights tensor
    ne10,              // number of elements of src1, dim 0
    s11,               // stride for src1, dim 1
    s12,               // stride for src1, dim 2
    s13,               // stride for src1, dim 3
    ne10_padded,       // number of elements src1, dim 0 padded
    ne11,              // number of elements src1, dim 1
    ne12,              // number of elements src1, dim 2
    ne13,              // number of elements src1, dim 3
    stream);           // the cuda stream
```

ggml-cuda/quantize.cu:
```c++
void quantize_row_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    GGML_ASSERT(!ids);
    GGML_ASSERT(ne0 % QK8_1 == 0);

    const uint3 ne2_fastdiv = init_fastdiv_values(ne2);

    const int64_t block_num_x = (ne0 + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
    const dim3 num_blocks(block_num_x, ne1, ne2*ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(num_blocks, block_size, 0, stream);
    ggml_cuda_kernel_launch(quantize_q8_1, launch_params, x, vy, ne00, s01, s02, s03, ne0, ne1, ne2_fastdiv);
    GGML_UNUSED(type_src0);
}
```
So notice that this 

That commit would also allow quantize_row_q8_1_cuda to remove its `ids`
and `type_src0` parameters (which are not used) from the function
signature. I'll open a follow up PR or include a commit for that which
ever makes reviewing easiest.



__wip__

Just a note about the name `switch` in some of the function names. So the target
function is `mul_mat_f_cuda` which is templatized on the number of columns per block.
If we know the number of columns to process per block the kernel can be made to
perform better like loop unrolling, constant folding, and better register
allocation. If a value is only known at runtime that prevents these optimizations.
So instead we have switch cases for the number of columns per block and then have
them call specialized templatized versions of the kernel.

```c++
template <typename T, int cols_per_block>
void mul_mat_f_cuda(
        const T * x, const float * y, const int32_t * ids, float * dst,
        const int64_t ncols_x, const int64_t nrows_x, const int64_t ncols_dst,
        const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst,
        const int64_t stride_col_id, const int64_t stride_row_id,
        const int64_t nchannels_x, const int64_t nchannels_y, const int64_t nchannels_dst,
        const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst, const int64_t nsamples_x,
        const int64_t nsamples_dst, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
        cudaStream_t stream, const mmf_ids_data * ids_data) {
    typedef tile<16, 8, T>     tile_A_16;
    typedef tile<32, 8, T>     tile_A_32;
    typedef tile<16, 8, T>     tile_B_16;
    typedef tile< 8, 8, T>     tile_B_8;
    ...


    switch (nwarps_best) {
        ...
        case 7: {
            mul_mat_f_switch_ids<T, cols_per_block, 7>(
                x, y, ids, dst, ncols_x, ncols_dst, nchannels_dst, stride_row, stride_col_y, stride_col_dst,
                stride_col_id, stride_row_id, channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, block_nums, block_dims, nbytes_shared_total, stream,
                ids_data);
        } break;
        ...
```
And finally we arrive at the kernel itself:
```c++
template <typename T, int rows_per_block, int cols_per_block, int nwarps, bool has_ids>
__launch_bounds__(ggml_cuda_get_physical_warp_size()*nwarps, 1)
static __global__ void mul_mat_f(
        const T * __restrict__ x, const float * __restrict__ y, const int32_t * __restrict__ ids, float * __restrict__ dst,
        const int ncols, const int ncols_dst_total, const int nchannels_dst, const int stride_row, const int stride_col_y, const int stride_col_dst,
        const int stride_col_id, const int stride_row_id,
        const int channel_ratio, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int sample_ratio, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst) {

```
We can see the concrete template specialization by looking at the backtrace:
```console
(cuda-gdb) f
mul_mat_f<__half2, 32, 7, 7, false> (x=0x7fff5e3a9c00, y=0x7ffefa1c3080,
    ids=0x0 <block_reduce_policy<(block_reduce_method)1, float2>::sentinel()>, dst=0x7ffefa383080, ncols=448, ncols_dst_total=7,
    nchannels_dst=1, stride_row=448, stride_col_y=448, stride_col_dst=896, stride_col_id=0, stride_row_id=0, channel_ratio=1,
    stride_channel_x=401408, stride_channel_y=6272, stride_channel_dst=6272, sample_ratio=1, stride_sample_x=401408,
    stride_sample_y=6272, stride_sample_dst=6272)
    at /home/danbev/work/ai/llama.cpp/ggml/src/ggml-cuda/template-instances/../mmf.cuh:30
30	        const int sample_ratio, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst) {
```

Next, we have:
```c++
    const int channel_x   = has_ids ? expert_idx : (channel_dst / channel_ratio);
    const int channel_y   = channel_dst;
    const int sample_dst  = blockIdx.z;
    const int sample_x    = sample_dst / sample_ratio;
    const int sample_y    = sample_dst;

    x   += int64_t(sample_x)  *stride_sample_x   + channel_x  *stride_channel_x  + row0*stride_row ;
    y   += int64_t(sample_y)  *stride_sample_y   + (has_ids ? 0 : channel_y  *stride_channel_y);
    dst += int64_t(sample_dst)*stride_sample_dst + (has_ids ? 0 : channel_dst*stride_channel_dst);

    extern __shared__ char data_mmv[];

    char * shmem_base = data_mmv;
    int  * slot_map   = (int *) shmem_base;
    char * compute_base = has_ids ? (shmem_base + GGML_PAD(cols_per_block, 16) * sizeof(int)) : shmem_base;

    tile_C C[ntA][ntB];

    T * tile_xy = (T *) compute_base + threadIdx.y*(tile_A::I * tile_k_padded);
```
Notice that we are defining shared memory here with for `data_mmv`.

Next we have the main loop:
```c++
    for (int col = threadIdx.y*warp_size + threadIdx.x; col < ncols; col += nwarps*warp_size) {
```
```console
(cuda-gdb) p ncols
$33 = 448
(cuda-gdb) p threadIdx.y
$34 = 0
(cuda-gdb) p warp_size
$35 = 32
```
And `nwarps` is 7 from the template instantiation above. So for this thread (0)
it will first process column 0, and then column 224: 
```console
(cuda-gdb) p warp_size * 7
$39 = 224
```
```console
7 warps × 32 threads/warp = 224 threads total
ncols                     = 448 columns
stride                    = 7 * 32 = 224
```

Matrix Multiplication Dimensions:
```
For the operation C = A × B:

A: M × K  (M rows, K columns)
B: K × N  (K rows, N columns)
C: M × N  (M rows, N columns)

      A          B          C
   [     ]     [   ]     [     ]
M    ...    K   ...    M   ...
     ...       [   ]       ...
   [     ]               [     ]
      K          N          N
```
To get a better understanding of what is going on below lets take a look at the
matrices that we will be multiplying:
And the current matrices look like this:
```console
(cuda-gdb) p dst->src[0]->name
$3 = "blk.0.attn_q.weight", '\000' <repeats 44 times>
(cuda-gdb) p dst->src[0]->ne
$4 = {896, 896, 1, 1}

$5 = "attn_norm-0", '\000' <repeats 52 times>
(cuda-gdb) p dst->src[1]->ne
$6 = {896, 7, 1, 1}

src0 is our A matrix, and src1 is our B matrix:
```console
   C(896,7) = A(896,896) * B^T(896,7)
               A                          B^T
   0   [0       ...         895]      0 [0 ... 6]
   1   [0       ...         895]      1 [0 ... 6]
   2   [0       ...         895]      2 [0 ... 6]
   3   [0       ...         895]  *   3 [0 ... 6]
   4   [0       ...         895]      4 [0 ... 6]
   5   [0       ...         895]      5 [0 ... 6]
   6   [0       ...         895]      6 [0 ... 6]
   ...          ...                       ...
   859 [0       ...         895]    895 [0 ... 6]
```
Now, B is not transposed explicitly but as we will discuss later.

The current thread (thread 0) will process:
```console
Iteration 0: col = 0
Iteration 1: col = 224
Iteration 2: col = 448 (exit loop since col >= ncols)
                                        448 <= 448
```
Warp 0 (y = 0, all threads in lockstep):
```console
Thread x=0:  col = 0                x[... + 0] <- memory address N
Thread x=1:  col = 1                x[... + 1] <- memory address N + 4 bytes
Thread x=2:  col = 2                x[... + 2] <- memory address N + 8 bytes
...
Thread x=31: col = 31               x[... + 31]<- memory address N + 124 bytes
```
All threads will issue the loads in parallel and they are 32 consecutive float16
values which enables the memory controller to fetch all 32 values in one memory
transaction. 

Warp 1 (y = 1):
```console
Thread x=0:  col = 32
Thread x=1:  col = 33
...
Thread x=31: col = 63
```
Warp 2 (y = 2):
```console
Thread x=0:  col = 64
Thread x=1:  col = 65
...
Thread x=31: col = 95
```
Warp 3 (y = 3):
```console
Thread x=0:  col = 96
Thread x=1:  col = 97
...
Thread x=31: col = 127
```
Warp 4 (y = 4):
```console
Thread x=0:  col = 128
Thread x=1:  col = 129
...
Thread x=31: col = 159
```
Warp 5 (y = 5):
```console
Thread x=0:  col = 160
Thread x=1:  col = 161
...
Thread x=31: col = 191
```
Warp 6 (y = 6):
```console
Thread x=0:  col = 192
Thread x=1:  col = 193
...
Thread x=31: col = 223
```

Lets look at the first inner loop (and remember we are processing column 0):
And also keep in mind the operation we are performing is `C += A * B`.
```c++
    for (int col = threadIdx.y*warp_size + threadIdx.x; col < ncols; col += nwarps*warp_size) {
        tile_A A[ntA][warp_size / tile_A::J];
```
So tile_A is defining a tile for a fragment of the A matrix:
```console
(cuda-gdb) p ntA
$41 = 2
(cuda-gdb) p warp_size
$42 = 32

(cuda-gdb) ptype A
type = @local struct _ZN13ggml_cuda_mma4tileILi16ELi8E7__half2LNS_11data_layoutE0EEE {
    @local half2 x[4];
} [2][4]
```
If we demangle ILi16ELi8E7__half2LNS_11data_layoutE0EEE we get:
* ILi16E:   This means I_ = 16. So, the tile represents 16 rows.
* Li8E:     This means J_ = 8. So, the tile represents 8 columns.
* 7__half2: This means T = __half2 (which is half2, the CUDA type for FP16 pairs).
* LNS_11data_layoutE0EEE: This means ds_ = DATA_LAYOUT_I_MAJOR (where 0 is the enum value for DATA_LAYOUT_I_MAJOR).

So the tile of matrix A would be something like this:
```console
               A
   0   [0     15           ...         895]
   1   [0     15           ...         895]
   2   [0     15           ...         895]
   3   [0     15           ...         895]
   4   [0     15           ...         895]
   5   [0     15           ...         895]
   6   [0     15           ...         895]
   7   [0     15           ...         895]
   8   [0     15           ...         895]
   9   [0     15           ...         895]
   10  [0     15           ...         895]
   11  [0     15           ...         895]
   12  [0     15           ...         895]
   13  [0     15           ...         895]
   14  [0     15           ...         895]
   15  [0     15           ...         895]
   ...                     ...
   859 [0                  ...         895]

And one tile from this matrix would look like this::

    0 [H H H H H H H H H H H H H H H H]
    1 [H H H H H H H H H H H H H H H H]
    2 [H H H H H H H H H H H H H H H H]
    3 [H H H H H H H H H H H H H H H H]
    4 [H H H H H H H H H H H H H H H H]
    5 [H H H H H H H H H H H H H H H H]
    6 [H H H H H H H H H H H H H H H H]
    7 [H H H H H H H H H H H H H H H H]
    8 [H H H H H H H H H H H H H H H H]
    9 [H H H H H H H H H H H H H H H H]
    10[H H H H H H H H H H H H H H H H]
    11[H H H H H H H H H H H H H H H H]
    12[H H H H H H H H H H H H H H H H]
    13[H H H H H H H H H H H H H H H H]
    14[H H H H H H H H H H H H H H H H]
    15[H H H H H H H H H H H H H H H H]

Total: 16 rows × 16 columns = 256 FP16 values
```
And each thread in a warp will operate on a fragment of this tile.

So tile_A is `tile<16, 8, half2, DATA_LAYOUT_I_MAJOR>`, so tile_A::I = 16 and
tile_A::J = 8.

```console
        tile_A A[ntA][warp_size / tile_A::J];
        tile_A A[ 2 ][32        / 8        ];
        tile_A A[ 2 ][4];

A[2][4]           = 8 tile_A structs per thread
Each tile_A       = half2 x[4] = 4 × half2
Each half2        = 32 bits (two FP16 values)

Total per thread:
    8 structs × 4 half2 × 32 bits = 1024 bits = 32 registers (32-bit each)
```
Recall that a half2 is a CUDA type that contains 2 FP16 values. So the A tile
will take up 32 registers per thread.
```c++
struct __half2 {
    __half x;  // First FP16 value  (16 bits)
    __half y;  // Second FP16 value (16 bits)
};
```
Together all the threads in the warp hold the complete 16x8 tile which we showed
above:
```console
Warp (32 threads) → Full 16×8 tile distributed across threads
  Thread 0 → fragment (half2 x[4]) → 8 FP16 values        32 registers
  Thread 1 → fragment (half2 x[4]) → 8 FP16 values        32 registers
  ...
  Thread 31 → fragment (half2 x[4]) → 8 FP16 values       32 registers
```

So tile_A represents a fragment, 8 FP16 values from the 16x16 logical tile, of
the A matrix that this thread going to operate on. The A variable is a local
varibable that each thread will have.

It first has to load the data for the fragment from global memory
into shared memory (every thread in the warp doing this in lock step so we get
the optimized load). And from there we will load them in to registers in the
correct format that mma expects.

Next we are going to iterate over 2 (ntA) tiles of the matrix x.
First, remember what ntA is:
```c++
  ntA = rows_per_block / tile_A::I
```
* rows_per_block is a template parameter for the kernel, typically 32. This is
  the total number of rows of the output matrix that this entire CUDA block is
  responsible for computing.

* tile_A::I is 16. This is the number of rows in one tile_A fragment that the
  mma instruction operates on.


The following is what one thread will load from global memory:
```console
Thread (y=0, x=0) when col=0:

Fragment 0 (itA=0):          Fragment 1 (itA=1):
         col 0                        col 0
Row  0:   [H]                Row 16:   [H]
Row  1:   [H]                Row 17:   [H]
Row  2:   [H]                Row 18:   [H]
...                          ...
Row 15:   [H]                Row 31:   [H]

16 elements from column 0    16 elements from column 0
```

And we have 32 threads in the warp doing this in parallel so we load:
```console
        Thread 0  Thread 1  Thread 2  ...  Thread 31
         (col 0)   (col 1)   (col 2)       (col 31)
Row  0:    H         H         H      ...     H
Row  1:    H         H         H      ...     H
...
Row 15:    H         H         H      ...     H
         ↑ Fragment 0 ↑

Row 16:    H         H         H      ...     H
Row 17:    H         H         H      ...     H
...
Row 31:    H         H         H      ...     H
         ↑ Fragment 1 ↑
```
Shape: 32 rows × 32 columns


So the outer loop will iterate over these two vertical fragments of matrix x,
and the inner loop will load the 16 rows for each fragment into shared memory.

Recall that tile_A::I = 16 and tile_A::J = 8.
```c++
        for (int itA = 0; itA < ntA; ++itA) {
            for (int i = 0; i < tile_A::I; ++i) {
                tile_xy[i*tile_k_padded + threadIdx.x] = x[(itA*tile_A::I + i)*stride_row  + col];
            }

            for (int k0 = 0; k0 < warp_size; k0 += tile_A::J) {
                load_ldmatrix(A[itA][k0/tile_A::J], tile_xy + k0, tile_k_padded);
            }
        }
```
I need to step back a bit to really understand this. If we break in
ggml_cuda_mul_mat_f and inspect the tensors we are dealing with:
```console
(cuda-gdb) p dst->op
$3 = GGML_OP_MUL_MAT

(cuda-gdb) p dst->src[0]->name
$4 = "blk.0.attn_q.weight", '\000' <repeats 44 times>
(cuda-gdb) p dst->src[0]->ne
$5 = {896, 896, 1, 1}

(cuda-gdb) p dst->src[1]->name
$6 = "attn_norm-0", '\000' <repeats 52 times>
(cuda-gdb) p dst->src[1]->ne  
$7 = {896, 7, 1, 1}
```
The model I'm using is Qwen2.5-0.5B-Instruct.gguf and it has:
```console
qwen2.embedding_length = 896

11:     802816 |   896,   896,     1,     1 | F16     | blk.0.attn_q.weight
```
```console
       A                     B^T

 0  [0  ... 895]     0   [0 ... 6]
 1  [0  ... 895]     1   [0 ... 6]
 ...                  ...
895 [0  ... 895]     895 [0 ... 6]
```
So the outer loop:
```c++
    for (int col = threadIdx.y*warp_size + threadIdx.x; col < ncols; col += nwarps*warp_size) {
```
ncols=448, nwarps=7, warp_size=32. So thread 0 will process columns 0 first and
in the next iteration column 224.

Matrix x layout in memory (896×896, but stride_row=448):
```console
    Row 0:  x[0]      x[1]      ... x[447]    (stride to next row = 448)
    Row 1:  x[448]    x[449]    ... x[895]
    Row 2:  x[896]    x[897]    ... x[1343]
    ...
```

```
   thread 0, iteration 0, col = 0
       ↓
  0   [0                             895]
  1   [0                             895]
  2   [0                             895]
  3   [0                             895]
  4   [0                             895]
  ..
  ...
  895 [0                             895]
```

tile_A::I = 16, tile_A::J = 8.
```c++
        tile_A A[ntA][warp_size / tile_A::J];
        //        2     32      /      8

        for (int itA = 0; itA < ntA; ++itA) {
            //                     16
            for (int i = 0; i < tile_A::I; ++i) {
                tile_xy[i*tile_k_padded + threadIdx.x] = x[(itA*tile_A::I + i)*stride_row  + col];
                //                                                16            448          (0, then 224)
            }

        }
```
Thread 0 will load 32 elements into the shared memory tile_xy (I'm ignoring the
padding for now but this could be offset by the padding)
```console
iteration itA=0, i = 0: (itA*tile_A::I +  i)
                       (  0*16        +  0) = 0
                       (0) * stride_row + col = 
                       (0) * 448        + 0   = 0        tile_xy[0] = x[0]

iteration itA=0, i = 1: (itA*tile_A::I +  i)
                       (  0*16        +  1) = 1
                       (1) * stride_row + col = 
                       (1) * 448        + 0   = 448      tile_xy[1] = x[448]

iteration itA=0, i = 2: (itA*tile_A::I +  i)
                       (  0*16        +  1) = 2
                       (2) * stride_row + col = 
                       (2) * 448        + 0   = 896      tile_xy[2] = x[896]
...
iteration itA=0, i = 15: (itA*tile_A::I + i)
                       (  0*16        + 15) = 15
                       (15) * stride_row + col = 
                       (15) * 448        + 0   = 6720     tile_xy[15] = x[6720]


iteration itA=1, i = 0: (itA*tile_A::I +  i)
                       (  1*16         +  0) = 16
                       (16) * stride_row + col = 
                       (16) * 448        + 0   = 7168      tile_xy[16] = x[7168]

iteration itA=1, i = 1: (itA*tile_A::I +  i)
                       (  1*16         +  1) = 17
                       (17) * stride_row + col = 
                       (17) * 448        + 0   = 7616      tile_xy[17] = x[7616]

iteration itA=1, i = 2: (itA*tile_A::I +  i)
                       (  1*16         +  2) = 18
                       (18) * stride_row + col = 
                       (18) * 448        + 0   = 8064      tile_xy[18] = x[8064] 
...
iteration itA=1, i = 15: (itA*tile_A::I + i)
                       (  1*16         + 15) = 31
                       (31) * stride_row + col = 
                       (31) * 448        + 0   = 13888     tile_xy[31] = x[13888]
```
And notice that these values are from the is the first column of the matrix x:
```console
    Row 0:  x[0]      x[1]      ... x[447]    (stride to next row = 448)
    Row 1:  x[448]    x[449]    ... x[895]
    Row 2:  x[896]    x[897]    ... x[1343]
    ...
    Row 15: x[6720]   x[6721]   ... x[7167]
    Row 16: x[7168]   x[7169]   ... x[7615]
    Row 17: x[7616]   x[7617]   ... x[8063]
    Row 18: x[8064]   x[8065]   ... x[8511]
    ...
    Row 31: x[13888]  x[13889]  ... x[14335]
```
So hopefully this clarifies what the loop is doing, it is basically reading 32
elements from the first column. And the other 31 threads will also load in a
similar fashion which allows coalesced memory accesses. With all threads accessing
a half2 (4 bytes, 32 bits) we have 32*4 = 128 which can be fetched in a single
memory transaction.

And after we have loaded all 32 elements for the first iteration we will then
load them into registers in the format that mma expects:
```c++
            //                      32                 8
            for (int k0 = 0; k0 < warp_size; k0 += tile_A::J) {
                load_ldmatrix(A[itA][k0/tile_A::J], tile_xy + k0, tile_k_padded);
                //                            8
            }
```
So each thread will do 4 iterations, as warp_size is 32 and tile_A::J is 8.
We have 32 columns of data in shared memory and we are loading them 8 columns
at a time into registers as this is what ldmatrix operates on:
```console
k0 = 0:  load_ldmatrix(A[itA][0], tile_xy + 0,  tile_k_padded)
k0 = 8:  load_ldmatrix(A[itA][1], tile_xy + 8,  tile_k_padded)
k0 = 16: load_ldmatrix(A[itA][2], tile_xy + 16, tile_k_padded)
k0 = 24: load_ldmatrix(A[itA][3], tile_xy + 24, tile_k_padded)
```

So at this point, before the B loop. We have only loaded a tile of the A matrix,
and now we are doing to load the corresponding values from the B matrix so that
we can perform the dot product on those elements. And we add the dot products to
the C matrix.

```console
A is loaded into registers
A[0] = rows 0-15,  columns col to col+31
A[1] = rows 16-31, columns col to col+31
```

Next we have a loop for the B matrix, but first to recap, our B matrix is:
```console
(cuda-gdb) p dst->src[1]->name
$6 = "attn_norm-0"

(cuda-gdb) p dst->src[1]->ne  
$7 = {896, 7, 1, 1}
```
And this is the operation we are performing:
```console
C = A × B^T

Where:
A is 896×896 (blk.0.attn_q.weight)
B is 896×7   (attn_norm-0)
C is 896×7   (output)
```
And just recall this is happening inside the outer loop, so we are currently
processing one tile of A.

The mma instruction, mma.sync.aligned.m16n8k16, is the core of the computation,
and it operates on small fragments of the matrices:
```console
  C_frag(16,8) += A_frag(16,16) * B_frag(16,8)

  0 [0 ... 15]    0 [0 ... 7]
  1 [0 ... 15]    1 [0 ... 7]
  2 [0 ... 15]    2 [0 ... 7]
  3 [0 ... 15]    3 [0 ... 7]
  4 [0 ... 15]    4 [0 ... 7]
  5 [0 ... 15]    5 [0 ... 7]
  6 [0 ... 15]    6 [0 ... 7]
  7 [0 ... 15]    7 [0 ... 7]
  8 [0 ... 15]    8 [0 ... 7]
  9 [0 ... 15]    9 [0 ... 7]
 10 [0 ... 15]   10 [0 ... 7]
 11 [0 ... 15]   11 [0 ... 7]
 12 [0 ... 15]   12 [0 ... 7]
 13 [0 ... 15]   13 [0 ... 7]
 14 [0 ... 15]   14 [0 ... 7]
 15 [0 ... 15]   15 [0 ... 7]
                           ↑
                         padding so all columns are zero.
```
The mma instruction requires a B fragment of size 16x8 but we only have 7 columns
as B is 896x7. So we will pad the B fragment to 16x8 by adding an extra column of
zeros.

For reference, y2 is a pointer to B in global memory and is of type, but y is
a float * pointer so it points to FP32 values):
```c++
    const float2 * y2 = (const float2 *) y;
```
By doing this we can read 2 floats at once which we will see shortly.

```c++
        for (int itB = 0; itB < ntB; ++itB) {
            if constexpr (std::is_same_v<T, float>) {
               ... // path not taken in this case
            } else if constexpr (std::is_same_v<T, half2> || std::is_same_v<T, nv_bfloat162>) {
                for (int j0 = 0; j0 < tile_B::I; ++j0) {
                    const int j = j0 + itB*tile_B::I;

                    if constexpr (!has_ids) {
                        // y2 is the pointer to B in global memory
                        //                       7              |     448
                        //                       ↓              ↓      ↓      
                        const float2 tmp = j < cols_per_block ? y2[j*stride_col_y + col] : make_float2(0.0f, 0.0f);
                        // The above reads two consecutive FP32 values from B
                        // tmp.x = B[j][col]
                        // tmp.y = B[j][col + 1]
                        tile_xy[j0*tile_k_padded + threadIdx.x] = ggml_cuda_cast<T>(tmp);
                    } else {
```
Now, tmp will be a float2:
```console
(cuda-gdb) p tmp
$30 = {x = 0.128237233, y = -0.0229022708}
```
The ggml_cuda_cast does the following conversion:
```console
  Input:  float2 {x=0.128237233 (FP32), y=-0.0229022708 (FP32)}
  Output: half2  {x=0.128237233 (FP16), y=-0.0229022708 (FP16)}
```
Each FP32 to FP16 conversion:
* Reduces precision (23-bit mantissa -> 10-bit mantissa)
* Reduces range (8-bit exponent -> 5-bit exponent)
* Saves memory and enables tensor core usage

Now there is no explicit transpose operation of B. It stays in its original form.

```console
Iteration j0=0, itB=0: j = 0 + 0*16 = 0
                         y2[0*448 + col]
                         y2[0 + col]  ← For thread 0, this is y2[0]
                         tile_xy[0*36 + 0] = ...

Iteration j0=1, itB=0: j = 1 + 0*16 = 1
                         y2[1*448 + col]
                         y2[448 + col]  ← For thread 0, this is y2[448]
                         tile_xy[1*36 + 0] = ...
```
So y2[448] is reading the second element from
```console
        B
  0 [0  .... 895]
  1 [0  .... 895]
  2 [0  .... 895]
  3 [0  .... 895]
  4 [0  .... 895]
  5 [0  .... 895]
  6 [0  .... 895]

B in memory (row-major, 896×7):
  Address 0:    [row0_col0, row0_col1, ..., row0_col6]
  Address 448:  [row1_col0, row1_col1, ..., row1_col6]
  Address 896:  [row2_col0, row2_col1, ..., row2_col6]
```
This is again doing the strided access pattern to get coalesced memory reads.

After we have loaded the tile of B into shared memory we then load it into
registers in the format that mma expects and also perform the mma operation:
```c++
            for (int k0 = 0; k0 < warp_size; k0 += tile_B::J) {
                tile_B B;
                load_ldmatrix(B, tile_xy + k0, tile_k_padded);

                for (int itA = 0; itA < ntA; ++itA) {
                    mma(C[itA][itB], A[itA][k0/tile_B::J], B);
                }
            }
```
And the mma implementation is as follows (in mma.cuh). Notice that we are passing
in C, A, and B tiles:
```c++
    static __device__ __forceinline__ void mma(
            tile<16, 8, float> & D, const tile<16, 8, half2> & A, const tile<8, 8, half2> & B) {
#ifdef TURING_MMA_AVAILABLE
        const int * Axi = (const int *) A.x;
        const int * Bxi = (const int *) B.x;
        int       * Dxi = (int       *) D.x;
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
---->   asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[1]));
#else
        // On Turing m16n8k16 mma is not available, use 2x m8n8k8 mma instead:
        asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
            : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Bxi[0]));
        asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
            : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3])
            : "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[1]));
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#else
        GGML_UNUSED_VARS(D, A, B);
        NO_DEVICE_CODE;
#endif // TURING_MMA_AVAILABLE
    }
```

```console
                                     
               A rows           D type    B type
                  ↓               ↓       ↓
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
                     ↑                 ↑       ↑
                   cols
                                    A type   C type
      {%0, %1, %2, %3},           // D (output) - 4 × 32-bit registers
      {%4, %5, %6, %7},           // A (input)  - 4 × 32-bit registers
      {%8, %9},                   // B (input)  - 2 × 32-bit registers
      {%0, %1, %2, %3};           // C (accumulator) - same as D

row = A layout is row-major
col = B layout is column-major
```
Notice that the this is specifying a constract with the hardware and we are
telling the tensor core that we are providing the A fragment in row-major layout,
and the B fragment in column-major layout. So that would mean that it expects the
B fragment to be transposed. But I think that B is actually also in row-major
in global memory so how does this work?  
The data for the B tile is first loaded from global memory into shared memory (
into tile_xy), and still row-major.
The `load_ldmatrix` instruction is specifically designed to read a tile from
shared memory and distribute it across the registers of all 32 threads in a
warp in the exact format the `mma` instruction requires. It reads the row-major
data from tile_xy and shuffles it into the registers of the 32 threads in a way
that, when the mma instruction executes, the hardware sees a coherent 16x8
column-major matrix.

To understand this we should consider that the ldmatrix (Load Matrix) and mma
(Matrix Multiply-Accumulate) were introduced together as a pair with NVIDIA's
Tensor Cores (starting with the Volta architecture).

So what would happen if I was not aware of this and thought that I need to transpose
the B fragment before calling mma?  If I did that I'd get incorrect results as
ldmatrix would still load the data in the same way as before. But there are
other flavours of the mma instruction:
```console
* mma ... .row.col ... (A is row-major, B is col-major) -> Computes A * B^T
* mma ... .row.row ... (A is row-major, B is row-major) -> Computes A * B
* mma ... .col.row ... (A is col-major, B is row-major) -> Computes A^T * B
* mma ... .col.col ... (A is col-major, B is col-major) -> Computes A^T * B^T
```
And there is a flavor/version of the ldmatrix instruction that can transpose
which we would also have to use in this case:
```console
ldmatrix.sync.aligned.m8n8.x4.b16.trans {%r0, %r1, ...}, [%rN];
```
This tells the hardware to load the matrix from shared memory and transpose it
as it moves into the registers. And in ggml-cuda there is a load_ldmatrix_trans
function that does this.

```
         A                B                   C/D
  0 [0    ...   15]    0 [0 ... 7]       0 [0 ... 7]
  1 [0    ...   15]    1 [0 ... 7]       1 [0 ... 7]
  2 [0    ...   15]    2 [0 ... 7]       2 [0 ... 7]
  3 [0    ...   15]    3 [0 ... 7]       3 [0 ... 7]
  4 [0    ...   15]    4 [0 ... 7]       4 [0 ... 7]
  5 [0    ...   15]    5 [0 ... 7]       5 [0 ... 7]
  6 [0    ...   15]    6 [0 ... 7]       6 [0 ... 7]
  7 [0    ...   15]    7 [0 ... 7]       7 [0 ... 7]
  8 [0    ...   15]    8 [0 ... 7]       8 [0 ... 7]
  9 [0    ...   15]    9 [0 ... 7]       9 [0 ... 7]
  10[0   ...    15]   10 [0 ... 7]      10 [0 ... 7]
  11[0   ...    15]   11 [0 ... 7]      11 [0 ... 7]
  12[0   ...    15]   12 [0 ... 7]      12 [0 ... 7]
  13[0   ...    15]   13 [0 ... 7]      13 [0 ... 7]
  14[0   ...    15]   14 [0 ... 7]      14 [0 ... 7]
  15[0   ...    15]   15 [0 ... 7]      15 [0 ... 7]
```
The warp collectively executes this as a single Tensor Core operation:
1. All 32 threads invoke mma simultaneously
2. Each thread provides its A, B, C fragments
3. The Tensor Core hardware performs the full 16×16×8 matrix multiply
4. Each thread's D registers are updated with its result fragment

So thread 0 will contribute 4 half2 values (8 FP16 values) from tile A and 4
half2 values from tile B.


### Quantization
So I initially had this backwards. If we have weights that are quantized, lets
say as Q4_K and activation that are in F32, I thought we would dequantize the
weighs to be able to perform the operation. But that is not the case and the
type of quantization/dequantization is determined by the type of quantized
weights. If we look in ggml-cuda.cu 
```c++
static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
```
```console
(cuda-gdb) p dst->ne
$27 = {4096, 2, 1, 1}
(cuda-gdb) p dst->name
$28 = "Qcur-0", '\000' <repeats 57 times>
(cuda-gdb) p dst->op  
$29 = GGML_OP_MUL_MAT

```

the activation will be quantized to Q8_1 and then the matrix multiplication will
be performed on those types.
Hmm, but this might also depended on the type of batch. If we are doing prompt
processing the operation might be MMQ instead. I need to understand this better.


__wip__


### Testing
```console
(gdb) p t1.ne
$19 = {4096, 16, 1, 1}
```

```c++
            double err = ud->tc->err(f1.data(), f2.data(), f1.size());
```

```c++
    // the number of elements that differ between the batched and the single-token results
    double err(const float * a, const float * b, size_t n) override {
        // We don't use the CPU results in this function, only the backends
        // output as we want to see if it agrees with itself.

        GGML_UNUSED(b);
        const size_t half = n/2;
        size_t ndiff = 0;
        double max_diff = 0.0;
        for (size_t i = 0; i < half; i++) {
            if (a[i] != a[half + i]) {
                ndiff++;
                max_diff = std::max(max_diff, (double) std::fabs(a[i] - a[half + i]));
            }
        }
        if (ndiff > 0) {
            printf("[%zu/%zu differ, max diff %g] ", ndiff, half, max_diff);
        }
        return ndiff;
    }
```


### ???
For this session I'm stepping through a test in test-backend-ops which I started
using the following options:
```console
gdb --args ${build_dir}/bin/${cmd} batch -b CUDA0 -o 'MUL_MAT(type_a=q4_K,m=4096,k=4096,n=8)'
```
And I'm setting an initial breakpoint in `ggml_cuda_mul_mat_q` (quantized):
```c++
void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        onst ggml_tensor * ids,
        ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
```
So lets see what we are working on here, src0 will be the weights matrix and
src1 the activations:
```console
(gdb) p src0->name
$1 = "a", '\000' <repeats 62 times>
(gdb) p src0->ne
$2 = {4096, 4096, 1, 1}
(gdb) p src0->type
$5 = GGML_TYPE_Q4_K

(gdb) p src1->name
$3 = "b", '\000' <repeats 62 times>
(gdb) p src1->ne
$4 = {4096, 8, 1, 1}
(gdb) p src1->type
$6 = GGML_TYPE_F32
```

Recall that the pattern of the local variables is that ne (number of elements),
then the index of the src index, followed by the dimension.
```c++
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;
```
```console
(gdb) p ne10_padded
$11 = 4096
(gdb) p s01
$12 = 16
(gdb) p s1
$13 = 4096
(gdb) p s02
$14 = 65536
(gdb) p s2
$15 = 32768
(gdb) p s03
$16 = 65536
(gdb) p s3
$17 = 32768
```

```c++
    const bool use_native_fp4 = prec_src1 == GGML_PREC_Q4;
    const size_t y_block_size       = use_native_fp4 ? sizeof(block_fp4_mmq) : sizeof(block_q8_1_mmq);
    const size_t y_values_per_block = use_native_fp4 ? QK_FP4_MMQ            : QK8_1_MMQ;
```
```console
(gdb) p use_native_fp4
$22 = false
(gdb) p y_block_size
$23 = 144
(gdb) p y_values_per_block
$24 = 128
```
```c++
    int J_best        = 0;
    int nthreads_best = 0;
    {
        int64_t ncols_opt = ne11;
        ...
        if (ctx.batch_invariant) {
            // Pin the smallest valid tile width regardless of the actual batch size so the
            // same template instantiation (and thus the same reduction order).
            ncols_opt = 1;
        }

        J_best = ggml_cuda_mmq_get_J_best(src0->type, ncols_opt, fallback, cc, smpbo, prec_src1, &nthreads_best);
```
```c++
static int ggml_cuda_mmq_get_J_best(
        const ggml_type type,
        const int64_t ncols,
        const bool fallback,
        const int cc,
        const size_t smpbo,
        const ggml_prec prec_src1,
        int * nthreads_best = nullptr) {
    int J_best        = 0;
    int ntiles_J_best = INT_MAX;

    for (int J = 8; J <= 128 && ntiles_J_best > 1; J += 8) {
        const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, fallback, cc, prec_src1);
        if (config.type == GGML_TYPE_COUNT) {
            continue;
        }

        if (mmq_get_nbytes_shared(config, cc) > smpbo) {
            continue;
        }

        const int ntiles_x = (ncols + config.J - 1) / config.J;

        if (ntiles_x < ntiles_J_best) {
            J_best = J;
            if (nthreads_best) {
                *nthreads_best = config.nthreads;
            }
            ntiles_J_best = ntiles_x;
        }
    }
    return J_best;
}
```
```console
(gdb) s
ggml_cuda_mmq_get_J_best (type=GGML_TYPE_Q4_K, ncols=1, fallback=false, cc=1210, smpbo=101376,
    prec_src1=GGML_PREC_Q8, nthreads_best=0xffffffffd65c)
    at /home/danbev/work/llama.cpp/ggml/src/ggml-cuda/mmq.cu:137
```
If we look at the loop above we can see that is getting a `ggml_cuda_mmq_config`
for the current type which is `GGML_TYPE_Q4_K`, cc which is 1210 (GB10), and
smpbo (SM per bytes something?)

A config has the following members:
```c++
struct ggml_cuda_mmq_config {
    ggml_type                 type;        // src0->type
    int                       nthreads;    // Number of threads per CUDA block.
    int                       occupancy;   // Targeted occupancy for the MMA kernel.
    int                       I;           // SRAM tile width in src0->ne[1]/dst->ne[0] direction.
    int                       J;           // SRAM tile width in src1->ne[1]/dst->ne[1] direction.
    ggml_cuda_mmq_sram_layout sram_layout; // SRAM tile length in src0->ne[0]/src1->ne[0] direction (physical 32 bit elements).
    int                       K_vram;      // VRAM tile length in src0->ne[0]/src1->ne[0] direction (logical elements).
    bool                      stream_k;    // Whether or not to use stream-k decomposition.
    bool                      fallback;    // Whether a fallback for out-of-bounds check in src0->ne[1] direction is needed.
```
```c++
static __host__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config(const ggml_type type, const int J, const bool fallback, const int cc, const ggml_prec prec_src1) {
    if (GGML_CUDA_CC_IS_AMD(cc)) {
        ...
    }

    if (blackwell_mma_available(cc)) {
        // only src1 at Q4 uses the native FP4 config, higher precisions keep src1 at Q8_1
        if (prec_src1 != GGML_PREC_Q4 && (type == GGML_TYPE_NVFP4 || type == GGML_TYPE_MXFP4)) {
            return ggml_cuda_mmq_get_config_ampere(type, J, fallback);
        }
        return ggml_cuda_mmq_get_config_blackwell(type, J, fallback);
    }
    ...
}
```
In `src/ggml-cuda/mmq-config-blackwell.cuh` we then have:
```c++
static constexpr __host__ __device__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config_blackwell(ggml_type type, int J, bool fallback) {
    CASE(GGML_TYPE_MXFP4, 256, 1, 128,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, true);
    ...
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, true);
    ...
    return ggml_cuda_mmq_get_config_ampere(type, J, fallback);
```
Where the CASE macro is defined back in mmq.cuh.
```c++
#define CASE(type_, nthreads_, occupancy_, I_, J_, sram_layout_, K_vram_, stream_k_, fallback_)                                           \
    if (type == (type_) && J == (J_) && fallback == (fallback_)) {                                                                        \
        static_assert((nthreads_) %  32 == 0 && (nthreads_)       <= 512, "bad nthreads");                                                \
        static_assert(                          (occupancy_)      <=   8, "bad occupancy");                                               \
        static_assert((I_)        %  32 == 0,                             "bad I");                                                       \
        static_assert((J_)        %   8 == 0 && (J_)              <= 128, "bad J");         \
        static_assert((K_vram_)   % 256 == 0,                             "bad K_vram"); \
        return ggml_cuda_mmq_config((type_), (nthreads_), (occupancy_), (I_), (J_), (sram_layout_), (K_vram_), (stream_k_), (fallback_)); \
    }
```
Notice that these cases are specifying the type, number of threads, occupancy
(which I need to understand better), I, J, the layout of shared memory, K_vram(?)
and stream_k (?), and the fallback (for when we don't have exactly multiples and
need to compute the remaining).

```console
(gdb) p config
$25 = {
    type = GGML_TYPE_Q4_K,
    nthreads = 256,
    occupancy = 1,
    I = 128,
    J = 8,
    sram_layout = GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1,
    K_vram = 256,
    stream_k = true,
    fallback = false}
```

Notice how this loop will start at J=8, and also jump 8 steps for each iteration.
And for each iteration it will call `get_config` passing in J which is the only
thing that will change.
```c+
    for (int J = 8; J <= 128 && ntiles_J_best > 1; J += 8) {
        const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, fallback, cc, prec_src1);
        if (config.type == GGML_TYPE_COUNT) {
            continue;
        }
```
What we are doing here is deciding the shape of the box/tile of the output that
each thread block computes. One side of this box is already fixed, this is I
which we can see above is 128. This is choosing the other size J which is the
number of columns.
Recall that dst is our output (C matrix):
```console
(gdb) p dst->ne
$26 = {4096, 8, 1, 1}
```

Next we check if this config will fit in share memory (shared memory per block,
opt-in smpbo):
```c++
        if (mmq_get_nbytes_shared(config, cc) > smpbo) {
            continue;
        }
```
So `mmq_get_bytes_shared` is asking how must shared memory do we need:
```c++
static size_t mmq_get_nbytes_shared(const ggml_cuda_mmq_config & config, const int cc) {
    const size_t nbs_ids = config.J*sizeof(int);
    const size_t nbs_x = ggml_cuda_mmq_get_nbytes_shared_x(config, cc);
    const size_t nbs_y = config.J * (sizeof(block_q8_1_mmq));
    return nbs_ids + nbs_x + GGML_PAD(nbs_y, config.nthreads*sizeof(int));
}
```
J=8 means we have/want/need 8 colums of the output tile C, so 8 tokens. And 
a tile is IxJ = 128 rows * 8 columns:
```console
         J=8 colums (tokens)
        c0 c1 c2 c3 c4 c5 c6 c7
row 0   [. .  .  .  .  .  .  . ]
row 1   [. .  .  .  .  .  .  . ]  I = 128 rows
...
row 127 [. .  .  .  .  .  .  . ]

128*8 = 1024 output cells per block.
```
But the output cells is not what goes into shared memory, it holds the inputs
the block needs to compute those cells. So shared memory will contain a tile
of A and a tile of B.
`nbs_ids` will be 32 in our case.

```c++
static __host__ int ggml_cuda_mmq_get_nbytes_shared_x(const ggml_cuda_mmq_config & config, const int cc) {
    if (config.use_mma_data_layout(cc)) {
        return config.I * ggml_cuda_mmq_get_sram_stride(config.sram_layout) * 4;
    }
    const tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(config.type, config.I);
    return (txs.qs + txs.dm + txs.sc) * 4;
}
```
```c++
#define MMQ_TILE_NE_K 32
#define QI8_0 (QK8_0 / (4 * QR8_0))
#define QK8_0 32
#define QR8_0 1

static constexpr __host__ __device__ int ggml_cuda_mmq_get_sram_stride(ggml_cuda_mmq_sram_layout sram_layout) {
    switch (sram_layout) {
        case GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0:
            return 2*MMQ_TILE_NE_K + 2*MMQ_TILE_NE_K/QI8_0 + 4;

        case GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1:
            return 2*MMQ_TILE_NE_K + 2*MMQ_TILE_NE_K/QI8_1 + 4;
            ...

        default:
            return -1;
    }
}
```
Back in mmq.cu we have:
```c++
    if (!ids) {
        const int ntj = (ne1 + J_best - 1) / J_best;
```
This is number of tiles in the j direction, that is how many tile colums are
needed to cover all the output colums. Which is only one in our case.

```c++
        const size_t nbytes_src1_q8_1 = ne13*ne12 * ne11*ne10_padded *
            y_block_size/y_values_per_block + src1_q8_1_padding;
        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);
        ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());
```
This buffer holds src1, the activations, converted to q8_1. The mmq kernel needs
int8 values on both sides so before mul_mat_q runs the `quantize_mmq_q8_1`
kernel converts the F32 activations into this buffer. The MMQ kernel will then
copy peices of it into its shared tile_y in `mul_mat_q_process_tile).

```c++
        {
            const int64_t s11 = src1->nb[1] / ts_src1;
            const int64_t s12 = src1->nb[2] / ts_src1;
            const int64_t s13 = src1->nb[3] / ts_src1;
            if (use_native_fp4) {
                ...
            } else {
                quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded,
                                       ne11, ne12, ne13, stream);
            }
```
```console
(cuda-gdb) p num_blocks
$2 = {x = 8, y = 8, z = 1}

(cuda-gdb) p block_size
$3 = {x = 128, y = 1, z = 1}
```
So we can see here that we have 8*8=64 blocks. Each of those 64 blocks has
128 threads. And threads are grouped in 32 so we have 128/32=4 warp.

And recall that a block always runs on a single SM which makes sense as shared
memory is per SM and all threads in the block can communitcate using it and by
synchronizing with `__syncthread()`


Setting a break point in the `quantize_mmq_q8_1_cuda`, make sure you are using
cuda-gdb and also the correct type of build, and set a breakpoint on a non-const
expr line.
```console
```

We can list the blocks using:
```console
(cuda-gdb) info cuda blocks
  BlockIdx   State Dev SM
Kernel 0
*  (0,0,0) running   0  0
   (1,0,0) running   0  1
   (2,0,0) running   0  8
   (3,0,0) running   0  9
   (4,0,0) running   0 16
   (5,0,0) running   0 17
   (6,0,0) running   0 24
   (7,0,0) running   0 25
   (0,1,0) running   0 32
   (1,1,0) running   0 33
   (2,1,0) running   0 40
   (3,1,0) running   0 41
   (4,1,0) running   0  2
   (5,1,0) running   0  3
   (6,1,0) running   0 10
   (7,1,0) running   0 11
   (0,2,0) running   0 18
   (1,2,0) running   0 19
   (2,2,0) running   0 26
   (3,2,0) running   0 27
   (4,2,0) running   0 34
   (5,2,0) running   0 35
   (6,2,0) running   0 42
   (7,2,0) running   0 43
   (0,3,0) running   0  4
   (1,3,0) running   0  5
   (2,3,0) running   0 12
   (3,3,0) running   0 13
   (4,3,0) running   0 20
   (5,3,0) running   0 21
   (6,3,0) running   0 28
   (7,3,0) running   0 29
   (0,4,0) running   0 36
   (1,4,0) running   0 37
   (2,4,0) running   0 44
   (3,4,0) running   0 45
   (4,4,0) running   0  6
   (5,4,0) running   0  7
   (6,4,0) running   0 14
   (7,4,0) running   0 15
   (0,5,0) running   0 22
   (1,5,0) running   0 23
   (2,5,0) running   0 30
   (3,5,0) running   0 31
   (4,5,0) running   0 38
   (5,5,0) running   0 39
   (6,5,0) running   0 46
   (7,5,0) running   0 47
   (0,6,0) running   0  0
   (1,6,0) running   0  1
   (2,6,0) running   0  8
   (3,6,0) running   0  9
   (4,6,0) running   0  2
   (5,6,0) running   0  3
   (6,6,0) running   0 10
   (7,6,0) running   0 11
   (0,7,0) running   0  4
   (1,7,0) running   0  5
   (2,7,0) running   0 12
   (3,7,0) running   0 13
   (4,7,0) running   0  6
   (5,7,0) running   0  7
   (6,7,0) running   0 14
   (7,7,0) running   0 15
```
Notice that some blocks are on the same SMs. And how many blocks can run is
detemined by how much resources that the kernel need, like number of registers,
shared memory etc. With this kernel that does not use any shared memory and very
few registers we can fit many onto each SM.


The GPU does not start all blocks at once. A hardware block scheduler hands
blocks to SMs as long as they have room (registers, shared mem, warp slots etc).
The remaining blocks wait in a queue. Whenever a block finishes the next one from
the queue starts on that SM. So we only see a subset that are currently running.

```console
(cuda-gdb) info cuda kernels
  Kernel Parent Dev Grid Status       SMs Mask GridDim  BlockDim Invocation
*      0      -   0    1 Active 0xffffffffffff (8,8,1) (128,1,1) quantize_mmq_q8_1<(mmq_q8_1_ds_layout)1, false>(x=0x330d02000, ids=0x0 <_INTERNAL_a1696bc1_11_quantize_cu_ba19f1fc::cudaTriggerProgrammaticLaunchCompletion()>, vy=0x340000000, ne00=4096, s01=4096, s02=32768, s03=32768, ne0=4096, ne1=8, ne2=1, n_expert_used=0)
```

```console
(cuda-gdb) set cuda coalescing off

```

So we can inspect the current block using:
```console
(cuda-gdb) cuda block
block (0,0,0)
```
And then use the following command to list all the threads in this block:
```console
(cuda-gdb) info cuda threads block (0,0,0)
  BlockIdx ThreadIdx                 PC Dev SM Wp Ln                                                   Filename  Line
Kernel 0
*  (0,0,0)   (0,0,0) 0x0000000325b66d20   0  0  0  0 /home/danbev/work/llama.cpp/ggml/src/ggml-cuda/quantize.cu   465
   (0,0,0)   (1,0,0) 0x0000000325b66d20   0  0  0  1 /home/danbev/work/llama.cpp/ggml/src/ggml-cuda/quantize.cu   465
   (0,0,0)   (2,0,0) 0x0000000325b66d20   0  0  0  2 /home/danbev/work/llama.cpp/ggml/src/ggml-cuda/quantize.cu   465
   (0,0,0)   (3,0,0) 0x0000000325b66d20   0  0  0  3 /home/danbev/work/llama.cpp/ggml/src/ggml-cuda/quantize.cu   465
   (0,0,0)   (4,0,0) 0x0000000325b66d20   0  0  0  4 /home/danbev/work/llama.cpp/ggml/src/ggml-cuda/quantize.cu   465
...
(0,0,0) (124,0,0) 0x0000000325b66d20   0  0  3 28 /home/danbev/work/llama.cpp/ggml/src/ggml-cuda/quantize.cu   465 
   (0,0,0) (125,0,0) 0x0000000325b66d20   0  0  3 29 /home/danbev/work/llama.cpp/ggml/src/ggml-cuda/quantize.cu   465 
   (0,0,0) (126,0,0) 0x0000000325b66d20   0  0  3 30 /home/danbev/work/llama.cpp/ggml/src/ggml-cuda/quantize.cu   465 
   (0,0,0) (127,0,0) 0x0000000325b66d20   0  0  3 31 /home/danbev/work/llama.cpp/ggml/src/ggml-cuda/quantize.cu   465
```
So here we can see that we have 128 threads, and we have 3 warps (0..3) which of
which have 32 threads each.

Now, for shared memory and when using mmq (quantizied computation) the data in
shared memory will be in different formats. Each types tile loader unpacks
them into a format that the tensor-core int8 instructions can use directly.

To switch threads in the same block we can use the following command:
```c++
(cuda-gdb) cuda thread (3,0,0)
```
This is nice as thread (0,0,0) often has 0 as all its local variable values.

Recall that this kernel is about quantizing the activations from F32 to q8_1
and write the results into the global buffer src1_q8_1.
Each thread will handle 4 consecutive values or one column, one float4. And we
also have a scale value that covers 32 values.
1 thread  = 4 values   loads one float4, computes its local max |x|
8 threads = 32 values  share one scale. __shfl_xor_sync exchanges the max

I'm finding that the parameter names are somewhat difficult to follow. This is
what the quantization function looks like where we have names like x, yv, ne00,
s01, s02, s03, ne0, ne1, ne2:
```c++
quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded,
                                       ne11, ne12, ne13, stream);

void quantize_mmq_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {

template <mmq_q8_1_ds_layout ds_layout, bool scatter>
static __global__ void quantize_mmq_q8_1(
        const float * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2, const int n_expert_used) {
```
So x is src1 the activations, yv is the global device buffer  src1_q8_1.

```c++
    const float4 * x4 = (const float4 *) x;
```
So this is a pointer to the activation values in float4.

```c++
    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;
```
And vy is a pointer to the device global memory where the quantized data will
be stored.

Then we have:
```c++
    const int64_t k_block = i0 / QK8_1_MMQ; // column block in the channel
    const int64_t iqs     = i0 % QK8_1_MMQ; // quant index in block

    const float4 xi = i0 < ne00 ? x4[(base_idx + i00)/4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
```
The above is checking for real data or if we should pad with just 0.0. make_float4
is a cuda library function (or perhaps a macro).
The base_idx is just where this token's row start in x counted in floats.

And then we calculate the max value of these 4 floats, calling fabsf with the
same values so that the max is preserved.
```c++
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));
```
Then we have the following loop which will iterate over 4, 2, and 1 offsets. The
intrinsic function here lets threads within the same warp read a register directly
from a partner thread whose lane index is calculate by a bitwise XOR
(lane_id ^ lane_mask). 
Lets looks at an example:
```console
T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width = warpSize);

lane:             0 1 2 3 4 5 6 7
start (own amax)  3 9 1 4 7 2 8 5

offset 4: partner = lane ^ 0100
                 0  0000 ^ 0100 = 0100 = 4
                 1  0001 ^ 0100 = 0101 = 5
                 2  0010 ^ 0100 = 0110 = 6
                 3  0011 ^ 0100 = 0111 = 7
                 4  0100 ^ 0100 = 0000 = 0
                 5  0101 ^ 0100 = 0001 = 1
                 6  0110 ^ 0100 = 0010 = 2
                 7  0111 ^ 0100 = 0011 = 3

lane:             0 1 2 3 4 5 6 7
start (own amax)  3 9 1 4 7 2 8 5

So lane 0 and 4 will swap values (keeping the largest):
lane:             0 1 2 3 4 5 6 7
                  7 9 1 4 7 2 8 5

And lane 1 and 5 will swap values:
lane:             0 1 2 3 4 5 6 7
                  7 9 1 4 7 9 8 5

And lane 2 and 6 will swap values:
lane:             0 1 2 3 4 5 6 7
                  7 9 8 4 7 9 8 5

And lane 3 and 7 will swap values:
lane:             0 1 2 3 4 5 6 7
                  7 9 8 5 7 9 8 5

For the next round the offset/mask will be 2:
offset 2: partner = lane ^ 0010
                 0  0000 ^ 0010 = 0010 = 2
                 1  0001 ^ 0010 = 0011 = 3
                 2  0010 ^ 0010 = 0000 = 0
                 3  0011 ^ 0010 = 0001 = 1
                 4  0100 ^ 0010 = 0110 = 6
                 5  0101 ^ 0010 = 0111 = 7
                 6  0110 ^ 0010 = 0100 = 4
                 7  0111 ^ 0010 = 0101 = 5

lane:             0 1 2 3 4 5 6 7
                  7 9 8 5 7 9 8 5
So lane 0 and 2 will swap values (keeping the largest):
                  8 9 8 5 7 9 8 5

And lane 1 and 3 will swap values:
                  8 9 8 9 7 9 8 5

And lane 2 and 0 will swap values:
                  8 9 8 9 7 9 8 5
And lane 3 and 1 will swap values:
                  9 9 8 9 7 9 8 5
And lane 4 and 6 will swap values:
                  9 9 8 9 8 9 8 5
And lane 5 and 7 will swap values:
                  9 9 8 9 8 9 8 9
And lane 6 and 4 will swap values:
                  9 9 8 9 8 9 8 9
And lane 7 and 5 will swap values:
                  9 9 8 9 8 9 8 9

For the next round the offset/mask will be 1:
offset 1: partner = lane ^ 0001
                 0  0000 ^ 0001 = 0000 = 1
                 1  0001 ^ 0001 = 0000 = 0
                 2  0010 ^ 0001 = 0011 = 3
                 3  0011 ^ 0001 = 0010 = 2
                 4  0100 ^ 0001 = 0101 = 5
                 5  0101 ^ 0001 = 0100 = 4
                 6  0110 ^ 0001 = 0111 = 7
                 7  0111 ^ 0001 = 0110 = 6

So lane 0 and 1 will swap values (keeping the largest):
                  9 9 8 9 8 9 8 9
And lane 1 and 0 will swap values:
                  9 9 8 9 8 9 8 9
And lane 2 and 3 will swap values:
                  9 9 9 9 8 9 8 9
And lane 3 and 2 will swap values:
                  9 9 9 9 8 9 8 9
And lane 4 and 5 will swap values:
                  9 9 9 9 9 9 8 9
And lane 5 and 4 will swap values:
                  9 9 9 9 9 9 8 9
And lane 6 and 7 will swap values:
                  9 9 9 9 9 9 9 9
And lane 7 and 6 will swap values:
                  9 9 9 9 9 9 9 9
```
So we can see that that after this all threads will have the same value.

```c++
    for (int offset = vals_per_scale/8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }
```
Following have the quantization of 4 floats. And recall that the to quantize
we divide each float by a scale, d = amax/127.0. We are quantizing the floats
into [-127, 127] integers by round(f_i / d) but to avoid division we use multiplication
by the inverse.
```console
    const float d_inv = 127.0f / amax;   // 1/d
    char4 q;
    q.x = roundf(xi.x * d_inv);   // quantize float 0
    q.y = roundf(xi.y * d_inv);   // quantize float 1
    q.z = roundf(xi.z * d_inv);   // quantize float 2
    q.w = roundf(xi.w * d_inv);   // quantize float 3

    const float d = 1.0f / d_inv; // perform the division upfront so we can
                                  // later dequantize using multiplication.
```

Next we are using `__shfl_xor_sync` with the same butterfly(offset) as before
but this time we are adding so every thread in an 8-lane group will hold the
sum of the 32 `quantized` values.
Keep in mind that we are quantizing the activations (src1) to q8_1, but for a
matmul with a Q4_K W (src0, the weights), `dst = W * X` where W is Q4_K.
So Q4_K is asymmetric (both scale and min), while Q8_1 is symmetric (scale only).

When we take the dot product between an asymmetric weight sub-block and a
symmetric activation sub-block, factoring out the constants produces a cross-term
that depends only on the sum of the quantized activations.
```console
Weight:
w_i = a * q_w_i + b

Activation:
x_i = d_x * q_x_i


The operation we want to perform later is matmul which becomes:

    31
    Σ   (a * q_w_i + b) (d_x * q_x_i)
    i=0

Expanded:
    31
    Σ   (a * d_x * q_w_i * q_x_i + b * d_x * q_x_i)
    i=0

Rewritten:
            31                         31
    a * d_x Σ   q_w_i * q_x_i + b (d_x Σ   q_x_i)
            i=0                        i=0
                                  {offset correction}
```
Notice the last term, this is what the sum below is all about:
```c++
    float sum = 0.0f;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        // integer addition as we are summing the quantized (int values) values.
        int sum_q = int(q.x) + int(q.y) + int(q.z) + int(q.w);

        for (int offset = vals_per_sum/8; offset > 0; offset >>= 1) {
            sum_q += __shfl_xor_sync(0xFFFFFFFF, sum_q, offset, WARP_SIZE);
        }

        // cast sum_q which are integer values.
        sum = d * float(sum_q);
    }
```
By doing it this was the matmul only has to do a a single `b * sum` in the epilog
instead of per operation. We are doing it here upfront. And this allows the
MMQ kernel never has to sum q_x for every weight row. And the inner loop is
pure raw integer multiplication.
SO we have to store a scale (delta) and a sum for each 32 elements.

So next we need to write these quantized values to the global memory that is
pointed to by `vy` and which `y` is also a pointer to but not a raw pointer but:
```console
cuda-gdb) ptype y
type = @generic struct block_q8_1_mmq {
    @generic __C13;
    @generic int8_t qs[128];
} * @register
```

All threads write the quantized data to the global buffer:
```c++
        // Write back 4 int8 values as a single 32 bit value for better memory bandwidth:
        char4 * yqs4 = (char4 *) y[ib].qs;
        yqs4[iqs/4] = q;
```

But the scale and sum only applies to a group of elements, 32. So not all thread
should write this information as it is shared by the group.
```
        if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
            if (iqs % 16 == 0 && iqs < 96) {
                y[ib].d2s6[2 + iqs/16] = sum;
                if (iqs % 64 == 0) {
                    y[ib].d2s6[iqs/64] = d;
                }
            }
        } else if (iqs % 32 == 0) {
            if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
                y[ib].ds4[iqs/32] = make_half2(d, sum);
            } else {
                y[ib].d4[iqs/32]  = d;
            }
        }
```
So one `block_q8_1_mmq` stores 128 value of one token:
```c++
#define QK8_1 32
static constexpr int QK8_1_MMQ  = 4*QK8_1;
struct block_q8_1_mmq {
    half2 ds4[4];         // delta and sum pairs in one half2
    int8_t qs[QK8_1_MMQ]; //int8_t qs[4 * 32 = 128];
};
```

And that is the last of the quantization. At this point the quantized activations
are now available in global memory. And so that kernel will finish execution
and we will return to the host code in quantize.cu:
```c++
void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    ...

                quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded,

        const int64_t s12 = use_native_fp4 ?
                                ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int)) :
                                ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
        const int64_t s13 = ne12*s12;

        const mmq_args args = {
            src0_d, src0->type, (const int *) src1_q8_1.ptr, nullptr, nullptr, dst_d,
            src0->type == GGML_TYPE_NVFP4 && use_native_fp4 ? src1_scale.ptr : nullptr,
            ne00, ne01, ne1, s01, ne11, s1,
            ne02, ne12, s02, s12, s2,
            ne03, ne13, s03, s13, s3,
            ntj, J_best, fusion_device};
        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream, prec_src1);
        return;
```
So this is setting up the mmq arguments that will be passed into the switch type
function. If we look at the mmq_args struct we can see the following:
```console
(cuda-gdb) set print symbol off
(cuda-gdb) p args              
$10 = {
  x = 0x330401000 "",
  type_x = GGML_TYPE_Q4_K,
  y = 0x340000000,
  ids_dst = 0x0,
  tile_spec = 0x0,
  dst = 0x330d23000,
  y_scale = 0x0,
  ncols_x = 4096,
  nrows_x = 4096,
  ncols_dst = 8,
  stride_row_x = 16,
  ncols_y = 8,
  nrows_dst = 4096,
  nchannels_x = 1,
  nchannels_y = 1,
  stride_channel_x = 65536,
  stride_channel_y = 9216,
  stride_channel_dst = 32768,
  nsamples_x = 1,
  nsamples_y = 1,
  stride_sample_x = 65536,
  stride_sample_y = 9216,
  stride_sample_dst = 32768,
  ntj = 1,
  J_best = 8,
  fusion = {
    x_bias = 0x0,
    gate = 0x0,
    gate_bias = 0x0,
    x_scale = 0x0,
    gate_scale = 0x0,
    glu_op = GGML_GLU_OP_REGLU,
    glu_limit = 0,
    shared_up = 0x0,
    shared_gate = 0x0,
    shared_dst = 0x0,
    shared_stride_col_dst = 0
  }
}
```
Just a note about the naming. Notice we have 'channels' and 'samples' for dimension
2 and 3. These terms come from image tensors where NCHW where N samplese, C
channels, height, width).
```console
ggml dim:  ne[0]      ne[1]    ne[2]        new[3]
           columns    rows     "channels"   "samples"
```
And `x` is just a pointer to raw bytes and not a string, it is set as src0_d, the
pointer to weights which can be any quantized type like Q4_K, Q6_K, Q8_0 etc.

So we will then call `ggml_cuda_mul_mat_q_switch_type`, which is part for the
runtime "switching" to the correct template specializtion for the current types
we are dealing with:
```c++
static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream, const ggml_prec prec_src1) {
    switch (args.type_x) {
        ...
        case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K, GGML_PREC_Q8>(ctx, args, stream);
            break;
        ...
```
So this is checking the type of the weights tensor (src0->type) and in our
case this was `GGML_TYPE_Q4_K` and we further switch (not sure if that is the
proper name for it) to further "filter/detect" the kernel that we eventually
want to run. So we are calling a specialization of mul_mat_q_case template for
the `GGML_TYPE_Q4_K, and GGML_PREC_Q8. 

So the templete function looks like this:
```c++
template <ggml_type type, ggml_prec prec_src1>
void mul_mat_q_case(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    if (ggml_cuda_mmq_needs_fallback(args.nrows_x)) {
        constexpr bool fallback = true;
        mul_mat_q_switch_fusion<type, fallback, prec_src1>(ctx, args, stream);
    } else {
        constexpr bool fallback = false;
        mul_mat_q_switch_fusion<type, fallback, prec_src1>(ctx, args, stream);
    }
}

#define DECL_MMQ_CASE(type)                                                        \
    template void mul_mat_q_case<type, GGML_PREC_Q8>(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) \

extern DECL_MMQ_CASE(GGML_TYPE_Q4_K);
```

```c++
static __host__ bool ggml_cuda_mmq_needs_fallback(const int64_t nrows_x) {
    return nrows_x % 128 != 0;
}
```
Notice the usage of `constexpr` above so that if/else is actually calling two
different specializations of the mul_mat_q_switch_fusion template. The args.nrows.x
is a runtime value. So the is just the normal way to test a runtime value and
then call the matching compile time instance.

That takes us down into:
```c++
template <ggml_type type, bool fallback, ggml_prec prec_src1>
static void mul_mat_q_switch_fusion(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    if constexpr (!fallback && ggml_cuda_mmq_fusion_supported(type)) {
        if (args.fusion.gate) {
            mul_mat_q_switch_J<type, fallback, prec_src1, true>(ctx, args, stream);
            return;
        }
    }
    GGML_ASSERT(!args.fusion.gate);
    mul_mat_q_switch_J<type, fallback, prec_src1, false>(ctx, args, stream);
}
```
In this case `args.fusion.gate` is null so we will not take that branch and
instead call the last line. And like before we are calling specialized templates
but here just using a constant instead of a named constexpr constant. So false
here is the `has_fusion` template parameter.

```c++
template <ggml_type type, bool fallback, ggml_prec prec_src1, bool has_fusion>
void mul_mat_q_switch_J(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    switch (args.J_best) {
        case   8:
            launch_mul_mat_q<type,   8, fallback, prec_src1, has_fusion>(ctx, args, stream);
            break;
```
And we are now about to launch the matrix multiplication kernel:
```c++
template <ggml_type type, int J, bool fallback, ggml_prec prec_src1, bool has_fusion>
static void launch_mul_mat_q(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    const int nsm = ggml_cuda_info().devices[id].nsm;
    const int warp_size = ggml_cuda_info().devices[id].warp_size;

    const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, fallback, cc, prec_src1);
```
```console
(cuda-gdb) p id
$5 = 0
(cuda-gdb) p cc
$6 = 65535
(cuda-gdb) p nsm
$7 = 48
(cuda-gdb) p warp_size
$8 = 32

(cuda-gdb) p config
$9 = {type = GGML_TYPE_Q4_K, nthreads = 256, occupancy = 1, I = 128, J = 8,
  sram_layout = GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, K_vram = 256, stream_k = true, fallback = false}

```
```c++
    const int nwarps = config.nthreads / warp_size;
    const int nbytes_shared = mmq_get_nbytes_shared(config, cc);
```
```console
(cuda-gdb) p nwarps
$10 = 8
(cuda-gdb) p nbytes_shared
$11 = 40992
```

```c++
    const dim3 block_dims(warp_size, nwarps, 1);
```
```console
(cuda-gdb) p block_dims
$12 = {x = 32, y = 8, z = 1}
```

Next we have a macro to set the shared memory limit. Now, notice that we have
two calls here. The first one is calling the macro with fallback set to false
and also passing in has_fusion, and the second one is conditioned on has_fusion
and then passing in true as the fallback. But this function also has fallback
as a template parameter:
```c++
template <ggml_type type, int J, bool fallback, ggml_prec prec_src1, bool has_fusion>
    ...
    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q<type, J, /*fallback =*/ false, prec_src1, has_fusion>), nbytes_shared);
    if constexpr (!has_fusion) {
        CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q<type, J, /*fallback =*/ true, prec_src1, false>), nbytes_shared);
    }

#    define CUDA_SET_SHARED_MEMORY_LIMIT(kernel, nbytes)                                                       \
        do {                                                                                                   \
            static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = { false };                         \
            const int   id                                                = ggml_cuda_get_device();            \
            if (!shared_memory_limit_raised[id]) {                                                             \
                CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes)); \
                shared_memory_limit_raised[id] = true;                                                         \
            }                                                                                                  \
        } while (0)
#else
#    define CUDA_SET_SHARED_MEMORY_LIMIT(kernel, nbytes) \
        do {                                             \
            GGML_UNUSED(nbytes);                         \
        } while (0)
#endif // !defined(GGML_USE_HIP)
```
A kernel may only use more that 48KB of shared memory after having called 
cudaFuncSetAttribute with cudaFuncAttributeMaxDynamicSharedMemorySize.

And the macro keeps track of the device which has called this using
shared_memory_limit_raised with the device id as the index.

So currently the code will call this for the kernel and the n_bytes are the same
in both calls. The kernel passed is just the signature of mul_mat_q with the
template parameters specified so these are two separate kernels. But this
function only calls one of these function not both
(this was not always the case though):
```c++
        mul_mat_q<type, J, fallback, prec_src1, has_fusion><<<block_nums_xy_tiling, block_dims, nbytes_shared, stream>>>
        ...
    mul_mat_q<type, J, fallback, prec_src1, has_fusion><<<block_nums_stream_k, block_dims, nbytes_shared, stream>>>
```
The fallback template parameter as added in Commit 6eddde06a4f25d55d538b5d15628dcc2b6882147
("CUDA: refactor MMQ kernel configuration (#24127)") g
```console
-template <ggml_type type, int mmq_x>
+template <ggml_type type, int J, bool fallback>

 static void launch_mul_mat_q(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
```
And later the `has_fusion` template parameter was also added in Commit
c49edf0963a76d5e78b2a7278c9bd5b71d26c84f ("CUDA: fuse FFN gate/up matmuls and GLU in MMQ with interleaved gate/up rows (#29948)")  which refactored this code to have a constexpr if case for
the non-fusion templates. 

So at some point this `launch_mul_mat_q` actually did call different kernels but
during refactoring this has changes and now this is decided earlier in the call
chain but this code was just missed. My understanding of this that previously
there were multiple kernels called by this function but after refactoring and
the addition of template parameters it only calls a single specialized template
and it only needs to set the shared memory limit for one and not both.

_wip_


So how we get from one row to the next can differ depending on the layout:
```console
(gdb) p config.sram_layout
$29 = GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1
```
The weight tile in shared memory is a 2D array of 32-bit words. It has I rows
and each stride word long.

```c++
#define MMQ_TILE_NE_K 32
#define MMQ_TILE_Y_K     (MMQ_TILE_NE_K + MMQ_TILE_NE_K / QI8_1)

static constexpr __host__ __device__ int ggml_cuda_mmq_get_sram_stride(ggml_cuda_mmq_sram_layout sram_layout) {
    switch (sram_layout) {
        case GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0:
            return 2*MMQ_TILE_NE_K + 2*MMQ_TILE_NE_K/QI8_0 + 4;

        case GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1:
            return 2*MMQ_TILE_NE_K + 2*MMQ_TILE_NE_K/QI8_1 + 4;
        //             64          + 64/8                  + 4 = 76 words

        case GGML_CUDA_MMQ_SRAM_LAYOUT_Q2_K:
            return 2*MMQ_TILE_NE_K + MMQ_TILE_NE_K         + 4;
        case GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K:
            return 2*MMQ_TILE_NE_K + MMQ_TILE_NE_K/2       + 4;
        case GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K:
            return 2*MMQ_TILE_NE_K + MMQ_TILE_NE_K/QI6_K   + MMQ_TILE_NE_K/8 + 7;
        case GGML_CUDA_MMQ_SRAM_LAYOUT_FP4:
            return 2*MMQ_TILE_NE_K + 8                     + 4;
        case GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4:
            return 2*MMQ_TILE_NE_K + MMQ_TILE_NE_K/2       + 4;
        default:
            return -1;
    }
}
```
