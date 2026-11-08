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
