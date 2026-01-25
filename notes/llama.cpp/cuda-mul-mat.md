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
static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx,
                              const ggml_tensor * src0,
                              const ggml_tensor * src1,
                              ggml_tensor * dst) {
    const bool split = ggml_backend_buft_is_cuda_split(src0->buffer->buft);

    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE
        && ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) && src0->view_src;

    bool use_mul_mat_vec_f = (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;
    bool use_mul_mat_f     = !ggml_is_quantized(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;
    bool use_mul_mat_vec_q = ggml_is_quantized(src0->type) && !bad_padding_clear
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32
        && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;
    bool use_mul_mat_q     = ggml_is_quantized(src0->type) && !bad_padding_clear
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;
```
Depending on the src0 tensor type we will be calling different matrix
multiplication.
```console
(cuda-gdb) p use_mul_mat_vec_f
$2 = true
(cuda-gdb) p use_mul_mat_f
$3 = true
```
Next, depending on if we have split GPUs there are different path taken:
```c++
    if (split) {
        ggml_backend_cuda_split_buffer_type_context * buft_ctx = (ggml_backend_cuda_split_buffer_type_context *) src0->buffer->buft->context;
        auto & tensor_split = buft_ctx->tensor_split;
        for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
            // skip devices that are not going to do any work:
            if (tensor_split[id] >= (id + 1 < ggml_backend_cuda_get_device_count() ? tensor_split[id + 1] : 1.0f)) {
                continue;
            }

            const int cc            = ggml_cuda_info().devices[id].cc;
            const int warp_size     = ggml_cuda_info().devices[id].warp_size;
            use_mul_mat_q           = use_mul_mat_q             && ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[1], /*n_experts=*/0);
            use_mul_mat_f           = use_mul_mat_f             && ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, src1->ne[1], /*mul_mat_id=*/false);
            use_mul_mat_vec_f       = use_mul_mat_vec_f         && ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, src1->ne[1]);
            any_gpus_with_slow_fp16 = any_gpus_with_slow_fp16   || !fast_fp16_hardware_available(cc);
        }
    } else {
        const int cc            = ggml_cuda_info().devices[ctx.device].cc;
        const int warp_size     = ggml_cuda_info().devices[ctx.device].warp_size;
        use_mul_mat_q           = use_mul_mat_q             && ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[1], /*n_experts=*/0);
        use_mul_mat_f           = use_mul_mat_f             && ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, src1->ne[1], /*mul_mat_id=*/false);
        use_mul_mat_vec_f       = use_mul_mat_vec_f         && ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, src1->ne[1]);
        any_gpus_with_slow_fp16 = any_gpus_with_slow_fp16   || !fast_fp16_hardware_available(cc);
    }
```
The above is figuring out which multiplication kernel to use based on the GPU
capabilities. MMQ is the matrix multiplication for quantized data, MMF is for
floating point data, and MMVF is for vectorized floating point data.
```console
(cuda-gdb) p cc
$7 = 890
(cuda-gdb) p warp_size
$8 = 32

(cuda-gdb) p use_mul_mat_f
$9 = true
(cuda-gdb) p use_mul_mat_q
$10 = false
(cuda-gdb) p use_mul_mat_vec_f
$11 = false
```
Notice that this has norrowed the options down to just use_mul_mat_f being true.
Following that we have:
```c++
    const int cc                 = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    bool use_batched_cublas_f16  = src0->type == GGML_TYPE_F16 && (src1->type == GGML_TYPE_F16 || !any_gpus_with_slow_fp16);
    bool use_batched_cublas_bf16 = src0->type == GGML_TYPE_BF16 && bf16_mma_hardware_available(cc);
    bool use_batched_cublas_f32  = src0->type == GGML_TYPE_F32;

    if (!split && use_mul_mat_vec_f) {
        // the custom F16 vector kernel can be used over batched cuBLAS GEMM
        // but this is only faster for GPUs without tensor cores or with a thin src0 matrix (particularly KQV in attention)
        ggml_cuda_mul_mat_vec_f(ctx, src0, src1, nullptr, dst);
    } else if (!split && use_mul_mat_f) {
        ggml_cuda_mul_mat_f(ctx, src0, src1, nullptr, dst);
    } else if (!split && use_mul_mat_vec_q) {
        ggml_cuda_mul_mat_vec_q(ctx, src0, src1, nullptr, dst);
    } else if (!split && use_mul_mat_q) {
        ggml_cuda_mul_mat_q(ctx, src0, src1, nullptr, dst);
    } else if (!split && (use_batched_cublas_f16 || use_batched_cublas_bf16 || use_batched_cublas_f32)
        && !ggml_is_transposed(src0) && !ggml_is_transposed(src1) && src1->ne[2]*src1->ne[3] > 1) {
        // general KQ + KQV multi-batch without FlashAttention
        ggml_cuda_mul_mat_batched_cublas(ctx, src0, src1, dst);
    } else if (use_mul_mat_vec_f) {
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_mul_mat_vec_f, nullptr);
    } else if (use_mul_mat_vec_q) {
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_mul_mat_vec_q, quantize_row_q8_1_cuda);
    } else if (use_mul_mat_q) {
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_mul_mat_q, quantize_mmq_q8_1_cuda);
    } else {
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_mul_mat_cublas, nullptr);
    }
```
In this case we will be calling `ggml_cuda_mul_mat_f` which is defined in
mmf.cu:
```c++
void ggml_cuda_mul_mat_f(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(!ids ||  ids->type == GGML_TYPE_I32);
    GGML_ASSERT(         dst->type == GGML_TYPE_F32);


    GGML_TENSOR_BINARY_OP_LOCALS;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    ...

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;
```
And above we are just creating pointers to the data of the src1, ids, and dst
tensors.
Next we have:
```c++
    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = src1->nb[1] / ts_src1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s12 = src1->nb[2] / ts_src1;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s13 = src1->nb[3] / ts_src1;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const int64_t ids_s0 = ids ? ids->nb[0] / ggml_type_size(ids->type) : 0;
    const int64_t ids_s1 = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;
```
The `s` stands for stride and what this is doing is calculating the stride for
each dimension of the tensors. While we could use the nb array this would lead
to code like:
```c++
char data_bytes = (char )data;
size_t offset_bytes = 2 * nb[1] + 1 * nb[0];
float value = ((float )(data_bytes + offset_bytes));
```
But by using the approach in the code we can do:
```c++
size_t offset_elements = 2 * s1 + 1 * s0;
float value = data[offset_elements];
```
Next we have:
```c++
    mmf_ids_data ids_info{};
    mmf_ids_data * ids_info_ptr = nullptr;
```
The `mmf_ids_data` struct is defined in mmf.cu as:
```c++
struct mmf_ids_data {
    const int32_t * ids_src_compact = nullptr;
    const int32_t * ids_dst_compact = nullptr;
    const int32_t * expert_bounds_dev = nullptr;
    int n_experts = 0;
    int sis1 = 0;
};
```
And this is used for mixtures of experts to make the computation efficient. What
is done is that we first shuffle the tokens assigned to each experts so that
all tokens for expert 0 come first, then all tokens for expert 1, etc. With this
data now contiguous in memory we can do a single matrix multiplication per expert.
And the we unshuffle the results back to the original token order. This struct
stores information needed to do this. The `experts_bounds_dev` specifies where
the section for expoert 'i' starts in the shuffled data.
`ids_src_compact` is a list that tells the kernal how to build the shuffled input.

Next we have a few ggml_cuda_pool_allocators for the above data:
```c++
    ggml_cuda_pool_alloc<int32_t> ids_src_compact_dev;
    ggml_cuda_pool_alloc<int32_t> ids_dst_compact_dev;
    ggml_cuda_pool_alloc<int32_t> expert_bounds_dev;
```
This is a memory allocator for GPU device memory. So initially there is a large
block of global device memory allocated and then smaller chunks are handed out
by using an allocator.

Next we have:
```c++
    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_dst      = ids ? ne1 : ne2;

    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_col_y       = ids ? s12  : s11;
    const int64_t stride_channel_dst = ids ? s1 : s2;

    int64_t stride_channel_y         = ids ? s11  : s12;
    int64_t nchannels_y              = ids ? ne11 : ne12;
```
The term channel is a more general concept than row and represents and independant
stream of computation. For example when we are processing a batched `MUL_MAT` we
are processing a batch of matrices and each matrix in the batch is an independent
stream (a channel).
For example, lets say we have a batch of 4 matrices each has 16 rows and 512
columns:
```console
batch = [512, 16, 4]
```
So the number of channels, `n_channels_dst`, is 4, the number of batches/channels.
And the number of columns is 16, `n_cols_dst`.

But for `MUL_MAT_ID` we have something different. In this case we are processing
4 tokens, and for each of them we are producing an output vector of 512 features.
But we still want to use the same kernel.
```
batch = [512, 4]

nchannels_dst = 4
ncols_dst = 1 (this is a 2d tensor so the second dimension is 1)
```

```c++
    switch (src0->type) {
        ...
        case GGML_TYPE_F16: {
            const half2 * src0_d = (const half2 *) src0->data;
            constexpr int vals_per_T = 2;
            mul_mat_f_switch_cols_per_block(
                src0_d, src1_d, ids_d, dst_d, ne00/vals_per_T, ne01, ncols_dst, s01/vals_per_T, stride_col_y/vals_per_T, stride_col_dst,
                ids_s0, ids_s1, ne02, nchannels_y, nchannels_dst, s02/vals_per_T, stride_channel_y, stride_channel_dst,
                ne03, ne3, s03/vals_per_T, s13, s3, ctx.stream(), ids_info_ptr);
        } break;
```
mul_mat_f_switch_cols_per_block can be found in mmf.cuf:
```c++
template <typename T>
static void mul_mat_f_switch_cols_per_block(
        const T * x, const float * y, const int32_t * ids, float * dst,
        const int64_t ncols_x, const int64_t nrows_x, const int64_t ncols_dst,
        const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst,
        const int64_t stride_col_id, const int stride_row_id,
        const int64_t nchannels_x, const int64_t nchannels_y, const int64_t nchannels_dst,
        const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst, const int64_t nsamples_x,
        const int64_t nsamples_dst, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
        cudaStream_t stream, const mmf_ids_data * ids_data) {

    const int ncols_case = (ids && ncols_dst > 16) ? 16 : ncols_dst;

    GGML_ASSERT(ids || ncols_dst <= 16);

    switch (ncols_case) {
    ...
        case  7: {
            mul_mat_f_cuda<T,  7>(x, y, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row, stride_col_y, stride_col_dst,
                stride_col_id, stride_row_id, nchannels_x, nchannels_y,  nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, stream, ids_data);
        } break;
```
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
Now, B is not transposed explicitly but as we will discuss later the values are
extracted in a "transposed order".

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
and now we are doing to load the corresponding values from the B matrix (transposed)
so that we can perform the dot product on those elements. And we add the dot
products to the C matrix.

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
Instead the indexing pattern y2[j*stride_col_y + col] reads the data in a way
that logically transposes it. So the data in shared memory has the transposed
layout the matrix multiplication needs.

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

So we are reading from the first column of the transposed B matrix.

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

__wip__
