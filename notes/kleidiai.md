## Kleidiai
This is an open source library from ARMs to access CPU instructions that are optimized
for operations like matrix multiplication which are often found in machine learning and
AI applications. It is a little bit like intrinsics but provided as a library.

This micro kernels that the library provides often, perhaps always fused operations, like
if there is a matrix multiplication an activation function can be specified and will be
done as part of the same operation.

### Installation
There is an example in [kleidiai](../fundamentals/kleidiai) and there is a make recipe to
to install and build kleidiai.

### i8mm feature extension set
This is the feature set that consists of a number of instructions that are part of the CPU
instructions, so these need to be supported by the hardware. So there is no i8mm instruction
but instead there are a number of instructions that are part of the i8mm feature set.
For example Signed Matrix Multiply-Accumulate (SMMLA).

Lets take a looks at one of these instructions in isolation, that is not using the Kleidiai
library and see how it works. In [smmla.s](../fundamentals/kleidiai/src/smmla.s) we can see
an example and run it in lldb:
```console
$ cd fundamentals/kleidiai
$ make smmla
$ lldb smmla
(lldb) br set -n do_smmla
```
In the source code (for some reason in the disassembly in lldb the operands are not displayed)
we can see the:
```asm
    smmla   v2.4s, v0.16b, v1.16b
```
So we have the instruction, and v2 is the vector 2 register that will hold the result which will
be 4 32-bit signed integers. v0 and v1 are the input registers that hold 16 8-bit signed integers.

```console
(lldb) si
Process 2363 stopped
* thread #1, queue = 'com.apple.main-thread', stop reason = instruction step into
    frame #0: 0x00000001000003c0 smmla`do_smmla + 8
smmla`do_smmla:
->  0x1000003c0 <+8>:  smmla
    0x1000003c4 <+12>: adrp   x2, 4
    0x1000003c8 <+16>: add    x2, x2, #0x20 ; result_matrix
    0x1000003cc <+20>: st1.4s { v2 }, [x2]
Target 0: (smmla) stopped.

(lldb) register read v2 --format int32
      v2 = {0 0 0 0}

(lldb) register read v0 --format int8
      v0 = {1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16}

(lldb) register read v1 --format int8
      v1 = {1 3 5 7 9 1 3 5 2 4 6 8 10 2 4 6}

(lldb) si
Process 2363 stopped
* thread #1, queue = 'com.apple.main-thread', stop reason = instruction step into
    frame #0: 0x00000001000003c4 smmla`do_smmla + 12
smmla`do_smmla:
->  0x1000003c4 <+12>: adrp   x2, 4
    0x1000003c8 <+16>: add    x2, x2, #0x20 ; result_matrix
    0x1000003cc <+20>: st1.4s { v2 }, [x2]
    0x1000003d0 <+24>: ldp    x29, x30, [sp], #0x10
Target 0: (smmla) stopped.

(lldb) register read v2 --format int32
      v2 = {162 198 434 534}
```

So the point of this was to show what the raw instructions do, and hopefully this will
help understand what the Kleidiai library is doing under the hood and also why it is
helpful in taking care of data layout/preparation.

### Function naming convention
Before using the library it is important to understand the [naming convention] used for the
functions and the header files.
```
kai_<op>_<fused_ops>_<dst_info>_<input_0_info, input_1_info, ...>_<m_step x n_step>_<simd_engine>_<feature>_<instruction>_<uarch>
```
Where:
* kai - the library prefix
* op - the operation, like matmul.
* fused_ops - any fused operations, like clamp.
Keep in mind that `clamp` can also be used to implement Relu, Relu6, Hard Tanh.
* dst_info - output information, like the data type and layout.
* input_0_info - the input information (matrix A), like the data type and layout.
* input_1_info - the input information (matrix B), like the data type and layout.
* m_step x n_step - the blocking size of the micro kernel. TODO: explain this better.
* simd_engine - the SIMD engine used, like neon, sme, sme2.
* feature - any special feature set used, like i8mm, dotprod.
* instruction - (Optional) any special instruction set used, like mla, smmla.
* uarch - (Optional) any special micro architecture optimizations.

### Packing
So we have the matrix multiplication operation, with an adding of the existing output
matrix, and then a clamp operation.

We have seen the  SMMLA instruction operates on:
* Left operand: 2×8 matrix (int8 values)
* Right operand: 8×2 matrix (int8 values)
* Output: 2×2 matrix (int32 accumulation)

```
        A (input_0)      B (input_1)     C (output) 
 [1, 2, 3, 4, 5, 6, 7, 8] [1, 2]      =  [162 198]
 [9,10,11,12,13,14,15,16] [3, 4]         [434 534]
                          [5, 6]
                          [7, 8]
                          [9,10]
                          [1, 2]
                          [3, 4]
                          [5, 6]
```
Now, if we have matrices that are larger than this we need to break them down into smaller
blocks that fit into this micro kernel. This is called blocking or tiling and this is what
packing in Kleidiai is about. The matrices need to be rearranged in memory to optimize
SMMLA access patterns.

```
       A (4×8)                    B (8×2)          C (4×2)
[1,  2,  3,  4,  5,  6,  7,  8] × [1, 2]        = [204, 248]
[9, 10, 11, 12, 13, 14, 15, 16]   [3, 4]          [588, 728]
[17,18, 19, 20, 21, 22, 23, 24]   [5, 6]          [972,1208]
[25,26, 27, 28, 29, 30, 31, 32]   [7, 8]          [1356,1688]
                                  [9,10]
                                  [1, 2]
                                  [3, 4]
                                  [5, 6]
```
So will have to split this into two `2x8` tiles:
Tile0:
```
           A_0 (2x8)              B (8x2)         C_0 (2x2)
[1,  2,  3,  4,  5,  6,  7,  8] × [1, 2]       = [204, 248]
[9, 10, 11, 12, 13, 14, 15, 16]   [3, 4]         [588, 728]
                                  [5, 6]
                                  [7, 8]
                                  [9,10]
                                  [1, 2]
                                  [3, 4]
                                  [5, 6]
```
Tile1:
```
           A_1 (2x8)              B (8x2)         C_1 (2x2)
[17,18, 19, 20, 21, 22, 23, 24] x [1, 2]       = [972,1208]
[25,26, 27, 28, 29, 30, 31, 32]   [3, 4]         [1356,1688]
                                  [5, 6]
                                  [7, 8]
                                  [9,10]
                                  [5, 6]
                                  [1, 2]
                                  [3, 4]
                                  [5, 6]
```

* qs - Quantized Symmetric
* qa - Quantized Asymmetric
* i  - Signed integer
* u  - Unsigned integer
* 4  - 4-bit quantization
* 8  - 8-bit quantization
* dx - Per-dimension quantization
* cx - Per-channel quantization
* c32 - Per block quantization (32 elements per block)
* scalef16 - Scale factor is float16
* p   - indicates that data is packed
* s16s0 - Packing order is interleaved 16-bit, stride 0
* s1s0  - Packing order is sequential 8-bit, stride 0
* s - scale factors are packed into the data structure/buffer
* b - bias values are packed into the data structure/buffer
* x data type agnostic. Used when describing moving data around in packing micro-kernels

For example 
```
qai8dxp4x8
```
This is an asymmetric quantized (qa) 8-bit integer (i8), with per-dimension quantization (dx),
whichis packed (p) into blocks of 4 rows and 8 columns (4x8).

### Quantization
Similar to GGML values are grouped into blocks that share quantization parameters. This is
```
Original float32 values: [1.2  3.4  2.1  4.8  1.9  3.2  2.7  4.1  ... 8.5]               
                           0    1    2    3    4    5   6     7   ...  63]
                         {    Block 0      }  {      Block 1   }  ... 
```
This could be quantized into a 8x8 blocks of 4-bit integers using a block structure like:
```
Scale factor (delta): 1 float32 per channel/block.
Zero point          : 0 (symmetric quantization).
Quantized values    : 4 bits per weight.
Block size          : Typically 8×8 = 64 values sharing one scale (delta) value.
```
So in this example the original weight matrix where we have an 8x8 matrix of float32 values,
this would have a single block because the quantization is quantizing each float32 value into
a 4-bit value, and the block can store 64 values. And the whole block will share one
scale/delta value.
```
Original: 8×8 = 64 float32 values
     ↓
Single quantization block:
- 1 scale/delta value (shared by all 64 values)
- 64 quantized 4-bit values
- Total storage: 1 float32 (4 bytes) + 64×4bits (32 bytes) = 36 bytes

Block 0 (covers entire 8×8 matrix):
┌─────────────────────────────────┐
│ Scale: 0.2 (float32)            │
├─────────────────────────────────┤
│ Quantized values (4-bit each):  │
│ [10,9,15,13,8,12,14,11,...]     │ ← 64 values, 4 bits each
│ Packed into 32 bytes            │
└─────────────────────────────────┘
```
Process:
```
Original 8×8 matrix:
[2.1, 1.8, 3.2, 2.7, 1.6, 2.4, 2.8, 2.2]
[1.9, 3.1, 2.6, 3.0, 2.3, 1.7, 2.9, 2.5]
[... 6 more rows ...]

Step 1: Find scale for entire block
scale = max_abs_value / 7  // 4-bit signed: -7 to +7 range
scale = 3.2 / 7 = 0.457

Step 2: Quantize all 64 values using same scale
q_values = round(original_values / scale)
q_values = [5, 4, 7, 6, 4, 5, 6, 5, 4, 7, 6, 7, 5, 4, 6, 5, ...]

Step 3: Pack into memory
All 64 quantized values share the single scale = 0.457
```

In Kleidiai there are descriptors that tell the system how matrices have been pre-packed:
```
mr x kr (written as mrxkr)
nr x kr (written as nrxkr)

where:
mr : number of rows in the LHS (matrix A) side matrix packed together
nr : number of columns in the RHS (matrix B) side matrix packed together
kr : number of columns in the LHS side matrix (or rows in the RHS side matrix) packed together
     which is the reduction dimension.
```
So the example above we have:
```
A (LHS): 4×8 matrix     B (RHS): 8×2 matrix
```
For optimal SMMLA packing:
```
mr = 2, pack 2 rows of A together which fits with the ssmla instruction requirements
nr = 2, pack 2 columns of B together which fits with the ssmla instruction requirements
kr = 8, pack 8 columns of A (or rows of B) together which fits with the ssmla instruction requirements
```
So for LHS this would be mrxkr = 2x8 and for RHS this would be nrxkr = 2x8.

### API
Just a short note about the API which I think is targeted to framework developers where
full control over memory and data layout is desired. So there is not just a single include
and then you are good to go but multiple which I found a bit confusing at first.

```console
kleidiai/install/include/kai/
    ukernels/matmul/
        matmul_clamp_f32_qai8dxp_qsi4cxp/
```
So this is a header file directory is for matrix multiplications with clamp operation, where
the output if float32, the input is `qai8dxp` and `qsi4cxp`. So the top level is the operation
and the data types.

There are a number of files in this directory but let start with the interface file:
```console
kai_matmul_clamp_f32_qai8dxp_qsi4cxp_interface.h
````
/// Micro-kernel interface
struct kai_matmul_clamp_f32_qai8dxp_qsi4cxp_ukernel {
    kai_matmul_clamp_f32_qai8dxp_qsi4cxp_run_matmul_func_t run_matmul;

    kai_matmul_clamp_f32_qai8dxp_qsi4cxp_get_m_step_func_t get_m_step;
    kai_matmul_clamp_f32_qai8dxp_qsi4cxp_get_n_step_func_t get_n_step;
    kai_matmul_clamp_f32_qai8dxp_qsi4cxp_get_mr_func_t     get_mr;
    kai_matmul_clamp_f32_qai8dxp_qsi4cxp_get_nr_func_t     get_nr;
    kai_matmul_clamp_f32_qai8dxp_qsi4cxp_get_kr_func_t     get_kr;
    kai_matmul_clamp_f32_qai8dxp_qsi4cxp_get_sr_func_t     get_sr;

    kai_matmul_clamp_f32_qai8dxp_qsi4cxp_get_lhs_packed_offset_func_t get_lhs_packed_offset;
    kai_matmul_clamp_f32_qai8dxp_qsi4cxp_get_rhs_packed_offset_func_t get_rhs_packed_offset;
    kai_matmul_clamp_f32_qai8dxp_qsi4cxp_get_dst_offset_func_t        get_dst_offset;
    kai_matmul_clamp_f32_qai8dxp_qsi4cxp_get_dst_size_func_t          get_dst_size;

};
```
So the main function is the `run_matmul` which is the micro-kernel core function:
```c
/// Micro-kernel core function ("run" method)
typedef void (*kai_matmul_clamp_f32_qai8dxp_qsi4cxp_run_matmul_func_t)(
    size_t m,
    size_t n,
    size_t k,
    const void* lhs_p,
    const void* rhs_p,
    float* dst,
    size_t dst_stride_row,
    size_t dst_stride_col,
    float scalar_min,
    float scalar_max);
```
And for the different variants
```
get_m_step() // How many rows this micro-kernel processes at once
get_n_step() // How many columns this micro-kernel processes at once
get_mr()     // LHS packing: rows packed together
get_nr()     // RHS packing: columns packed together
get_kr()     // Reduction dimension packing
get_sr()     // Scale packing (for quantization blocks)
```
Now, if we look at a specific implementation we will find these values defined,
for example in `kleidiai/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm.c`:
```c
static const size_t kai_m_step = 8;
static const size_t kai_n_step = 8;
static const size_t kai_mr = 4;
static const size_t kai_nr = 8;
static const size_t kai_kr = 16;
static const size_t kai_sr = 2;
static const size_t kai_num_bytes_multiplier_lhs = sizeof(float);
static const size_t kai_num_bytes_multiplier_rhs = sizeof(float);
static const size_t kai_num_bytes_offset_lhs = sizeof(int32_t);
static const size_t kai_num_bytes_sum_rhs = sizeof(int32_t);
static const size_t kai_num_bytes_bias = sizeof(float);
```

The files inside the above directory represent specific micro kernel variants with different
packing formats (mrxkr), tile sizes, hardware targets like neon, sme2, and feature sets like
i8mm, dotprod, sdot, mopa. So just like we saw with the smmla instruction it required its data
in a specific format, but if neon is the target for this matric multiplication then the data
may have to be packed differently. And if sme2 is the target then it may have to be packed
differently again.
```
The last p is for packing but there can be multiple different ways to pack the data, so there are
multiple files in this
```console
$ ls kleidiai/install/include/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp
kai_matmul_clamp_f32_qai8dx p1vlx8  _qsi4cxp4vlx8  _1vlx4vl _sme2 _mopa.h
kai_matmul_clamp_f32_qai8dx p1x4    _qsi4cxp4vlx4  _1x4vl   _sme2 _sdot.h
kai_matmul_clamp_f32_qai8dx p1x4    _qsi4cxp4x4    _1x4     _neon _dotprod.h
kai_matmul_clamp_f32_qai8dx p1x8    _qsi4cxp4x8    _1x4x32  _neon _dotprod.h
kai_matmul_clamp_f32_qai8dx p1x8    _qsi4cxp8x8    _1x8x32  _neon _dotprod.h
kai_matmul_clamp_f32_qai8dx p4x4    _qsi4cxp8x4    _8x8x32  _neon _dotprod.h
kai_matmul_clamp_f32_qai8dx p4x8    _qsi4cxp4x4    _16x4x32 _neon _dotprod.h
kai_matmul_clamp_f32_qai8dx p4x8    _qsi4cxp4x8    _4x4x32  _neon _i8mm.h
kai_matmul_clamp_f32_qai8dx p4x8    _qsi4cxp4x8    _8x4x32  _neon _i8mm.h
kai_matmul_clamp_f32_qai8dx p4x8    _qsi4cxp8x8    _4x8x32  _neon _i8mm.h
kai_matmul_clamp_f32_qai8dx p4x8    _qsi4cxp8x8    _8x8x32  _neon _i8mm.h
```

### rs value
Specifies how many elements share the same scale factor along the reduction dimension (k).
```c
void kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(
    size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, const uint8_t* rhs, const float* bias,
    const float* scale, void* rhs_packed, size_t extra_bytes,
    const struct kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_params* params) {
    KAI_ASSERT((kr % sr) == 0);
    ...
    const size_t block_length_in_bytes = kr / sr;
```
So if kr=32, and sr=2, then every 16 elements share the same scale factor.
So with `sr=2`, every 16 bytes (32 elements ÷ 2) of the packed data share the same
quantization parameters.

__wip__

### Left Hand Side (LHS) and Right Hand Side (RHS)

```
Output = Input × Weights + Bias
           ↓       ↓
          LHS  ×  RHS
 ```

LHS are the input or activations (if coming from a previous layer) and these change with
every input (image, text, etc). The only exist during inference and unknown what the user
will input. Runtime quantization needs to be fast and effiecient.

RHS on the other hand are fixed after training and never change during inference. So these can
be quantized aggressively as they are know before hand and can be analyzed and processed offline
and stored in the model.

RHS: qsi4cxp (Weights)
```
q: Quantized
s: Symmetric (centered around zero - good for weights)
i4: 4-bit integers (very aggressive compression)
c: Per-channel (each output channel gets its own scale)
x: ?
p: Packed format
```

LHS: qai8dxp (Activations)
```
q: Quantized
a: Asymmetric (might not be centered - good for activations)
i8: 8-bit integers (more conservative, higher precision)
d: Per-row (each input row gets its own scale)
x: ?
p: Packed format
```

Libraries like BLAS were designed for scientific computing where:
* Both matrices are "just data"
* No distinction between "static parameters" and "dynamic inputs"

But in AI inference:
* The LHS/RHS roles are very different
* We process millions of operations with the same weights
* Every optimization counts for deployment

So this might be why we don't seen the same emphasis on packing and LHS and RHS in traditional
BLAS libraries.


[naming convention]: https://gitlab.arm.com/kleidi/kleidiai/-/blob/main/kai/ukernels/matmul/README.md#micro-kernel-naming


