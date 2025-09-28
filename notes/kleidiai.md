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


### API
Just a short note about the API which I think is targeted to framework developers where
full control over memory and data layout is desired. So there is not just a single include
and then you are good to go but multiple which I found a bit confusing at first.

```console
kleidiai/install/include/kai/
    ukernels/matmul/
        matmul_clamp_f32_qai8dxp_qsi4cxp/
            kai_matmul_clamp_f32_ai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm.h
```
So this is a header file for a matrix multiplication with clamp operation, where the
output if float32, the input is `qai8dxp` and `qsi4cxp` and the micro kernel

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


