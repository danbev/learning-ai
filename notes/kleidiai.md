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


### API
Just a short note about the API which I think is targeted to framework developers where
full control over memory and data layout is desired. So there is not just a single include
and then you are good to go but multiple which I found a bit confusing at first.

```console
kleidiai/install/include/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm.h
```

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
* dst_info - 
* input_info - the input information, like the data type and layout.

__wip__

[naming convention]: https://gitlab.arm.com/kleidi/kleidiai/-/blob/main/kai/ukernels/matmul/README.md#micro-kernel-naming


