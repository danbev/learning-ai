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
