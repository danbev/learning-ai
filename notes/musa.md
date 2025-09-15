## Multi-Unit Streaming Architecture (MUSA)
This is from the Chinese company named Moore Threads and is a GPU programming
plaform similar to CUDA and ROCm.

The focus on General-purpose GPU compute + AI so they also target the gaming
market. They these are more targeting general usecases whereas [CANN](./cann.md)
is more focused on AI workloads.

Like AMD's HIP, MUSA provides a programming interface that's very similar to
CUDA, making it easier to port CUDA applications. For example, in ggml we have
the following in ggml/src/ggml-cuda/common.cuh:
```
#if defined(GGML_USE_HIP)
#define GGML_COMMON_DECL_HIP
#define GGML_COMMON_IMPL_HIP
#else
#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#if defined(GGML_USE_MUSA)
#define GGML_COMMON_DECL_MUSA
#define GGML_COMMON_IMPL_MUSA
#endif
#endif
...
#if defined(GGML_USE_HIP)
#include "vendors/hip.h"
#elif defined(GGML_USE_MUSA)
#include "vendors/musa.h"
#else
#include "vendors/cuda.h"
#endif // defined(GGML_USE_HIP)
```
And in ggml/src/ggml-cuda/vendors there are the different vendor-specific
headers. For example in `musa.h` we have:
```console
#pragma once

#include <musa_runtime.h>
#include <musa.h>
#include <mublas.h>
#include <musa_bf16.h>
#include <musa_fp16.h>
#define CUBLAS_COMPUTE_16F CUDA_R_16F
#define CUBLAS_COMPUTE_32F CUDA_R_32F
#define CUBLAS_COMPUTE_32F_FAST_16F MUBLAS_COMPUTE_32F_FAST_16F
#define CUBLAS_GEMM_DEFAULT MUBLAS_GEMM_DEFAULT
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP MUBLAS_GEMM_DEFAULT
#define CUBLAS_OP_N MUBLAS_OP_N
#define CUBLAS_OP_T MUBLAS_OP_T
#define CUBLAS_STATUS_SUCCESS MUBLAS_STATUS_SUCCESS
#define CUBLAS_TF32_TENSOR_OP_MATH MUBLAS_TENSOR_OP_MATH
#define CUDA_R_16F  MUSA_R_16F
#define CUDA_R_16BF MUSA_R_16BF
#define CUDA_R_32F  MUSA_R_32F
#define cublasComputeType_t cudaDataType_t
#define cublasCreate mublasCreate
#define cublasDestroy mublasDestroy
...
```
