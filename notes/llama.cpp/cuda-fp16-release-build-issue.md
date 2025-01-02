## CUDA FP16 Release Build Issue
I ran into this issue when working on the Llama 3.2 Vision Instruct support
and only when compiling a Release build.

```console
/home/danbev/work/ai/llama-vision-api/ggml/src/ggml-cuda/mmv.cu:51: ERROR: CUDA kernel mul_mat_vec has no device code compatible with CUDA arch 520. ggml-cuda.cu was compiled for: 520
/home/danbev/work/ai/llama-vision-api/ggml/src/ggml-cuda/mmv.cu:51: ERROR: CUDA kernel mul_mat_vec has no device code compatible with CUDA arch 520. ggml-cuda.cu was compiled for: 520
/home/danbev/work/ai/llama-vision-api/ggml/src/ggml-cuda/mmv.cu:51: ERROR: CUDA kernel mul_mat_vec has no device code compatible with CUDA arch 520. ggml-cuda.cu was compiled for: 520
/home/danbev/work/ai/llama-vision-api/ggml/src/ggml-cuda/mmv.cu:51: ERROR: CUDA kernel mul_mat_vec has no device code compatible with CUDA arch 520. ggml-cuda.cu was compiled for: 520
/home/danbev/work/ai/llama-vision-api/ggml/src/ggml-cuda/mmv.cu:51: ERROR: CUDA kernel mul_mat_vec has no device code compatible with CUDA arch 520. ggml-cuda.cu was compiled for: 520
/home/danbev/work/ai/llama-vision-api/ggml/src/ggml-cuda/mmv.cu:51: ERROR: CUDA kernel mul_mat_vec has no device code compatible with CUDA arch 520. ggml-cuda.cu was compiled for: 520
/home/danbev/work/ai/llama-vision-api/ggml/src/ggml-cuda/mmv.cu:51: ERROR: CUDA kernel mul_mat_vec has no device code compatible with CUDA arch 520. ggml-cuda.cu was compiled for: 520
/home/danbev/work/ai/llama-vision-api/ggml/src/ggml-cuda/mmv.cu:51: ERROR: CUDA kernel mul_mat_vec has no device code compatible with CUDA arch 520. ggml-cuda.cu was compiled for: 520
/home/danbev/work/ai/llama-vision-api/ggml/src/ggml-cuda/mmv.cu:51: ERROR: CUDA kernel mul_mat_vec has no device code compatible with CUDA arch 520. ggml-cuda.cu was compiled for: 520
/home/danbev/work/ai/llama-vision-api/ggml/src/ggml-cuda/mmv.cu:51: ERROR: CUDA kernel mul_mat_vec has no device code compatible with CUDA arch 520. ggml-cuda.cu was compiled for: 520
/home/danbev/work/ai/llama-vision-api/ggml/src/ggml-cuda/mmv.cu:51: ERROR: CUDA kernel mul_mat_vec has no device code compatible with CUDA arch 520. ggml-cuda.cu was compiled for: 520
/home/danbev/work/ai/llama-vision-api/ggml/src/ggml-cuda/mmv.cu:51: ERROR: CUDA kernel mul_mat_vec has no device code compatible with CUDA arch 520. ggml-cuda.cu was compiled for: 520
/home/danbev/work/ai/llama-vision-api/ggml/src/ggml-cuda/mmv.cu:51: ERROR: CUDA kernel mul_mat_vec has no device code compatible with CUDA arch 520. ggml-cuda.cu was compiled for: 520
/home/danbev/work/ai/llama-vision-api/ggml/src/ggml-cuda/mmv.cu:51: ERROR: CUDA kernel mul_mat_vec has no device code compatible with CUDA arch 520. ggml-cuda.cu was compiled for: 520
/home/danbev/work/ai/llama-vision-api/ggml/src/ggml-cuda/mmv.cu:51: ERROR: CUDA kernel mul_mat_vec has no device code compatible with CUDA arch 520. ggml-cuda.cu was compiled for: 520
/home/danbev/work/ai/llama-vision-api/ggml/src/ggml-cuda/mmv.cu:51: ERROR: CUDA kernel mul_mat_vec has no device code compatible with CUDA arch 520. ggml-cuda.cu was compiled for: 520
/home/danbev/work/ai/llama-vision-api/ggml/src/ggml-cuda/mmv.cu:51: ERROR: CUDA kernel mul_mat_vec has no device code compatible with CUDA arch 520. ggml-cuda.cu was compiled for: 520
/home/danbev/work/ai/llama-vision-api/ggml/src/ggml-cuda/mmv.cu:51: ERROR: CUDA kernel mul_mat_vec has no device code compatible with CUDA arch 520. ggml-cuda.cu was compiled for: 520
/home/danbev/work/ai/llama-vision-api/ggml/src/ggml-cuda/mmv.cu:51: ERROR: CUDA kernel mul_mat_vec has no device code compatible with CUDA arch 520. ggml-cuda.cu was compiled for: 520
/home/danbev/work/ai/llama-vision-api/ggml/src/ggml-cuda/mmv.cu:51: ERROR: CUDA kernel mul_mat_vec has no device code compatible with CUDA arch 520. ggml-cuda.cu was compiled for: 520
/home/danbev/work/ai/llama-vision-api/ggml/src/ggml-cuda/mmv.cu:51: ERROR: CUDA kernel mul_mat_vec has no device code compatible with CUDA arch 520. ggml-cuda.cu was compiled for: 520
/home/danbev/work/ai/llama-vision-api/ggml/src/ggml-cuda/mmv.cu:51: ERROR: CUDA kernel mul_mat_vec has no device code compatible with CUDA arch 520. ggml-cuda.cu was compiled for: 520
/home/danbev/work/ai/llama-vision-api/ggml/src/ggml-cuda/mmv.cu:51: ERROR: CUDA kernel mul_mat_vec has no device code compatible with CUDA arch 520. ggml-cuda.cu was compiled for: 520
/home/danbev/work/ai/llama-vision-api/ggml/src/ggml-cuda/mmv.cu:51: ERROR: CUDA kernel mul_mat_vec has no device code compatible with CUDA arch 520. ggml-cuda.cu was compiled for: 520
/home/danbev/work/ai/llama-vision-api/ggml/src/ggml-cuda/mmv.cu:51: ERROR: CUDA kernel mul_mat_vec has no device code compatible with CUDA arch 520. ggml-cuda.cu was compiled for: 520
/home/danbev/work/ai/llama-vision-api/ggml/src/ggml-cuda/mmv.cu:51: ERROR: CUDA kernel mul_mat_vec has no device code compatible with CUDA arch 520. ggml-cuda.cu was compiled for: 520
/home/danbev/work/ai/llama-vision-api/ggml/src/ggml-cuda/mmv.cu:51: ERROR: CUDA kernel mul_mat_vec has no device code compatible with CUDA arch 520. ggml-cuda.cu was compiled for: 520
/home/danbev/work/ai/llama-vision-api/ggml/src/ggml-cuda/mmv.cu:51: ERROR: CUDA kernel mul_mat_vec has no device code compatible with CUDA arch 520. ggml-cuda.cu was compiled for: 520
/home/danbev/work/ai/llama-vision-api/ggml/src/ggml-cuda/mmv.cu:51: ERROR: CUDA kernel mul_mat_vec has no device code compatible with CUDA arch 520. ggml-cuda.cu was compiled for: 520
/home/danbev/work/ai/llama-vision-api/ggml/src/ggml-cuda/mmv.cu:51: ERROR: CUDA kernel mul_mat_vec has no device code compatible with CUDA arch 520. ggml-cuda.cu was compiled for: 520
/home/danbev/work/ai/llama-vision-api/ggml/src/ggml-cuda/mmv.cu:51: ERROR: CUDA kernel mul_mat_vec has no device code compatible with CUDA arch 520. ggml-cuda.cu was compiled for: 520
/home/danbev/work/ai/llama-vision-api/ggml/src/ggml-cuda/mmv.cu:51: ERROR: CUDA kernel mul_mat_vec has no device code compatible with CUDA arch 520. ggml-cuda.cu was compiled for: 520
/home/danbev/work/ai/llama-vision-api/ggml/src/ggml-cuda/ggml-cuda.cu:70: CUDA error
CUDA error: unspecified launch failure
  current device: 0, in function ggml_backend_cuda_synchronize at /home/danbev/work/ai/llama-vision-api/ggml/src/ggml-cuda/ggml-cuda.cu:2282
  cudaStreamSynchronize(cuda_ctx->stream())
Could not attach to process.  If your uid matches the uid of the target
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
ptrace: Operation not permitted.
No stack.
The program is not being run.
./run-simple-vision-mllama.sh: line 32: 35452 Aborted                 (core dumped) ./build/bin/llama-simple-vision-mllama -m ${MODEL} -v -ngl 20 --image ${IMAGE}
```
Looking at where the error is coming from, ggml/src/ggml-cuda/mmv.cu:51 we can
see the following:
```c
#ifdef FP16_AVAILABLE
        half2 sumh2 = make_half2(0.0f, 0.0f);

        for (int64_t col2 = tid; col2 < ncols2; col2 += block_size) {
            const float2 tmp = y2[col2];
            sumh2 += x2[col2] * make_half2(tmp.x, tmp.y);
        }

        sumf = __low2float(sumh2) + __high2float(sumh2);
#else
        NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
    }
```
So in my case `FP16_AVAILABLE` is not defined and the `NO_DEVICE_CODE` macro is
being called. The `NO_DEVICE_CODE` macro is defined in ggml/src/ggml-cuda/common.cu
```c
#ifdef __CUDA_ARCH__
#define NO_DEVICE_CODE no_device_code(__FILE__, __LINE__, __FUNCTION__, __CUDA_ARCH__, STRINGIZE(__CUDA_ARCH_LIST__))
#else
#define NO_DEVICE_CODE //GGML_ABORT("NO_DEVICE_CODE not valid in host code.")
#endif // __CUDA_ARCH__
```
Now, `FP16_AVAILABLE` is defined in ggml/src/ggml-cuda/common.h
```c
#define GGML_CUDA_CC_PASCAL     600
#define GGML_CUDA_CC_DP4A       610 // minimum compute capability for __dp4a, an intrinsic for byte-wise dot products
#define GGML_CUDA_CC_VOLTA      700
#define GGML_CUDA_CC_TURING     750
#define GGML_CUDA_CC_AMPERE     800
#define GGML_CUDA_CC_OFFSET_AMD 1000000


#if (defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)) || __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
#define FP16_AVAILABLE
#endif // (defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)) || __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
```

I have a NVIDIA GeForce RTX 4070 GPU and the CUDA version is 12.6 which is of
the Ada Loveace architecture. 

```console
$ nvidia-smi
Thu Jan  2 10:51:15 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.35.05              Driver Version: 560.35.05      CUDA Version: 12.6     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4070        Off |   00000000:04:00.0 Off |                  N/A |
|  0%   39C    P8             11W /  200W |       9MiB /  12282MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A      2361      G   /usr/bin/gnome-shell                            2MiB |
+-----------------------------------------------------------------------------------------+
```
```console

$ nvidia-smi --query-gpu=compute_cap --format=csv
compute_cap
8.9
```
Notice that my GPU has a compute capability of 8.9 which is greater than the
reported 520.
So there is something wrong with the way the compute capability is being
detected.
