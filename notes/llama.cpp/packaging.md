## Packaging/Distributions in llama.cpp
This document contains note about how to package and distribute `llama.cpp`.

By llama.cpp we mean a binary distribution of the project and all the tools or
just a subset of the tools?

The ones I think that are most relevant are:
* llama-cli
* llama-server
* quantize
* tokenize

But including everything is also an option and not the difficult part.

### Existing release packages
A release of llama.cpp currently contains the following artifacts:
[b6558](https://github.com/ggml-org/llama.cpp/releases/tag/b6558)

#### Windows
* cudart-llama-bin-win-cuda-12.4-x64.zip
CUDA runtime for Windows x64 with CUDA 12.4.
```console
Archive:  cudart-llama-bin-win-cuda-12.4-x64.zip
  inflating: cublas64_12.dll
  inflating: cublasLt64_12.dll
  inflating: cudart64_12.dll
```

* llama-b6558-bin-win-cuda-12.4-x64.zip
Binary distribution for Windows x64 with CUDA 12.4 support.
```console
Archive:  llama-b6558-bin-win-cuda-12.4-x64.zip
  inflating: ggml-cuda.dll
  inflating: LICENSE-curl
  inflating: llama-gemma3-cli.exe
  inflating: ggml.dll
  inflating: ggml-base.dll
  inflating: llama-minicpmv-cli.exe
  inflating: mtmd.dll
  inflating: libcurl-x64.dll
  inflating: ggml-cpu-haswell.dll
  inflating: llama.dll
  inflating: llama-perplexity.exe
  inflating: llama-mtmd-cli.exe
  inflating: llama-bench.exe
  inflating: llama-tokenize.exe
  inflating: ggml-cpu-icelake.dll
  inflating: LICENSE-linenoise
  inflating: llama-imatrix.exe
  inflating: llama-server.exe
  inflating: llama-tts.exe
  inflating: LICENSE-jsonhpp
  inflating: rpc-server.exe
  inflating: libomp140.x86_64.dll
  inflating: ggml-cpu-sandybridge.dll
  inflating: ggml-cpu-sapphirerapids.dll
  inflating: LICENSE-httplib
  inflating: ggml-cpu-skylakex.dll
  inflating: llama-batched-bench.exe
  inflating: llama-qwen2vl-cli.exe
  inflating: ggml-cpu-alderlake.dll
  inflating: llama-gguf-split.exe
  inflating: ggml-rpc.dll
  inflating: llama-run.exe
  inflating: llama-cli.exe
  inflating: ggml-cpu-sse42.dll
  inflating: llama-llava-cli.exe
  inflating: ggml-cpu-x64.dll
  inflating: llama-quantize.exe
```

* llama-b6558-bin-win-cpu-arm64.zip
Binary distribution for Windows ARM64.

* llama-b6558-bin-win-hip-radeon-x64.zip
Binary distribution for Windows x64 with HIP support for AMD Radeon GPUs.

* llama-b6558-bin-win-opencl-adreno-arm64.zip
Binary distribution for Windows ARM64 with OpenCL support for Adreno GPUs.

* llama-b6558-bin-win-sycl-x64.zip
Binary distribution for Windows x64 with SYCL support.

* llama-b6558-bin-win-vulkan-x64.zip
Binary distribution for Windows x64 with Vulkan support.

* llama-b6558-bin-win-cpu-x64.zip
Binary distribution for Windows x64.


#### Linux
* llama-b6558-bin-ubuntu-vulkan-x64.zip
Binary distribution for Ubuntu x64 with Vulkan support.
```console
Archive:  llama-b6558-bin-ubuntu-vulkan-x64.zip
  inflating: build/bin/LICENSE
  inflating: build/bin/LICENSE-curl
  inflating: build/bin/LICENSE-httplib
  inflating: build/bin/LICENSE-jsonhpp
  inflating: build/bin/LICENSE-linenoise
  inflating: build/bin/libggml-base.so
  inflating: build/bin/libggml-cpu-alderlake.so
  inflating: build/bin/libggml-cpu-haswell.so
  inflating: build/bin/libggml-cpu-icelake.so
  inflating: build/bin/libggml-cpu-sandybridge.so
  inflating: build/bin/libggml-cpu-sapphirerapids.so
  inflating: build/bin/libggml-cpu-skylakex.so
  inflating: build/bin/libggml-cpu-sse42.so
  inflating: build/bin/libggml-cpu-x64.so
  inflating: build/bin/libggml-rpc.so
  inflating: build/bin/libggml-vulkan.so
  inflating: build/bin/libggml.so
  inflating: build/bin/libllama.so
  inflating: build/bin/libmtmd.so
  inflating: build/bin/llama-batched-bench
  inflating: build/bin/llama-bench
  inflating: build/bin/llama-cli
  inflating: build/bin/llama-gemma3-cli
  inflating: build/bin/llama-gguf-split
  inflating: build/bin/llama-imatrix
  inflating: build/bin/llama-llava-cli
  inflating: build/bin/llama-minicpmv-cli
  inflating: build/bin/llama-mtmd-cli
  inflating: build/bin/llama-perplexity
  inflating: build/bin/llama-quantize
  inflating: build/bin/llama-qwen2vl-cli
  inflating: build/bin/llama-run
  inflating: build/bin/llama-server
  inflating: build/bin/llama-tokenize
  inflating: build/bin/llama-tts
  inflating: build/bin/rpc-server
```
So this contains all the cpu backends and the vulkan backend. But notice that
there are no shared libraries for Vulkan, so the user must have Vulkan installed.

* llama-b6558-bin-ubuntu-x64.zip
Binary distribution for Ubuntu x64.


#### MacOS
* llama-b6558-bin-macos-arm64.zip
Binary distribution for macOS ARM64 (Apple Silicon).

* llama-b6558-bin-macos-x64.zip
Binary distribution for macOS x64 (Intel).

* llama-b6558-xcframework.zip
XCFramework for iOS and macOS.


### Packaging backends
Currently llama.cpp supports the following backends:

* ggml-blas
* ggml-cuda
* ggml-opencl
* ggml-cann
* ggml-hip
* ggml-rpc
* ggml-vulkan
* ggml-sycl
* ggml-webgpu
* ggml-amx
* ggml-metal
* ggml-zdnn
* ggml-cpu
* ggml-musa

