## WebGPU
Is the successor to WebGL, and is a new web standard for GPU acceleration which
provides access to the capabilities of modern GPUs from a web browser. Despite
the name, WebGPU does not require a browser to work but can also be run
standalone using implementations like [Dawn](https://dawn.googlesource.com/dawn)
or [wgpu](https://wgpu.rs/).

WebGPU provides a unified API across Windows (D3D12), macOS (Metal), and Linux
(Vulkan). One codebase works everywhere, unlike having separate CUDA/OpenCL/Metal
backends. We write a "kernel" called a shader which is then compiled by the
specific system to run on a particular GPU.

The WebGPU implementation handles translation:
- WGSL → HLSL (for D3D12/Windows)
- WGSL → MSL (Metal Shading Language for macOS/iOS)
- WGSL → SPIR-V (for Vulkan/Linux)

Example workflow:
1. Write a matrix multiply kernel in WGSL
2. On Windows: WebGPU runtime compiles WGSL → HLSL   → GPU bytecode
3. On macOS:   WebGPU runtime compiles WGSL → MSL    → GPU bytecode
4. On Linux:   WebGPU runtime compiles WGSL → SPIR-V → GPU bytecode

### Overview

```console
    +--------------------+ +--------------------+
    | JavaScript         | |     webgpu.h       |
    +--------------------+ +--------------------+
    +-------------------------------------------+
    |                 WebGPU                    |  
    +-------------------------------------------+
    +--------------+ +--------------+ +---------+
    |  Metal       | | DirectX 12   | | Vulkan  |
    +--------------+ +--------------+ +---------+
    +-------------------------------------------+
    |     Hardware (GPU etc)                    |
    +-------------------------------------------+ 
```

### Vertex Shaders
TODO:

### Fragment Shaders
TODO:

### Compute Shaders
Are written in a shading language called WebGPU Shading Language (WGSL).

### WebGPU Shading Language (WGSL)
[Shaders](./gpu.md) are written in a new shading language called WebGPU Shading
Language.


### Links
* https://github.com/webgpu-native/webgpu-headers/


### Chrome
First enable WebGPU in chrome://flags and search for `WebGPU` and enable the
flags. We can then inspect status using `chrome://gpu`.

Then start chrome with the following flag:
```
$ google-chrome --enable-features=Vulkan
```
I added this to `/usr/share/applications/google-chrome.desktop`.
And after restarting chrome, we can see the following in `chrome://gpu`:
```
*   Vulkan: Enabled
```

### llama.cpp
ggml has a backend for WebGPU, which can be used with llama-cli. First install
```console
$ cmake -S . -B out/Release -DDAWN_ENABLE_INSTALL=ON -DDAWN_BUILD_MONOLITHIC_LIBRARY=SHARED -DCMAKE_BUILD_TYPE=Release
$ cmake --build out/Release -j8
$ cmake --install out/Release --prefix install/Release --verbose
```

[Dawn](https://dawn.googlesource.com/dawn/+/refs/heads/main/docs/quickstart-cmake.md).

Then build llama.cpp with WebGPU support:
```console
export CMAKE_PREFIX_PATH=/home/danbev/work/webgpu/dawn/install/Release
cmake -B build -DGGML_WEBGPU=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -- -j8
```
```console
$ export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
$ build/bin/llama-cli --list-devices
ggml_webgpu: adapter_info: vendor_id: 4318 | vendor: nvidialovelaceNVIDIA GeForce RTX 4070NVIDIA: 560.35.05 560.35.5.0 | architecture: lovelaceNVIDIA GeForce RTX 4070NVIDIA: 560.35.05 560.35.5.0 | device_id: 10118 | name: NVIDIA GeForce RTX 4070NVIDIA: 560.35.05 560.35.5.0 | device_desc: NVIDIA: 560.35.05 560.35.5.0
Available devices:
  WebGPU: NVIDIA: 560.35.05 560.35.5.0 (1048576 MiB, 1048576 MiB free)
```

```console
$ build/bin/llama-cli -m models/ --device webgpu
```

