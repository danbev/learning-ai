## WebGPU
Is the successor to WebGL, and is a new web standard for GPU acceleration which
provides access to the capabilities of modern GPUs from a web browser. Despite
the name, WebGPU does not require a browser to work but can also be run
standalone using implementations like [Dawn](https://dawn.googlesource.com/dawn)
or [wgpu](https://wgpu.rs/).

It provides graphics and compute APIs designed specifically for the web simliar
to it as the web's equivalent to low-level graphics APIs like Vulkan,
Direct3D 12, and Metal. 

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
