## MESA 
[Mesa](https://mesa3d.org/) was originally an open-source implementation of the
OpenGL specification. It was evolved and now also provides implementations of other
graphics APIs such as Vulkan, OpenGL ES, and others.

Vulkan compute shaders need to talk to the GPU through a Vulkan driver and Mesa
provides several Vulkan ICDs:
* RADV - Mesa's Vulkan driver for AMD GPUs
* ANV - Mesa's Vulkan driver for Intel GPUs
* NVK - Mesa's experimental Vulkan driver for NVIDIA GPUs

```
Application → Vulkan Loader -> Vulkan layers → Mesa ICD (like RADV) → AMD GPU hardware
```

```console
VK_LAYER_MESA_device_select (Linux device selection layer) Vulkan version 1.3.211, layer version 1:
  Layer Extensions: count = 0
  Devices: count = 2
  GPU id = 0 (Intel(R) Graphics (ADL GT2))
  Layer-Device Extensions: count = 0

  GPU id = 1 (llvmpipe (LLVM 19.1.1, 256 bits))
  Layer-Device Extensions: count = 0

  VK_LAYER_MESA_overlay (Mesa Overlay layer) Vulkan version 1.3.211, layer version 1:
  Layer Extensions: count = 0
  Devices: count = 2
  GPU id = 0 (Intel(R) Graphics (ADL GT2))
  Layer-Device Extensions: count = 0

  GPU id = 1 (llvmpipe (LLVM 19.1.1, 256 bits))
  Layer-Device Extensions: count = 0
 

VkPhysicalDeviceVulkan12Properties:                                             
-----------------------------------                                             
  driverID                                             = DRIVER_ID_INTEL_OPEN_SOURCE_MESA
  driverName                                           = Intel open-source Mesa driver
  driverInfo                                           = Mesa 24.2.8-1ubuntu1~24.04.1
  conformanceVersion:                                                           
    major    = 1                                                                
    minor    = 3                                                                    
    subminor = 6                                                                
    patch    = 0                                       
...

VkPhysicalDeviceVulkan12Properties:
-----------------------------------
  driverID                                             = DRIVER_ID_MESA_LLVMPIPE
  driverName                                           = llvmpipe
  driverInfo                                           = Mesa 24.2.8-1ubuntu1~24.04.1 (LLVM 19.1.1)
  conformanceVersion:
    major    = 1
    minor    = 3
    subminor = 1
    patch    = 1
 ```
This is an implicit layer, so loaded by default, on my system.
/usr/share/vulkan/implicit_layer.d/VkLayer_MESA_device_select.json:
```
$ cat /usr/share/vulkan/implicit_layer.d/VkLayer_MESA_device_select.json
{
  "file_format_version" : "1.0.0",
  "layer" : {
    "name": "VK_LAYER_MESA_device_select",
    "type": "GLOBAL",
    "library_path": "libVkLayer_MESA_device_select.so",
    "api_version": "1.3.211",
    "implementation_version": "1",
    "description": "Linux device selection layer",
    "functions": {
      "vkNegotiateLoaderLayerInterfaceVersion": "vkNegotiateLoaderLayerInterfaceVersion"
    },
    "disable_environment": {
      "NODEVICE_SELECT": "1"
    }
  }
}
```
The source code for this layer can be found here:
https://gitlab.freedesktop.org/mesa/mesa/-/blob/main/src/vulkan/device-select-layer/device_select_layer.c

The Mesa device select layer intercepts Vulkan device enumeration calls like
`vkEnumeratePhysicalDevices`. When your application asks "what GPUs are available?",
this layer can:
* Reorder the device list (put preferred GPU first)
* Filter out devices based on environment variables
* Apply device selection policies


### list layer devices
```console
$ MESA_VK_DEVICE_SELECT=list vulkaninfo
WARNING: [Loader Message] Code 0 : Layer VK_LAYER_MESA_device_select uses API version 1.3 which is older than the application specified API version of 1.4. May cause issues.
selectable devices:
  GPU 0: 10de:2786 "NVIDIA GeForce RTX 4070" discrete GPU 0000:3c:00.0
  GPU 1: 8086:46a6 "Intel(R) Graphics (ADL GT2)" integrated GPU 0000:00:02.0
  GPU 2: 10005:0 "llvmpipe (LLVM 19.1.1, 256 bits)" CPU 0000:00:00.0
```

### llvmpipe
This is a software implementation/driver that runs on the CPU. The name comes
from LLVM + "pipe" (referring to the graphics pipeline). It was originally built
for OpenGL but got extended with "lavapipe" for Vulkan support.

* Uses multiple CPU threads to simulate GPU parallelism
* Breaks work into tiles/blocks that run on different CPU cores
* Each "GPU thread" in your compute shader becomes actual CPU thread execution

It uses LLVM to compile shaders to optimized x86/x64 machine code at runtime.
Your SPIR-V compute shaders get:
```
SPIR-V bytecode → LLVM IR → JIT compiled x86 code → Runs on CPU cores
```
```
Your compute shader: "run this kernel on 1024 threads"
↓
LLVM: "compile this kernel to x86 code"
↓
CPU: "spawn 16 threads, each handles 64 iterations of the kernel"
```

