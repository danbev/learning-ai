## Google Dawn example
This is a WebGPU example in C++ that used Google's Dawn library.


### Install Dawn
```console
$ git clone https://chromium.googlesource.com/chromium/tools/depot_tools.git
$ export PATH="${PWD}/depot_tools:${PATH}"
```
There is a `set-env.sh` script that can be used to set the environment
variable.

Next, clone the Dawn repository:
```console
$ git clone https://dawn.googlesource.com/dawn
$ cd dawn
```

Fetch dependencies using gclient:
```console
$ cp scripts/standalone.gclient .gclient
$ gclient sync
```

```console
$ sudo apt install libxinerama-dev libxcursor-dev libx11-xcb-dev
```
Build Dawn:
```console
$ mkdir -p out/Debug
$ cd out/Debug
$ cmake ../..
$ make -j8
```

### Building with CMake
https://github.com/google/dawn/blob/main/docs/quickstart-cmake.md

The above document has instructions for configuring CMake to build Dawn, but
I found that the install command did not work as expected. There are no errors
but nothing is installed to the specified install directory. What I needed to do
to get this working was add `DWAN_BUILD_MONOLITHIC_LIBRARY=SHARED` to get anything
copied to the install directory:
```console
#!/bin/bash

set -e

cmake -S . -B out/Release -DDAWN_BUILD_MONOLITHIC_LIBRARY=SHARED \
    -DDAWN_FETCH_DEPENDENCIES=ON \
    -DDAWN_ENABLE_INSTALL=ON \
    -DDAWN_USE_BUILT_DXC=ON \
    -DCMAKE_BUILD_TYPE=Release

cmake --build out/Release -j8

cmake --install out/Release --prefix install/Release
```

### Timeout waits are not supported
```console
$ make run-list-adapters
env LD_LIBRARY_PATH=/home/danbev/work/webgpu/dawn/install/Release/lib \
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
    bin/list-adapters
Attempting to create WebGPU instance...
WebGPU instance created successfully!
Requesting adapter...
Adapter found successfully!
Error: Timeout waits are either not enabled or not supported.
Adapter request completed successfully!
Adapter limits retrieved successfully!
Adapter Information:
  Vendor: nvidia
  Architecture: lovelace
  Device: NVIDIA GeForce RTX 4070
  Description: NVIDIA: 560.35.05 560.35.5.0
```
This error is coming from `src/dawn/native/EventManager.cpp`:
```c++
wgpu::WaitStatus EventManager::WaitAny(size_t count, FutureWaitInfo* infos, Nanoseconds timeout) {
    DAWN_ASSERT(!IsShutDown());

    // Validate for feature support.
    if (timeout > Nanoseconds(0)) {
        if (!mTimedWaitAnyEnable) {
            mInstance->EmitLog(WGPULoggingType_Error,
                               "Timeout waits are either not enabled or not supported.");
            return wgpu::WaitStatus::Error;
        }
```
So is appears that `mTimedWaitAnyEnable` is not set to true. This is set in:
```c++
MaybeError EventManager::Initialize(const UnpackedPtr<InstanceDescriptor>& descriptor) {
    if (descriptor) {
        for (auto feature :
             std::span(descriptor->requiredFeatures, descriptor->requiredFeatureCount)) {
            if (feature == wgpu::InstanceFeatureName::TimedWaitAny) {
                mTimedWaitAnyEnable = true;
                break;
            }
        }
```
So this looks like a feature that has to be set.
```c++
    wgpu::InstanceDescriptor instanceDescriptor{};

    // This is an array of required features for the instanceDescriptor above.
    wgpu::InstanceFeatureName requiredFeatures[] = {
        wgpu::InstanceFeatureName::TimedWaitAny
    };
    instanceDescriptor.requiredFeatureCount = 1;  // requiring 1 feature
    instanceDescriptor.requiredFeatures = requiredFeatures;
```

### ShaderF16 support
When testing the WebGPU backend in llama.cpp I ran into the following error:
```console
ggml_webgpu: Failed to get a device: Invalid feature required: Requested feature FeatureName::ShaderF16 is not supported.
    at CreateDeviceInternal (/home/danbev/work/webgpu/dawn/src/dawn/native/Adapter.cpp:290)

ggml_webgpu: Device lost! Reason: 4, Message: Failed to create device:
Invalid feature required: Requested feature FeatureName::ShaderF16 is not supported.
    at CreateDeviceInternal (/home/danbev/work/webgpu/dawn/src/dawn/native/Adapter.cpp:290)

/home/danbev/work/ai/llama.cpp/ggml/src/ggml-webgpu/ggml-webgpu.cpp:1202: GGML_ASSERT(ctx->device != nullptr) failed

Program received signal SIGABRT, Aborted.
__pthread_kill_implementation (no_tid=0, signo=6, threadid=<optimized out>) at ./nptl/pthread_kill.c:44
warning: 44	./nptl/pthread_kill.c: No such file or directory
```
I started looking to make sure that by Vulkan driver supports F16 shaders, and
then also that I was building Dawn correctly.

Vulkan info:
```console
$ vulkaninfo | grep -A 20 -B 5 -i "float16\|16bit\|shader.*16"
...

VkPhysicalDeviceVulkan12Features:
---------------------------------
	samplerMirrorClampToEdge                           = true
	drawIndirectCount                                  = true
	storageBuffer8BitAccess                            = true
	uniformAndStorageBuffer8BitAccess                  = true
	storagePushConstant8                               = true
	shaderBufferInt64Atomics                           = true
	shaderSharedInt64Atomics                           = true
	shaderFloat16                                      = true
```
So my hardware, Vulkan driver, and extensions all support F16. The issue is
definitely that Dawn WebGPU doesn't properly expose this Vulkan F16 capability
through its WebGPU ShaderF16 feature.

Digging into the Dawn code, I found the following in
`src/dawn/native/vulkan/PhysicalDeviceVk.cpp`:
```c++
void PhysicalDevice::InitializeSupportedFeaturesImpl() {
    ...

    bool shaderF16Enabled = false;
    if (mDeviceInfo.HasExt(DeviceExt::ShaderFloat16Int8) &&
        mDeviceInfo.HasExt(DeviceExt::_16BitStorage) &&
        mDeviceInfo.shaderFloat16Int8Features.shaderFloat16 == VK_TRUE &&
        mDeviceInfo._16BitStorageFeatures.storageBuffer16BitAccess == VK_TRUE &&
        mDeviceInfo._16BitStorageFeatures.uniformAndStorageBuffer16BitAccess == VK_TRUE) {
        // TODO(crbug.com/tint/2164): Investigate crashes in f16 CTS tests to enable on NVIDIA.
        if (!gpu_info::IsNvidia(GetVendorId())) {
            EnableFeature(Feature::ShaderF16);
            shaderF16Enabled = true;
        } else {
            printf("[danbev] ShaderF16 feature is not enabled on NVIDIA devices due to known issues.\n");
        }
    }
    ...
```
So it looks like this in only an issue with NVIDIA drivers:
```console
$ make features
g++ -L/home/danbev/work/webgpu/dawn/install/Release/lib -ldl \
	-std=c++20 -Wall -Wextra -g -I/home/danbev/work/webgpu/dawn/install/Release/include \
	src/features.cpp -o bin/features -lwebgpu_dawn
m$ make run-features
g++ -L/home/danbev/work/webgpu/dawn/install/Release/lib -ldl \
	-std=c++20 -Wall -Wextra -g -I/home/danbev/work/webgpu/dawn/install/Release/include \
	src/features.cpp -o bin/features -lwebgpu_dawn
env LD_LIBRARY_PATH=/home/danbev/work/webgpu/dawn/install/Release/lib \
	VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
	bin/features
Testing ShaderF16 support...
[danbev] ShaderF16 feature is not enabled on NVIDIA devices due to known issues.
Testing on: NVIDIA GeForce RTX 4070 (Backend: Vulkan)

Testing ShaderF16 support...
✗ ShaderF16 NOT supported
  Error: Invalid feature required: Requested feature FeatureName::ShaderF16 is not supported.
    at CreateDeviceInternal (/home/danbev/work/webgpu/dawn/src/dawn/native/Adapter.cpp:290)


Testing device creation without ShaderF16...
✓ Basic device creation works

=== RESULTS ===
ShaderF16: ✗ NOT SUPPORTED
This is why GGML fails with your current setup.
Basic WebGPU: ✓ WORKING
Your WebGPU setup is functional, just missing F16 support.
make: *** [Makefile:57: run-features] Error 1
```
The issue seems to be that there are tests that are failing in Dawn and this
is why it is disabled. I tried forcing the feature and it seems to work for me
at least I can get llama.cpp up and running. Which is pretty much all I want
at this point so I can learn more about WebGPU.
