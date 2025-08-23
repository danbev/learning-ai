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
$ cmake -S . -B out/Release -DDAWN_BUILD_MONOLITHIC_LIBRARY=SHARED -DDAWN_FETCH_DEPENDENCIES=ON -DDAWN_ENABLE_INSTALL=ON -DCMAKE_BUILD_TYPE=Release
$ cmake --build out/Release -j8
$ cmake --install out/Release --prefix install/Release
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
