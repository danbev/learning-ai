## Vulkan API for graphics and compute
Is an opensource project that is led by the Khronos Group. The Vulkan API is a
low-level graphics and compute API that provides high-efficiency, cross-platform
access to modern GPUs used in a wide variety of devices from PCs and consoles to
mobile phones and embedded platforms. 

There are proprietary alternatives like Nvidia's CUDA and Apple's Metal. Vulkan
is cross-platform and cross-vendor.

Kompute is general purpose GPU computing framework built upon Vulkan.

It is simlar to DirectX 12 or Metal. It can be used for low resource
environments such as mobile phones.

### Vulkan architecture
```
      My App
        ↓
Vulkan Loader (libvulkan.so)
        ↓
[Layers] ← Optional middleware (separate from ICDs)
        ↓
ICDs (vendor-specific drivers)
        ↓
GPU Hardware
```

### Layers
So a layer is like a middleware that can intercept Vulkan API calls and 
exist between the loader and the ICD. These are loaded by separate files:
```console
  33 [Vulkan Loader] LAYER:             Found the following files:
  34 [Vulkan Loader] LAYER:                /usr/share/vulkan/implicit_layer.d/nvidia_layers.json
  35 [Vulkan Loader] LAYER:                /usr/share/vulkan/implicit_layer.d/VkLayer_MESA_device_select.json
```
Lets take a look at `nvidia_layers.json`:
```console
{
    "file_format_version" : "1.0.1",
    "layers": [{
        "name": "VK_LAYER_NV_optimus",
        "type": "INSTANCE",
        "library_path": "libGLX_nvidia.so.0",
        "api_version" : "1.3.280",
        "implementation_version" : "1",
        "description" : "NVIDIA Optimus layer",
        "functions": {
            "vkGetInstanceProcAddr": "vk_optimusGetInstanceProcAddr",
            "vkGetDeviceProcAddr": "vk_optimusGetDeviceProcAddr"
        },
        "enable_environment": {
            "__NV_PRIME_RENDER_OFFLOAD": "1"
        },
        "disable_environment": {
            "DISABLE_LAYER_NV_OPTIMUS_1": ""
        }
    }]
}
```

I have the following:
```console
(venv) $ ls /usr/share/vulkan/implicit_layer.d/
nvidia_layers.json  VkLayer_MESA_device_select.json

(venv) $ ls /usr/share/vulkan/explicit_layer.d/
VkLayer_INTEL_nullhw.json  VkLayer_khronos_validation.json  VkLayer_MESA_overlay.json
```
What are the difference between implicit and explicit layers?
Implicit layers are automatically loaded by the Vulkan loader.

Explicit layers must be manually requested by your application code:
```c++
// You have to explicitly ask for these layers
const char* requestedLayers[] = { "VK_LAYER_KHRONOS_validation" };

VkInstanceCreateInfo createInfo = {};
createInfo.enabledLayerCount = 1;
createInfo.ppEnabledLayerNames = requestedLayers;
vkCreateInstance(&createInfo, NULL, &instance);
```
What will happen is that the loader will search through all the layer manifest
files to find a match for the `name` attribute and if found and load it using
the `library_path` attribute.


Now, the layers are specified by files in the filesystem, typically in
```console
(venv) $ ls /usr/share/vulkan/explicit_layer.d/
VkLayer_INTEL_nullhw.json  VkLayer_khronos_validation.json  VkLayer_MESA_overlay.json
```
The `VkLayer_khronos_validation.json` file contains a lot of information but
lets take a look at some of it:
```
{
    "file_format_version": "1.2.0",
    "layer": {
        "name": "VK_LAYER_KHRONOS_validation",
        "type": "GLOBAL",
        "library_path": "libVkLayer_khronos_validation.so",
        "api_version": "1.4.309",
```
So we can see the layer name the shared object file name.
After that we have the instance extensions:
```
        "instance_extensions": [
            {
                "name": "VK_EXT_debug_report",
                "spec_version": "9"
            },
            {
                "name": "VK_EXT_debug_utils",
                "spec_version": "1"
            },
            {
                "name": "VK_EXT_layer_settings",
                "spec_version": "2"
            },
            {
                "name": "VK_EXT_validation_features",
                "spec_version": "2"
            }
    ],
```
Instance means that these extentions are enabled/effect the entire Vulkan
instance and are often used for system wide features like debugging/validation.
These are enabled during vkCreateInstance().

Next we have device extensions which are for logical device level and they
provide functionality specific to a particular GPU/device and are enabled
per-device during vkCreateDevice().
```
        "device_extensions": [
            {
                "name": "VK_EXT_debug_marker",
                "spec_version": "4",
                "entrypoints": [
                    "vkDebugMarkerSetObjectTagEXT",
                    "vkDebugMarkerSetObjectNameEXT",
                    "vkCmdDebugMarkerBeginEXT",
                    "vkCmdDebugMarkerEndEXT",
                    "vkCmdDebugMarkerInsertEXT"
                ]
            },
            {
                "name": "VK_EXT_validation_cache",
                "spec_version": "1",
                "entrypoints": [
                    "vkCreateValidationCacheEXT",
                    "vkDestroyValidationCacheEXT",
                    "vkGetValidationCacheDataEXT",
                    "vkMergeValidationCachesEXT"
                ]
            },
            {
                "name": "VK_EXT_tooling_info",
                "spec_version": "1",
                "entrypoints": [
                    "vkGetPhysicalDeviceToolPropertiesEXT"
                ]
            }
        ],
```
Next we have features which is a large section in the file which defines
validation presets and configurations for the Khronos validation layer:
```
        "features": {
            "presets": [
                {
                    "label": "Standard",
                    "description": "Good default validation setup that balance validation coverage and performance.",
                    "platforms": [ "WINDOWS", "LINUX", "MACOS", "ANDROID" ],
                    "status": "STABLE",
                    "settings": [
                        { "key": "validate_core", "value": true },
                        { "key": "check_image_layout", "value": true },
                        { "key": "check_command_buffer", "value": true },
                        { "key": "check_object_in_use", "value": true },
                        { "key": "check_query", "value": true },
                        { "key": "check_shaders", "value": true },
                        { "key": "check_shaders_caching", "value": true },
                        { "key": "unique_handles", "value": true },
                        { "key": "object_lifetime", "value": true },
                        { "key": "stateless_param", "value": true },
                        { "key": "thread_safety", "value": false },
                        { "key": "validate_sync", "value": false },
                        { "key": "printf_enable", "value": false },
                        { "key": "gpuav_enable", "value": false },
                        { "key": "validate_best_practices", "value": false },
                        { "key": "report_flags", "value": [ "error", "warn" ] },
                        { "key": "debug_action", "value": [ "VK_DBG_LAYER_ACTION_LOG_MSG" ] },
                        { "key": "enable_message_limit", "value": true }
                    ]
                },
                ...
            }
            "settings": [
                {
                    "key": "validation_control",
                    "label": "Validation Areas",
                    "description": "Control of the Validation layer validation",
                    "type": "GROUP",
                    "expanded": true,
                    "settings": [
                        {
                            "key": "fine_grained_locking",
                            "env": "VK_LAYER_FINE_GRAINED_LOCKING",
                            "label": "Fine Grained Locking",
                            "description": "Enable fine grained locking for Core Validation, which should improve performance in multithreaded applications. This setting allows the optimization to be disabled for debugging.",
                            "type": "BOOL",
                            "default": true,
                            "platforms": [ "WINDOWS", "LINUX", "MACOS", "ANDROID" ]
                        },
                        {
                            "key": "validate_core",
                            "label": "Core",
                            "description": "The main, heavy-duty validation checks. This may be valuable early in the development cycle to reduce validation output while correcting parameter/object usage errors.",
                            "type": "BOOL",
                            "default": true,
                            "settings": [
                                {
                                    "key": "check_image_layout",
                                    "label": "Image Layout",
                                    "description": "Check that the layout of each image subresource is correct whenever it is used by a command buffer. These checks are very CPU intensive for some applications.",
                                    "type": "BOOL",
                                    "default": true,
                                    "dependence": {
                                        "mode": "ALL",
                                        "settings": [
                                            { "key": "validate_core", "value": true }
                                        ]
                                    }
                                },
```
Notice how `validate_core` is specified in the preset and then defined later
in the settings section.

Alright, so we have ICD that have extensions that can be enabled, and layers
(middleware). The layers extensions can be enabled regardless of the ICD. Like
they work on the intercepted invocation and don't depend on information from the
GPU. Whereas the device extensions actually are on/in the ICD and can directly
manipulate the target GPU (with commands/instructions)

Layer Extensions (Instance-level, ICD-independent):
* Work by intercepting Vulkan API calls between your application and the ICD
* Don't depend on GPU hardware capabilities
* Can be enabled regardless of which ICD/GPU you're using

Device Extensions (ICD-dependent, GPU-specific):

* Implemented inside the ICD/driver
* Directly control GPU hardware features and capabilities
* Can only be enabled if the specific GPU/driver supports them
* Add new functionality that requires hardware support


_wip_

```
Your vkCreateInstance() call
    ↓
Loader intercepts
    ↓
Validation layer (if enabled) - performs checks
    ↓
NVIDIA ICD's vkCreateInstance() - creates instance for RTX 4080
    ↓
AMD ICD's vkCreateInstance() - creates instance for RX 7800 XT
    ↓
Intel ICD's vkCreateInstance() - creates instance for integrated GPU
```

### ICD (Installable Client Driver)
An installable client driver (ICD) is a vendor-specific driver that implements
the actual Vulkan functionality for a specific GPU. 
For example, when we call vkCreateBuffer() we are not talking directly to the
GPU but we are doing through the ICD.

Vulkan has a loader that is responsible for loading the ICDs and routing the
calls to the appropriate ICD.

We can find the ICDs on our system by using the following command:
```console
$ export VK_LOADER_DEBUG=all
$ vulkaninfo --summary | grep Driver | grep icd | grep 'Found the following files'
...
[Vulkan Loader] DRIVER:            Found the following files:
[Vulkan Loader] DRIVER:               /usr/share/vulkan/icd.d/radeon_icd.x86_64.json
[Vulkan Loader] DRIVER:               /usr/share/vulkan/icd.d/nvidia_icd.json
[Vulkan Loader] DRIVER:               /usr/share/vulkan/icd.d/intel_icd.x86_64.json
[Vulkan Loader] DRIVER:               /usr/share/vulkan/icd.d/lvp_icd.x86_64.json
[Vulkan Loader] DRIVER:               /usr/share/vulkan/icd.d/intel_hasvk_icd.x86_64.json
[Vulkan Loader] DRIVER:               /usr/share/vulkan/icd.d/nouveau_icd.x86_64.json
[Vulkan Loader] DRIVER:               /usr/share/vulkan/icd.d/virtio_icd.x86_64.json
```
And we can list them using:
```console
$ ls /usr/share/vulkan/icd.d/
intel_hasvk_icd.x86_64.json  lvp_icd.x86_64.json      nvidia_icd.json         virtio_icd.x86_64.json
intel_icd.x86_64.json        nouveau_icd.x86_64.json  radeon_icd.x86_64.json
```
Now, lets take a closer look at `nvidia_icd.json`:
```console
$ cat /usr/share/vulkan/icd.d/nvidia_icd.json
{
    "file_format_version" : "1.0.1",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version" : "1.3.280"
    }
}
```

### ggml
```console
$ cmake -S . -B build -DGGML_VULKAN=ON --debug-find-pkg=Vulkan -DCMAKE_VERBOSE_MAKEFILE=ON
...
CMake Debug Log at /usr/share/cmake-3.28/Modules/FindVulkan.cmake:304 (find_path):
  find_path called with the following settings:
```
If we inspect `/usr/share/cmake-3.28/Modules/FindVulkan.cmake` we can find the
following:
```console
# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindVulkan
----------

.. versionadded:: 3.7

Find Vulkan, which is a low-overhead, cross-platform 3D graphics
and computing API.

...

#]=======================================================================]
```
This is a reStructuredText comment block. This can be displayed by CMake's help
system:
```console
$ cmake --help-module FindVulkan
```

### Cooperative Matrices
This is a Vulkan extension that provides direct access to specialized
tensor/matrix hardware units on modern GPUs.
It is a new matrix type


### Vulkan in ggml
The first interaction, at least when running `llama-cli` is the following:
```console
(gdb) bt
#0  ggml_vk_instance_init () at /home/danbev/work/ai/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp:4213
#1  0x00007ffff4cc87c5 in ggml_backend_vk_reg () at /home/danbev/work/ai/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp:11992
#2  0x00007ffff7fa96d6 in ggml_backend_registry::ggml_backend_registry (this=0x7ffff7fbc120 <get_reg()::reg>)
    at /home/danbev/work/ai/llama.cpp/ggml/src/ggml-backend-reg.cpp:182
#3  0x00007ffff7fa6cce in get_reg () at /home/danbev/work/ai/llama.cpp/ggml/src/ggml-backend-reg.cpp:312
#4  0x00007ffff7fa7f3b in ggml_backend_load_best (name=0x7ffff7fb3717 "vulkan", silent=false, user_search_path=0x0)
    at /home/danbev/work/ai/llama.cpp/ggml/src/ggml-backend-reg.cpp:564
#5  0x00007ffff7fa8410 in ggml_backend_load_all_from_path (dir_path=0x0)
    at /home/danbev/work/ai/llama.cpp/ggml/src/ggml-backend-reg.cpp:591
#6  0x00007ffff7fa8331 in ggml_backend_load_all () at /home/danbev/work/ai/llama.cpp/ggml/src/ggml-backend-reg.cpp:574
#7  0x0000555555601e60 in common_params_parser_init (params=..., ex=LLAMA_EXAMPLE_MAIN,
    print_usage=0x5555555d795e <print_usage(int, char**)>) at /home/danbev/work/ai/llama.cpp/common/arg.cpp:1268
#8  0x00005555555f6ed2 in common_params_parse (argc=10, argv=0x7fffffffd698, params=..., ex=LLAMA_EXAMPLE_MAIN,
    print_usage=0x5555555d795e <print_usage(int, char**)>) at /home/danbev/work/ai/llama.cpp/common/arg.cpp:1222
#9  0x00005555555d7fbd in main (argc=10, argv=0x7fffffffd698) at /home/danbev/work/ai/llama.cpp/tools/main/main.cpp:89
```
So we can see that `ggml_backend_load_all` is called which will trickle down
into `ggml_backend_vk_reg` which will call `ggml_vk_instance_init`.
```c++
static void ggml_vk_instance_init() {
    if (vk_instance_initialized) {
        return;
    }
    VK_LOG_DEBUG("ggml_vk_instance_init()");

    uint32_t api_version = vk::enumerateInstanceVersion();

```

### validate_extensions issue
```c++
static void ggml_vk_instance_init() {
    ...
    const bool validation_ext = ggml_vk_instance_validation_ext_available(instance_extensions);

```
```c++
static bool ggml_vk_instance_validation_ext_available(const std::vector<vk::ExtensionProperties>& instance_extensions) {
#ifdef GGML_VULKAN_VALIDATE
    bool portability_enumeration_ext = false;
    // Check for portability enumeration extension for MoltenVK support
    for (const auto& properties : instance_extensions) {
        if (strcmp("VK_KHR_portability_enumeration", properties.extensionName) == 0) {
            return true;
        }
    }
    if (!portability_enumeration_ext) {
        std::cerr << "ggml_vulkan: WARNING: Instance extension VK_KHR_portability_enumeration not found." << std::endl;
    }
#endif
    return false;

    UNUSED(instance_extensions);
}
```
And this looks very similar to:
```c++
static bool ggml_vk_instance_portability_enumeration_ext_available(const std::vector<vk::ExtensionProperties>& instance_extensions) {
#ifdef __APPLE__
    bool portability_enumeration_ext = false;
    // Check for portability enumeration extension for MoltenVK support
    for (const auto& properties : instance_extensions) {
        if (strcmp("VK_KHR_portability_enumeration", properties.extensionName) == 0) {
            return true;
        }
    }
    if (!portability_enumeration_ext) {
        std::cerr << "ggml_vulkan: WARNING: Instance extension VK_KHR_portability_enumeration not found." << std::endl;
    }
#endif
    return false;

    UNUSED(instance_extensions);
}
```
`validation_ext` is later used here:
```c++
    if (validation_ext) {
        layers.push_back("VK_LAYER_KHRONOS_validation");
    }
    std::vector<const char*> extensions;
    if (validation_ext) {
        extensions.push_back("VK_EXT_validation_features");
    }
```
I wonder if this should be using `VK_KHR_portability_enumeration` instead:
```c++
static bool ggml_vk_instance_validation_ext_available(const std::vector<vk::ExtensionProperties>& instance_extensions) {
#ifdef GGML_VULKAN_VALIDATE
    // Check for validation feature extension
    for (const auto& properties : instance_extensions) {
        if (strcmp("VK_EXT_validation_features", properties.extensionName) == 0) {
            return true;
        }
    }
    std::cerr << "ggml_vulkan: WARNING: Instance extension VK_KHR_portability_enumeration not found." << std::endl;
#endif
    return false;

    UNUSED(instance_extensions);
}
```




### enumerateInstanceExtensionProperties
The loader will handle this just like everything else and will delegate the
call to the NVIDIA ICD and ask what extensions it supports. And likewise for
the other ICDs.
```console
    const std::vector<vk::ExtensionProperties> instance_extensions = vk::enumerateInstanceExtensionProperties();
```
Now, the above is from ggml and it used the C++ wrapper api:
```c++
#include <vulkan/vulkan.hpp>
```
This is what the function does internally:
```c++
    // What the wrapper does internally:
    uint32_t extensionCount = 0;
    vk::enumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

    std::vector<vk::ExtensionProperties> extensions(extensionCount);
    vk::enumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

    return extensions; // Returns the vector directly
```

This is asking the ICDs for if it has an extension but it is asking for the
wrong extension. How do I know which extension they original author had in mind.
The later use the returned boolean like this:
```c++
    if (validation_ext) {
        layers.push_back("VK_LAYER_KHRONOS_validation");
    }
    std::vector<const char*> extensions;
    if (validation_ext) {
        extensions.push_back("VK_EXT_validation_features");
```
So one is adding the layer validation, and then an extension
`VK_EXT_validation_features`.  The `VK_EXT_validation_features` extension
provides enhanced validation capabilities:
* Fine-grained control over which validation checks to enable/disable
* GPU-assisted validation for more thorough checking
* Performance optimization by selectively enabling only needed validation

But this extension isn't always available - it depends on:
* Vulkan SDK version
* Validation layer version
* ICD support for the extension


So I have the following instance extensions (which recall are extensions that
the ICD provides):
```console
Breakpoint 1, ggml_vk_instance_init () at /home/danbev/work/ai/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp:4228
4228	    const bool validation_ext = ggml_vk_instance_validation_ext_available(instance_extensions);
(gdb) p instance_extensions 
$1 = std::vector of length 25, capacity 25 = {{extensionName = {<std::array<char, 256>> = {
        _M_elems = "VK_KHR_device_group_creation", '\000' <repeats 227 times>}, <No data fields>}, specVersion = 1}, {
    extensionName = {<std::array<char, 256>> = {_M_elems = "VK_KHR_display", '\000' <repeats 241 times>}, <No data fields>}, 
    specVersion = 23}, {extensionName = {<std::array<char, 256>> = {
        _M_elems = "VK_KHR_external_fence_capabilities", '\000' <repeats 221 times>}, <No data fields>}, specVersion = 1}, {
    extensionName = {<std::array<char, 256>> = {
        _M_elems = "VK_KHR_external_memory_capabilities", '\000' <repeats 220 times>}, <No data fields>}, specVersion = 1}, {
    extensionName = {<std::array<char, 256>> = {
        _M_elems = "VK_KHR_external_semaphore_capabilities", '\000' <repeats 217 times>}, <No data fields>}, specVersion = 1}, {
    extensionName = {<std::array<char, 256>> = {
        _M_elems = "VK_KHR_get_display_properties2", '\000' <repeats 225 times>}, <No data fields>}, specVersion = 1}, {
    extensionName = {<std::array<char, 256>> = {
        _M_elems = "VK_KHR_get_physical_device_properties2", '\000' <repeats 217 times>}, <No data fields>}, specVersion = 2}, {
    extensionName = {<std::array<char, 256>> = {
        _M_elems = "VK_KHR_get_surface_capabilities2", '\000' <repeats 223 times>}, <No data fields>}, specVersion = 1}, {
    extensionName = {<std::array<char, 256>> = {_M_elems = "VK_KHR_surface", '\000' <repeats 241 times>}, <No data fields>}, 
    specVersion = 25}, {extensionName = {<std::array<char, 256>> = {
        _M_elems = "VK_KHR_surface_protected_capabilities", '\000' <repeats 218 times>}, <No data fields>}, specVersion = 1}, {
    extensionName = {<std::array<char, 256>> = {_M_elems = "VK_KHR_wayland_surface", '\000' <repeats 233 times>}, <No data fields>}, 
    specVersion = 6}, {extensionName = {<std::array<char, 256>> = {
        _M_elems = "VK_KHR_xcb_surface", '\000' <repeats 237 times>}, <No data fields>}, specVersion = 6}, {
    extensionName = {<std::array<char, 256>> = {_M_elems = "VK_KHR_xlib_surface", '\000' <repeats 236 times>}, <No data fields>}, 
    specVersion = 6}, {extensionName = {<std::array<char, 256>> = {
        _M_elems = "VK_EXT_acquire_drm_display", '\000' <repeats 229 times>}, <No data fields>}, specVersion = 1}, {
    extensionName = {<std::array<char, 256>> = {
        _M_elems = "VK_EXT_acquire_xlib_display", '\000' <repeats 228 times>}, <No data fields>}, specVersion = 1}, {
    extensionName = {<std::array<char, 256>> = {_M_elems = "VK_EXT_debug_report", '\000' <repeats 236 times>}, <No data fields>}, 
    specVersion = 10}, {extensionName = {<std::array<char, 256>> = {
        _M_elems = "VK_EXT_debug_utils", '\000' <repeats 237 times>}, <No data fields>}, specVersion = 2}, {
    extensionName = {<std::array<char, 256>> = {
        _M_elems = "VK_EXT_direct_mode_display", '\000' <repeats 229 times>}, <No data fields>}, specVersion = 1}, {
    extensionName = {<std::array<char, 256>> = {
        _M_elems = "VK_EXT_display_surface_counter", '\000' <repeats 225 times>}, <No data fields>}, specVersion = 1}, {
    extensionName = {<std::array<char, 256>> = {
        _M_elems = "VK_EXT_headless_surface", '\000' <repeats 232 times>}, <No data fields>}, specVersion = 1}, {
    extensionName = {<std::array<char, 256>> = {
        _M_elems = "VK_EXT_surface_maintenance1", '\000' <repeats 228 times>}, <No data fields>}, specVersion = 1}, {
    extensionName = {<std::array<char, 256>> = {
        _M_elems = "VK_EXT_swapchain_colorspace", '\000' <repeats 228 times>}, <No data fields>}, specVersion = 4}, {
    extensionName = {<std::array<char, 256>> = {
        _M_elems = "VK_NV_display_stereo", '\000' <repeats 235 times>, "\377"}, <No data fields>}, specVersion = 1}, {
    extensionName = {<std::array<char, 256>> = {
        _M_elems = "VK_KHR_portability_enumeration", '\000' <repeats 225 times>}, <No data fields>}, specVersion = 1}, {
    extensionName = {<std::array<char, 256>> = {
        _M_elems = "VK_LUNARG_direct_driver_loading", '\000' <repeats 224 times>}, <No data fields>}, specVersion = 1}}
```
Notice that I have `VK_KHR_portability_enumeration`.
```c++
    const bool validation_ext = ggml_vk_instance_validation_ext_available(instance_extensions);
```
Now, the name of this function is vk_instane_validation_ext_available which
seems a little strange because it is checking for portability enumeration. And
like we mentioned earlier this might be an error but I'm not sure yet.
But either way there is the `portability_enumeration_ext` variable which is 
set to false and never updated so it could be removed.
```c++
static bool ggml_vk_instance_validation_ext_available(const std::vector<vk::ExtensionProperties>& instance_extensions) {
#ifdef GGML_VULKAN_VALIDATE
    bool portability_enumeration_ext = false;
    // Check for portability enumeration extension for MoltenVK support
    for (const auto& properties : instance_extensions) {
        if (strcmp("VK_KHR_portability_enumeration", properties.extensionName) == 0) {
            return true;
        }
    }
    if (!portability_enumeration_ext) {
        std::cerr << "ggml_vulkan: WARNING: Instance extension VK_KHR_portability_enumeration not found." << std::endl;
    }
#endif
    return false;

    UNUSED(instance_extensions);
}
```
So in my case this function will return true because I have the
`VK_KHR_portability_enumeration` extension.
```console
(gdb) p validation_ext
$3 = true
```
This variable is then used here:
```c++
    if (validation_ext) {
        layers.push_back("VK_LAYER_KHRONOS_validation");
    }
```
```console
(gdb) p layers
$5 = std::vector of length 1, capacity 1 = {0x7ffff4d716ff "VK_LAYER_KHRONOS_validation"}
```
I don't understand why the ICD extensions should be considered here. My
understanding is that layers come from the Vulkan SDK and are not tied to a
specific ICD but are separate, that they are a layer inbetween the Vulkan loader
and the ICD. So it should be possible I though to enable
`VK_LAYER_KHRONOS_validation` regardless of the ICD extensions.

Next, the variable is used here:
```c++
    std::vector<const char*> extensions;
    if (validation_ext) {
        extensions.push_back("VK_EXT_validation_features");
    }
```
Now, for my system this will actually add the `VK_EXT_validation_features` which
my ICD does not have.
```console
(gdb) p extensions
$4 = std::vector of length 1, capacity 1 = {0x7ffff4d7171b "VK_EXT_validation_features"}
```
```c++
    vk::InstanceCreateInfo instance_create_info(vk::InstanceCreateFlags{}, &app_info, layers, extensions);
```
The variable is used again here:
```c++
    std::vector<vk::ValidationFeatureEnableEXT> features_enable;
    vk::ValidationFeaturesEXT validation_features;

    if (validation_ext) {
        features_enable = { vk::ValidationFeatureEnableEXT::eBestPractices };
        validation_features = { features_enable, {}, };
        validation_features.setPNext(nullptr);
        instance_create_info.setPNext(&validation_features);
        GGML_LOG_DEBUG("ggml_vulkan: Validation layers enabled\n");
    }
```
From https://registry.khronos.org/vulkan/specs/latest/man/html/VkValidationFeatureEnableEXT.html
we can read that:  
`VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT` specifies that Vulkan
best-practices validation is enabled. Activating this feature enables the output
of warnings related to common misuse of the API, but which are not explicitly
prohibited by the specification. This feature is disabled by default.

And then the instance is created using the `instance_create_info`:
```c++
    vk_instance.instance = vk::createInstance(instance_create_info);
    vk_instance_initialized = true;
```
I get the follow warning:
```console
Validation Warning: [ BestPractices-deprecated-extension ] | MessageID = 0xda8260ba
vkCreateInstance(): Attempting to enable deprecated extension VK_EXT_validation_features, but this extension has been deprecated by VK_EXT_layer_settings.

Validation Warning: [ BestPractices-specialuse-extension ] | MessageID = 0x675dc32e
vkCreateInstance(): Attempting to enable extension VK_EXT_validation_features, but this extension is intended to support use by applications when debugging and it is strongly recommended that it be otherwise avoided.
```

Now, I if I try adding an extension that I know does not exist, for example:
```c++
    if (validation_ext) {
        extensions.push_back("VK_EXT_validation_features_bajja");
    }
```
This will not cause a visible error but when I run:
```c++
    vk_instance.instance = vk::createInstance(instance_create_info);
    vk_instance_initialized = true;
```
The call will not continue to the second line here but will instead break
out of it.
So I get here:
```console
ggml_vulkan: Validation layers enabled
4270	    vk_instance.instance = vk::createInstance(instance_create_info);
```
But this will then break out of the function and the program will just
continue.
```console
(gdb) n
register_backend: registered backend CPU (1 devices)
register_device: registered device CPU (12th Gen Intel(R) Core(TM) i7-1260P)
load_backend: failed to find ggml_backend_init in /home/danbev/work/ai/llama.cpp/build/bin/libggml-vulkan.so
load_backend: failed to find ggml_backend_init in /home/danbev/work/ai/llama.cpp/build/bin/libggml-cpu.so
warning: no usable GPU found, --gpu-layers option will be ignored
warning: one possible reason is that llama.cpp was compiled without GPU support
warning: consult docs/build.md for compilation instructions
[New Thread 0x7fffdf5ff6c0 (LWP 77285)]
build: 6316 (009b709d6) with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu (debug)
main: llama backend init
```

Hmm, actually if I enable `-DGGML_VULKAN_DEBUG=ON` I can see that there is
an exception thrown but I was not seeing it before because it was logged at
debug:
```c++
ggml_backend_reg_t ggml_backend_vk_reg() {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_vk_reg_i,
        /* .context     = */ nullptr,
    };
    try {
        ggml_vk_instance_init();
        return &reg;
    } catch (const vk::SystemError& e) {
        VK_LOG_DEBUG("ggml_backend_vk_reg() -> Error: System error: " << e.what());
        return nullptr;
    }
}
```
Enabling this and running I get:
```console
ggml_backend_vk_reg() -> Error: System error: vk::createInstance: ErrorExtensionNotPresent
```

```c++
        extension_prop = get_extension_property(pCreateInfo->ppEnabledExtensionNames[i], icd_exts);
```
```console
(gdb) p pCreateInfo->ppEnabledExtensionNames[0]
$3 = 0x7ffff4d88c53 "VK_EXT_validation_features"

(gdb) p extension_prop
$10 = (VkExtensionProperties *) 0x0
```
So this extension is not in the icd_extension list which knew.

