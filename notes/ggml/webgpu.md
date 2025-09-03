### GGML WebGPU Backend notes

### ShaderF16 not supported
When I try using the WebGPU backend I get the following error:
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
This is a feature that is enabled in:
```c++
static ggml_backend_dev_t ggml_backend_webgpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    WEBGPU_LOG_DEBUG("ggml_backend_reg_get_device()");

    ggml_backend_webgpu_reg_context * reg_ctx = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);

    webgpu_context ctx = reg_ctx->webgpu_ctx;

    wgpu::RequestAdapterOptions options = {};
    auto                        callback =
        [](wgpu::RequestAdapterStatus status, wgpu::Adapter adapter, const char * message, void * userdata) {
            if (status != wgpu::RequestAdapterStatus::Success) {
                GGML_LOG_ERROR("ggml_webgpu: Failed to get an adapter: %s\n", message);
                return;
            }
            *static_cast<wgpu::Adapter *>(userdata) = std::move(adapter);
        };
    void * userdata = &ctx->adapter;
    ctx->instance.WaitAny(
        ctx->instance.RequestAdapter(&options, wgpu::CallbackMode::AllowSpontaneous, callback, userdata), UINT64_MAX);
    GGML_ASSERT(ctx->adapter != nullptr);

    ctx->adapter.GetLimits(&ctx->limits);

    wgpu::AdapterInfo info{};
    ctx->adapter.GetInfo(&info);

    // Initialize device
    std::vector<wgpu::FeatureName> required_features = { wgpu::FeatureName::ShaderF16,
                                                        wgpu::FeatureName::ImplicitDeviceSynchronization };
    wgpu::DeviceDescriptor         dev_desc;
    dev_desc.requiredLimits       = &ctx->limits;
    dev_desc.requiredFeatures     = required_features.data();
    dev_desc.requiredFeatureCount = required_features.size();
```
And this is also used in the shaders, for example in `set_rows.wgsl`:
```
enable f16;

@group(0) @binding(0)
var<storage, read_write> src: array<f32>;
```
The problem is that my GPU driver does not seem to support this feature. I've got
a NVIDIA GeForce RTX 4070.

The shaders are loaded using functions like the following which loads the shader
for `set_rows.wgsl`:
```c++
static void ggml_webgpu_init_set_rows_pipeline(webgpu_context & webgpu_ctx) {
    std::vector<wgpu::ConstantEntry> constants(1);
    constants[0].key   = "wg_size";
    constants[0].value = webgpu_ctx->limits.maxComputeWorkgroupSizeX;
    ggml_webgpu_create_pipeline(
        webgpu_ctx->device, webgpu_ctx->set_rows_pipeline, wgsl_set_rows, "set_rows", constants);
}
```
If we search for wgsl_set_rows, which is string we won't find it in the codebase
as it is generated. We have to look in the build directory:
```
$ cat build/ggml/src/ggml-webgpu/generated/ggml-wgsl-shaders.hpp
...
const char* wgsl_set_rows = R"(enable f16;

@group(0) @binding(0)
var<storage, read_write> src: array<f32>;

@group(0) @binding(1)
var<storage, read_write> idx: array<u32>;

@group(0) @binding(2)
var<storage, read_write> dst: array<f16>;

@group(0) @binding(3)
var<storage, read_write> error: atomic<u32>;
...
```
In CMakeLists.txt we can find:
```cmake
# Shader locations
set(SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/wgsl-shaders")
set(SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(SHADER_HEADER "${SHADER_OUTPUT_DIR}/ggml-wgsl-shaders.hpp")
file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR})

message(STATUS "Shader output dir: ${SHADER_OUTPUT_DIR}")

# Find all WGSL files
file(GLOB WGSL_SHADER_FILES "${SHADER_DIR}/*.wgsl")

# Generate the header using a Python script
add_custom_command(
    OUTPUT ${SHADER_HEADER}
    COMMAND ${CMAKE_COMMAND} -E echo "Embedding WGSL shaders to ggml-wgsl-shaders.hpp"
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_OUTPUT_DIR}
    COMMAND ${CMAKE_COMMAND} -E env PYTHONIOENCODING=utf-8
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/wgsl-shaders/embed_wgsl.py
            --input_dir "${SHADER_DIR}"
            --output_file "${SHADER_HEADER}"
    DEPENDS ${WGSL_SHADER_FILES} ${CMAKE_CURRENT_SOURCE_DIR}/wgsl-shaders/embed_wgsl.py
    VERBATIM
)

add_custom_target(generate_shaders DEPENDS ${SHADER_HEADER})

ggml_add_backend_library(ggml-webgpu
    ggml-webgpu.cpp
    ${SHADER_HEADER}
    ../../include/ggml-webgpu.h
)

add_dependencies(ggml-webgpu generate_shaders)
```
Just to clarify this the custom command has an output of `ggml-wgsl-shaders.hpp`
and a custom target is addded that depends on this output (think prerequisite in
make terms). And this is added to the `ggml-webgpu` library which will require
that file to be generated before the library can be built.

We can force the generation of the shaders by running:
```console
$ make -B generate_shaders
[100%] Generating generated/ggml-wgsl-shaders.hpp
Embedding WGSL shaders to ggml-wgsl-shaders.hpp
[100%] Built target generate_shaders
```
