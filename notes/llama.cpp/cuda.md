## llama.cpp CUDA

### example/main
When llama-cli is run the CUDA backend is loaded by the following call in
arg.cpp:
```c++
common_params_context common_params_parser_init(common_params & params, llama_example ex, void(*print_usage)(int, char **)) {
    // load dynamic backends
    ggml_backend_load_all();
    ...
```
This function can be found in ggml-backend-reg.cpp:
```c++
void ggml_backend_load_all() {
    ggml_backend_load_all_from_path(nullptr);
}
```
Notice that nullptr is pass in as the `dir_path`:
```c++
void ggml_backend_load_all_from_path(const char * dir_path) {
#ifdef NDEBUG
    bool silent = true;
#else
    bool silent = false;
#endif

    ggml_backend_load_best("blas", silent, dir_path);
    ggml_backend_load_best("cann", silent, dir_path);
    ggml_backend_load_best("cuda", silent, dir_path);
    ggml_backend_load_best("hip", silent, dir_path);
    ggml_backend_load_best("kompute", silent, dir_path);
    ggml_backend_load_best("metal", silent, dir_path);
    ggml_backend_load_best("rpc", silent, dir_path);
    ggml_backend_load_best("sycl", silent, dir_path);
    ggml_backend_load_best("vulkan", silent, dir_path);
    ggml_backend_load_best("opencl", silent, dir_path);
    ggml_backend_load_best("musa", silent, dir_path);
    ggml_backend_load_best("cpu", silent, dir_path);
    // check the environment variable GGML_BACKEND_PATH to load an out-of-tree backend
    const char * backend_path = std::getenv("GGML_BACKEND_PATH");
    if (backend_path) {
        ggml_backend_load(backend_path);
    }
}
```
Lets step through `ggml_backend_load_best("cuda", silent, dir_path);`:
```c++
static ggml_backend_reg_t ggml_backend_load_best(const char * name, bool silent, const char * user_search_path) {
    // enumerate all the files that match [lib]ggml-name-*.[so|dll] in the search paths
    const fs::path name_path = fs::u8path(name);
    const fs::path file_prefix = backend_filename_prefix().native() + name_path.native() + fs::u8path("-").native();
    const fs::path file_extension = backend_filename_extension();
```
```console
(gdb) p name_path
$2 = filesystem::path "cuda"
(gdb) p backend_filename_prefix()
$3 = filesystem::path "libggml-"
(gdb) p file_prefix
$5 = filesystem::path "libggml-cuda-"
```
The shared libaray will then be loaded by the following code:
```c++
    int best_score = 0;
    fs::path best_path;

    for (const auto & search_path : search_paths) {
        if (!fs::exists(search_path)) {
            GGML_LOG_DEBUG("%s: search path %s does not exist\n", __func__, path_str(search_path).c_str());
            continue;
        }
        fs::directory_iterator dir_it(search_path, fs::directory_options::skip_permission_denied);
        for (const auto & entry : dir_it) {
            if (entry.is_regular_file()) {
                auto filename = entry.path().filename().native();
                auto ext = entry.path().extension().native();
                if (filename.find(file_prefix) == 0 && ext == file_extension) {
                    dl_handle_ptr handle { dl_load_library(entry) };
                    if (!handle && !silent) {
                        GGML_LOG_ERROR("%s: failed to load %s\n", __func__, path_str(entry.path()).c_str());
                    }
                    if (handle) {
                        auto score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
                        if (score_fn) {
                            int s = score_fn();
#ifndef NDEBUG
                            GGML_LOG_DEBUG("%s: %s score: %d\n", __func__, path_str(entry.path()).c_str(), s);
#endif
                            if (s > best_score) {
                                best_score = s;
                                best_path = entry.path();
                            }
                        } else {
                            if (!silent) {
                                GGML_LOG_INFO("%s: failed to find ggml_backend_score in %s\n", __func__, path_str(entry.path()).c_str());
                            }
                        }
                    }
                }
            }
        }
    }
```
Following that we have:
```c++
    if (best_score == 0) {
        // try to load the base backend
        for (const auto & search_path : search_paths) {
            fs::path filename = backend_filename_prefix().native() + name_path.native() + backend_filename_extension().native();
            fs::path path = search_path.native() + filename.native();
            if (fs::exists(path)) {
                return get_reg().load_backend(path, silent);
            }
        }
        return nullptr;
    }
```
Now, get_reg looks like this:
```c++
static ggml_backend_registry & get_reg() {
    static ggml_backend_registry reg;
    return reg;
}
```
This will initialize the `ggml_backend_registry` struct which is defined as:
```c++
struct ggml_backend_registry {
    std::vector<ggml_backend_reg_entry> backends;
    std::vector<ggml_backend_dev_t> devices;

    ggml_backend_registry() {
#ifdef GGML_USE_CUDA
        register_backend(ggml_backend_cuda_reg());
#endif
#ifdef GGML_USE_METAL
        register_backend(ggml_backend_metal_reg());
#endif
#ifdef GGML_USE_SYCL
        register_backend(ggml_backend_sycl_reg());
#endif
#ifdef GGML_USE_VULKAN
        register_backend(ggml_backend_vk_reg());
#endif
#ifdef GGML_USE_OPENCL
        register_backend(ggml_backend_opencl_reg());
#endif
#ifdef GGML_USE_CANN
        register_backend(ggml_backend_cann_reg());
#endif
#ifdef GGML_USE_BLAS
        register_backend(ggml_backend_blas_reg());
#endif
#ifdef GGML_USE_RPC
        register_backend(ggml_backend_rpc_reg());
#endif
#ifdef GGML_USE_KOMPUTE
        register_backend(ggml_backend_kompute_reg());
#endif
#ifdef GGML_USE_CPU
        register_backend(ggml_backend_cpu_reg());
#endif
    }
```
So in our case this will call `ggml_backend_cuda_reg()`:
```c++
// backend registry
ggml_backend_reg_t ggml_backend_cuda_reg() {
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_backend_cuda_reg_context * ctx = new ggml_backend_cuda_reg_context;

            for (int i = 0; i < ggml_cuda_info().device_count; i++) {
                ggml_backend_cuda_device_context * dev_ctx = new ggml_backend_cuda_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_CUDA_NAME + std::to_string(i);

                ggml_cuda_set_device(i);
                cudaDeviceProp prop;
                CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
                dev_ctx->description = prop.name;

                ggml_backend_dev_t dev = new ggml_backend_device {
                    /* .iface   = */ ggml_backend_cuda_device_interface,
                    /* .reg     = */ &reg,
                    /* .context = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_cuda_reg_interface,
                /* .context     = */ ctx
            };
        }

        initialized = true;
    }

    return &reg;
}
```
Notice that this is accessing CUDA API's like `cudaGetDeviceProperties` which
I initially thougth was not possible libggml-cuda.so has not been loaded yet,
but the contents of that shared libaray is the ggml code. The CUDA libraries
are linked with the llama executable.
When `ggml_cuda_info` is called it will call ggml_cuda_init:
```c++
const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = ggml_cuda_init();
    return info;
}
```
```c++
static ggml_cuda_device_info ggml_cuda_init() {
    ...
    ggml_cuda_device_info info = {};

    cudaError_t err = cudaGetDeviceCount(&info.device_count);

#if defined(GGML_USE_VMM)
        CUdevice device;
        CU_CHECK(cuDeviceGet(&device, id));
        CU_CHECK(cuDeviceGetAttribute(&device_vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device));

        if (device_vmm) {
            CUmemAllocationProp alloc_prop = {};
            alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            alloc_prop.location.id = id;
            CU_CHECK(cuMemGetAllocationGranularity(&info.devices[id].vmm_granularity, &alloc_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        }
#endif // defined(GGML_USE_VMM)
```
The `cuDeviceGetAttribute` function takes a pointer to a variable which will
be populated with 1 if the device supports virtual memory management and 0
otherwise. 
```console
(gdb) p device_vmm
$4 = 1
```
This is about virtual memory on the GPU which enables to create mappings between
the host and device memory. This is used in the `ggml_cuda_device_context` struct
Then a `CUmemAllocationProp` struct is created and populated with the following
with `CU_MEM_ALLOCATION_TYPE_PINNED` which is used to allocate pinned memory which
is memory that cannot be paged out. And that this memory should be located on
the device and not the host.
The granularity of the memory allocation is then queried with
`cuMemGetAllocationGranularity` which the alignment of the memory allocation. This
value is then stored in `info.devices[id].vmm_granularity`.
```console
(gdb) p info.devices[0].vmm_granularity
$7 = 2097152
```
So this is 2MB.
Following that we have:
```c++
info.devices[id].vmm = !!device_vmm;
```
This is first converting the integer to a boolean value which for a non-zero
value will become false. And the second ! will turn this into true. So it will
convert any non-zero value to true.
```c++
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
```
Next all the device properties are retrieved.
```c++
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, id));

        info.default_tensor_split[id] = total_vram;
        total_vram += prop.totalGlobalMem;

        info.devices[id].nsm       = prop.multiProcessorCount;
        info.devices[id].smpb      = prop.sharedMemPerBlock;
        info.devices[id].warp_size = prop.warpSize;
```
```console
(gdb) p prop.totalGlobalMem
$17 = 12481724416
(gdb) p prop.multiProcessorCount
$18 = 46
(gdb) p prop.sharedMemPerBlock
$19 = 49152
(gdb) p prop.warpSize
$20 = 32
```
So my GPU has 12GB of memory, 46 Streaming Multiprocessors, 48KB of shared memory per block and
a warpSize of 32 which is the number of threads in a warp.
Notice that these are stored for each device and the are in number of sm (nsm),
shared memory per block (smpb), warp size (warp_size).
```console
(gdb) p prop.sharedMemPerBlockOptin
$21 = 101376
```
This is an option that has to be specifically opted-in for and represents the
maximum amount of shared memory per block that can be allocated. The shared
memory and the L1 cache share the same physical memory space on each SM.
So, each SM has fixed amount of memory that can be used as either L1 cache or
shared memory. Now, if we only have one block on a SM the it has 48KB of shared
memory available to the block, and there would be about 52KB L1 cache available.
If a second block get scheduled to the same SM then it will also get 48K of
shared memory but this will mean that there is less memory available for the L1
cache. If a third block is attempted to be placed on the same SM this will not
work as there is not enough shared memory available for it.

This will then return back to ggml_backend_cuda_reg:
```c++
            for (int i = 0; i < ggml_cuda_info().device_count; i++) {
                ggml_backend_cuda_device_context * dev_ctx = new ggml_backend_cuda_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_CUDA_NAME + std::to_string(i);

                ggml_cuda_set_device(i);
                cudaDeviceProp prop;
                CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
                dev_ctx->description = prop.name;

                ggml_backend_dev_t dev = new ggml_backend_device {
                    /* .iface   = */ ggml_backend_cuda_device_interface,
                    /* .reg     = */ &reg,
                    /* .context = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_cuda_reg_interface,
                /* .context     = */ ctx
            };
        }

        initialized = true;

```
