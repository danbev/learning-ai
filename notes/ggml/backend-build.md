## GGML Backend build notes
When building ggml it is possible to specify the GGML_BACKEND_DL, for dynamic
library (shared library) to be built. When this is enabled the backend will
be built as a module library, which means a .so file on Linux, a .dylib file on
windows which can be loaded at runtime.

When we build normally without GGML_BACKEND_DL the backends are still shared
object but the are not linked dynamically at runtime, instead they are linked
statically at build time.
```console
$ ldd build/bin/libggml.so
	linux-vdso.so.1 (0x00007aaa82bce000)
	libggml-cpu.so.0 => /home/danbev/work/ai/llama.cpp/build/bin/libggml-cpu.so.0 (0x00007aaa82800000)
	libggml-base.so.0 => /home/danbev/work/ai/llama.cpp/build/bin/libggml-base.so.0 (0x00007aaa82a8a000)
	libstdc++.so.6 => /lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007aaa82400000)
	libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007aaa82a44000)
	libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007aaa82000000)
	libgomp.so.1 => /lib/x86_64-linux-gnu/libgomp.so.1 (0x00007aaa827aa000)
	libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007aaa826c1000)
	/lib64/ld-linux-x86-64.so.2 (0x00007aaa82bd0000)
```
And if we compare this with the same library built with GGML_BACKEND_DL we can
see that it does not link against ligggml-cpu.so.0:
```console
$ ldd build-backends/bin/libggml.so
	linux-vdso.so.1 (0x000073f8d9dee000)
	libggml-base.so.0 => /home/danbev/work/ai/llama.cpp/build-backends/bin/libggml-base.so.0 (0x000073f8d9caa000)
	libstdc++.so.6 => /lib/x86_64-linux-gnu/libstdc++.so.6 (0x000073f8d9a00000)
	libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x000073f8d99d2000)
	libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x000073f8d9600000)
	libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x000073f8d98e9000)
	/lib64/ld-linux-x86-64.so.2 (0x000073f8d9df0000)
```

### libraries
There is ggml-base which is the foundation library which contains backend agnostic
code, tensor operations, quantization code etc.
```cmake
add_library(ggml-base
            ../include/ggml.h
            ../include/ggml-alloc.h
            ../include/ggml-backend.h
            ../include/ggml-cpp.h
            ../include/ggml-opt.h
            ../include/gguf.h
            ggml.c
            ggml.cpp
            ggml-alloc.c
            ggml-backend.cpp
            ggml-opt.cpp
            ggml-threading.cpp
            ggml-threading.h
            ggml-quants.c
            ggml-quants.h
            gguf.cpp)

set_target_properties(ggml-base PROPERTIES
    VERSION ${GGML_VERSION}
    SOVERSION ${GGML_VERSION_MAJOR}
)

target_include_directories(ggml-base PRIVATE .)
if (GGML_BACKEND_DL)
    target_compile_definitions(ggml-base PUBLIC GGML_BACKEND_DL)
endif()

if (GGML_SCHED_NO_REALLOC)
    target_compile_definitions(ggml-base PUBLIC GGML_SCHED_NO_REALLOC)
endif()
```

Then there is ggml which builts upon ggml-base and adds backend-reg.cpp:
```cmake
add_library(ggml
            ggml-backend-reg.cpp)
add_library(ggml::ggml ALIAS ggml)

set_target_properties(ggml PROPERTIES
    VERSION ${GGML_VERSION}
    SOVERSION ${GGML_VERSION_MAJOR}
)
```

Backends are added like this:
```cmake
ggml_add_backend(BLAS)
ggml_add_backend(CANN)
ggml_add_backend(CUDA)
ggml_add_backend(HIP)
ggml_add_backend(METAL)
ggml_add_backend(MUSA)
ggml_add_backend(RPC)
ggml_add_backend(SYCL)
ggml_add_backend(Vulkan)
ggml_add_backend(WebGPU)
ggml_add_backend(zDNN)
ggml_add_backend(OpenCL)
ggml_add_backend(Hexagon)
```
```cmake
function(ggml_add_backend backend)
    string(TOUPPER "GGML_${backend}" backend_id) // GGML_CUDA
    if (${backend_id})
        string(TOLOWER "ggml-${backend}" backend_target) // ggml-cuda
        add_subdirectory(${backend_target})   // add ggml-cuda directory
        message(STATUS "Including ${backend} backend")
        if (NOT GGML_BACKEND_DL) // only when not building dynamic backends
            string(TOUPPER "GGML_USE_${backend}" backend_use)
            target_compile_definitions(ggml PUBLIC ${backend_use})
        endif()
    endif()
endfunction()
```

```cmake
function(ggml_add_backend_library backend)
    if (GGML_BACKEND_DL)
        add_library(${backend} MODULE ${ARGN}) // creates a backend as a module librar (.so/.dylib)
        # write the shared library to the output directory
        set_target_properties(${backend} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
        target_compile_definitions(${backend} PRIVATE GGML_BACKEND_DL)

        add_dependencies(ggml ${backend}) // build order dependency

        if (GGML_BACKEND_DIR)
            install(TARGETS ${backend} LIBRARY DESTINATION ${GGML_BACKEND_DIR})
        else()
            install(TARGETS ${backend} LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR})
        endif()
    else()
        add_library(${backend} ${ARGN})
        target_link_libraries(ggml PUBLIC ${backend})
        install(TARGETS ${backend} LIBRARY)
    endif()
```
`add_dependencies(A B)` tells CMake that target A depends on target B meaning
that whenever we build target A, CMake must first make sure that target B is built.
This is about order of compilation not the final linkage.
This makes sure that if we make a change to a backend and then try to build the
ggml target, CMake will first rebuild the backend before rebuilding ggml and this
keeps things consistent.

If we don't have this it would be possible that after everything is built and
we have libggml-base.so, libggml.so, and libggml-vulkan.so in the build directory.
If we would now make a change to the ggml-vulkan backend and then try to build
using a targeting ggml, cmake would check ggml and determine that it is up to
date and nothing would be rebuilt. So our changes would not be compiled. But
with this build order dependency CMake will first rebuild the ggml-vulkan backend
before rebuilding ggml.

target_link_libraries is what adds a linkage dependency.

### Building
```console
cmake --fresh -S . -B build-backends -DCMAKE_BUILD_TYPE=Debug\
    -DGGML_BACKEND_DL=ON \
    -DGGML_CPU_ALL_VARIANTS=ON \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="89" \
    -DGGML_CPU_AARCH64=OFF \
    -DGGML_CUDA_F16=ON \
    -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_LLGUIDANCE=ON

cmake --build build-backends -j8
```
The CPU backend is always built and also GGML_NATIVE is also on by default. But
when GGML_NATIVE is on the build system will inspect the host CPU and build
an optimized version of the CPU backend for the host CPU. This means that it
might not be runnable on other CPUS that don't match the host CPU features, the
machine we are building on. So this combination is not allowed and will produce
the following error:
```console
- x86 detected
CMake Error at ggml/src/ggml-cpu/CMakeLists.txt:377 (message):
  GGML_NATIVE is not compatible with GGML_BACKEND_DL, consider using
  GGML_CPU_ALL_VARIANTS
Call Stack (most recent call first):
  ggml/src/CMakeLists.txt:430 (ggml_add_cpu_backend_variant_impl)


-- Configuring incomplete, errors occurred!
```
