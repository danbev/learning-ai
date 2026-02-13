## Kleidiai install issue notes
When building with 
```console
cmake --fresh -S . -B build-install -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CPU_KLEIDIAI=ON \
  -DGGML_OPENMP=OFF \
  -DGGML_LLAMAFILE=OFF \
  -DGGML_CCACHE=OFF \
  -DCMAKE_INSTALL_PREFIX="${PWD}/install"
cmake --build build-install --parallel 12
cmake --install build-install
```
The following error is generated:
```console
[100%] Built target llama-server
-- Install configuration: "Release"
-- Installing: /Users/danbev/work/ai/llama.cpp/install/lib/libggml-cpu.0.9.5.dylib
-- Installing: /Users/danbev/work/ai/llama.cpp/install/lib/libggml-cpu.0.dylib
-- Installing: /Users/danbev/work/ai/llama.cpp/install/lib/libggml-cpu.dylib
CMake Error at build-install/_deps/kleidiai_download-build/cmake_install.cmake:41 (file):
  file INSTALL cannot find
  "/Users/danbev/work/ai/llama.cpp/build-install/bin/libkleidiai.dylib": No
  such file or directory.
Call Stack (most recent call first):
  build-install/ggml/src/cmake_install.cmake:72 (include)
  build-install/ggml/cmake_install.cmake:42 (include)
  build-install/cmake_install.cmake:42 (include)
```
In ggml/src/ggml-cpu/CMakeLists.txt we have:
```cmake
    if (GGML_CPU_KLEIDIAI)
        message(STATUS "Using KleidiAI optimized kernels if applicable")

    ...

        FetchContent_Declare(KleidiAI_Download
            URL ${KLEIDIAI_DOWNLOAD_URL}
            DOWNLOAD_EXTRACT_TIMESTAMP NEW
            URL_HASH MD5=${KLEIDIAI_ARCHIVE_MD5})

        FetchContent_MakeAvailable(KleidiAI_Download)
```
The call FetchContent_MakeAvailable will call add_subdirectory on the kleidiai
download directory which will register the install target. But we don't actually
build using KleidiAIs build system, instead we include the directories manuall:
```cmake
        include_directories(
            ${KLEIDIAI_SRC}/
            ${KLEIDIAI_SRC}/kai/
            ...
```
And individual source files are added and then addes to the ggml cpu sources:
```cmake
        list(APPEND GGML_KLEIDIAI_SOURCES
            ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32.c
            ...

        list(APPEND GGML_CPU_SOURCES ${GGML_KLEIDIAI_SOURCES})
```
```cmake
    target_sources(${GGML_CPU_NAME} PRIVATE ${GGML_CPU_SOURCES})
```
There is also a removal of the  kleidiai target after making it available since
we don't really use the target:
```cmake
        # Remove kleidiai target after fetching it
        if (TARGET kleidiai)
            set_target_properties(kleidiai PROPERTIES EXCLUDE_FROM_ALL TRUE)
        endif()
```
But we have already registered the install target at this point and this is why
the error occurs.

We could use FetchContent_Populate(KleidiAI_Download) to download and extract
based on the URL/hash from FetchContentDeclare but with the difference that this
does not call add_subdirectory and thus does not register the install target.
