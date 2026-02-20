### CMake lib install issue
Configuring CMake and setting `GGML_LIB_INSTALL_DIR` and `LLAMA_LIB_INSTALL_DIR`
to a location does not cause the libraries to be installed into those locations.
These variables only affect the generated cmake config files, the ones that are
used by cmake for the find_package command for example.

For example, take the following CMake configuration and install commands:
```console
cmake --fresh -S . -B build-install -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_BACKEND_DL=ON \
  -DGGML_CPU_ALL_VARIANTS=ON \
  -DLLAMA_TESTS_INSTALL=OFF \
  -DCMAKE_INSTALL_PREFIX="${PWD}/install/" \
  -DGGML_BACKEND_DIR="${PWD}/install/lib/llama.cpp" \
  -DGGML_LIB_INSTALL_DIR="${PWD}/install/llama.cpp" \
  -DLLAMA_LIB_INSTALL_DIR="${PWD}/install/llama.cpp"

cmake --build build-install --parallel 12
cmake --install build-install
```
The install directory will now look like this:
```console
install/
bin include lib
```
lib will have a cmake directory, and a llama.cpp directory. And notice that the
ggml and llama libraries (apart from the backend shared libraries) are in this
directory:
```console
$ ls install/lib/
cmake              libggml-base.so.0.9.7  libggml.so.0.9.7  libllama.so.0.0.8115  libmtmd.so.0.0.8115
libggml-base.so    libggml.so             libllama.so       libmtmd.so            llama.cpp
libggml-base.so.0  libggml.so.0           libllama.so.0     libmtmd.so.0          pkgconfig
```

If we create a simple project and use the following CMakeLists.txt:
```cmake
cmake_minimum_required(VERSION 3.14)
project(test-find-ggml)

# Point to the install directory
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../install")

find_package(ggml REQUIRED)

add_executable(test-ggml test.cpp)
target_link_libraries(test-ggml PRIVATE ggml::ggml)
```
Configuring this will fail with the following error:
```console
$ cmake -S . -B build
CMake Error at /home/danbev/work/ai/llama.cpp/install/lib/cmake/ggml/ggml-config.cmake:11 (message):
  File or directory /home/danbev/work/ai/llama.cpp/install/llama.cpp
  referenced by variable GGML_LIB_DIR does not exist !
Call Stack (most recent call first):
  /home/danbev/work/ai/llama.cpp/install/lib/cmake/ggml/ggml-config.cmake:259 (set_and_check)
  CMakeLists.txt:7 (find_package)


-- Configuring incomplete, errors occurred!
```
And if we inspect ggml-config.cmake we can find:
```console
$ grep -n GGML_LIB_DIR ../install/lib/cmake/ggml/ggml-config.cmake
259:set_and_check(GGML_LIB_DIR "${PACKAGE_PREFIX_DIR}/llama.cpp")
```
And PACKAGE_PREFIX_DIR is what CMAKE_INSTALL_PREFIX was set to which was install
in this case. So currently this setting is broken if we want to set these
variables.

If we look at how ggml and llama handle the install commands
```cmake
install(TARGETS ggml LIBRARY PUBLIC_HEADER)
install(TARGETS ggml-base LIBRARY)
```
There is not DESTINATION set for the library so it will be installed to the
default location which is CMAKE_INSTALL_PREFIX/lib.

If we want to install it to a custom location we need to set the DESTINATION for
the library and the public header:
```cmake
install(TARGETS ggml
    LIBRARY DESTINATION ${GGML_LIB_INSTALL_DIR}
    PUBLIC_HEADER DESTINATION ${GGML_INCLUDE_INSTALL_DIR})
install(TARGETS ggml-base LIBRARY DESTINATION ${GGML_LIB_INSTALL_DIR})
```
And likewise for llama.cpp:
```cmake
install(TARGETS llama
    LIBRARY DESTINATION ${LLAMA_LIB_INSTALL_DIR}
    PUBLIC_HEADER DESTINATION ${LLAMA_INCLUDE_INSTALL_DIR})
```
With these changes the above example cmake project was able to configure without
any problem. But it was still not able to find the ggml headers:
```console
$ cmake --build build
[ 50%] Building CXX object CMakeFiles/test-ggml.dir/test.cpp.o
/home/danbev/work/ai/llama.cpp/test-find-package/test.cpp:1:10: fatal error: ggml.h: No such file or directory
    1 | #include <ggml.h>
      |          ^~~~~~~~
compilation terminated.
gmake[2]: *** [CMakeFiles/test-ggml.dir/build.make:76: CMakeFiles/test-ggml.dir/test.cpp.o] Error 1
gmake[1]: *** [CMakeFiles/Makefile2:83: CMakeFiles/test-ggml.dir/all] Error 2
gmake: *** [Makefile:91: all] Error 2
```
If we look in ggml/cmake/ggml-config.cmake.in we can find:
```cmake
    set_target_properties(ggml::ggml
        PROPERTIES
            IMPORTED_LOCATION "${GGML_LIBRARY}")
```
The above is not setting the include directories for the target, so we need to add:
```cmake
    set_target_properties(ggml::ggml
        PROPERTIES
            IMPORTED_LOCATION "${GGML_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${GGML_INCLUDE_DIR}")
```
The `INTERFACE_INCLUDE_DIRECTORIES` property is needed because when a downstream
project does `target_link_libraries(some_prog PRIVATE ggml::ggml)`, CMake needs
to know not only where the library file is but also where the headers are so it
can add `-I<include-path>` to compile commands. This is what was missing and
the reason for the compilation error.

