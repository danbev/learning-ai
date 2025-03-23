## GGML_NATIVE
By default this option is true and this will quiery the current cpu for available
features and enable them. This allows for an optimized build for the current cpu.
But if you want to build and distribute a binary this may not work unles you
know that the target cpu has the same features as the build machine.

Enabling `GGML_NATIVE` will add `-march=native` to the compiler flags. 
`-march=native` will cause the compiler to query/probe the current cpu for
features that it supports, using something like `cpuid` on x86. Depening on what
is supported the compiler will add flags to enable those features.



### Setting GGML_NATIVE issue
I was going to try to set GGML_NATIVE=OFF but it seems that it is not possible
to override this but just setting like this:
```console
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DGGML_NATIVE=OFF
cmake --build build
```

If we inspect `ggml/CMakeLists.txt` we have the following with some debug
statements added:
```console
  message(STATUS "-------------> GGML_NATIVE before: ${GGML_NATIVE}")             
  message(STATUS "-------------> GGML_NATIVE_DEFAULT before: ${GGML_NATIVE_DEFAULT}")
  if (CMAKE_CROSSCOMPILING OR DEFINED ENV{SOURCE_DATE_EPOCH})                     
      set(GGML_NATIVE_DEFAULT OFF)                                                
  else()                                                                          
      set(GGML_NATIVE_DEFAULT ON)                                                 
  endif()                                                                         
  message(STATUS "-------------> GGML_NATIVE after: ${GGML_NATIVE}")              
  message(STATUS "-------------> GGML_NATIVE_DEFAULT after: ${GGML_NATIVE_DEFAULT}")
```
This produces the following output:
```console
-- -------------> GGML_NATIVE before: OFF
-- -------------> GGML_NATIVE_DEFAULT before: 
-- -------------> GGML_NATIVE after: OFF
-- -------------> GGML_NATIVE_DEFAULT after: ON
```
Then further down we have:
```console
 option(GGML_NATIVE "ggml: optimize the build for the current system" ${GGML_NATIVE_DEFAULT})
```

This is making it difficult, unless we set the environment variable 
`SOURCE_DATE_EPOCH`:
```console
env SOURCE_DATE_EPOCH=1234567890 cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug \
    -DGGML_NATIVE=OFF
cmake --build build

$ ./build-native.sh
-- -------------> GGML_NATIVE before: OFF
-- -------------> GGML_NATIVE_DEFAULT before:
-- -------------> GGML_NATIVE after: OFF
-- -------------> GGML_NATIVE_DEFAULT after: OFF
-- -------------> INS_ENB: OFF
-- ccache found, compilation results will be cached. Disable with GGML_CCACHE=OFF.
-- CMAKE_SYSTEM_PROCESSOR: x86_64
-- Including CPU backend
-- x86 detected
-- Adding CPU backend variant ggml-cpu: -msse4.2;-mf16c;-mfma;-mbmi2 GGML_SSE42;GGML_F16C;GGML_FMA;GGML_BMI2
-- Configuring done (0.1s)
-- Generating done (0.0s)
-- Build files have been written to: /home/danbev/work/ai/whisper-work/build
```
`SOURCE_DATE_EPOCH` is used to set the build time to a fixed value, so that the
build is reproducible. It's used to make builds deterministic by setting a fixed
timestamp for all file creation and modification times during the build process.

And this will allow us to explicitly enable native features using GGML options:
```console
$ rm -rf build
env SOURCE_DATE_EPOCH=1234567890 cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug \
    -DGGML_NATIVE=OFF \
    -DGGML_AVX=ON
cmake --build build
-- The C compiler identification is GNU 13.3.0
-- The CXX compiler identification is GNU 13.3.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Found Git: /usr/bin/git (found version "2.43.0") 
-- -------------> GGML_NATIVE before: OFF
-- -------------> GGML_NATIVE_DEFAULT before: 
-- -------------> GGML_NATIVE after: OFF
-- -------------> GGML_NATIVE_DEFAULT after: OFF
-- -------------> INS_ENB: OFF
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Success
-- Found Threads: TRUE  
-- ccache found, compilation results will be cached. Disable with GGML_CCACHE=OFF.
-- CMAKE_SYSTEM_PROCESSOR: x86_64
-- Including CPU backend
-- Found OpenMP_C: -fopenmp (found version "4.5") 
-- Found OpenMP_CXX: -fopenmp (found version "4.5") 
-- Found OpenMP: TRUE (found version "4.5")  
-- x86 detected
-- Adding CPU backend variant ggml-cpu: -msse4.2;-mavx GGML_SSE42;GGML_AVX
-- Configuring done (0.7s)
-- Generating done (0.0s)
-- Build files have been written to: /home/danbev/work/ai/whisper-work/build
```

Unless I'm missing something I was expecting that if I set `GGML_NATIVE=OFF` it
would disable native features. This is not the case at the moment and I'd like
to ask if this is on purpose or if it's a bug?
Adding the following to `ggml/CMakeLists.txt` would make it possible to disable:
```console
if (CMAKE_CROSSCOMPILING OR DEFINED ENV{SOURCE_DATE_EPOCH} OR (DEFINED GGML_NATIVE AND GGML_NATIVE STREQUAL "OFF"))
    set(GGML_NATIVE_DEFAULT OFF)
else()
    set(GGML_NATIVE_DEFAULT ON)
endif()
```
