### ARM i8mm issue

### CI error
The following error happens on CI:
```console
cmake .. -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DBUILD_SHARED_LIBS=ON
  -DCMAKE_INSTALL_PREFIX=/Users/runner/work/ggml/ggml/installed -DGGML_METAL=OFF\
  -- ARM -mcpu not found, -mcpu=native will be used
  -- Performing Test GGML_MACHINE_SUPPORTS_dotprod
  -- Performing Test GGML_MACHINE_SUPPORTS_dotprod - Success
  -- Performing Test GGML_MACHINE_SUPPORTS_i8mm
  -- Performing Test GGML_MACHINE_SUPPORTS_i8mm - Failed
  -- Performing Test GGML_MACHINE_SUPPORTS_noi8mm
  -- Performing Test GGML_MACHINE_SUPPORTS_noi8mm - Success
  -- Performing Test GGML_MACHINE_SUPPORTS_sve
  -- Performing Test GGML_MACHINE_SUPPORTS_sve - Failed
  -- Performing Test GGML_MACHINE_SUPPORTS_nosve
  -- Performing Test GGML_MACHINE_SUPPORTS_nosve - Success
  -- Performing Test GGML_MACHINE_SUPPORTS_sme
  -- Performing Test GGML_MACHINE_SUPPORTS_sme - Failed
  -- Performing Test GGML_MACHINE_SUPPORTS_nosme
  -- Performing Test GGML_MACHINE_SUPPORTS_nosme - Success
  -- ARM feature DOTPROD enabled
  -- ARM feature MATMUL_INT8 enabled
  -- ARM feature FMA enabled
  -- ARM feature FP16_VECTOR_ARITHMETIC enabled
  -- Adding CPU backend variant ggml-cpu: -mcpu=native+dotprod+noi8mm+nosve+nosme \
```
Notice that it fails to detect i8mm support, but it also detects MATMUL_INT8 support.

```console
  FAILED: [code=1] src/CMakeFiles/ggml-cpu.dir/ggml-cpu/arch/arm/quants.c.o
  /usr/bin/clang -DACCELERATE_LAPACK_ILP64 -DACCELERATE_NEW_LAPACK -DGGML_BACKEND_BUILD -DGGML_BACKEND_SHARED
  -DGGML_SCHED_MAX_COPIES=4 -DGGML_SHARED -DGGML_USE_ACCELERATE -DGGML_USE_CPU_REPACK -D_DARWIN_C_SOURCE
  -D_XOPEN_SOURCE=600 -Dggml_cpu_EXPORTS -I/Users/runner/work/ggml/ggml/src/.. -I/Users/runner/work/ggml/ggml/src/.
  -I/Users/runner/work/ggml/ggml/src/ggml-cpu -I/Users/runner/work/ggml/ggml/src/../include -F/Applications/Xcode_16.4.
  app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX15.5.sdk/System/Library/Frameworks -O3 -DNDEBUG
   -std=gnu11 -arch arm64 -isysroot
  /Applications/Xcode_16.4.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX15.5.sdk -fPIC
  -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes -Werror=implicit-int
  -Werror=implicit-function-declaration -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function
  -Wunreachable-code-break -Wunreachable-code-return -Wdouble-promotion -mcpu=native+dotprod+noi8mm+nosve+nosme -MD -MT
   src/CMakeFiles/ggml-cpu.dir/ggml-cpu/arch/arm/quants.c.o -MF
  src/CMakeFiles/ggml-cpu.dir/ggml-cpu/arch/arm/quants.c.o.d -o
  src/CMakeFiles/ggml-cpu.dir/ggml-cpu/arch/arm/quants.c.o -c
  /Users/runner/work/ggml/ggml/src/ggml-cpu/arch/arm/quants.c
  /Users/runner/work/ggml/ggml/src/ggml-cpu/arch/arm/quants.c:217:88: error: always_inline function 'vmmlaq_s32'
  requires target feature 'i8mm', but would be inlined into function 'ggml_vec_dot_q4_0_q8_0' that is compiled without
  support for 'i8mm'
    217 |             sumv0 =
  vmlaq_f32(sumv0,(vcvtq_f32_s32(vmmlaq_s32((vmmlaq_s32((vmmlaq_s32((vmmlaq_s32(vdupq_n_s32(0), l0, r0)),
        |                                                                                        ^
  /Users/runner/work/ggml/ggml/src/ggml-cpu/arch/arm/quants.c:217:76: error: always_inline function 'vmmlaq_s32'
  requires target feature 'i8mm', but would be inlined into function 'ggml_vec_dot_q4_0_q8_0' that is compiled without
  support for 'i8mm'
    217 |             sumv0 =
  vmlaq_f32(sumv0,(vcvtq_f32_s32(vmmlaq_s32((vmmlaq_s32((vmmlaq_s32((vmmlaq_s32(vdupq_n_s32(0), l0, r0)),
        |                                                                            ^
  /Users/runner/work/ggml/ggml/src/ggml-cpu/arch/arm/quants.c:217:64: error: always_inline function 'vmmlaq_s32'
  requires target feature 'i8mm', but would be inlined into function 'ggml_vec_dot_q4_0_q8_0' that is compiled without
  support for 'i8mm'
    217 |             sumv0 =
  vmlaq_f32(sumv0,(vcvtq_f32_s32(vmmlaq_s32((vmmlaq_s32((vmmlaq_s32((vmmlaq_s32(vdupq_n_s32(0), l0, r0)),
        |                                                                ^
  /Users/runner/work/ggml/ggml/src/ggml-cpu/arch/arm/quants.c:217:52: error: always_inline function 'vmmlaq_s32'
  requires target feature 'i8mm', but would be inlined into function 'ggml_vec_dot_q4_0_q8_0' that is compiled without
  support for 'i8mm'
    217 |             sumv0 =
  vmlaq_f32(sumv0,(vcvtq_f32_s32(vmmlaq_s32((vmmlaq_s32((vmmlaq_s32((vmmlaq_s32(vdupq_n_s32(0), l0, r0)),
        |                                                    ^
  4 errors generated.
  [20/107] Building CXX object src/CMakeFiles/ggml-cpu.dir/ggml-cpu/unary-ops.cpp.o
  [21/107] Building CXX object src/CMakeFiles/ggml-cpu.dir/ggml-cpu/arch/arm/repack.cpp.o
  [22/107] Building CXX object src/CMakeFiles/ggml-cpu.dir/ggml-cpu/binary-ops.cpp.o
  [23/107] Building CXX object src/CMakeFiles/ggml-cpu.dir/ggml-cpu/ops.cpp.o
  ninja: build stopped: subcommand failed.
```
Notice tha the `+noi8mm` flag is present in the compile command:
```console
  -Wunreachable-code-break -Wunreachable-code-return -Wdouble-promotion -mcpu=native+dotprod+noi8mm+nosve+nosme -MD -MT
```
And this is added by src/ggml-cpu/CMakeLists.txt:
```cmake
                function(check_arm_feature tag code)
                    set(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
                    set(CMAKE_REQUIRED_FLAGS "${ARM_MCPU_FLAG}+${tag}")
                    check_cxx_source_runs("${code}" GGML_MACHINE_SUPPORTS_${tag})
                    if (GGML_MACHINE_SUPPORTS_${tag})
                        set(ARM_MCPU_FLAG_FIX "${ARM_MCPU_FLAG_FIX}+${tag}" PARENT_SCOPE)
                    else()
                        set(CMAKE_REQUIRED_FLAGS "${ARM_MCPU_FLAG}+no${tag}")
                        check_cxx_source_compiles("int main() { return 0; }" GGML_MACHINE_SUPPORTS_no${tag})
                        if (GGML_MACHINE_SUPPORTS_no${tag})
                            set(ARM_MCPU_FLAG_FIX "${ARM_MCPU_FLAG_FIX}+no${tag}" PARENT_SCOPE)
                        endif()
                    endif()
                    set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})
                endfunction()

                check_arm_feature(i8mm    "#include <arm_neon.h>\nint main() { int8x16_t _a, _b; volatile int32x4_t _s = vmmlaq_s32(_s, _a, _b); return 0; }")
```

If we look at the line where this error is generated from:
```c++
void ggml_vec_dot_q4_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    ...
#if defined(__ARM_FEATURE_MATMUL_INT8)
    ...
            sumv0 = vmlaq_f32(sumv0,(vcvtq_f32_s32(vmmlaq_s32((vmmlaq_s32((vmmlaq_s32((vmmlaq_s32(vdupq_n_s32(0), l0, r0)),
            ...
#endif
```
So somehow the `__ARM_FEATURE_MATMUL_INT8` is defined, but not the `__ARM_FEATURE_I8MM`. This could happen I think
as the check for `GGML_INTERNAL_MATMUL_INT8` is done after the i8mm check above.

```console
                    if (GGML_INTERNAL_MATMUL_INT8)
                        set(ARM_MCPU "armv8.6-a")
                        set(ARCH_TAGS "${ARCH_TAGS}+i8mm")
                        list(APPEND ARCH_DEFINITIONS GGML_USE_MATMUL_INT8)
                    endif()
```
So perhaps this should also check `GGML_MACHINE_SUPPORTS_i8mm` as well:
```cmake

                    if (GGML_INTERNAL_MATMUL_INT8 AND GGML_MACHINE_SUPPORTS_i8mm)
                        set(ARM_MCPU "armv8.6-a")
                        set(ARCH_TAGS "${ARCH_TAGS}+i8mm")
                        list(APPEND ARCH_DEFINITIONS GGML_USE_MATMUL_INT8)
```
