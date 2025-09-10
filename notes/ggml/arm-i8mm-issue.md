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

### Troubleshooting locally
Notice that the `+noi8mm` flag is present in the compile command:
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

But we also have the following that detects features:
```cmake
            execute_process(
                COMMAND ${CMAKE_C_COMPILER} ${ARCH_FLAGS} -dM -E -
                INPUT_FILE ${FEAT_INPUT_FILE}
                OUTPUT_VARIABLE ARM_FEATURE
                RESULT_VARIABLE ARM_FEATURE_RESULT
            )
            message(STATUS "ARM_FEATURE: ${ARM_FEATURE}")
            if (ARM_FEATURE_RESULT)
                message(WARNING "Failed to get ARM features")
            else()
                foreach(feature DOTPROD SVE MATMUL_INT8 FMA FP16_VECTOR_ARITHMETIC SME)
                    string(FIND "${ARM_FEATURE}" "__ARM_FEATURE_${feature} 1" feature_pos)
                    if (NOT ${feature_pos} EQUAL -1)
                        message(STATUS "ARM feature ${feature} enabled")
                    endif()
                endforeach()
            endif()
        endif()
```
The command here is printing the macro definitions (`-dM`) and only running the preprocessor (`-E`):
```console
$ echo "" | /usr/bin/clang -mcpu=native+dotprod+noi8mm+nosve+nosme -dM -E -
#define TARGET_IPHONE_SIMULATOR 0
#define TARGET_OS_ARROW 1
#define TARGET_OS_BRIDGE 0
#define TARGET_OS_DRIVERKIT 0
#define TARGET_OS_EMBEDDED 0
#define TARGET_OS_IOS 0
#define TARGET_OS_IOSMAC 0
#define TARGET_OS_IPHONE 0
#define TARGET_OS_LINUX 0
#define TARGET_OS_MAC 1
#define TARGET_OS_MACCATALYST 0
#define TARGET_OS_NANO 0
#define TARGET_OS_OSX 1
#define TARGET_OS_SIMULATOR 0
#define TARGET_OS_TV 0
#define TARGET_OS_UIKITFORMAC 0
#define TARGET_OS_UNIX 0
#define TARGET_OS_VISION 0
#define TARGET_OS_WATCH 0
#define TARGET_OS_WIN32 0
#define TARGET_OS_WINDOWS 0
#define TARGET_OS_XR 0
#define _LP64 1
#define __AARCH64EL__ 1
#define __AARCH64_CMODEL_SMALL__ 1
#define __AARCH64_SIMD__ 1
#define __APPLE_CC__ 6000
#define __APPLE__ 1
#define __ARM64_ARCH_8__ 1
#define __ARM_64BIT_STATE 1
#define __ARM_ACLE 200
#define __ARM_ALIGN_MAX_STACK_PWR 4
#define __ARM_ARCH 8
#define __ARM_ARCH_8_3__ 1
#define __ARM_ARCH_8_4__ 1
#define __ARM_ARCH_8_5__ 1
#define __ARM_ARCH_8_6__ 1
#define __ARM_ARCH_ISA_A64 1
#define __ARM_ARCH_PROFILE 'A'
#define __ARM_BF16_FORMAT_ALTERNATIVE 1
#define __ARM_FEATURE_AES 1
#define __ARM_FEATURE_ATOMICS 1
#define __ARM_FEATURE_BF16 1
#define __ARM_FEATURE_BF16_SCALAR_ARITHMETIC 1
#define __ARM_FEATURE_BF16_VECTOR_ARITHMETIC 1
#define __ARM_FEATURE_BTI 1
#define __ARM_FEATURE_CLZ 1
#define __ARM_FEATURE_COMPLEX 1
#define __ARM_FEATURE_CRC32 1
#define __ARM_FEATURE_CRYPTO 1
#define __ARM_FEATURE_DIRECTED_ROUNDING 1
#define __ARM_FEATURE_DIV 1
#define __ARM_FEATURE_DOTPROD 1
#define __ARM_FEATURE_FMA 1
#define __ARM_FEATURE_FP16_FML 1
#define __ARM_FEATURE_FP16_SCALAR_ARITHMETIC 1
#define __ARM_FEATURE_FP16_VECTOR_ARITHMETIC 1
#define __ARM_FEATURE_FRINT 1
#define __ARM_FEATURE_IDIV 1
#define __ARM_FEATURE_JCVT 1
#define __ARM_FEATURE_LDREX 0xF
#define __ARM_FEATURE_MATMUL_INT8 1       <------------------- 
#define __ARM_FEATURE_NUMERIC_MAXMIN 1
#define __ARM_FEATURE_PAUTH 1
#define __ARM_FEATURE_QRDMX 1
#define __ARM_FEATURE_RCPC 1
#define __ARM_FEATURE_SHA2 1
#define __ARM_FEATURE_SHA3 1
#define __ARM_FEATURE_SHA512 1
#define __ARM_FEATURE_UNALIGNED 1
...
```
And this is using the proper `-mcpu=native+dotprod+noi8mm+nosve+nosme` flag but the feature is
still in the returned list of supported features, it has a one.
My understanding is that MATMUL_INT8 does depend on the i8mm architectural feature.

Lets chech the clang verbose output:
```console
$ /usr/bin/clang -### -mcpu=native+dotprod+noi8mm+nosve+nosme -c -x c /dev/null
Apple clang version 17.0.0 (clang-1700.0.13.5)
Target: arm64-apple-darwin24.6.0
Thread model: posix
InstalledDir: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin
 "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang" "-cc1" "-triple" "arm64-apple-macosx15.0.0" "-Wundef-prefix=TARGET_OS_" "-Wdeprecated-objc-isa-usage" "-Werror=deprecated-objc-isa-usage" "-Werror=implicit-function-declaration" "-emit-obj" "-disable-free" "-clear-ast-before-backend" "-disable-llvm-verifier" "-discard-value-names" "-main-file-name" "null" "-mrelocation-model" "pic" "-pic-level" "2" "-mframe-pointer=non-leaf" "-fno-strict-return" "-ffp-contract=on" "-fno-rounding-math" "-funwind-tables=1" "-fobjc-msgsend-selector-stubs" "-target-sdk-version=15.5" "-fvisibility-inlines-hidden-static-local-var" "-fdefine-target-os-macros" "-fno-assume-unique-vtables" "-fno-modulemap-allow-subdirectory-search" "-target-cpu" "apple-m3" "-target-feature" "+zcm" "-target-feature" "+zcz" "-target-feature" "+v8.6a" "-target-feature" "+aes" "-target-feature" "+bf16" "-target-feature" "+complxnum" "-target-feature" "+crc" "-target-feature" "+dotprod" "-target-feature" "+fp-armv8" "-target-feature" "+fp16fml" "-target-feature" "+fpac" "-target-feature" "+fullfp16" "-target-feature" "+hcx" "-target-feature" "-i8mm" "-target-feature" "+jsconv" "-target-feature" "+lse" "-target-feature" "+neon" "-target-feature" "+pauth" "-target-feature" "+perfmon" "-target-feature" "+ras" "-target-feature" "+rcpc" "-target-feature" "+rdm" "-target-feature" "+sha2" "-target-feature" "+sha3" "-target-feature" "+ssbs" "-target-abi" "darwinpcs" "-debugger-tuning=lldb" "-fdebug-compilation-dir=/Users/danbev/work/ai/ggml" "-target-linker-version" "1167.5" "-fcoverage-compilation-dir=/Users/danbev/work/ai/ggml" "-resource-dir" "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/17" "-isysroot" "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk" "-I/usr/local/include" "-internal-isystem" "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/local/include" "-internal-isystem" "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/17/include" "-internal-externc-isystem" "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include" "-internal-externc-isystem" "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include" "-Wno-reorder-init-list" "-Wno-implicit-int-float-conversion" "-Wno-c99-designator" "-Wno-final-dtor-non-final-class" "-Wno-extra-semi-stmt" "-Wno-misleading-indentation" "-Wno-quoted-include-in-framework-header" "-Wno-implicit-fallthrough" "-Wno-enum-enum-conversion" "-Wno-enum-float-conversion" "-Wno-elaborated-enum-base" "-Wno-reserved-identifier" "-Wno-gnu-folding-constant" "-ferror-limit" "19" "-stack-protector" "1" "-fstack-check" "-mdarwin-stkchk-strong-link" "-fblocks" "-fencode-extended-block-signature" "-fregister-global-dtors-with-atexit" "-fgnuc-version=4.2.1" "-fskip-odr-check-in-gmf" "-fmax-type-align=16" "-fcommon" "-fcolor-diagnostics" "-clang-vendor-feature=+disableNonDependentMemberExprInCurrentInstantiation" "-fno-odr-hash-protocols" "-clang-vendor-feature=+enableAggressiveVLAFolding" "-clang-vendor-feature=+revert09abecef7bbf" "-clang-vendor-feature=+thisNoAlignAttr" "-clang-vendor-feature=+thisNoNullAttr" "-clang-vendor-feature=+disableAtImportPrivateFrameworkInImplementationError" "-D__GCC_HAVE_DWARF2_CFI_ASM=1" "-o" "null.o" "-x" "c" "/dev/null"
 ```
This has `-i8mm` disabled.

So in cmake it can't detect i8mm support but it is still enabled the MATMUL_INT8 feature which is causing the
compilation failure.
 
### Troubleshooting on CI
So the above was running on my local machine and forcing i8mm to be disabled.
I've added a debug step to the CI server:
```console
    - name: Debug
      run: |
        /usr/bin/clang -### -mcpu=native+dotprod+noi8mm+nosve+nosme -c -x c /dev/null
        sysctl -a | grep hw.optional.arm.
```

To see what it actually supports:
```console
/usr/bin/clang -### -mcpu=native+dotprod+noi8mm+nosve+nosme -c -x c /dev/null
  sysctl -a | grep hw.optional.arm.
  shell: /bin/bash -e {0}
Apple clang version 17.0.0 (clang-1700.0.13.5)
Target: arm64-apple-darwin24.5.0
Thread model: posix
InstalledDir: /Applications/Xcode_16.4.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin
 "/Applications/Xcode_16.4.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang" "-cc1" "-triple" "arm64-apple-macosx15.0.0" "-Wundef-prefix=TARGET_OS_" "-Wdeprecated-objc-isa-usage" "-Werror=deprecated-objc-isa-usage" "-Werror=implicit-function-declaration" "-emit-obj" "-disable-free" "-clear-ast-before-backend" "-disable-llvm-verifier" "-discard-value-names" "-main-file-name" "null" "-mrelocation-model" "pic" "-pic-level" "2" "-mframe-pointer=non-leaf" "-fno-strict-return" "-ffp-contract=on" "-fno-rounding-math" "-funwind-tables=1" "-fobjc-msgsend-selector-stubs" "-target-sdk-version=15.5" "-fvisibility-inlines-hidden-static-local-var" "-fdefine-target-os-macros" "-fno-assume-unique-vtables" "-fno-modulemap-allow-subdirectory-search" "-target-cpu" "apple-m3" "-target-feature" "+zcm" "-target-feature" "+zcz" "-target-feature" "+v8.6a" "-target-feature" "+aes" "-target-feature" "+bf16" "-target-feature" "+complxnum" "-target-feature" "+crc" "-target-feature" "+dotprod" "-target-feature" "+

hw.optional.arm.FEAT_CRC32: 1
hw.optional.arm.FEAT_FlagM: 1
hw.optional.arm.FEAT_FlagM2: 1
hw.optional.arm.FEAT_FHM: 1
hw.optional.arm.FEAT_DotProd: 1
hw.optional.arm.FEAT_SHA3: 1
hw.optional.arm.FEAT_RDM: 1
hw.optional.arm.FEAT_LSE: 1
hw.optional.arm.FEAT_SHA256: 1
hw.optional.arm.FEAT_SHA512: 1
hw.optional.arm.FEAT_SHA1: 1
hw.optional.arm.FEAT_AES: 1
hw.optional.arm.FEAT_PMULL: 1
hw.optional.arm.FEAT_SPECRES: 0
hw.optional.arm.FEAT_SPECRES2: 0
hw.optional.arm.FEAT_SB: 1
hw.optional.arm.FEAT_FRINTTS: 1
hw.optional.arm.FEAT_PACIMP: 1
hw.optional.arm.FEAT_LRCPC: 1
hw.optional.arm.FEAT_LRCPC2: 1
hw.optional.arm.FEAT_FCMA: 1
hw.optional.arm.FEAT_JSCVT: 1
hw.optional.arm.FEAT_PAuth: 1
hw.optional.arm.FEAT_PAuth2: 0
hw.optional.arm.FEAT_FPAC: 0
hw.optional.arm.FEAT_FPACCOMBINE: 0
hw.optional.arm.FEAT_DPB: 1
hw.optional.arm.FEAT_DPB2: 1
hw.optional.arm.FEAT_BF16: 0
hw.optional.arm.FEAT_EBF16: 0
hw.optional.arm.FEAT_I8MM: 0           <-------------------
hw.optional.arm.FEAT_WFxT: 0
hw.optional.arm.FEAT_RPRES: 0
hw.optional.arm.FEAT_CSSC: 0
hw.optional.arm.FEAT_HBC: 0
hw.optional.arm.FEAT_ECV: 0
hw.optional.arm.FEAT_AFP: 0
hw.optional.arm.FEAT_LSE2: 1
hw.optional.arm.FEAT_CSV2: 1
hw.optional.arm.FEAT_CSV3: 1
hw.optional.arm.FEAT_DIT: 1
hw.optional.arm.AdvSIMD: 1
hw.optional.arm.AdvSIMD_HPFPCvt: 1
hw.optional.arm.FEAT_FP16: 1
hw.optional.arm.FEAT_SSBS: 0
hw.optional.arm.FEAT_BTI: 0
hw.optional.arm.FEAT_SME: 0
hw.optional.arm.FEAT_SME2: 0
hw.optional.arm.SME_F32F32: 0
hw.optional.arm.SME_BI32I32: 0
hw.optional.arm.SME_B16F32: 0
hw.optional.arm.SME_F16F32: 0
hw.optional.arm.SME_I8I32: 0
hw.optional.arm.SME_I16I32: 0
hw.optional.arm.FEAT_SME_F64F64: 0
hw.optional.arm.FEAT_SME_I16I64: 0
hw.optional.arm.FP_SyncExceptions: 1
hw.optional.arm.caps: 292171059125284863
hw.optional.arm.sme_max_svl_b: 0
hw.optional.armv8_crc32: 1
hw.optional.armv8_gpi: 1
hw.optional.armv8_1_atomics: 1
hw.optional.armv8_2_fhm: 1
hw.optional.armv8_2_sha512: 1
hw.optional.armv8_2_sha3: 1
hw.optional.armv8_3_compnum: 1
hw.optional.arm64: 1
```
And the following is the output of `ARM_FEATURE`:
```console
#define TARGET_OS_IOS 0
#define TARGET_OS_IOSMAC 0
#define TARGET_OS_IPHONE 0
#define TARGET_OS_LINUX 0
#define TARGET_OS_MAC 1
#define TARGET_OS_MACCATALYST 0
#define TARGET_OS_NANO 0
#define TARGET_OS_OSX 1
#define TARGET_OS_SIMULATOR 0
#define TARGET_OS_TV 0
#define TARGET_OS_UIKITFORMAC 0
#define TARGET_OS_UNIX 0
#define TARGET_OS_VISION 0
#define TARGET_OS_WATCH 0
#define TARGET_OS_WIN32 0
#define TARGET_OS_WINDOWS 0
#define TARGET_OS_XR 0
#define _LP64 1
#define __AARCH64EL__ 1
#define __AARCH64_CMODEL_SMALL__ 1
#define __AARCH64_SIMD__ 1
#define __APPLE_CC__ 6000
#define __APPLE__ 1
#define __ARM64_ARCH_8__ 1
#define __ARM_64BIT_STATE 1
#define __ARM_ACLE 200
#define __ARM_ALIGN_MAX_STACK_PWR 4
#define __ARM_ARCH 8
#define __ARM_ARCH_8_3__ 1
#define __ARM_ARCH_8_4__ 1
#define __ARM_ARCH_8_5__ 1
#define __ARM_ARCH_8_6__ 1
#define __ARM_ARCH_ISA_A64 1
#define __ARM_ARCH_PROFILE 'A'
#define __ARM_BF16_FORMAT_ALTERNATIVE 1

#define __ARM_FEATURE_AES 1
#define __ARM_FEATURE_ATOMICS 1
#define __ARM_FEATURE_BF16 1
#define __ARM_FEATURE_BF16_SCALAR_ARITHMETIC 1
#define __ARM_FEATURE_BF16_VECTOR_ARITHMETIC 1
#define __ARM_FEATURE_BTI 1
#define __ARM_FEATURE_CLZ 1
#define __ARM_FEATURE_COMPLEX 1
#define __ARM_FEATURE_CRC32 1
#define __ARM_FEATURE_CRYPTO 1
#define __ARM_FEATURE_DIRECTED_ROUNDING 1
#define __ARM_FEATURE_DIV 1
#define __ARM_FEATURE_DOTPROD 1
#define __ARM_FEATURE_FMA 1
#define __ARM_FEATURE_FP16_FML 1
#define __ARM_FEATURE_FP16_SCALAR_ARITHMETIC 1
#define __ARM_FEATURE_FP16_VECTOR_ARITHMETIC 1
#define __ARM_FEATURE_FRINT 1
#define __ARM_FEATURE_IDIV 1
#define __ARM_FEATURE_JCVT 1
#define __ARM_FEATURE_LDREX 0xF
#define __ARM_FEATURE_MATMUL_INT8 1         <------------
#define __ARM_FEATURE_NUMERIC_MAXMIN 1
#define __ARM_FEATURE_PAUTH 1
#define __ARM_FEATURE_QRDMX 1
#define __ARM_FEATURE_RCPC 1
#define __ARM_FEATURE_SHA2 1
#define __ARM_FEATURE_SHA3 1
#define __ARM_FEATURE_SHA512 1
#define __ARM_FEATURE_UNALIGNED 1

#define __ARM_FP 0xE
#define __ARM_FP16_ARGS 1
#define __ARM_FP16_FORMAT_IEEE 1
#define __ARM_NEON 1
#define __ARM_NEON_FP 0xE
#define __ARM_NEON__ 1
#define __ARM_PCS_AAPCS64 1
#define __ARM_SIZEOF_MINIMAL_ENUM 4
#define __ARM_SIZEOF_WCHAR_T 4
#define __ARM_STATE_ZA 1
#define __ARM_STATE_ZT0 1
#define __ATOMIC_ACQUIRE 2
#define __ATOMIC_ACQ_REL 4
#define __ATOMIC_CONSUME 1
#define __ATOMIC_RELAXED 0
#define __ATOMIC_RELEASE 3
#define __ATOMIC_SEQ_CST 5
#define __BIGGEST_ALIGNMENT__ 8
#define __BITINT_MAXWIDTH__ 128
#define __BLOCKS__ 1
#define __BOOL_WIDTH__ 8
#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
#define __CHAR16_TYPE__ unsigned short
#define __CHAR32_TYPE__ unsigned int
#define __CHAR_BIT__ 8
#define __CLANG_ATOMIC_BOOL_LOCK_FREE 2
#define __CLANG_ATOMIC_CHAR16_T_LOCK_FREE 2
#define __CLANG_ATOMIC_CHAR32_T_LOCK_FREE 2
#define __CLANG_ATOMIC_CHAR_LOCK_FREE 2
#define __CLANG_ATOMIC_INT_LOCK_FREE 2
#define __CLANG_ATOMIC_LLONG_LOCK_FREE 2
#define __CLANG_ATOMIC_LONG_LOCK_FREE 2
#define __CLANG_ATOMIC_POINTER_LOCK_FREE 2
#define __CLANG_ATOMIC_SHORT_LOCK_FREE 2
#define __CLANG_ATOMIC_WCHAR_T_LOCK_FREE 2
#define __CONSTANT_CFSTRINGS__ 1
#define __DBL_DECIMAL_DIG__ 17
#define __DBL_DENORM_MIN__ 4.9406564584124654e-324
#define __DBL_DIG__ 15
#define __DBL_EPSILON__ 2.2204460492503131e-16
#define __DBL_HAS_DENORM__ 1
#define __DBL_HAS_INFINITY__ 1
#define __DBL_HAS_QUIET_NAN__ 1
#define __DBL_MANT_DIG__ 53
#define __DBL_MAX_10_EXP__ 308
#define __DBL_MAX_EXP__ 1024
#define __DBL_MAX__ 1.7976931348623157e+308
#define __DBL_MIN_10_EXP__ (-307)
#define __DBL_MIN_EXP__ (-1021)
#define __DBL_MIN__ 2.2250738585072014e-308
#define __DBL_NORM_MAX__ 1.7976931348623157e+308
#define __DECIMAL_DIG__ __LDBL_DECIMAL_DIG__
#define __DYNAMIC__ 1
#define __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ 150000
#define __ENVIRONMENT_OS_VERSION_MIN_REQUIRED__ 150000
#define __FINITE_MATH_ONLY__ 0
#define __FLT16_DECIMAL_DIG__ 5
#define __FLT16_DENORM_MIN__ 5.9604644775390625e-8F16
#define __FLT16_DIG__ 3
#define __FLT16_EPSILON__ 9.765625e-4F16
#define __FLT16_HAS_DENORM__ 1
#define __FLT16_HAS_INFINITY__ 1
#define __FLT16_HAS_QUIET_NAN__ 1
#define __FLT16_MANT_DIG__ 11
#define __FLT16_MAX_10_EXP__ 4
#define __FLT16_MAX_EXP__ 16
#define __FLT16_MAX__ 6.5504e+4F16
#define __FLT16_MIN_10_EXP__ (-4)
#define __FLT16_MIN_EXP__ (-13)
#define __FLT16_MIN__ 6.103515625e-5F16
#define __FLT16_NORM_MAX__ 6.5504e+4F16
#define __FLT_DECIMAL_DIG__ 9
#define __FLT_DENORM_MIN__ 1.40129846e-45F
#define __FLT_DIG__ 6
#define __FLT_EPSILON__ 1.19209290e-7F
#define __FLT_HAS_DENORM__ 1
#define __FLT_HAS_INFINITY__ 1
#define __FLT_HAS_QUIET_NAN__ 1
#define __FLT_MANT_DIG__ 24
#define __FLT_MAX_10_EXP__ 38
#define __FLT_MAX_EXP__ 128
#define __FLT_MAX__ 3.40282347e+38F
#define __FLT_MIN_10_EXP__ (-37)
#define __FLT_MIN_EXP__ (-125)
#define __FLT_MIN__ 1.17549435e-38F
#define __FLT_NORM_MAX__ 3.40282347e+38F
#define __FLT_RADIX__ 2
#define __FPCLASS_NEGINF 0x0004
#define __FPCLASS_NEGNORMAL 0x0008
#define __FPCLASS_NEGSUBNORMAL 0x0010
#define __FPCLASS_NEGZERO 0x0020
#define __FPCLASS_POSINF 0x0200
#define __FPCLASS_POSNORMAL 0x0100
#define __FPCLASS_POSSUBNORMAL 0x0080
#define __FPCLASS_POSZERO 0x0040
#define __FPCLASS_QNAN 0x0002
#define __FPCLASS_SNAN 0x0001
#define __FP_FAST_FMA 1
#define __FP_FAST_FMAF 1
#define __GCC_ASM_FLAG_OUTPUTS__ 1
#define __GCC_ATOMIC_BOOL_LOCK_FREE 2
#define __GCC_ATOMIC_CHAR16_T_LOCK_FREE 2
#define __GCC_ATOMIC_CHAR32_T_LOCK_FREE 2
#define __GCC_ATOMIC_CHAR_LOCK_FREE 2
#define __GCC_ATOMIC_INT_LOCK_FREE 2
#define __GCC_ATOMIC_LLONG_LOCK_FREE 2
#define __GCC_ATOMIC_LONG_LOCK_FREE 2
#define __GCC_ATOMIC_POINTER_LOCK_FREE 2
#define __GCC_ATOMIC_SHORT_LOCK_FREE 2
#define __GCC_ATOMIC_TEST_AND_SET_TRUEVAL 1
#define __GCC_ATOMIC_WCHAR_T_LOCK_FREE 2
#define __GCC_CONSTRUCTIVE_SIZE 64
#define __GCC_DESTRUCTIVE_SIZE 64
#define __GCC_HAVE_DWARF2_CFI_ASM 1
#define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_1 1
#define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_16 1
#define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_2 1
#define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_4 1
#define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_8 1
#define __GNUC_MINOR__ 2
#define __GNUC_PATCHLEVEL__ 1
#define __GNUC_STDC_INLINE__ 1
#define __GNUC__ 4
#define __GXX_ABI_VERSION 1002
#define __HAVE_FUNCTION_MULTI_VERSIONING 1
#define __INT16_C_SUFFIX__
#define __INT16_FMTd__ "hd"
#define __INT16_FMTi__ "hi"
#define __INT16_MAX__ 32767
#define __INT16_TYPE__ short
#define __INT32_C_SUFFIX__
#define __INT32_FMTd__ "d"
#define __INT32_FMTi__ "i"
#define __INT32_MAX__ 2147483647
#define __INT32_TYPE__ int
#define __INT64_C_SUFFIX__ LL
#define __INT64_FMTd__ "lld"
#define __INT64_FMTi__ "lli"
#define __INT64_MAX__ 9223372036854775807LL
#define __INT64_TYPE__ long long int
#define __INT8_C_SUFFIX__
#define __INT8_FMTd__ "hhd"
#define __INT8_FMTi__ "hhi"
#define __INT8_MAX__ 127
#define __INT8_TYPE__ signed char
#define __INTMAX_C_SUFFIX__ L
#define __INTMAX_FMTd__ "ld"
#define __INTMAX_FMTi__ "li"
#define __INTMAX_MAX__ 9223372036854775807L
#define __INTMAX_TYPE__ long int
#define __INTMAX_WIDTH__ 64
#define __INTPTR_FMTd__ "ld"
#define __INTPTR_FMTi__ "li"
#define __INTPTR_MAX__ 9223372036854775807L
#define __INTPTR_TYPE__ long int
#define __INTPTR_WIDTH__ 64
#define __INT_FAST16_FMTd__ "hd"
#define __INT_FAST16_FMTi__ "hi"
#define __INT_FAST16_MAX__ 32767
#define __INT_FAST16_TYPE__ short
#define __INT_FAST16_WIDTH__ 16
#define __INT_FAST32_FMTd__ "d"
#define __INT_FAST32_FMTi__ "i"
#define __INT_FAST32_MAX__ 2147483647
#define __INT_FAST32_TYPE__ int
#define __INT_FAST32_WIDTH__ 32
#define __INT_FAST64_FMTd__ "lld"
#define __INT_FAST64_FMTi__ "lli"
#define __INT_FAST64_MAX__ 9223372036854775807LL
#define __INT_FAST64_TYPE__ long long int
#define __INT_FAST64_WIDTH__ 64
#define __INT_FAST8_FMTd__ "hhd"
#define __INT_FAST8_FMTi__ "hhi"
#define __INT_FAST8_MAX__ 127
#define __INT_FAST8_TYPE__ signed char
#define __INT_FAST8_WIDTH__ 8
#define __INT_LEAST16_FMTd__ "hd"
#define __INT_LEAST16_FMTi__ "hi"
#define __INT_LEAST16_MAX__ 32767
#define __INT_LEAST16_TYPE__ short
#define __INT_LEAST16_WIDTH__ 16
#define __INT_LEAST32_FMTd__ "d"
#define __INT_LEAST32_FMTi__ "i"
#define __INT_LEAST32_MAX__ 2147483647
#define __INT_LEAST32_TYPE__ int
#define __INT_LEAST32_WIDTH__ 32
#define __INT_LEAST64_FMTd__ "lld"
#define __INT_LEAST64_FMTi__ "lli"
#define __INT_LEAST64_MAX__ 9223372036854775807LL
#define __INT_LEAST64_TYPE__ long long int
#define __INT_LEAST64_WIDTH__ 64
#define __INT_LEAST8_FMTd__ "hhd"
#define __INT_LEAST8_FMTi__ "hhi"
#define __INT_LEAST8_MAX__ 127
#define __INT_LEAST8_TYPE__ signed char
#define __INT_LEAST8_WIDTH__ 8
#define __INT_MAX__ 2147483647
#define __INT_WIDTH__ 32
#define __LDBL_DECIMAL_DIG__ 17
#define __LDBL_DENORM_MIN__ 4.9406564584124654e-324L
#define __LDBL_DIG__ 15
#define __LDBL_EPSILON__ 2.2204460492503131e-16L
#define __LDBL_HAS_DENORM__ 1
#define __LDBL_HAS_INFINITY__ 1
#define __LDBL_HAS_QUIET_NAN__ 1
#define __LDBL_MANT_DIG__ 53
#define __LDBL_MAX_10_EXP__ 308
#define __LDBL_MAX_EXP__ 1024
#define __LDBL_MAX__ 1.7976931348623157e+308L
#define __LDBL_MIN_10_EXP__ (-307)
#define __LDBL_MIN_EXP__ (-1021)
#define __LDBL_MIN__ 2.2250738585072014e-308L
#define __LDBL_NORM_MAX__ 1.7976931348623157e+308L
#define __LITTLE_ENDIAN__ 1
#define __LLONG_WIDTH__ 64
#define __LONG_LONG_MAX__ 9223372036854775807LL
#define __LONG_MAX__ 9223372036854775807L
#define __LONG_WIDTH__ 64
#define __LP64__ 1
#define __MACH__ 1
#define __MEMORY_SCOPE_DEVICE 1
#define __MEMORY_SCOPE_SINGLE 4
#define __MEMORY_SCOPE_SYSTEM 0
#define __MEMORY_SCOPE_WRKGRP 2
#define __MEMORY_SCOPE_WVFRNT 3
#define __NO_INLINE__ 1
#define __NO_MATH_ERRNO__ 1
#define __OBJC_BOOL_IS_BOOL 1
#define __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES 3
#define __OPENCL_MEMORY_SCOPE_DEVICE 2
#define __OPENCL_MEMORY_SCOPE_SUB_GROUP 4
#define __OPENCL_MEMORY_SCOPE_WORK_GROUP 1
#define __OPENCL_MEMORY_SCOPE_WORK_ITEM 0
#define __ORDER_BIG_ENDIAN__ 4321
#define __ORDER_LITTLE_ENDIAN__ 1234
#define __ORDER_PDP_ENDIAN__ 3412
#define __PIC__ 2
#define __POINTER_WIDTH__ 64
#define __PRAGMA_REDEFINE_EXTNAME 1
#define __PTRDIFF_FMTd__ "ld"
#define __PTRDIFF_FMTi__ "li"
#define __PTRDIFF_MAX__ 9223372036854775807L
#define __PTRDIFF_TYPE__ long int
#define __PTRDIFF_WIDTH__ 64
#define __REGISTER_PREFIX__
#define __SCHAR_MAX__ 127
#define __SHRT_MAX__ 32767
#define __SHRT_WIDTH__ 16
#define __SIG_ATOMIC_MAX__ 2147483647
#define __SIG_ATOMIC_WIDTH__ 32
#define __SIZEOF_DOUBLE__ 8
#define __SIZEOF_FLOAT__ 4
#define __SIZEOF_INT128__ 16
#define __SIZEOF_INT__ 4
#define __SIZEOF_LONG_DOUBLE__ 8
#define __SIZEOF_LONG_LONG__ 8
#define __SIZEOF_LONG__ 8
#define __SIZEOF_POINTER__ 8
#define __SIZEOF_PTRDIFF_T__ 8
#define __SIZEOF_SHORT__ 2
#define __SIZEOF_SIZE_T__ 8
#define __SIZEOF_WCHAR_T__ 4
#define __SIZEOF_WINT_T__ 4
#define __SIZE_FMTX__ "lX"
#define __SIZE_FMTo__ "lo"
#define __SIZE_FMTu__ "lu"
#define __SIZE_FMTx__ "lx"
#define __SIZE_MAX__ 18446744073709551615UL
#define __SIZE_TYPE__ long unsigned int
#define __SIZE_WIDTH__ 64
#define __SSP__ 1
#define __STDC_EMBED_EMPTY__ 2
#define __STDC_EMBED_FOUND__ 1
#define __STDC_EMBED_NOT_FOUND__ 0
#define __STDC_HOSTED__ 1
#define __STDC_NO_THREADS__ 1
#define __STDC_UTF_16__ 1
#define __STDC_UTF_32__ 1
#define __STDC_VERSION__ 201710L
#define __STDC__ 1
#define __UINT16_C_SUFFIX__
#define __UINT16_FMTX__ "hX"
#define __UINT16_FMTo__ "ho"
#define __UINT16_FMTu__ "hu"
#define __UINT16_FMTx__ "hx"
#define __UINT16_MAX__ 65535
#define __UINT16_TYPE__ unsigned short
#define __UINT32_C_SUFFIX__ U
#define __UINT32_FMTX__ "X"
#define __UINT32_FMTo__ "o"
#define __UINT32_FMTu__ "u"
#define __UINT32_FMTx__ "x"
#define __UINT32_MAX__ 4294967295U
#define __UINT32_TYPE__ unsigned int
#define __UINT64_C_SUFFIX__ ULL
#define __UINT64_FMTX__ "llX"
#define __UINT64_FMTo__ "llo"
#define __UINT64_FMTu__ "llu"
#define __UINT64_FMTx__ "llx"
#define __UINT64_MAX__ 18446744073709551615ULL
#define __UINT64_TYPE__ long long unsigned int
#define __UINT8_C_SUFFIX__
#define __UINT8_FMTX__ "hhX"
#define __UINT8_FMTo__ "hho"
#define __UINT8_FMTu__ "hhu"
#define __UINT8_FMTx__ "hhx"
#define __UINT8_MAX__ 255
#define __UINT8_TYPE__ unsigned char
#define __UINTMAX_C_SUFFIX__ UL
#define __UINTMAX_FMTX__ "lX"
#define __UINTMAX_FMTo__ "lo"
#define __UINTMAX_FMTu__ "lu"
#define __UINTMAX_FMTx__ "lx"
#define __UINTMAX_MAX__ 18446744073709551615UL
#define __UINTMAX_TYPE__ long unsigned int
#define __UINTMAX_WIDTH__ 64
#define __UINTPTR_FMTX__ "lX"
#define __UINTPTR_FMTo__ "lo"
#define __UINTPTR_FMTu__ "lu"
#define __UINTPTR_FMTx__ "lx"
#define __UINTPTR_MAX__ 18446744073709551615UL
#define __UINTPTR_TYPE__ long unsigned int
#define __UINTPTR_WIDTH__ 64
#define __UINT_FAST16_FMTX__ "hX"
#define __UINT_FAST16_FMTo__ "ho"
#define __UINT_FAST16_FMTu__ "hu"
#define __UINT_FAST16_FMTx__ "hx"
#define __UINT_FAST16_MAX__ 65535
#define __UINT_FAST16_TYPE__ unsigned short
#define __UINT_FAST32_FMTX__ "X"
#define __UINT_FAST32_FMTo__ "o"
#define __UINT_FAST32_FMTu__ "u"
#define __UINT_FAST32_FMTx__ "x"
#define __UINT_FAST32_MAX__ 4294967295U
#define __UINT_FAST32_TYPE__ unsigned int
#define __UINT_FAST64_FMTX__ "llX"
#define __UINT_FAST64_FMTo__ "llo"
#define __UINT_FAST64_FMTu__ "llu"
#define __UINT_FAST64_FMTx__ "llx"
#define __UINT_FAST64_MAX__ 18446744073709551615ULL
#define __UINT_FAST64_TYPE__ long long unsigned int
#define __UINT_FAST8_FMTX__ "hhX"
#define __UINT_FAST8_FMTo__ "hho"
#define __UINT_FAST8_FMTu__ "hhu"
#define __UINT_FAST8_FMTx__ "hhx"
#define __UINT_FAST8_MAX__ 255
#define __UINT_FAST8_TYPE__ unsigned char
#define __UINT_LEAST16_FMTX__ "hX"
#define __UINT_LEAST16_FMTo__ "ho"
#define __UINT_LEAST16_FMTu__ "hu"
#define __UINT_LEAST16_FMTx__ "hx"
#define __UINT_LEAST16_MAX__ 65535
#define __UINT_LEAST16_TYPE__ unsigned short
#define __UINT_LEAST32_FMTX__ "X"
#define __UINT_LEAST32_FMTo__ "o"
#define __UINT_LEAST32_FMTu__ "u"
#define __UINT_LEAST32_FMTx__ "x"
#define __UINT_LEAST32_MAX__ 4294967295U
#define __UINT_LEAST32_TYPE__ unsigned int
#define __UINT_LEAST64_FMTX__ "llX"
#define __UINT_LEAST64_FMTo__ "llo"
#define __UINT_LEAST64_FMTu__ "llu"
#define __UINT_LEAST64_FMTx__ "llx"
#define __UINT_LEAST64_MAX__ 18446744073709551615ULL
#define __UINT_LEAST64_TYPE__ long long unsigned int
#define __UINT_LEAST8_FMTX__ "hhX"
#define __UINT_LEAST8_FMTo__ "hho"
#define __UINT_LEAST8_FMTu__ "hhu"
#define __UINT_LEAST8_FMTx__ "hhx"
#define __UINT_LEAST8_MAX__ 255
#define __UINT_LEAST8_TYPE__ unsigned char
#define __USER_LABEL_PREFIX__ _
#define __VERSION__ "Apple LLVM 17.0.0 (clang-1700.0.13.5)"
#define __WCHAR_MAX__ 2147483647
#define __WCHAR_TYPE__ int
#define __WCHAR_WIDTH__ 32
#define __WINT_MAX__ 2147483647
#define __WINT_TYPE__ int
#define __WINT_WIDTH__ 32
#define __aarch64__ 1
#define __apple_build_version__ 17000013
#define __arm64 1
#define __arm64__ 1
#define __block __attribute__((__blocks__(byref)))
#define __clang__ 1
#define __clang_literal_encoding__ "UTF-8"
#define __clang_major__ 17
#define __clang_minor__ 0
#define __clang_patchlevel__ 0
#define __clang_version__ "17.0.0 (clang-1700.0.13.5)"
#define __clang_wide_literal_encoding__ "UTF-32"
#define __llvm__ 1
#define __nonnull _Nonnull
#define __null_unspecified _Null_unspecified
#define __nullable _Nullable
#define __pic__ 2
#define __strong
#define __unsafe_unretained
#define __weak __attribute__((objc_gc(weak)))
```
So it seems like the CI runner does not support `i8mm` but the compiler is still
enabling the `MATMUL_INT8` feature. One way might be to check for this situation:
```cmake
                # Check if __ARM_FEATURE_MATMUL_INT8 is enabled but machine doesn't support i8mm
                string(FIND "${ARM_FEATURE}" "__ARM_FEATURE_MATMUL_INT8 1" matmul_int8_pos)
                if (NOT ${matmul_int8_pos} EQUAL -1 AND GGML_MACHINE_SUPPORTS_noi8mm)
                    message(STATUS "Unsetting __ARM_FEATURE_MATMUL_INT8 due to machine not supporting i8mm")
                    list(APPEND ARCH_FLAGS -U__ARM_FEATURE_MATMUL_INT8)
                endif()
```
