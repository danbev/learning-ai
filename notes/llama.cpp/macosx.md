## Macos notes for llama.cpp

### Building with xcodebuild
We need to install an iOS platform install to build which can be installed using:
```console
$ xcodebuild -downloadPlatform iOS
```

To build the project with xcodebuild, use the following command:
```console
$ xcodebuild -scheme llama-Package -destination "generic/platform=iOS"
```

### Building swifutui example
```console
xcodebuild -project examples/llama.swiftui/llama.swiftui.xcodeproj \
           -scheme llama.swiftui \
           -sdk iphoneos \
           HEADER_SEARCH_PATHS="/Users/danbev/work/llama.cpp/include /Users/danbev/work/llama.cpp/ggml/include" \
	   LIBRARY_SEARCH_PATHS="/Users/danbev/work/llama.cpp/build/bin/Release" \
           OTHER_LDFLAGS="-lllama -lggml" \
           CODE_SIGNING_REQUIRED=NO \
           CODE_SIGN_IDENTITY= \
           -destination generic/platform=iOS build
```

### multiple resources named 'ggml-metal.metal' in target 'llama'
I ran into this issue when trying to build the project with xcodebuild. The error message is:
```console
$ xcodebuild -scheme llama-Package -destination "generic/platform=iOS"
Command line invocation:
    /Applications/Xcode.app/Contents/Developer/usr/bin/xcodebuild -scheme llama -destination generic/platform=iOS

User defaults from command line:
    IDEPackageSupportUseBuiltinSCM = YES

Resolve Package Graph
multiple resources named 'ggml-metal.metal' in target 'llama'multiple resources named 'ggml-metal-embed.metal' in target 'llama'

Resolved source packages:
  llama: /Users/danbev/work/llama.cpp

2024-11-01 07:02:29.107 xcodebuild[45972:9437010] Writing error result bundle to /var/folders/7h/g216wj3x0qldxw27twph8mwr0000gn/T/ResultBundle_2024-01-11_07-02-0029.xcresult
xcodebuild: error: Could not resolve package dependencies:
  multiple resources named 'ggml-metal.metal' in target 'llama'
  multiple resources named 'ggml-metal-embed.metal' in target 'llama'
```
This was due to there being a `ggml-metal.metal` file in the build directory:
```console
$ find . -name ggml-metal.metal
./build/bin/ggml-metal.metal
./ggml/src/ggml-metal.metal
```
Removing this file allowed the build to progress.

### system information
```console
$ arch
arm64
```
```console
$ system_profiler SPHardwareDataType
Hardware:

    Hardware Overview:

      Model Name: MacBook Pro
      Model Identifier: Mac15,3
      Model Number: Z1C8001TNKS/A
      Chip: Apple M3
      Total Number of Cores: 8 (4 performance and 4 efficiency)
      Memory: 24 GB
      System Firmware Version: 10151.140.19
      OS Loader Version: 10151.140.19
      Serial Number (system): FC4PK6W1J4
      Hardware UUID: B382C8B4-D4AF-594C-B235-7E700F336892
      Provisioning UDID: 00008122-000258CA0180001C
      Activation Lock Status: Disabled
```
```console
$ uname -m
arm64
```
```console
$ sysctl -n machdep.cpu.brand_string
Apple M3
```
For arm64:
```console
$ sysctl hw.optional
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
hw.optional.arm.FEAT_SB: 1
hw.optional.arm.FEAT_FRINTTS: 1
hw.optional.arm.FEAT_LRCPC: 1
hw.optional.arm.FEAT_LRCPC2: 1
hw.optional.arm.FEAT_FCMA: 1
hw.optional.arm.FEAT_JSCVT: 1
hw.optional.arm.FEAT_PAuth: 1
hw.optional.arm.FEAT_PAuth2: 1
hw.optional.arm.FEAT_FPAC: 1
hw.optional.arm.FEAT_DPB: 1
hw.optional.arm.FEAT_DPB2: 1
hw.optional.arm.FEAT_BF16: 1
hw.optional.arm.FEAT_I8MM: 1
hw.optional.arm.FEAT_WFxT: 0
hw.optional.arm.FEAT_RPRES: 1
hw.optional.arm.FEAT_ECV: 1
hw.optional.arm.FEAT_AFP: 1
hw.optional.arm.FEAT_LSE2: 1
hw.optional.arm.FEAT_CSV2: 1
hw.optional.arm.FEAT_CSV3: 1
hw.optional.arm.FEAT_DIT: 1
hw.optional.arm.FEAT_FP16: 1
hw.optional.arm.FEAT_SSBS: 1
hw.optional.arm.FEAT_BTI: 1
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
hw.optional.floatingpoint: 1
hw.optional.neon: 1
hw.optional.neon_hpfp: 1
hw.optional.neon_fp16: 1
hw.optional.armv8_1_atomics: 1
hw.optional.armv8_2_fhm: 1
hw.optional.armv8_2_sha512: 1
hw.optional.armv8_2_sha3: 1
hw.optional.armv8_3_compnum: 1
hw.optional.watchpoint: 4
hw.optional.breakpoint: 6
hw.optional.armv8_crc32: 1
hw.optional.armv8_gpi: 1
hw.optional.AdvSIMD: 1
hw.optional.AdvSIMD_HPFPCvt: 1
hw.optional.ucnormal_mem: 1
hw.optional.arm64: 1

### Check for a specific feature
```console
$ sysctl hw.optional.arm.FEAT_DotProd
hw.optional.arm.FEAT_DotProd: 1
```
In this case the DotProd feature is available. This is available in c++ code
as `__ARM_FEATURE_DOTPROD`.


### Feature issue
```console
/llama.cpp/ggml/src/ggml-cpu/ggml-cpu-quants.c:2108:31: error:
always_inline function 'vdotq_s32' requires target feature 'dotprod',
      but would be inlined into function 'ggml_vec_dot_q4_0_q8_0' that is compiled without support for 'dotprod'
 2108 |         const int32x4_t p_1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), v0_1ls, v1_1l), v0_1hs, v1_1h);
      |                               ^
In file included from /llama.cpp/ggml/src/ggml-cpu/ggml-cpu-quants.c:7:
/llama.cpp/ggml/src/ggml-cpu/ggml-cpu-impl.h:323:33: note: expanded from macro 'ggml_vdotq_s32'
  323 | #define ggml_vdotq_s32(a, b, c) vdotq_s32(a, b, c)
````
The user also reported that the following was displayed during cmake configuration:
```console
-- ARM -mcpu not found, -mcpu=native will be used
-- Performing Test GGML_MACHINE_SUPPORTS_dotprod
-- Performing Test GGML_MACHINE_SUPPORTS_dotprod - Failed
```
It sounds like they might not have support for the dotprod feature on their machine's cpu, but
for some reason. In `ggml-cpu-quants.c` line 2108 we have the following line of code:
```c++
        const int32x4_t p_1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), v0_1ls, v1_1l), v0_1hs, v1_1h);
```
This actualy a macro which is defined in `ggml-cpu-impl.h`
```c++
#if !defined(__ARM_FEATURE_DOTPROD)

inline static int32x4_t ggml_vdotq_s32(int32x4_t acc, int8x16_t a, int8x16_t b) {
    const int16x8_t p0 = vmull_s8(vget_low_s8 (a), vget_low_s8 (b));
    const int16x8_t p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));

    return vaddq_s32(acc, vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)));
}

#else

#define ggml_vdotq_s32(a, b, c) vdotq_s32(a, b, c)

#endif // !defined(__ARM_FEATURE_DOTPROD)
```
In this case the `ggml_vdot_s32` function is defined as a macro which expands to `vdotq_s32`
which is what the user is running into. But since there cpu does not have this instruction
it should not have taken this route.

This is happening because of the following in 
```console
    if (CMAKE_OSX_ARCHITECTURES      STREQUAL "arm64" OR
        CMAKE_GENERATOR_PLATFORM_LWR STREQUAL "arm64" OR
        (NOT CMAKE_OSX_ARCHITECTURES AND NOT CMAKE_GENERATOR_PLATFORM_LWR AND
            CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm.*|ARM64)$"))

        message(STATUS "ARM detected")

        if (MSVC AND NOT CMAKE_C_COMPILER_ID STREQUAL "Clang")
            message(FATAL_ERROR "MSVC is not supported for ARM, use clang")
        else()
            check_cxx_compiler_flag(-mfp16-format=ieee GGML_COMPILER_SUPPORTS_FP16_FORMAT_I3E)
            if (NOT "${GGML_COMPILER_SUPPORTS_FP16_FORMAT_I3E}" STREQUAL "")
                list(APPEND ARCH_FLAGS -mfp16-format=ieee)
            endif()

            if (GGML_NATIVE)
                # -mcpu=native does not always enable all the features in some compilers,
                # so we check for them manually and enable them if available

                execute_process(
                    COMMAND ${CMAKE_C_COMPILER} -mcpu=native -E -v -
                    INPUT_FILE "/dev/null"
                    OUTPUT_QUIET
                    ERROR_VARIABLE ARM_MCPU
                    RESULT_VARIABLE ARM_MCPU_RESULT
                )
                if (NOT ARM_MCPU_RESULT)
                    string(REGEX MATCH "-mcpu=[^ ']+" ARM_MCPU_FLAG "${ARM_MCPU}")
                endif()
                if ("${ARM_MCPU_FLAG}" STREQUAL "")
                    set(ARM_MCPU_FLAG -mcpu=native)
                    message(STATUS "ARM -mcpu not found, -mcpu=native will be used")
                endif()

                include(CheckCXXSourceRuns)

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

                check_arm_feature(dotprod "#include <arm_neon.h>\nint main() { int8x16_t _a, _b; volatile int32x4_t _s = vdotq_s32(_s, _a, _b); return 0; }")
```
This issue is that if `GGML_NATIVE` is set then it will check for the `dotprod` feature using the
current compiler. This not what we want in this case as we are cross compiling.


```console
/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/libtool: for architecture: x86_64 file: /Users/danbev/work/llama.cpp/build-ios-sim/ggml/src/Release-iphonesimulator/libggml-cpu.a(amx.o) has no symbols
```
This is a message from libtool and notice the `amx` which is Apple Matrix Accelerator which is only
available on Apple silicon processors (ARM architectures) and not available on `x86_64`. The compiler
is seeing the amx.o and for `x86_64` it does not find any symbols for that architecture.
So this is just a warning and can be ignored. We might be able to surpress this using
`no_warning_for_no_symbols`:
```console
$ libtool -static -no_warning_for_no_symbols -o "${base_dir}/${output_lib}" \
```
