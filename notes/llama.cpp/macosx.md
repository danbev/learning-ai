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

### Architecture support

ios 14.0 (Released 2020)
 - No cblas_gemm support

ios 15.0 (Released 2021)
 - No cblas_gemm support

ios 16.0 (Released 2022)
 - Metal 3 support
 - Improved neural engine support
 - Advanced ML features

ios 17.0 (Released 2023)
 - Further ML improvements
 - Advanced ML features

### Frameworks
A framework contains resources like shared libraries, nib files, image files, strings files information
property lists (Info.plists), headers. There are also unbrella framworks which can contain other
frameworks. 
These are bundled as a directory with the `.framework` extension.

A framework can contain multiple versions of the frameworks code and header files in the same bundle.
So the same framework can ship older versions of the code and header files in the same bundle and this
is called a versioned bundle.

Framework structure:
- Versions/Current -> llama
- Headers -> Versions/Current/Headers
- Modules -> Versions/Current/Modules
- Resources -> Versions/Current/Resources
- llamas -> Versions/Current/llama

For example, for llama.cpp one framework, we will have multiple frameworks as there is one for
each architecture. So we will have:
```console
$ ls build-apple/llama.xcframework/ios-arm64/llama.framework/
Headers		Modules		Resources	Versions	llama

$ ls -l  build-apple/llama.xcframework/ios-arm64/llama.framework/Versions/
total 0
drwxr-xr-x@ 7 danbev  staff  224 Feb 27 07:27 A
lrwxr-xr-x@ 1 danbev  staff    1 Feb 27 07:27 Current -> A
```

### dSYM files
These are directores (bundles) that contain debugging informations for object files. On linux
these are usually added as sections to the object files themselves. 

If we use the same example framework as above the .dSYM bundle/directory will have a directory
next to the llama.framework directory:
```console
$ ls -l  build-apple/llama.xcframework/ios-arm64/
dSYMs/           llama.framework/
```
```console
$ ls -R build-apple/llama.xcframework/ios-arm64/dSYMs/llama.dSYM/
Contents

build-apple/llama.xcframework/ios-arm64/dSYMs/llama.dSYM//Contents:
Info.plist	Resources

build-apple/llama.xcframework/ios-arm64/dSYMs/llama.dSYM//Contents/Resources:
DWARF

build-apple/llama.xcframework/ios-arm64/dSYMs/llama.dSYM//Contents/Resources/DWARF:
llama
```

This is what Info.plist looks like:
```console
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple Computer//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
    <dict>
        <key>CFBundleDevelopmentRegion</key>
        <string>English</string>
        <key>CFBundleIdentifier</key>
        <string>ggml-org.llama.dsym</string>
        <key>CFBundleInfoDictionaryVersion</key>
        <string>6.0</string>
        <key>CFBundlePackageType</key>
        <string>dSYM</string>
        <key>CFBundleSignature</key>
        <string>????</string>
        <key>CFBundleShortVersionString</key>
        <string>1.0</string>
        <key>CFBundleVersion</key>
        <string>1</string>
    </dict>
</plist>
```
The `????` is just a placeholder for a legacy value that is no longer used and can be anything.

Each such .dSYM file is linked to an executable of library using an UUID, so they have to have he
same id for the linnker to match them. When Xcode compiles an object file or perhaps a group of
them in the case of a library, it generates a UUID which is embedded in the header of the object
file. The .dSYM file also contains this UUID and the debugger can use this to match the object file.
Inspecting the UUID of the object file can be done using dwarfdump:
```console
$ dwarfdump --uuid build-apple/llama.xcframework/ios-arm64/dSYMs/llama.dSYM/Contents/Resources/DWARF/llama
```
But this does not should any UUID, and there is something wrong with how I'm created the 
.dsym! Now, this has to do with the way we are building this. We are following Apple's guidelines
when it comes to xcframeworks to have static libraries instead of dynamic libraries as they might
not work across different version/cpu architectures. But this means that the .dSYM files are not
generated by Xcode, instead this generated by us manually. This is why the UUID is not present in
the .dSYM file. The .dSYM file is required by the App Store validation process so this is a compromise
and we still have debug symbols in the individual object files.

When a crash occurs or a debugger is attached the system looks for this UUID of the binary and
seaches for a dsym with a matching UUID. When/if found the debug information can be used to map
addresses in the binary to source code lines.

The process looks something like this when an app crashes:
* Create a binary image list containing all the loaded images (executables and libraries)
* Store each image's UUID
* Store the memory address where the crash occured.

The lookup process is then:
* Looks in the app bundle itself for embedded debug symbols

Each binary (executable or library in this case, not every object file) as a UUID which is stored
in it's `_LINKEDIT` segment. This can be inspected (if there is one) using:
```console
$ dwarfdump --uuid build-apple/llama.xcframework/ios-arm64/dSYMs/llama.dSYM/Contents/Resources/DWARF/llama
```

Dwarf is a "play" on the word ELF which is the Executable and Linkable Format. Dwarf is the Debugging
Attributed Record Format.

Inspect an object file:
```console
$ ar -x build-ios-device/ggml/src/Release-iphoneos/libggml.a ggml-backend-reg.o
$ dwarfdump ggml-backend-reg.o | head
ggml-backend-reg.o:     file format Mach-O arm64

.debug_info contents:
0x00000000: Compile Unit: length = 0x000330aa, format = DWARF32, version = 0x0004, abbr_offset = 0x0000, addr_size = 0x08 (next unit at 0x000330ae)

0x0000000b: DW_TAG_compile_unit
              DW_AT_producer    ("Apple clang version 15.0.0 (clang-1500.3.9.4)")
              DW_AT_language    (DW_LANG_C_plus_plus_14)
              DW_AT_name        ("/Users/danbev/work/llama.cpp/ggml/src/ggml-backend-reg.cpp")
              DW_AT_LLVM_sysroot        ("/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS17.5.sdk")
error: write on a pipe with no reader
```

### Validate an application
In xcode we can validate an application by using the `Product -> Archive` menu item. First make sure
that you have a `Team` selected for the `Release` configuration in `Signing & Capabilities`, and
doulbe check that you are not on the `Debug` configuration. Then select `Product -> Archive` and
and `Validate`. If there are any errors with application, or the xcframeworks (dependencies
of the app) then they should be displayed here.

You might run into an issue where the application name is already take and the validation fails
becauase of that. We can rename the app by going to `Build Settings` and search for `Product Name`
and update it. 

Also make sure to select a device like `Any iOS Device`.

Set `Skip Install` to `YES` in the `Build Settings` for the `Release` configuration. 


An archive is a build of your app, including debugging information, that Xcode stores in a bundle.
Xcode repackages the archive’s contents based on the distribution configuration you choose for
your distribution.


### Signing
The framework provider (llama.cpp in this case) does NOT need to sign the XCFramework for distribution. You can build and publish the XCFramework unsigned.
The end user (app developer) who incorporates the XCFramework is responsible for signing everything in their app bundle, including third-party frameworks, during their own build process.


### bitcode issue
Bitcode is the binary representation of llvm IR (intermediate representation) which was used by Apple
to enable them to potentially recompile the app for different architectures. This is not used anymore
and it is not clear to me if it was used but it was required for submission to the app store at some
point. It has now been deprecated and is not required for submission to the app store.

But I ran into this when building an xcframework for llama.cpp:
```console
/Users/danbev/work/llama.cpp/build-apple/llama.xcframework/ios-arm64/llama.framework bitcode_strip /Users/danbev/work/llama.cpp/build-apple/llama.xcframework/ios-arm64/llama.framework/Versions/A/llama.dSYM/Contents/Resources/DWARF/llama: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/bitcode_strip exited with 1
```
```console
bitcode_strip /Users/danbev/work/llama.cpp/build-apple/llama.xcframework/ios-arm64/llama.framework/Versions/A/llama.dSYM/Contents/Resources/DWARF/llama


```console
    -DCMAKE_XCODE_ATTRIBUTE_ENABLE_BITCODE=NO
    -DCMAKE_XCODE_ATTRIBUTE_BITCODE_GENERATION_MODE=none
```

### Signing issue
When trying to validate an application in xcode I got the following error:
```console
Missing signing identifier at "/var/folders/7h/g216wj3x0qldxw27twph8mwr0000gn/T/XcodeDistPipeline.~~~9whq1k/Root/Payload/llama.swiftui.app/Frameworks/llama.framework/Libraries/libggml-base.dylib".
```

```console
Missing signing identifier at "/var/folders/7h/g216wj3x0qldxw27twph8mwr0000gn/T/XcodeDistPipeline.~~~0erbQK/Root/Payload/llama.swiftui.app/Frameworks/llama.framework/Versions/A/llama.dSYM/Contents/Resources/DWARF/llama".
```

### dSYM
Debugging symbols are extracted and stored in a .dSYM bundle. Each time we build our app xcode will
generate an UUID for the app and this UUID is stored in the binary. The .dSYM bundle also contains
this UUID and the debugger can use this to match the binary with the .dSYM bundle. This is how the
For example:
```console
$ dwarfdump -u build-apple/llama.xcframework/ios-arm64/dSYMs/llama.dSYM/Contents/Resources/DWARF/llama
UUID: 57B0892D-F062-35BB-B7D9-65D2941C7594 (arm64) build-apple/llama.xcframework/ios-arm64/dSYMs/llama.dSYM/Contents/Resources/DWARF/llama
```
When a crash occurs the device will collect information about the specific exception, the stack trace
which only contains raw memory addresses. A list of images (user and system frameworks) loaded by
the application (each one has a UUID).
This information is then sent to the crash reporting service which will then look for the .dSYM
bundle with the matching UUID. If found the crash reporting service can then symbolicate the stack
trace and provide the developer with a human readable stack trace.

Now, these should not be in the actual bundle of a framework but separate from it or there will
be validation errors saying that the file containing the symbols is a binary executable and not
allowed in the bundle. For example:
```console
“llama.swiftui.app/Frameworks/llama.framework/llama.dSYM/Contents/Resources/DWARF/llama” binary file
is not permitted. Your app cannot contain standalone executables or libraries, other than a valid
CFBundleExecutable of supported bundles.
For details, visit: https://developer.apple.com/documentation/bundleresources/placing_content_in_a_bundle (ID: ba3bef0c-9f49-47e6-8250-5f3488bcb1a3)
```
In a framework there should be a .dSYM bundle next to the framework bundle and not inside of it.

### Missing CFBundleIconName
```console
Validation failed
Missing Info.plist value. A value for the Info.plist key 'CFBundleIconName' is missing in the bundle 'ggml-org.llama'. Apps built with iOS 11 or later SDK must supply app icons in an asset catalog and must also provide a value for this Info.plist key. For more information see http://help.apple.com/xcode/mac/current/#/dev10510b1f7. (ID: c7ab3ccb-c2ec-4dbe-b950-9fef14371fae)
```
There is now icon in UI/Assets.xcassets/AppIcon.appiconset/ so I added one. Not sure if there was one before
or not. 

```console
Validation failed
Missing required icon file. The bundle does not contain an app icon for iPhone / iPod Touch of exactly '120x120' pixels, in .png format for iOS versions >= 10.0. To support older versions of iOS, the icon may be required in the bundle outside of an asset catalog. Make sure the Info.plist file includes appropriate entries referencing the file. See https://developer.apple.com/documentation/bundleresources/information_property_list/user_interface (ID: a173b605-820c-46b7-9386-cb5ee4043e2d)
```

```console
Validation failed
Invalid bundle structure. The “llama.swiftui.app/Frameworks/llama.framework/Versions/A/llama” binary file is not permitted. Your app cannot contain standalone executables or libraries, other than a valid CFBundleExecutable of supported bundles. For details, visit: https://developer.apple.com/documentation/bundleresources/placing_content_in_a_bundle (ID: 33e8f4a4-59b4-4c7d-b77c-7e239ecdc1eb)
```

### iOS App Store Package (IPA)
This is an archive file, like a zip file with a specific strucutre.
It contains a Payload directory which contains the app bundle.

```console
$ file ~/work/llama.cpp/validation-builds/ios/iOSLlamaTest.ipa
/Users/danbev/work/llama.cpp/validation-builds/ios/iOSLlamaTest.ipa: Zip archive data, at least v1.0 to extract, compression method=store
```
```console
$ ls ~/work/llama.cpp/validation-builds/ios/temp/Payload/iOSLlamaTest.app/
Frameworks	Info.plist	PkgInfo		iOSLlamaTest
```
We can inspect the Info.plist file:
```console
$ plutil -p /Users/danbev/work/llama.cpp/validation-builds/ios/temp/Payload/iOSLlamaTest.app/Info.plist
{
  "BuildMachineOSBuild" => "23G80"
  "CFBundleDevelopmentRegion" => "en"
  "CFBundleExecutable" => "iOSLlamaTest"
  "CFBundleIdentifier" => "org.ggml.iOSLlamaTest"
  "CFBundleInfoDictionaryVersion" => "6.0"
  "CFBundleName" => "iOSLlamaTest"
  "CFBundlePackageType" => "APPL"
  "CFBundleShortVersionString" => "1.0"
  "CFBundleSupportedPlatforms" => [
    0 => "iPhoneOS"
  ]
  "CFBundleVersion" => "1"
  "DTCompiler" => "com.apple.compilers.llvm.clang.1_0"
  "DTPlatformBuild" => "22C146"
  "DTPlatformName" => "iphoneos"
  "DTPlatformVersion" => "18.2"
  "DTSDKBuild" => "22C146"
  "DTSDKName" => "iphoneos18.2"
  "DTXcode" => "1620"
  "DTXcodeBuild" => "16C5032a"
  "LSRequiresIPhoneOS" => 1
  "MinimumOSVersion" => "16.4"
  "UIDeviceFamily" => [
    0 => 1
    1 => 2
  ]
  "UILaunchScreen" => {
  }
  "UIRequiredDeviceCapabilities" => [
    0 => "arm64"
  ]
  "UISupportedInterfaceOrientations" => [
    0 => "UIInterfaceOrientationPortrait"
  ]
}
```


### Frameworks structure
The framework structure is a bit different for macos and ios which can be important when creating
a xcframework. Each arch will need to be packages in the way that is expected by the platform.
For macos we have a structure like this:
```console


```

### Version warning
I got this warning when validating an app:
```console
SDK version issue. This app was built with the iOS 17.5 SDK. Starting April 24, 2025, all iOS and
iPadOS apps must be built with the iOS 18 SDK or later, included in Xcode 16 or later, in order to
be uploaded to App Store Connect or submitted for distribution.
```
But the rest of the validation process passed. 


### Framework structure

#### macOS Frameworks
On macOS, frameworks are usually organized in a versioned bundle structure. This means
that within the framework bundle you’ll typically find a Versions folder that holds
different iterations (like A, B, etc.) of the binary and its resources. This structure
lets the system manage backward compatibility by letting apps reference a specific
version or the current one via a symlink.

#### iOS (and visionOS) Frameworks
For iOS, frameworks are packaged in a much flatter format. Since iOS apps bundle the
frameworks they need and the platform imposes stricter rules (such as code signing and
sandboxing), there isn’t the same need for a multi-version structure.

With visionOS being a new platform from Apple, its framework layout aligns more with the
modern, streamlined approach seen on iOS. In practice, this means visionOS frameworks are
typically structured as a flat bundle rather than using the older versioning scheme found
on macOS.


## VisionOS target issue
```console
Creating dynamic library for visionos.
clang++: error: unknown argument: '-mxros-version-min=1.0'
```

```
-mtargetos=xros1.0
```

### App specific password
https://account.apple.com/account/manage



### Metal backend in ggml
The Metal backend in ggml needs kernels function to be compiled and these need to
be loaded by the GPU. Just recall that even though we compile the kernel on a
compiler the binary generated is actually run on the GPU.
```console
$ xcrun metal -c src/kernel.metal -o kernel.air
```
And then we can compile the air (Apple Intermediate Representation) file to a
metallib file:
```console
xcrun metallib kernel.air -o kernel.metallib
```
This `kernel.metalib` file contains the compiled kernels and can be loaded by the
GPU. So this file, or its contents need to be available to the ggml-metal backend
in some way.

In ggml the metal library can be embedded which means the source code in src/ggml-metal.metal
is embedded into the object file (most often the dynamic library). This can then be accessed
and it can be compiled and loaded into the GPU.
```cmake
if (GGML_METAL_EMBED_LIBRARY)
    enable_language(ASM)

    add_compile_definitions(GGML_METAL_EMBED_LIBRARY)

    set(METALLIB_COMMON "${CMAKE_CURRENT_SOURCE_DIR}/../ggml-common.h")
    set(METALLIB_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/ggml-metal.metal")
    set(METALLIB_IMPL   "${CMAKE_CURRENT_SOURCE_DIR}/ggml-metal-impl.h")

    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/autogenerated")

    # merge ggml-common.h and ggml-metal.metal into a single file
    set(METALLIB_EMBED_ASM        "${CMAKE_BINARY_DIR}/autogenerated/ggml-metal-embed.s")
    set(METALLIB_SOURCE_EMBED     "${CMAKE_BINARY_DIR}/autogenerated/ggml-metal-embed.metal")
    set(METALLIB_SOURCE_EMBED_TMP "${CMAKE_BINARY_DIR}/autogenerated/ggml-metal-embed.metal.tmp")

    add_custom_command(
        OUTPUT ${METALLIB_EMBED_ASM}
        COMMAND echo "Embedding Metal library"
        COMMAND sed -e '/__embed_ggml-common.h__/r         ${METALLIB_COMMON}' -e '/__embed_ggml-common.h__/d'         < ${METALLIB_SOURCE}           > ${METALLIB_SOURCE_EMBED_TMP}
        COMMAND sed -e '/\#include \"ggml-metal-impl.h\"/r ${METALLIB_IMPL}'   -e '/\#include \"ggml-metal-impl.h\"/d' < ${METALLIB_SOURCE_EMBED_TMP} > ${METALLIB_SOURCE_EMBED}
        COMMAND echo ".section __DATA,__ggml_metallib"          >  ${METALLIB_EMBED_ASM}
        COMMAND echo ".globl _ggml_metallib_start"              >> ${METALLIB_EMBED_ASM}
        COMMAND echo "_ggml_metallib_start:"                    >> ${METALLIB_EMBED_ASM}
        COMMAND echo ".incbin \\\"${METALLIB_SOURCE_EMBED}\\\"" >> ${METALLIB_EMBED_ASM}
        COMMAND echo ".globl _ggml_metallib_end"                >> ${METALLIB_EMBED_ASM}
        COMMAND echo "_ggml_metallib_end:"                      >> ${METALLIB_EMBED_ASM}
        DEPENDS ../ggml-common.h ggml-metal.metal ggml-metal-impl.h
        COMMENT "Generate assembly for embedded Metal library"
    )

    target_sources(ggml-metal PRIVATE ${METALLIB_EMBED_ASM})
```
This will generate an assembly file which will be compiled and linked into the
ggml-metal backend. The assembly file will contain the following:
```console
$ cat build/autogenerated/ggml-metal-embed.s
.section __DATA,__ggml_metallib
.globl _ggml_metallib_start
_ggml_metallib_start:
.incbin "/Users/danbev/work/llama.cpp/build/autogenerated/ggml-metal-embed.metal"
.globl _ggml_metallib_end
_ggml_metallib_end:
```
We can see that `ggml-metal-embed.metal` is included in the assembly file and
notice that the symbols `_ggml_metallib_start` and `_ggml_metallib_end` are defined
here. These are then used in `ggml_metal_init`:
```
static struct ggml_backend_metal_context * ggml_metal_init(ggml_backend_dev_t dev) {
    GGML_LOG_INFO("%s: allocating\n", __func__);

#if TARGET_OS_OSX && !GGML_METAL_NDEBUG
    // Show all the Metal device instances in the system
    NSArray * devices = MTLCopyAllDevices();
    for (id<MTLDevice> device in devices) {
        GGML_LOG_INFO("%s: found device: %s\n", __func__, [[device name] UTF8String]);
    }
    [devices release]; // since it was created by a *Copy* C method
#endif

    // init context
    struct ggml_backend_metal_context * ctx = calloc(1, sizeof(struct ggml_backend_metal_context));
    struct ggml_backend_metal_device_context * ctx_dev = dev->context;

    id<MTLDevice> device = ggml_backend_metal_device_acq(ctx_dev);
    GGML_LOG_INFO("%s: picking default device: %s\n", __func__, [[device name] UTF8String]);

    ctx->queue  = [device newCommandQueue];
    if (ctx->queue == nil) {
        GGML_LOG_ERROR("%s: error: failed to create command queue\n", __func__);
        return NULL;
    }

    ctx->d_queue = dispatch_queue_create("ggml-metal", DISPATCH_QUEUE_CONCURRENT);

    id<MTLLibrary> metal_library = nil;

    // load library
    //
    // - first check if the library is embedded
    // - then check if the library is in the bundle
    // - if not found, load the source and compile it
    // - if that fails, return NULL
    {
        NSError * error = nil;
        NSString * src = nil;

#if GGML_METAL_EMBED_LIBRARY
        GGML_LOG_INFO("%s: using embedded metal library\n", __func__);

        extern const char ggml_metallib_start[];
        extern const char ggml_metallib_end[];

        src = [[NSString alloc] initWithBytes:ggml_metallib_start length:(ggml_metallib_end-ggml_metallib_start) encoding:NSUTF8StringEncoding];

#else
    ...
```
Notice the `extern const char ggml_metallib_start[];` and `extern const char ggml_metallib_end[];`
This is then read into a string:
```c++
src = [NSString stringWithContentsOfFile:path_source encoding:NSUTF8StringEncoding error:&error];
```
And a little later used to call the newLibraryWithSource method on the device instance:
```c++
    MTLCompileOptions * options = [MTLCompileOptions new];
    options.preprocessorMacros = prep;

    //[options setFastMathEnabled:false];

    metal_library = [device newLibraryWithSource:src options:options error:&error];
```
This will compile the sources into a MTLLibrary instance which contains the compiled
functions.
So we can build ggml using `GGML_METAL_EMBED_LIBRARY=OFF` in which case  and the sources will be embedded

```c++
        NSBundle * bundle = [NSBundle bundleForClass:[GGMLMetalClass class]];
        NSString * path_lib = [bundle pathForResource:@"default" ofType:@"metallib"];
```
So the first line will use what is called a marker class to get the bundle that contains
the class. The bundle in this case is the `bin` directory because I'm running this as a 
command line tool, but if this was part of an iso app then this would use the bundle to
look up this file.
This is then used to get a resource name `default`, with the extension `metallib`.
```console
(lldb) br set -f ggml-metal.m -n ggml_metal_init
(lldb) p bundle
(NSBundle *) 0x0000600002fb4050 @"/Users/danbev/work/llama.cpp/build/bin"
(lldb) p path_lib
(__NSCFString *) 0x0000600002fb32f0 @"/Users/danbev/work/llama.cpp/build/bin/default.metallib"
```
This path is then used.
```c++
        if (path_lib != nil) {
            // pre-compiled library found
            NSURL * libURL = [NSURL fileURLWithPath:path_lib];
            GGML_LOG_INFO("%s: loading '%s'\n", __func__, [path_lib UTF8String]);

            metal_library = [device newLibraryWithURL:libURL error:&error];
```
So the above will load the precompiled library.

After this the kernel functions will be loaded.
```c++
    // load kernels
    {
        NSError * error = nil;

        for (int i = 0; i < GGML_METAL_KERNEL_TYPE_COUNT; ++i) {
            ctx->kernels[i].pipeline = nil;
        }

#define GGML_METAL_ADD_KERNEL(e, name, supported) \
        if (supported) { \
            struct ggml_metal_kernel * kernel = &ctx->kernels[e]; \
            id<MTLFunction> metal_function = [metal_library newFunctionWithName:@"kernel_"#name]; \
            kernel->pipeline = [device newComputePipelineStateWithFunction:metal_function error:&error]; \
            GGML_LOG_DEBUG("%s: loaded %-40s %16p | th_max = %4d | th_width = %4d\n", __func__, "kernel_"#name, (void *) kernel->pipeline, \
                    (int) kernel->pipeline.maxTotalThreadsPerThreadgroup, \
                    (int) kernel->pipeline.threadExecutionWidth); \
            [metal_function release]; \
            if (error) { \
                GGML_LOG_ERROR("%s: error: load pipeline error: %s\n", __func__, [[error description] UTF8String]); \
                [metal_library release]; \
                return NULL; \
            } \
        } else { \
            GGML_LOG_WARN("%s: skipping %-40s (not supported)\n", __func__, "kernel_"#name); \
        }

        const bool has_simdgroup_mm        = ctx_dev->has_simdgroup_mm;
        const bool has_simdgroup_reduction = ctx_dev->has_simdgroup_reduction;
        const bool use_bfloat              = ctx_dev->use_bfloat;

        // simd_sum and simd_max requires MTLGPUFamilyApple7

        GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_ADD,                           add,                            true);
```
A `ggml_metal_kernel` is a struct that contains a pointer to an `MTLComputePipelineState`
instance:
```c++
struct ggml_metal_kernel {
    id<MTLComputePipelineState> pipeline;
};

enum ggml_metal_kernel_type {
    GGML_METAL_KERNEL_TYPE_ADD,
    GGML_METAL_KERNEL_TYPE_ADD_ROW,
    ...
    GGML_METAL_KERNEL_TYPE_COUNT
};
```
And the `context->kernels` is an array these and they are index by.
```c++
    struct ggml_metal_kernel kernels[GGML_METAL_KERNEL_TYPE_COUNT];
```

We can expand the macro above and inspect it:
```console
$ clang++ -E ggml/src/ggml-metal/ggml-metal.m
...
        if (1) { struct ggml_metal_kernel * kernel = &ctx->kernels[GGML_METAL_KERNEL_TYPE_ADD]; id<MTLFunction> metal_function = [metal_library newFunctionWithName:@"kernel_""add"]; kernel->pipeline = [device newComputePipelineStateWithFunction:metal_function error:&error]; GGML_LOG_DEBUG("%s: loaded %-40s %16p | th_max = %4d | th_width = %4d\n", __func__, "kernel_""add", (void *) kernel->pipeline, (int) kernel->pipeline.maxTotalThreadsPerThreadgroup, (int) kernel->pipeline.threadExecutionWidth); [metal_function release]; if (error) { GGML_LOG_ERROR("%s: error: load pipeline error: %s\n", __func__, [[error description] UTF8String]); [metal_library release]; return ((void *)0); } } else { GGML_LOG_WARN("%s: skipping %-40s (not supported)\n", __func__, "kernel_""add"); };
```
And we can clean that up a bit:
```c++
        if (1) {
            struct ggml_metal_kernel * kernel = &ctx->kernels[GGML_METAL_KERNEL_TYPE_ADD];
            id<MTLFunction> metal_function = [metal_library newFunctionWithName:@"kernel_""add"];
            kernel->pipeline = [device newComputePipelineStateWithFunction:metal_function error:&error];
            [metal_function release];
            if (error) {
                [metal_library release]; return ((void *)0); }
        } else {
            GGML_LOG_WARN("%s: skipping %-40s (not supported)\n", __func__, "kernel_""add"); 
        };
```
Notice that this is using `metal_library` which contains all the compiled kernels. The
`newFunctionWithName` is not actually creating a function but rather looking up a function
in the library. The `new` part is from Objective-C's ownership policy and signals that this
is a new reference that the caller is responsible for releasing.

`newComputePipelineStateWithFunction` takes the `kernal_add` function and compiles it into
a pipeline state which is a compiled, validated, and optimized representation of the 
GPU code to be executed, the hardware state configuration.
Creating a pipeline state involves:
* Validating that the function can run on the current GPU
* Optimizing the code for the specific GPU architecture
* Pre-configuring all the fixed-function hardware settings
* Allocating any GPU-side resources needed

And the above is done for all the kernels in the `gggml_metal_kernel_type` enum.

The actual kernel function is defined in ggml/src/ggml-metal/ggml-metal.metal:
```c++
kernel void kernel_add(
        constant ggml_metal_kargs_bin & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i03 = tgpig.z;
    const int i02 = tgpig.y;
    const int i01 = tgpig.x;

    const int i13 = i03%args.ne13;
    const int i12 = i02%args.ne12;
    const int i11 = i01%args.ne11;

    device const char * src0_ptr = src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + args.offs;
    device const char * src1_ptr = src1 + i13*args.nb13 + i12*args.nb12 + i11*args.nb11;
    device       char * dst_ptr  = dst  + i03*args.nb3  + i02*args.nb2  + i01*args.nb1  + args.offs;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        const int i10 = i0%args.ne10;
        *((device float *)(dst_ptr + i0*args.nb0)) = *((device float *)(src0_ptr + i0*args.nb00)) + *((device float *)(src1_ptr + i10*args.nb10));
    }
}
```
So when `ggml_metal_init` completes all the kernel functions have been compiled and loaded
and are ready to be executed on the GPU.

So let now take a look at how the kernels are actually executed. When a graph is executed,
for example by `llama_decode_impl` the `ggml_metal_graph_compute` will be call (indirectly):
```c++
static enum ggml_status ggml_metal_graph_compute(
            ggml_backend_t   backend,
        struct ggml_cgraph * gf) {
    struct ggml_backend_metal_context        * ctx     = backend->context;
    struct ggml_backend_metal_device_context * ctx_dev = backend->device->context;

    // number of nodes encoded by the main thread (empirically determined)
    const int n_main = 128;

    // number of threads in addition to the main thread
    const int n_cb = ctx->n_cb;

    @autoreleasepool {
        ctx->gf = gf;

        ctx->n_nodes_0 = MIN(n_main, gf->n_nodes);
        ctx->n_nodes_1 = gf->n_nodes - ctx->n_nodes_0;

        ctx->n_nodes_per_cb = (ctx->n_nodes_1 + ctx->n_cb - 1) / ctx->n_cb;

        const bool should_capture = ctx->capture_next_compute;
        if (should_capture) {
            ...
        }

        // the main thread commits the first few commands immediately
        // command_buffer[n_cb]
        {
            id<MTLCommandBuffer> command_buffer = [ctx->queue commandBufferWithUnretainedReferences];
            ctx->command_buffers[n_cb] = command_buffer;

            [command_buffer enqueue];
            ctx->encode_async(n_cb);
        }
```
So the above is creating a pointer (id) to a `MTLCommandBuffer` and storing this command buffer
in the context command buffers array. Notice that is is being stored in the `n_cb` index:
```console
(lldb) p ctx->command_buffers
(id[9]) {
  [0] = nil
  [1] = 0x0000000105304220
  [2] = nil
  [3] = nil
  [4] = nil
  [5] = nil
  [6] = nil
  [7] = nil
  [8] = nil
}
(lldb) p ctx->n_cb
(int) 1
```
We need to understand a little bit about the context.

```c++
struct ggml_backend_metal_context {
    id<MTLCommandQueue> queue;

    dispatch_queue_t d_queue;

    struct ggml_metal_kernel kernels[GGML_METAL_KERNEL_TYPE_COUNT];

    // capture state
    bool capture_next_compute;
    bool capture_started;

    id<MTLCaptureScope> capture_scope;

    // command buffer state
    int n_cb;           // number of extra threads used to submit the command buffers
    int n_nodes_0;      // number of nodes submitted by the main thread
    int n_nodes_1;      // remaining number of nodes submitted by the n_cb threads
    int n_nodes_per_cb;

    struct ggml_cgraph * gf;

    // the callback given to the thread pool
    void (^encode_async)(size_t ith);

    // n_cb command buffers + 1 used by the main thread
    id<MTLCommandBuffer> command_buffers[GGML_METAL_MAX_COMMAND_BUFFERS + 1];

    // abort ggml_metal_graph_compute if callback returns true
    ggml_abort_callback abort_callback;
    void *              abort_callback_data;
};
```
So each context has a pointer to a `MTLCommandQueue` which is a serial queue
that manages the execution of command buffers. Command buffers are containers
that encode (record/describe) GPU operations. 
The following will add the command buffer to the `ctx->queue`:
```
            [command_buffer enqueue];
```
This does not start any execution, it only adds it to the queue. One reason this is
done is to ensure the order of the command buffers..

The next thing that happens is that `encode_async` is called with the `n_cb` index:
```c++
            ctx->encode_async(n_cb);
```
Now, this is a function that is defined in `ggml_backend_metal_set_n_cb` which is called
from `ggml_backend_metal_device_init`:
```c++
static void ggml_backend_metal_set_n_cb(ggml_backend_t backend, int n_cb) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    struct ggml_backend_metal_context * ctx = (struct ggml_backend_metal_context *)backend->context;

    ...

    ctx->encode_async = Block_copy(^(size_t iter) {
        const int cb_idx = iter;
        const int n_cb_l = ctx->n_cb;

        const int n_nodes_0 = ctx->n_nodes_0;
        const int n_nodes_1 = ctx->n_nodes_1;

        const int n_nodes_per_cb = ctx->n_nodes_per_cb;

        id<MTLCommandBuffer> command_buffer  = ctx->command_buffers[cb_idx];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

        int node_start = 0;
        int node_end   = n_nodes_0;

        if (cb_idx < n_cb_l) {
            node_start = n_nodes_0 + (                                         (cb_idx + 0) * n_nodes_per_cb);
            node_end   = n_nodes_0 + (MIN((cb_idx == n_cb_l - 1) ? n_nodes_1 : (cb_idx + 1) * n_nodes_per_cb, n_nodes_1));
        }

        const bool should_capture = ctx->capture_next_compute;

        for (int idx = node_start; idx < node_end; ++idx) {
            if (should_capture) {
                [encoder pushDebugGroup:[NSString stringWithCString:ggml_op_desc(ggml_graph_node(ctx->gf, idx)) encoding:NSUTF8StringEncoding]];
            }

            ggml_metal_encode_node(backend, idx, encoder);

            if (should_capture) {
                [encoder popDebugGroup];
            }
        }

        [encoder endEncoding];

        if (cb_idx < 2 || ctx->abort_callback == NULL) {
            [command_buffer commit];
        }
    });
}
```
The `^` denotes the block syntax in Objective-C, this is like a lambda or closure in other
languages. So `^(size_t iter) {..}` is a block that takes a `size_t` argument and does something. 
`Block_copy` is a function that copies the block to the heap so that it can be used after the
current scope has ended. This is because the block is created on the stack and will be deallocated
when the scope ends. This is needed becuase it is getting assigned to a member of the context.
So previously we saw the following code:
```c++
            [command_buffer enqueue];
            ctx->encode_async(n_cb);
```
And this is calling the above block passing in the `n_cb` index.
In the block/lambda a command buffer is created as is an encoder. This encoder will then be
passed to `ggml_metal_encode_node`.
```c++
static void ggml_metal_encode_node(
                        ggml_backend_t   backend,
                                   int   idx,
          id<MTLComputeCommandEncoder>   encoder) {
    struct ggml_backend_metal_context        * ctx     = backend->context;
    struct ggml_backend_metal_device_context * ctx_dev = backend->device->context;

    struct ggml_cgraph * gf = ctx->gf;

    struct ggml_tensor * node = ggml_graph_node(gf, idx);

    //GGML_LOG_INFO("%s: encoding node %3d, op = %8s\n", __func__, idx, ggml_op_name(node->op));

    struct ggml_tensor * src0 = node->src[0];
    struct ggml_tensor * src1 = node->src[1];
    struct ggml_tensor * src2 = node->src[2];
    struct ggml_tensor * dst  = node;

    ...
    id<MTLBuffer> id_src0 = src0 ? ggml_metal_get_buffer(src0, &offs_src0) : nil;
    id<MTLBuffer> id_src1 = src1 ? ggml_metal_get_buffer(src1, &offs_src1) : nil;
    id<MTLBuffer> id_src2 = src2 ? ggml_metal_get_buffer(src2, &offs_src2) : nil;
    id<MTLBuffer> id_dst  = dst  ? ggml_metal_get_buffer(dst,  &offs_dst)  : nil;

    if (ggml_is_empty(dst)) {
        return;
    }

    ...

    id<MTLDevice> device = ctx_dev->mtl_device;

    switch (dst->op) {
        ...
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            {
                GGML_ASSERT(src0t == GGML_TYPE_F32);
                GGML_ASSERT(src1t == GGML_TYPE_F32);

                const size_t offs = 0;

                bool bcast_row = false;

                id<MTLComputePipelineState> pipeline = nil;

                if (ggml_nelements(src1) == ne10 && ggml_is_contiguous(src1) && ne00 % 4 == 0 && ne10 % 4 == 0) {
                    GGML_ASSERT(ggml_is_contiguous(src0));

                    // src1 is a row
                    GGML_ASSERT(ne11 == 1);

                    switch (dst->op) {
                        case GGML_OP_ADD: pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_ADD_ROW].pipeline; break;
                        case GGML_OP_SUB: pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_SUB_ROW].pipeline; break;
                        case GGML_OP_MUL: pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_ROW].pipeline; break;
                        case GGML_OP_DIV: pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_DIV_ROW].pipeline; break;
                        default: GGML_ABORT("fatal error");
                    }

                    bcast_row = true;
                } else {
                    switch (dst->op) {
                        case GGML_OP_ADD: pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_ADD].pipeline; break;
                        case GGML_OP_SUB: pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_SUB].pipeline; break;
                        case GGML_OP_MUL: pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL].pipeline; break;
                        case GGML_OP_DIV: pipeline = ctx->kernels[GGML_METAL_KERNEL_TYPE_DIV].pipeline; break;
                        default: GGML_ABORT("fatal error");
                    }
                }

                ggml_metal_kargs_bin args = {
                    /*.ne00 =*/ ne00,
                    /*.ne01 =*/ ne01,
                    /*.ne02 =*/ ne02,
                    /*.ne03 =*/ ne03,
                    /*.nb00 =*/ nb00,
                    /*.nb01 =*/ nb01,
                    /*.nb02 =*/ nb02,
                    /*.nb03 =*/ nb03,
                    /*.ne10 =*/ ne10,
                    /*.ne11 =*/ ne11,
                    /*.ne12 =*/ ne12,
                    /*.ne13 =*/ ne13,
                    /*.nb10 =*/ nb10,
                    /*.nb11 =*/ nb11,
                    /*.nb12 =*/ nb12,
                    /*.nb13 =*/ nb13,
                    /*.ne0  =*/ ne0,
                    /*.ne1  =*/ ne1,
                    /*.ne2  =*/ ne2,
                    /*.ne3  =*/ ne3,
                    /*.nb0  =*/ nb0,
                    /*.nb1  =*/ nb1,
                    /*.nb2  =*/ nb2,
                    /*.nb3  =*/ nb3,
                    /*.offs =*/ offs,
                };

                [encoder setComputePipelineState:pipeline];
                [encoder setBytes:&args length:sizeof(args) atIndex:0];
                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:1];
                [encoder setBuffer:id_src1 offset:offs_src1 atIndex:2];
                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:3];

                if (bcast_row) {
                    const int64_t n = ggml_nelements(dst)/4;

                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                } else {
                    const int nth = MIN((int) pipeline.maxTotalThreadsPerThreadgroup, ne0);

                    [encoder dispatchThreadgroups:MTLSizeMake(ne01, ne02, ne03) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
                }
            } break;
```
So the above is setting up the encoder with the arguments for the kernel function and then
dispatching the threads. The `dispatchThreadgroups` method records a compute command to the
encoder and specifies how many threads to launch, how many threads per group to use. But this
does not run anything yet.

When all the nodes have been encoded the encoder is ended and the command buffer is committed:
```
            ggml_metal_encode_node(backend, idx, encoder);

            if (should_capture) {
                [encoder popDebugGroup];
            }
        }

        [encoder endEncoding];

        if (cb_idx < 2 || ctx->abort_callback == NULL) {
            [command_buffer commit];
        }
```
The `commit` method will submit the command buffer to the queue for execution. 

One thing that I did not notice during my first pass through this the following:
```c++
    struct ggml_tensor * src0 = node->src[0];
    ...
    size_t offs_src0 = 0;
    id<MTLBuffer> id_src0 = src0 ? ggml_metal_get_buffer(src0, &offs_src0) : nil;
```
So the source tensor (source of an operation) is passed to `ggml_metal_get_buffer`
which will use the tensors buffer to get the buffer context:
```c++
static id<MTLBuffer> ggml_metal_get_buffer(struct ggml_tensor * t, size_t * offs) {
    const int64_t tsize = ggml_nbytes(t);

    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;

    struct ggml_backend_metal_buffer_context * buf_ctx = (struct ggml_backend_metal_buffer_context *) buffer->context;

    // find the view that contains the tensor fully
    for (int i = 0; i < buf_ctx->n_buffers; ++i) {
        const int64_t ioffs = (int64_t) t->data - (int64_t) buf_ctx->buffers[i].data;

        if (ioffs >= 0 && ioffs + tsize <= (int64_t) buf_ctx->buffers[i].size) {
            *offs = (size_t) ioffs;

            return buf_ctx->buffers[i].metal;
        }
    }

    GGML_LOG_ERROR("%s: error: tensor '%s' buffer is nil\n", __func__, t->name);

    return nil;
}
```
And buffers is an array of `ggml_backend_metal_buffer`:
```c++
struct ggml_backend_metal_buffer_context {
    void * all_data;
    size_t all_size;
    bool owned;

    // multiple buffers are used only to avoid the maximum buffer size limitation when using mmap
    int n_buffers;
    struct ggml_backend_metal_buffer buffers[GGML_METAL_MAX_BUFFERS];

    // optional MTLResidencySet
    id rset;
};
```
And one of these buffer looks like this:
```c++
struct ggml_backend_metal_buffer {
    void   * data;
    size_t   size;

    id<MTLBuffer> metal;
};
```
And notice in the code above that `.metal` is returned. So this is returning a pointer
to a MTBuffer. So to understand how this we need to look at how the buffer is created.
```c++
static ggml_backend_buffer_t ggml_backend_metal_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    struct ggml_backend_metal_buffer_context * ctx = calloc(1, sizeof(struct ggml_backend_metal_buffer_context));

    const size_t size_page = sysconf(_SC_PAGESIZE);

    size_t size_aligned = size;
    if ((size_aligned % size_page) != 0) {
        size_aligned += (size_page - (size_aligned % size_page));
    }

    struct ggml_backend_metal_device_context * ctx_dev = (struct ggml_backend_metal_device_context *)buft->device->context;
    id<MTLDevice> device = ggml_backend_metal_device_acq(ctx_dev);

    ctx->all_data = ggml_metal_host_malloc(size_aligned);
    ctx->all_size = size_aligned;
    ctx->owned = true;
    ctx->n_buffers = 1;

    if (ctx->all_data != NULL) {
        ctx->buffers[0].data  = ctx->all_data;
        ctx->buffers[0].size  = size;
        ctx->buffers[0].metal = nil;

        if (size_aligned > 0) {
            ctx->buffers[0].metal = [device newBufferWithBytesNoCopy:ctx->all_data
                                            length:size_aligned
                                            options:MTLResourceStorageModeShared
                                            deallocator:nil];
        }
    }

    if (size_aligned > 0 && (ctx->all_data == NULL || ctx->buffers[0].metal == nil)) {
        GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f MiB\n", __func__, size_aligned / 1024.0 / 1024.0);
        free(ctx);
        ggml_backend_metal_device_rel(ctx_dev);
        return NULL;
    }

    if (!ggml_backend_metal_buffer_rset_init(ctx, ctx_dev, device)) {
        GGML_LOG_ERROR("%s: error: failed to initialize residency set\n", __func__);
        free(ctx);
        ggml_backend_metal_device_rel(ctx_dev);
        return NULL;
    }

    //ggml_backend_metal_log_allocated_size(device, size_aligned);

    return ggml_backend_buffer_init(buft, ggml_backend_metal_buffer_i, ctx, size);
}
```
Notice the `newBufferWithBytesNoCopy`:
```c++
            ctx->buffers[0].metal = [device newBufferWithBytesNoCopy:ctx->all_data
                                            length:size_aligned
                                            options:MTLResourceStorageModeShared
                                            deallocator:nil];
```
This is creating a new buffer that is pointing to CPU memory pointed to by `ctx->all_data`. This
is possible by using Apple's unified memory architecture (UMA) which is available in M1, M2 and
later chips. The deallocator is nil which means that the buffer will not be deallocated when the
buffer is deallocated so this is managed by ggml.

After that and if there current plaform support ResidencySet the buffers are added to the residency
set:
```c++
    if (!ggml_backend_metal_buffer_rset_init(ctx, ctx_dev, device)) {
        GGML_LOG_ERROR("%s: error: failed to initialize residency set\n", __func__);
        free(ctx);
        ggml_backend_metal_device_rel(ctx_dev);
        return NULL;
    }
```
```c++
// rset init
static bool ggml_backend_metal_buffer_rset_init(
        struct ggml_backend_metal_buffer_context * ctx,
        struct ggml_backend_metal_device_context * ctx_dev,
        id<MTLDevice> device) {
    ctx->rset = nil;

    if (!ctx_dev->has_residency_sets) {
        return true;
    }

#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
    if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, *)) {
        MTLResidencySetDescriptor * desc = [[MTLResidencySetDescriptor alloc] init];
        desc.label = @"ggml_backend_metal";
        desc.initialCapacity = ctx->n_buffers;

        NSError * error;
        ctx->rset = [device newResidencySetWithDescriptor:desc error:&error];
        if (error) {
            GGML_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
            [desc release];
            return false;
        }

        [desc release];

        for (int i = 0; i < ctx->n_buffers; i++) {
            [ctx->rset addAllocation:ctx->buffers[i].metal];
        }

        [ctx->rset commit];
        [ctx->rset requestResidency];

        return true;
    }
#else
    GGML_UNUSED(ctx_dev);
    GGML_UNUSED(device);
#endif

    return true;
}
```
The label is just for debugging purposes. And the buffers will be added to the residency set
and can be used like the normally would. The requestResidency will tell Metal to try to keep
there reousces in GPU memory.
