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
This is a file (archive?) that contains everything needed to run an app and distributed it for testing
to the app store.


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
