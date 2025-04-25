## Whisper Java Bindings Issue on Windows
This document describes and issue with Whisper on Windows when running
the Java bindings tests.

```console
> .\gradlew build --info
WhisperCppTest STANDARD_OUT
    JNA Library Path: C:\Users\danie\work\ai\whisper.cpp\bindings\java\build\generated\resources\main
    Working directory: C:\Users\danie\work\ai\whisper.cpp\bindings\java
    Library file exists: true at C:\Users\danie\work\ai\whisper.cpp\bindings\java\build\generated\resources\main\whisper.dll

WhisperCppTest STANDARD_ERROR
    Apr 25, 2025 8:15:11 AM com.sun.jna.Native extractFromResourcePath
    INFO: Looking in classpath from jdk.internal.loader.ClassLoaders$AppClassLoader@11a4022a for /com/sun/jna/win32-x86-64/jnidispatch.dll
    Apr 25, 2025 8:15:11 AM com.sun.jna.Native extractFromResourcePath
    INFO: Found library resource at jar:file:/C:/Users/danie/.gradle/caches/modules-2/files-2.1/net.java.dev.jna/jna/5.13.0/1200e7ebeedbe0d10062093f32925a912020e747/jna-5.13.0.jar!/com/sun/jna/win32-x86-64/jnidispatch.dll
    Apr 25, 2025 8:15:11 AM com.sun.jna.Native extractFromResourcePath
    INFO: Extracting library to C:\Users\danie\AppData\Local\Temp\jna-95350893\jna9802014600080209138.dll
WARNING: A restricted method in java.lang.System has been called
WARNING: java.lang.System::load has been called by com.sun.jna.Native in an unnamed module (file:/C:/Users/danie/.gradle/caches/modules-2/files-2.1/net.java.dev.jna/jna/5.13.0/1200e7ebeedbe0d10062093f32925a912020e747/jna-5.13.0.jar)
WARNING: Use --enable-native-access=ALL-UNNAMED to avoid a warning for callers in this module
WARNING: Restricted methods will be blocked in a future release unless native access is enabled

    Apr 25, 2025 8:15:11 AM com.sun.jna.NativeLibrary loadLibrary
    INFO: Looking for library 'whisper'
    Apr 25, 2025 8:15:11 AM com.sun.jna.NativeLibrary loadLibrary
    INFO: Adding paths from jna.library.path: C:\Users\danie\work\ai\whisper.cpp\bindings\java\build\generated\resources\main
    Apr 25, 2025 8:15:11 AM com.sun.jna.NativeLibrary loadLibrary
    INFO: Trying C:\Users\danie\work\ai\whisper.cpp\bindings\java\build\generated\resources\main\whisper.dll
    Apr 25, 2025 8:15:11 AM com.sun.jna.NativeLibrary loadLibrary
    INFO: Found library 'whisper' at C:\Users\danie\work\ai\whisper.cpp\bindings\java\build\generated\resources\main\whisper.dll
whisper_init_from_file_with_params_no_state: loading model from '../../models/ggml-tiny.en.bin'
whisper_init_with_params_no_state: use gpu    = 1
whisper_init_with_params_no_state: flash attn = 0
whisper_init_with_params_no_state: gpu_device = 0
whisper_init_with_params_no_state: dtw        = 0

WhisperCppTest > initializationError FAILED
    java.lang.Error: Invalid memory access
        at com.sun.jna.Native.invokePointer(Native Method)
        at com.sun.jna.Function.invokePointer(Function.java:497)
        at com.sun.jna.Function.invoke(Function.java:441)
        at com.sun.jna.Function.invoke(Function.java:361)
        at com.sun.jna.Library$Handler.invoke(Library.java:270)
        at jdk.proxy3/jdk.proxy3.$Proxy12.whisper_init_from_file_with_params(Unknown Source)
        at io.github.ggerganov.whispercpp.WhisperCpp.initContextImpl(WhisperCpp.java:63)
        at io.github.ggerganov.whispercpp.WhisperCpp.initContext(WhisperCpp.java:39)
        at io.github.ggerganov.whispercpp.WhisperCppTest.init(WhisperCppTest.java:28)
<===========--> 85% EXECUTING [35s]
> :test > 1 test completed, 1 failed
> :test > Executing test io.github.ggerganov.whispercpp.WhisperJnaLibraryTest
```
After quite a bit of debugging I managed to track this down to ggml/src/ggml-threading.cpp:
```
#include "ggml-threading.h"
#include <mutex>

std::mutex ggml_critical_section_mutex;

void ggml_critical_section_start() {
    ggml_critical_section_mutex.lock();
}

void ggml_critical_section_end(void) {
    ggml_critical_section_mutex.unlock();
}
```
When we compile whisper.cpp on Windows a version of Visual Studio is used, on my current machine
this is	Visual Studio 2022. `std::mutex` is included in the MSVCP140.dll which is part of the
system libraries and can be found in:
```console
C:\Windows\System32\MSVCP140.dll
C:\Windows\System32\VCRUNTIME140.dll
C:\Windows\System32\VCRUNTIME140_1.dll
```
Now, the problem seems to be that the Visual Studio headers may include code or expect features
rather from a newer version of MSVCP140.dll that what is installed on the system.

For this particular case the issue is.
```console
> echo ((Get-Command msvcp140.dll).Path)
C:\WINDOWS\system32\msvcp140.dll

> echo ((Get-Command msvcp140.dll).Version)

Major  Minor  Build  Revision
-----  -----  -----  --------
14     42     34438  0
```


```console
> echo ((Get-Command vcruntime140.dll).Path)
C:\WINDOWS\system32\vcruntime140.dll
```

```console
> echo ((Get-Command vcruntime140.dll).Version)

Major  Minor  Build  Revision
-----  -----  -----  --------
14     42     34438  0
```

### The issue
So the issue is that with never versions of the standard library, it will have a constexpr 
constructor which just sets the the mutex_impl field to nullptr. But an older version of 
the standard library will expect the constructor to have initialized this pointer, and when 
it tries to dereference it there will be an invalid memory access.

The `_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR` macro forces the mutex constructor to be non-constexpr
and initializes the internal structures in a way that's compatible with older runtime libraries.

### Workaround
A workaround is to add the following macro to `ggml-threading.cpp`:
```c++
#define DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR
```
I've verified that this works on my machine and on CI (github action workflow).
I'm not sure modifying `ggml-threading.cpp` is a good solution as this seems
to be very specific to Java and JNA if I've understood things correctly.
This can also be included in the CMakeLists.txt in whisper.cpp:                   
```console
target_compile_definitions(ggml-base PRIVATE _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
```
