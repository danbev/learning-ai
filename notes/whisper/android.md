## whisper.android
There are two android related examples in whisper, one written in Java
(whisper.android.java) and the other in Kotlin (whisper.android).

Both examples use JNI and they build ggml as part of the build process.

### whisper.android (Kotlin)
The JNI code is located in `whisper.android/lib/src/main/jni/whisper` where
we can find the cmake file and a `jni.c`.

The cmake file has support for building using external ggml library/directory:
```cmake
function(build_library target_name)
    add_library(
        ${target_name}
        SHARED
        ${SOURCE_FILES}
    )
    ...

    if (GGML_HOME)
        include(FetchContent)
        FetchContent_Declare(ggml SOURCE_DIR ${GGML_HOME})
        FetchContent_MakeAvailable(ggml)

        target_compile_options(ggml PRIVATE ${GGML_COMPILE_OPTIONS})
        target_link_libraries(${target_name} ${LOG_LIB} android ggml)
    else()
        target_link_libraries(${target_name} ${LOG_LIB} android)
    endif()
```
What `FetchContent` does is usually downloads an external library and makes it
available but in this case it is using this to include a local directory.

Now, we have the following TODO in this file:
```cmake
# TODO: this needs to be updated to work with the new ggml CMakeLists

if (NOT GGML_HOME)
    set(
        SOURCE_FILES
        ${SOURCE_FILES}
        ${WHISPER_LIB_DIR}/ggml/src/ggml.c
        ${WHISPER_LIB_DIR}/ggml/src/ggml-alloc.c
        ${WHISPER_LIB_DIR}/ggml/src/ggml-backend.cpp
        ${WHISPER_LIB_DIR}/ggml/src/ggml-backend-reg.cpp
        ${WHISPER_LIB_DIR}/ggml/src/ggml-quants.c
        ${WHISPER_LIB_DIR}/ggml/src/ggml-threading.cpp
        ${WHISPER_LIB_DIR}/ggml/src/ggml-cpu/ggml-cpu.c
        ${WHISPER_LIB_DIR}/ggml/src/ggml-cpu/ggml-cpu.cpp
        ${WHISPER_LIB_DIR}/ggml/src/ggml-cpu/hbm.cpp
        ${WHISPER_LIB_DIR}/ggml/src/ggml-cpu/traits.cpp
        ${WHISPER_LIB_DIR}/ggml/src/ggml-cpu/unary-ops.cpp
        ${WHISPER_LIB_DIR}/ggml/src/ggml-cpu/binary-ops.cpp
        ${WHISPER_LIB_DIR}/ggml/src/ggml-cpu/vec.cpp
        ${WHISPER_LIB_DIR}/ggml/src/ggml-cpu/ops.cpp
        ${WHISPER_LIB_DIR}/ggml/src/ggml-cpu/arch/arm/quants.c
        ${WHISPER_LIB_DIR}/ggml/src/ggml-cpu/arch/arm/repack.cpp
        ${WHISPER_LIB_DIR}/ggml/src/ggml-cpu/quants.c
        )
endif()
```
Now, this is not great as when updates are made to ggml cmake build files we
sometimes need to manually update this section. We need avoid this.
