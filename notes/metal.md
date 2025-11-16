## Metal
Is Apple's low-level hardware-accelerated 3D graphics and compute framework.

### Shaders
Are written in Metal Shading Language (MSL) which is based on C++14.
There are different types of shaders:
* Vertex shaders
* Fragment shaders
* Compute shaders

As an example we can look at [kernel.metal](../gpu/metal/src/kernel.metal) which is a compute shader.

```metal
#include <metal_stdlib>

using namespace metal;

kernel void simpleMultiply(const device float* input  [[buffer(0)]],
                                 device float* output [[buffer(1)]],
                           uint id [[thread_position_in_grid]]) {
    output[id] = input[id] * 2.0;
}
```
We compile this kernel into Apple Intermediate Representation (AIR) using the `metal` compiler:
```console
$ xcrun metal -c src/kernel.metal -o kernel.air
```
AIR is a LLVM like bitcode which is a binary representation of the LLVM IR but is not standard LLVM bitcode.

Multiple kernels can be packaged together into a single library using the `metallib` tool:
```console
$ xcrun metallib kernel.air -o kernel.metallib
```

To inspect the symbols:
```console
$ xcrun metal-nm kernel.metallib
0000005c T simpleMultiply
```

To disassemble the metallib:
```console
$ xcrun metal-objdump -d kernel.metallib
```

Shaders are compiled into a `.metallib` file which is then loaded by the application.

The metalib can then be loaded by a Swift or Objective-C/C++ application using the Metal framework.
An Objective-C source file would have the extension `.m` and an Objective-C++ source file
would have the extension `.mm`. Normally when we use c++ libraries/features then we would use
the `.mm` extension.

### Parameter syntax
These are specified with double brackets which are called attribute specifiers which provide
extra information to the compiler. This is similar to Rust's attributes which are specified
using `#[...]`.

Lets look at the `simpleMultiply` kernel:
```c++
kernel void simple_multiply(const device float* input  [[buffer(0)]],
                                  device float* output [[buffer(1)]],
                           uint id [[thread_position_in_grid]]) {
```
The `[[buffer(0)]]` specifies that this parameter is bound to buffer 0.

Also notice the `device` keyword which specifies that the buffer is stored in device memory.
And `[[thread_position_in_grid]]` is also an attribute specifier which specifies that this parameter
is the thread position in the grid.


### Thread groups
A compute shader is executed in thread.


### File extensions
* .metal contains Metal Shading Language (MSL) code.
* .m contains Objective-C code.
* .mm contains Objective-C++ code.
* .metallib contains compiled Metal code.


### Objective-C brush up
Function calls are done with square brackets:
```objc
id<MTLLibrary> defaultLibrary = [device newLibraryWithFile:libraryPath error:&error];
```
This is calling a method on the `device` object named `newLibraryWithFile`. The first
parameter is part of the method name and the additional parameters need to be named,
like error above.

### Exploration
In the example [project](../gpu/metal) we have the kernel in [kernel.metal](../gpu/metal/src/kernel.metal).
We compile this using `metal`:
```console
$ xcrun metal --help
OVERVIEW: clang LLVM compiler

USAGE: metal [options] file...

OPTIONS:
  -###                    Print (but do not run) the commands to run for this compilation
  --amdgpu-arch-tool=<value>
                          Tool used for detecting AMD GPU arch in the system.
  --analyzer-output <value>
                          Static analyzer report output format (html|plist|plist-multi-file|plist-html|sarif|sarif-html|text).
  --analyze               Run the static analyzer
```
Notice that this says clang and the output if very similiar to a normal llvm tool chain. So I'm
guessing that Metal is a frontend to clang.

So we compile the kernel into an object file:
```console
$ xcrun metal -c src/kernel.metal -o kernel.air
```
This produces a file with the extension `.air` (Apple Intermediate Representation) which is LLVM
bitcode:
```console
$ file kernel.air
kernel.air: LLVM bitcode, wrapper
```
And recall that bitcode is a binary representation of the LLVM IR.
We can inspect this format using:
```console
$ xcrun metal-objdump -d kernel.air

kernel.air:	file format LLVM IR

0x00000000000000 -- simpleMultiply:
; ModuleID = 'kernel.air'
source_filename = "src/kernel.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64-apple-macosx14.0.0"

; Function Attrs: argmemonly mustprogress nofree norecurse nosync nounwind willreturn
define void @simpleMultiply(float addrspace(1)* nocapture noundef readonly "air-buffer-no-alias" %0, float addrspace(1)* nocapture noundef writeonly "air-buffer-no-alias" %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = zext i32 %2 to i64
  %5 = getelementptr inbounds float, float addrspace(1)* %0, i64 %4
  %6 = load float, float addrspace(1)* %5, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %7 = fmul fast float %6, 2.000000e+00
  %8 = getelementptr inbounds float, float addrspace(1)* %1, i64 %4
  store float %7, float addrspace(1)* %8, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  ret void
}

attributes #0 = { argmemonly mustprogress nofree norecurse nosync nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!15, !16, !17}
!llvm.ident = !{!18}
!air.version = !{!19}
!air.language_version = !{!20}
!air.source_file_name = !{!21}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 14, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{void (float addrspace(1)*, float addrspace(1)*, i32)* @simpleMultiply, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"input"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"output"}
!14 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"id"}
!15 = !{!"air.compile.denorms_disable"}
!16 = !{!"air.compile.fast_math_enable"}
!17 = !{!"air.compile.framebuffer_fetch_enable"}
!18 = !{!"Apple metal version 32023.155 (metalfe-32023.155)"}
!19 = !{i32 2, i32 6, i32 0}
!20 = !{!"Metal", i32 3, i32 1, i32 0}
!21 = !{!"/Users/danbev/work/ai/learning-ai/gpu/metal/src/kernel.metal"}
!22 = !{!23, !23, i64 0}
!23 = !{!"float", !24, i64 0}
!24 = !{!"omnipotent char", !25, i64 0}
!25 = !{!"Simple C++ TBAA"}
!26 = !{!27}
!27 = distinct !{!27, !28, !"air-alias-scope-arg(0)"}
!28 = distinct !{!28, !"air-alias-scopes(simpleMultiply)"}
!29 = !{!30}
!30 = distinct !{!30, !28, !"air-alias-scope-arg(1)"}
```

We can then compile this into a binary library:
```console
xcrun metallib kernel.air -o kernel.metallib
```
metallib is the backend/linker and will perform multiple things:
* Optimize the code specifically for Metal GPU targets
* Packages all shaders, there can be more than one like we have above, into a single library file.

We can think of the .metallib in a similar way to .dylibs (a compiled binary that contains multiple
functions).
```swift
    guard let defaultLibrary = try? device.makeLibrary(filepath: libraryPath) else {
```
`defaultLibrary` will be of type `MTLLibrary` which is like a handle to a loaded dynamic
library. When we call `makeFunction` that is similar to using `dlsym` to get a function pointer
from a dynamic library.

For additional details and a brush up on objective-c syntax see [simple.mm](../gpu/metal/src/simple.mm).

Lambdas/closures in objective-c are done using blocks. For example:
```objc
^(size_t iter) { ... }
```

### MTLComputePipelineState
This is what compiles the AIR to run on the GPU, and is similar to compiling from PTX to SASS in CUDA.
This returned object is an optimized and ready-to-run GPU program.

```objective-c
id<MTLComputePipelineState> computePipelineState = [device newComputePipelineStateWithFunction:kernelFunction error:&error];
```
This is an expensive operation and should be done once and reused.


### GGML_USE_METAL
It was no clear to me where this variable was being set. I searched for it but could not find it:
```console
$ find . -name CMakeLists.txt | xargs grep -n GGML_USE_METAL
$
```
This is because is set programatically in cmake. If we look in `ggml/src/CMakeLists.txt` we see:
```cmake
ggml_add_backend(BLAS)
ggml_add_backend(CANN)
ggml_add_backend(CUDA)
ggml_add_backend(HIP)
ggml_add_backend(Kompute)
ggml_add_backend(METAL)
ggml_add_backend(MUSA)
ggml_add_backend(RPC)
ggml_add_backend(SYCL)
ggml_add_backend(Vulkan)
ggml_add_backend(OpenCL)

function(ggml_add_backend backend)
    string(TOUPPER "GGML_${backend}" backend_id)
    if (${backend_id})
        string(TOLOWER "ggml-${backend}" backend_target)
        add_subdirectory(${backend_target})
        message(STATUS "Including ${backend} backend")
        if (NOT GGML_BACKEND_DL)
            string(TOUPPER "GGML_USE_${backend}" backend_use)
            target_compile_definitions(ggml PUBLIC ${backend_use})
        endif()
    endif()
endfunction()
```

### Adding an new operation to the metal backend
Apart from implementing the actual operation in a metal kernel we also need to enable the

#### Add a new struct for the operation
This is done by adding a new struct in ggml/src/ggml-metal/ggml-metal-impl.h:
```c
typedef struct {
    float repeat;
    float freq;
    float present;
    int32_t n_vocab;
} ggml_metal_kargs_penalties;
```

#### Add device support for the new operation
This is done by adding a new case in ggml_metal_device_supports_op in
ggml/src/ggml-metal/ggml-metal-device.m:
```objc
bool ggml_metal_device_supports_op(ggml_metal_device_t dev, const struct ggml_tensor * op) {
    ...
    switch (op->op) {
        ...
        case GGML_OP_PENALTIES:
            return op->src[0]->type == GGML_TYPE_F32 &&  // logits
                   op->src[1]->type == GGML_TYPE_I32 &&  // history
                   op->src[2]->type == GGML_TYPE_I32;    // n_history
       ...
    }
```

#### Add the operation
First add the operation to the operations header file ggml/src/ggml-metal/ggml-metal-op.h:
```c++
int ggml_metal_op_penalties         (ggml_metal_op_t ctx, int idx);
```
And then add a case to the ggml_metal_op_encode_impl function in
```c++
static int ggml_metal_op_encode_impl(ggml_metal_op_t ctx, int idx) {
    ...
    switch (node->op) {
        ...
        case GGML_OP_PENALTIES:
            {
                n_fuse = ggml_metal_op_penalties(ctx, idx);
            } break;
        ...
    }
```
And we add this function to the same file:
```c++
int ggml_metal_op_penalties(ggml_metal_op_t ctx, int idx) {
    ...
}
```
And the kernel itself is in ggml/src/ggml-metal/ggml-metal.metal:
```metal
kernel void kernel_penalties_f32(
        constant ggml_metal_kargs_penalties & args,
        device const float  * logits,        // src[0] - logits to penalize
        device const int    * history,       // src[1] - token history
        device const int    * n_history_ptr, // src[2] - number of valid tokens in history
        device       float  * dst,           // output - penalized logits
        uint tpig[[thread_position_in_grid]]) {
    ...
}
```
