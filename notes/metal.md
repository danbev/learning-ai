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

kernel void simpleMultiply(const device float* input [[buffer(0)]],
                           device float* output [[buffer(1)]],
                           uint id [[thread_position_in_grid]]) {
    output[id] = input[id] * 2.0;
}
```

Shaders are compiled into a `.metallib` file which is then loaded by the application.

### Parameter syntax
These are specified with double brackets which are called attribute specifiers which provide
extra information to the compiler. This is similar to Rust's attributes which are specified
using `#[...]`.

Lets look at the `simpleMultiply` kernel:
```c++
kernel void simpleMultiply(const device float* input [[buffer(0)]],
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
