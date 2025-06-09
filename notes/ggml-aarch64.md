## ggml-cpu-aarch64
The `aarch64` in this file name is a specific name for ARM's 64-bit architecture,
which is commonly used in mobile devices and some servers. Also I have this on
my MacBook Pro M3 which uses the Apple Silicon chip which uses aarch64.

There is an interesting [discussion](https://github.com/ggml-org/llama.cpp/pull/13720#issuecomment-2912509254)
about the naming of this file and it is in the process of being renamed.

In src/ggml-cpu/ggml-cpu.cpp we have:
```c++
#include "ggml-backend.h"                                                          
#include "ggml-backend-impl.h"                                                     
#include "ggml-cpu.h"                                                           
#include "ggml-cpu-aarch64.h"                                                   
#include "ggml-cpu-traits.h"                                                    
#include "ggml-impl.h"                                                             
#include "amx/amx.h"
```
And ggml-cpu-aarch64.h looks like this:
```c++
#pragma once

#include "ggml-cpu-traits.h"
#include "ggml.h"


// GGML internal header

ggml_backend_buffer_type_t ggml_backend_cpu_aarch64_buffer_type(void);
```

Now when we compile all cpu variants will be compiled, including aarch64:
```console
[  9%] Building C object src/CMakeFiles/ggml-cpu.dir/ggml-cpu/ggml-cpu.c.o
[ 10%] Building CXX object src/CMakeFiles/ggml-cpu.dir/ggml-cpu/ggml-cpu.cpp.o
[ 11%] Building CXX object src/CMakeFiles/ggml-cpu.dir/ggml-cpu/ggml-cpu-aarch64.cpp.o
[ 12%] Building CXX object src/CMakeFiles/ggml-cpu.dir/ggml-cpu/ggml-cpu-hbm.cpp.o
[ 13%] Building C object src/CMakeFiles/ggml-cpu.dir/ggml-cpu/ggml-cpu-quants.c.o
[ 14%] Building CXX object src/CMakeFiles/ggml-cpu.dir/ggml-cpu/ggml-cpu-traits.cpp.o
[ 15%] Building CXX object src/CMakeFiles/ggml-cpu.dir/ggml-cpu/amx/amx.cpp.o
[ 16%] Building CXX object src/CMakeFiles/ggml-cpu.dir/ggml-cpu/amx/mmq.cpp.o
[ 17%] Building CXX object src/CMakeFiles/ggml-cpu.dir/ggml-cpu/binary-ops.cpp.o
[ 18%] Building CXX object src/CMakeFiles/ggml-cpu.dir/ggml-cpu/unary-ops.cpp.o
[ 19%] Building CXX object src/CMakeFiles/ggml-cpu.dir/ggml-cpu/vec.cpp.o
[ 20%] Building CXX object src/CMakeFiles/ggml-cpu.dir/ggml-cpu/ops.cpp.o
[ 21%] Linking CXX shared library libggml-cpu.so
[ 21%] Built target ggml-cpu
```

