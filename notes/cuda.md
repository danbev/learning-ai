## CUDA (Compute Unified Device Architecture)
CUDA is a parallel computing platform and programming model developed by Nvidia
for general computing on its own GPUs (graphics processing units).

### Colab
For this I chose a runtime with a A100 GPU which is an NVIDIA GPU which uses
their Ampere architecture. The A100 is a high-end GPU that is used for
accelerating machine learning and data science workloads.
```
!nvcc --version

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Sep_21_10:33:58_PDT_2022
Cuda compilation tools, release 11.8, V11.8.89
Build cuda_11.8.r11.8/compiler.31833905_0
```

We also need a notebook extension to run nvcc in a notebook:
```
!pip install git+https://github.com/andreinechaev/nvcc4jupyter.git
Collecting git+https://github.com/andreinechaev/nvcc4jupyter.git
  Cloning https://github.com/andreinechaev/nvcc4jupyter.git to /tmp/pip-req-build-yn5nyi2b
  Running command git clone --filter=blob:none --quiet https://github.com/andreinechaev/nvcc4jupyter.git /tmp/pip-req-build-yn5nyi2b
  Resolved https://github.com/andreinechaev/nvcc4jupyter.git to commit 0a71d56e5dce3ff1f0dd2c47c29367629262f527
  Preparing metadata (setup.py) ... done
Building wheels for collected packages: NVCCPlugin
  Building wheel for NVCCPlugin (setup.py) ... done
  Created wheel for NVCCPlugin: filename=NVCCPlugin-0.0.2-py3-none-any.whl size=4295 sha256=faeb31d6421f878d57c1eac339673df8030789df132c7757c4710b6b15fb139f
  Stored in directory: /tmp/pip-ephem-wheel-cache-frkomf2o/wheels/a8/b9/18/23f8ef71ceb0f63297dd1903aedd067e6243a68ea756d6feea
Successfully built NVCCPlugin
Installing collected packages: NVCCPlugin
Successfully installed NVCCPlugin-0.0.2
```
And load the plugin using the following command:
```
%load_ext nvcc_plugin
created output directory at /content/src
Out bin /content/result.out
```

We can compile a simple C++ program to see that is works:
```
%%cu
#include <iostream>
    int main() {
    std::cout << "Compiling a C++ program in a Notbook\n";
    return 0;
}
```
The `%%cu` magic command is used to compile the cell using nvcc. To see all
the available magics use `%lsmagic`.

A CUDA program would have a `.cu` suffix. The nvcc compiler will separate out
the host (CPU) code from the device (GPU) code. The host code is compiled by
the host compiler (gcc, clang, etc.) and the device code is compiled into an
intermediate representation (PTX) which is later converted into binary code
for the GPU.

So the host code is compiled by the host and the device code is compiled on
the GPU. The host will call code on the GPU using something that is called
kernel calls. A kernel is a function that is executed on the GPU.
Each kernel functions runs the same code but on different data.

```c++
__global__ void add_arrays(int *a, int *b, int *c, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        c[tid] = a[tid] + b[tid];
    }
}
```
The `__global__` keyword is used to indicate that this function is a kernel
function. The `add_arrays` function will be executed on the GPU. The `tid`
variable is the thread id. The `blockIdx.x` is the block id and the
`threadIdx.x` is the thread id within the block. The `blockDim.x` is the
number of threads in a block. The `blockIdx.x * blockDim.x + threadIdx.x`
is the global thread id.

There is an example in [cuda_exploration](../cuda/cuda_exploration.cu) that
shows the above example.

### Parallel Thread Execution (PTX)
When you compile CUDA code with nvcc, the device code doesn't immediately get
translated to machine code. Instead, it first gets compiled to this intermediate
PTX format. The abstraction allows CUDA code to be compiled and then later
be translated into the binary code of a specific GPU.

### nvcc
nvcc is the CUDA compiler driver. It is used to compile CUDA programs. nvcc
accepts a range of conventional compiler options, such as for defining macros
and include/library paths, and for steering the compilation process. nvcc also
accepts a range of CUDA-specific options for defining the virtual architecture
targeting which the CUDA program is compiled, and for defining the memory model
used, and for steering the compilation process.
