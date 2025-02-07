## CUDA (Compute Unified Device Architecture)
The motivation for this page is that I want to get a basic understanding of
how GPUs work from a programming perspective. I can understand that there are
libraries like Torch and TensorFlow that can do this for us but is just seems
like magic to me at the moment.

CUDA is a parallel computing platform and programming model developed by Nvidia
for general computing on its own GPUs (graphics processing units).

### CUDA Ecosystem
```
Dev tools:     NVIDIA SMI  Data Center GPU Mgr  GPU REST Engine

Libraries:     cuBlas   cuFFT  cuSPARSE  cuSOLVER  AGM-X
               Thrust   CUB    cuDNN     cuRand    NCCL

Compilers:     nvcc,nvc  CUDA-GDB  NVIDIA Nsight NVIDIA Visual Profiler PAPI CUDA
               nvc++
               nvfortran

Programming:   CUDA       OpenMP API   OpenACC   OpenCL   PyCUDA
models

Drivers:       Linux and Windows device drivers and runtime (no mac?)
```

`CUB` (CUDA UnBound) is a library of high-performance primitives for CUDA.
`AMG-X` (Adaptive General Matrix eXponentiation)
`NCCL` (NVIDIA Collective Communications Library) is a library that provides
`multi-GPU` and `multi-node` collective communication primitives.

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

### Compilation units
A CUDA program would have a `.cu` suffix. The nvcc compiler will separate out
the host (CPU) code from the device (GPU) code. The host code is compiled by
the host compiler (gcc, clang, etc.) and the device code is compiled into an
intermediate representation (PTX) which is later converted into binary code
for the GPU.

The host will call code on the GPU using something that is called kernel calls.
A kernel is a function that is executed on the GPU.  Each kernel functions runs
the same code but on different data.

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

There is an example in [array_add](../gpu/cuda/src/array_add.cu) that shows the
above example.

### Parallel Thread Execution (PTX)
When you compile CUDA code with nvcc, the device code doesn't immediately get
translated to machine code. Instead, it first gets compiled to this intermediate
PTX format. The abstraction allows CUDA code to be compiled and then later
be translated into the binary code of a specific GPU.

### CUBIN (CUDA Binary)
Is an ELF-formatted file. THis contains CUDA executable code sections and
sections containing symbols, relocators, debug info. nvcc embeds cubin files
into the host executable file.

### nvcc
nvcc is the CUDA compiler driver (think gcc or clang). It is used to compile
CUDA programs. nvcc accepts a range of conventional compiler options, such as
for defining macros and include/library paths, and for steering the compilation
process. nvcc also accepts a range of CUDA-specific options for defining the
virtual architecture targeting which the CUDA program is compiled, and for
defining the memory model used, and for steering the compilation process.

### Questions
So when we are going to train a large language model on a GPU we first need to
load the model weights into the host's memory, then copy them over to the GPU's
memory, and then call a kernel function to execute. Is this how it is done in
practice or are there ways to avoid the memory copying?

When training large language models (or any deep learning models) on GPUs, the
model's weights and the data do need to reside in the GPU's memory. However, the
process is a bit more nuanced than just loading everything into the host's
memory and then copying it to the GPU's memory. Here's a breakdown of how it
typically works in practice:

Model Initialization:

When you initialize a model using deep learning frameworks like TensorFlow or
PyTorch, and you've set the device to a GPU, the model's weights are often
directly initialized in the GPU's memory. There's no need to first initialize
them on the CPU and then transfer them.
When you instruct these frameworks to initialize tensors (or model parameters)
on the GPU, a series of steps occur:

* The framework communicates with the GPU through a driver API (e.g., CUDA for
NVIDIA GPUs).
* Memory on the GPU is allocated to store the tensor.
* Initialization operations (like random number generation for weight
initialization) are executed as GPU kernels. These operations fill the allocated
memory with the initial values.

The key takeaway is that these operations occur directly on the GPU without the
need for an intermediary step on the CPU.


Data Loading and Batching:


Training data is usually read in batches. Instead of loading the entire dataset
into the host's memory and then transferring it to the GPU, data is typically
loaded batch-by-batch. Each batch is transferred to the GPU just before it's
needed for training.
Modern deep learning frameworks and data loaders handle this process
efficiently, often using asynchronous operations to overlap data loading on the
CPU with computation on the GPU.

Once the model's weights are on the GPU, they typically stay there throughout
the training process. Forward passes, backward passes, and weight updates all
happen on the GPU. The weights aren't constantly moved back and forth between
the host and the GPU.
It's only if you need to save the model's weights or inspect them on the CPU
that you'd transfer them back to the host's memory.

### Streams
A stream in CUDA is a sequence of commands that execute in order. It is like a
queue of commands that are executed one after the other. Now, what I've been 
doing in my examples is just using synchronous commands, like `cudaMemcpy` and
and not specifying a stream when calling a kernel. When we call a function
like `cudaMemcpy` this is a synchronous operation, the host program will not
progress (it will block) until the entire memory transfer is complete.
When we call `cudaMemcpy` this function is added to the default CUDA stream. The
default stream operations are all executed in the order they are called/added.
In this wasy `cudaMemcpy` acts as a synchronization point between the host and
the device.

When using streams we use the async version of memcpy, like `cudaMemcpyAsync`,
and for launching a kernel we specify a stream as an argument.

An example can be found in [streams.cu](../gpu/cuda/src/streams.cu).

###  CMAKE_CUDA_FLAGS
This is a cmake variable that is passed to `nvcc` when compiling CUDA code.

#### FASTFP16_AVAILABLE
This flag tells nvcc to enable the FP16 code paths in the CUDA code.
```console
$ cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_FLAGS="-DFASTFP16_AVAILABLE"
```
