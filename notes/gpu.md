## Graphics Processing Unit (GPU)
Keep in mind that CPU's also have vector hardware like MMX, SIMD, and SSE, but
these are less powerful than GPU's. GPU's are designed to do many calculations
in parallel, which is why they are so good at matrix multiplication. Moving
memory is also an important aspect of GPU's, so they have a lot of memory
bandwidth. GPU's are also designed to be very power efficient, so they are
designed to do a lot of work with a small amount of power.

```
+-----------------------------------+
|          Host System              |
|                                   |
|    +---------------------------+  |
|    |       Main Memory         |  |
|    +---------------------------+  |
|                |                  |
|                | PCIe Bus         |
|                |                  |
+----------------|------------------+
                 |
+----------------|------------------+
|                V                  |
|        GPU Architecture           |
|                                   |
|    +---------------------------+  |
|    | SM1  | SM2  | ... | SMn   |  |    SM = Streaming Multiprocessor
|    +---------------------------+  |
|    |   L2 Cache                |  |
|    +---------------------------+  |
|    |  Global Memory/VRAM/      |  |
|    |  GPU RAM/DRAM             |  |
|    +---------------------------+  |
+-----------------------------------+


+-----------------------------------+
|       Streaming Multiprocessor    |
|                                   |
|    +---------------------------+  |
|    |       Shared Memory       |  |
|    +---------------------------+  |
|    |                           |  |
|    |      CUDA Cores /         |  |
|    |      Processing Units     |  |
|    |                           |  |
|    |  [Registers (per core)]   |  |
|    +---------------------------+  |
|    |       L1 Cache            |  |
|    +---------------------------+  |
|                                   |
+-----------------------------------+
```
For example the GPU I have has got 12 GB of Global Memory (VRAM). And it has
the following number of cores:
```console
$ nvidia-settings -q TotalDedicatedGPUMemory -t
12282
$ nvidia-settings -q CUDACores -t
5888
```
And it looks like this GPU has 46 SMs which would give 128 cores per SM(
(5888/46) = 128 cores/SM).

I've read that the L1 Cache, on-chip SRAM, for my card is 128 KB (per SM).
128 * 46 = 5888 KB = 5.75 MB. So the L1 Cache is 5.75 MB in total for my card.

Lets think about input data to a process, for example an array that we want to
sum all the element of. 

1) Input data is copied from the host's main memory to the GPU's global memory. 

2) Data needed for processing by a specific set of threads within an SM is then
copied to the SM's shared memory. This memory is shared among all cores in the
streaming multiprocessor and this is something that I think the programmer has
to do explicitly.

For data that might not fit into the SMs shared memory, it will be accessed from
the GPUs global memory, and this data will be cached in the SMs L1 cache. Note
that the L1 cache is not involved in the SMs shared memory accesses.

3) Data in the shared memory can be read and written by the cores on the same
SM, for examples this can be temporary results of computations.

4) The results of the computations are then copied back to the GPUs global
memory.

5) Finally, the results are copied back to the host's main memory.


The registers on the SMs are per GPU thread which is not the case for a CPU
where the registers are per CPU core. Every core in an SM has a register file,
which is a collection of registers used exclusively by that core. The term
register "file" has always confused me but in computing "file" has older roots,
where it was used to describe a collection of related data. In early computing,
a "file" could refer to a collection of data cards, for example. So a register
file is a collection of registers which are memory locations on the chip itself.

There are about 255 registers per thread on modern GPUs and they are typically
32 bits registers.

Threads are executed by GPU cores and they execute the kernel. Just a note about
the name "kernel" as the first thing I thought about was the linux kernel or
something like that. In this case I think it comes from that what we want
executed is a small portion of our program, a part of it that we want to
optimize (and which can benefit from parallelization). So this is the "kernel",
it is the "core" computation unit of our program, or the "essential" part that
is being computed.

Each thread has its own program counter, registers, stack and local memory (off
chip so is slower than registers). But individual threads are not the unit of
execution on the cores, instead something called a warp is the unit of
execution. A warp is a collection of 32 threads that execute the same
instruction in a SIMD fashion. So all the threads in a warp execute the same
instruction at the same time, but on different data. 

A block is a group of threads that execute the same kernel and can communicate
with each other via shared memory. This means that a block is specific to an
SM as the shared memory is per SM.

A grid is a collection of blocks that execute the same kernel. This grid is
distributed across the SMs of the GPU. So while an individual block is specific
to an SM, other blocks in the same grid can be on other SMs.

Each SM can execute multiple blocks at the same time. And each SM has multiple
/many cores, and each of these cores can execute one or more threads at the
same time.

We specify the number of blocks, which is the same thing as the size of the
grid, when we create the kernel. The GPUs scheduler will then distribute the
blocks across the available SMs. So each block is assigned to an SM and the SM
is responsible for executing the blocks  assigned to it, managing the warps and
threads within those blocks.

The cores are what actually do the work and execute the threads. Each core can
execute one thread at a time.

There is shared memory within a thread block/workgroup which is much faster than
global memory (~10-20 cycles). The size of this is memory is limited and typically
around 48-164 KB per block. This needs to be explicitely managed by the programmer.
Global memory is the slowest memory on the GPU and is typically around 400-600 cycles.



So, we have something like this:
```
Registers    : 32 bit, ~1 cycle access time, ~255 registers per thread, 255*32 bits = ~8160 bytes
Shared memory: ~10-20 cycles access time, ~48-164 KB per block
Global memory: ~400-600 cycles access time, ~GBs in size
```

Now, lets say that we want to multiply a 4096x4096 matrix. One row will be:
```
row size: 4096 elements * 4 bytes/element = 16,384 bytes = 16 KB
```
And one thread will need to access one row and one column to compute one element of
the result matrix. So one row and one column will be:
```
one row = 16 KB
one column = 16 KB
Total: 32 KB
```
And we cannot fit 32 KB into the threads registers which can be around 1KB.
So instead of one thread handling one complete dot product, using one row and
one column to compute one output result, threads in a block can cooperate and compute
a `tile` of the output matrix.

So we have our input row and column in global memory:
```
row   : 16 KB 4096 elements
column: 16 KB 4096 elements
```
And lets say we have a block of 256 threads (16x16=256). And each thread will compute a 16x16 tile
of the input matrices (lets call them A an B), which results in a 16x16 tile of the output matrix C.
```
16x16 A tile = 1KB (16x16*4 bytes/element = 1024)
16x16 B tile = 1KB (16x16*4 bytes/element = 1024)
Total: 2KB in shared memory.

Each thread will get:
- 16 elements (64 bytes) from tile A in shared memory
- 16 elements (64 bytes) from tile B in shared memory
```
So each thread will need to load 64 + 64 = 128 bytes from shared memory to its registers. Then the
thread will compute the dot product of these two 16 element vectors to produce one output element
which will be stored in an accumulator register. So thread one will have the output for the first
element in its accumulator register, thread two will have the output for the second element
and so on. And each thread writes this output to global memory directly which out having to go
through shared memory.

We know from above that 32 KB will not fit into a thread's registers so this will instead be loaded
into the shared memory which can fit 2 KB without any problems.

Each thread's registers will hold:
```
1 accumulator for its output bits   : 4 bytes (32 bits so this fits in one register).
Temporary values for the computation: ~16 bytes
Total per thread: ~20 bytes (not problem to fit within the ~1KB register size)
```
So we will have 256 threads that will run in parallel to compute the output for this matrix
multiplication operation. ALL 256 threads do this `simultaneously`:
```
For k = 0 to 4096 step 16  (because 4096/256 = 16 iterations):
  
  // So the following is processing 0-15
  Thread(0, 0) loads A[tile_row=0, tile_col=0, k_offset] → shared_A[0,0]
  Thread(0, 1) loads A[tile_row=0, tile_col=1, k_offset] → shared_A[0,1]  
  Thread(1, 0) loads A[tile_row=1, tile_col=0, k_offset] → shared_A[1,0]
  ...
  Thread(15, 15) loads A[tile_row=15, tile_col=15, k_offset] → shared_A[15,15]
  
  Thread(0, 0) loads B[k_offset, tile_col=0] → shared_B[0,0]
  Thread(0, 1) loads B[k_offset, tile_col=1] → shared_B[0,1]
  ...
  
  __syncthreads()  // Barrier: wait for ALL threads to finish loading
  
  // PARALLEL: All 256 threads compute simultaneously  
  Thread(0, 0) computes: shared_A[0,:] dot shared_B[:,0] → accumulator(0,0)
  Thread(0, 1) computes: shared_A[0,:] dot shared_B[:,1] → accumulator(0,1)
  Thread(1, 0) computes: shared_A[1,:] dot shared_B[:,0] → accumulator(1,0)
  ...
  Thread(15, 15) computes: shared_A[15,:] dot shared_B[:,15] → accumulator(15,15)
```

### GPU Terminology
```
Metal              CUDA              Vulkan/WebGPU
─────────────────────────────────────────────────────
Thread             Thread            Invocation  
Threadgroup        Thread Block      Workgroup
Grid               Grid              Dispatch
SIMD-group         Warp              Subgroup (32 threads on M3)
```

_wip_

```
+-------------------------------------+
|               Grid                  |
|                                     |
|    +------------+  +------------+   |
|    |   Block 1  |  |   Block 2  |   |
|    |            |  |            |   |
|    | T1 T2 .. Tn|  | T1 T2 .. Tn|   |
|    +------------+  +------------+   |
|    +------------+  +------------+   |
|    |   Block 3  |  |   Block 4  |   |
|    |            |  |            |   |
|    | T1 T2 .. Tn|  | T1 T2 .. Tn|   |
|    +------------+  +------------+   |
|              ...                    |
|    +------------+  +------------+   |
|    |  Block N-1 |  |   Block N  |   |
|    |            |  |            |   |
|    | T1 T2 .. Tn|  | T1 T2 .. Tn|   |
|    +------------+  +------------+   |
|                                     |
+-------------------------------------+
```
So if we launch a kernel with the following configuration and we specify how
many blocks and threads per block we want to use:
```c++
    int thread_blocks = 2;
    int threads_per_block = 4;
    helloWorld<<<thread_blocks, threads_per_block>>>();
```
The we would have something like:
```
Grid:
+---------+---------+
| Block 0 | Block 1 |
+---------+---------+

Each Block (4 Threads):
Block 0: +----+----+----+----+   T0 = thread_id = 0
         | T0 | T1 | T2 | T3 |   T1 = thread_id = 1
         +----+----+----+----+   T2 = thread_id = 2
                                 T3 = thread_id = 3

Block 1: +----+----+----+----+   T0 = thread_id = 0
         | T0 | T1 | T2 | T3 |   T1 = thread_id = 1
         +----+----+----+----+   T2 = thread_id = 2
                                 T3 = thread_id = 3
```
So 8 threads in total will execute the kernel. On thing to notice is that within
a block the thread ids are the same (0, 1, 2, 3 in this case). To get a unique
thread id we also need to take the block into account.

A warp is a group of 32 threads that execute the same instruction at the same
time. So for our above examples this would be something like:
```
Warp Execution (32 Threads per Warp):
+-----+-----+-----+-----+-----+-----+-----+-----+----+ ... +----+
|T0_b₀|T1_b₀|T2_b₀|T3_b₀|T0_b₁|T1_b₁|T2_b₁|T3_b₁| XX | ... | XX |
+-----+-----+-----+-----+-----+-----+-----+-----+----+ ... +----+
   0     1     2     3     4     5     6     7    8    ...   31

XX = inactive threads
```
Each SM can execute multiple blocks at the same time.
The GPU scheduler will distribute the blocks accross the available SMs. So each
block is assigned to an SM and the SM is responsible for executing all the
threads in that block. If we specify more blocks than there are SMs then the
scheduler will queue the blocks and execute as SMs become available.
Inside each block, threads are grouped into warps which contain 32 threads each.

The built-in variable threadIdx is a 3-component vector (x, y, z) that holds the
index of the current thread within the block.
The built-in variable blockIdx is a 3-component vector (x, y, z) that holds the
index of the current block within the grid. blockDim contains the dimensions of
the block which is how many threads there are in each dimension of the block.
If we only have one dimension we can use the x component of the block dimension:
```c++
int threadId = blockIdx.x * blockDim.x + threadIdx.x;
```
So if we access the 5th element in an array then blockIdx.x would be 1, and
blockDim.x would be 4. So the threadId would be 5.

It is also possible to have multiple buses between the GPU and system memory
which is called parallelization. This way the GPU can keep transferring data
from system memory into its own memory while it is processing data.

In most systems, the GPU has its own dedicated memory (VRAM - Video RAM) that
is separate from the system's main RAM. VRAM is optimized for the GPU's
high-throughput, parallel processing needs.

### Memory transfers
Initially, data (like a matrix for multiplication or texture data for rendering)
resides in the system's main memory. To be processed by the GPU, this data needs
to be transferred to the GPU's memory. This transfer typically happens over the
PCIe (Peripheral Component Interconnect Express) bus, a high-speed interface
that connects the GPU to the motherboard and through it to the system memory.

PCIe Bandwidth: The PCIe bus has a certain bandwidth limit, which dictates how
fast data can be transferred between the system memory and the GPU memory. This
transfer can become a bottleneck, especially for large data sets or when
frequent data transfers are required.

Once the data is in the GPU's memory (HBM), data that a kernel operates on has
to be copied over to the SRAM of the GPU (or the SM?). And once the operation is
done the results have to copied back to the GPU's memory (HBM). Now, if we have
multiple kernels that are called in sequence and use the results from another
kernel it can make sense to merge, or fuse, them into a single kernel and avoid
this memory transfer overhead. This is what is known as kernel fusion.

If the results of the GPU's computations need to be used by the CPU or need to
be stored back in the system memory, they are ransferred back via the PCIe bus.

### Single Instruction Multiple Data (SIMD)
This means that all the cores on the GPU are executing the same instruction at
the same time, but on different data. This is why GPU's are so good at matrix
multiplication. The GPU can multiply each row of the first matrix by each column
of the second matrix in parallel. This is called a dot product.
So if we have two matrices, a and b, to produce the matrix c we can see that the
value of each element in C does not depend on any other element in C. This means
that we can calculate each element in C in parallel. All need to have access
to the matrices a an b but they don't depend on any of the other dot products
which allows them to be calculated in parallel.

```
  +-----+-----+       +-----+-----+
  |a₁.b₁|a₁.b₂|       | c₁₁ | C₁₂ |
  +-----+-----+   =   +-----+-----+
  |a₂.b₁|a₂.b₂|       | C₂₁ | C₂₁ |
  +-----+-----+       +-----+-----+
```
In a CPU lets say we can only compute one of the dot products at a time so this
would take four cycles to compute. In a GPU we can compute all four dot products
at the same time so it would only take one cycle to compute. With larger
matrices we can start to see the power of the GPU.

Lets take another example where we want to sum all the elements of an array.
In this case there is a dependency on the computations because we need to
compute the sum of the first two elements before we can compute the sum of the
first three elements. But we can still compute this is parallel by splitting
the array into pairs and computing the sum of each pair in parallel.

Lets first look at doing this sequencially:
```
  +----+   +----+
  | 1  |   | 2  |
  +----+   +----+
    |         | 
    +----+----+
         |
      +-----+   +-----+
      |  3  |   |  3  |
      +-----+   +-----+
         |         |
         +----+----+
              |
           +-----+   +-----+
           |  6  |   |  4  |
           +-----+   +-----+
              |         |
              +----+----+
                    |
                 +-----+
                 |  10 |
                 +-----+
```
But we can also do this in parallel:
```
  +----+   +----+    +----+    +----+
  | 1  |   | 2  |    | 3  |    | 4  |
  +----+   +----+    +----+    +----+
    |         |          |        |
    +----+----+          +---+----+
         |                   |
      +-----+              +-----+
      |  3  |              |  7  |
      +-----+              +-----+
         |                   |
         +----------+--------+
                    |
                 +-----+
                 |  10 |
                 +-----+
```

To illustrate what is happening in the GPU we can look at the following diagram:
```
    +----+   +----+    +----+    +----+
    | 1  |   | 2  |    | 3  |    | 4  |
    +----+   +----+    +----+    +----+
      |        |          |        |
      +--------+          +--------+
            |                 |
       +--------+         +--------+ 
       | GPU 0  |         | GPU 1  |
       +--------+         +--------+
            |                   |
            +----------+--------+
                       |
                 +---------+  
                 | GPU 0   |
                 +---------+  
```
Above the partial output of the operation on GPU 1 is needed by GPU 0 to compute
the final result. This means that this data needs to be shared in memory and
accessed by different GPU cores. In a CPU we might have 4,8, 12, 16, 32, or
perhaps 128 cors but in a GPU we might have 1000's of cores. This means that
sharing memory becomes more difficult and other solutions are needed for a GPU
compared to a CPU.

```
   +-----------------------------------------------------+
   | +------------------------------------------------+
   | |  Registers                                     |
   | +------------------------------------------------+
   | +--------------+ +--------------+ +--------------+
   | |  INT Cores   | |  FP Cores    | |  Tensor Cores|
   | |              | |              | |              |
   | |              | |              | |              |
   | |              | |              | |              |
   | |              | |              | |              |
   | |              | |              | |              |
   | +--------------+ +--------------+ +--------------+
   | +------------------------------------------------+
   | |  L1 Data Cache / Shared Memory                 |
   | +------------------------------------------------+
   +-----------------------------------------------------+
```
The L1 cache/shared memory is how the cores can work together and be able to
share data.

### FP32 Cores
These perform single precision floating point operations.

### FP64 Cores
These perform double precision floating point operations.

### Integer Cores
These perform integer operations like address computations, and that can
execute instructions concurrently with the floating-point math datapath.


### Tensor Cores
Are used to serve tensor operations in AI/ML and are speciallized execution
units for matrix multiplication. The can have modes like INT8 or INT4 precision
which can tolerate quantization and don't required FP16 precision (but can also
use FP16 (half) precision)


### Ray Tracing Cores
These are used to serve hyper-realistic graphics rendering.

### Shaders
Shaders are blocks of code that excute on the GPU, similar to how CUDA kernels.
And in a similar way to CUDA kernels which are defined in a C-like language,
shaders are defined in languages like High Level Shading Language (HLSL) or
OpenGL Shading Language (GLSL).

HLSL is used with DirectX (Microsoft) and primary support is for Window, and has
a c-like syntax. Is often pre-compiled into intermediate bytecode which is then
executed by the GPU.

GLSL is used with OpenGL (Khronos) and is cross-platform and also has a c-like
syntax. Are typically compiled at runtime but it is possible to pre-compile them
as well.



#### Vertex Shaders
For some background to be able to understand what a shader does the
following image is defining a triangle in a program consisting of three points
in 3D space (x, y, z).
```
              +
            /   \
           /     \
          /       \
         +---------+
```

The points are called vertices and the triangle is defined by the three plus
signs above. Apart from the x, y, and z coordinates a vertex can have additional
attributes like color, texture coordinates, and normals.
These are then copied from the main memory by the CPU part of the program,
similar to what is done with CUDA programs.

A vertext shader is a block of code define in HLSL or GLSL that is executed on
the GPU. This program operates on a single vertex and transforms its position
from model space (3D) into screen space (2D). In Model space the origin might be
the middle of the triangle, but in screen space the coordinates typically have
their origin in the top left corner of the screen. 
The vertex shader may also update other vertex attributes.
When this is done the output of each shader (recall that each vertex shader
operation is performed in parallel on a separate core/SM) and the result is
written back to HBM. So the vertices have now been updated and the next shader
can then operate on these updated vertices.

#### Rasterization
This is the process of converting the vertices into fragments. A fragment is
basically a pixel on the screen. So the rasterizer takes the vertices and
interpolates the values of the vertices to determine the values of the pixels
in between the vertices. 
```
              +
            /|||\
           /|||||\
          /|||||||\
         +---------+
```
Is is not really about creating pixels but rather determining which pixels on
the scsreen are covered by the shape formed by the vertices (a triangle in our
case).

#### Fragment Shaders
This shader, which again is a block of code written in HLSL or GLSL, and is
executed per fragment (in parallel). This is where the color of the each pixel
is determined.

#### General compute "shaders"
Initially GPU's were only used for graphics processing and I've read that people
started using shaders to do general purpose computing on the GPU. And that this
later led to the development of general purpose GPU programming languages like
CUDA and OpenCL. 

#### OpenGL Shading Language (GLSL)
Lets take a peak at what a vertex shader could look like in GLSL:
```glsl
#version 330 core               // uses version 3.30, and core (no deprecated features)

// layout specifies the location of the shaders input. `in` means that we are
// specifing an input for this shader. Then we have the type, vec3, which is an
// three dimensional vector. And then we have the name of the variable.
layout (location = 0) in vec3 position;

// uniforms are used to pass data from the OpenGL program (cpu part?) to the
// shaders. mat4 means a 4x4 matrix.
uniform mat4 worldViewProj;

// main function of the shader, the part that will be excuted on a GPU core.
void main() {
    // The following line transforms the vertex position from model space to
    // screen space. position is first extended with one dimention to match the
    // dimension of the matrix.
    gl_Position = worldViewProj * vec4(position, 1.0);
    // gl_Position is a built-in GLSL variable in that is used to store the
    // position of the vertex in clip space.
}
```
So the wordViewProj metrix is a combination of the world, view, and projection
operations (think of this as being 3 separate matrices that have been combinded
into one).
The World transform matrix is what transforms the vertices from model space to
world space. The View transform matrix is what transforms the vertices from
world space to view space. And the Projection transform matrix is what
transforms the vertices from view space to clip space.

The Fragment shader might look something like this:
```glsl
#version 330 core
out vec4 color;

void main() {
    color = vec4(1.0, 0.0, 0.0, 1.0); // Red color
}
```

### Vulkan
Is an API for 3D graphics and compute. It is a low-level API that is designed
for high-performance. There was recently a pull request opened for llama.cpp
which includes support for a [Vulkan] (or rather [Kompute]) backend.
It is simlar to CUDA and Metal but is open-source, and it is cross-platform and
cross-vendor.
It can be used for low resource environments such as mobile phones.

In Vulkan it does not use GLSL or HLSL but instead uses SPIR-V 
(Standard Portable Intermediate Representation) is a bytecode format for
shaders. It is a binary format that is designed to be used with Vulkan and
OpenCL. It is designed to be used with multiple languages and is not tied to a
specific language. 

Vulkan provides very fine grained controll over the GPU and is designed to
allow the programmer to have more control over the GPU. This means that the
programmer has to do more work to get the same result compared to OpenGL. But
this also means that the programmer can get better performance and more control
over the GPU. The disadvantage is that it pretty verbose.

[Vulkan]: https://vulkan.org/
[Kompute]: https://github.com/KomputeProject/kompute

### Kompute
Is built on top of Vulkan and addresses the issue of verbosity as it was observed
that many project ended up writing simliar boilerplate code to use Vulkan in
their project. It is easier to use and highly optimized and mobile friendly
general purpose GPU compute framework (recall that Vulkan has both the graphics
and compute APIs).


### Ubuntu NVIDIA Drivers

```console
$ sudo apt-get install -V nvidia-kernel-source-550-open cuda-drivers-550
Reading package lists... Done
Building dependency tree... Done
Reading state information... Done
The following additional packages will be installed:
   cpp-12 (12.3.0-1ubuntu1~22.04)
   dctrl-tools (2.24-3build2)
   dkms (2.8.7-2ubuntu2.2)
   gcc-12 (12.3.0-1ubuntu1~22.04)
   gcc-12-base:i386 (12.3.0-1ubuntu1~22.04)
   krb5-locales (1.19.2-2ubuntu0.3)
   libasan8 (12.3.0-1ubuntu1~22.04)
   libbsd0:i386 (0.11.5-1)
   libc6:i386 (2.35-0ubuntu3.7)
   libcom-err2:i386 (1.46.5-2ubuntu1.1)
   libcrypt1:i386 (1:4.4.27-1)
   libgcc-12-dev (12.3.0-1ubuntu1~22.04)
   libgcc-s1:i386 (12.3.0-1ubuntu1~22.04)
   libgssapi-krb5-2:i386 (1.19.2-2ubuntu0.3)
   libidn2-0:i386 (2.3.2-2build1)
   libk5crypto3:i386 (1.19.2-2ubuntu0.3)
   libkeyutils1:i386 (1.6.1-2ubuntu3)
   libkrb5-3:i386 (1.19.2-2ubuntu0.3)
   libkrb5support0:i386 (1.19.2-2ubuntu0.3)
   libmd0:i386 (1.0.4-1build1)
   libnsl2:i386 (1.3.0-2build2)
   libnss-nis:i386 (3.1-0ubuntu6)
   libnss-nisplus:i386 (1.3-0ubuntu6)
   libnvidia-cfg1-550 (550.54.15-0ubuntu1)
   libnvidia-common-550 (550.54.15-0ubuntu1)
   libnvidia-compute-550 (550.54.15-0ubuntu1)
   libnvidia-compute-550:i386 (550.54.15-0ubuntu1)
   libnvidia-decode-550 (550.54.15-0ubuntu1)
   libnvidia-decode-550:i386 (550.54.15-0ubuntu1)
   libnvidia-encode-550 (550.54.15-0ubuntu1)
   libnvidia-encode-550:i386 (550.54.15-0ubuntu1)
   libnvidia-extra-550 (550.54.15-0ubuntu1)
   libnvidia-fbc1-550 (550.54.15-0ubuntu1)
   libnvidia-fbc1-550:i386 (550.54.15-0ubuntu1)
   libnvidia-gl-550 (550.54.15-0ubuntu1)
   libnvidia-gl-550:i386 (550.54.15-0ubuntu1)
   libssl3:i386 (3.0.2-0ubuntu1.15)
   libtirpc3:i386 (1.3.2-2ubuntu0.1)
   libtsan2 (12.3.0-1ubuntu1~22.04)
   libunistring2:i386 (1.0-1)
   libvdpau1 (1.4-3build2)
   libx11-6:i386 (2:1.7.5-1ubuntu0.3)
   libxau6:i386 (1:1.0.9-1build5)
   libxcb1:i386 (1.14-3ubuntu3)
   libxdmcp6:i386 (1:1.1.3-0ubuntu5)
   libxext6:i386 (2:1.3.4-1build1)
   libxnvctrl0 (550.54.15-0ubuntu1)
   mesa-vdpau-drivers (23.2.1-1ubuntu3.1~22.04.2)
   nvidia-compute-utils-550 (550.54.15-0ubuntu1)
   nvidia-dkms-550-open (550.54.15-0ubuntu1)
   nvidia-driver-550-open (550.54.15-0ubuntu1)
   nvidia-firmware-550-550.54.15 (550.54.15-0ubuntu1)
   nvidia-kernel-common-550 (550.54.15-0ubuntu1)
   nvidia-prime (0.8.17.1)
   nvidia-settings (550.54.15-0ubuntu1)
   nvidia-utils-550 (550.54.15-0ubuntu1)
   screen-resolution-extra (0.18.2)
   vdpau-driver-all (1.4-3build2)
   xserver-xorg-video-nvidia-550 (550.54.15-0ubuntu1)
Suggested packages:
   gcc-12-locales (12.3.0-1ubuntu1~22.04)
   cpp-12-doc (12.3.0-1ubuntu1~22.04)
   debtags (2.1.5)
   menu (2.1.47ubuntu4)
   gcc-12-multilib (12.3.0-1ubuntu1~22.04)
   gcc-12-doc (12.3.0-1ubuntu1~22.04)
   glibc-doc:i386
   locales:i386
   krb5-doc:i386
   krb5-user:i386 (1.19.2-2ubuntu0.3)
   nvidia-driver-550
   libvdpau-va-gl1 (0.4.2-1build2)
The following packages will be REMOVED:
   libnvidia-common-550-server (550.54.15-0ubuntu0.22.04.2)
The following NEW packages will be installed:
   cpp-12 (12.3.0-1ubuntu1~22.04)
   cuda-drivers-550 (550.54.15-1)
   dctrl-tools (2.24-3build2)
   dkms (2.8.7-2ubuntu2.2)
   gcc-12 (12.3.0-1ubuntu1~22.04)
   gcc-12-base:i386 (12.3.0-1ubuntu1~22.04)
   krb5-locales (1.19.2-2ubuntu0.3)
   libasan8 (12.3.0-1ubuntu1~22.04)
   libbsd0:i386 (0.11.5-1)
   libc6:i386 (2.35-0ubuntu3.7)
   libcom-err2:i386 (1.46.5-2ubuntu1.1)
   libcrypt1:i386 (1:4.4.27-1)
   libgcc-12-dev (12.3.0-1ubuntu1~22.04)
   libgcc-s1:i386 (12.3.0-1ubuntu1~22.04)
   libgssapi-krb5-2:i386 (1.19.2-2ubuntu0.3)
   libidn2-0:i386 (2.3.2-2build1)
   libk5crypto3:i386 (1.19.2-2ubuntu0.3)
   libkeyutils1:i386 (1.6.1-2ubuntu3)
   libkrb5-3:i386 (1.19.2-2ubuntu0.3)
   libkrb5support0:i386 (1.19.2-2ubuntu0.3)
   libmd0:i386 (1.0.4-1build1)
   libnsl2:i386 (1.3.0-2build2)
   libnss-nis:i386 (3.1-0ubuntu6)
   libnss-nisplus:i386 (1.3-0ubuntu6)
   libnvidia-cfg1-550 (550.54.15-0ubuntu1)
   libnvidia-common-550 (550.54.15-0ubuntu1)
   libnvidia-compute-550 (550.54.15-0ubuntu1)
   libnvidia-compute-550:i386 (550.54.15-0ubuntu1)
   libnvidia-decode-550 (550.54.15-0ubuntu1)
   libnvidia-decode-550:i386 (550.54.15-0ubuntu1)
   libnvidia-encode-550 (550.54.15-0ubuntu1)
   libnvidia-encode-550:i386 (550.54.15-0ubuntu1)
   libnvidia-extra-550 (550.54.15-0ubuntu1)
   libnvidia-fbc1-550 (550.54.15-0ubuntu1)
   libnvidia-fbc1-550:i386 (550.54.15-0ubuntu1)
   libnvidia-gl-550 (550.54.15-0ubuntu1)
   libnvidia-gl-550:i386 (550.54.15-0ubuntu1)
   libssl3:i386 (3.0.2-0ubuntu1.15)
   libtirpc3:i386 (1.3.2-2ubuntu0.1)
   libtsan2 (12.3.0-1ubuntu1~22.04)
   libunistring2:i386 (1.0-1)
   libvdpau1 (1.4-3build2)
   libx11-6:i386 (2:1.7.5-1ubuntu0.3)
   libxau6:i386 (1:1.0.9-1build5)
   libxcb1:i386 (1.14-3ubuntu3)
   libxdmcp6:i386 (1:1.1.3-0ubuntu5)
   libxext6:i386 (2:1.3.4-1build1)
   libxnvctrl0 (550.54.15-0ubuntu1)
   mesa-vdpau-drivers (23.2.1-1ubuntu3.1~22.04.2)
   nvidia-compute-utils-550 (550.54.15-0ubuntu1)
   nvidia-dkms-550-open (550.54.15-0ubuntu1)
   nvidia-driver-550-open (550.54.15-0ubuntu1)
   nvidia-firmware-550-550.54.15 (550.54.15-0ubuntu1)
   nvidia-kernel-common-550 (550.54.15-0ubuntu1)
   nvidia-kernel-source-550-open (550.54.15-0ubuntu1)
   nvidia-prime (0.8.17.1)
   nvidia-settings (550.54.15-0ubuntu1)
   nvidia-utils-550 (550.54.15-0ubuntu1)
   screen-resolution-extra (0.18.2)
   vdpau-driver-all (1.4-3build2)
   xserver-xorg-video-nvidia-550 (550.54.15-0ubuntu1)
0 upgraded, 61 newly installed, 1 to remove and 3 not upgraded.
Need to get 51,7 MB/355 MB of archives.
After this operation, 1 176 MB of additional disk space will be used.
Do you want to continue? [Y/n] y
Get:1 file:/var/cuda-repo-ubuntu2204-12-4-local  libnvidia-common-550 550.54.15-0ubuntu1 [17,1 kB]
Get:3 http://se.archive.ubuntu.com/ubuntu jammy-updates/main amd64 libasan8 amd64 12.3.0-1ubuntu1~22.04 [2 442 kB]
Get:2 http://se.archive.ubuntu.com/ubuntu jammy-updates/main amd64 cpp-12 amd64 12.3.0-1ubuntu1~22.04 [10,8 MB]
Get:4 file:/var/cuda-repo-ubuntu2204-12-4-local  libnvidia-compute-550 550.54.15-0ubuntu1 [49,5 MB]
Get:5 http://se.archive.ubuntu.com/ubuntu jammy-updates/main amd64 libtsan2 amd64 12.3.0-1ubuntu1~22.04 [2 477 kB]
Get:6 http://se.archive.ubuntu.com/ubuntu jammy-updates/main amd64 libgcc-12-dev amd64 12.3.0-1ubuntu1~22.04 [2 618 kB]
Get:8 http://se.archive.ubuntu.com/ubuntu jammy/main amd64 dctrl-tools amd64 2.24-3build2 [66,9 kB]
Get:9 http://se.archive.ubuntu.com/ubuntu jammy-updates/main amd64 dkms all 2.8.7-2ubuntu2.2 [70,1 kB]
Get:10 http://se.archive.ubuntu.com/ubuntu jammy-updates/main i386 gcc-12-base i386 12.3.0-1ubuntu1~22.04 [20,1 kB]
Get:11 http://se.archive.ubuntu.com/ubuntu jammy-updates/main i386 libgcc-s1 i386 12.3.0-1ubuntu1~22.04 [64,0 kB]
Get:12 http://se.archive.ubuntu.com/ubuntu jammy/main i386 libcrypt1 i386 1:4.4.27-1 [97,2 kB]
Get:13 http://se.archive.ubuntu.com/ubuntu jammy-updates/main i386 libc6 i386 2.35-0ubuntu3.7 [3 013 kB]
Get:14 file:/var/cuda-repo-ubuntu2204-12-4-local  libnvidia-gl-550 550.54.15-0ubuntu1 [136 MB]
Get:15 file:/var/cuda-repo-ubuntu2204-12-4-local  nvidia-kernel-source-550-open 550.54.15-0ubuntu1 [4 864 kB]
Get:16 file:/var/cuda-repo-ubuntu2204-12-4-local  nvidia-firmware-550-550.54.15 550.54.15-0ubuntu1 [36,8 MB]
Get:17 file:/var/cuda-repo-ubuntu2204-12-4-local  nvidia-kernel-common-550 550.54.15-0ubuntu1 [109 kB]
Get:18 file:/var/cuda-repo-ubuntu2204-12-4-local  nvidia-dkms-550-open 550.54.15-0ubuntu1 [16,8 kB]
Get:19 file:/var/cuda-repo-ubuntu2204-12-4-local  libnvidia-extra-550 550.54.15-0ubuntu1 [71,1 kB]
Get:20 file:/var/cuda-repo-ubuntu2204-12-4-local  nvidia-compute-utils-550 550.54.15-0ubuntu1 [118 kB]
Get:21 file:/var/cuda-repo-ubuntu2204-12-4-local  libnvidia-decode-550 550.54.15-0ubuntu1 [1 783 kB]
Get:22 file:/var/cuda-repo-ubuntu2204-12-4-local  libnvidia-encode-550 550.54.15-0ubuntu1 [100 kB]
Get:23 file:/var/cuda-repo-ubuntu2204-12-4-local  nvidia-utils-550 550.54.15-0ubuntu1 [494 kB]
Get:24 file:/var/cuda-repo-ubuntu2204-12-4-local  libnvidia-cfg1-550 550.54.15-0ubuntu1 [145 kB]
Get:25 file:/var/cuda-repo-ubuntu2204-12-4-local  xserver-xorg-video-nvidia-550 550.54.15-0ubuntu1 [1 534 kB]
Get:26 file:/var/cuda-repo-ubuntu2204-12-4-local  libnvidia-fbc1-550 550.54.15-0ubuntu1 [54,9 kB]
Get:27 file:/var/cuda-repo-ubuntu2204-12-4-local  nvidia-driver-550-open 550.54.15-0ubuntu1 [14,4 kB]
Get:28 file:/var/cuda-repo-ubuntu2204-12-4-local  cuda-drivers-550 550.54.15-1 [2 546 B]
Get:29 http://se.archive.ubuntu.com/ubuntu jammy-updates/main i386 libcom-err2 i386 1.46.5-2ubuntu1.1 [9 614 B]
Get:30 http://se.archive.ubuntu.com/ubuntu jammy-updates/main i386 libkrb5support0 i386 1.19.2-2ubuntu0.3 [36,9 kB]
Get:31 http://se.archive.ubuntu.com/ubuntu jammy-updates/main i386 libk5crypto3 i386 1.19.2-2ubuntu0.3 [91,0 kB]
Get:32 http://se.archive.ubuntu.com/ubuntu jammy/main i386 libkeyutils1 i386 1.6.1-2ubuntu3 [10,7 kB]
Get:33 http://se.archive.ubuntu.com/ubuntu jammy-updates/main i386 libssl3 i386 3.0.2-0ubuntu1.15 [1 946 kB]
Get:34 file:/var/cuda-repo-ubuntu2204-12-4-local  libnvidia-compute-550 550.54.15-0ubuntu1 [51,0 MB]
Get:35 file:/var/cuda-repo-ubuntu2204-12-4-local  libnvidia-decode-550 550.54.15-0ubuntu1 [2 129 kB]   
Get:36 file:/var/cuda-repo-ubuntu2204-12-4-local  libnvidia-encode-550 550.54.15-0ubuntu1 [107 kB]     
Get:37 file:/var/cuda-repo-ubuntu2204-12-4-local  libnvidia-fbc1-550 550.54.15-0ubuntu1 [59,7 kB]      
Get:38 http://se.archive.ubuntu.com/ubuntu jammy-updates/main i386 libkrb5-3 i386 1.19.2-2ubuntu0.3 [403 kB]
Get:39 http://se.archive.ubuntu.com/ubuntu jammy-updates/main i386 libgssapi-krb5-2 i386 1.19.2-2ubuntu0.3 [154 kB]
Get:40 file:/var/cuda-repo-ubuntu2204-12-4-local  libnvidia-gl-550 550.54.15-0ubuntu1 [17,8 MB]        
Get:41 file:/var/cuda-repo-ubuntu2204-12-4-local  libxnvctrl0 550.54.15-0ubuntu1 [21,3 kB]             
Get:42 file:/var/cuda-repo-ubuntu2204-12-4-local  nvidia-settings 550.54.15-0ubuntu1 [947 kB]          
Get:43 http://se.archive.ubuntu.com/ubuntu jammy-updates/main i386 libtirpc3 i386 1.3.2-2ubuntu0.1 [92,8 kB]
Get:44 http://se.archive.ubuntu.com/ubuntu jammy/main i386 libnsl2 i386 1.3.0-2build2 [46,2 kB]        
Get:45 http://se.archive.ubuntu.com/ubuntu jammy/main i386 libmd0 i386 1.0.4-1build1 [23,8 kB]         
Get:46 http://se.archive.ubuntu.com/ubuntu jammy/main i386 libbsd0 i386 0.11.5-1 [48,3 kB]             
Get:47 http://se.archive.ubuntu.com/ubuntu jammy/main i386 libunistring2 i386 1.0-1 [554 kB]           
Get:48 http://se.archive.ubuntu.com/ubuntu jammy/main i386 libidn2-0 i386 2.3.2-2build1 [71,1 kB]      
Get:49 http://se.archive.ubuntu.com/ubuntu jammy/main i386 libxau6 i386 1:1.0.9-1build5 [7 924 B]      
Get:50 http://se.archive.ubuntu.com/ubuntu jammy/main i386 libxdmcp6 i386 1:1.1.3-0ubuntu5 [11,4 kB]   
Get:51 http://se.archive.ubuntu.com/ubuntu jammy/main i386 libxcb1 i386 1.14-3ubuntu3 [55,4 kB]        
Get:52 http://se.archive.ubuntu.com/ubuntu jammy-updates/main i386 libx11-6 i386 2:1.7.5-1ubuntu0.3 [694 kB]
Get:53 http://se.archive.ubuntu.com/ubuntu jammy/main i386 libxext6 i386 2:1.3.4-1build1 [34,8 kB]     
Get:54 http://se.archive.ubuntu.com/ubuntu jammy-updates/main amd64 krb5-locales all 1.19.2-2ubuntu0.3 [11,7 kB]
Get:55 http://se.archive.ubuntu.com/ubuntu jammy/main i386 libnss-nis i386 3.1-0ubuntu6 [28,2 kB]      
Get:56 http://se.archive.ubuntu.com/ubuntu jammy/main i386 libnss-nisplus i386 1.3-0ubuntu6 [23,7 kB]  
Get:57 http://se.archive.ubuntu.com/ubuntu jammy/main amd64 libvdpau1 amd64 1.4-3build2 [27,0 kB]      
Get:58 http://se.archive.ubuntu.com/ubuntu jammy-updates/main amd64 mesa-vdpau-drivers amd64 23.2.1-1ubuntu3.1~22.04.2 [3 820 kB]
Get:7 http://se.archive.ubuntu.com/ubuntu jammy-updates/main amd64 gcc-12 amd64 12.3.0-1ubuntu1~22.04 [21,7 MB]
Get:59 http://se.archive.ubuntu.com/ubuntu jammy/main amd64 nvidia-prime all 0.8.17.1 [9 956 B]        
Get:60 http://se.archive.ubuntu.com/ubuntu jammy/main amd64 screen-resolution-extra all 0.18.2 [4 396 B]
Get:61 http://se.archive.ubuntu.com/ubuntu jammy/main amd64 vdpau-driver-all amd64 1.4-3build2 [4 510 B]
Fetched 51,7 MB in 3s (20,0 MB/s)                                                                      
Extracting templates from packages: 100%
Preconfiguring packages ...
(Reading database ... 214842 files and directories currently installed.)
Removing libnvidia-common-550-server (550.54.15-0ubuntu0.22.04.2) ...
Selecting previously unselected package cpp-12.
(Reading database ... 214836 files and directories currently installed.)
Preparing to unpack .../00-cpp-12_12.3.0-1ubuntu1~22.04_amd64.deb ...
Unpacking cpp-12 (12.3.0-1ubuntu1~22.04) ...
Selecting previously unselected package libasan8:amd64.
Preparing to unpack .../01-libasan8_12.3.0-1ubuntu1~22.04_amd64.deb ...
Unpacking libasan8:amd64 (12.3.0-1ubuntu1~22.04) ...
Selecting previously unselected package libtsan2:amd64.
Preparing to unpack .../02-libtsan2_12.3.0-1ubuntu1~22.04_amd64.deb ...
Unpacking libtsan2:amd64 (12.3.0-1ubuntu1~22.04) ...
Selecting previously unselected package libgcc-12-dev:amd64.
Preparing to unpack .../03-libgcc-12-dev_12.3.0-1ubuntu1~22.04_amd64.deb ...
Unpacking libgcc-12-dev:amd64 (12.3.0-1ubuntu1~22.04) ...
Selecting previously unselected package gcc-12.
Preparing to unpack .../04-gcc-12_12.3.0-1ubuntu1~22.04_amd64.deb ...
Unpacking gcc-12 (12.3.0-1ubuntu1~22.04) ...
Selecting previously unselected package dctrl-tools.
Preparing to unpack .../05-dctrl-tools_2.24-3build2_amd64.deb ...
Unpacking dctrl-tools (2.24-3build2) ...
Selecting previously unselected package dkms.
Preparing to unpack .../06-dkms_2.8.7-2ubuntu2.2_all.deb ...
Unpacking dkms (2.8.7-2ubuntu2.2) ...
Selecting previously unselected package gcc-12-base:i386.
Preparing to unpack .../07-gcc-12-base_12.3.0-1ubuntu1~22.04_i386.deb ...
Unpacking gcc-12-base:i386 (12.3.0-1ubuntu1~22.04) ...
Selecting previously unselected package libgcc-s1:i386.
Preparing to unpack .../08-libgcc-s1_12.3.0-1ubuntu1~22.04_i386.deb ...
Unpacking libgcc-s1:i386 (12.3.0-1ubuntu1~22.04) ...
Selecting previously unselected package libcrypt1:i386.
Preparing to unpack .../09-libcrypt1_1%3a4.4.27-1_i386.deb ...
Unpacking libcrypt1:i386 (1:4.4.27-1) ...
Selecting previously unselected package libc6:i386.
Preparing to unpack .../10-libc6_2.35-0ubuntu3.7_i386.deb ...
Unpacking libc6:i386 (2.35-0ubuntu3.7) ...
Selecting previously unselected package libcom-err2:i386.
Preparing to unpack .../11-libcom-err2_1.46.5-2ubuntu1.1_i386.deb ...
Unpacking libcom-err2:i386 (1.46.5-2ubuntu1.1) ...
Selecting previously unselected package libkrb5support0:i386.
Preparing to unpack .../12-libkrb5support0_1.19.2-2ubuntu0.3_i386.deb ...
Unpacking libkrb5support0:i386 (1.19.2-2ubuntu0.3) ...
Selecting previously unselected package libk5crypto3:i386.
Preparing to unpack .../13-libk5crypto3_1.19.2-2ubuntu0.3_i386.deb ...
Unpacking libk5crypto3:i386 (1.19.2-2ubuntu0.3) ...
Selecting previously unselected package libkeyutils1:i386.
Preparing to unpack .../14-libkeyutils1_1.6.1-2ubuntu3_i386.deb ...
Unpacking libkeyutils1:i386 (1.6.1-2ubuntu3) ...
Selecting previously unselected package libssl3:i386.
Preparing to unpack .../15-libssl3_3.0.2-0ubuntu1.15_i386.deb ...
Unpacking libssl3:i386 (3.0.2-0ubuntu1.15) ...
Selecting previously unselected package libkrb5-3:i386.
Preparing to unpack .../16-libkrb5-3_1.19.2-2ubuntu0.3_i386.deb ...
Unpacking libkrb5-3:i386 (1.19.2-2ubuntu0.3) ...
Selecting previously unselected package libgssapi-krb5-2:i386.
Preparing to unpack .../17-libgssapi-krb5-2_1.19.2-2ubuntu0.3_i386.deb ...
Unpacking libgssapi-krb5-2:i386 (1.19.2-2ubuntu0.3) ...
Selecting previously unselected package libtirpc3:i386.
Preparing to unpack .../18-libtirpc3_1.3.2-2ubuntu0.1_i386.deb ...
Unpacking libtirpc3:i386 (1.3.2-2ubuntu0.1) ...
Selecting previously unselected package libnsl2:i386.
Preparing to unpack .../19-libnsl2_1.3.0-2build2_i386.deb ...
Unpacking libnsl2:i386 (1.3.0-2build2) ...
Selecting previously unselected package libmd0:i386.
Preparing to unpack .../20-libmd0_1.0.4-1build1_i386.deb ...
Unpacking libmd0:i386 (1.0.4-1build1) ...
Selecting previously unselected package libbsd0:i386.
Preparing to unpack .../21-libbsd0_0.11.5-1_i386.deb ...
Unpacking libbsd0:i386 (0.11.5-1) ...
Selecting previously unselected package libunistring2:i386.
Preparing to unpack .../22-libunistring2_1.0-1_i386.deb ...
Unpacking libunistring2:i386 (1.0-1) ...
Selecting previously unselected package libidn2-0:i386.
Preparing to unpack .../23-libidn2-0_2.3.2-2build1_i386.deb ...
Unpacking libidn2-0:i386 (2.3.2-2build1) ...
Selecting previously unselected package libxau6:i386.
Preparing to unpack .../24-libxau6_1%3a1.0.9-1build5_i386.deb ...
Unpacking libxau6:i386 (1:1.0.9-1build5) ...
Selecting previously unselected package libxdmcp6:i386.
Preparing to unpack .../25-libxdmcp6_1%3a1.1.3-0ubuntu5_i386.deb ...
Unpacking libxdmcp6:i386 (1:1.1.3-0ubuntu5) ...
Selecting previously unselected package libxcb1:i386.
Preparing to unpack .../26-libxcb1_1.14-3ubuntu3_i386.deb ...
Unpacking libxcb1:i386 (1.14-3ubuntu3) ...
Selecting previously unselected package libx11-6:i386.
Preparing to unpack .../27-libx11-6_2%3a1.7.5-1ubuntu0.3_i386.deb ...
Unpacking libx11-6:i386 (2:1.7.5-1ubuntu0.3) ...
Selecting previously unselected package libxext6:i386.
Preparing to unpack .../28-libxext6_2%3a1.3.4-1build1_i386.deb ...
Unpacking libxext6:i386 (2:1.3.4-1build1) ...
Selecting previously unselected package libnvidia-common-550.
Preparing to unpack .../29-libnvidia-common-550_550.54.15-0ubuntu1_all.deb ...
Unpacking libnvidia-common-550 (550.54.15-0ubuntu1) ...
Selecting previously unselected package libnvidia-compute-550:amd64.
Preparing to unpack .../30-libnvidia-compute-550_550.54.15-0ubuntu1_amd64.deb ...
Unpacking libnvidia-compute-550:amd64 (550.54.15-0ubuntu1) ...
Selecting previously unselected package libnvidia-gl-550:amd64.
Preparing to unpack .../31-libnvidia-gl-550_550.54.15-0ubuntu1_amd64.deb ...
dpkg-query: no packages found matching libnvidia-gl-535
Unpacking libnvidia-gl-550:amd64 (550.54.15-0ubuntu1) ...
Selecting previously unselected package nvidia-kernel-source-550-open.
Preparing to unpack .../32-nvidia-kernel-source-550-open_550.54.15-0ubuntu1_amd64.deb ...
Unpacking nvidia-kernel-source-550-open (550.54.15-0ubuntu1) ...
Selecting previously unselected package nvidia-firmware-550-550.54.15.
Preparing to unpack .../33-nvidia-firmware-550-550.54.15_550.54.15-0ubuntu1_amd64.deb ...
Unpacking nvidia-firmware-550-550.54.15 (550.54.15-0ubuntu1) ...
Selecting previously unselected package nvidia-kernel-common-550.
Preparing to unpack .../34-nvidia-kernel-common-550_550.54.15-0ubuntu1_amd64.deb ...
Unpacking nvidia-kernel-common-550 (550.54.15-0ubuntu1) ...
Selecting previously unselected package nvidia-dkms-550-open.
Preparing to unpack .../35-nvidia-dkms-550-open_550.54.15-0ubuntu1_amd64.deb ...
Unpacking nvidia-dkms-550-open (550.54.15-0ubuntu1) ...
Selecting previously unselected package libnvidia-extra-550:amd64.
Preparing to unpack .../36-libnvidia-extra-550_550.54.15-0ubuntu1_amd64.deb ...
Unpacking libnvidia-extra-550:amd64 (550.54.15-0ubuntu1) ...
Selecting previously unselected package nvidia-compute-utils-550.
Preparing to unpack .../37-nvidia-compute-utils-550_550.54.15-0ubuntu1_amd64.deb ...
Unpacking nvidia-compute-utils-550 (550.54.15-0ubuntu1) ...
Selecting previously unselected package libnvidia-decode-550:amd64.
Preparing to unpack .../38-libnvidia-decode-550_550.54.15-0ubuntu1_amd64.deb ...
Unpacking libnvidia-decode-550:amd64 (550.54.15-0ubuntu1) ...
Selecting previously unselected package libnvidia-encode-550:amd64.
Preparing to unpack .../39-libnvidia-encode-550_550.54.15-0ubuntu1_amd64.deb ...
Unpacking libnvidia-encode-550:amd64 (550.54.15-0ubuntu1) ...
Selecting previously unselected package nvidia-utils-550.
Preparing to unpack .../40-nvidia-utils-550_550.54.15-0ubuntu1_amd64.deb ...
Unpacking nvidia-utils-550 (550.54.15-0ubuntu1) ...
Selecting previously unselected package libnvidia-cfg1-550:amd64.
Preparing to unpack .../41-libnvidia-cfg1-550_550.54.15-0ubuntu1_amd64.deb ...
Unpacking libnvidia-cfg1-550:amd64 (550.54.15-0ubuntu1) ...
Selecting previously unselected package xserver-xorg-video-nvidia-550.
Preparing to unpack .../42-xserver-xorg-video-nvidia-550_550.54.15-0ubuntu1_amd64.deb ...
Unpacking xserver-xorg-video-nvidia-550 (550.54.15-0ubuntu1) ...
Selecting previously unselected package libnvidia-fbc1-550:amd64.
Preparing to unpack .../43-libnvidia-fbc1-550_550.54.15-0ubuntu1_amd64.deb ...
Unpacking libnvidia-fbc1-550:amd64 (550.54.15-0ubuntu1) ...
Selecting previously unselected package nvidia-driver-550-open.
Preparing to unpack .../44-nvidia-driver-550-open_550.54.15-0ubuntu1_amd64.deb ...
Unpacking nvidia-driver-550-open (550.54.15-0ubuntu1) ...
Selecting previously unselected package cuda-drivers-550.
Preparing to unpack .../45-cuda-drivers-550_550.54.15-1_amd64.deb ...
Unpacking cuda-drivers-550 (550.54.15-1) ...
Selecting previously unselected package krb5-locales.
Preparing to unpack .../46-krb5-locales_1.19.2-2ubuntu0.3_all.deb ...
Unpacking krb5-locales (1.19.2-2ubuntu0.3) ...
Selecting previously unselected package libnss-nis:i386.
Preparing to unpack .../47-libnss-nis_3.1-0ubuntu6_i386.deb ...
Unpacking libnss-nis:i386 (3.1-0ubuntu6) ...
Selecting previously unselected package libnss-nisplus:i386.
Preparing to unpack .../48-libnss-nisplus_1.3-0ubuntu6_i386.deb ...
Unpacking libnss-nisplus:i386 (1.3-0ubuntu6) ...
Selecting previously unselected package libnvidia-compute-550:i386.
Preparing to unpack .../49-libnvidia-compute-550_550.54.15-0ubuntu1_i386.deb ...
Unpacking libnvidia-compute-550:i386 (550.54.15-0ubuntu1) ...
Selecting previously unselected package libnvidia-decode-550:i386.
Preparing to unpack .../50-libnvidia-decode-550_550.54.15-0ubuntu1_i386.deb ...
Unpacking libnvidia-decode-550:i386 (550.54.15-0ubuntu1) ...
Selecting previously unselected package libnvidia-encode-550:i386.
Preparing to unpack .../51-libnvidia-encode-550_550.54.15-0ubuntu1_i386.deb ...
Unpacking libnvidia-encode-550:i386 (550.54.15-0ubuntu1) ...
Selecting previously unselected package libnvidia-fbc1-550:i386.
Preparing to unpack .../52-libnvidia-fbc1-550_550.54.15-0ubuntu1_i386.deb ...
Unpacking libnvidia-fbc1-550:i386 (550.54.15-0ubuntu1) ...
Selecting previously unselected package libnvidia-gl-550:i386.
Preparing to unpack .../53-libnvidia-gl-550_550.54.15-0ubuntu1_i386.deb ...
dpkg-query: no packages found matching libnvidia-gl-535
Unpacking libnvidia-gl-550:i386 (550.54.15-0ubuntu1) ...
Selecting previously unselected package libvdpau1:amd64.
Preparing to unpack .../54-libvdpau1_1.4-3build2_amd64.deb ...
Unpacking libvdpau1:amd64 (1.4-3build2) ...
Selecting previously unselected package libxnvctrl0:amd64.
Preparing to unpack .../55-libxnvctrl0_550.54.15-0ubuntu1_amd64.deb ...
Unpacking libxnvctrl0:amd64 (550.54.15-0ubuntu1) ...
Selecting previously unselected package mesa-vdpau-drivers:amd64.
Preparing to unpack .../56-mesa-vdpau-drivers_23.2.1-1ubuntu3.1~22.04.2_amd64.deb ...
Unpacking mesa-vdpau-drivers:amd64 (23.2.1-1ubuntu3.1~22.04.2) ...
Selecting previously unselected package nvidia-prime.
Preparing to unpack .../57-nvidia-prime_0.8.17.1_all.deb ...
Unpacking nvidia-prime (0.8.17.1) ...
Selecting previously unselected package screen-resolution-extra.
Preparing to unpack .../58-screen-resolution-extra_0.18.2_all.deb ...
Unpacking screen-resolution-extra (0.18.2) ...
Selecting previously unselected package nvidia-settings.
Preparing to unpack .../59-nvidia-settings_550.54.15-0ubuntu1_amd64.deb ...
Unpacking nvidia-settings (550.54.15-0ubuntu1) ...
Selecting previously unselected package vdpau-driver-all:amd64.
Preparing to unpack .../60-vdpau-driver-all_1.4-3build2_amd64.deb ...
Unpacking vdpau-driver-all:amd64 (1.4-3build2) ...
Setting up cpp-12 (12.3.0-1ubuntu1~22.04) ...
Setting up libnvidia-compute-550:amd64 (550.54.15-0ubuntu1) ...
Setting up nvidia-prime (0.8.17.1) ...
Setting up libnvidia-common-550 (550.54.15-0ubuntu1) ...
Setting up nvidia-firmware-550-550.54.15 (550.54.15-0ubuntu1) ...
Setting up nvidia-utils-550 (550.54.15-0ubuntu1) ...
Setting up krb5-locales (1.19.2-2ubuntu0.3) ...
Setting up libnvidia-fbc1-550:amd64 (550.54.15-0ubuntu1) ...
Setting up libnvidia-cfg1-550:amd64 (550.54.15-0ubuntu1) ...
Setting up nvidia-compute-utils-550 (550.54.15-0ubuntu1) ...
Warning: The home dir /nonexistent you specified can't be accessed: No such file or directory
Adding system user `nvidia-persistenced' (UID 129) ...
Adding new group `nvidia-persistenced' (GID 137) ...
Adding new user `nvidia-persistenced' (UID 129) with group `nvidia-persistenced' ...
Not creating home directory `/nonexistent'.
Setting up gcc-12-base:i386 (12.3.0-1ubuntu1~22.04) ...
Setting up libxnvctrl0:amd64 (550.54.15-0ubuntu1) ...
Setting up screen-resolution-extra (0.18.2) ...
Setting up libnvidia-gl-550:amd64 (550.54.15-0ubuntu1) ...
Setting up nvidia-kernel-common-550 (550.54.15-0ubuntu1) ...
update-initramfs: deferring update (trigger activated)
Created symlink /etc/systemd/system/systemd-hibernate.service.wants/nvidia-hibernate.service → /lib/systemd/system/nvidia-hibernate.service.
Created symlink /etc/systemd/system/systemd-suspend.service.wants/nvidia-resume.service → /lib/systemd/system/nvidia-resume.service.
Created symlink /etc/systemd/system/systemd-hibernate.service.wants/nvidia-resume.service → /lib/systemd/system/nvidia-resume.service.
Created symlink /etc/systemd/system/systemd-suspend.service.wants/nvidia-suspend.service → /lib/systemd/system/nvidia-suspend.service.
Setting up libnvidia-extra-550:amd64 (550.54.15-0ubuntu1) ...
Setting up libvdpau1:amd64 (1.4-3build2) ...
Setting up libasan8:amd64 (12.3.0-1ubuntu1~22.04) ...
Setting up nvidia-settings (550.54.15-0ubuntu1) ...
Setting up nvidia-kernel-source-550-open (550.54.15-0ubuntu1) ...
Setting up libtsan2:amd64 (12.3.0-1ubuntu1~22.04) ...
Setting up dctrl-tools (2.24-3build2) ...
Setting up mesa-vdpau-drivers:amd64 (23.2.1-1ubuntu3.1~22.04.2) ...
Setting up libnvidia-decode-550:amd64 (550.54.15-0ubuntu1) ...
Setting up xserver-xorg-video-nvidia-550 (550.54.15-0ubuntu1) ...
Setting up libnvidia-encode-550:amd64 (550.54.15-0ubuntu1) ...
Setting up libgcc-12-dev:amd64 (12.3.0-1ubuntu1~22.04) ...
Setting up vdpau-driver-all:amd64 (1.4-3build2) ...
Setting up gcc-12 (12.3.0-1ubuntu1~22.04) ...
Setting up dkms (2.8.7-2ubuntu2.2) ...
Setting up nvidia-dkms-550-open (550.54.15-0ubuntu1) ...
update-initramfs: deferring update (trigger activated)
INFO:Enable nvidia
DEBUG:Parsing /usr/share/ubuntu-drivers-common/quirks/dell_latitude
DEBUG:Parsing /usr/share/ubuntu-drivers-common/quirks/put_your_quirks_here
DEBUG:Parsing /usr/share/ubuntu-drivers-common/quirks/lenovo_thinkpad
Loading new nvidia-550.54.15 DKMS files...
Building for 6.5.0-28-generic
Building for architecture x86_64
Building initial module for 6.5.0-28-generic
Can't load /var/lib/shim-signed/mok/.rnd into RNG
4027C216C5740000:error:12000079:random number generator:RAND_load_file:Cannot open file:../crypto/rand/randfile.c:106:Filename=/var/lib/shim-signed/mok/.rnd
.........+.........+.....+...+.+......+.....+.............+...........+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*.+...+...+..+...+.......+........+...+.......+..+..................+.......+...+...+.....+.+...+.....+.......+......+...+..............+..........+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*....+.+...+.....+.........+.......+......+...+..+.........+.+..............+...+......+.....................+.+.........+......+.....+...+.+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
..+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*..+......+...+......+....+.....+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*.+..+.......+...+..+......+.......+........+.....................+...+.+......+...+.....+...+....+........+......+......+...+..........+.....+......+......+....+..+.........+....+..+....+.....+....+..............+...............+.+...........+.+...+..+.......+...+........+.+...........+...+............+............+...+....+......+.........+.....+..........+......+...+......+........+...+...+....+...+...+.....+........................+...+....+...+......+.....+....+......+.....+.......+...+......+...+.....+......+.......+.....+..........+...+..+.........+....+.....+.+...+...+..+...+.........+.+........+.......+...+...........+.............+..+.........+......+...+.+......+...+..+....+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
-----
Done.

nvidia.ko:
Running module version sanity check.
 - Original module
   - No original module exists within this kernel
 - Installation
   - Installing to /lib/modules/6.5.0-28-generic/updates/dkms/

nvidia-modeset.ko:
Running module version sanity check.
 - Original module
   - No original module exists within this kernel
 - Installation
   - Installing to /lib/modules/6.5.0-28-generic/updates/dkms/

nvidia-drm.ko:
Running module version sanity check.
 - Original module
   - No original module exists within this kernel
 - Installation
   - Installing to /lib/modules/6.5.0-28-generic/updates/dkms/

nvidia-uvm.ko:
Running module version sanity check.
 - Original module
   - No original module exists within this kernel
 - Installation
   - Installing to /lib/modules/6.5.0-28-generic/updates/dkms/

nvidia-peermem.ko:
Running module version sanity check.
 - Original module
   - No original module exists within this kernel
 - Installation
   - Installing to /lib/modules/6.5.0-28-generic/updates/dkms/

depmod...
Setting up nvidia-driver-550-open (550.54.15-0ubuntu1) ...
Setting up cuda-drivers-550 (550.54.15-1) ...
Setting up libcrypt1:i386 (1:4.4.27-1) ...
Setting up libgcc-s1:i386 (12.3.0-1ubuntu1~22.04) ...
Setting up libc6:i386 (2.35-0ubuntu3.7) ...
Setting up libmd0:i386 (1.0.4-1build1) ...
Setting up libbsd0:i386 (0.11.5-1) ...
Setting up libxau6:i386 (1:1.0.9-1build5) ...
Setting up libxdmcp6:i386 (1:1.1.3-0ubuntu5) ...
Setting up libkeyutils1:i386 (1.6.1-2ubuntu3) ...
Setting up libxcb1:i386 (1.14-3ubuntu3) ...
Setting up libnvidia-compute-550:i386 (550.54.15-0ubuntu1) ...
Setting up libssl3:i386 (3.0.2-0ubuntu1.15) ...
Setting up libunistring2:i386 (1.0-1) ...
Setting up libidn2-0:i386 (2.3.2-2build1) ...
Setting up libcom-err2:i386 (1.46.5-2ubuntu1.1) ...
Setting up libkrb5support0:i386 (1.19.2-2ubuntu0.3) ...
Setting up libk5crypto3:i386 (1.19.2-2ubuntu0.3) ...
Setting up libx11-6:i386 (2:1.7.5-1ubuntu0.3) ...
Setting up libkrb5-3:i386 (1.19.2-2ubuntu0.3) ...
Setting up libxext6:i386 (2:1.3.4-1build1) ...
Setting up libnvidia-fbc1-550:i386 (550.54.15-0ubuntu1) ...
Setting up libgssapi-krb5-2:i386 (1.19.2-2ubuntu0.3) ...
Setting up libnvidia-gl-550:i386 (550.54.15-0ubuntu1) ...
Setting up libtirpc3:i386 (1.3.2-2ubuntu0.1) ...
Setting up libnvidia-decode-550:i386 (550.54.15-0ubuntu1) ...
Setting up libnvidia-encode-550:i386 (550.54.15-0ubuntu1) ...
Setting up libnsl2:i386 (1.3.0-2build2) ...
Setting up libnss-nisplus:i386 (1.3-0ubuntu6) ...
Setting up libnss-nis:i386 (3.1-0ubuntu6) ...
Processing triggers for mailcap (3.70+nmu1ubuntu1) ...
Processing triggers for desktop-file-utils (0.26-1ubuntu3) ...
Processing triggers for initramfs-tools (0.140ubuntu13.5) ...
update-initramfs: Generating /boot/initrd.img-6.5.0-28-generic
Processing triggers for gnome-menus (3.36.0-1ubuntu3) ...
Processing triggers for libc-bin (2.35-0ubuntu3.7) ...
Processing triggers for ccache (4.5.1-1) ...
Updating symlinks in /usr/lib/ccache ...
Processing triggers for man-db (2.10.2-1) ...
```
The above installation will prompt for a MOK( Machine Owner Key) password which
needs to be provided and is used for loading third party kernel modules.
Reboot and choose `enroll` from the MOK manager menu to complete the
installation.
```

After that I was finally to run nvidia-smi and see the following output:
```
$ nvidia-smi 
Sat Apr 20 08:15:56 2024       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4080 ...    Off |   00000000:01:00.0  On |                  N/A |
| N/A   42C    P8              2W /   90W |      67MiB /  12282MiB |     10%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A      2132      G   /usr/lib/xorg/Xorg                             57MiB |
+-----------------------------------------------------------------------------------------+
```
And this also allowed my external Dell 24" monitor to work with the laptop.

### Installing CUDA 11.8
I needed to install CUDA 11.8 for a project I was working on and these are the
steps I took to install it. Note that I'm using ubuntu 22.04:
```console
$ lsb_release -r
Release:	22.04
``` 
So this is adding the CUDA 11.8 repository to the system in addition to CUDA
12.4.

```console
$ https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network
$ sudo dpkg -i cuda-keyring_1.0-1_all.deb
$ sudo apt-get update
$ sudo apt install cuda-toolkit-11-8
```

The update the PATH and LD_LIBRARY_PATH variables your environment. I've save
this in a script so that I can switch between different versions of CUDA if
needed:
```bash
export PATH=/usr/local/cuda-11.8/bin:$PATH                                      
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH                 
nvcc --version 
```

```console
apt list --installed
```
