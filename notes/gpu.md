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
where the registers are per CPU core.
Every core in an SM has a register file, which is a collection of registers
used exclusively by that core. The term register "file" has always confused me
but in computing "file" has older roots, where it was used to describe a
collection of related data. In early computing, a "file" could refer to a
collection of data cards, for example. So a register file is a collection of
registers which are memory locations on the chip itself.

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
So is we access the 5th element in an array then blockIdx.x would be 1, and
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
So if we have to matrices, a and b, to produce the matrix c we can see that the
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
The are used to serve hyper-realistic graphics rendering.

### Shaders
Shaders are blocks of code that excute on the GPU, similar to how CUDA kernels.
And in a similar way to CUDA kernels which are defined in a C-like language,
shaders are defined in languages like High Level Shading Language (HLSL) or
OpenGL Shading Language (GLSL).

HLSL is used with DirectX (Microsoft) and primary support is for Window. Has and
c-like syntax. Is often pre-compiled into intermediate bytecode which is then
executed by the GPU.

GLSL is used with OpenGL (Khronos) and is cross-platform and also has a c-like
syntax. Are typically compiled at runtime but it is possible to pre-compile them
as well.



#### Vertex Shaders
For some background to be able to understand the what this shader does and the
following, imaging defining a triangle in a program consisting of three points
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
