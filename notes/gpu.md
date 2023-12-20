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
|    |       Global Memory       |  |
|    +---------------------------+  |
|    | SM1  | SM2  | ... | SMn   |  |
|    +---------------------------+  |
|    |  Shared/Global Memory     |  |
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
Lets think about input data to a process, for example an array that we want to
sum all the element of. 

1) Input data is indeed copied from the host's main memory to the GPU's global
memory. 

2) Data needed for processing by a specific set of threads within an SM is then
copied to the SM's shared memory. This memory is shared among all cores and this
is something that I think the programmer has to do explicitly.
For data that might not fit into the SMs shared memory, it will be accessed from
the GPUs global memory, and this data will be cached in the SMs L1 cache. Note
that the L1 cache is not involved in the SMs shared memory accesses.

3) Data in the shared memory can be read and written by the cores on the same
SM, for examples this can be temporary results of computations.

4) The results of the computations are then copied back to the GPUs global
memory.

5) Finally, the results are copied back to the host's main memory.


The registers on the SMs are are per GPU thread which is not the case for a CPU
where the registers are per CPU core.
Every core in an SM has a register file, which is a collection of registers
used exclusively by that core.

Threads are the smallest unit of execution in a GPU and execute part of the
kernel. Just a note about the name "kernel" as the first thing I thought about
was the linux kernel. In this case I think it comes from that what we want
executed is a small portion of our program, a part of it that we want to
optimize (and which can benifit from parallelization). So this is the "kernel",
it is the "core" computation unit of our program, or the "essential" part that
is being computed.

A block is a group of threads that execute the same kernel and can communicate
with each other via shared memory. This means that a block is specific to an
SM as the shared memory is per SM.

A grid is a collection of blocks that execute the same kernel. This grid is
distributed across the SMs of the GPU. So while an individual block is specific
to an SM, other blocks in the same grid can be on other SMs.

Each SM can execute multiple blocks at the same time. And each SM has multiple
/many cores, and each of these cores can execute one or more threads at the
same time.

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
TODO: add notes about threadIdx, blockIdx, blockDim, gridDim


It is also possible to have multiple buses between the GPU and system memory
which is called parallelization. This way the GPU can keep transferring data
from system memory into its own memory while it is processing data.

In most systems, the GPU has its own dedicated memory (VRAM - Video RAM) that
is separate from the system's main RAM. VRAM is optimized for the GPU's
high-throughput, parallel processing needs.

CPU to GPU Data Transfer: Initially, data (like a matrix for multiplication or
texture data for rendering) resides in the system's main memory. To be processed
by the GPU, this data needs to be transferred to the GPU's memory. This transfer
typically happens over the PCIe (Peripheral Component Interconnect Express) bus,
a high-speed interface that connects the GPU to the motherboard and through it
to the system memory.

PCIe Bandwidth: The PCIe bus has a certain bandwidth limit, which dictates how
fast data can be transferred between the system memory and the GPU memory. This
transfer can become a bottleneck, especially for large data sets or when
frequent data transfers are required.

Data Fetch by GPU: Once the data is in the GPU's memory, the GPU cores can
access it much faster. The GPU's design allows it to fetch large blocks of data
concurrently, leveraging its parallel processing capabilities. This is
particularly efficient for tasks that can process large chunks of data in
parallel, like rendering graphics or performing computations on matrices.

Data Processing: After fetching the data, the GPU performs the required
processing - this could be rendering frames for a game, computing physics
simulations, or running machine learning algorithms.

Result Transfer Back to CPU: If the results of the GPU's computations need to be
used by the CPU or need to be stored back in the system memory, they are 
ransferred back via the PCIe bus.

Optimizations: To minimize the bottleneck caused by data transfers over PCIe,
software and hardware optimizations are often used. These can include minimizing
the amount of data that needs to be transferred, using techniques like
compression, or organizing data in a way that reduces the need for transfers.

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
