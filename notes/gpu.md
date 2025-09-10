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
|    |      Processing Units (SP)|  |
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

### CUDA example
I'm going to take a bottom up approach to explain how my CUDA GPU works. I hope
this will make the most sense without getting confused with the higher level
concepts and teminology.

So lets take a very basic matric multiplication example, the one that will be
used for this section is [matrix-mul](../gpu/cuda/src/matrix-mul.cu).
It is going to perform the following operation:
```
     Matrix A:          Matrix B:             Matrix C:
   0    1   2   3     0    1   2   3          0   1  2   3
  +---+---+---+---+  +---+---+---+---+      +---+---+---+---+
0 |a00|a01|a02|a03|  |b00|b01|b02|b03|      |c00|c01|c02|c03|
  +---+---+---+---+  +---+---+---+---+      +---+---+---+---+
1 |a10|a11|a12|a13|  |b10|b11|b12|b13|      |c10|c11|c12|c13|
  +---+---+---+---+  +---+---+---+---+  =   +---+---+---+---+
2 |a20|a21|a22|a23|  |b20|b21|b22|b23|      |c20|c21|c22|c23|
  +---+---+---+---+  +---+---+---+---+      +---+---+---+---+
3 |a30|a31|a32|a33|  |b30|b31|b32|b33|      |c30|c31|c32|c33|
  +---+---+---+---+  +---+---+---+---+      +---+---+---+---+
4 |a40|a41|a42|a43|                         |c40|c41|c42|c43|
  +---+---+---+---+                         +---+---+---+---+
5 |a50|a51|a52|a53|                         |c50|c51|c52|c53|
  +---+---+---+---+                         +---+---+---+---+
6 |a60|a61|a62|a63|                         |c60|c61|c62|c63|
  +---+---+---+---+                         +---+---+---+---+
7 |a70|a71|a72|a73|                         |c70|c71|c72|c73|
  +---+---+---+---+                         +---+---+---+---+
  Total: 32 elements  Total: 16 elements    Total: 32 elements
```
Now, each element of the output matrix is the dot product of one row in matrix A
with one column of matrix B. For example lets look at `c11`, this the second
row in matrix A and the second column in matrix B:
```
  +---+---+---+---+  +---+
1 |a10|a11|a12|a13|  |b01|
  +---+---+---+---+  +---+
                     |b11|
                     +---+
                     |b21|
                     +---+
                     |b31|
                     +---+
```
I know this seems very basic but I want to refer back to this example later on.
Notice that we have 32 elements in the output matrix, and that to compute the
result for one element we need to read 4 elements from matrix A and 4 elements
matrix B. So we will have 32 threads running in parallel.

So our kernel looks like this:
```c++
__global__ void matrix_mul(float* A, float* B, float* C) {
    int tid = threadIdx.x;  // 0 to 31

    int row = tid / 4;
    int col = tid % 4;

    float result = 0.0f; // local variable is stored in a register
    float a_val, b_val;

    // This loop computes one element of the output matrix C.
    for (int k = 0; k < 4; k++) {
        a_val = A[row * 4 + k]; // Global memory read by all threads
        b_val = B[k * 4 + col]; // Global memory read by all threads

        result += a_val * b_val;
    }
    C[row * 4 + col] = result; // Global memory write
}
```
So this will execute on 32 processing units/cores in parallel. Each thread will
have a different thread id. We get the row and column using:
```
0  row = 0/4 = 0, col = 0%4 = 0   a0 . b0 = c00
1  row = 1/4 = 0, col = 1%4 = 1   a0 . b1 = c01
2  row = 2/4 = 0, col = 2%4 = 2   a0 . b2 = c02
3  row = 3/4 = 0, col = 3%4 = 3   a0 . b3 = c03

4  row = 4/4 = 1, col = 4%4 = 0   a1 . b0 = c10
5  row = 5/4 = 1, col = 5%4 = 1   a1 . b1 = c11
6  row = 6/4 = 1, col = 6%4 = 2   a1 . b2 = c12
7  row = 7/4 = 1, col = 7%4 = 3   a1 . b3 = c13

8  row = 8/4 = 2, col = 8%4 = 0   a2 . b0 = c20
9  row = 9/4 = 2, col = 9%4 = 1   a2 . b1 = c21
10 row =10/4 = 2, col =10%4 = 2   a2 . b2 = c22
11 row =11/4 = 2, col =11%4 = 3   a2 . b3 = c23

12 row =12/4 = 3, col =12%4 = 0   a3 . b0 = c30
13 row =13/4 = 3, col =13%4 = 1   a3 . b1 = c31
14 row =14/4 = 3, col =14%4 = 2   a3 . b2 = c32
15 row =15/4 = 3, col =15%4 = 3   a3 . b3 = c33

16 row =16/4 = 4, col =16%4 = 0   a4 . b0 = c40
17 row =17/4 = 4, col =17%4 = 1   a4 . b1 = c41
18 row =18/4 = 4, col =18%4 = 2   a4 . b2 = c42
19 row =19/4 = 4, col =19%4 = 3   a4 . b3 = c43

20 row =20/4 = 5, col =20%4 = 0   a5 . b0 = c50
21 row =21/4 = 5, col =21%4 = 1   a5 . b1 = c51
22 row =22/4 = 5, col =22%4 = 2   a5 . b2 = c52
23 row =23/4 = 5, col =23%4 = 3   a5 . b3 = c53

24 row =24/4 = 6, col =24%4 = 0   a6 . b0 = c60
25 row =25/4 = 6, col =25%4 = 1   a6 . b1 = c61
26 row =26/4 = 6, col =26%4 = 2   a6 . b2 = c62
27 row =27/4 = 6, col =27%4 = 3   a6 . b3 = c63

28 row =28/4 = 7, col =28%4 = 0   a7 . b0 = c70
29 row =29/4 = 7, col =29%4 = 1   a7 . b1 = c71
30 row =30/4 = 7, col =30%4 = 2   a7 . b2 = c72
31 row =31/4 = 7, col =31%4 = 3   a7 . b3 = c73
```
Notice that each thread contains the same for loop, but that it reads from
from different global memory addresses (A and B are in global memory but more
on this later). And the memory layout it a flat array in row-major order:
```
      0    1    2    3    4    5    6    7    8   9    10   11   12   13   14   15   16   17   18   19   20   21   22   23   24   25   26   27   28   29   30   31
A = [a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33, a40, a41, a42, a43, a50, a51, a52, a53, a60, a61, a62, a63, a70, a71, a72, a73]
      0    1    2    3    4    5    6    7    8   9    10   11   12   13   14   15
B = [b00, b01, b02, b03, b10, b11, b12, b13, b20, b21, b22, b23, b30, b31, b32, b33]
```
So any of the 32 threads, will index into A using the thread id times the column
stride which is 4 in this case.
Lets take thread 10 which will access, row 2 and column 2:
```
    for (int k = 0; k < 4; k++) {
        a_val = A[row * 4 + k]; // Global memory read by all threads
        b_val = B[k * 4 + col]; // Global memory read by all threads
    }
k = 0, A[2 * 4 + 0] = A[8]  = a20
k = 1, A[2 * 4 + 1] = A[9]  = a21
k = 2, A[2 * 4 + 2] = A[10] = a22
k = 3, A[2 * 4 + 3] = A[15] = a23

k = 0, B[0 * 4 + 2] = B[2]  = b02
k = 1, B[1 * 4 + 2] = B[6]  = b12
k = 2, B[2 * 4 + 2] = B[10] = b22
k = 3, B[3 * 4 + 2] = B[14] = b32

  +---+---+---+---+  +---+
  |a20|a21|a22|a23|  |b02|
  +---+---+---+---+  +---+
                     |b12|
                     +---+
                     |b22|
                     +---+
                     |b32|
                     +---+
```
```console
$ cuda-gdb matrix-mul
NVIDIA (R) cuda-gdb 12.6
Reading symbols from matrix-mul...
(cuda-gdb) br matrix-mul.cu:13
Breakpoint 1 at 0xb367: file src/matrix-mul.cu, line 20.
(cuda-gdb) r
Starting program: /home/danbev/work/ai/learning-ai/gpu/cuda/matrix-mul
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
Matrix A:
 1.0  2.0  3.0  4.0
 5.0  6.0  7.0  8.0
 9.0 10.0 11.0 12.0
13.0 14.0 15.0 16.0

Matrix B:
 1.0  2.0  3.0  4.0
 5.0  6.0  7.0  8.0
 9.0 10.0 11.0 12.0
13.0 14.0 15.0 16.0
[Switching focus to CUDA kernel 0, grid 1, block (0,0,0), thread (0,0,0), device 0, sm 0, warp 0, lane 0]

CUDA thread hit Breakpoint 1, matrix_mul<<<(1,1,1),(32,1,1)>>> (A=0x7fffd7e00000, B=0x7fffd7e00200, C=0x7fffd7e00400)
    at src/matrix-mul.cu:13
13	    for (int k = 0; k < 4; k++) {
```
Now, lets switch to thread 10:
```console
(cuda-gdb) cuda lane 10
[Switching focus to CUDA kernel 0, grid 1, block (0,0,0), thread (10,0,0), device 0, sm 0, warp 0, lane 10]
13	    for (int k = 0; k < 4; k++) {
```
And we can verify that our row and col are correct:
```
(cuda-gdb) p row
$3 = 2
(cuda-gdb) p col
$4 = 2
```
And lets see what gt
```
$5 = 8
(cuda-gdb) p A[8]
$6 = 9
(cuda-gdb) p k * 4 + col
$7 = 2
(cuda-gdb) p B[2]
$8 = 3
```
And this matches our expectations from above. If we inspect all the thread/lanes
we can see the following:
```console
(cuda-gdb) info cuda lanes
  Ln  State         PC         ThreadIdx Exception Device 0 SM 0 Warp 0
   0 active 0x00007fffdb259fd0   (0,0,0)    None
   1 active 0x00007fffdb259fd0   (1,0,0)    None
   2 active 0x00007fffdb259fd0   (2,0,0)    None
   3 active 0x00007fffdb259fd0   (3,0,0)    None
   4 active 0x00007fffdb259fd0   (4,0,0)    None
   5 active 0x00007fffdb259fd0   (5,0,0)    None
   6 active 0x00007fffdb259fd0   (6,0,0)    None
   7 active 0x00007fffdb259fd0   (7,0,0)    None
   8 active 0x00007fffdb259fd0   (8,0,0)    None
   9 active 0x00007fffdb259fd0   (9,0,0)    None
* 10 active 0x00007fffdb259fd0  (10,0,0)    None
  11 active 0x00007fffdb259fd0  (11,0,0)    None
  12 active 0x00007fffdb259fd0  (12,0,0)    None
  13 active 0x00007fffdb259fd0  (13,0,0)    None
  14 active 0x00007fffdb259fd0  (14,0,0)    None
  15 active 0x00007fffdb259fd0  (15,0,0)    None
  16 active 0x00007fffdb259fd0  (16,0,0)    None
  17 active 0x00007fffdb259fd0  (17,0,0)    None
  18 active 0x00007fffdb259fd0  (18,0,0)    None
  19 active 0x00007fffdb259fd0  (19,0,0)    None
  20 active 0x00007fffdb259fd0  (20,0,0)    None
  21 active 0x00007fffdb259fd0  (21,0,0)    None
  22 active 0x00007fffdb259fd0  (22,0,0)    None
  23 active 0x00007fffdb259fd0  (23,0,0)    None
  24 active 0x00007fffdb259fd0  (24,0,0)    None
  25 active 0x00007fffdb259fd0  (25,0,0)    None
  26 active 0x00007fffdb259fd0  (26,0,0)    None
  27 active 0x00007fffdb259fd0  (27,0,0)    None
  28 active 0x00007fffdb259fd0  (28,0,0)    None
  29 active 0x00007fffdb259fd0  (29,0,0)    None
  30 active 0x00007fffdb259fd0  (30,0,0)    None
  31 active 0x00007fffdb259fd0  (31,0,0)    None
                   ↑
             Program counter
```
We can see here that there is only one program counter for all threads:
```console
(cuda-gdb) print $pc
$10 = (void (*)(void)) 0x7fffdb259fd0 <matrix_mul(float*, float*, float*)+2512>
```
We can inspect the instruction at this program counter:
```
(cuda-gdb) x/i $pc
=> 0x7fffdb259fd0 <_Z10matrix_mulPfS_S_+2512>:	IMAD.SHL R8, R10, 0x4, RZ
```
Notice that all thread have the same program counter. This is a fundamental
difference from a CPU thread which would have its own program counter register.
This is what is meant by the threads executing in lock step, they are all executing
the exact same instruction for each step, but the operate on different data which
we say when we changed to a different thread/lane.
The address for
```console
(cuda-gdb) p A
$9 = (@generic float * @parameter) 0x7fffd7e00000
```
Now, the memory address for A is in the GPUs global memory. Looking a little bit
futher in the assembly code we can see the following load instruction:
```console
=> 0x00007fffdb259fd0 <+2512>:	IMAD.SHL R8, R10, 0x4, RZ
   0x00007fffdb259fe0 <+2528>:	IADD3 R8, R8, R13, RZ
   0x00007fffdb259ff0 <+2544>:	MOV R8, R8
   0x00007fffdb25a000 <+2560>:	SHF.R.S32.HI R9, RZ, 0x1f, R8
   0x00007fffdb25a010 <+2576>:	MOV R14, R8
   0x00007fffdb25a020 <+2592>:	MOV R8, R14
   0x00007fffdb25a030 <+2608>:	MOV R9, R9
   0x00007fffdb25a040 <+2624>:	SHF.L.U64.HI R9, R8, 0x2, R9
   0x00007fffdb25a050 <+2640>:	SHF.L.U32 R8, R8, 0x2, RZ
   0x00007fffdb25a060 <+2656>:	IADD3 R8, P0, R4, R8, RZ
   0x00007fffdb25a070 <+2672>:	IADD3.X R9, R5, R9, RZ, P0, !PT
   0x00007fffdb25a080 <+2688>:	MOV R8, R8
   0x00007fffdb25a090 <+2704>:	MOV R9, R9
   0x00007fffdb25a0a0 <+2720>:	MOV R8, R8
   0x00007fffdb25a0b0 <+2736>:	MOV R9, R9
   0x00007fffdb25a0c0 <+2752>:	R2UR UR4, R16
   0x00007fffdb25a0d0 <+2768>:	R2UR UR5, R17
   0x00007fffdb25a0e0 <+2784>:	LD.E R8, [R8.64]
```
Here `LD.E` is the load from global memory instruction with caching enabled, `R8`
is the destination register, and `[R8.64]` is a 64-bit address stored in `R8`.
```c++
a_val = A[row * 4 + k];
```
Now, because a single thread does not have its own program counter this means
that it is external from the processing unit itself. And 32 threads is the unit
of execution and this is referred to as a warp. There is a warp scheduler which
actually manages the execution of warps on the SM. There are actually multiple
warp schedulers but I'll return to this later. Each time we step in the debugger,
the warp scheduler will fetch another instruction from the instruction cache and
decodes the instruction. And depending on the instruction it will dispatch to
different processing units. For example, a floating point instruction might be
dispatched to the floating point units, while a memory load/store instruction
would be dispatched to the Load/Store Unit (LSU).

So a warp scheduler on the SM performs the following:
```console
1. Fetches LD.E instruction from instruction cache
2. Decodes: "This is a memory load operation"
3. Checks: "All 32 threads need to load from memory"
4. Dispatches to: Load/Store Unit (LSU)
```

Load/Store Unit (LSU) will:
```console
Load/Store Unit receives:
├── 32 memory addresses (one per thread)
├── Thread 0: Address in R8 = A[0*4 + k]
├── Thread 1: Address in R8 = A[0*4 + k+1]
├── Thread 4: Address in R8 = A[1*4 + k]
├── Thread 5: Address in R8 = A[1*4 + k+1]
└── ... (32 different addresses total)

LSU analyzes access pattern:
├── Are these consecutive? (YES - good coalescing)
├── Do they fit in cache lines? (YES - 128 bytes)
├── How many memory transactions needed? (4 transactions)
└── Generate optimized memory requests
```
LSU sends requests to memory hierarchy:
```console
L1 Cache Check:
├── Cache hit? → Return data immediately (4 cycles)
└── Cache miss? → Forward to L2 Cache

L2 Cache Check:
├── Cache hit? → Return data (20-40 cycles)
└── Cache miss? → Forward to GDDR6X memory

GDDR6X Memory:
├── Access DRAM (200+ cycles)
├── Load 128-byte cache line
└── Return requested data
```
So the warp scheduler has passed this off to the LSU and now it can handle other
work while the memory access in progress:
```console
Warp Scheduler:
1. Marks Warp 0 as "MEMORY_PENDING"
2. Switches to execute different ready warp
3. Keeps ALUs busy with other warps

When memory returns (200+ cycles later):
1. Warp 0 marked as "READY"
2. Data written to register R8 for all 32 threads
3. Warp scheduler can pick Warp 0 for execution again
```
One thing to notice here is that in contrast to CPU thread context switching where
the threads registers have to be save this does not happen on the GPU. This
registers are resident and stay allocated to the warp for its entire lifetime. And
this is the reason for the much much larger register file on a GPU compared to a
CPU core. The LSU is very important and it can, if the data access patterns
allow it (coalesced memory accesses), combine multiple memory requests into
fewer memory transactions. This is very important for performance. And this is
something that the programmer has to be aware of when writing CUDA code as it
can help improve performance significantly. So thinking about how data is layed
out and accessed is very important and I'll come back to this later.


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

TODO: figure out a better order of introducing the warp schdulers and registers
and execution process. I'm kind of jumping around a bit here as I've learned more.

The registers on the SMs are per GPU thread which is not the case for a CPU
where the registers are per CPU core. Every core/processing unit in an SM has a
register file, which is a collection of registers used exclusively by that core.
The term register "file" has always confused me but in computing "file" has
older roots, where it was used to describe a collection of related data. In early
computing, a "file" could refer to a collection of data cards, for example. So
a register file is a collection of registers which are memory locations on the
chip itself.

There are about 64 registers per thread on modern GPUs and they are typically
32 bits registers. So each SM has a huge register file (e.g., 65,536 32-bit
registers), much more than a CPU core which is interesting.
Now, I though that the reason for this was that GPUs have big register files
because there are lots of threads and that seemed to make sense. But if we look
at the number the have way more register/memory than a CPU, like 50000x more.

The reason for this is that these registers are resident. There is no thread
context switching that occurs when a different warp is executed. Instead the
registers stay set (resident) and can be used again when the same warp is
executed again. When a warp get created it gets permanent allocation of
registers for its lifetime.

```console
Warp 0:
├── Program Counter: 0x1A4C (stored in warp scheduler)
├── Registers R0-R31: Allocated in register file [0-1023]
├── Predicate registers: For conditional execution
├── Status flags: Ready/Stalled/Memory_pending
└── Active mask: Which threads are active

Warp 1:
├── Program Counter: 0x2B8F
├── Registers R0-R31: Allocated in register file [1024-2047]
├── Predicate registers
├── Status flags
└── Active mask

... (up to 64 warps)
```

Threads are executed by GPU cores/compute units, and they execute the kernel.
Just a note about the name "kernel" as the first thing I thought about was the
linux kernel or something like that. In this case I think it comes from that
what we want executed is a small portion of our program, a part of it that we
want to optimize (and which can benefit from parallelization). So this is the
"kernel", it is the "core" computation unit of our program, or the "essential"
part that is being computed.

Each computation unit has 32 ALUs which can execute 32 threads simultaneously.
So the smallest unit of execution is not an individual threads but instead 32
threads at once is the smallest unit of execution. 

This is actually more important that it might seem at first. Each computation
unit on a GPU has 32 ALUs which means that it can execute 32 threads at the same
time.
If we think about how a thread executes on a CPU, it has its own program counter,
stack, and its own registers. It executes instructions one at a time, and a loop
could be implemented something like this :
```console
loop_start:
    ; loop body instructions
    add k, 1
    cmp k, 4
    jl loop_start    ; conditional jump backward
```
This no problem and it can jump as it wants.

But this is not how the execution works on a GPU. Instead, we have something
similar to the following:
```console
; GPU execution - ALL threads execute the same instruction
instruction_0:  load a_val from A[row*4 + k]   ; All 32 threads, different addresses
instruction_1:  load b_val from B[k*4 + col]   ; All 32 threads, different addresses  
instruction_2:  fma result, a_val, b_val       ; All 32 threads, same operation
instruction_3:  add k, 1                       ; All 32 threads, same increment
instruction_4:  cmp k, 4                       ; All 32 threads, same comparison
instruction_5:  branch_if_less instruction_0   ; All 32 threads take same branch
```
There is no program counter per thread, that is instead per warp and is managed
by the warp scheduler. Each SM has an instruction cache which stores the compiled
CUDA kernel code (PTX->SASS machine code). The size of this instruction cache is
typically 32-128KB. This is what is meant by "lock step" execution, all 32
threads in a warp execute the same instruction at the same time, but on
different data.


```
Compute Unit Architecture:
┌─────────────────────────────────────────────────────┐
│ Compute Unit (256 threads max)                      │
│                                                     │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐     │
│ │SIMD Group 0 │ │SIMD Group 1 │ │SIMD Group 7 │     │
│ │(32 threads) │ │(32 threads) │ │(32 threads) │     │
│ └─────────────┘ └─────────────┘ └─────────────┘     │
│        │               │               │            │
│        ▼               ▼               ▼            │
│ ┌─────────────────────────────────────────────────┐ │
│ │        32 ALUs (Arithmetic Logic Units)         │ │
│ │     (Shared by SIMD groups via time-slicing)    │ │
│ └─────────────────────────────────────────────────┘ │
│                                                     │
│ ┌─────────────────────────────────────────────────┐ │
│ │         Shared Memory (32KB)                    │ │
│ │        (Shared by ALL 256 threads)              │ │
│ └─────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```
So we have 8 SIMD groups of 32 threads each, giving a total of 256 threads.

And there are 32 ALUs are what execute all the operations, computations, memory
loads/stores etc. And we can have 32 instructions executed in one cycle!

Examples of operations that the ALUs execute:
- Arithmetic         : add, multiply, fma operations
- Memory loads       : from global memory → registers
- Memory stores      : from registers → global memory
- Address calculation: computing memory offsets
- Control flow       : branch decisions, loop counters
- Atomic operations  : atomic_add, compare_and_swap

In CUDA these 32 threads that execute the same instruction are called a warp.
So all the threads in a warp execute the same instruction at the same time, but
on different data.
```
32 Physical ALUs → 32 threads minimum execution unit → "Warp/SIMD-group"
```
We can't efficiently use fewer than 32 threads (ALUs sit idle) and we can't
execute more the 32 threads simulationsly either (needs time slicing).
Now, not all GPUS have 32 ALUs, some have 16 and some have 64:
```
NVIDIA RTX 4090: 32 ALUs → 32-thread warps
AMD RX 7900    : 64 ALUs → 64-thread wavefronts  
Apple M3       : 32 ALUs → 32-thread SIMD-groups
Intel Arc      : 16 ALUs → 16-thread subgroups
```

So we have a compute using which can have up to 256 threads. Each thread has
its own 64 registers, but they are all from the same register file/memory, so
we have 256 * 64 = 16,384 registers in total, and the register size would then
be 65,536:
```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │     Physical Register File (e.g., 65,536 regs)   │   │
│  │  ┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐│   │
│  │  │ Reg0 ││ Reg1 ││ Reg2 ││ Reg3 ││ Reg4 ││ Reg5 ││   │
│  │  └──────┘└──────┘└──────┘└──────┘└──────┘└──────┘│   │
│  │  │  ...thousands more registers...               │   │
│  └──────────────────────────────────────────────────┘   │
│           ▲                                  ▲          │
│           │                                  │          │
│  ┌─────────────┐                    ┌─────────────┐     │
│  │   32 ALUs   │                    │  Register   │     │
│  │             │◄──────────────────►│  Crossbar   │     │
│  │ ALU0  ALU1  │                    │   Switch    │     │
│  │ ALU2  ALU3  │                    │             │     │
│  │    ...      │                    │             │     │
│  └─────────────┘                    └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```
Recall that each thread is executing the same instructions, but the registers
each thread uses are unique to that thread. So if we have an assembly load
instruction that uses a register like `R1`, then each thread will use its own
register.

8 SIMD groups of 32 threads each = 256 total threads, but only 1 SIMD group uses
the 32 ALUs at any given cycle:
```
Cycle 1: SIMD Group 0 (threads 0-31)    → all 32 ALUs busy
Cycle 2: SIMD Group 1 (threads 32-63)   → all 32 ALUs busy
Cycle 3: SIMD Group 2 (threads 64-95)   → all 32 ALUs busy
...
Cycle 8: SIMD Group 7 (threads 224-255) → all 32 ALUs busy
Cycle 9: Back to SIMD Group 0...
```

```
Registers    : 32 bit, ~1 cycle access time, 64 registers per thread
Shared memory: ~10-20 cycles access time, ~48-164 KB per block
Global memory: ~400-600 cycles access time, ~GBs in size
```

Now, lets say that we want to multiply a 4096x4096 matrix. 
```
               Matrix A
                 4096                      datatype: float (4 bytes)
     +--------------------------------+
     |                                |
     |                                |
     |                                |
     |                                |
     |                                |  4096
     |                                |
     |                                |
     |                                |
     |                                |
     |                                |
     +--------------------------------+

Total: 4096x4096 elements * 4 bytes/element = 67,108,864 bytes = 64 MB 
```
One row will be:
```
row size: 4096 elements * 4 bytes/element = 16,384 bytes = 16 KB
```
And one thread will need to access one row and one column to compute one element
of the result matrix. So one row and one column will be:
```
one row    = 16 KB (4096 elements per row, and each 4 bytes/element)
one column = 16 KB (4096 elements per column, and each 4 bytes/element)
Total      = 32 KB
```
We can't fit 32 KB into the threads register's which can be around 1KB.

So instead of one thread handling one complete dot product, using one row and
one column to compute one output result, threads in a block can cooperate and
compute a `tile` of the output matrix.
We can take a small tile of the input matrix at a time, for example a 16x16 tile:
```
        16       4096                      datatype: float (4 bytes)
     +-------+------------------------+
 16  |       |                        |
     |       |                        |
     +-------+                        |
     |                                |
     |                                |  4096
     |                                |
     |                                |
     |                                |
     |                                |
     |                                |
     +--------------------------------+
```
So we store this tile tile in `shared memory`:
```
16×16 A tile = 1KB (shared by ALL 256 threads)
16×16 B tile = 1KB (shared by ALL 256 threads)
Total in shared memory: 2KB
```

And each each thread will only holds a small slice of this in its registers:
```
16 elements from A tile = 64 bytes (one row from shared A tile)
16 elements from B tile = 64 bytes (one column from shared B tile)
Total: 128 bytes in registers for row and column

1 accumulator = 4 bytes
Temporary values = ~16 bytes
Total per thread: ~148 bytes (fits comfortably in ~1KB register space)
```
So each thread will need to load `64 + 64 = 128 bytes` from shared memory to its
registers. Then the thread will compute the dot product of these two 16 element
vectors to produce one output element which will be stored in an accumulator
register. So thread one will have the output for the first element in its
accumulator register, thread two will have the output for the second element
and so on. And each thread writes this output to global memory directly which
out having to go through shared memory.

So we will have 256 threads that will run in parallel to compute the output for
this matrix multiplication operation. ALL 256 threads do this `simultaneously`.
```
For k = 0 to 4096 step 16  (because 4096/16 = 256 iterations):
  
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

Now, lets take a broader look a the compute unit/streaming multiprocessor:
```
┌─────────────────────────────────────────────────────────────┐
│                    Compute Unit                             │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │        Instruction Cache & Fetch Unit                │   │
│  │     (Fetches instructions for all threads)           │   │
│  └──────────────────────────────────────────────────────┘   │
│                            │                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           Warp Scheduler                             │   │
│  │    (Decides which SIMD group executes next)          │   │
│  └──────────────────────────────────────────────────────┘   │
│                            │                                │
│  ┌─────────────┐  ┌──────────┐  ┌─────────────────────────┐ │
│  │    32x      │  │    4x    │  │    Register File        │ │
│  │   ALUs      │  │   SFUs   │  │    (65,536 regs)        │ │
│  │             │  │          │  │                         │ │
│  │ • Int/FP    │  │ • sin()  │  │                         │ │
│  │ • Logic     │  │ • cos()  │  │                         │ │
│  │ • Compare   │  │ • log()  │  │                         │ │
│  │ • FMA       │  │ • exp()  │  │                         │ │
│  └─────────────┘  │ • sqrt() │  └─────────────────────────┘ │
│                   │ • rsqrt()│                              │
│                   │ • rcp()  │                              │
│                   └──────────┘                              │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │               Shared Memory                          │   │
│  │        (32-48KB, user-managed cache)                 │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                L1 Cache                              │   │
│  │         (128-256KB, automatic caching)               │   │
│  └──────────────────────────────────────────────────────┘   │
│                            │                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Load/Store Units                        │   │
│  │        (Handle memory transactions)                  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Memory Subsystem                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  L2 Cache   │  │ Texture     │  │   Global Memory     │  │
│  │ (4-8MB)     │  │ Cache       │  │    (VRAM/RAM)       │  │
│  │ (shared)    │  │             │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Special Function Units (SFUs)
Fast approximate math operations (1-2 cycles):
* Transcendental functions: sin, cos, tan
* Exponentials: exp, exp2, log, log2
* Square roots: sqrt, rsqrt (reciprocal square root)
* Reciprocals: rcp (1/x)
* Some integer operations: __clz (count leading zeros)

Without SFUs sin(x) would take ~20 ALU instructions → 20 cycles per thread.
With SFUs sin(x) takes 1 SFU instruction → 8 cycles for 32 threads (80x speedup
for transcendental math!).

### Scheduler
Each cycle, the scheduler asks:
1. Which warps are ready to execute? (not waiting for memory/barriers)
2. What instruction does each ready warp need next?
3. Which execution units are available this cycle?
4. Assign ready warp → available execution unit

Now since all 32 threads execute the same instruction at the same time, each
thread does not have an instruction pointer register, but instead the warp/schduler
has this:
```
Warp State (stored in scheduler):
┌─────────────────────────────────────────────────────────┐
│ Warp 0:                                                 │
│ ├─ Program Counter (PC): 0x2040                         │
│ ├─ Active Mask: 11111111111111111111111111111111        │
│ ├─ State: READY/WAITING/STALLED                         │
│ └─ Next Instruction: ADD R1, R2, R3                     │
├─────────────────────────────────────────────────────────┤
│ Warp 1:                                                 │
│ ├─ Program Counter (PC): 0x2044                         │
│ ├─ Active Mask: 11111111111111111111111111111111        │
│ ├─ State: WAITING (for memory)                          │
│ └─ Next Instruction: LD R4, [global_addr]               │
└─────────────────────────────────────────────────────────┘
```
All instructions are stored in the instruction cache which is what each warps
program counter points to (the next instruction to execute for that warp.
```
┌─────────────────────────────────────────────────────────┐
│                 Compute Unit                            │
│                                                         │
│ ┌─────────────────────────────────────────────────────┐ │
│ │            Instruction Cache                        │ │
│ │        (Stores compiled kernel code)                │ │
│ └─────────────────────────────────────────────────────┘ │
│                           ▲                             │
│                           │                             │
│ ┌─────────────────────────────────────────────────────┐ │
│ │              Warp Scheduler                         │ │
│ │                                                     │ │
│ │ Warp 0: PC=0x2040 → fetch instruction at 0x2040     │ │
│ │ Warp 1: PC=0x2044 → fetch instruction at 0x2044     │ │
│ │ Warp 2: PC=0x2048 → fetch instruction at 0x2048     │ │
│ │ ...                                                 │ │
│ │                                                     │ │
│ │ Decision: Execute Warp 0 this cycle                 │ │
│ └─────────────────────────────────────────────────────┘ │
│                           │                             │
│                           ▼                             │
│ ┌─────────────────────────────────────────────────────┐ │
│ │                32 ALUs                              │ │
│ │ Execute "ADD R1, R2, R3" on all 32 threads          │ │
│ └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```
So this allows the scheduler to inspect the next instruction for a warp and
determine if is should run now or if it should wait. For example if the next
instruction is a memory load and the data is not yet available then the warp
will be put into a waiting state until the data arrives. This is important
because memory loads can take hundreds of cycles to complete, and we don't want
to waste cycles waiting for memory loads to complete.
The warp manages state something like this:
```c++
struct WarpState {
    uint32_t program_counter;      // Where this warp is in the code
    uint32_t active_mask;          // Which of the 32 threads are active  
    enum WarpStatus status;        // READY/WAITING/STALLED
    uint32_t wait_cycles;          // How long waiting for memory/barrier
    uint32_t last_instruction;     // For dependency tracking
};

WarpState warp_states[MAX_WARPS_PER_SM];  // Scheduler's state table
```
So the scheduler would have an array of these warp states. And for each warp
state that is in the READY state select one to execute based on some criteria
like round robin or prefer warps with instructions that matches available units
like SFUs or ALUs.

```
Warp States:
┌─────────────┐    instruction     ┌─────────────┐
│   Ready     │ ─────────────────► │  Executing  │
│             │                    │             │
└─────────────┘                    └─────────────┘
       ▲                                  │
       │                                  │ 
       │ data arrives                     │ memory request/
       │                                  │ barrier sync
       │                                  ▼
┌─────────────┐                    ┌─────────────┐
│   Stalled   │ ◄───────────────── │   Waiting   │
│             │    timeout/retry   │             │  
└─────────────┘                    └─────────────┘
```
```
Cycle 1: Warp 0 issues memory load → goes to "Waiting" state
Cycle 2: Warp 1 executes arithmetic → uses ALUs
Cycle 3: Warp 2 executes arithmetic → uses ALUs
...
Cycle 300: Warp 0's memory load completes → back to "Ready" state
Cycle 301: Warp 0 executes next instruction → uses ALUs
```

### Load/Store Units (LSUs)
This is a speciallized hardware component that handles all memory operations.
* address calculations
* cache access
* alignment handling
* atomic operations
* coalescing memory accesses from multiple threads (instead of performing individual
  memory operations for each thread, LSUs combine them into fewer, larger operations to
  one memory transaction which is more efficient).

Example of coalescing:
```
32 threads request memory:
Thread 0: address 0x1000
Thread 1: address 0x1004
Thread 2: address 0x1008
...
Thread 31: address 0x107C

LSU detects pattern sequential addresses, 4-byte aligned
→ Coalesces into ONE 128-byte transaction (0x1000-0x107F)
→ Single memory request instead of 32!

Bad pattern:
Thread 0: address 0x1000
Thread 1: address 0x2000  ← Far apart!
Thread 2: address 0x3000
...
→ LSU must issue 32 separate transactions
→ 32x slower memory access
```
Example of instructions that are handled by the LSUs:
```
float value = input[id];           // Load
output[id] = result;               // Store
atomicAdd(&counter, 1);            // Atomic (special LSU operation)
texture2D(tex, u, v);              // Texture load (specialized LSU)

// Shared memory access:
shared_data[tid] = value;          // Still uses LSU (different path)
```

### Grid, Block, Warp, Thread
When we launch a kernel in CUDA we use kernel launch configuration syntax, for
example:
```c++
op_clamp_kernel<<<num_blocks, CUDA_CLAMP_BLOCK_SIZE, 0   , stream>>>(x, dst, min, max, k);
//             <<<grid_dim  ,  block_dim           ,shmem, stream>>>
```
```
Grid (num_blocks = 4):
┌─────────────────────────────────────────────────────────────┐
│  Block 0      Block 1      Block 2      Block 3             │
│ ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐          │
│ │256 thds │  │256 thds │  │256 thds │  │256 thds │          │
│ │8 warps  │  │8 warps  │  │8 warps  │  │8 warps  │          │
│ └─────────┘  └─────────┘  └─────────┘  └─────────┘          │
└─────────────────────────────────────────────────────────────┘
```
So each warp has 32 threads, remember that the smallest unit of execution, and
32*8=256 threads per block. And we have 4 blocks in this case so we have
256*4=1024 threads in total.

```c++
const int i = blockDim.x * blockIdx.x + threadIdx.x;
                 256       0,1,2,3       0..255

Block 0: i = 256 * 0 + threadIdx.x = 0-255    (elements 0-255)
Block 1: i = 256 * 1 + threadIdx.x = 256-511  (elements 256-511)
Block 2: i = 256 * 2 + threadIdx.x = 512-767  (elements 512-767)
Block 3: i = 256 * 3 + threadIdx.x = 768-1023 (elements 768-1023)
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

### cuda-gdb
```console
$ cuda-gdb ./matrix-mul
NVIDIA (R) cuda-gdb 12.6
Portions Copyright (C) 2007-2024 NVIDIA Corporation
Based on GNU gdb 13.2

Reading symbols from ./matrix-mul...
(cuda-gdb) br matrix-mul.cu:4
(cuda-gdb) r

(cuda-gdb) info cuda devices
  Dev PCI Bus/Dev ID                    Name Description SM Type SMs Warps/SM Lanes/Warp Max Regs/Lane Active SMs Mask 
*   0        3c:00.0 NVIDIA GeForce RTX 4070     AD104-A   sm_89  46       48         32           255  0x000000000001

(cuda-gdb) info cuda sms
  SM  Active Warps Mask
Device 0
*  0 0x0000000000000001

(cuda-gdb) info cuda warps
  Wp Active Lanes Mask Divergent Lanes Mask          Active PC Kernel BlockIdx First Active ThreadIdx
Device 0 SM 0
   0        0xffffffff           0x00000000 0x00007fffdb2597f0      0  (0,0,0)                (0,0,0)


(cuda-gdb) info cuda threads
  BlockIdx ThreadIdx To BlockIdx To ThreadIdx Count      PC              Filename          Line
Kernel 0
*  (0,0,0)   (0,0,0)     (0,0,0)     (31,0,0)    32  0x00007fffdb2597b0 src/matrix-mul.cu   4

(cuda-gdb) info cuda blocks
  BlockIdx To BlockIdx Count   State
Kernel 0
*  (0,0,0)     (0,0,0)     1 running

(cuda-gdb) info cuda kernels
  Kernel Parent Dev Grid Status       SMs Mask GridDim BlockDim Invocation
*      0      -   0    1 Active 0x000000000001 (1,1,1) (32,1,1) matrix_mul()

(cuda-gdb) info cuda lanes
  Ln  State         PC         ThreadIdx Exception
Device 0 SM 0 Warp 0
*  0 active 0x00007fffdb2597b0   (0,0,0)    None
   1 active 0x00007fffdb2597b0   (1,0,0)    None
   2 active 0x00007fffdb2597b0   (2,0,0)    None
   3 active 0x00007fffdb2597b0   (3,0,0)    None
   4 active 0x00007fffdb2597b0   (4,0,0)    None
   5 active 0x00007fffdb2597b0   (5,0,0)    None
   6 active 0x00007fffdb2597b0   (6,0,0)    None
   7 active 0x00007fffdb2597b0   (7,0,0)    None
   8 active 0x00007fffdb2597b0   (8,0,0)    None
   9 active 0x00007fffdb2597b0   (9,0,0)    None
  10 active 0x00007fffdb2597b0  (10,0,0)    None
  11 active 0x00007fffdb2597b0  (11,0,0)    None
  12 active 0x00007fffdb2597b0  (12,0,0)    None
  13 active 0x00007fffdb2597b0  (13,0,0)    None
  14 active 0x00007fffdb2597b0  (14,0,0)    None
  15 active 0x00007fffdb2597b0  (15,0,0)    None
  16 active 0x00007fffdb2597b0  (16,0,0)    None
  17 active 0x00007fffdb2597b0  (17,0,0)    None
  18 active 0x00007fffdb2597b0  (18,0,0)    None
  19 active 0x00007fffdb2597b0  (19,0,0)    None
  20 active 0x00007fffdb2597b0  (20,0,0)    None
  21 active 0x00007fffdb2597b0  (21,0,0)    None
  22 active 0x00007fffdb2597b0  (22,0,0)    None
  23 active 0x00007fffdb2597b0  (23,0,0)    None
  24 active 0x00007fffdb2597b0  (24,0,0)    None
  25 active 0x00007fffdb2597b0  (25,0,0)    None
  26 active 0x00007fffdb2597b0  (26,0,0)    None
  27 active 0x00007fffdb2597b0  (27,0,0)    None
  28 active 0x00007fffdb2597b0  (28,0,0)    None
  29 active 0x00007fffdb2597b0  (29,0,0)    None
  30 active 0x00007fffdb2597b0  (30,0,0)    None
  31 active 0x00007fffdb2597b0  (31,0,0)    None
```
We can switch to a different lane/thread using:
```console
(cuda-gdb) cuda lane 1
```
This allows us to step and then switch to another thread to inspect that threads
specific values.

>>>>> Put this is some appropratate place later or delete it:
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
<<<<<
