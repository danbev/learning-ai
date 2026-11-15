### Streaming Assembly (SASS)


### Format
The following is a snippet from `vec_add.ptx` lowered Streaming Assembly (SASS)
```console
	code for sm_120
	.target	sm_120

		Function : vec_add
	.headerflags	@"EF_CUDA_SM120 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM89)"
        /*0000*/                   LDC R1, c[0x0][0x37c] ;                 /* 0x0000df00ff017b82 */
                                                                           /* 0x000fe20000000800 */
        /*0010*/                   S2R R0, SR_TID.X ;                      /* 0x0000000000007919 */
        ...
```
It starts with a `code` which specifies for which archtecture the code in this
file targets.
The `.target` looks like a directive and in this case it contains the same value
TODO: look into the difference and if this is in fact a directive.
Next we have the function name followed by the .headerflags which in this case
specify ELF metadata which tells the CUDA driver the pyhsical SASS target, SM120
and the virtual PTX architecture, SM89 it was compiled from.

After that we have the instructions. The start with a command which is the offset
and notice that this is 10, 16 in hex. Each instructions is 16 bytes so we have
increments of 0x10. The we have the instruction, and the comment on the rights
is the are the raw 120-bit (16-byte) encoded machine instructions themselves.

So that is the highlevel format of a .sass. Now lets look closer to the actual
instruction format:

```console
        /*0000*/                   LDC R1, c[0x0][0x37c] ;                 /* 0x0000df00ff017b82 */
                                                                           /* 0x000fe20000000800 */
```
This loading a constant into register R1, but where is this 'c' coming from?
The CUDA driver places all launch metadata and kernel arguments (param_a, param_b
param_c, and param_n in to a hardware cached region of memory called constant
memory, which can be accessed by c[bank][byte_offset].
Bank 0x0 is owned by the CUDA driver and every time we launch a kernel the driver
packs our kernel arguments and grid/block metadata into bank 0x0 before it starts
running the grid. This would look something like this for our example on sm_120:
```console
Offset     Size           Contains
0x360      4 bytes        blockDim.x
0x37c      4 bytes        Initial per thread ABI stack pointer offset.
0x380      8 bytes        param_a
0x388      8 bytes        param_b
0x390      8 bytes        param_c
0x398      8 bytes        param_n
```
R1 is reserved by convention for the stack pointer (SP). This is part of a
function prologue and might be present in most/all functions I think.

We can inspect this:
```console
$ make launcher
$ cuda-gdb --args launcher
(cuda-gdb) br vec_add
(cuda-gdb) c
Continuing.
[Switching focus to CUDA kernel 0, grid 1, block (0,0,0), thread (0,0,0), device 0, sm 0, warp 0, lane 0]

CUDA thread hit Breakpoint 2, 0x0000000325b66900 in vec_add<<<(4,1,1),(256,1,1)>>> ()
(cuda-gdb) disassemble 
Dump of assembler code for function vec_add:
=> 0x0000000325b66900 <+0>:	LDC R1,c[0x0][0x37c]

(cuda-gdb) p $R1
$2 = 0

(cuda-gdb) si
0x0000000325b66910 in vec_add<<<(4,1,1),(256,1,1)>>> ()
(cuda-gdb) p $R1
$3 = 16776640

(cuda-gdb) p /x $R1
$5 = 0xfffdc0
```
So this is the stack pointer (not the program counter just to avoid confusion)
and every GPU thread as its own private stack. This was introduced in sm_20
(Fermi) when NVIDA added function calls and recursion. But there not exactly the
same a the stacks we are used to in GPUs. Simliar to the CPU stack the SP on
the GPU grows downward. On this stack is layed out like this:
```
0x00000000 to 0x00ffffff
```
This is 16 MiB and the value we see is 0xfffdc0 is the top with some alignment

We don't have push/op instructions like on a CPU the implicitly modify RSP,
instead explicit IADD3 arithmetic on R1 paired with STL/LDL are used. Every
thread on a CPU has its own SP but that is not the case on the GPU:
```console
(cuda-gdb) cuda thread
thread (0,0,0)
```
Lets switch to a another thread and verify this:
```console
(cuda-gdb) cuda thread (1, 0, 0)
[Switching focus to CUDA kernel 0, grid 1, block (0,0,0), thread (1,0,0), device 0, sm 0, warp 0, lane 1]
0x0000000325b66910 in vec_add<<<(4,1,1),(256,1,1)>>> ()
(cuda-gdb) p /x $R1
$6 = 0xfffdc0
```
Indeed we have the same SP for both of them.

_wip_

```console
        /*0010*/                   S2R R0, SR_TID.X ;                      /* 0x0000000000007919 */
                                                                           /* 0x000e2e0000002100 */
        /*0020*/                   S2UR UR4, SR_CTAID.X ;                  /* 0x00000000000479c3 */
                                                                           /* 0x000e220000002500 */
        /*0030*/                   LDCU UR5, c[0x0][0x398] ;               /* 0x00007300ff0577ac */
                                                                           /* 0x000e6e0008000800 */
        /*0040*/                   LDC R9, c[0x0][0x360] ;                 /* 0x0000d800ff097b82 */
                                                                           /* 0x000e240000000800 */
        /*0050*/                   IMAD R9, R9, UR4, R0 ;                  /* 0x0000000409097c24 */
                                                                           /* 0x001fca000f8e0200 */
        /*0060*/                   ISETP.GE.U32.AND P0, PT, R9, UR5, PT ;  /* 0x0000000509007c0c */
                                                                           /* 0x002fda000bf06070 */
        /*0070*/               @P0 EXIT ;                                  /* 0x000000000000094d */
```
