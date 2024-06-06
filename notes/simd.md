## Single Instruction, Multiple Data (SIMD)
Intel and AMD processors use AVX and AVX3 instructions to perform SIMD
operations. ARM processors use NEON instructions.


### Introduction
SIMD utilizes special registers which can hold 128, 256, 512, or even 1024 bits
of data. The register used is divided into smaller blocks of 8, 16, 32, or 64
bits and perform the same operation on all the blocks simultaneously.

Now, the processor itself needs to have physical support for the instructions,
and the compiler needs to be able to generate the instructions. In addition
the operating system needs to be able to save the state of the registers so 
it also needs to have some support (though this is at runtime not compile time).

### Compiler flags
The following flags can be used with the compiler. To see what extensions are
available we can use:
```console
$ gcc --target-help
```

* -mavx512vl (Vector Length Extension) enables AVX512 ops on 256/128 bit regs
* -mavx512bw (Byte and Word) enables AVX512 ops on 512 bit regs
* -mavx512vbmi (Vector Byte Manipulation Instructions) extens the existing AVX512
instruction set byte and work operations (8/16 bit operations).
* -mfma (Fused Multiply-Add) enables FMA instructions
* -mf16c (Half Precision Floating Point Conversion) provides support for
converting between 32-bit single precision floats and 16-bit half precision
floats.

* -march=native (Optimize for the host processor) enables all instruction set
extensions supported by the host processor. The compiler probes the host
system's processor for its capabilities like using cpuid and then adds then
appropriate compiler flags that are available on the current system.

### Matrix Math Extension (MMX) 1997
Registers (64-bit):
* MM0-MM7

Defines 8 64-bit MMX registers (MM0-MM7) which can hold 64-bit integers. So we
could have 2 32-bit integers in each register or 4 16-bit integers, or 8 8-bit
integers. MMX only provides integer operations.

### 3dNow! 1998
From AMD which addes 32-bit floating point operations to the MMX registers.

### Streaming SIMD Extensions (SSE) 1999
Registers (128-bit):
* XMM0-XMM7

Notice that the registers have a new name.
Increases the width of the SIMD registers from 64 to 128 bits. Introduced
double-precision floating point instructions in addition to the single-precision
floating point and integer instructions found in SSE. 
70 instructions in total.

### SSE2 2000
Registers (128-bit):
* XMM0-XMM7

144 new instructions in total.

### Advanced Vector Extensions (AVX)
Registers (256-bit):
* YMM0-YMM7

Notice that the registers have a new name.
AVX increases the width of the SIMD registers from 128 to to 256 bits. The
registers were renamed from XMM0–XMM7 to YMM0–YMM7. It adds 8 new registers
(YMM0-YMM15) which are 256 bits wide so each one can hold 8 32-bit floats
(8*32=256), 4 64-bit (4*64=256) doubles.

```
-mavx  Support MMX, SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2 and AVX built-in functions and code generation
```

#### Functions format
```
_mm<bit_width>_<name<data_type>
```
`_mm` is a prefix for all SIMD functions. The `<bit_width>` is the width of the
return type register. The `<name>` is the name of the function and the
`<data_type>` is the data type of the function arguments:
Data types:
* __m128     128-bit register 4 floats
* __m128d    128-bit register 2 doubles
* __m128i    128-bit register 4 integers (char, short, int, long, long long)
* __m256     256-bit register 8 floats
* __m256d    256-bit register 4 doubles
* __m256i    256-bit register 8 integers

In the function names `ps` stands for packed single precision (floats), `pd`
stands for packed double precision (doubles), `epi` stands for extended packed
integer, `si` stands for scalar integer, `sd` stands for scalar double, `ss`
stands for scalar single, `epi8` stands for extended packed integer 8-bit.

### maskload
The idea here is that one might have a data type that does not fill a complete
vector register. In this one can add a mask to the operation to only operate
on the elements that are set in the mask. For examples:
```c
#include <stdio.h>
#include <immintrin.h>

int main() {
    int i;
    int int_array[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    
    __m256i mask = _mm256_setr_epi32(-1, -1, -1, -1, 0, 0, 0, 0);
    
    __m256i result = _mm256_maskload_epi32(int_array, mask);
    
    int* res = (int*)&result;
    printf("%d %d %d %d %d %d %d %d\n", 
        res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7]);
    
    return 0;
}
```
This will generate:
```console
$ ./bin/masking 
1 2 3 4 0 0 0 0
```
My initial though would be that -1 would be used to indicate that the elements
should be included in the operation but the reason for using -1 is that that
produces a 1 in all bits of the integer and the most signficant bit is 1 so this
is a simple check to be performed. This is using -1 so the processor can check
the MSB and if it is 1 then it knows to include this element, and if zero it
knows not to. Lets say we used 0 to determine if the element should be included
instead then it would have to check the first bit if it is 0 and also the other
bits to make sure they are all zeros as well which would be more work.


### immintrin.h
This header provides immidate access to SIMD instructions. Immediate in that one
only has to include this single header file to get access to all the SIMD and
then compiler flags will determine which headers to include.
```c
#include <x86gprintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <wmmintrin.h>
#include <avxintrin.h>
```

### Instructions
When compling for x86 architectures (Intel and AMD) we can specify compiler
flags to limit the instructions set to a specific instruction set. For example
to enable AVX we can specify `-mavx` and to enable AVX2 we can specify `-mavx2`.
We can also explicitely disable AVX2 by specifying `-mno-avx2`.

But how do we verify that our code is actually using the instructions we think
it is using?  

We can use 'objdump' to disassemble the binary and look for the instructions
we are interested in. For example:
```console
$ objdump -d somthing.so | grep vbroadcastss

230231:  123fc3:	c4 e2 7d 18 55 b0    	vbroadcastss ymm2,DWORD PTR [rbp-0x50]
```
Now, AVX uses 256-bit registers which are YMM0-YMM15 and AVX2 uses 512-bit
which are ZMM0-ZMM31. So if we see YMM2 in the disassembly then we know that
Now AVX is mostly focused on floating point operations and AVX2 extends this to
integer operations but still uses the YMM registers.

