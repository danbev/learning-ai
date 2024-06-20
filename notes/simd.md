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

### AVX instructions
```
__m256d _mm256_add_pd (__m256d a, __m256d b)                             vaddps
__m256 _mm256_add_ps (__m256 a, __m256 b)                                vaddsubpd
__m256d _mm256_addsub_pd (__m256d a, __m256d b)                          vaddsubps
__m256 _mm256_addsub_ps (__m256 a, __m256 b)                             vandpd
__m256d _mm256_and_pd (__m256d a, __m256d b)                             vandps
__m256 _mm256_and_ps (__m256 a, __m256 b)                                vandnpd
__m256d _mm256_andnot_pd (__m256d a, __m256d b)                          vandnps
__m256 _mm256_andnot_ps (__m256 a, __m256 b)                             vblendpd
__m256d _mm256_blend_pd (__m256d a, __m256d b, const int imm8)           vblendps
__m256 _mm256_blend_ps (__m256 a, __m256 b, const int imm8)              vblendvpd
__m256d _mm256_blendv_pd (__m256d a, __m256d b, __m256d mask)            vblendvps
__m256 _mm256_blendv_ps (__m256 a, __m256 b, __m256 mask)                vbroadcastf128
__m256d _mm256_broadcast_pd (__m128d const * mem_addr)                   vbroadcastf128
__m256 _mm256_broadcast_ps (__m128 const * mem_addr)                     vbroadcastsd
__m256d _mm256_broadcast_sd (double const * mem_addr)                    vbroadcastss
__m128 _mm_broadcast_ss (float const * mem_addr)                         vbroadcastss
__m256 _mm256_broadcast_ss (float const * mem_addr) __m256 _mm256_castpd_ps (__m256d a)
__m256i _mm256_castpd_si256 (__m256d a) __m256d _mm256_castpd128_pd256 (__m128d a)
__m128d _mm256_castpd256_pd128 (__m256d a) __m256d _mm256_castps_pd (__m256 a)
__m256i _mm256_castps_si256 (__m256 a) __m256 _mm256_castps128_ps256 (__m128 a)
__m128 _mm256_castps256_ps128 (__m256 a) __m256i _mm256_castsi128_si256 (__m128i a)
__m256d _mm256_castsi256_pd (__m256i a) __m256 _mm256_castsi256_ps (__m256i a)
__m128i _mm256_castsi256_si128 (__m256i a)                               vroundpd
__m256d _mm256_ceil_pd (__m256d a)                                       vroundps
__m256 _mm256_ceil_ps (__m256 a)                                         vcmppd
__m128d _mm_cmp_pd (__m128d a, __m128d b, const int imm8)                vcmppd
__m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)             vcmpps
__m128 _mm_cmp_ps (__m128 a, __m128 b, const int imm8)                   vcmpps
__m256 _mm256_cmp_ps (__m256 a, __m256 b, const int imm8)                vcmpsd
__m128d _mm_cmp_sd (__m128d a, __m128d b, const int imm8)                vcmpss
__m128 _mm_cmp_ss (__m128 a, __m128 b, const int imm8)                   vcvtdq2pd
__m256d _mm256_cvtepi32_pd (__m128i a)                                   vcvtdq2ps
__m256 _mm256_cvtepi32_ps (__m256i a)                                    vcvtpd2dq
__m128i _mm256_cvtpd_epi32 (__m256d a)                                   vcvtpd2ps
__m128 _mm256_cvtpd_ps (__m256d a)                                       vcvtps2dq
__m256i _mm256_cvtps_epi32 (__m256 a)                                    vcvtps2pd
__m256d _mm256_cvtps_pd (__m128 a)                                       vmovsd
double _mm256_cvtsd_f64 (__m256d a)                                      vmovd
int _mm256_cvtsi256_si32 (__m256i a)                                     vmovss
float _mm256_cvtss_f32 (__m256 a)                                        vcvttpd2dq
__m128i _mm256_cvttpd_epi32 (__m256d a)                                  vcvttps2dq
__m256i _mm256_cvttps_epi32 (__m256 a)                                   vdivpd
__m256d _mm256_div_pd (__m256d a, __m256d b)                             vdivps
__m256 _mm256_div_ps (__m256 a, __m256 b)                                vdpps
__m256 _mm256_dp_ps (__m256 a, __m256 b, const int imm8)
__int32 _mm256_extract_epi32 (__m256i a, const int index)
__int64 _mm256_extract_epi64 (__m256i a, const int index)                vextractf128
__m128d _mm256_extractf128_pd (__m256d a, const int imm8)                vextractf128
__m128 _mm256_extractf128_ps (__m256 a, const int imm8)                  vextractf128
__m128i _mm256_extractf128_si256 (__m256i a, const int imm8)             vroundpd
__m256d _mm256_floor_pd (__m256d a)                                      vroundps
__m256 _mm256_floor_ps (__m256 a)                                        vhaddpd
__m256d _mm256_hadd_pd (__m256d a, __m256d b)                            vhaddps
__m256 _mm256_hadd_ps (__m256 a, __m256 b)                               vhsubpd
__m256d _mm256_hsub_pd (__m256d a, __m256d b)                            vhsubps
__m256 _mm256_hsub_ps (__m256 a, __m256 b)
__m256i _mm256_insert_epi16 (__m256i a, __int16 i, const int index)
__m256i _mm256_insert_epi32 (__m256i a, __int32 i, const int index)
__m256i _mm256_insert_epi64 (__m256i a, __int64 i, const int index)
__m256i _mm256_insert_epi8 (__m256i a, __int8 i, const int index)        vinsertf128
__m256d _mm256_insertf128_pd (__m256d a, __m128d b, int imm8)            vinsertf128
__m256 _mm256_insertf128_ps (__m256 a, __m128 b, int imm8)               vinsertf128
__m256i _mm256_insertf128_si256 (__m256i a, __m128i b, int imm8)         vlddqu
__m256i _mm256_lddqu_si256 (__m256i const * mem_addr)                    vmovapd
__m256d _mm256_load_pd (double const * mem_addr)                         vmovaps
__m256 _mm256_load_ps (float const * mem_addr)                           vmovdqa
__m256i _mm256_load_si256 (__m256i const * mem_addr)                     vmovupd
__m256d _mm256_loadu_pd (double const * mem_addr)                        vmovups
__m256 _mm256_loadu_ps (float const * mem_addr)                          vmovdqu
__m256i _mm256_loadu_si256 (__m256i const * mem_addr)
__m256 _mm256_loadu2_m128 (float const* hiaddr, float const* loaddr)
__m256d _mm256_loadu2_m128d (double const* hiaddr, double const* loaddr)
__m256i _mm256_loadu2_m128i (__m128i const* hiaddr, __m128i const* loaddr) vmaskmovpd
__m128d _mm_maskload_pd (double const * mem_addr, __m128i mask)            vmaskmovpd
__m256d _mm256_maskload_pd (double const * mem_addr, __m256i mask)         vmaskmovps
__m128 _mm_maskload_ps (float const * mem_addr, __m128i mask)              vmaskmovps
__m256 _mm256_maskload_ps (float const * mem_addr, __m256i mask)           vmaskmovpd
void _mm_maskstore_pd (double * mem_addr, __m128i mask, __m128d a)         vmaskmovpd
void _mm256_maskstore_pd (double * mem_addr, __m256i mask, __m256d a)      vmaskmovps
void _mm_maskstore_ps (float * mem_addr, __m128i mask, __m128 a)           vmaskmovps
void _mm256_maskstore_ps (float * mem_addr, __m256i mask, __m256 a)        vmaxpd
__m256d _mm256_max_pd (__m256d a, __m256d b)                               vmaxps
__m256 _mm256_max_ps (__m256 a, __m256 b)                                  vminpd
__m256d _mm256_min_pd (__m256d a, __m256d b)                               vminps
__m256 _mm256_min_ps (__m256 a, __m256 b)                                  vmovddup
__m256d _mm256_movedup_pd (__m256d a)                                      vmovshdup
__m256 _mm256_movehdup_ps (__m256 a)                                       vmovsldup
__m256 _mm256_moveldup_ps (__m256 a)                                       vmovmskpd
int _mm256_movemask_pd (__m256d a)                                         vmovmskps
int _mm256_movemask_ps (__m256 a)                                          vmulpd
__m256d _mm256_mul_pd (__m256d a, __m256d b)                               vmulps
__m256 _mm256_mul_ps (__m256 a, __m256 b)                                  vorpd
__m256d _mm256_or_pd (__m256d a, __m256d b)                                vorps
__m256 _mm256_or_ps (__m256 a, __m256 b)                                   vpermilpd
__m128d _mm_permute_pd (__m128d a, int imm8)                               vpermilpd
__m256d _mm256_permute_pd (__m256d a, int imm8)                            vpermilps
__m128 _mm_permute_ps (__m128 a, int imm8)                                 vpermilps
__m256 _mm256_permute_ps (__m256 a, int imm8)                              vperm2f128
__m256d _mm256_permute2f128_pd (__m256d a, __m256d b, int imm8)            vperm2f128
__m256 _mm256_permute2f128_ps (__m256 a, __m256 b, int imm8)               vperm2f128
__m256i _mm256_permute2f128_si256 (__m256i a, __m256i b, int imm8)         vpermilpd
__m128d _mm_permutevar_pd (__m128d a, __m128i b)                           vpermilpd
__m256d _mm256_permutevar_pd (__m256d a, __m256i b)                        vpermilps
__m128 _mm_permutevar_ps (__m128 a, __m128i b)                             vpermilps
__m256 _mm256_permutevar_ps (__m256 a, __m256i b)                          vrcpps
__m256 _mm256_rcp_ps (__m256 a)                                            vroundpd
__m256d _mm256_round_pd (__m256d a, int rounding)                          vroundps
__m256 _mm256_round_ps (__m256 a, int rounding)                            vrsqrtps
__m256 _mm256_rsqrt_ps (__m256 a) ...
__m256i _mm256_set_epi16 (short e15, short e14, short e13, short e12, short e11, short e10, short e9, short e8, short e7, short e6, short e5, short e4, short e3, short e2, short e1, short e0)
__m256i _mm256_set_epi32 (int e7, int e6, int e5, int e4, int e3, int e2, int e1, int e0)
__m256i _mm256_set_epi64x (__int64 e3, __int64 e2, __int64 e1, __int64 e0)
__m256i _mm256_set_epi8 (char e31, char e30, char e29, char e28, char e27, char e26, char e25, char e24, char e23, char e22, char e21, char e20, char e19, char e18, char e17, char e16, char e15, char e14, char e13, char e12, char e11, char e10, char e9, char e8, char e7, char e6, char e5, char e4, char e3, char e2, char e1, char e0) vinsertf128
__m256 _mm256_set_m128 (__m128 hi, __m128 lo)                              vinsertf128
__m256d _mm256_set_m128d (__m128d hi, __m128d lo)                          vinsertf128
__m256i _mm256_set_m128i (__m128i hi, __m128i lo)
__m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
__m256 _mm256_set_ps (float e7, float e6, float e5, float e4, float e3, float e2, float e1, float e0)
__m256i _mm256_set1_epi16 (short a)
__m256i _mm256_set1_epi32 (int a)
__m256i _mm256_set1_epi64x (long long a)
__m256i _mm256_set1_epi8 (char a)
__m256d _mm256_set1_pd (double a)
__m256 _mm256_set1_ps (float a)
__m256i _mm256_setr_epi16 (short e15, short e14, short e13, short e12, short e11, short e10, short e9, short e8, short e7, short e6, short e5, short e4, short e3, short e2, short e1, short e0)
__m256i _mm256_setr_epi32 (int e7, int e6, int e5, int e4, int e3, int e2, int e1, int e0)
__m256i _mm256_setr_epi64x (__int64 e3, __int64 e2, __int64 e1, __int64 e0)
__m256i _mm256_setr_epi8 (char e31, char e30, char e29, char e28, char e27, char e26, char e25, char e24, char e23, char e22, char e21, char e20, char e19, char e18, char e17, char e16, char e15, char e14, char e13, char e12, char e11, char e10, char e9, char e8, char e7, char e6, char e5, char e4, char e3, char e2, char e1, char e0) vinsertf128
__m256 _mm256_setr_m128 (__m128 lo, __m128 hi)                                  vinsertf128
__m256d _mm256_setr_m128d (__m128d lo, __m128d hi)                              vinsertf128
__m256i _mm256_setr_m128i (__m128i lo, __m128i hi)
__m256d _mm256_setr_pd (double e3, double e2, double e1, double e0)
__m256 _mm256_setr_ps (float e7, float e6, float e5, float e4, float e3, float e2, float e1, float e0) vxorpd
__m256d _mm256_setzero_pd (void)                                                                       vxorps
__m256 _mm256_setzero_ps (void)                                                                        vpxor
__m256i _mm256_setzero_si256 (void)                                                                    vshufpd
__m256d _mm256_shuffle_pd (__m256d a, __m256d b, const int imm8)                                       vshufps
__m256 _mm256_shuffle_ps (__m256 a, __m256 b, const int imm8)                                          vsqrtpd
__m256d _mm256_sqrt_pd (__m256d a)                                                                     vsqrtps
__m256 _mm256_sqrt_ps (__m256 a)                                                                       vmovapd
void _mm256_store_pd (double * mem_addr, __m256d a)                                                    vmovaps
void _mm256_store_ps (float * mem_addr, __m256 a)                                                      vmovdqa
void _mm256_store_si256 (__m256i * mem_addr, __m256i a)                                                vmovupd
void _mm256_storeu_pd (double * mem_addr, __m256d a)                                                   vmovups
void _mm256_storeu_ps (float * mem_addr, __m256 a)                                                     vmovdqu
void _mm256_storeu_si256 (__m256i * mem_addr, __m256i a)
void _mm256_storeu2_m128 (float* hiaddr, float* loaddr, __m256 a)
void _mm256_storeu2_m128d (double* hiaddr, double* loaddr, __m256d a)
void _mm256_storeu2_m128i (__m128i* hiaddr, __m128i* loaddr, __m256i a)                                vmovntpd
void _mm256_stream_pd (void* mem_addr, __m256d a)                                                      vmovntps
void _mm256_stream_ps (void* mem_addr, __m256 a)                                                       vmovntdq
void _mm256_stream_si256 (void* mem_addr, __m256i a)                                                   vsubpd
__m256d _mm256_sub_pd (__m256d a, __m256d b)                                                           vsubps
__m256 _mm256_sub_ps (__m256 a, __m256 b)                                                              vtestpd
int _mm_testc_pd (__m128d a, __m128d b)                                                                vtestpd
int _mm256_testc_pd (__m256d a, __m256d b)                                                             vtestps
int _mm_testc_ps (__m128 a, __m128 b)                                                                  vtestps
int _mm256_testc_ps (__m256 a, __m256 b) vptest
int _mm256_testc_si256 (__m256i a, __m256i b) vtestpd
int _mm_testnzc_pd (__m128d a, __m128d b) vtestpd
int _mm256_testnzc_pd (__m256d a, __m256d b) vtestps
int _mm_testnzc_ps (__m128 a, __m128 b) vtestps
int _mm256_testnzc_ps (__m256 a, __m256 b) vptest
int _mm256_testnzc_si256 (__m256i a, __m256i b) vtestpd
int _mm_testz_pd (__m128d a, __m128d b) vtestpd
int _mm256_testz_pd (__m256d a, __m256d b) vtestps
int _mm_testz_ps (__m128 a, __m128 b) vtestps
int _mm256_testz_ps (__m256 a, __m256 b) vptest
int _mm256_testz_si256 (__m256i a, __m256i b)
__m256d _mm256_undefined_pd (void)
__m256 _mm256_undefined_ps (void)
__m256i _mm256_undefined_si256 (void) vunpckhpd
__m256d _mm256_unpackhi_pd (__m256d a, __m256d b) vunpckhps
__m256 _mm256_unpackhi_ps (__m256 a, __m256 b) vunpcklpd
__m256d _mm256_unpacklo_pd (__m256d a, __m256d b) vunpcklps
__m256 _mm256_unpacklo_ps (__m256 a, __m256 b) vxorpd
__m256d _mm256_xor_pd (__m256d a, __m256d b) vxorps
__m256 _mm256_xor_ps (__m256 a, __m256 b) vzeroall
void _mm256_zeroall (void) vzeroupper
void _mm256_zeroupper (void)
__m256d _mm256_zextpd128_pd256 (__m128d a)
__m256 _mm256_zextps128_ps256 (__m128 a)
__m256i _mm256_zextsi128_si256 (__m128i a)

```

### AVX2 instructions
```
__m256i _mm256_abs_epi16 (__m256i a) vpabsd
__m256i _mm256_abs_epi32 (__m256i a) vpabsb
__m256i _mm256_abs_epi8 (__m256i a) vpaddw
__m256i _mm256_add_epi16 (__m256i a, __m256i b) vpaddd
__m256i _mm256_add_epi32 (__m256i a, __m256i b) vpaddq
__m256i _mm256_add_epi64 (__m256i a, __m256i b) vpaddb
__m256i _mm256_add_epi8 (__m256i a, __m256i b) vpaddsw
__m256i _mm256_adds_epi16 (__m256i a, __m256i b) vpaddsb
__m256i _mm256_adds_epi8 (__m256i a, __m256i b) vpaddusw
__m256i _mm256_adds_epu16 (__m256i a, __m256i b) vpaddusb
__m256i _mm256_adds_epu8 (__m256i a, __m256i b) vpalignr
__m256i _mm256_alignr_epi8 (__m256i a, __m256i b, const int imm8) vpand
__m256i _mm256_and_si256 (__m256i a, __m256i b) vpandn
__m256i _mm256_andnot_si256 (__m256i a, __m256i b) vpavgw
__m256i _mm256_avg_epu16 (__m256i a, __m256i b) vpavgb
__m256i _mm256_avg_epu8 (__m256i a, __m256i b) vpblendw
__m256i _mm256_blend_epi16 (__m256i a, __m256i b, const int imm8) vpblendd
__m128i _mm_blend_epi32 (__m128i a, __m128i b, const int imm8) vpblendd
__m256i _mm256_blend_epi32 (__m256i a, __m256i b, const int imm8) vpblendvb
__m256i _mm256_blendv_epi8 (__m256i a, __m256i b, __m256i mask) vpbroadcastb
__m128i _mm_broadcastb_epi8 (__m128i a) vpbroadcastb
__m256i _mm256_broadcastb_epi8 (__m128i a) vpbroadcastd
__m128i _mm_broadcastd_epi32 (__m128i a) vpbroadcastd
__m256i _mm256_broadcastd_epi32 (__m128i a) vpbroadcastq
__m128i _mm_broadcastq_epi64 (__m128i a) vpbroadcastq
__m256i _mm256_broadcastq_epi64 (__m128i a) movddup
__m128d _mm_broadcastsd_pd (__m128d a) vbroadcastsd
__m256d _mm256_broadcastsd_pd (__m128d a) vbroadcasti128
__m256i _mm_broadcastsi128_si256 (__m128i a) vbroadcasti128
__m256i _mm256_broadcastsi128_si256 (__m128i a) vbroadcastss
__m128 _mm_broadcastss_ps (__m128 a) vbroadcastss
__m256 _mm256_broadcastss_ps (__m128 a) vpbroadcastw
__m128i _mm_broadcastw_epi16 (__m128i a) vpbroadcastw
__m256i _mm256_broadcastw_epi16 (__m128i a) vpslldq
__m256i _mm256_bslli_epi128 (__m256i a, const int imm8) vpsrldq
__m256i _mm256_bsrli_epi128 (__m256i a, const int imm8) vpcmpeqw
__m256i _mm256_cmpeq_epi16 (__m256i a, __m256i b) vpcmpeqd
__m256i _mm256_cmpeq_epi32 (__m256i a, __m256i b) vpcmpeqq
__m256i _mm256_cmpeq_epi64 (__m256i a, __m256i b) vpcmpeqb
__m256i _mm256_cmpeq_epi8 (__m256i a, __m256i b) vpcmpgtw
__m256i _mm256_cmpgt_epi16 (__m256i a, __m256i b) vpcmpgtd
__m256i _mm256_cmpgt_epi32 (__m256i a, __m256i b) vpcmpgtq
__m256i _mm256_cmpgt_epi64 (__m256i a, __m256i b) vpcmpgtb
__m256i _mm256_cmpgt_epi8 (__m256i a, __m256i b) vpmovsxwd
__m256i _mm256_cvtepi16_epi32 (__m128i a) vpmovsxwq
__m256i _mm256_cvtepi16_epi64 (__m128i a) vpmovsxdq
__m256i _mm256_cvtepi32_epi64 (__m128i a) vpmovsxbw
__m256i _mm256_cvtepi8_epi16 (__m128i a) vpmovsxbd
__m256i _mm256_cvtepi8_epi32 (__m128i a) vpmovsxbq
__m256i _mm256_cvtepi8_epi64 (__m128i a) vpmovzxwd
__m256i _mm256_cvtepu16_epi32 (__m128i a) vpmovzxwq
__m256i _mm256_cvtepu16_epi64 (__m128i a) vpmovzxdq
__m256i _mm256_cvtepu32_epi64 (__m128i a) vpmovzxbw
__m256i _mm256_cvtepu8_epi16 (__m128i a) vpmovzxbd
__m256i _mm256_cvtepu8_epi32 (__m128i a) vpmovzxbq

__m256i _mm256_cvtepu8_epi64 (__m128i a) ...
int _mm256_extract_epi16 (__m256i a, const int index)
...
int _mm256_extract_epi8 (__m256i a, const int index) vextracti128

__m128i _mm256_extracti128_si256 (__m256i a, const int imm8) vphaddw
__m256i _mm256_hadd_epi16 (__m256i a, __m256i b) vphaddd
__m256i _mm256_hadd_epi32 (__m256i a, __m256i b) vphaddsw
__m256i _mm256_hadds_epi16 (__m256i a, __m256i b) vphsubw
__m256i _mm256_hsub_epi16 (__m256i a, __m256i b) vphsubd
__m256i _mm256_hsub_epi32 (__m256i a, __m256i b) vphsubsw
__m256i _mm256_hsubs_epi16 (__m256i a, __m256i b) vpgatherdd
__m128i _mm_i32gather_epi32 (int const* base_addr, __m128i vindex, const int scale) vpgatherdd
__m128i _mm_mask_i32gather_epi32 (__m128i src, int const* base_addr, __m128i vindex, __m128i mask, const int scale) vpgatherdd
__m256i _mm256_i32gather_epi32 (int const* base_addr, __m256i vindex, const int scale) vpgatherdd
__m256i _mm256_mask_i32gather_epi32 (__m256i src, int const* base_addr, __m256i vindex, __m256i mask, const int scale) vpgatherdq
__m128i _mm_i32gather_epi64 (__int64 const* base_addr, __m128i vindex, const int scale) vpgatherdq
__m128i _mm_mask_i32gather_epi64 (__m128i src, __int64 const* base_addr, __m128i vindex, __m128i mask, const int scale) vpgatherdq
__m256i _mm256_i32gather_epi64 (__int64 const* base_addr, __m128i vindex, const int scale) vpgatherdq
__m256i _mm256_mask_i32gather_epi64 (__m256i src, __int64 const* base_addr, __m128i vindex, __m256i mask, const int scale) vgatherdpd
__m128d _mm_i32gather_pd (double const* base_addr, __m128i vindex, const int scale) vgatherdpd
__m128d _mm_mask_i32gather_pd (__m128d src, double const* base_addr, __m128i vindex, __m128d mask, const int scale) vgatherdpd
__m256d _mm256_i32gather_pd (double const* base_addr, __m128i vindex, const int scale) vgatherdpd
__m256d _mm256_mask_i32gather_pd (__m256d src, double const* base_addr, __m128i vindex, __m256d mask, const int scale) vgatherdps
__m128 _mm_i32gather_ps (float const* base_addr, __m128i vindex, const int scale) vgatherdps
__m128 _mm_mask_i32gather_ps (__m128 src, float const* base_addr, __m128i vindex, __m128 mask, const int scale) vgatherdps
__m256 _mm256_i32gather_ps (float const* base_addr, __m256i vindex, const int scale) vgatherdps
__m256 _mm256_mask_i32gather_ps (__m256 src, float const* base_addr, __m256i vindex, __m256 mask, const int scale) vpgatherqd
__m128i _mm_i64gather_epi32 (int const* base_addr, __m128i vindex, const int scale) vpgatherqd
__m128i _mm_mask_i64gather_epi32 (__m128i src, int const* base_addr, __m128i vindex, __m128i mask, const int scale) vpgatherqd
__m128i _mm256_i64gather_epi32 (int const* base_addr, __m256i vindex, const int scale) vpgatherqd
__m128i _mm256_mask_i64gather_epi32 (__m128i src, int const* base_addr, __m256i vindex, __m128i mask, const int scale) vpgatherqq
__m128i _mm_i64gather_epi64 (__int64 const* base_addr, __m128i vindex, const int scale) vpgatherqq
__m128i _mm_mask_i64gather_epi64 (__m128i src, __int64 const* base_addr, __m128i vindex, __m128i mask, const int scale) vpgatherqq
__m256i _mm256_i64gather_epi64 (__int64 const* base_addr, __m256i vindex, const int scale) vpgatherqq
__m256i _mm256_mask_i64gather_epi64 (__m256i src, __int64 const* base_addr, __m256i vindex, __m256i mask, const int scale) vgatherqpd
__m128d _mm_i64gather_pd (double const* base_addr, __m128i vindex, const int scale) vgatherqpd
__m128d _mm_mask_i64gather_pd (__m128d src, double const* base_addr, __m128i vindex, __m128d mask, const int scale)
vgatherqpd
__m256d _mm256_i64gather_pd (double const* base_addr, __m256i vindex, const int scale) vgatherqpd
__m256d _mm256_mask_i64gather_pd (__m256d src, double const* base_addr, __m256i vindex, __m256d mask, const int scale) vgatherqps
__m128 _mm_i64gather_ps (float const* base_addr, __m128i vindex, const int scale) vgatherqps
__m128 _mm_mask_i64gather_ps (__m128 src, float const* base_addr, __m128i vindex, __m128 mask, const int scale) vgatherqps
__m128 _mm256_i64gather_ps (float const* base_addr, __m256i vindex, const int scale) vgatherqps
__m128 _mm256_mask_i64gather_ps (__m128 src, float const* base_addr, __m256i vindex, __m128 mask, const int scale) vinserti128
__m256i _mm256_inserti128_si256 (__m256i a, __m128i b, const int imm8) vpmaddwd
__m256i _mm256_madd_epi16 (__m256i a, __m256i b) vpmaddubsw
__m256i _mm256_maddubs_epi16 (__m256i a, __m256i b) vpmaskmovd
__m128i _mm_maskload_epi32 (int const* mem_addr, __m128i mask) vpmaskmovd
__m256i _mm256_maskload_epi32 (int const* mem_addr, __m256i mask) vpmaskmovq
__m128i _mm_maskload_epi64 (__int64 const* mem_addr, __m128i mask) vpmaskmovq
__m256i _mm256_maskload_epi64 (__int64 const* mem_addr, __m256i mask) vpmaskmovd
void _mm_maskstore_epi32 (int* mem_addr, __m128i mask, __m128i a) vpmaskmovd
void _mm256_maskstore_epi32 (int* mem_addr, __m256i mask, __m256i a) vpmaskmovq
void _mm_maskstore_epi64 (__int64* mem_addr, __m128i mask, __m128i a) vpmaskmovq
void _mm256_maskstore_epi64 (__int64* mem_addr, __m256i mask, __m256i a) vpmaxsw
__m256i _mm256_max_epi16 (__m256i a, __m256i b) vpmaxsd
__m256i _mm256_max_epi32 (__m256i a, __m256i b) vpmaxsb
__m256i _mm256_max_epi8 (__m256i a, __m256i b) vpmaxuw
__m256i _mm256_max_epu16 (__m256i a, __m256i b) vpmaxud
__m256i _mm256_max_epu32 (__m256i a, __m256i b) vpmaxub
__m256i _mm256_max_epu8 (__m256i a, __m256i b) vpminsw
__m256i _mm256_min_epi16 (__m256i a, __m256i b) vpminsd
__m256i _mm256_min_epi32 (__m256i a, __m256i b) vpminsb
__m256i _mm256_min_epi8 (__m256i a, __m256i b) vpminuw
__m256i _mm256_min_epu16 (__m256i a, __m256i b) vpminud
__m256i _mm256_min_epu32 (__m256i a, __m256i b) vpminub
__m256i _mm256_min_epu8 (__m256i a, __m256i b) vpmovmskb
int _mm256_movemask_epi8 (__m256i a) vmpsadbw
__m256i _mm256_mpsadbw_epu8 (__m256i a, __m256i b, const int imm8) vpmuldq
__m256i _mm256_mul_epi32 (__m256i a, __m256i b) vpmuludq
__m256i _mm256_mul_epu32 (__m256i a, __m256i b) vpmulhw
__m256i _mm256_mulhi_epi16 (__m256i a, __m256i b) vpmulhuw
__m256i _mm256_mulhi_epu16 (__m256i a, __m256i b) vpmulhrsw
__m256i _mm256_mulhrs_epi16 (__m256i a, __m256i b) vpmullw
__m256i _mm256_mullo_epi16 (__m256i a, __m256i b) vpmulld
__m256i _mm256_mullo_epi32 (__m256i a, __m256i b) vpor
__m256i _mm256_or_si256 (__m256i a, __m256i b) vpacksswb
__m256i _mm256_packs_epi16 (__m256i a, __m256i b) vpackssdw
__m256i _mm256_packs_epi32 (__m256i a, __m256i b) vpackuswb
__m256i _mm256_packus_epi16 (__m256i a, __m256i b) vpackusdw
__m256i _mm256_packus_epi32 (__m256i a, __m256i b) vperm2i128
__m256i _mm256_permute2x128_si256 (__m256i a, __m256i b, const int imm8) vpermq
__m256i _mm256_permute4x64_epi64 (__m256i a, const int imm8) vpermpd
__m256d _mm256_permute4x64_pd (__m256d a, const int imm8) vpermd
__m256i _mm256_permutevar8x32_epi32 (__m256i a, __m256i idx) vpermps
__m256 _mm256_permutevar8x32_ps (__m256 a, __m256i idx) vpsadbw
__m256i _mm256_sad_epu8 (__m256i a, __m256i b) vpshufd
__m256i _mm256_shuffle_epi32 (__m256i a, const int imm8) vpshufb
__m256i _mm256_shuffle_epi8 (__m256i a, __m256i b) vpshufhw
__m256i _mm256_shufflehi_epi16 (__m256i a, const int imm8) vpshuflw
__m256i _mm256_shufflelo_epi16 (__m256i a, const int imm8) vpsignw
__m256i _mm256_sign_epi16 (__m256i a, __m256i b) vpsignd
__m256i _mm256_sign_epi32 (__m256i a, __m256i b) vpsignb
__m256i _mm256_sign_epi8 (__m256i a, __m256i b) vpsllw
__m256i _mm256_sll_epi16 (__m256i a, __m128i count) vpslld
__m256i _mm256_sll_epi32 (__m256i a, __m128i count) vpsllq
__m256i _mm256_sll_epi64 (__m256i a, __m128i count) vpsllw
__m256i _mm256_slli_epi16 (__m256i a, int imm8) vpslld
__m256i _mm256_slli_epi32 (__m256i a, int imm8) vpsllq
__m256i _mm256_slli_epi64 (__m256i a, int imm8) vpslldq
__m256i _mm256_slli_si256 (__m256i a, const int imm8) vpsllvd
__m128i _mm_sllv_epi32 (__m128i a, __m128i count) vpsllvd
__m256i _mm256_sllv_epi32 (__m256i a, __m256i count) vpsllvq
__m128i _mm_sllv_epi64 (__m128i a, __m128i count) vpsllvq
__m256i _mm256_sllv_epi64 (__m256i a, __m256i count) vpsraw
__m256i _mm256_sra_epi16 (__m256i a, __m128i count) vpsrad
__m256i _mm256_sra_epi32 (__m256i a, __m128i count) vpsraw
__m256i _mm256_srai_epi16 (__m256i a, int imm8) vpsrad
__m256i _mm256_srai_epi32 (__m256i a, int imm8) vpsravd
__m128i _mm_srav_epi32 (__m128i a, __m128i count) vpsravd
__m256i _mm256_srav_epi32 (__m256i a, __m256i count) vpsrlw
__m256i _mm256_srl_epi16 (__m256i a, __m128i count) vpsrld
__m256i _mm256_srl_epi32 (__m256i a, __m128i count) vpsrlq
__m256i _mm256_srl_epi64 (__m256i a, __m128i count) vpsrlw
__m256i _mm256_srli_epi16 (__m256i a, int imm8) vpsrld
__m256i _mm256_srli_epi32 (__m256i a, int imm8) vpsrlq
__m256i _mm256_srli_epi64 (__m256i a, int imm8) vpsrldq
__m256i _mm256_srli_si256 (__m256i a, const int imm8) vpsrlvd
__m128i _mm_srlv_epi32 (__m128i a, __m128i count) vpsrlvd
__m256i _mm256_srlv_epi32 (__m256i a, __m256i count) vpsrlvq
__m128i _mm_srlv_epi64 (__m128i a, __m128i count) vpsrlvq
__m256i _mm256_srlv_epi64 (__m256i a, __m256i count) vmovntdqa
__m256i _mm256_stream_load_si256 (void const* mem_addr) vpsubw
__m256i _mm256_sub_epi16 (__m256i a, __m256i b) vpsubd
__m256i _mm256_sub_epi32 (__m256i a, __m256i b) vpsubq
__m256i _mm256_sub_epi64 (__m256i a, __m256i b) vpsubb
__m256i _mm256_sub_epi8 (__m256i a, __m256i b) vpsubsw
__m256i _mm256_subs_epi16 (__m256i a, __m256i b) vpsubsb
__m256i _mm256_subs_epi8 (__m256i a, __m256i b) vpsubusw
__m256i _mm256_subs_epu16 (__m256i a, __m256i b) vpsubusb
__m256i _mm256_subs_epu8 (__m256i a, __m256i b) vpunpckhwd
__m256i _mm256_unpackhi_epi16 (__m256i a, __m256i b) vpunpckhdq
__m256i _mm256_unpackhi_epi32 (__m256i a, __m256i b) vpunpckhqdq
__m256i _mm256_unpackhi_epi64 (__m256i a, __m256i b) vpunpckhbw
__m256i _mm256_unpackhi_epi8 (__m256i a, __m256i b) vpunpcklwd
__m256i _mm256_unpacklo_epi16 (__m256i a, __m256i b) vpunpckldq
__m256i _mm256_unpacklo_epi32 (__m256i a, __m256i b) vpunpcklqdq
__m256i _mm256_unpacklo_epi64 (__m256i a, __m256i b) vpunpcklbw
__m256i _mm256_unpacklo_epi8 (__m256i a, __m256i b) vpxor
__m256i _mm256_xor_si256 (__m256i a, __m256i b)
```

### Instructions common to AVX and AVX2
```
AVX  : __m256 _mm256_broadcast_ps (__m128 const * mem_addr) vbroadcastsd
AVX2 : __m128d _mm_broadcastsd_pd (__m128d a)               vbroadcastsd

AVX  : __m256d _mm256_broadcast_sd (double const * mem_addr) vbroadcastss
AVX  : __m256d _mm256_broadcast_sd (double const * mem_addr) vbroadcastss
AVX  : __m128 _mm_broadcast_ss (float const * mem_addr)      vbroadcastss
AVX2 : __m256i _mm256_broadcastsi128_si256 (__m128i a)       vbroadcastss
AVX2 : __m128 _mm_broadcastss_ps (__m128 a)                  vbroadcastss

AVX  : __m256 _mm256_setzero_ps (void)                       vpxor
AVX2 : __m256i _mm256_unpacklo_epi8 (__m256i a, __m256i b)   vpxor

### Order of instruction set extensions flags
This section is about GCC and an issue that I have encountered related to the
order of machine specific options (`-m`).

The intention of this test is to check the availability of the SSSE3
macro is being set correctly. This is sensitive to the order of options
specified to the compiler. For example, if -mssse3 is specified and later
-mavx is specified, the SSSE3 macro will be defined which can be somewhat
surprising. This is not really an issue with this simple test and one
Makefile but in a larger project using CMake and including multiple
directories all with their own CMakeLists.txt files, it can be difficult
to ensure that the correct flags are being passed to the compiler. Whan even
more concerning is that you probably won't notice this issue until you
run the code or inspect it.

### 
```c
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)
// multiply int8_t, add results pairwise twice
static inline __m128i mul_sum_i8_pairs(const __m128i x, const __m128i y) {
    // Get absolute values of x vectors. This instruction is available in SSSE3
    const __m128i ax = _mm_sign_epi8(x, x);

    // Sign the values of the y vectors. This instruction is available in SSSE3
    const __m128i sy = _mm_sign_epi8(y, x);

    // Perform multiplication and create 16-bit values. This instruction is available in SSSE3
    const __m128i dot = _mm_maddubs_epi16(ax, sy);

    // Sets all elements of the vector 128-bit integers to 1. From SSE2
    const __m128i ones = _mm_set1_epi16(1);

    // Multiply packed signed 16-bit integers, from SSE2
    return _mm_madd_epi16(ones, dot);
}
```
If we dump the assembly code for this function we see something interesting:
```console
000000000048609f <mul_sum_i8_pairs>:
  48609f:	55                   	push   rbp
  4860a0:	48 89 e5             	mov    rbp,rsp
  4860a3:	48 81 ec 88 00 00 00 	sub    rsp,0x88
  4860aa:	c5 f9 7f 85 10 ff ff 	vmovdqa XMMWORD PTR [rbp-0xf0],xmm0
  4860b1:	ff 
  4860b2:	c5 f9 7f 8d 00 ff ff 	vmovdqa XMMWORD PTR [rbp-0x100],xmm1
  4860b9:	ff 
  4860ba:	c5 f9 6f 85 10 ff ff 	vmovdqa xmm0,XMMWORD PTR [rbp-0xf0]
  4860c1:	ff 
  4860c2:	c5 f9 7f 45 e0       	vmovdqa XMMWORD PTR [rbp-0x20],xmm0
  4860c7:	c5 f9 6f 85 10 ff ff 	vmovdqa xmm0,XMMWORD PTR [rbp-0xf0]
  4860ce:	ff 
  4860cf:	c5 f9 7f 45 f0       	vmovdqa XMMWORD PTR [rbp-0x10],xmm0
  4860d4:	c5 f9 6f 4d f0       	vmovdqa xmm1,XMMWORD PTR [rbp-0x10]
  4860d9:	c5 f9 6f 45 e0       	vmovdqa xmm0,XMMWORD PTR [rbp-0x20]
  4860de:	c4 e2 79 08 c1       	vpsignb xmm0,xmm0,xmm1
  4860e3:	c5 f9 7f 85 40 ff ff 	vmovdqa XMMWORD PTR [rbp-0xc0],xmm0
  4860ea:	ff 
  4860eb:	c5 f9 6f 85 00 ff ff 	vmovdqa xmm0,XMMWORD PTR [rbp-0x100]
  4860f2:	ff 
  4860f3:	c5 f9 7f 45 c0       	vmovdqa XMMWORD PTR [rbp-0x40],xmm0
  4860f8:	c5 f9 6f 85 10 ff ff 	vmovdqa xmm0,XMMWORD PTR [rbp-0xf0]
  4860ff:	ff 
  486100:	c5 f9 7f 45 d0       	vmovdqa XMMWORD PTR [rbp-0x30],xmm0
  486105:	c5 f9 6f 4d d0       	vmovdqa xmm1,XMMWORD PTR [rbp-0x30]
  48610a:	c5 f9 6f 45 c0       	vmovdqa xmm0,XMMWORD PTR [rbp-0x40]
  48610f:	c4 e2 79 08 c1       	vpsignb xmm0,xmm0,xmm1
  486114:	c5 f9 7f 85 50 ff ff 	vmovdqa XMMWORD PTR [rbp-0xb0],xmm0
  48611b:	ff 
  48611c:	c5 f9 6f 85 40 ff ff 	vmovdqa xmm0,XMMWORD PTR [rbp-0xc0]
  486123:	ff 
  486124:	c5 f9 7f 45 a0       	vmovdqa XMMWORD PTR [rbp-0x60],xmm0
  486129:	c5 f9 6f 85 50 ff ff 	vmovdqa xmm0,XMMWORD PTR [rbp-0xb0]
  486130:	ff 
  486131:	c5 f9 7f 45 b0       	vmovdqa XMMWORD PTR [rbp-0x50],xmm0
  486136:	c5 f9 6f 4d b0       	vmovdqa xmm1,XMMWORD PTR [rbp-0x50]
  48613b:	c5 f9 6f 45 a0       	vmovdqa xmm0,XMMWORD PTR [rbp-0x60]
  486140:	c4 e2 79 04 c1       	vpmaddubsw xmm0,xmm0,xmm1
  486145:	c5 f9 7f 85 60 ff ff 	vmovdqa XMMWORD PTR [rbp-0xa0],xmm0
  48614c:	ff 
```
Notice the `vpmaddubsw` instruction which is only availabe in AVX and AVX2.
```
486140:	c4 e2 79 04 c1       	vpmaddubsw xmm0,xmm0,xmm1
```
Now, the `xmm0` and `xmm1` registers are 128-bit registers and the `vpmaddubsw`
and AVX2 uses 256-bit registers `ymm0, ymm1...ymm31`. This is a clear indication
