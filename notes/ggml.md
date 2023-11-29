## GGML (Georgi Gerganov Machine Learning)
"GG" refers to the initials of its originator (Georgi Gerganov) and I think
ML is just for machine learning. It is a
[C library](https://github.com/rustformers/llm/blob/main/crates/ggml/README.md).

This is a basic example in [ggml/c-example](fundamentals/ggml/c-example/README.md)
of how to use GGML.

For me it help to draw parallels between GGML and part1 of zero-to-hero, where
we created a Value struct which in addition of holding a value, would also have
a gradient, and an operation and children if the value was created by an
operation. The Value struct also supported automatic differentiation. In a
simliar manner GGML has a `tensor` struct which holds a value, a gradient, and
an operation and a `src` array which is simlar to the children in zero-to-hero
in that is would contain the left hand side and right hand side of an operation.

The tensor struct also supports ability do generate dot draphs. GGML contains
more than that but then tensor one of the is the basic structure.

GGML files contain binary-encoded data, including version number,
hyperparameters, vocabulary, and weights.

So GGML is used by llama.cpp, and whisper.cpp. In addition it GGML is what the
is used by the Rust `llm` crate. So learning about GGML will help understand
all of these project better.

### GGML walk through

```console
$ gdb --args ./main -m models/llama-2-13b-chat.Q4_0.gguf --prompt "What is LoRA?</s>"

(gdb) br
Breakpoint 2 at 0x429ea4: file ggml.c, line 4589.

(gdb) bt
#0  ggml_init (params=...) at ggml.c:4589
#1  0x00000000004802b5 in llama_backend_init (numa=false) at llama.cpp:6623
#2  0x000000000040b235 in main (argc=5, argv=0x7fffffffd1c8) at examples/main/main.cpp:171

(gdb) p ggml_init
$25 = {struct ggml_context *(struct ggml_init_params)} 0x429d89 <ggml_init>
```
So the function `ggml_init` returns an instance of `struct ggml_context`
initialized with the parameters. The `struct ggml_context` is an opaque struct
so we can simply use `ptype ggml_context` to see what it contains. But we can
list it using:
```console
(gdb) list ggml.c:4188
4183	
4184	//
4185	// ggml context
4186	//
4187	
4188	struct ggml_context {
4189	    size_t mem_size;
4190	    void * mem_buffer;
4191	    bool   mem_buffer_owned;
4192	    bool   no_alloc;
(gdb) 
4193	    bool   no_alloc_save; // this is used to save the no_alloc state when using scratch buffers
4194	
4195	    int    n_objects;
4196	
4197	    struct ggml_object * objects_begin;
4198	    struct ggml_object * objects_end;
4199	
4200	    struct ggml_scratch scratch;
4201	    struct ggml_scratch scratch_save;
4202	};
```

Turning our attention back to the `ggml_init` function we can see that it
has the following initialization of some tables:
```console
gdb) l
4579	        // initialize GELU, Quick GELU, SILU and EXP F32 tables
4580	        {
4581	            const uint64_t t_start = ggml_time_us(); UNUSED(t_start);
4582	
4583	            ggml_fp16_t ii;
4584	            for (int i = 0; i < (1 << 16); ++i) {
4585	                uint16_t ui = i;
4586	                memcpy(&ii, &ui, sizeof(ii));
4587	                const float f = table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(ii);
4588	                table_gelu_f16[i] = GGML_FP32_TO_FP16(ggml_gelu_f32(f));
4589	                table_gelu_quick_f16[i] = GGML_FP32_TO_FP16(ggml_gelu_quick_f32(f));
4590	                table_silu_f16[i] = GGML_FP32_TO_FP16(ggml_silu_f32(f));
4591	                table_exp_f16[i]  = GGML_FP32_TO_FP16(expf(f));
4592	            }
```
So lets take a closer look at what is going on here. First we have the `ii`
variable which is of type `ggml_fp16_t` which is defined as:
```c
typedef uint16_t ggml_fp16_t;
```
On my machine this is an unsiged 16 bit integer.
```console
(gdb) ptype ii
type = unsigned short
(gdb) p sizeof(ii)
$11 = 2 (bytes)
```
So this is a 16 bit unsigned integer, that is 2 bytes and the range is 0 to
65535.
Next we have a look which will iterate over 65536 elements:
```console
(gdb) p/d (1 << 16)
$8 = 65536
(gdb) p/t (1 << 16)
$7 = 10000000000000000
```
The first thing that happens in the loop is that the integer i, note that this
is a 32 bit integer (4 bytes), is copied into the 16 bit unsigned integer ui.
```
0000 0001 0000 0000 0000 0000 0000 0000 = 65536
                    0000 0000 0000 0000 = 0
```
Notice that we are discarding the upper 16 bits of the 32 bit integer. And
recall that our loop only iterates 65536 times. So the value of ui will be
0, 1, 2, 3, ..., 65535. Next we have the memcpy:
```console
4586	                memcpy(&ii, &ui, sizeof(ii));
```
The usage of `memcpy` was not obvious to me at first and I thought that after
looking at the actual types used a simple assignment would be sufficient:
```c
    ii = (ggml_fp16_t)ui;  // Implicit conversion
```
But `memcpy` is used instead to safely copy bits from the `uint16_t` variable
`ui` to the `ggml_fp16_t` variable `ii` without relying on implicit type
conversion which might not behave as desired, especially if `ggml_fp16_t` and
`uint16_t` have different internal representations. `memcpy` ensures abit-wise
copy, preserving the exact bit pattern from `ui` to `ii` over the specified
number of bytes (`sizeof(ii)`), which is crucial when dealing with different
data types or when precise control over memory is required.

Next we have the following line:
```console
4587  const float f = table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(ii);
```
This is a compound assignment and is equivalent to:
```c
    table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(ii);
    const float f = table_f32_f16[i];
```
So 16-bit floats, or half-floats are popular for storing data in memory but
computations are typically done using 32-bit floats. The macro
`GGML_COMPUTE_FP16_TO_FP32` is used to convert the 16-bit float to a 32-bit
float. The macro is defined as:
```c
#define GGML_COMPUTE_FP16_TO_FP32(x) _cvtsh_ss(x)                               
```
Notice that this uses an intrinsic function `_cvtsh_ss`, so it is implemented
by some low-level code that can directly use special hardware operations to
convert the 16-bit float to a 32-bit float. Alright, so we now understand that
we are storing values as 16-bit floats in memory but we are converting them to
32-bit floats when we need to do computations.

The table `table_f32_f16` is an array and contains all possible 16-bit values
(from 0 to 65535), which cover all possible half-precision floating-point
numbers.
```
static float table_f32_f16[65536];
```
And we are poplulating this array with the values of the
`GGML_COMPUTE_FP16_TO_FP32` macro above which will be single-precision floating
point values which are 32-bits. Later when we need to do computations we can
just cast a ggml_fp16_t value to an unsigned short and use that as an index
to get the single-precision 32-bit value without having to do any conversions.

So if we have a value of type
`ggml_fp16_t` we should be able to look that up in the table. After four
iterations the table will look like this (only showing the first 5 elements):
```console
(gdb) p table_f32_f16[0]
$43 = 0

(gdb) p table_f32_f16[1]
$44 = 5.96046448e-08

(gdb) p table_f32_f16[2]
$45 = 1.1920929e-07

(gdb) p table_f32_f16[3]
$46 = 1.78813934e-07

(gdb) p table_f32_f16[4]
$47 = 0
```
If we take a look at the second element it is 5.96046448e-08, which is the
smallest positive value that can be represented by a 32-bit float.

```console
(gdb) x table_f32_f16 
0x6011e0 <table_f32_f16>:	00000000000000000000000000000000

(gdb) p table_f32_f16[1]
$69 = 5.96046448e-08
(gdb) x/w table_f32_f16 +  1
0x6011e4 <table_f32_f16+4>:	00110011100000000000000000000000
```
Lets take the entry of index 1. We have a 32-bit float, which is 4 bytes:
```
   0011 0011 1000 0000 0000 0000 0000 0000
   0 01100111 000 0000 0000 0000 0000 0000
   ↑    ↑      ↑   ↑    ↑    ↑    ↑    ↑
   S=0  E= 103   M= 00000000000000000000000

S = sign bit
E = exponent, 8 bits
M = mantissa, 23 bits

(-1)ˢ × 2ᴱ⁻¹⁵⁷ × 1.F

(-1)⁰ × 2¹⁰³⁻¹²⁷ × 1.00000000000000000000000
= 2¹⁰³⁻¹²⁷
= 2⁻²⁴
= 5.96046448e-08
```

```
gdb) p table_f32_f16[2]
$70 = 1.1920929e-07
```
First we write 1.92e-07 in m x 2^e format:
```
1.1920929 × 10^−7 = 1.0101010100 ×2^−23
```
In IEEE 754 single-precision, an exponent bias of 127 is used, so the stored
exponent value is −23 + 127 = 104, or 01101000 in binary.
Recall that the exponent field is 8 bits, so we can have values between 0 and
255. But we need to be able to represnt negative exponents as well, What well
do is subtract 127 from the 8 bit binary value to get the actual exponent.
```
Binary Exponent	Unsigned Integer	Bias	Actual Exponent
00000000        0	                127	    -127
00000001        1	                127	    -126
00000010        2	                127	    -125
...             ...                 ...     ...
01111110        126	                127	     -1
01111111        127	                127	      0
10000000        128	                127	      1
10000001        129	                127	      2
...             ...                 ...     ...
11111110	    254	                127	    127
11111111	    255	                127	    (Reserved)
```
The binary exponent value of 11111111 (255 in decimal) is reserved for special
values like infinity and NaN (Not a Number), so it's not used for normal
floating-point numbers.

In IEEE 754 single precision floating point format, 1.0101010100 ×2^−23 is
represented as:
```
   0011 0100 0000 0000 0000 0000 0000 0000
   0 01101000 000 0000 0000 0000 0000 0000
   ↑    ↑      ↑   ↑    ↑    ↑    ↑    ↑
   S=0  E= 104   M= 00000000000000000000000


(gdb) x/t table_f32_f16 + 2
0x6011e8 <table_f32_f16+8>:	00110100000000000000000000000000

```
(gdb) p table_f32_f16[3]
$111 = 1.78813934e-07

(gdb) p/t table_f32_f16[3]
$110 = 110100010000000000000000000000

```
After that detour we have:
```console
4588	                table_gelu_f16[i] = GGML_FP32_TO_FP16(ggml_gelu_f32(f));
```
Here we are populating a lookup table with the values of the `ggml_gelu_f32`
converted to half-precision floating point values. The `ggml_gelu_f32` function
looks like this:
```console
3812	inline static float ggml_gelu_f32(float x) {
3813	    return 0.5f*x*(1.0f + tanhf(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)));
3814	}
```
This is done for all the possibe half-precision floating point numbers.
Similarlty this is also done for other activation functions.

Lets turn our attention to the llama_init_from_gpt_params function:
```console
```console
std::tie(model, ctx) = llama_init_from_gpt_params(params);
```

### Tensor
