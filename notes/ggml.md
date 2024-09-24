## GGML (Georgi Gerganov Machine Learning)
"GG" refers to the initials of its originator (Georgi Gerganov) and I think
ML is just for machine learning. It is a
[C library](https://github.com/rustformers/llm/blob/main/crates/ggml/README.md).

This is a basic example in [ggml](fundamentals/ggml/README.md) of how to use
GGML.

For me it helps to draw parallels between GGML and part1 of [zero-to-hero],
where we created a Value struct which in addition of holding a value, would also
have a gradient, an operation, and children if the value was created by an
operation. The Value struct also supported automatic differentiation. In a
simliar manner GGML has a `tensor` struct which holds a value, a gradient, an
operation and a `src` array which is simlar to the children in zero-to-hero
in that is would contain the left hand side and right hand side of an operation.

In the case of zero-to-hero when we perform an operation on a Value the
operation itself will create a new Value which will contain the output of the
operation, and it will also "connect" this with the left and right operands.
This will form a graph of Values which we can then at some point backpropagate
through. In GGML the graph is created upfront before the actual operation is
performed. 

The tensor struct also supports ability to generate dot draphs. GGML contains
more than that, but the tensor is one of the the basic structure.

GGML model files contain binary-encoded data, including version number,
hyperparameters, vocabulary, and weights.

So GGML is used by llama.cpp, and whisper.cpp. In addition GGML is what is the
used by the Rust `llm` crate, and also by llm-chain. So learning about GGML will
help understand all of these project better.

### Memory usage
We start off by specifying the memory that GGML will use. This is done by
creating a `struct ggml_init_params` and passing it to the `ggml_init`:
```c 
  struct ggml_init_params params = {
    .mem_size   = 16*1024*1024, // 16 MB
    .mem_buffer = NULL,
    .no_alloc   = false,
  };
  struct ggml_context* ctx = ggml_init(params);
```
So in this case we will have 16 MB of memory available for GGML to use.
For this example I'm using [graph.c](../fundamentals/ggml/src/graph.c) and
the next thing it does is it creates a graph:
```c
  struct ggml_cgraph* c_graph = ggml_new_graph(ctx);
```
This call will end up in:
```c
struct ggml_cgraph * ggml_new_graph(struct ggml_context * ctx) {
    return ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);
}
```
The GGML_DEFAULT_GRAPH_SIZE is 2048, and `false` is for the `grad` parameter in
that gradients should not be computed.

Next the size of the object will be computed
```c
struct ggml_cgraph * ggml_new_graph_custom(struct ggml_context * ctx, size_t size, bool grads) {
    const size_t obj_size = ggml_graph_nbytes(size, grads);
```
```console
(gdb) p obj_size
$3 = 65640 
```

```c
struct ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_GRAPH, obj_size);

    enum ggml_object_type {
        GGML_OBJECT_TENSOR,
        GGML_OBJECT_GRAPH,
        GGML_OBJECT_WORK_BUFFER
    };
```
So we can see that 3 objects types are currently supported.

```c
static struct ggml_object * ggml_new_object(struct ggml_context* ctx,
    enum ggml_object_type type, size_t size) {
    // always insert objects at the end of the context's memory pool
    struct ggml_object * obj_cur = ctx->objects_end;

    const size_t cur_offs = obj_cur == NULL ? 0 : obj_cur->offs;
    const size_t cur_size = obj_cur == NULL ? 0 : obj_cur->size;
    const size_t cur_end  = cur_offs + cur_size;

    // align to GGML_MEM_ALIGN
    size_t size_needed = GGML_PAD(size, GGML_MEM_ALIGN);

    char * const mem_buffer = ctx->mem_buffer;
    struct ggml_object * const obj_new = (struct ggml_object *)(mem_buffer + cur_end);
```
This is the first thing to be added to the memory that was allocated in the
`ggml_init` call. So this is creating a pointer to a `ggml_object` which will be
the first thing in the memory area that was allocated. 

So what is a `ggml_object`?  
It is defined as:
```c
struct ggml_object {
    size_t offs;
    size_t size;
    struct ggml_object * next;
    enum ggml_object_type type;
    char padding[4];
};
```
I think the `offs` is the offset into `mem_buffer`, and size is the size of that
memory.

To get a visual of this perhaps we can think about the memory as the following
initially: 
```
    mem_buffer
       ↓
       +---------------------------------------------------------
       |                                                      ... 
       +---------------------------------------------------------
```
Then we dedicate a portion of this memory area to an `ggml_object`:
```
    mem_buffer + cur_end
       ↓
       +---------------------------------------------------------
       | ggml_object          |                               ... 
       +---------------------------------------------------------
       ↑
     obj_new
```
And this memory is then updated:
```c
    *obj_new = (struct ggml_object) {
        .offs = cur_end + GGML_OBJECT_SIZE,
        .size = size_needed,
        .next = NULL,
        .type = type,
    };
```
And just to verify this:
```console
(gdb) p *(struct ggml_object*) mem_buffer
$10 = {offs = 32, size = 65648, next = 0x0, type = GGML_OBJECT_GRAPH, padding = "\000\000\000"}
(gdb) p sizeof(struct ggml_object)
$11 = 32
```
Next, if the current object is not null (which is not the case currently in
this debugging session) we will update the next pointer of the current object
to point to the new object.
```c
    if (obj_cur != NULL) {
        obj_cur->next = obj_new;
    } else {
        // this is the first object in this context
        ctx->objects_begin = obj_new;
    }

    ctx->objects_end = obj_new;
```
And on the last line above we update the context `objects_end` to point to the
newly created object. So now our memory looks like this:
```
    mem_buffer
       ↓
       +---------------------------------------------------------
       | ggml_object          |                               ... 
       +---------------------------------------------------------
       ↑
     obj_new
       ↑
     ctx->objects_end
```
After this the pointer to the new object is returned.

Just to recap where we were:
```console
(gdb) bt
#0  ggml_new_object (ctx=0x7ffff7fba008 <g_state+8>, type=GGML_OBJECT_GRAPH, size=65640)
    at /home/danielbevenius/work/ai/learning-ai/fundamentals/ggml/ggml/src/ggml.c:2446
#1  0x00007ffff7ec0cee in ggml_new_graph_custom (ctx=0x7ffff7fba008 <g_state+8>, size=2048, grads=false)
    at /home/danielbevenius/work/ai/learning-ai/fundamentals/ggml/ggml/src/ggml.c:15635
#2  0x00007ffff7ec0ed0 in ggml_new_graph (ctx=0x7ffff7fba008 <g_state+8>)
    at /home/danielbevenius/work/ai/learning-ai/fundamentals/ggml/ggml/src/ggml.c:15670
#3  0x00000000004012d0 in main (argc=1, argv=0x7fffffffd0b8) at src/graph.c:19
```
So we were in `ggml_new_graph_custom`:
```c
    struct ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_GRAPH, obj_size);
    struct ggml_cgraph * cgraph = (struct ggml_cgraph *) ((char *) ctx->mem_buffer + obj->offs);
```
So we created an obj and this and `ctx->mem_buffer` currently point to the same
location:
```console
(gdb) p ctx->mem_buffer
$18 = (void *) 0x7ffff6ba0010
(gdb) p obj
$19 = (struct ggml_object *) 0x7ffff6ba0010
(gdb) p obj->offs
$13 = 32
```
And now we are creating a pointer to a `ggml_cgraph` (compute graph) which is
pointing to `mem_buffer + 32`. So we have the obj which is currently pointing to
`mem_buffer` because this is the first object created, and every object as data
as in `ggml_object` which we can sort of think of as a base struct. And then
after the base struct we can have different "subtypes". In the above case we are
saying that we memory area after the `obj->off` is going to store a `ggml_cgraph`
struct.
```
    mem_buffer
       ↓
       +---------------------------------------------------------
       | ggml_object     |ggml_cgraph |                       ... 
       +---------------------------------------------------------
       ↑                 ↑
     obj_new            off
       ↑
     ctx->objects_end
```
The `ggml_cgraph` struct is defined as:
```console
(gdb) ptype struct ggml_cgraph
type = struct ggml_cgraph {
    int size;
    int n_nodes;
    int n_leafs;
    struct ggml_tensor** nodes;
    struct ggml_tensor** grads;
    struct ggml_tensor** leafs;
    struct ggml_hash_set visited_hash_table;
    enum ggml_cgraph_eval_order order;
    int perf_runs;
    int64_t perf_cycles;
    int64_t perf_time_us;
}

(gdb) p *cgraph
$28 = {size = 0, n_nodes = 0, n_leafs = 0, nodes = 0x0, grads = 0x0, leafs = 0x0, visited_hash_table = {size = 0, 
    keys = 0x0}, order = GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT, perf_runs = 0, perf_cycles = 0, perf_time_us = 0}
```
Next, we have:
```c
    struct ggml_tensor** data_start = (struct ggml_tensor **) (cgraph + 1);
```
This is creating a pointer to a pointer to a `ggml_tensor` and the location will
be cgraph + 80 because the size of the `ggml_cgraph` is 80 bytes:
```console
(gdb) p sizeof(struct ggml_cgraph)
$30 = 80
(gdb) p cgraph 
$31 = (struct ggml_cgraph *) 0x7ffff6ba0030
(gdb) p cgraph + 1
$32 = (struct ggml_cgraph *) 0x7ffff6ba0080
(gdb) p 0x7ffff6ba0080 - 0x7ffff6ba0030
$33 = 80
```
So this will be a pointer to the end of the `cgraph` struct in memory.
```c
    size_t hash_size = ggml_hash_size(size * 2);
    struct ggml_tensor** nodes_ptr = data_start;

    struct ggml_tensor** leafs_ptr = nodes_ptr + size;

    struct ggml_tensor** hash_keys_ptr = leafs_ptr + size;

    struct ggml_tensor** grads_ptr = grads ? hash_keys_ptr + hash_size : NULL;
```
`hash_size` will be:
```
15640	    size_t hash_size = ggml_hash_size(size * 2);
Value returned is $36 = 4099
```
Next we can see that `nodes_ptr` will point to the same location as
`data_start`. Next, `leafs_ptr` will be set to point to `nodes_ptr + size` which
is:
```console
(gdb) p sizeof(struct ggml_tensor**)
$45 = 8
(gdb) p size * 8
$46 = 16384

(gdb) p leafs_ptr 
$48 = (struct ggml_tensor **) 0x7ffff6ba4080
(gdb) p notes_ptr
No symbol "notes_ptr" in current context.
(gdb) p nodes_ptr
$49 = (struct ggml_tensor **) 0x7ffff6ba0080
(gdb) p 0x7ffff6ba4080 - 0x7ffff6ba0080
$50 = 16384
```
So, that is it will point to the memory after all the `nodes_ptrs`.
And something similar will happen for `hash_keys_ptr` and `grads_ptr` (unless
grads is null).

Next the hash keys will be zeroed out:
```c
    memset(hash_keys_ptr, 0, hash_size * sizeof(struct ggml_tensor *));
```
And then the cgraph will be initialized and returned:
```c
    *cgraph = (struct ggml_cgraph) {
        /*.size         =*/ size,
        /*.n_nodes      =*/ 0,
        /*.n_leafs      =*/ 0,
        /*.nodes        =*/ nodes_ptr,
        /*.grads        =*/ grads_ptr,
        /*.leafs        =*/ leafs_ptr,
        /*.hash_table   =*/ { hash_size, hash_keys_ptr },
        /*.order        =*/ GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT,
        /*.perf_runs    =*/ 0,
        /*.perf_cycles  =*/ 0,
        /*.perf_time_us =*/ 0,
    };

    return cgraph;
```
That will return us back in our program graph.c.

Now, lets take a look at how a <span id="tensor_mem">tensor</span>
tensor is represented in memory:
```c
  struct ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
```
```console
(gdb) l
2566	struct ggml_tensor * ggml_new_tensor_1d(
2567	        struct ggml_context* ctx,
2568	        enum ggml_type type,
2569	        int64_t ne0) {
2570	    return ggml_new_tensor(ctx, type, 1, &ne0);
2571	}
```
`ne0` is the number of elements in the first dimension. 
```console
(gdb) l
2558	struct ggml_tensor * ggml_new_tensor(
2559	        struct ggml_context * ctx,
2560	        enum   ggml_type      type,
2561	        int                   n_dims,
2562	        const int64_t       * ne) {
2563	    return ggml_new_tensor_impl(ctx, type, n_dims, ne, NULL, 0);
2564	}

(gdb) l
2461	
2462	static struct ggml_tensor * ggml_new_tensor_impl(
2463	        struct ggml_context* ctx,
2464	        enum ggml_type type,
2465	        int n_dims,
2466	        const int64_t* ne,
2467	        struct ggml_tensor* view_src,
2468	        size_t view_offs) {
```
Notice that the last two arguments are NULL and 0 which are the `view_src` and
`view_offs`.
Skipping some things about views which I'm not familiar with yet and also
skipping the scratch buffer.
TODO: take a closer look at views and scratch buffers.

Next, we have something that might now look familar:
```c
    struct ggml_object * const obj_new = ggml_new_object(ctx,
        GGML_OBJECT_TENSOR,
        GGML_TENSOR_SIZE + obj_alloc_size);
```
But this time instead of a `GGML_OBJECT_GRAPH` we are creating a
`GGML_OBJECT_TENSOR`.
```console
(gdb) p sizeof(struct ggml_tensor)
$60 = 384
(gdb) p obj_alloc_size
$61 = 4
(gdb) p sizeof(struct ggml_tensor) + obj_alloc_size
$62 = 388
```

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
so we can't simply use `ptype ggml_context` to see what it contains. But we can
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

### Tensors
When reading about matrix multiplication we often see it in the format that
we create a matrix like 3x2 which means 3 rows and 2 columns. When working
with GGML, and I think this is also common with graphics libraries in general,
that one first specifies the x-axis, that is the horizontal axis/number of
columns.
If we have multiple dimensions then we have another value that specifies the
size of the y-axis, that is the vertical axis/number of rows. So think of this
as building a matrix from the bottom up and specifying one dimension at a time.

So if we want to create a 3x2 matrix in GGML we do the following:
```c
  struct ggml_tensor* matrix = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 3);
```
This is because we are first specifying the number of elements in the first
dimension (x-axis), and then the number of elements in the second dimension
(y-axis).

Which can be visualized like this:
```
    +---+---+
    | 0 | 1 |
    +---+---+
    | 2 | 3 |
    +---+---+
    | 4 | 5 |
    +---+---+ 

ne[0] = 2 (Number of elements in the first dimension) 
ne[1] = 3 (Number of elements in the second dimension)

nb[0] = 4 (size of each element, moving this number will move to the next column)
nb[1] = 8 (stride, moving this number will move to the next row)

Memory layout:
0000 0001    0010 0011    0100 0101
  0    1      2     3       4    5
    row 1      row 2        row 3
             ↑
             8 (ne[1])
```  

### views
If we inspect a tensor we can see that it contains the following:
```console
(gdb) p *x
$7 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0, ne = {10, 1, 1, 1}, nb = {
    4, 40, 40, 40}, op = GGML_OP_NONE, op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0, 
  src = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, perf_runs = 0, perf_cycles = 0, 
  perf_time_us = 0, view_src = 0x0, view_offs = 0, data = 0x7ffff6a001a0, 
  name = '\000' <repeats 63 times>, extra = 0x0, padding = "\000\000\000\000\000\000\000"}
```
So this is storing 10 elements of type float (GGML_TYPE_F32):
```console
(gdb) p ((float*) x->data)[0]
$11 = 1
(gdb) p ((float*) x->data)[1]
$12 = 2
(gdb) p ((float*) x->data)[2]
$13 = 3
(gdb) p ((float*) x->data)[4]
$14 = 5
(gdb) p ((float*) x->data)[3]
$15 = 4
(gdb) p ((float*) x->data)[9]
$16 = 10
```
Now if we create a 1d view of this tensor using:
```c
  struct ggml_tensor* view = ggml_view_1d(ctx, x, 5, (5-1) * ggml_type_size(x->type));
```
Where the first int argument is the number of elements and the second integer is
the offset in bytes. Notice that we have take into account the size for the
elements stored. Keep in mind that the index is zero based so the offset should
reflect this too.

So if our x tensor is:
```console
x[0]: 1.000000
x[1]: 2.000000
x[2]: 3.000000
x[3]: 4.000000
x[4]: 5.000000
x[5]: 6.000000
x[6]: 7.000000
x[7]: 8.000000
x[8]: 9.000000
x[9]: 10.000000
```
Then the view would be:
```console
view[0]: 5.000000
view[1]: 6.000000
view[2]: 7.000000
view[3]: 8.000000
view[4]: 9.000000
```
The element type in this case if 4. And what we are saying above is that we
want a view of 5 elements starting at the offset 16:
```
 0     4    8   12   16   20   24   28   32   36 
 [0    1    2    3    4    5    6    7    8    9]
 [1    2    3    4    5    6    7    8    9   10]

```


```c
    struct ggml_tensor * result = ggml_view_impl(ctx, a, 1, &ne0, offset);
```

```c
static struct ggml_tensor * ggml_view_impl(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   n_dims,
        const int64_t       * ne,
        size_t                offset) {
    ...
       
    struct ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, n_dims, ne, a, offset);
    ggml_format_name(result, "%s (view)", a->name);
```
If we step into this function we can see the arguments are as follows:
```console
(gdb) s
ggml_new_tensor_impl (ctx=0x7ffff7fba2a8 <g_state+8>, type=GGML_TYPE_F32, n_dims=1, ne=0x7fffffffc7e8, 
    view_src=0x7ffff6a00030, view_offs=5)
    at /home/danielbevenius/work/ai/learning-ai/fundamentals/ggml/ggml/src/ggml.c:2928
```
Notice that `view_src` is a pointer to the `x` tensor:
```console
(gdb) p x
$20 = (struct ggml_tensor *) 0x7ffff6a00030
(gdb) down
(gdb) p a
$21 = (struct ggml_tensor *) 0x7ffff6a00030
```

[zero-to-hero]: ../fundamentals/rust/zero-to-hero/README.md

### Backend
Exploration code can be found in [backend.c](../fundamentals/ggml/src/backend.c).

What is a backend in ggml?  
A backend in ggml is an interface which describes and abstracts operations on a
buffer. Recall that a buffer is a contiguous block of memory of fixed size and
intended to hold data while moving that data between places (in this case the
host and a device). And a buffer has a fixed size. This buffer can be on an
accelerator, like a GPU, or on the host.

All backends implement the same interface which enables them to be used in a
uniform way and there can be multiple backends available at the same time.

The backend interface is declared in `ggml/include/ggml/ggml-backend.h` and this
header contains the functions of the interface to the backend. The actual
backend is an opaque pointer:
```c
    typedef struct ggml_backend * ggml_backend_t;
```
This is a way of implementing an abstract data type in C and allows for
encapsulation. The actual implementation of the backend is hidden from the usero
of the backend. 

The actual definition of `ggml_backend` can be found in
ggml/src/ggml-backend-impl.h:
```c
    struct ggml_backend {
        ggml_guid_t guid;

        struct ggml_backend_i iface;
        ggml_backend_context_t context;
    };

    typedef void * ggml_backend_context_t;
```
So a backend has a global unique identifier (guid), an interface and a context.
And notice that the context can be anything, since it is a void pointer.

The backend interface, `iface` above, is what defines the operations that are
available for a backend which every backend must implement.
`struct ggml_backend_i` has functions like (simplified for readability):
```c
    const char* get_name();
    void free();

    ggml_backend_buffer_type_t get_default_buffer_type();
    void set_tensor_async(struct ggml_tensor* tensor,
                          const void* data,
                          size_t offset,
                          size_t size);
    void get_tensor_async(const struct ggml_tensor* tensor,
                         void* data,
                         size_t offset,
                         size_t size);
    bool cpy_tensor_async(ggml_backend_t backend_dst,
                          const struct ggml_tensor* src,
                          struct ggml_tensor * dst);

   void synchronize();
   ggml_backend_graph_plan_t graph_plan_create(const struct ggml_cgraph* cgraph);
   enum ggml_status graph_plan_compute(ggml_backend_graph_plan_t plan);
   enum ggml_status graph_compute(struct ggml_cgraph* cgraph);
   bool supports_op(const struct ggml_tensor* op);
   bool offload_op(const struct ggml_tensor* op);

   ggml_backend_event_t event_new();
   void event_free(ggml_backend_event_t event);
   void event_record(ggml_backend_event_t event);
   void event_wait(ggml_backend_event_t event);
   void event_synchronize(ggml_backend_event_t event);
```
Not all backends support async operations, for example the CPU backend does not
and the same goes for the support of events.

### `ggml_backend_buffer_type_t`
Lets now take a closer look at the buffer type, `ggml_backend_buffer_type_t`
which is the type returned from `get_default_buffer_type()` above.

It is a typedef in `ggml/include/ggml/ggml-alloc.h`:
```c
    typedef struct ggml_backend_buffer_type * ggml_backend_buffer_type_t;
```
And the definition can be found in `ggml/src/ggml-backend-impl.h`: 
```c
    struct ggml_backend_buffer_type {
        struct ggml_backend_buffer_type_i  iface;
        ggml_backend_buffer_type_context_t context;
    };
    typedef void * ggml_backend_buffer_type_context_t;
```
Notice that a buffer type also has an interface and a context which is also a
void pointer just like the context of a backend.

So first we have an interface which describes the buffer, the buffer type
interface:
```c
    struct ggml_backend_buffer_type_i {
        const char* get_name(ggml_backend_buffer_type_t buft);
        size_t get_alignment(ggml_backend_buffer_type_t buft);
        size_t get_max_size(ggml_backend_buffer_type_t buft);
        size_t get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor);

        bool is_host(ggml_backend_buffer_type_t buft);
        bool supports_backend(ggml_backend_buffer_type_t buft, ggml_backend_t backend); 

        ggml_backend_buffer_t alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);
    };
```
So we first have functions that describe the buffer type like the max size that
can be allocated by this buffer, the memory alignment, if it is a host or
device buffer etc. These are just describing a buffer, `alloc_buffer` returns a
`ggml_backend_buffer_t` (typedef in ggml/include/ggml/ggml-alloc.h) which is the
actual buffer that the type describes:
```c
    typedef struct ggml_backend_buffer* ggml_backend_buffer_t;

    struct ggml_backend_buffer {
        struct ggml_backend_buffer_i  iface;
        ggml_backend_buffer_type_t    buft;
        ggml_backend_buffer_context_t context;
        size_t size;
        enum ggml_backend_buffer_usage usage;
    };
```
The `iface` is the interface of a backend buffer which looks like this:
```c
    struct ggml_backend_buffer_i {
        const char* get_name()
        void free_buffer()
        void get_base();
        void init_tensor(struct ggml_tensor* tensor);
        void set_tensor(struct ggml_tensor* tensor,
                        const void* data,
                        size_t offset,
                        size_t size);
        void get_tensor(const struct ggml_tensor* tensor,
                        void* data,
                        size_t offset,
                        size_t size);
        bool cpy_tensor(const struct ggml_tensor* src,
                        struct ggml_tensor * dst);
        void clear(uint8_t value);
        void reset();
    };
```
`ggml_backend_buffer_type_t` is something we've already seen earlier as it the
context.
`ggml_backend_buffer_usage` is defined as follows:
```
    enum ggml_backend_buffer_usage {
        GGML_BACKEND_BUFFER_USAGE_ANY = 0,
        GGML_BACKEND_BUFFER_USAGE_WEIGHTS = 1,
    };
```
So `ggml_backend_buffer_i` is the bottom most interface and this is what
interacts with the actual memory. For example, `set_tensor` will set the tensor
on a backend, which for a device would mean copying the data from the host to
the device:
```c
static void ggml_backend_cuda_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                ggml_tensor* tensor,
                                                const void* data,
                                                size_t offset,
                                                size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(
        cudaMemcpyAsync((char *)tensor->data + offset,
                        data,
                        size,
                        cudaMemcpyHostToDevice,
                        cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}
```

But of the CPU backend would just be `memcpy` function call:
```c
static void ggml_backend_cpu_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                               struct ggml_tensor* tensor,
                                               const void* data,
                                               size_t offset,
                                               size_t size) {
    memcpy((char *)tensor->data + offset, data, size);
    GGML_UNUSED(buffer);
}
```

`cpu_backend_i` is defined in ggml/src/ggml-backend-cpu.c:
```c++
static struct ggml_backend_i cpu_backend_i = {
    /* .get_name                = */ ggml_backend_cpu_name,
    /* .free                    = */ ggml_backend_cpu_free,
    /* .get_default_buffer_type = */ ggml_backend_cpu_get_default_buffer_type,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ ggml_backend_cpu_graph_plan_create,
    /* .graph_plan_free         = */ ggml_backend_cpu_graph_plan_free,
    /* .graph_plan_compute      = */ ggml_backend_cpu_graph_plan_compute,
    /* .graph_compute           = */ ggml_backend_cpu_graph_compute,
    /* .supports_op             = */ ggml_backend_cpu_supports_op,
    /* .offload_op              = */ NULL,
    /* .event_new               = */ NULL,
    /* .event_free              = */ NULL,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .event_synchronize       = */ NULL,
};
```
Notice that the CPU backend does not support asynchronous operations, and also
does not support events which I'm guessing are used for the asynchronous
operations.

And the `ggml_backend_cpu_buffer_type` function is defined as follows:
```
GGML_CALL ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface = */ {
            /* .get_name         = */ ggml_backend_cpu_buffer_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .supports_backend = */ ggml_backend_cpu_buffer_type_supports_backend,
            /* .is_host          = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}
```

Every tensor has a backend type and and may have buffer:
```console
(gdb) p x
$14 = (struct ggml_tensor *) 0x7ffff6a00030
(gdb) p *x
$15 = {type = GGML_TYPE_F32,
       backend = GGML_BACKEND_TYPE_CPU,
       buffer = 0x0, ne = {10, 1, 1, 1}, nb = {4, 40, 40, 40},
       op = GGML_OP_NONE, op_params = {0 <repeats 16 times>},
       flags = 0, grad = 0x0, src = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
       0x0, 0x0}, perf_runs = 0, perf_cycles = 0, perf_time_us = 0,
       view_src = 0x0, view_offs = 0, data = 0x7ffff6a001a0,
       name = '\000' <repeats 63 times>, extra = 0x0, padding = "\000\000\000\000\000\000\000"}

(gdb) p x.backend
$16 = GGML_BACKEND_TYPE_CPU
(gdb) p x.buffer
$17 = (struct ggml_backend_buffer *) 0x0
```

One thing I noticed is that when we call `ggml_backend_tensor_set` the backend
of the tensor is still CPU in the CUDA case. I would have expected that the
backend would be GPU. It looks like some of the backends set the tensors backend
to GPU but the CPU backend does not. For example the sycl backend does as does
the kompute backend.

The following is a suggestion to update the CUDA backend to also set the
tensor backend to GPU:
```console
$ git diff src/ggml-cuda.cu
diff --git a/src/ggml-cuda.cu b/src/ggml-cuda.cu
index be8e33a..3d93d6b 100644
--- a/src/ggml-cuda.cu
+++ b/src/ggml-cuda.cu
@@ -418,6 +418,7 @@ GGML_CALL static void ggml_backend_cuda_buffer_init_tensor(ggml_backend_buffer_t
             CUDA_CHECK(cudaMemset((char *)tensor->data + original_size, 0, padded_size - original_size));
         }
     }
+    tensor->backend = GGML_BACKEND_TYPE_GPU;
 }
 
 GGML_CALL static void ggml_backend_cuda_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
```

When we call `ggml_backend_alloc_ctx_tensors`, which is a call that allocate
the passed in ggml_contexts tensors to the backend, like this:
```c
      ggml_backend_buffer_t t = ggml_backend_alloc_ctx_tensors(ctx, cuda_backend);
```

```console
$ !gdb
gdb --args ./bin/backend 
Reading symbols from ./bin/backend...
(gdb) br backend.c:62
Breakpoint 1 at 0x404e16: file src/backend.c, line 62.
(gdb) r
```
This function call  will end up in ggml-alloc.c:
```c
ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors(struct ggml_context * ctx, ggml_backend_t backend) {
    return ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(backend));
}
```
And `ggml_backend_alloc_ctx_tensors_from_buft` can also be found in ggml-alloc.c:
```c
ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors_from_buft(struct ggml_context * ctx,
    ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    size_t alignment = ggml_backend_buft_get_alignment(buft);
    size_t max_size = ggml_backend_buft_get_max_size(buft);
    ggml_backend_buffer_t * buffers = NULL;
    size_t n_buffers = 0;

    size_t cur_buf_size = 0;
    struct ggml_tensor * first = ggml_get_first_tensor(ctx);
    for (struct ggml_tensor * t = first; t != NULL; t = ggml_get_next_tensor(ctx, t)) {
```
```console
(gdb) p alignment 
$4 = 128
(gdb) p max_size 
$5 = 18446744073709551615

(gdb) p *first
$8 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU,
      buffer = 0x0, ne = {10, 1, 1, 1}, nb = {4, 40, 40, 40},
      op = GGML_OP_NONE, op_params = {0 <repeats 16 times>},
      flags = 0, grad = 0x0, src = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
      0x0, 0x0, 0x0, 0x0},
      perf_runs = 0, perf_cycles = 0, perf_time_us = 0,
      view_src = 0x0, view_offs = 0,
      data = 0x0,
      name = "x", '\000' <repeats 62 times>, extra = 0x0, padding = "\000\000\000\000\000\000\000"}
```
Inside the above for loop we have:
```console
(gdb) l
926	
927	    size_t cur_buf_size = 0;
928	    struct ggml_tensor * first = ggml_get_first_tensor(ctx);
929	    for (struct ggml_tensor * t = first; t != NULL; t = ggml_get_next_tensor(ctx, t)) {
930	        size_t this_size = 0;
931	        if (t->data == NULL && t->view_src == NULL) {
932	            this_size = GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
933	        }
934	
935	        if (this_size > max_size) {

(gdb) p t->data
$9 = (void *) 0x0
(gdb) p t->view_src
$10 = (struct ggml_tensor *) 0x0

(gdb) p this_size
$14 = 128

(gdb) l
959	    // allocate remaining tensors
960	    if (cur_buf_size > 0) {
961	        if (!alloc_tensor_range(ctx, first, NULL, buft, cur_buf_size, &buffers, &n_buffers)) {
962	            return NULL;
963	        }
964	    }
(gdb) s

(gdb) l
878	
879	static bool alloc_tensor_range(struct ggml_context * ctx,
880	        struct ggml_tensor * first, struct ggml_tensor * last,
881	        ggml_backend_buffer_type_t buft, size_t size,
882	        ggml_backend_buffer_t ** buffers, size_t * n_buffers) {
883	    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, size);
884	    if (buffer == NULL) {
885	#ifndef NDEBUG
886	        fprintf(stderr, "%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), size);
887	#endif
```
`ggml_backed_buft_alloc_buffer` will
```console
21	ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
22	    return buft->iface.alloc_buffer(buft, size);
23	}

(gdb) l
492	GGML_CALL static ggml_backend_buffer_t ggml_backend_cuda_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
493	    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;
494	
495	    ggml_cuda_set_device(buft_ctx->device);
496	
497	    size = std::max(size, (size_t)1); // cudaMalloc returns null for size 0
498	
499	    void * dev_ptr;
500	    cudaError_t err = cudaMalloc(&dev_ptr, size);
501	    if (err != cudaSuccess) {
```
Notice that this is allocating 128 bytes of memory on the GPU which the device
pointer (dev_ptr) will point to if successful.

```console
506	    ggml_backend_cuda_buffer_context * ctx = new ggml_backend_cuda_buffer_context(buft_ctx->device, dev_ptr);
507	
508	    return ggml_backend_buffer_init(buft, ggml_backend_cuda_buffer_interface, ctx, size);
509	}

(gdb) p *ctx
$4 = {device = 0, dev_ptr = 0x7fff94c00200, name = "CUDA0"}
```
The final thing to happen in this function is that ggml_backend_buffer_init is
called.
```c
GGML_CALL ggml_backend_buffer_t ggml_backend_buffer_init(
               ggml_backend_buffer_type_t      buft,
        struct ggml_backend_buffer_i           iface,
               ggml_backend_buffer_context_t   context,
               size_t                          size) {
    ggml_backend_buffer_t buffer = malloc(sizeof(struct ggml_backend_buffer));

    (*buffer) = (struct ggml_backend_buffer) {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_BUFFER_USAGE_ANY
    };

    return buffer;
}
```

```console

895	    struct ggml_tallocr tallocr = ggml_tallocr_new(buffer);
896	
897	    for (struct ggml_tensor * t = first; t != last; t = ggml_get_next_tensor(ctx, t)) {
898	        if (t->data == NULL) {
899	            if (t->view_src == NULL) {
900	                ggml_tallocr_alloc(&tallocr, t);
901	            } else if (t->buffer == NULL) {
902	                ggml_backend_view_init(buffer, t);
903	            }
904	        } else {
```
```c
void ggml_tallocr_alloc(struct ggml_tallocr * talloc, struct ggml_tensor * tensor) {
    size_t size = ggml_backend_buffer_get_alloc_size(talloc->buffer, tensor);
    size = GGML_PAD(size, talloc->alignment);

    if (talloc->offset + size > ggml_backend_buffer_get_size(talloc->buffer)) {
        fprintf(stderr, "%s: not enough space in the buffer to allocate %s (needed %zu, available %zu)\n",
                __func__, tensor->name, size, ggml_backend_buffer_get_size(talloc->buffer) - talloc->offset);
        GGML_ASSERT(!"not enough space in the buffer");
        return;
    }

    void * addr = (char *)ggml_backend_buffer_get_base(talloc->buffer) + talloc->offset;
    talloc->offset += size;

    assert(((uintptr_t)addr % talloc->alignment) == 0);

    ggml_backend_tensor_alloc(talloc->buffer, tensor, addr);
}
``` 
The call to `ggml_backend_tensor_alloc` will set the tensors buffer and
data (which my be null):
```c
void ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr) {
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->data == NULL);
    GGML_ASSERT(tensor->view_src == NULL);
    GGML_ASSERT(addr >= ggml_backend_buffer_get_base(buffer));
    GGML_ASSERT((char *)addr + ggml_backend_buffer_get_alloc_size(buffer, tensor) <=
                (char *)ggml_backend_buffer_get_base(buffer) + ggml_backend_buffer_get_size(buffer));

    tensor->buffer = buffer;
    tensor->data = addr;
    ggml_backend_buffer_init_tensor(buffer, tensor);
}
```
The last function call will then `ggml_backend_buffer_init_tensor` will then
...

Before this call the tensor looks like this:
```console
(gdb) p *tensor
$38 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x41ac880, ne = {10, 1, 1, 1}, nb = {4, 40, 
    40, 40}, op = GGML_OP_NONE, op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0, src = {0x0, 0x0, 0x0, 
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, perf_runs = 0, perf_cycles = 0, perf_time_us = 0, view_src = 0x0, 
  view_offs = 0, data = 0x7fff94c00200, name = "x", '\000' <repeats 62 times>, extra = 0x0, 
  padding = "\000\000\000\000\000\000\000"}
```

```c
GGML_CALL void ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    // init_tensor is optional
    if (buffer->iface.init_tensor) {
        buffer->iface.init_tensor(buffer, tensor);
    }
}
```
```console
(gdb) p buffer->iface.init_tensor
$39 = (void (*)(ggml_backend_buffer_t, 
    struct ggml_tensor *)) 0x4e20c4 <ggml_backend_cuda_buffer_init_tensor(ggml_backend_buffer_t, ggml_tensor*)>
```
So that will land us in `ggml_backend_cuda_buffer_init_tensor`:
```console
(gdb) s
ggml_backend_cuda_buffer_init_tensor (buffer=0x41ac880, tensor=0x7fffcaa00030)
    at /home/danielbevenius/work/ai/learning-ai/fundamentals/ggml/ggml/src/ggml-cuda.cu:402
402	    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
```
So lets take a closer look at this function:
```c
GGML_CALL static void ggml_backend_cuda_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    if (tensor->view_src != NULL) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        return;
    }

    if (ggml_is_quantized(tensor->type)) {
        // initialize padding to 0 to avoid possible NaN values
        size_t original_size = ggml_nbytes(tensor);
        size_t padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size && tensor->view_src == nullptr) {
            ggml_cuda_set_device(ctx->device);
            CUDA_CHECK(cudaMemset((char *)tensor->data + original_size, 0, padded_size - original_size));
        }
    }
    // The following line was added by me as a suggestion that the cuda backend
    // should set this to GPU (change it from CPU).
    tensor->backend = GGML_BACKEND_TYPE_GPU;
}
```
```console
(gdb) p *ctx
$41 = {device = 0, dev_ptr = 0x7fff94c00200, name = "CUDA0"}
(gdb) p tensor->view_src
$42 = (ggml_tensor *) 0x0
(gdb) p ggml_is_quantized(tensor->type)
$43 = false
```

_wip_

#### Forward pass
```c
  struct ggml_cgraph* f_graph = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, true);
  ggml_build_forward_expand(f_graph, mul);
```

```c
struct ggml_cgraph * ggml_new_graph_custom(struct ggml_context * ctx, size_t size, bool grads) {
    ...
    void* p = cgraph + 1;

    struct ggml_tensor** nodes_ptr = incr_ptr_aligned(&p, size* sizeof(struct ggml_tensor*),
        sizeof(struct ggml_tensor*));
```
The `incr_ptr_aligned` is used to increment a pointer while ensuring proper alignment. So p is
the pointer to be be incremented, and size is how much we want to increment the pointer by.
And align is how it should be aligned which in this case is the size of a tensor pointer.
```c
static void * incr_ptr_aligned(void ** p, size_t size, size_t align) {
    void * ptr = *p;
    ptr = (void *) GGML_PAD((uintptr_t) ptr, align);
    *p = (void *) ((char *) ptr + size);
    return ptr;
}
```
So above we first deref the pointer that we want to increment, then pad it using the
`GGML_PAD` macro. The third line will increment (the char cast is just to enable pointer
arithmetic) the pointer by the size we want to increment it by.

And then we do the same for the leafs in the graph:
```c
    struct ggml_tensor** leafs_ptr = incr_ptr_aligned(&p, size* sizeof(struct ggml_tensor*),
        sizeof(struct ggml_tensor*));
```
And similarly for the hash keys and grads:
```c

    struct ggml_tensor** hash_keys_ptr = incr_ptr_aligned(&p,
        hash_size* sizeof(struct ggml_tensor*), sizeof(struct ggml_tensor*));

    struct ggml_tensor** grads_ptr = grads ?
        incr_ptr_aligned(&p, size* sizeof(struct ggml_tensor*), sizeof(struct ggml_tensor*)) :
        NULL;

    ggml_bitset_t* hash_used = incr_ptr_aligned(&p, ggml_bitset_size(hash_size) * sizeof(ggml_bitset_t), sizeof(ggml_bitset_t));
```
These pointer will then be used to create a `ggml_cgraph` struct:
```c
    *cgraph = (struct ggml_cgraph) {
        /*.size         =*/ size,
        /*.n_nodes      =*/ 0,
        /*.n_leafs      =*/ 0,
        /*.nodes        =*/ nodes_ptr,
        /*.grads        =*/ grads_ptr,
        /*.leafs        =*/ leafs_ptr,
        /*.hash_table   =*/ { hash_size, hash_used, hash_keys_ptr },
        /*.order        =*/ GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT,
};
```
Just to clarify that all tensors in the graph are nodes. In ggml any node that is an operation
of and any node that participates in gradient computation.
Notice that we create a `.hash_table` which is a struct with a size, a pointer to
a bitset and a pointer to the hash keys.
```c
    typedef uint32_t ggml_bitset_t;

    struct ggml_hash_set {
        size_t size;
        ggml_bitset_t * used;
        struct ggml_tensor ** keys;
    };
```
```console
(lldb) p cgraph->visited_hash_set
(ggml_hash_set) {
  size = 4099
  used = 0x000000015c014980
  keys = 0x000000015c008968
}
```
Next we have:
```c
    ggml_hash_set_reset(&cgraph->visited_hash_set);
```
```c
void ggml_hash_set_reset(struct ggml_hash_set * hash_set) {
    memset(hash_set->used, 0, sizeof(ggml_bitset_t) * ggml_bitset_size(hash_set->size));
}
```
So this is writing 516 bytes of 0 to the used `visited_hash_set` which is just to initialize
it to 0. After that the compute graph is ready to be used and returned.

Lets inspect the compute graph:
```console
(lldb) p *cgraph
(ggml_cgraph) {
  size = 2048
  n_nodes = 0
  n_leafs = 0
  nodes = 0x0000000136000968
  grads = 0x0000000136010980
  leafs = 0x0000000136004968
  visited_hash_set = {
    size = 4099
    used = 0x0000000136014980
    keys = 0x0000000136008968
  }
  order = GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT
}
```
Next lets look at the `ggml_build_forward_expand` function:
```c
void ggml_build_forward_expand(struct ggml_cgraph * cgraph, struct ggml_tensor * tensor) {
    ggml_build_forward_impl(cgraph, tensor, true);
}
```
Notice that this is passing `true` as `expand`:
```c
static void ggml_build_forward_impl(struct ggml_cgraph * cgraph,
    struct ggml_tensor * tensor, bool expand) {
    if (!expand) {
        // TODO: this branch isn't accessible anymore, maybe move this to ggml_build_forward_expand
        ggml_graph_clear(cgraph);
    }

    const int n0 = cgraph->n_nodes;

    ggml_visit_parents(cgraph, tensor);

    const int n_new = cgraph->n_nodes - n0;
    GGML_PRINT_DEBUG("%s: visited %d new nodes\n", __func__, n_new);

    if (n_new > 0) {
        // the last added node should always be starting point
        GGML_ASSERT(cgraph->nodes[cgraph->n_nodes - 1] == tensor);
    }
}
```
In our case `n0` is 0. Lets take a look at the `ggml_visit_parents` function:
```c
static void ggml_visit_parents(struct ggml_cgraph * cgraph, struct ggml_tensor * node) {
    if (node->grad == NULL) {
        if (node->op != GGML_OP_NONE) {
            //GGML_PRINT_DEBUG("%s: warning: node %p has no grad, but op %d\n", __func__, (void *) node, node->op);
        }
    }

    // check if already visited
    if (ggml_hash_insert(&cgraph->visited_hash_set, node) == GGML_HASHSET_ALREADY_EXISTS) {
        return;
    }
```
Next lets look at `ggml_hash_insert`. The `node` is the tensor that we passed in to
`ggml_build_forward_expand` which is:
```console
(lldb) p tensor->op
(ggml_op) GGML_OP_MUL
```
```c
static size_t ggml_hash_insert(struct ggml_hash_set * hash_set, struct ggml_tensor * key) {
    size_t h = ggml_hash(key) % hash_set->size;

    // linear probing
    size_t i = h;
    do {
        if (!ggml_bitset_get(hash_set->used, i)) {
            ggml_bitset_set(hash_set->used, i);
            hash_set->keys[i] = key;
            return i;
        }
        if (hash_set->keys[i] == key) {
            return GGML_HASHSET_ALREADY_EXISTS;
        }
        i = (i + 1) % hash_set->size;
    } while (i != h);

    // visited all hash table entries -> not found
    GGML_ABORT("fatal error");
}
```
Now, `ggml_hash` is uses the pointer of the tensor (minus the padding) as the hash:
```c
// hash function for ggml_tensor
static inline size_t ggml_hash(const struct ggml_tensor * p) {
    // the last 4 bits are always zero due to alignment
    return (size_t)(uintptr_t)p >> 4;
}
```
So we have a `hash_set->size` of 4099 and we need the map the hash value into this
range of values which is what the modulo operation is used for.

```c
static size_t ggml_hash_insert(struct ggml_hash_set * hash_set, struct ggml_tensor * key) {
    size_t h = ggml_hash(key) % hash_set->size;
```
Next, we have the following where the hash value is used (and the key is the tensor):
```c
    // linear probing
    size_t i = h;
    do {
        if (!ggml_bitset_get(hash_set->used, i)) {
            ggml_bitset_set(hash_set->used, i);
            hash_set->keys[i] = key;
            return i;
        }
        if (hash_set->keys[i] == key) {
            return GGML_HASHSET_ALREADY_EXISTS;
        }
        i = (i + 1) % hash_set->size;
    } while (i != h);
```

```c
#define BITSET_SHR 5 // log2(sizeof(ggml_bitset_t)*8)
#define BITSET_MASK (sizeof(ggml_bitset_t)*8 - 1) // 31

static inline bool ggml_bitset_get(const ggml_bitset_t * bitset, size_t i) {
    return !!(bitset[i >> BITSET_SHR] & (1u << (i & BITSET_MASK)));
}
```
So lets take a look at a concreate example and lets say that `i` is 37:
```
i = 37
BITSET_SHR = 5
BITSET_MASK = 31 (binary: 00011111)

i >> BITSET_SHR:
37 in binary:        00100101
37 >> 5:             00000001

i & BITSET_MASK:
37 in binary:        00100101
BITSET_MASK:         00011111
37 & BITSET_MASK:    00000101

1u << (i & BITSET_MASK)
1 in binary:         00000001
1 << 5:              00100000  (32 in decimal)

Now, the bitset is passed but lets say it is 10100000:
bitset[1]:           10100000
Bit mask:            00100000
bitset[1] & mask:    00100000

!!(00100000)

First !: 00100000 becomes 0
Second !: 0 becomes 1
Final result: 1 (true, the bit is set)
```
In our case (debugging session, not the above example) the bit is not set and we will
enter the if block:
```c
        if (!ggml_bitset_get(hash_set->used, i)) {
            ggml_bitset_set(hash_set->used, i);
            hash_set->keys[i] = key;
            return i;
        }
```
Next the bitset will be set for this hash (i) and then the key will be set to the
tensor that we passed in and the hash returned (recall that we are in `ggml_hash_insert`):
```c
    // check if already visited
    if (ggml_hash_insert(&cgraph->visited_hash_set, node) == GGML_HASHSET_ALREADY_EXISTS) {
        return;
    }
```
Following that we have:
```c
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        const int k =
            (cgraph->order == GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT) ? i :
            (cgraph->order == GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT) ? (GGML_MAX_SRC-1-i) :
            /* unknown order, just fall back to using i*/ i;
        if (node->src[k]) {
            ggml_visit_parents(cgraph, node->src[k]);
        }
    }

#define GGML_MAX_SRC            10
```
This will visit the parents of the children of the tensor that was passed in. In our case
`k` will be 0 so we will visit the first child:
```console
(lldb) p node->src[0]->name
(char[64]) "a"
```
So we will go through the same process we did above but now for the a tensor. And we will
also do the same for the b tensor.

Next we have:
```c
    if (node->op == GGML_OP_NONE && node->grad == NULL) {
        // reached a leaf node, not part of the gradient graph (e.g. a constant)
        GGML_ASSERT(cgraph->n_leafs < cgraph->size);

        if (strlen(node->name) == 0) {
            ggml_format_name(node, "leaf_%d", cgraph->n_leafs);
        }

        cgraph->leafs[cgraph->n_leafs] = node;
        cgraph->n_leafs++;
```
This is checking if the tensor is an operation, and that it does not have a gradient which
indicates that this is a leaf node (a constant). Recall that node is the multiplcation operator
node so the above block will not be executed.

```c
    } else {
        GGML_ASSERT(cgraph->n_nodes < cgraph->size);

        if (strlen(node->name) == 0) {
            ggml_format_name(node, "node_%d", cgraph->n_nodes);
        }

        cgraph->nodes[cgraph->n_nodes] = node;
        if (cgraph->grads) {
            cgraph->grads[cgraph->n_nodes] = node->grad;
        }
        cgraph->n_nodes++;
    }
```
We can see that if the node/tensor does not have a name it will be given one now.
```console
(lldb) p node->name
(char[64]) "node_2"
```
Then we can see that the node is added to the cgraph nodes array, and the nodes counter is
incremented. Note that we already recursed through the children of the node so we have already
visited the children and they would also have been added and incremented `n_nodes` just in case
it is confusing that this is `node_2`.
And that is bascially the how the forward pass is built up.

We can print the graph (I've added a few columns for readability):
```console
=== GRAPH ===
n_nodes = 3
          ne[0]   ne[1]  ne[2]             OP   Param (x) or Gradient (g)
 -   0: [     1,     1,     1]             NONE x
 -   1: [     1,     1,     1]             NONE x
 -   2: [     1,     1,     1]              MUL g
n_leafs = 1
          ne[0]   ne[1]     OP                  Name
 -   0: [     1,     1]     NONE                5
========================================
```

After we have created a cgraph and called `ggml_build_forward_expand` we can
then call `ggml_graph_compute`:
```c
  int n_threads = 1;
  ggml_graph_compute_with_ctx(ctx, c_graph, n_threads);
```
I'm just using 1 thread to simplfy stepping through the code in the debugger.

```c
enum ggml_status ggml_graph_compute_with_ctx(struct ggml_context * ctx,
    struct ggml_cgraph * cgraph, int n_threads) {
    struct ggml_cplan cplan = ggml_graph_plan(cgraph, n_threads);

    struct ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_WORK_BUFFER, cplan.work_size);

    cplan.work_data = (uint8_t *)ctx->mem_buffer + obj->offs;

    return ggml_graph_compute(cgraph, &cplan);
}
```
So before we can compute we need to create a plan. So what is a plan? 
```c   
    struct ggml_cplan {
        size_t    work_size;
        uint8_t * work_data;

        int n_threads;
        struct ggml_threadpool * threadpool;

        // abort ggml_graph_compute when true
        ggml_abort_callback abort_callback;
        void *              abort_callback_data;
    };
```
TODO: look into the threadpool.
I'm going to skip the threadpool for now and revisit as my focus is on the
computation in this section.

```c
struct ggml_cplan ggml_graph_plan(
          const struct ggml_cgraph * cgraph,
                               int   n_threads,
            struct ggml_threadpool * threadpool) {

    if (threadpool == NULL) {
        GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
    }
    if (n_threads <= 0) {
        n_threads = threadpool ? threadpool->n_threads_max : GGML_DEFAULT_N_THREADS;
    }

    size_t work_size = 0;

    struct ggml_cplan cplan;
    memset(&cplan, 0, sizeof(struct ggml_cplan));
    ...
```

### Backward expand
In the example we have:
```c
  struct ggml_cgraph* b_graph = ggml_graph_dup(ctx, f_graph);
  ggml_build_backward_expand(ctx, f_graph, b_graph, /* keep gradients */ false);
```

```c
void ggml_build_backward_expand(struct ggml_context * ctx, struct ggml_cgraph * gf,
    struct ggml_cgraph * gb, bool keep) {

    ...

    // remember original gradients which start with zero values
    struct ggml_hash_set zero_table = ggml_hash_set_new(gf->size);
    for (int i = 0; i < gf->n_nodes; i++) {
        if (gf->grads[i]) {
            ggml_hash_insert(&zero_table, gf->grads[i]);
        }
    }
    ...
}
```
Notice that this is using the forward compute graph (gf).

We can inspect the gradients:
```console
(lldb) p gf->grads[0]->name
(char[64]) "a (grad)"
(lldb) p gf->grads[1]->name
(char[64]) "b (grad)"
(lldb) p gf->grads[2]
(ggml_tensor *) 0x00000001358007a0
(lldb) p gf->nodes[2]->grad
(ggml_tensor *) 0x00000001358007a0
```
Lets start with this hash set named `zero_table`. Keeping track of the gradients that are zero
might enable optimizations, like skipping some calculations or using speciallized algorithms
for sparse gradients. But at this point in the code the `zero_table` is simply being populated
with all the gradient tensors (for a, b, and mul in our case). Recall that the keys
in the hash set are the hashes like we mentioned earlier. But when debugging we can see
the hash values and check, for example for the `b (grad)` tensor:
```console
(lldb) p *zero_table->keys[1392]->name
(char) 'b'
```
And notice that if the grandient tensor (key) is not in the set it will be added and
set to they key, and the hash returned.

After that we have where we will iterate over all the nodes in the graph, but
in reverse order:
```c
    for (int i = gf->n_nodes - 1; i >= 0; i--) {
        struct ggml_tensor * node = gf->nodes[i];

        // inplace operations to add gradients are not created by ggml_compute_backward
        // use allocator to automatically make inplace operations
        if (node->grad) {
            ggml_compute_backward(ctx, node, &zero_table);
        }
    }
```
This will start with `mul`. And notice that we are passing in the `zero_table` to
`ggml_compute_backward`. Now, recall that we are currently in
`ggml_build_backward_expand` and I was not expecting to see this
`ggml_compute_backward`  function call here! But this is constructing the graph
that will later be used during backpropagation.

Lets take a look at this function:
```c
static void ggml_compute_backward(struct ggml_context * ctx, struct ggml_tensor * tensor,
    struct ggml_hash_set * zero_table) {
    struct ggml_tensor * src0 = tensor->src[0];
    struct ggml_tensor * src1 = tensor->src[1];
    struct ggml_tensor * src2 = tensor->src[2];

    switch (tensor->op) {
```
We can check that `src0` is `a` and `src1` is `b`:
```console
(lldb) p src0->name
(char[64]) "a"
(lldb) p src1->name
(char[64]) "b"
```
And after that we have a switch statement and different cases for the different operations.
```c
        case GGML_OP_MUL:
            {
                if (src0->grad) {
                    src0->grad =
                        ggml_add_or_set(ctx,
                                src0->grad,
                                ggml_mul(ctx, src1, tensor->grad),
                                zero_table);
                }
                if (src1->grad) {
                    src1->grad =
                        ggml_add_or_set(ctx,
                                src1->grad,
                                ggml_mul(ctx, src0, tensor->grad),
                                zero_table);
                }
            } break;
```
We can see that this is setting the `src0->grad` tensor to some tensor.
Now, recall we have something that looks like this `mul = a * b`:
Think of this as:
```
mul    =    a * b
tensor = src0 * src1
```
And we have the derivative function definition:
```
 f(a+h) - f(a)
 -------------
      h

f(a) = a * b
```
And we want to find out how our function `f(a * b)` will change if we make
a small change to 'a' (the partial derivative of a that is):
```
 df/da = ((a + h) * b) - (a * b) / h
 df/da = ab + hb - ab / h
 df/da = ab - ab + hb / h
 df/da = hb / h
 df/da = b
```
So that is the partial derivitive of `a`. Now we want to find out how our new
loss value will effect a. This can be done by taking derivative of the loss
with respect to the output of the multiplication operation node multiplied
by the derivative of a:
```
∂L/∂a = (∂L/∂z) * (∂z/∂a) = (∂L/∂z) * b
```
This is what the following line is doing:
```c
    ggml_mul(ctx, src1, tensor->grad),
```
So the result of this operation is what we want to add to the gradient of a. We
need add this value to the gradient of a becuase `a` might be used in other
places in the graph and we don't want to loose any information which is what
would happen if we just replaced the gradient. Also keep in mind that this is
only defining an operation and returning the tensor representing that operation.

The loss value indicates how well our system is performing relative to what we
want it to do.

After that `ggml_add_or_set` is called with this new tensor:
```c
static struct ggml_tensor * ggml_add_or_set(struct ggml_context * ctx, struct ggml_tensor * a,
    struct ggml_tensor * b, struct ggml_hash_set * zero_table) {
    if (ggml_hash_contains(zero_table, a)) {
        return b;
    } else {
        return ggml_add_impl(ctx, a, b, false);
    }
}
```
Keep in mind that a is the grandient (`a_grad`) in this case and that tensor of the
multiplication operation of src1 and tensor-grad. In this case remember that
the b tensor is the gradient of the multiplication operation node and not 'b'
in our example (was not the best choice of names perhaps on my part for this).
So if the a->grad is in the zero hash set then we will return b, which makes sense,
as we don't have to perform the add operation in that case.

The result of this is then set to update the gradient of `a`:
```c
        src0->grad =
```
The same is done for src1.

The last thing to be done in `ggml_compute_backward` is just the following checks:
```c
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (tensor->src[i] && tensor->src[i]->grad) {
            GGML_ASSERT(ggml_are_same_shape(tensor->src[i], tensor->src[i]->grad));
        }
}
```
This will then return to `ggml_build_backward_expand` and we will continue with the next node
which is `b`.

```c
    for (int i = gf->n_nodes - 1; i >= 0; i--) {
        struct ggml_tensor * node = gf->nodes[i];

        // inplace operations to add gradients are not created by ggml_compute_backward
        // use allocator to automatically make inplace operations
        if (node->grad) {
            ggml_compute_backward(ctx, node, &zero_table);
        }
    }
}
```
Then the same will be done for the `b` tensor but this tensor does not have
an operation so nothing will be done for it, and the same goes for the `a`
tensor.

Following that we have:
```c
    for (int i = 0; i < gf->n_nodes; i++) {
        struct ggml_tensor * node = gf->nodes[i];

        if (node->flags & GGML_TENSOR_FLAG_PARAM) {
            GGML_PRINT_DEBUG("%s: found root node %p\n", __func__, (void *) node);
            ggml_build_forward_expand(gb, node->grad);
        }
    }
```
This is iterating over all the nodes in the forward graph so it will start
with tensor a. But notice that it is using the backward computation graph (gb)
and it is passing it's gradient tensor to `ggml_build_forward_expand`. 
Now, just so we understand what is going on here. Recall that in the we
added a gradient to a in `ggml_compute_backward`:
```c
        case GGML_OP_MUL:
            {
                if (src0->grad) {
                    src0->grad =
                        ggml_add_or_set(ctx,
                                src0->grad,
                                ggml_mul(ctx, src1, tensor->grad),
                                zero_table);
                }
                if (src1->grad) {
                    src1->grad =
                        ggml_add_or_set(ctx,
                                src1->grad,
                                ggml_mul(ctx, src0, tensor->grad),
                                zero_table);
                }
            } break;
```
And if we recall this was set to the multiplation operation of b and the loss.
Now, the `b` tensor has already been added to the compuation graph but the
loss tensor has not yet (and note that this tensor will be added as a leaf node.

And the last thing to happen in `ggml_build_backward_expand` is:
```c
    ggml_hash_set_free(&zero_table);
```
Now with the backward graph constructed we can actually compute the gradients.
First we compute a loss function and set the output nodes gradient to this
value:
```
  ggml_set_f32(mul->grad, 2.0f);
  ggml_graph_compute_with_ctx(ctx, b_graph, 1);
```
Notice that we are using the backward graph (b_graph) and not the forward graph
this time, and threads are still 1.
```c
enum ggml_status ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads) {
    struct ggml_cplan cplan = ggml_graph_plan(cgraph, n_threads, NULL);

    struct ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_WORK_BUFFER, cplan.work_size);

    cplan.work_data = (uint8_t *)ctx->mem_buffer + obj->offs;

    return ggml_graph_compute(cgraph, &cplan);
}
```
So we create a plan and then set a work buffer and then we call
`ggml_graph_compute`.
```c
enum ggml_status ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan) {
    ...
    } else {
        ggml_graph_compute_thread(&threadpool->workers[0]);
    }
```
I'm skipping the threadpool/threading for now and will revisit that later as
mentioned before.
But lets take a look at the `ggml_graph_compute_thread` function:
```c
static thread_ret_t ggml_graph_compute_thread(void * data) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;

    const struct ggml_cgraph * cgraph = state->threadpool->cgraph;
    const struct ggml_cplan  * cplan  = state->threadpool->cplan;

    set_numa_thread_affinity(state->ith);

    struct ggml_compute_params params = {
        /*.ith       =*/ state->ith,
        /*.nth       =*/ state->threadpool->n_threads_cur,
        /*.wsize     =*/ cplan->work_size,
        /*.wdata     =*/ cplan->work_data,
        /*.threadpool=*/ state->threadpool,
    };

    for (int node_n = 0; node_n < cgraph->n_nodes; node_n++) {
        struct ggml_tensor * node = cgraph->nodes[node_n];

        ggml_compute_forward(&params, node);

        if (state->ith == 0 && cplan->abort_callback && cplan->abort_callback(cplan->abort_callback_data)) {
            state->threadpool->ec = GGML_STATUS_ABORTED;
        }

        ggml_barrier(state->threadpool);

        if (state->threadpool->ec != GGML_STATUS_SUCCESS) {
            break;
        }
    }

    return 0;
}
```
TODO: add some notes about NUMA. But in short this is about pinning/binding a
thread to a specific CPU core.
We can inspect the ggml_compute_state:
```console
$4 = {threadpool = 0x5555556b0740, ith = 0}
```
So in this case we have the threadpool pointer and the thread index
(index thread=ith).

Just to get an overview of the threadpool here is what this struct looks like:
```console
(gdb) ptype *state->threadpool
type = struct ggml_threadpool {
    ggml_mutex_t mutex;
    ggml_cond_t cond;
    struct ggml_cgraph *cgraph;
    struct ggml_cplan *cplan;
    atomic_int n_graph;
    atomic_int n_barrier;
    atomic_int n_barrier_passed;
    atomic_int current_chunk;
    atomic_bool stop;
    atomic_bool pause;
    struct ggml_compute_state *workers;
    int n_threads_max;
    int n_threads_cur;
    int32_t prio;
    uint32_t poll;
    enum ggml_status ec;
}
```
After creating the compute params the main look is iterating over all the nodes
in the graph:
```c
    for (int node_n = 0; node_n < cgraph->n_nodes; node_n++) {
        struct ggml_tensor * node = cgraph->nodes[node_n];

        ggml_compute_forward(&params, node);

        if (state->ith == 0 && cplan->abort_callback && cplan->abort_callback(cplan->abort_callback_data)) {
            state->threadpool->ec = GGML_STATUS_ABORTED;
        }

        ggml_barrier(state->threadpool);

        if (state->threadpool->ec != GGML_STATUS_SUCCESS) {
            break;
        }
    }
```
TODO: I've actually written about this part before and should consolidate this
into a section that is easier to find.
The first node in this case will be tensor `a`:
`ggml_compute_forward`.
```console
(gdb) p node->name
$11 = "a"
(gdb) p node->op
$10 = GGML_OP_NONE
```
And this will be passed to `ggml_compute_forward`:
```c
static void ggml_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor) {
    GGML_ASSERT(params);

    if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {
        return;
    }
```
Notice that in this case `a` does not have an operation so this will just
return at this point.
The same will happen for `b`.

Just a note about the `ggml_is_empty` function:
```c
GGML_CALL bool ggml_is_empty(const struct ggml_tensor * tensor) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] == 0) {
            // empty if any dimension has no elements
            return true;
        }
    }
    return false;
}
```
If a tensor has any dimension that is 0 then it is considered empty. This makes
sense as if we have a 2d tensor of size 4x0 (4 x-axis element and 0 y-axis) ther
are no elements in the tensor.

After that we have the multiplication operation:
```console
(gdb) p node->name
$17 = "mul", '\000' <repeats 60 times>
(gdb) p node->op
$18 = GGML_OP_MUL
```
This time we will reach the switch statement as the mul tensor has an operation:
```c
static void ggml_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor) {
    GGML_ASSERT(params);

    if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {
        return;
    }

    switch (tensor->op) {
        ...
        case GGML_OP_MUL:
            {
                ggml_compute_forward_mul(params, tensor);
            } break;
        ...
    }
```
So dst will be the mul tensor, and src0 and src1 the a and b tensors in the
following function:
```c
static void ggml_compute_forward_mul(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32 && "only f32 src1 supported for now");

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_mul_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}
```
```c
static void ggml_compute_forward_mul_f32(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    const int ith = params->ith;  // 0
    const int nth = params->nth;  // 1 (only using one thread for this example)

    const int64_t nr = ggml_nrows(src0);

    GGML_TENSOR_BINARY_OP_LOCALS

    if (nb10 == sizeof(float)) {
```
The above is checking if nb10 (number of bytes) which is the first dimension of
the second source which is `b` in our case:
```console
(gdb) p src1->nb[0]
$32 = 4
(gdb) p nb10
$33 = 4
(gdb) p nb10 == sizeof(float)
$36 = 1
```
So we will enter this if block:
Now, we can see that `ir` is set to the thread index (0 in this specific case
as we are only using 1 thread).
```c
        for (int64_t ir = ith; ir < nr; ir += nth) {
            // src0 and dst are same shape => same indices
            const int64_t i03 = ir/(ne02*ne01);
            const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
            const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);
```
Now, `ne02` (src0)->ne[2]) is the number of elements of the second dimension of
the first source which is `a`. So `ir` is the current row being processed.
`i03` is the index of the 4th dimension, `i02` is the index of the 3rd dimension
and `i01` is the index of the 2nd dimension.


_wip_

### GGML_TENSOR_BINARY_OP_LOCALS
`GGML_TENSOR_BINARY_OP_LOCALS` expands local variables for the src tensors and
this is used in many if not all the operation functions so it can be good to
understand/know about them as it will make reading the code easier (otherwise
it just looks like variables are magically being used)
```c
    const int64_t ne00 = (src0)->ne[0];
    const int64_t ne01 = (src0)->ne[1];
    const int64_t ne02 = (src0)->ne[2];
    const int64_t ne03 = (src0)->ne[3];

    const size_t nb00 = (src0)->nb[0];
    const size_t nb01 = (src0)->nb[1];
    const size_t nb02 = (src0)->nb[2];
    const size_t nb03 = (src0)->nb[3];

    const int64_t ne10 = (src1)->ne[0];
    const int64_t ne11 = (src1)->ne[1];
    const int64_t ne12 = (src1)->ne[2];
    const int64_t ne13 = (src1)->ne[3];

    const size_t nb10 = (src1)->nb[0];
    const size_t nb11 = (src1)->nb[1];
    const size_t nb12 = (src1)->nb[2];
    const size_t nb13 = (src1)->nb[3];

    const int64_t ne0 = (dst)->ne[0];
    const int64_t ne1 = (dst)->ne[1];
    const int64_t ne2 = (dst)->ne[2];
    const int64_t ne3 = (dst)->ne[3];

    const size_t nb0 = (dst)->nb[0];
    const size_t nb1 = (dst)->nb[1];
    const size_t nb2 = (dst)->nb[2];
    const size_t nb3 = (dst)->nb[3];
```
There are also casts are just to avoid warnings about unused variables which
I've removed for clarity here.
Notice the pattern here that is used. For the source tensor we have `ne` for
number of elements and `nb` for number of bytes (stride), followed the source
index (src0, src1 etc), and then the dimension index.
For the destination tensor (mul in this case), these are named only using
ne and nb followed by the dimension index (there is only one dst tensor).

### `ggml_get_rows`
```c
GGML_CALL int64_t ggml_nrows(const struct ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}
```
Notice that for a 2d case we have d0 which are the number of elements, and d1
is the number of rows. For more dimensions we have d1*d2 rows and so on. This
is basically what the above is caclulating.



And notice that the operation is of type `GGML_OP_MUL_MAT`.
```
        switch (node->op) {
        ...
            case GGML_OP_MUL_MAT:
                {
                    const enum ggml_type vec_dot_type = type_traits[node->src[0]->type].vec_dot_type;

                    if (node->src[1]->type != vec_dot_type) {
                        cur = ggml_row_size(vec_dot_type, ggml_nelements(node->src[1]));
                    }
                } break;
```
`type_traits` is an array of struct that has the following members:
```console
(gdb) ptype ggml_type_traits_t
type = struct {
    const char *type_name;
    int blck_size;
    size_t type_size;
    _Bool is_quantized;
    ggml_to_float_t to_float;
    ggml_from_float_t from_float;
    ggml_from_float_t from_float_reference;
    ggml_vec_dot_t vec_dot;
    enum ggml_type vec_dot_type;
    int64_t nrows;
}
(gdb) p node->src[0]->type
$49 = GGML_TYPE_F32

(gdb) p type_traits[node->src[0]->type]
$50 = {type_name = 0x5555555ea428 "f32", blck_size = 1, type_size = 4, is_quantized = false, to_float = 0x0, from_float = 0x0, 
  from_float_reference = 0x0, vec_dot = 0x55555555a0ed <ggml_vec_dot_f32>, vec_dot_type = GGML_TYPE_F32, nrows = 1}
(gdb) p type_traits[node->src[0]->type].vec_dot_type
$51 = GGML_TYPE_F32
```
Notice that the above is checking the types of the parent/src nodes and if they
are the same type, which is the case here, then it will not set `cur`.
```c
    cplan.n_threads = MIN(max_tasks, n_threads);
    cplan.work_size = work_size;   // 0 
    cplan.work_data = NULL;       

    return cplan;
```
Back in `ggml_graph_compute_with_ctx` we  have created the cplan and will now
create a new object of type `GGML_OBJECT_TYPE_WORK_BUFFER`.
```c
enum ggml_status ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads) {
    struct ggml_cplan cplan = ggml_graph_plan(cgraph, n_threads);

    struct ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_WORK_BUFFER, cplan.work_size);

    cplan.work_data = (uint8_t *)ctx->mem_buffer + obj->offs;

    return ggml_graph_compute(cgraph, &cplan);
}
```
I've gone through `ggml_new_object` before so I won't go through it again.
TODO: look into the context `mem_buffer` and how that works.

So lets now take a look at `ggml_graph_compute`:
```c
enum ggml_status ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan) {
    GGML_ASSERT(cplan);
    GGML_ASSERT(cplan->n_threads > 0);
    GGML_ASSERT(cplan->work_size == 0 || cplan->work_data != NULL);

    int n_threads = cplan->n_threads;

    struct ggml_compute_state_shared state_shared = {
        /*.cgraph                  =*/ cgraph,
        /*.cgraph_plan             =*/ cplan,
        /*.n_threads               =*/ n_threads,
        /*.n_barrier               =*/ 0,
        /*.n_barrier_passed        =*/ 0,
        /*.abort_callback          =*/ NULL,
        /*.abort_callback_data     =*/ NULL,
        /*.current_chunk           =*/ 0,
        /*.ec                      =*/ GGML_STATUS_SUCCESS,
    };

#ifdef GGML_USE_OPENMP
    if (n_threads > 1) {
        #pragma omp parallel num_threads(n_threads)
        {
            #pragma omp single
            {
                // update the number of threads from the actual number of threads that we got from OpenMP
                n_threads = omp_get_num_threads();
                state_shared.n_threads = n_threads;
            }

            struct ggml_compute_state worker = {
                .thrd   = 0,
                .ith    = omp_get_thread_num(),
                .shared = &state_shared,
            };
            ggml_graph_compute_thread(&worker);
        }
    } else {
        struct ggml_compute_state worker = {
            .thrd   = 0,
            .ith    = 0,
            .shared = &state_shared,
        };
        ggml_graph_compute_thread(&worker);
    }
```

```
        #pragma omp parallel num_threads(n_threads)
```
This is an OpenMP directive that specifies that the following block of code
should be executed in parallel by `n_threads`. So this will start 4 threads in
our case.

The next OMP directive is `#pragma omp single` which specifies that the block
should be executed by a single thread. So one of those four threads will execute
the block of code that follows, which in this case just gets the number of
threads from OpenMP and sets the `n_threads` to that value.

If we set a breakpoint in the single block and the parallel block we can
inspect the threads that have been created:
```console
18704	        #pragma omp parallel num_threads(n_threads)
(gdb) n
[New Thread 0x7ffff6bfe640 (LWP 94880)]
[New Thread 0x7ffff63fd640 (LWP 94881)]
[New Thread 0x7ffff5bfc640 (LWP 94882)]
[Switching to Thread 0x7ffff63fd640 (LWP 94881)]

Thread 3 "matrix-mul" hit Breakpoint 2, ggml_graph_compute._omp_fn.0 () at /home/danbev/work/ai/learning-ai/fundamentals/ggml/ggml/src/ggml.c:18709
18709	                n_threads = omp_get_num_threads();
(gdb) info threads
  Id   Target Id                                      Frame 
  1    Thread 0x7ffff7e65c00 (LWP 80908) "matrix-mul" 0x00007ffff7e8b0ca in ?? () from /lib/x86_64-linux-gnu/libgomp.so.1
  2    Thread 0x7ffff6bfe640 (LWP 94880) "matrix-mul" 0x00007ffff7e8b0ca in ?? () from /lib/x86_64-linux-gnu/libgomp.so.1
* 3    Thread 0x7ffff63fd640 (LWP 94881) "matrix-mul" ggml_graph_compute._omp_fn.0 ()
    at /home/danbev/work/ai/learning-ai/fundamentals/ggml/ggml/src/ggml.c:18709
  4    Thread 0x7ffff5bfc640 (LWP 94882) "matrix-mul" 0x00007ffff7e8b0ca in ?? () from /lib/x86_64-linux-gnu/libgomp.so.1

(gdb) p n_threads
$66 = 4
```
And we can switch between the threads using:
```console
(gdb) thread 1
[Switching to thread 1 (Thread 0x7ffff7e65c00 (LWP 80908))]
#0  0x00007ffff7e8b0ca in ?? () from /lib/x86_64-linux-gnu/libgomp.so.1
```
But lets follow thread 1 in this case:
```c
            struct ggml_compute_state worker = {
                .thrd   = 0,
                .ith    = omp_get_thread_num(),
                .shared = &state_shared,
            };
            ggml_graph_compute_thread(&worker);
```
```console
(gdb) thread 1
(gdb) set scheduler-locking on
```

So we will call `ggml_graph_compute_thread` with the worker struct:
```c
static thread_ret_t ggml_graph_compute_thread(void * data) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;

    const struct ggml_cgraph * cgraph = state->shared->cgraph;
    const struct ggml_cplan  * cplan  = state->shared->cplan;

    set_numa_thread_affinity(state->ith);

    struct ggml_compute_params params = {
        /*.ith   =*/ state->ith,
        /*.nth   =*/ state->shared->n_threads,
        /*.wsize =*/ cplan->work_size,
        /*.wdata =*/ cplan->work_data,
        /*.shared=*/ state->shared,
    };

    for (int node_n = 0; node_n < cgraph->n_nodes; node_n++) {
        struct ggml_tensor * node = cgraph->nodes[node_n];

        ggml_compute_forward(&params, node);

        if (state->ith == 0 && cplan->abort_callback && cplan->abort_callback(cplan->abort_callback_data)) {
            state->shared->ec = GGML_STATUS_ABORTED;
        }

        ggml_barrier(state->shared);

        if (state->shared->ec != GGML_STATUS_SUCCESS) {
            break;
        }
    }

    return 0;
}
```
So this is looping over all the nodes in the compute graph, which is only one
in this case.
```console
(gdb) p *node
$71 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0, ne = {3, 1, 1, 1}, nb = {4, 12, 12, 12}, 
  op = GGML_OP_MUL_MAT, op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0, src = {0x7ffff6bff030, 0x7ffff6bff1c0, 0x0, 0x0, 
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0, view_offs = 0, data = 0x7ffff6bff490, name = "result", '\000' <repeats 57 times>, 
  extra = 0x0}
```
And then calling `ggml_compute_forward`:

```c
static void ggml_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor) {
    GGML_ASSERT(params);

    if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {
        return;
    }

    switch (tensor->op) {
        ...
        case GGML_OP_MUL_MAT:
            {
                ggml_compute_forward_mul_mat(params, tensor);
            } break;
        ...
```

```
static void ggml_compute_forward_mul_mat(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;  // index of thread
    const int nth = params->nth;  // number of threads

    const enum ggml_type type = src0->type;

    enum ggml_type    const vec_dot_type          = type_traits[type].vec_dot_type;
    ggml_from_float_t const from_float_to_vec_dot = type_traits[vec_dot_type].from_float;
    int64_t           const vec_dot_num_rows      = type_traits[type].nrows;
```
So we have our two src tensors which are 'a' anb 'b'

`GGML_TENSOR_BINARY_OP_LOCALS` expands local variables for the src tensors and
are used later in the funcion:
```c
    const int64_t ne00 = (src0)->ne[0];
    (void)(ne00);
    const int64_t ne01 = (src0)->ne[1];
    (void)(ne01);
    const int64_t ne02 = (src0)->ne[2];
    (void)(ne02);
    const int64_t ne03 = (src0)->ne[3];
    (void)(ne03);
    const size_t nb00 = (src0)->nb[0];
    (void)(nb00);
    const size_t nb01 = (src0)->nb[1];
    (void)(nb01);
    const size_t nb02 = (src0)->nb[2]; (void)(nb02); const size_t nb03 = (src0)->nb[3]; (void)(nb03); const int64_t ne10 = (src1)->ne[0]; (void)(ne10); const int64_t ne11 = (src1)->ne[1]; (void)(ne11); const int64_t ne12 = (src1)->ne[2]; (void)(ne12); const int64_t ne13 = (src1)->ne[3]; (void)(ne13); const size_t nb10 = (src1)->nb[0]; (void)(nb10); const size_t nb11 = (src1)->nb[1]; (void)(nb11); const size_t nb12 = (src1)->nb[2]; (void)(nb12); const size_t nb13 = (src1)->nb[3]; (void)(nb13); const int64_t ne0 = (dst)->ne[0]; (void)(ne0); const int64_t ne1 = (dst)->ne[1]; (void)(ne1); const int64_t ne2 = (dst)->ne[2]; (void)(ne2); const int64_t ne3 = (dst)->ne[3]; (void)(ne3); const size_t nb0 = (dst)->nb[0]; (void)(nb0); const size_t nb1 = (dst)->nb[1]; (void)(nb1); const size_t nb2 = (dst)->nb[2]; (void)(nb2); const size_t nb3 = (dst)->nb[3]; (void)(nb3);
```
Next we have:
```c
    if (ith == 0) {
        // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
        atomic_store(&params->shared->current_chunk, nth);
    }

    ggml_barrier(params->shared);
```
Now, ith is the index of the thread and this is checking that it is 0.
The `atomic_store` functions is not compiled in on Linux and may be specific to
Windows. TODO: figure out how this is done. 
Then we have `ggml_barrier`:
```c
static void ggml_barrier(struct ggml_compute_state_shared * shared) {
    if (shared->n_threads == 1) {
        return;
    }

    #pragma omp barrier
}
```
When a thread reaches this point in the code, it must wait until all other
threads in the team reach the same point. Only when all threads have arrived at
the barrier can they all proceed. 

```
(gdb) br 12209
Breakpoint 4 at 0x5555555829d4: file /home/danbev/work/ai/learning-ai/fundamentals/ggml/ggml/src/ggml.c, line 12209.
(gdb) set scheduler-locking off
(gdb) continue 
Continuing.
[Switching to Thread 0x7ffff5bfc640 (LWP 94882)]

Thread 4 "matrix-mul" hit Breakpoint 4, ggml_compute_forward_mul_mat (params=0x7ffff5bfbd20, dst=0x7ffff6bff340) at /home/danbev/work/ai/learning-ai/fundamentals/ggml/ggml/src/ggml.c:12209
12209	    ggml_barrier(params->shared);
(gdb) info thread
  Id   Target Id                                      Frame 
  1    Thread 0x7ffff7e65c00 (LWP 80908) "matrix-mul" 0x00007ffff7e8b113 in ?? () from /lib/x86_64-linux-gnu/libgomp.so.1
  2    Thread 0x7ffff6bfe640 (LWP 94880) "matrix-mul" ggml_compute_forward_mul_mat (params=0x7ffff6bfdd20, dst=0x7ffff6bff340)
    at /home/danbev/work/ai/learning-ai/fundamentals/ggml/ggml/src/ggml.c:12209
  3    Thread 0x7ffff63fd640 (LWP 94881) "matrix-mul" ggml_compute_forward_mul_mat (params=0x7ffff63fcd20, dst=0x7ffff6bff340)
    at /home/danbev/work/ai/learning-ai/fundamentals/ggml/ggml/src/ggml.c:12209
* 4    Thread 0x7ffff5bfc640 (LWP 94882) "matrix-mul" ggml_compute_forward_mul_mat (params=0x7ffff5bfbd20, dst=0x7ffff6bff340)
    at /home/danbev/work/ai/learning-ai/fundamentals/ggml/ggml/src/ggml.c:12209
```

Now, the `ggml_compute_forward_mul_mat` function is inside of a omp parallel
block so mulitple threads will be running this function.
```c
static void ggml_compute_forward_mul_mat(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

```
So thread usage in GGML is using in a way that is kind of similar to how CUDA
kernels are executed as well, the same function runs but by different threads.
This enables compution on the same matrix but using different pieces of the
data. Take matrix multiplication for example which performs the dot product.
The dot product to the resulting output matrix, position 0,0 is the dot product
of the first row of the first matrix and the first column of the second matrix.
This can be handled by one thread.

### Threading
This section will look at how threading is used/implemented in GGML.

Lets take any of the exploration examples in [ggml](../fundamentals/ggml) and
and set a breakpoint in the `ggml_graph_compute_with_ctx` function. I'll use
the `rope` example just because its the last one I worked on:
```console
$ gdb --args bin/rope
(gdb) br ggml_graph_compute_with_ctx
(gdb) r
Breakpoint 1, ggml_graph_compute_with_ctx (ctx=0x55555568a808 <g_state+8>, cgraph=0x7ffff691d610, n_threads=4) at /home/danbev/work/ai/learning-ai/fundamentals/ggml/ggml/src/ggml.c:18771
18771	enum ggml_status ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads) {
```
We can see that this functions takes a `ggml_context` which we have talked about
previously, and also a computation graph, and the number of threads to use.
```c
enum ggml_status ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads) {
    struct ggml_cplan cplan = ggml_graph_plan(cgraph, n_threads);

    struct ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_WORK_BUFFER, cplan.work_size);

    cplan.work_data = (uint8_t *)ctx->mem_buffer + obj->offs;

    return ggml_graph_compute(cgraph, &cplan);
}
```
TOOD: look into the context `mem_buffer` and how that works.

First the construction of a `ggml_cplan` will happen, which is a struct that
looks like this:
```console
(gdb) ptype struct ggml_cplan
type = struct ggml_cplan {
    size_t work_size;
    uint8_t *work_data;
    int n_threads;
    ggml_abort_callback abort_callback;
    void *abort_callback_data;
}
```
We can see that is `n_threads` is not set then the default will be used which is
currently 4.
```c
struct ggml_cplan ggml_graph_plan(const struct ggml_cgraph * cgraph, int n_threads) {
    if (n_threads <= 0) {
        n_threads = GGML_DEFAULT_N_THREADS;
    }
```
This function will iterate over all the nodes in the compute graph which is
2 in our case:
```console
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        ...
    }
```
```console
(gdb) p cgraph->n_nodes
$1 = 2
```
And notice that a node is just a tensor:
```console
(gdb) p *node
$3 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0, ne = {128, 32, 6, 1}, nb = {4, 512, 16384, 98304}, 
op = GGML_OP_RESHAPE, op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0, src = {0x7ffff68ed030, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 
0x0, 0x0, 0x0}, view_src = 0x7ffff68ed030, view_offs = 0, data = 0x7ffff68ed180, name = "a", '\000' <repeats 62 times>, extra = 0x0}
```
Following that we have this line:
```c
        const int n_tasks = ggml_get_n_tasks(node, n_threads);
```
Note that the operation of the tensor is `GGML_OP_RESHAPE`:
```c
static int ggml_get_n_tasks(struct ggml_tensor * node, int n_threads) {
    int n_tasks = 0;

    if (ggml_is_empty(node)) {
        // no need to multi-thread a no-op
        n_tasks = 1;
        return n_tasks;
    }

    switch (node->op) {
      ...
        case GGML_OP_SCALE:
        case GGML_OP_SET:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_GET_ROWS_BACK:
        case GGML_OP_DIAG:
            {
                n_tasks = 1;
            } break;
    }
```
So in this case `n_tasks` will be set to 1. And this will be returned. Reshape is
actually a no-operation in the forward pass:
```c
static void ggml_compute_forward_reshape(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst) {
    // NOP
    UNUSED(params);
    UNUSED(dst);
}
```

Back in `ggml_graph_plan` there is a switch statement in the for loop over
the nodes in the compute graph:
```c
        switch (node->op) {
            ...
            default:
                break;
```
But for the current operation, `GGML_OP_RESHAPE` there is no special handling.

For the second node which is the following:
```console
(gdb) p node->op
$12 = GGML_OP_ROPE
```
For this operation `n_tasks` will be set to the number of threads:
```c
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
        case GGML_OP_ADD_REL_POS:
            {
                n_tasks = n_threads;
            } break;
```
And this will then be returned.

Again back in the for loop in `ggml_graph_plan` this time there is a case for
`GGML_OP_ROPE`:
```c
    size_t work_size = 0;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ...
        size_t cur = 0;

        switch (node->op) {
            ...
            case GGML_OP_ROPE:
                {
                    cur = ggml_type_size(GGML_TYPE_F32) * node->ne[0] * n_tasks;
                } break;
        }
    }

    if (work_size > 0) {
        work_size += CACHE_LINE_SIZE*(n_threads - 1);
    }

    cplan.n_threads = MIN(max_tasks, n_threads);
    cplan.work_size = work_size;
    cplan.work_data = NULL;

    return cplan;
```
We can inspect the values for the calculation of `cur`:
```console
(gdb) p node->ne[0]
$14 = 128
(gdb) p ggml_type_size(GGML_TYPE_F32)
$15 = 4
(gdb) p ggml_type_size(GGML_TYPE_F32) * node->ne[0] * n_tasks
$16 = 2048
```
So `work_size` will be set to 2048 in this case, which will later be extended
to make sure that different threads are not writing to the same cache line
(false sharing). The first thread does not need this spacing as it typically
starts at the base address. This will allow each thread to operate on different
cache lines.

Our `cplan` struct will then look like this:
```console
(gdb) p cplan
$20 = {work_size = 2240, work_data = 0x0, n_threads = 4, abort_callback = 0x0, abort_callback_data = 0x0}
```
This will then be retuned to `ggml_graph_compute_with_ctx`:
```c
enum ggml_status ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads) {
    struct ggml_cplan cplan = ggml_graph_plan(cgraph, n_threads);

    struct ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_WORK_BUFFER, cplan.work_size);
    cplan.work_data = (uint8_t *)ctx->mem_buffer + obj->offs;
    return ggml_graph_compute(cgraph, &cplan);
```
GGML currently has three types of objects and I think we have discussed the
other two previously:
```c
    enum ggml_object_type {
        GGML_OBJECT_TYPE_TENSOR,
        GGML_OBJECT_TYPE_GRAPH,
        GGML_OBJECT_TYPE_WORK_BUFFER
    };
```

Lets take a closer look at `ggml_graph_compute`:
```c
enum ggml_status ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan) {
    GGML_ASSERT(cplan);
    GGML_ASSERT(cplan->n_threads > 0);
    GGML_ASSERT(cplan->work_size == 0 || cplan->work_data != NULL);

    int n_threads = cplan->n_threads;

    struct ggml_compute_state_shared state_shared = {
        /*.cgraph                  =*/ cgraph,
        /*.cgraph_plan             =*/ cplan,
        /*.n_threads               =*/ n_threads,
        /*.n_barrier               =*/ 0,
        /*.n_barrier_passed        =*/ 0,
        /*.abort_callback          =*/ NULL,
        /*.abort_callback_data     =*/ NULL,
        /*.current_chunk           =*/ 0,
        /*.ec                      =*/ GGML_STATUS_SUCCESS,
    };

#ifdef GGML_USE_OPENMP
    if (n_threads > 1) {
        #pragma omp parallel num_threads(n_threads)
        {
            #pragma omp single
            {
                // update the number of threads from the actual number of threads that we got from OpenMP
                n_threads = omp_get_num_threads();
                state_shared.n_threads = n_threads;
            }

            struct ggml_compute_state worker = {
                .thrd   = 0,
                .ith    = omp_get_thread_num(),
                .shared = &state_shared,
            };
            ggml_graph_compute_thread(&worker);
        }
    } else {
        struct ggml_compute_state worker = {
            .thrd   = 0,
            .ith    = 0,
            .shared = &state_shared,
        };
        ggml_graph_compute_thread(&worker);
    }
```
The member `ec`in the `ggml_compute_state_shared` struct is the error code (ec)
which for some reason was not obvious to me initially.

```
        #pragma omp parallel num_threads(n_threads)
```
This is an OpenMP directive that specifies that the following block of code
should be executed in parallel by `n_threads`. So this will start 4 threads in
our case.

The next OMP directive is `#pragma omp single` which specifies that the block
should be executed by a single thread. So one of those four threads will execute
the block of code that follows, which in this case just gets the number of
threads from OpenMP and sets the `n_threads` to that value.

If we set a breakpoint in the single block and the parallel block we can
inspect the threads that have been created:
```console
(gdb) br ggml.c:18715
(gdb) continue

Breakpoint 3 at 0x5555555a8ef4: file /home/danbev/work/ai/learning-ai/fundamentals/ggml/ggml/src/ggml.c, line 18715.
(gdb) continue 
Continuing.
[New Thread 0x7ffff68ec640 (LWP 451923)]
[New Thread 0x7ffff60eb640 (LWP 451924)]
[New Thread 0x7ffff58ea640 (LWP 451925)]
[Switching to Thread 0x7ffff58ea640 (LWP 451925)]

Thread 4 "rope" hit Breakpoint 3, ggml_graph_compute._omp_fn.0 () at /home/danbev/work/ai/learning-ai/fundamentals/ggml/ggml/src/ggml.c:18715
18715	                n_threads = omp_get_num_threads();

(gdb) info thread
  Id   Target Id                                 Frame 
  1    Thread 0x7ffff7e64c00 (LWP 450569) "rope" 0x00007ffff7e8a0ca in ?? () from /lib/x86_64-linux-gnu/libgomp.so.1
  2    Thread 0x7ffff68ec640 (LWP 451923) "rope" 0x00007ffff7e8a0ca in ?? () from /lib/x86_64-linux-gnu/libgomp.so.1
  3    Thread 0x7ffff60eb640 (LWP 451924) "rope" 0x00007ffff7e8a0ca in ?? () from /lib/x86_64-linux-gnu/libgomp.so.1
* 4    Thread 0x7ffff58ea640 (LWP 451925) "rope" ggml_graph_compute._omp_fn.0 ()
    at /home/danbev/work/ai/learning-ai/fundamentals/ggml/ggml/src/ggml.c:18715
```
Now, lets enable thread locking:
```console
(gdb) set scheduler-locking on
```
This is the single thread block which will execute and set the
`state_shared.n_threads`:
```c
            #pragma omp single
            {
                // update the number of threads from the actual number of threads that we got from OpenMP
                n_threads = omp_get_num_threads();
                state_shared.n_threads = n_threads;
            }
```
So lets also set a break point in the after the single thread block to be able
to step through it.
```c
            struct ggml_compute_state worker = {
                .thrd   = 0,
                .ith    = omp_get_thread_num(),
                .shared = &state_shared,
            };
            ggml_graph_compute_thread(&worker);
```
So the above will get executed by each thread and each will set teh `ith` member
of the `ggml_compute_state` struct to the thread number. For example the current
thread will set it to 3:
```console
(gdb) p (int) omp_get_thread_num()
$23 = 3
```
Note that `thrd` is short for `thread` and its type is `ggml_thread_t`:
```console
(gdb) ptype struct ggml_compute_state
type = struct ggml_compute_state {
    ggml_thread_t thrd;
    int ith;
    struct ggml_compute_state_shared *shared;
}
```
And notice that all `ggml_compute_state`'s will have a pointer to the shared
compute state which contains:
```console
(gdb) p state_shared
$24 = {cgraph = 0x7ffff691d610,
       cplan = 0x7fffffffdd90,
       n_threads = 4,
       n_barrier = 0,
       n_barrier_passed = 0,
       abort_callback = 0x0, 
       abort_callback_data = 0x0,
       current_chunk = 0,
       ec = GGML_STATUS_SUCCESS}
```

After that the worker struct will be passed to `ggml_graph_compute_thread`:
```c
static thread_ret_t ggml_graph_compute_thread(void * data) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;

    const struct ggml_cgraph * cgraph = state->shared->cgraph;
    const struct ggml_cplan  * cplan  = state->shared->cplan;

    set_numa_thread_affinity(state->ith);

    struct ggml_compute_params params = {
        /*.ith   =*/ state->ith,
        /*.nth   =*/ state->shared->n_threads,
        /*.wsize =*/ cplan->work_size,
        /*.wdata =*/ cplan->work_data,
        /*.shared=*/ state->shared,
    };

    for (int node_n = 0; node_n < cgraph->n_nodes; node_n++) {
        struct ggml_tensor * node = cgraph->nodes[node_n];

        ggml_compute_forward(&params, node);

        if (state->ith == 0 && cplan->abort_callback && cplan->abort_callback(cplan->abort_callback_data)) {
            state->shared->ec = GGML_STATUS_ABORTED;
        }

        ggml_barrier(state->shared);

        if (state->shared->ec != GGML_STATUS_SUCCESS) {
            break;
        }
    }

    return 0;
}
```
A `ggml_compute_params` struct is created which contains the thread number (3
in the current session/thread), the number of threads (4), the work size (2240),
the work data, and also a pointer to the shared compute state.

Now, recall that all the threads will execute this function and not just one,
and all of them will loop through the nodes in the compute graph.
And in this case the first node is the rehape tensor which will be passed to
`ggml_compute_forward`:
```c
static void ggml_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor) {
    GGML_ASSERT(params);

    if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {
        return;
    }

    switch (tensor->op) {
        ...
        case GGML_OP_RESHAPE:
            {
                ggml_compute_forward_reshape(params, tensor);
            } break;
        ...
    }
```
And like we mentioned earlier rehape is a no-operation in the forward pass:
```c
static void ggml_compute_forward_reshape(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst) {
    // NOP
    UNUSED(params);
    UNUSED(dst);
}
```
Back in `ggml_graph_compute_thread` we have the following:
```c
        if (state->ith == 0 && cplan->abort_callback && cplan->abort_callback(cplan->abort_callback_data)) {
            state->shared->ec = GGML_STATUS_ABORTED;
        }

        ggml_barrier(state->shared);
```
In our case/thread the `state->ith` is 3 so that block will only be performed by
thread 0. Next we have the `ggml_barrier` which will a syncrhonization
construct which ensures that all threads in a parallel region (OpenMP block
which we are currently in). So all theads must wait for the others at this
point. For us this means that nothing else will happen as we enabled thread
locking earlier. So lets disable that and continue.
```console
(gdb) set scheduler-locking off
(gdb) continue
```
When we hit the next breakpoint we can inspect the node/tensor to see that it is
the `GGML_OP_ROPE` operation and then again enable thread locking.
What we are interested in is the usage of the params like `ith`
```c
    const int ith = params->ith;  // thread number
    const int nth = params->nth;  // number of threads
```
Next we have the calculation of the number of rows which I'd like to see how it
is implemented:
```c
    const int nr = ggml_nrows(dst);
```
```c
GGML_CALL int64_t ggml_nrows(const struct ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");
    return tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
}
```
Our tensor has the following shape:
```console
(gdb) p tensor->ne
$42 = {128, 32, 6, 1}
```
128 is the number of dimensions, 32 the number of heads, 6 is the number of
tokens in the sequence, and 1 is the batch size. So this will become
32 * 6 * 1 = 192 rows.

Next we have the calculation of the number of rows per thread:
```c
    // rows per thread
    const int dr = (nr + nth - 1)/nth;
```
This will become (192 + 4 - 1)/4 = 48. So each thread will handle 48 rows.
We can visualize this something like this:
```
    0                               128
0   [                                 ]  thread 0
                    ...
    [                                 ]
47  [                                 ]  thread 1
                    ...
    [                                 ]  
95  [                                 ]  thread 2
                    ...
    [                                 ]  
143 [                                 ]  thread 3
                    ...
    [                                 ] 
191 [                                 ]
```

```c
    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);
```
So `ir0` (index row) will be 48 * 3 = 144.

Next we have a local variable named `ir`:
```c
    // row index used to determine which thread to use
    int ir = 0;
```
I'm going to skip the details of the rope implementation as this has been
covered in [ggml-rope.md](positional-encoding/ggml-rope.md) and focus on the
code related to multithreading.
Below there is an outerloop that loops over the number of batches (which is just
one in this case), then it will loop over the number of tokens in the sequence
which is 6. And the it will loop over the number of heads:
```c
    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            ...
            for (int64_t i1 = 0; i1 < ne1; i1++) {
                if (ir++ < ir0) continue;
                if (ir   > ir1) break;
```
This is the part that is interesting to this section. Recall that `ir` was set
to 0 initially and `ir0` was set to 144 (the rows that this thread is going to
process). So what this is doing is that it contining the loop as long as `ir` 
is not rows that this thread is going to process. And if `ir` has reached `ir1`
then it will break out of the loop. So when the body of the loop is executed i1
will be in the range 143 to 191.
And I think this is how the multi-threading is implemented in GGML.

### Backpropagation
This section deals with the backpropagation in GGML.
The following example [backprop.c](../fundamentals/ggml/backprop.c) will be used
to explore backpropagation in this secion.

First lets take a look at the compute graph which we have seen ealier. 
```c
    struct ggml_cgraph {
        int size;
        int n_nodes;
        int n_leafs;

        struct ggml_tensor ** nodes;
        struct ggml_tensor ** grads;
        struct ggml_tensor ** leafs;

        struct ggml_hash_set visited_hash_set;

        enum ggml_cgraph_eval_order order;
    };
```
So just like the compute graph has nodes, and leafs it also has grads which are
the gradients.

Now, lets take a look at the `ggml_compute_backward` function:
```c
  struct ggml_cgraph* b_graph = ggml_graph_dup(ctx, f_graph);
  ggml_build_backward_expand(ctx, f_graph, b_graph, /* keep gradients */ false);
```

```c
void ggml_build_backward_expand(struct ggml_context * ctx,
    struct ggml_cgraph * gf, struct ggml_cgraph * gb, bool keep) {

    if (keep) {
        for (int i = 0; i < gf->n_nodes; i++) {
            struct ggml_tensor * node = gf->nodes[i];

            if (node->grad) {
                node->grad = ggml_dup_tensor(ctx, node);
                gf->grads[i] = node->grad;
            }
        }
    }
```
So the last parameter is false in our case which means that we don't want to keep the
gradients from the forward compute graph, so this is duplicating those tensors so they
won't be modified.
