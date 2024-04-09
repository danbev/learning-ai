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
  struct ggml_tensor* view = ggml_view_1d(ctx, x, 5, 5 * ggml_type_size(x->type));
```
Where the first int argument is the number of elements and the second integer is
the offset in bytes. Notice that we have take into account the size for the
elements stored.
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

So first we have an interface which describes the buffer, the buffer type:
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

