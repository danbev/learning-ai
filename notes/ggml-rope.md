## GGML RoPE implementation notes

This document contains a walkthrough of the RoPE function in GGML.

The code for this can be found in [rope.c](../fundamentals/ggml/src/rope.c).

```console
$ gdb --args bin/rope
```
The first function call we make is to set up the tensors and operations in the
context which is done by calling `ggml_rope_ext`.
```c
struct ggml_tensor * ggml_rope_ext(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        struct ggml_tensor  * c,
        int                   n_dims,
        int                   mode,
        int                   n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    return ggml_rope_impl(
        ctx, a, b, c, n_dims, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, false
    );
}
```
The tensor `a` is the tensor that is to be rotated. The `b` tensor is a one
dimensional vector that contains the positions. And I think that `c` is a
tensor that contains scaling factors but I've not gotten this far in my
understanding of this yet, and in the example it NULL is passed in.

`n_dims` is, `d` in RoPE. TODO: link with rope.md document.
`mode` is ?
`n_ctx_orig` is the models original context that it was trained on, this might
be used by PI and used to calculate 's' (s = L'/L) I think?
`freq_base` is the base frequency which is 10000 by default.
`freq_scale` is the scaling factor which I thought might be -2 because this is
what the paper uses but the default for this is 1.0f.
`ext_factor` might be the extrapolation factor but I'm not sure.
`attn_factor` ? 
`beta_fast` might be the scaling factor for YaRN and which should be used for
scaling the higher frequencies.
`beta_slow` also related to YaRN and might be  used to scale the lower
frequencies.

So that will set up the context and the tensor operations for RoPE. Next we
want to create a computation graph and run the forward pass:
```c
  struct ggml_cgraph* c_graph = ggml_new_graph(ctx);
  ggml_build_forward_expand(c_graph, s);
  ggml_graph_compute_with_ctx(ctx, c_graph, 4);
```

Now, we can set a breakpoint in `ggml_compute_forward_rope`:
```console
(gdb) br ggml_compute_forward_rope
Breakpoint 5 at 0x55555558acf8: file /home/danbev/work/ai/learning-ai/fundamentals/ggml/ggml/src/ggml.c, line 14061.
(gdb) r
```
Keep in mind that the excution in GGML is multithreaded and multiple threads
will be running when our break point is it. So just continuing stepping will
this using:
```console
(gdb) set scheduler-locking on
```
And now we can step only the current thread.

```c
static void ggml_compute_forward_rope(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_rope_f16(params, dst, true);
            } break;
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_rope_f32(params, dst, true);
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }
}
```
I need to readup on how this works but with some handwaving the
`ggml_compute_params` is part of GGMLs multithreading and contains information
about which thread and what part of the tensor this thread is working on.
```console
(gdb) p *dst->src[0]
$9 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0, ne = {128, 32, 512, 1}, nb = {4, 512, 16384, 8388608}, op = GGML_OP_RESHAPE, 
  op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0, src = {0x7ffff68ed030, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, 
  view_src = 0x7ffff68ed030, view_offs = 0, data = 0x7ffff68ed180, name = "a_reshaped", '\000' <repeats 53 times>, extra = 0x0}
```
In our case the type of the src tensor is F32:
```c
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_rope_f32(params, dst, true);
            } break;
```

```c
static void ggml_compute_forward_rope_f32(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst,
        const bool forward) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const struct ggml_tensor * src2 = dst->src[2];
```
```console
(gdb) p *src0
$12 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0, ne = {128, 32, 512, 1}, nb = {4, 512, 16384, 8388608}, op = GGML_OP_RESHAPE, 
  op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0, src = {0x7ffff68ed030, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, 
  view_src = 0x7ffff68ed030, view_offs = 0, data = 0x7ffff68ed180, name = "a_reshaped", '\000' <repeats 53 times>, extra = 0x0}

(gdb) p *src1
$13 = {type = GGML_TYPE_I32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0, ne = {512, 1, 1, 1}, nb = {4, 2048, 2048, 2048}, op = GGML_OP_NONE, 
  op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0, src = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0, view_offs = 0, 
  data = 0x7ffff70ed460, name = "pos", '\000' <repeats 60 times>, extra = 0x0}
```
And in this case we did not specify a c so src2 is null.
Next the parameters, that is the parameters that were set on the operation
parameters and not the computation params:
```console
(gdb) p dst.op_params
$16 = {0, 128, 0, 0, 4096, 1176256512, 1065353216, 0, 1065353216, 1107296256, 0, 0, 0, 0, 0, 0}
```
So these parameters are extracted into local variables:
```c
    const int n_dims     = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];

    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;

    memcpy(&freq_base,   (int32_t *) dst->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));


    GGML_TENSOR_UNARY_OP_LOCALS
```
The macro will create local variables like the following (which can be generated
by the make target `pre-ggml.c`).`:
```c
    const int64_t ne00 = (src0)->ne[0]; (void)(ne00);
    const int64_t ne01 = (src0)->ne[1]; (void)(ne01);
    const int64_t ne02 = (src0)->ne[2]; (void)(ne02);
    const int64_t ne03 = (src0)->ne[3]; (void)(ne03);

    const size_t  nb00 = (src0)->nb[0]; (void)(nb00);
    const size_t  nb01 = (src0)->nb[1]; (void)(nb01);
    const size_t  nb02 = (src0)->nb[2]; (void)(nb02);
    const size_t  nb03 = (src0)->nb[3]; (void)(nb03);

    const int64_t  ne0 = (dst)->ne[0]; (void)(ne0);
    const int64_t  ne1 = (dst)->ne[1]; (void)(ne1);
    const int64_t  ne2 = (dst)->ne[2]; (void)(ne2);
    const int64_t  ne3 = (dst)->ne[3]; (void)(ne3);

    const size_t   nb0 = (dst)->nb[0]; (void)(nb0);
    const size_t   nb1 = (dst)->nb[1]; (void)(nb1);
    const size_t   nb2 = (dst)->nb[2]; (void)(nb2);
    const size_t   nb3 = (dst)->nb[3]; (void)(nb3);
```
So this is simply extracting local variables for src0 and dst and the casts are
to avoid warnings that the variables might not be used.


A little further down we have the following:
```c
    const float theta_scale = powf(freq_base, -2.0f/n_dims);
```
Now, this looks familar (at least after reading [rope.md](rope.md) and notice
that this also clarifies that the `freq_scale` is not the scaling factor for
the frequency which I originally thought.
```console
(gdb) p theta_scale
$27 = 0.865964353
```
Following that we have this function:
```c
    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
```
And notice that this is where we use (or at least pass in) the original context
length, the `freq_base`, `beta_fast` and `beta_slow`. And notice that corr_dims
is an array of two floats that will be populated by the function.

```c
GGML_CALL void ggml_rope_yarn_corr_dims(
    int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]
) {
    // start and end correction dims
    float start = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    float end   =  ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = MAX(0, start);
    dims[1] = MIN(n_dims - 1, end);
}
```
Lets break this down a little and start by looking the function `ggml_rope_yarn_corr_dim`:
```c
static float ggml_rope_yarn_corr_dim(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * logf(n_ctx_orig / (n_rot * 2 * (float)M_PI)) / (2 * logf(base));
}
```
Notice that we are passing in `beta_fast` and `beta_slow` as `n_rot`.
```c
// Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)` for x, we get
// `corr_dim(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`
static float ggml_rope_yarn_corr_dim(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * logf(n_ctx_orig / (n_rot * 2 * (float)M_PI)) / (2 * logf(base));
}
```
What is going on here is that this function calculates the start and end
dimensions for which rotation corrections should be applied. So in our case
they should be applied for dimensions greater than 48 and less than 80.

From the paper we see:

```
       2 PI      
λ_d = -------  = 2PI b^(2d|D|)
       theta_d   

d = index of the hidden dimension.
b = base frequency.
D = number of hidden dimensions.
```
λ_d indicates how many tokens are required for the positional encoding at the
d-th dimension to cycle through a complete period.

This describes the number of tokens needed for the positional embedding at the 
d-th hidden dimension to complete a full rotation of 2PI.


```
          L         L 
r(d) = -------- = ----------
       lambda_d    2PI b^2d|D|

```

```console
(gdb) s
ggml_rope_yarn_corr_dim (n_dims=128, n_ctx_orig=4096, n_rot=32, base=10000) at /home/danbev/work/ai/learning-ai/fundamentals/ggml/ggml/src/ggml.c:13778
13778	    return n_dims * logf(n_ctx_orig / (n_rot * 2 * (float)M_PI)) / (2 * logf(base));
```
So this becomes `128 * log(4096 / (32 * 2 * pi)) / (2 * log(10000))` which is
`128 * log(4096 / 201.06193) / (2 * log(10000))` which is `128 * log(20.354) /
(2 * 4)` which is `128 * 3.015 / 8` which is `48`. So the start is 48 and the
end is 80. And this is the start and end of the correction dimensions.
