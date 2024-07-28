## GGML RoPE implementation notes

This document contains a walkthrough of the RoPE function in GGML.

```console
gdb --args ./llama-cli -m models/llama-2-7b.Q4_0.gguf --no-warmup --rope-scaling yarn --rope-freq-scale 1 --yarn-ext-factor 1.0 -ngl 10 -p "What is LoRA?" -n 10
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

* `n_dims` is, `d` in RoPE. TODO: link with rope.md document.
* `mode` is ?
* `n_ctx_orig` is the models original context that it was trained on, this might
be used by PI and used to calculate 's' (s = L'/L) I think?
* `freq_base` is the base frequency which is 10000 by default.
* `freq_scale` TODO: what is this?
* `ext_factor` might be the extrapolation factor but I'm not sure.
* `attn_factor` TODO: what is this? 
* `beta_fast` might be the scaling factor for YaRN and which should be used for
scaling the higher frequencies.
* `beta_slow` also related to YaRN and might be  used to scale the lower
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
will be running when our break point is hit. So we will disable this behavior
and lock the current thread using the following command:
```console
(gdb) set scheduler-locking on
```
And now we can step using only the current thread.

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
I need to read up on how this works but with some handwaving the
`ggml_compute_params` is part of GGMLs multithreading and contains information
about which thread and what part of the tensor this thread is working on.

```console
(gdb) p *dst->src[0]
$9 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0,
ne = {128, 32, 512, 1}, 
b = {4, 512, 16384, 8388608},
op = GGML_OP_RESHAPE, 
op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0,
src = {0x7ffff68ed030, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, 
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

Now `src0` is the tensor that is goint to be rotated and it will have a shape of
`{128, 32, sequence_length, 1}`. And `src1` is the tensor that contains the
positions and it be a vector of the same size of the sequence length.

And in this case we did not specify a `c` argument so `src2` is null.

Next the parameters, that is the tensor operation parameters not the computation
params:
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
by the make target `pre-ggml.c`):
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
So this is simply extracting local variables from src0 and dst and the casts are
to avoid warnings that the variables might not be used. Specifically it is
creating local variables for the number of elements and the number the strides.

A little further down we have the following:
```c
    const float theta_scale = powf(freq_base, -2.0f/n_dims);
```
Now, this looks familar (at least it might be after reading [rope.md](rope.md)
and notice that this also clarifies that the `freq_scale` is not the scaling
factor for the frequency which I initially thought (the -2f that is).
```console
(gdb) p theta_scale
$27 = 0.865964353
```

Following that we have `ggml_rope_yarn_corr_dims`:
```c
    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
```
So this is where `beta_fast` and `beta_slow` are used and they correspond to
the alpha and beta parameters in the YaRN paper. These are used to define the
boundaries for the different interpolation strategies.

Notice that `corr_dims` is an array of two floats that will be populated by the
function.

```c
GGML_CALL void ggml_rope_yarn_corr_dims(
    int n_dims,
    int n_ctx_orig,
    float freq_base,
    float beta_fast,
    float beta_slow,
    float dims[2]
) {
    // start and end correction dims
    float start = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));

    float end   =  ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));

    dims[0] = MAX(0, start);
    dims[1] = MIN(n_dims - 1, end);
}
```

```console
(gdb) s
ggml_rope_yarn_corr_dims (n_dims=128,
                          n_ctx_orig=4096,
                          freq_base=10000,
                          beta_fast=32,
                          beta_slow=1,
                          dims=0x7fffffffb220) at ggml/src/ggml.c:13941
13941	    float start = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
```

Lets break this down a little and start by looking first call to
`ggml_rope_yarn_corr_dim`:
And this will first call `ggml_rope_yarn_corr_dim`:
```console
(gdb) s
ggml_rope_yarn_corr_dim (n_dims=128, n_ctx_orig=4096, n_rot=0, base=10000) at /home/danbev/work/ai/learning-ai/fundamentals/ggml/ggml/src/ggml.c:13778
13778	    return n_dims * logf(n_ctx_orig / (n_rot * 2 * (float)M_PI)) / (2 * logf(base));
```
Notice that we are passing in `beta_fast` as `n_rot`. This function has the
following comment:
```c
// Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)` for x, we get
// `corr_dim(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`
```
This is calculating the dimension that would give us a certain number of
rotations (n_rot) at the maxium position (n_ctx_orig).
```console
n_dims=128, n_ctx_orig=4096, n_rot=32, base=10000
(gdb) p n_dims * logf(n_ctx_orig / (n_rot * 2 * (float)3.14)) / (2 * logf(base))
$10 = 20.9480038
```
The value `20.95` represents a specific dimension of the position embedding when
the sequence length is at the maximum value (n_ctx_orig).

So imagine we have a sequence length of 4096, what we are asking is that at what
dimension does the position embedding rotate 32 times.
Dimensions lower than 20.95 will rotate more than 32 times and dimensions higher
will rotate fewer times.
This value will be passed to floorf so it will become 20.

Then we will do the same for `beta_slow`:
```console
(gdb) s
ggml_rope_yarn_corr_dim (n_dims=128, n_ctx_orig=4096, n_rot=1, base=10000) at ggml/src/ggml.c:13918
13918	    return n_dims * logf(n_ctx_orig / (n_rot * 2 * (float)M_PI)) / (2 * logf(base));
(gdb) p n_dims * logf(n_ctx_orig / (n_rot * 2 * (float)3.14)) / (2 * logf(base))
$11 = 45.0304031
```
And similar to above here we are asking at which dimension the position rotates
only once (slower frequence compared to the above).
And this value will be passed to ceilf so it will become 46.

```
(gdb) p corr_dims 
$17 = {20, 46}
```
So the above will figurare out the range of dimensions for which interpolation
extrapolation will be performed and used later when we iterate over all the
token position embeddings.



The following code will iterate over all the batches, which is 1 in this case.
And then iterate over all the tokens in the sequence (ne2) which is my case is
only 6 as the prompt was "What is LoRA?".

Lets just take a quick look at the shape of the tensor that we are going token
rotate:
```console
(gdb) p src0.ne
$18 = {128, 32, 6, 1}
```
So we have 128 position embeddings, 32 heads, 6 tokens and 1 batch.

```
       +----------------------------------------------+
      /                                              /|
     /   [position embeddings]                      / |
    /                                             /   |
  0/                                            127   |
  +----------------------------------------------+    |
  |                                              |    |
  |                                              |    |
  |                                              |  [heads]
  |                                              |    |
  |                                              |   /5
  |                                              |  / tokens in sequence (6)
  |                                              | /0
  +----------------------------------------------+
31
```
Is this heads or layers?  TODO: I think we heads but verify this!

So we first iterate over the batches (ne3), followed by the number of tokens
in the sequence (ne2=6):
```c
    const int32_t * pos = (const int32_t *) src1->data;

    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            const int64_t p = pos[i2];

            float * cache = (float *) params->wdata + (ne0 + CACHE_LINE_SIZE_F32)*ith;
            ggml_rope_cache_init(p,
                                 freq_scale,
                                 freq_factors,
                                 corr_dims,
                                 ne0,
                                 ext_factor,
                                 attn_factor,
                                 cache, sin_sign, theta_scale);
```
So for each token we iterate over we will initialize a cache for the position
encodings.

One thing I missed the first time through this is that the position `p` is
passed as the `theta_base` so it will be 0 for the first iteration. 
```c
static void ggml_rope_cache_init(
     float theta_base,
     float freq_scale,
     const float * freq_factors,
     float corr_dims[2],
     int64_t ne0,
     float ext_factor,
     float mscale,
     float * cache,
     float sin_sign,
     float theta_scale) {

    float theta = theta_base;

    for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        const float ff = freq_factors ? freq_factors[i0/2] : 1.0f;
        rope_yarn(
            theta/ff, freq_scale, corr_dims, i0, ext_factor, mscale, &cache[i0 + 0], &cache[i0 + 1]
        );
        cache[i0 + 1] *= sin_sign;

        theta *= theta_scale;
    }
}
```
This will iterate over all the position embedding dimensions which is 128 for
a single token in the sequence.

And notice that this is done in pairs which is the pairwise rotation.
So this is where the third tensor, 'c' is used which is called frequency factors.
If this is null the frequency factor is 1.0f. 

These are the arguments passed to `ggml_rop_cache_init`:
```console
(gdb) s
ggml_rope_cache_init (
theta_base=0,
freq_scale=1,
freq_factors=0x0,
corr_dims=0x7ffeb6e90790,
ne0=128,
ext_factor=1,
mscale=1,
cache=0x5555645547b0,
sin_sign=1,
theta_scale=0.865964353) at ggml/src/ggml.c:13952
```
Like we mentioned before `theta_base` is the position and in this case it will
be 0 since this is the first token in the sequence. This value is set as the
initial value for the rotation angle `theta`. The code will iterate over the
number of position embedding dimensions which is 128 in this case pairwise.


We then call `rope_yarn`.
```c
static void rope_yarn(
    float theta_extrap,        // yarn_ext_factor parameter.
    float freq_scale,          // rope-freq-scale parameter.
    float corr_dims[2],
    int64_t i0,                // the current position embedding dimension.
    float ext_factor,          // yarn-ext-factor parameter.
    float mscale,              // yarn-attn-factor parameter.
    float * cos_theta,
    float * sin_theta) {

    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}
```
The function `rope_yarn_ramp` is related to the gamma function in the YaRN
paper. `i0` is the position embedding dimension we are currently iterating over.
```c
static float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / MAX(0.001f, high - low);
    return 1 - MIN(1, MAX(0, y));
}
```
Recall that we are iterating over all the position embedding dimensions which are
128 in this case, but we do this in pairs. This is the reason for `i0/2` in the
code above. So we are taking the current position (0) minus the value of low
which is 20 in our case. Then we are taking the maximum of 0.001 and the
difference between high and low which is 46 - 20 = 26. And we are then taking
the ratio of these values:
```
-20 / 26 = -0.769230769
```
Now,  low represents the dimension where interpolation starts to transition to
extrapolation and high is the dimension where the transition is complete. i0/2 is
the current dimension pair we are calculating the gamma function for.

* When i0/2 < low y will be negative.
* When i0/2  is between low and high y will be between 0 and 1.
* When i0/2 > high y will be greater than 1.

This value is clamped and inverted  using:
```
    return 1 - MIN(1, MAX(0, y));
```
So for our first dimension we will get:
-20 / 26 = -0.769230769
1 - MIN(1, MAX(0, -0.769230769)) = 
1 - MIN(1, -0.769230769) = 
1 - 0 = 1
```

So this will return:
* For dimensions < low we return 1 (full interpolation)
* For dimensions > high we return 0 (full extrapolation)
* For dimensions between low and high we return a value between 0 and 1.

We then use the value of `ramp_mix`:
```c
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
```
When `ramp_mix` is 0 we will use the interpolated value which is the case for
the current iteration:
```
theta = thread_interp * (1) + 0;
theta = thread_interp * 1;
theta = thread_interp;
```
But if `ramp_mix` is 1 we will use the extrapolated value:
```
theta = theta_interp * (0) + theta_extrap * 1;
theta = theta_extrap * 1;
theta = theta_extrap;
```
For a value between 0 and 1 we will get a mix:
```
theta = theta_interp * (1 - 0.5) + theta_extrap * 0.5;
theta = theta_interp * (0.5) + theta_extrap * 0.5;
```
We then scale the magnitude of the rotation:
```c
    mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
```
This is the "length scaling" trick referred to in the YaRN paper, in Equation
22. This is something that the authors observed that introducing a temperature
`t` on the logits before the softmax function so they modify they scaled
attention to:
```
        q^T_m k_n
softmax(----------)
          t√|D|
```
Recall that the scaling is usually √|D|.

This temperature controls how "sharp" or "soft" the attention is. A high
temperature will make the attention more uniform and a low temperature will make
the attention more focused. When extending to longer sequences the attention
patterns become diluted or less effective. For example, lets say we have an 
article that is less than the max context length that the model was trained on,
and when processing this article the model attention makes strong connections/
focuses on certain tokens. But if we now extend the article beyond the max
length the model was trained on the attention will be less focused and the
connections Will be weaker. Recall that the attention score is calculated using:
```
        q^T_m k_n
softmax(----------)
          t√|D|
```
And recall that softmax normalizes the scores so that they sum to 1. With longer
context this normalization is spread over more tokens. So the raw attention
score (pre softmax) are not lower, but because the normailzation is spread over
more tokens the effect of them is diluted.
For example:
```
   [A B C D]

Attention scores: [0.1, 0.7, 0.1, 0.1]

B has the highest attention score, 0.7 and the model will focus on this token.

   [A B C D E F G H I J]

Attention scores: [0.05, 0.3, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.05, 0.1]

B still has the highest attention score but it is now 0.3.
```
So that makes sense but how does the temperature `t` help?  
Lets take the same example but now we introduce a temperature `t`:
```
   [A B C D]

t = 1 (no scaling)
Raw scores:       [2,      8,   3,      1]
Attention scores: [0.04, 0.87, 0.08, 0.01]

B: 0.87
   [A B C D E F G H I J]

t = 1 (no scaling)
Raw scores        [2,      8,   3,     1,   2,    2,    1,      1]
Attention scores: [0.03, 0.71, 0.05, 0.01, 0.03, 0.03, 0.01, 0.01]
B: 0.71

t = 1.3
Raw scores        [2,     8,   3,     1,   2,    2,    1,      1]
                  [/1.3, /1.3, /1.3, /1.3, /1.3, /1.3, /1.3, /1.3]
Raw scaled        [1.54, 6.15, 2.31, 0.77, 1.54, 1.54, 0.77, 0.77]
Attention scores: [0.05, 0.62, 0.09, 0.02, 0.05, 0.05, 0.02, 0.02]
B: 0.62
```
The attention to B is still the highest, but it's not as extreme so the model
can still pay more attention to other relevant tokens. And the new tokens get
slightly higher attention scores enabling better long-range dependencies.


As sequences get longer (s increases), the temperature t increases slightly.

1.0f is the base value and it guarantees that the scaling factor is at least 1.
0.1f is a constant from the paper and was found to work well in practice.
1.0f / `freq_scale` is  equivalent to `√1/t` in the paper.

If `freq_scale` is 1.0:
```
    mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);

    mscale *= 1.0f + 0.1f * logf(1.0f / 1);
    mscale *= 1.0f + 0.1f * logf(1.0f / 1);
    mscale *= 1.0f + 0.1f * logf(1);
    mscale *= 1.0f + 0.1f * 0;
    mscale *= 1.0f + 0;
    mscale *= 1.0f;
```
So in this case there will be no scaling.

Instead of directly modifying the attention computation, they propose scaling
the RoPE embeddings. This achieves the same effect as scaling the attention
scores, but without changing the attention code.

Notice the last two lines:
```c
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
```
So this is calculating the angles for cosine and sine for the rotation, and
notice that it is also scaling the result of cos(theta) and sin(theta).
In the YaRN paper section 3.4 YaRN they describe this as introducing a
temperature `t` on the logits before the softmax function:
```
         q^T_m k_n
softmax(-----------)
          t√|D|
```
They use a `scaling trick` where they achive the same effect by scaling the
RoPE embeddings by √1/t which in our case would be √1/mscale or √1/attn_factor.
By doing this there are no changes to the model architecture and the only
difference is the scaling of the embeddings.

For each token in the sequence we will call `ggml_rope_cache_init` which will
iterate over all the embedding dimensions for the current token.
When that is done the cache as been populated with the cosine and sine values
for the rotation of the position embeddings.

We will then iterate over ne1 which is the number of heads I think:
```console
(gdb) p ne1
$31 = 32
```

This will first check that the current thread should be handling this span of
values (something like that) so we'll skip ahead using `until`:
```c
            for (int64_t i1 = 0; i1 < ne1; i1++) {
                if (ir++ < ir0) continue;
                if (ir   > ir1) break;
```
```console
(gdb) until 14055
ggml_compute_forward_rope_f32 (params=0x7ffeb6e90860, dst=0x7fffccaa37e0, forward=true) at ggml/src/ggml.c:14055
14055	                if (!is_neox) {
```

The following will iterate over all the 128 embedding dimensions for the current
token in the sequence:
```c
                if (!is_neox) {
                    for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                        const float cos_theta = cache[i0 + 0];
                        const float sin_theta = cache[i0 + 1];

                        const float * const src = (float *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                              float * dst_data  = (float *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                        const float x0 = src[0];
                        const float x1 = src[1];

                        dst_data[0] = x0*cos_theta - x1*sin_theta;
                        dst_data[1] = x0*sin_theta + x1*cos_theta;
                    }
                } else {
```

```c
const float * const src = (float *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
```
The cast to `char *` is enable pointer arithmetic. Lets remind ourselves what
src0 is:
```console
(gdb) p *src0
$43 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU,
buffer = 0x555563d35770, ne = {128, 32, 6, 1}, 
nb = {4, 512, 16384, 98304}, op = GGML_OP_RESHAPE, op_params = {0 <repeats 16 times>}, flags = 0, 
grad = 0x0, src = {0x7fffccaa3220, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, 
view_src = 0x7fffccaa3220, view_offs = 0, data = 0x7ffdf0231060, 
name = "Qcur-0 (reshaped)", '\000' <repeats 46 times>, extra = 0x0}
```
And `i3` is the current batch. Notice that the `nb` values are used which are
like stides into the tensor data. So what this is doing is it is getting setting
a pointer to the correct position of the src0 tensor for the current batch, head
and token in the sequence.
And since we use pairs of values we extract the two values:
```c
    const float x0 = src[0];
    const float x0 = src[0];
```
And then we perform the actual rotation:
```c
    dst_data[0] = x0*cos_theta - x1*sin_theta;
    dst_data[1] = x0*sin_theta + x1*cos_theta;
```
After we have iterated over all the dimensions (128 in this case).

The final for loop for when the input dimension ne0 is larger than the number
of dimensions used for RoPE:
```c
                for (int64_t i0 = n_dims; i0 < ne0; i0 += 2) {
                    const float * const src = (float *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                    float * dst_data  = (float *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                    dst_data[0] = src[0];
                    dst_data[1] = src[1];
                }
```
Notice that this actually does not perform a rotation but is simply copying the
values from the src tensor to the dst tensor.

In "standard" RoPE take the embedding input dimensions in pairs when we process
them. So we will take the 128 dimensions and simple process them as pairs in
order, (0,1), (2,3), (4,5), (6,7), ... (126,127).
But in the case of NeoX we will also create pairs but diffrently:
```
n = n_dim

0-n/2    =  (0,64)
1-n/2+1  =  (1,65)
2-n/2+2  =  (2,66)
3-n/2+3  =  (3,67)
...
n/2-1    =  (63,127)
```
The NeoX approach can potentially offer better performance in some hardware
architectures due to its memory access pattern.

__wip__



```c
static void rope_yarn(
    float theta_extrap,
    float freq_scale,
    float corr_dims[2],
    int64_t i0,
    float ext_factor,
    float mscale,             // attn_factor
    float * cos_theta,
    float * sin_theta) {

    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}
```
`theta_interp` corresponds to θ_d/s in the paper and `theta_extrap` corresponds to θ_d

A standalone example can be found in [rope.c](../fundamentals/ggml/src/rope.c).
