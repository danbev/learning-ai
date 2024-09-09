## Receptance Weight Key Value (RWKV): Reinventing RNNs for the Transformer era
Simliar to [Mamba](./mamba.md) this model architecture avoids the transformers
O(N²) complexity by using an RNN (or an SSM).

So with this, like Mamba, we have efficient training and also efficient
inference (which scales linearly with the length of the sequence and not
exponentially like it does with transformers). And there are no approximation
techniques like in the Performer/Reformer/Linear Transformer (perhaps others)
either instead this is exact.

It is possible to [decompose attention](../transformer.md#decomposing-as-vector-operations)
which is a way to work around the memory limits of having the Q, K, and V
matrices stored in memory, though it might mean more computation time as we
have to compute then sequentially and not in parallel.

There are currently 6 versions of the RWKV model and this document will focus
on v5 (Eagle) and v6 (Finch) which are the latest versions as of this writing
and also the version that are implemented in llama.cpp.

_I initially started this document reading the RWKV-4 paper which is why at the
moment there are still notes that are based on that paper. This will be updated
as I go through the new paper_.

Lets take a look at inference in this architecture:
```
input sequence   = ["Dan"        "loves"      "ice"         "cream"    ]
input tokens     = [ 223         45,          1212            67       ]
input embeddings = [ [0.5, -03], [0.7, 0.2], [-0.1, 0.8], [ 0.3, -0.5] ]

           +----------------------+
           |   input_embeddings   |
           +----------------------+
                      ↓
           +----------------------+
           |       LayerNorm      |
           +----------------------+
                      |
                      +-----------------------------+
Time mixing           ↓                             |
           +----------------------+                 |   
           |       LayerNorm      |                 |
           +----------------------+                 |
                      ↓                             |
           +--------------------------------------+ |
           |               μ                      | |
           | G = (μ_g ⊙ x_t + (1 - μ_g) ⊙ x_t-1)Wg| |
           | R = (μ_r ⊙ x_t + (1 - μ_r) ⊙ x_t-1)Wr| |
           | K = (μ_k ⊙ x_t + (1 - μ_k) ⊙ x_t-1)Wk| |
           | V = (μ_v ⊙ x_t + (1 - μ_v) ⊙ x_t-1)Wv| |
           +--------------------------------------+ |
              |      |    |        |        |       |
           +-----+ +----+ +--+  +-----+  +-----+    |       
           |  G  | | R  | |w |  |  K  |  |  V  |    |
           +-----+ +----+ +--+  +-----+  +-----+    |
              |       |    |      |        |        |  
            +-------+ |    | +------------------+   |
            | SiLU  | |    +-|   WKV Operator   |   |
            +-------+ |      +------------------+   |
               |      |               |             |
               |      +------------->(*)            |
               |                      |             |
               |           +----------------------+ |   
               |           |       LayerNorm      | |
               |           +----------------------+ |
               |                      |             |
               |                      |             |
               |          +-----+     |             |
               +----------| (*) |-----+             |
                          +-----+                   |
                             |                      |
                             ↓                      |
                          +-----+                   |
                          | out |                   |
                          +-----+                   |
                             |                      |
                             ↓                      |
                            (+)---------------------+
                             |
                             +----------------------+
Channel mixing               |                      |
                 +----------------------+           |
                 |       LayerNorm      |           |
                 +----------------------+           |
                             |                      |
           +---------------------------------------+|
           |               μ'                      ||
           | R'= (μ_r ⊙ x_t + (1 - μ_r) ⊙ x_t-1)Wr'||
           | K'= (μ_k ⊙ x_t + (1 - μ_k) ⊙ x_t-1)Wk'||
           +---------------------------------------+|
                 |                   |              |
            +------+            +-------------+     |
            |  R'   |           |    K'       |     |
            +------+            +-------------+     |
                 |                   |              |
            +-------+           +-------------+     |
            |Sigmoid|           | ReLU^2      |     |
            +-------+           +-------------+     |
                 |                   |              |
                 |     +------+      |              |
                 +-----| (*)  |------+              |
                       +------+                     |
                          |                         |
                         (+)------------------------+
                          |
                          ↓
                     (next layer)

(*) = Element wise multiplication (Hadamard product)
``` 

### Linear Interpolation (lerp) in Eagle (RWKV-5)
This is pretty much the same as in RWKV-4 so I've kept my notes from that
and I'll follow up with the notation used in the RWKV-5 paper after this.

This is what the `μ` box in the diagram above is doing.

Linear interpolation recap:
```
result = t * a + (1 - t) * b

t     = current token
1 - t = previous token
a     = current token decay factor
b     = previous token decay factor

Examples:
a = 1, b = 0
result = t * 1 + (1 - t) * 0
result = t 
This means that the current token is used and the previous token is ignored.

a = 0, b = 1
result = t * 0 + (1 - t) * 1
result = (1 - t)
This means that the current token is ignored and the previous token is used.

And any value in between will be a linear interpolation between the two.
```

In the paper this is defined as (showing only one component R but the same
is done for G, K, and V as well):
```
R = μ_r ⊙ x_t + (1 - μ_r) ⊙ x_t-1

R = [0.2, 0.9] ⊙ [0.5, -0.3] + (1 - [0.2, 0.9]) ⊙ [0.0, 0.0]
R = [0.1, -0.27] + (1 - [0.2, 0.9] ⊙ [0.0, 0.0]
R = [0.1, -0.27] + [0.8, 0.1] ⊙ [0.0, 0.0]
R = [0.1, -0.27] + 0
R = [0.1, -0.27]

μ_r   = learnable parameter vector.
x_t   = current token embedding.
x_t-1 = previous token embedding.
⊙     = element-wise multiplication (Hadamard product).
```

```
input embeddings = [ [0.5, -03], [0.7, 0.2], [-0.1, 0.8], [ 0.3, -0.5] ]

x_t   = [0.5, -0.3]
x_t-1 = [ 0.0, 0.0] (first token so there is no previous token)
μ_r   = [0.2, 0.9]


The following is doing a linear interpolation between the current token and the
previous token using the μ_r vector:
R = μ_r ⊙ x_t + (1 - μ_r) ⊙ x_t-1

R = [0.2, 0.9] ⊙ [0.5, -0.3] + (1 - [0.2, 0.9]) ⊙ [0.0, 0.0]
R = [0.1, -0.27] + (1 - [0.2, 0.9] ⊙ [0.0, 0.0]
R = [0.1, -0.27] + [0.8, 0.1] ⊙ [0.0, 0.0]
R = [0.1, -0.27] + 0
R = [0.1, -0.27]

```
The same operations are done for the K and V vectors but they have their own
`μ_k` an `μ_v` vectors.
If a value in these mu vectors is 1 then the current value of the token
embeddings is used. And if it is 0 then the previous value of the token
embedding would be used. And any value in between would be a linear
interpolation between the two.

In the RWKV-5 paper the notation is a little different:
```
lerp_ם(a, b) = a + (b -a ) ⊙  μ_ם

a = x_t-1
b = x_t

lerp_ם(x_t-1, x_t) = x_t-1 + (x_t - xt_-1) ⊙  μ_ם
                     x_t-1 + (x_t ⊙ µ□) - (x_t-1 ⊙ µ□)
                     (x_t ⊙ µ□) + (x_t-1 ⊙ (1 - µ□))
                     µ□ ⊙ x_t + (1 - µ□) ⊙ x_t-1

ם = one of g, r, k, or v
```
But it is really the same thing, just different notation. Also note that there
is an matrix multiplication in the first equation which is specific for each
μ vector (this can be seen in the diagram above).

### Data-Dependent Linear Interpolation (ddlerp) in Finch (RWKV-6)
Now this is a new concept in the RWKV-6 paper and it is used to calculate the
receptance, key, and value vectors.
There is a function named LoRA (Low Rank Adaptation) which I've written about in
[lora.md](../lora.md) and I was somewhat confused about its usage here. My
understanding was that LoRA was used to reduce the dimensionality of matrices
and I did not see how that would be applicable here. In the RWKV-6 paper it is
not used for parameter reduction but instead for data-dependent linear shift
mechanism. Is is called LoRA because of the similar structure of the LoRA update
function:

```
lora□(x) = λ□  + tanh(xA□) B□

lora(x) = λr + tanh(xAr) Br

λr is a learnable vector
Ar and Br are small learnable matrices
x is the input

lora□(x) = λ□  + tanh(xA□) B□
ddlerp□(a,b) = a + (b − a) ⊙ lora□(a +(b − a) ⊙ µx)
```
Note that there is a µx vector which is trained and this is used for g, r, k,
and v. And notice that the vectors and matrices in the lora function are
specific to the current component (g, r, k, v).

For example:
```
input sequence   = ["Dan"        "loves"      "ice"         "cream"    ]
input tokens     = [ 223         45,          1212            67       ]
input embeddings = [ [0.5, -0.3], [0.7, 0.2], [-0.1, 0.8], [ 0.3, -0.5] ]

Let's assume:
D = 2 (embedding dimension)
μr = [0.6, 0.7]
λr = [0.1, 0.2]
Ar = [[0.1, 0.2], [0.3, 0.4]]
Br = [[0.5, 0.6], [0.7, 0.8]]

For the second token "loves":

a = x_t   = [0.7, 0.2]   (current embedding for "loves")
b = x_t-1 = [0.5, -0.3]  (previous embedding for "Dan")

Step 1: Calculate (b - a) ⊙ μr
[0.5, -0.3] - [0.7, 0.2] = [-0.2, -0.5]
[-0.2, -0.5] ⊙ [0.6, 0.7] = [-0.12, -0.35]

Step 2: Add this to a
[0.7, 0.2] + [-0.12, -0.35] = [0.58, -0.15]

Step 3: Apply LoRA function
x = [0.58, -0.15]
tanh(xAr) = tanh([0.58*0.1 + (-0.15)*0.3, 0.58*0.2 + (-0.15)*0.4])
           = tanh([0.013, 0.056])
           ≈ [0.013, 0.056]

tanh(xAr)Br = [0.013*0.5 + 0.056*0.7, 0.013*0.6 + 0.056*0.8]
             = [0.0455, 0.0526]

lora(x) = [0.1, 0.2] + [0.0455, 0.0526] = [0.1455, 0.2526]

Step 4: Final ddlerp calculation
ddlerpr(a, b) = [0.7, 0.2] + ([-0.2, -0.5] ⊙ [0.1455, 0.2526])
              = [0.7, 0.2] + [-0.0291, -0.1263]
              = [0.6709, 0.0737]
```

### Eagle (RWKV-5) Time mixing
The forumla given in the paper looks like this:
```
□t = lerp□(xt ,xt−1) W□, 

□  = ∈ {r ,k, v, g }


Example:
r_t = lerp_r(xt ,xt−1) W_r 

This is represented by the μ "box" in the diagram above.

w = exp(−exp(ω))
This is represented by the w "box" in the diagram above.

                                 t-1
wkv_t = diag(u) * K_t^T * v_t +   Σ  diag(w)^t-1-i * K_i^T * v_i 
                                 i=1

```
The diag function is creating a diagonal matrix from a vector. And the u vector
is a learned parameter that is part of the Weighted Key Value (WKV) computation.
The "time-first" u is initialized to
```
r0(1 − i/(D−1)) + 0.1((i + 1) mod 3)
```
This is represented by the WKV Operator "box" in the diagram above.
So, `diag(u)` will create a matrix where the diagonal is the u vector and the
rest of the matrix is zeros. This matrix will be multipled by the transpose of
the K matrix which contains key values after the linear interpolation. And
that will then be multiplied by the V matrix which contains the values after
the linear interpolation also.

To this we add the sum all the past tokens up to the current token but not
includig the current token (t-1):
```
 t-1
  Σ  diag(w)^t-1-i * K_i^T * v_i 
 i=1
```
`K_i^T * v_i` is the key and value product for that token.
`diag(w)^t-1-i` is the decay factor applied to that token. Notice that tokens in
the past will have a larger exponent value which will make the decay factor
resulting in more decay for those tokens.

Alright lets break this down a little so let walk through this with an example.
```
Embedding dim  : 4
Attention heads: 2 (each head will deal with 4/2 = 2 dimensions)

Left hand term: diag(u) * K_t^T * v_t
u       = [0.9 0.7]
diag(u) = [ 0.9 0.0]
          [ 0.0 0.7]

K_t     = [0.3, 0.5]  (row vector)
K_t^T   = [0.3]
          [0.5]

v_t     = [0.2 0.4]   (row vector)

So lets start with computing K_t^T * v_t:
   [0.3] [0.2 0.4] = [0.06 0.12]
   [0.5]             [ 0.1  0.2] 


The we multiply this with diag(u):
   [ 0.9 0.0] [0.06 0.12] = [0.054 0.108]
   [ 0.0 0.7] [ 0.1  0.2]   [ 0.07  0.14]


w       = [0.8, 0.6] (decay factor)
k_t-1   = [0.4, 0.2] (key vector from previous step (t-1))
k_t-1^T = [0.4]
          [0.2]
v_t-1   = [0.1, 0.3]  (value vector from previous time step (t-1))

diag(w) = [0.8 0.0]
          [0.0 0.6]

We start with computing K_t-1^T * v_t-1:
[0.4] [0.1 0.3] = [0.04 0.12]
[0.2]             [0.02 0.06]

Then we multiply this with diag(w):
[0.8 0.0] [0.04 0.12] = [0.032 0.096]
[0.0 0.6] [0.02 0.06]   [0.012 0.036]

So both the left and right hand produce 2x2 vectors which are then added:

   [0.054 0.108]  + [0.032 0.096] = [0.086 0.204]
   [ 0.07  0.14]    [0.012 0.036]   [0.082 0.176]
```

The output will be the result of all the heads concatenated together and then
multipled by `W_o`.
```
o_t = concat( SiLU(g_t) ⊙ LayerNorm(r_t * wkv_t)) W_o
```
Now the result from wkv operation above (`wkv_t`) will be multiplied by the
retention vector (`r_t`) which is calculated in the previous step. This is a
learned parameter and is used to control how much of the information is retained
from the previous time step. This is then passed through the LayerNorm function
and then multipled, element wise, by the result of the SiLU activation function
applied to the `g_t` vector. And like we mentioned above this done for all heads
which are then concatenated together and multipled by `W_o`.

### Eagle (RWKV-5) Channel mixing)
Like the time mixing this also starts with a LayerNorm operation and then the
mu (μ) operations similar to the time mixing:
```
G'= (μ_g ⊙ x_t + (1 - μ_g) ⊙ x_t-1)Wg'
R'= (μ_r ⊙ x_t + (1 - μ_r) ⊙ x_t-1)Wr'
K'= (μ_k ⊙ x_t + (1 - μ_k) ⊙ x_t-1)Wk'
V'= (μ_v ⊙ x_t + (1 - μ_v) ⊙ x_t-1)Wv'
```
G' is passed to through the sigmoid activation function and R', K', and V' are
passed to the MLP function. This is a feed-forward neurual network but a little
different than the ones I've seen before. The ones I've seen usually have an
expension operation, followed by a non-linear function, and then a contraction
operation. In RWKV this is a little different:
```
r_t'  = lerp_r(x_t', x_t_' -1 ) W_r'   
k_t'  = lerp_k(x_t', x_t_' -1 ) W_k'
v_t'  = ReLU(k')^2 W_v'
o_t'  = σ(r_t') ⊙ v_t')
```
ReLU is max(0, x) and squared ReLU is max(0, x)^2. 
My understanding is that the dimension of k' is the same as the embedding
dimension divded by the number of heads.
Now, what we are doing here is that we are taking the current tokens keys and
applying a linear interpolation between the current and previous token producing
k'. These are then passed through ReLU^2 and the result of this is then
multiplied by the `W_v'` matrix to produce v'.

The last operations in the channel mixing are element-wise multiplication between
the sigmoid output of G' and the output of the MLP function (v'):
```
o_t'  = σ(r_t') ⊙ v_t')
```
And the output of this is then passed to the next layer.

I'm confused about the names of these, time mixing, and channel mixing. It seems
to me that channel mixing, or dimension mixing, is done in both of these.
_Actually this is not so different from the transformer where we have the
self attention which attends to all the tokens in the sequence (time mixing) and
then we have the feed-forward neural network which is applied to each token
independently (channel mixing)_.

### Finch (RWKV-6) Channel mixing)
This is pretty much the same as in Eagle (RWKV-5) but I'm still keeping this
section. When reading the paper I was a little confused because I was using
the Figure 1 from the paper as a reference for the diagram above. But when going
throught the llama.cpp implementation I noticed that the G' matrix is not
present. It seems like this might be a mistake in the paper, so I'll update
the digram to reflect this. It is stated [here](https://www.rwkv.com/) that there
was a mistake in the paper so the below is correct.
```
r_t'  = lerp_r(x_t', x_t_' -1 ) W_r'   
k_t'  = lerp_k(x_t', x_t_' -1 ) W_k'
v_t'  = ReLU(k')^2 W_v'
o_t'  = σ(r_t') ⊙ v_t')
```

### llama.cpp RWKV implementation
Now, lets take a look at the implementation of the RWKV 6 model in llama.cpp.   

The following example can be used start debugging and stepping through the
implementation. This requires a converted model which can be done using the
following:
```console
$ cd fundamentals/llama.cpp
$ make checkout-rwkv-model
$ make convert-rwkv-model
```

And then we can build and start a debugging session using:
```console
$ cd fundamentals/llama.cpp
$ make simple-prompt-multi
$ make debug-rwkv-simple-prompt-multi
```

Now, lets stick a break point in:
```console
(gdb) br build_rwkv6()
Breakpoint 2 at 0x5555556dc649: file src/llama.cpp, line 15073.
```
```c++
    ggml_cgraph * build_rwkv6() {
        ggml_cgraph *gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

        const int64_t n_seqs = batch.n_seqs;
        const int64_t n_seq_tokens = batch.n_seq_tokens;
        const int64_t n_tokens = batch.n_tokens;
```
One thing to note is that a computation graph is also built as part of the
`llama_new_context_with_model` so just `(gdb) continue` after the first
breakpoint so that we are in the function that we are interested in with the
correct batch as well.

Lets inspect these values to get an idea of the size of the input:
```console
(gdb) p n_seqs
$21 = 2

(gdb) p n_seq_tokens 
$22 = 4

(gdb) p n_tokens
$23 = 8

(gdb) p *this
$9 = {model = @0x555555ac1e70, lctx = @0x555555cd0470, hparams = @0x555555ac1ea0, cparams = @0x555555cd0478,
  batch = @0x7fffffffd470, kv_self = @0x555555cd1960, n_embd = 2048, n_layer = 24, n_rot = 0, n_ctx = 1024, n_head = 0,
  n_head_kv = 0, n_embd_head_k = 0, n_embd_k_gqa = 0, n_embd_head_v = 0, n_embd_v_gqa = 0, n_expert = 0, n_expert_used = 0,
  freq_base = 10000, freq_scale = 1, ext_factor = 0, attn_factor = 1, beta_fast = 32, beta_slow = 1, norm_eps = 9.99999975e-06,
  norm_rms_eps = 0, n_tokens = 8, n_kv = 2, n_outputs = 0, n_outputs_enc = 0, kv_head = 0, n_ctx_orig = 1048576, flash_attn = false,
  pooling_type = LLAMA_POOLING_TYPE_NONE, rope_type = LLAMA_ROPE_TYPE_NONE
```

Following that we have:
```c++
        struct ggml_tensor * state_copy = build_inp_s_copy();
```
And `build_inp_s_copy` is defined as:
```c++
    struct ggml_tensor * build_inp_s_copy() {
        lctx.inp_s_copy = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_kv);
        cb(lctx.inp_s_copy, "inp_s_copy", -1);
        ggml_set_input(lctx.inp_s_copy);
        return lctx.inp_s_copy;
    }
```
And we can see that `n_kv` is 2. So this is creating a new 1 dimensional tensor
which can hold two element of type `GGML_TYPE_I32`.  I'll get back to this once
we see how it is used later but this tensor will be used to store information
about which token states should be copied (used) for the current
processing/decoding.

Then we have:
```c++
        struct ggml_tensor * state_mask = build_inp_s_mask();
```
And `inp_s_mask` is created in the same way as `inp_s_copy` we saw above but
with an important difference in that this is a 2d tensor of type
`GGML_TYPE_F32` as opposed to the 1d tensor of type `GGML_TYPE_I32`:
```c++
    struct ggml_tensor * build_inp_s_mask() {
        lctx.inp_s_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_kv);
        cb(lctx.inp_s_mask, "inp_s_mask", -1);
        ggml_set_input(lctx.inp_s_mask);
        return lctx.inp_s_mask;
    }
```
Notice that this is creating a 2d tensor with 1 column and the number of
sequences as the number of rows. Notice again that this is a F32 tensor and
it has multiple rows (one for each sequence).

Next we build up the input embedding tensor:
```
        inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
```

```c++
static struct ggml_tensor * llm_build_inp_embd(
        struct ggml_context * ctx,
       struct llama_context & lctx,
        const llama_hparams & hparams,
         const llama_ubatch & batch,
         struct ggml_tensor * tok_embd,
         const llm_build_cb & cb) {
    const int64_t n_embd = hparams.n_embd;

    struct ggml_tensor * inpL;

    if (batch.token) {
        lctx.inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.n_tokens);
        cb(lctx.inp_tokens, "inp_tokens", -1);
        ggml_set_input(lctx.inp_tokens);

        inpL = ggml_get_rows(ctx, tok_embd, lctx.inp_tokens);
    } else {
       lctx.inp_embd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, batch.n_tokens);
        inpL = lctx.inp_embd;
        ggml_set_input(lctx.inp_embd);
    }

    cb(inpL, "inp_embd", -1);

    return inpL;
}
```
In this case we have batch.tokens so the above will create a 1d tensor with the
with a size of 8. Note that this will be assigned to `lctx.inp_tokens`.
Then `inpL` will get set to `ggml_get_rows` which is an operation that will
extract the _rows_ from `tok_embed` that are specified by the `lctx.inp_tokens`.
So at inference time the number of tokens in the `lctx.inp_tokens` that are
populatetd will be returned from this operation. But keep in mind that at this
stage we are only building up the computation graph and not actually running
any operations yet.

The returned `inpL` is then passed into `llm_build_norm`:
```c++
        inpL = llm_build_norm(ctx0, inpL, hparams, model.tok_norm, model.tok_norm_b, LLM_NORM, cb, -1);
```
So is this the first LayerNorm after the embeddings layer?

Next we are going to iterate over all the layers in the model which is 24 for
this model:
```c++
        for (int il = 0; il < n_layer; ++il) {
            const llama_layer * layer = &model.layers[il];

```

And below we can see the usage of both `state_copy` and `state_mask`:
```c++
            struct ggml_tensor * token_shift = llm_build_copy_mask_state(ctx0,
                    gf, kv_self.k_l[il], state_copy, state_mask,
                    hparams.n_embd_k_s(), kv_self.size, kv_head, n_kv, n_seqs);
```
`kv_self.k_l` is a vector of tensors in `kv_self`:
```++
std::vector<struct ggml_tensor *> k_l; // per layer
```
```console
(gdb) p kv_self.k_l.size()
$14 = 24
```

Each of these tensors (one for each layer) will contain two pieces of
information:
```
0                          2047                                 4095
[   att_shift                  |          ffn_shift                ]
   "Time mixing"                        "Channel mixing"
```
So the `token_shift` tensor contains two pieces of information. The first is the
`att_shift`. The second is the `ff_shift`.

* `att_shift` is the last tokens normalized output from the time mixing (from
   the previous layer).

* `ff_shift` is the last tokens normalized output from the channel mixing
   (from the previous layer).

```c++
static struct ggml_tensor * llm_build_copy_mask_state(
        struct ggml_context * ctx,
         struct ggml_cgraph * graph,
         struct ggml_tensor * s,             // kv_self.k_l[il]
         struct ggml_tensor * state_copy,    // inp_s_copy
         struct ggml_tensor * state_mask,    // inp_s_mask
                    int32_t   n_state,       // embd * 2 (2048 * 2 = 4096)
                    int32_t   kv_size,
                    int32_t   kv_head,
                    int32_t   n_kv,
                    int32_t   n_seqs) {
    struct ggml_tensor * states = ggml_reshape_2d(ctx, s, n_state, kv_size);
```
So `s` is `kv_self.k_l[il]` which is the tensor for the currently processed
layer and this is a 1d tensor. And `n_state` is the number of states 4096 (note
that this is 2048 for the result of time mixing process and 2048 for the channel
mixing process). The `kv_size` is 2 and `n_kv` is 2 depending on the number of
sequences. In this case `n_seqs` is also 2 (again this would be different for
a single sequence or multiple).

```c++
    struct ggml_tensor * states = ggml_reshape_2d(ctx, s, n_state, kv_size);
```
So we are reshaping the 1d tensor `s` (which is `kv_self.k_l[il]`) from:
```console
(gdb) p *s
$16 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x555555cd5000,
ne = {8192, 1, 1, 1}, nb = {4, 32768, 32768, 32768},
op = GGML_OP_NONE, op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0, src = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 
0x0, 0x0, 0x0}, view_src = 0x0, view_offs = 0, data = 0x7fff34e00020,
name = "cache_k_l0", '\000' <repeats 53 times>, extra = 0x0}
```
Into a 2d tensor. Since we have two sequences in this case `kv_size` will be 2,
and `n_state` is
4096. So this will create a new tensor 4096x2 (4096 columns and 2 rows).
```console
(gdb) p *states
$118 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0,
ne = {4096, 2, 1, 1}, nb = {4, 16384, 32768, 32768},
op = GGML_OP_RESHAPE, op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0, src = {0x555556e5f530, 0x0, 0x0, 0x0, 0x0, 0x0,
0x0, 0x0, 0x0, 0x0}, view_src = 0x555556e5f530, view_offs = 0, data = 0x7fff34e00020,
name = "cache_k_l0 (reshaped)", '\000' <repeats 42 times>, extra = 0x0}
```
We can visualize the reshaped states tensor as:
```
states:

        n_state = 4096
 0 [0....................4095]     kv_size = 2
 1 [0....................4095]     
```

Next we have the following which is getting rows from the states tensor where
`state_copy` is specifying which rows to get:
```c++
    states = ggml_get_rows(ctx, states, state_copy);
```
Recall that state copy is a copy of the `llama_context.inp_s_copy` tensor. And
this tensor holds information about which states should be used by the current
processing/decoding. Now I need to remind myself that what is happening here
is just building up the computation graph and not actually running any
computation so this tensor does not contains any actual values at this point.

It currently looks like the following and if we recall from about it is a 1d
tensor with with an number of elements equal to the sequence length (2 in this
case):
```console
(gdb) p *state_copy
$22 = {type = GGML_TYPE_I32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0,
ne = {2, 1, 1, 1}, nb = {4, 8, 8, 8}, op = GGML_OP_NONE,
op_params = {0 <repeats 16 times>}, flags = 1, grad = 0x0, src = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
view_src = 0x0, view_offs = 0, data = 0x0, name = "inp_s_copy", '\000' <repeats 53 times>, extra = 0x0}
```
Now, what this is doing is that `copy_state` will contain the indices of the
rows from the states tensor that should be used for the current processing.
One thing to keep in mind is that this is copying/moving rows and the shape of
states remains the same. The states that are specified in the `state_copy`
tensor will be copied to the top of the states tensor. 

To see how this works we need to take a look at the `llama_set_inputs` function
which is called after the `build_rwkv6` function but before `llama_decode`.

In `llama_set_input`s we have the following:
```cpp
    if (kv_self.recurrent) {
        const int64_t n_kv = kv_self.n;
        ...

        if (lctx.inp_s_copy) {
            GGML_ASSERT(ggml_backend_buffer_is_host(lctx.inp_s_copy->buffer));

            int32_t * data = (int32_t *) lctx.inp_s_copy->data;

            // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
            for (uint32_t i = 0; i < n_kv; ++i) {
                const uint32_t  cell_id = i + kv_self.head;
                llama_kv_cell & kv_cell = lctx.kv_self.cells[cell_id];

                // prevent out-of-bound sources
                if (kv_cell.src < 0 || (uint32_t) kv_cell.src >= kv_self.size) {
                    kv_cell.src = cell_id;
                }

                data[i] = kv_cell.src;

                // ensure copy only happens once
                if (kv_cell.src != (int32_t) cell_id) {
                    kv_cell.src = cell_id;
                }
            }
        }
```
Notice that the above will get a pointer to the data of the `inp_s_copy` tensor
and also notice that the type int32. Then we will iterate of the number or
key-value entries which is 2 in our case. The first `cell_id` will be 0.
```console
(gdb) p cell_id
$31 = 0
(gdb) p kv_self.cells[0]
(gdb) p kv_cell
$41 = (llama_kv_cell &) @0x555555cd6ed0: {pos = 3, delta = 0, src = 0, tail = 0, seq_id = std::set with 1 element = {[0] = 0}}
```
So we are going to set the first element in `inp_s_copy` to 0 to the value
of the `kv_cell.src` which is 0. This means that later in the `ggml_get_rows`
operation the 0/first row of the `states` tensor will be copied.

Notice that the `src` field is used with recurrent models like RWKV-6:
```c++
struct llama_kv_cell {
    llama_pos pos   = -1;
    llama_pos delta = 0;
    int32_t   src   = -1; // used by recurrent state models to copy states
```
A `src` value of less than 0 means that this cell is new or does not have any
valid state information. If it is greater then 0 then it indicates that this
cell has valid, perviously computed state information.

TODO: I'm not sure about what the last if statement is doing yet.

In our case `n_kv` is 2 so we will go through this loop once more, this time
the `cell_id` will be set to 1.
```console
(gdb) p kv_cell
$47 = (llama_kv_cell &) @0x555555cd6f10: {pos = 7, delta = 0, src = 1, tail = 1, seq_id = std::set with 1 element = {[0] = 1}}
```
Notice that the value of `src` i 1 for this entry so the second element in
`inp_s_copy` will be set to 1. This means that the second row of the `states`
tensor will be copied.
Now, that still leaves one question which is now did the `kv_cell.src` get set
to their respective values?  
I'll return to this question shortly but need to go through some other things
before that namely the `state_mask`.

Next, back in `llm_build_copy_mask_state` we have:
```c++
    // clear states of sequences which are starting at the beginning of this batch
    // FIXME: zero-out NANs?
    states = ggml_mul(ctx, states, state_mask);
```
So we are multiplying the `states` tensor (4096x2) with the `state_mask` (1x2)
tensor. Simliar to the `inp_s_copy` tensor we are only building the computation
graph at this point so the `state_mask` tensor does not contain any actual
values at this point. `inp_s_mask` is also populated in the `llama_set_inputs`:
```c++
        if (lctx.inp_s_mask) {
            GGML_ASSERT(ggml_backend_buffer_is_host(lctx.inp_s_mask->buffer));
            float * data = (float *) lctx.inp_s_mask->data;

            // clear unused states
            for (int i = 0; i < n_kv; ++i) {
                uint32_t        cell_id = i + kv_self.head;
                llama_kv_cell & kv_cell = lctx.kv_self.cells[cell_id];

                data[i] = (float) (kv_cell.src >= 0);

                // only clear once
                if (kv_cell.src < 0) {
                    kv_cell.src = cell_id;
                }
            }
        }
```
TODO: I'm not sure why `cell_id` is not const like it is in the `inp_s_copy` case.

Recall that this tensor if of type F32 and we are getting a pointer to this
tensors data. And we are going to iterate over all the key-value entries which
is 2 in our case. The first `cell_id` will be 0 because `kv_self.head` is 0.:
```console
(gdb) p cell_id
$59 = 0

(gdb) p kv_cell
$60 = (llama_kv_cell &) @0x555555cd6ed0:
{pos = 3, delta = 0, src = -1, tail = 0, seq_id = std::set with 1 element = {[0] = 0}}
```
Now, notice that `src` field is actually `-1` now which was not the case for
the `inp_s_copy` block. The `inp_s_mask` block actually comes before
`inp_s_copy` but I wanted to follow the order in the computation building
functions. We can also see that the sequence id is 0.

If the `src` field contains a valid previously computed state then we set that
entry in `inp_s_mask` to 1, and if it does not then we set it to 0:
```c++
                data[i] = (float) (kv_cell.src >= 0);
```
In this case `data[0] = 0` because `kv_cell.src` is `-1`.

And the last if statement will update `kv_cell.src` to 0 since it is currently
-1 (less than 0).
The second time through the loop `cell_id` will be 1 and `kv_cell` will look
like this:
```console
(gdb) p kv_cell
$70 = (llama_kv_cell &) @0x555555cd6f10:
{pos = 7, delta = 0, src = -1, tail = 1, seq_id = std::set with 1 element = {[0] = 1}}
```
The same thing will happen here and we are setting `data[1] = 0` because
`kv_cell.src` is `-1`. And the last if statement will update `kv_cell.src` to 1.

So after this loop has completed the state of `inp_s_mask` will look like this:
```console
(gdb) p data[0]
$75 = 0
(gdb) p data[1]
$76 = 0
```
So if we "jump" back to the `llm_build_copy_mask_state` function and look at the
following line again:
```c++
    states = ggml_mul(ctx, states, state_mask);
```
So we are multiplying the `states` tensor (4096x2) with the `state_mask` (1x2)
tensor. And we know that `state_mask` has two elements which are both 0.
```
0 [0....................4095]  [0]
1 [0....................4095]  [0]   
```
So this is in fact clearning out the states. But we need to keep in mind that
this is the first time we process the tokens so there are no prior states, so
clearing seems reasonable.

Now, for future decoding these previous states might be populated and used in
which case the mask would clear the states that are not required for this
current processing.

Following that we have:
```c++
    // copy states which won't be changed further (between n_seqs and n_rs)
    ggml_build_forward_expand(graph,
        ggml_cpy(ctx,
            ggml_view_1d(ctx, states, n_state*(n_kv - n_seqs),            n_seqs *n_state*ggml_element_size(states)),
            ggml_view_1d(ctx, s     , n_state*(n_kv - n_seqs), (kv_head + n_seqs)*n_state*ggml_element_size(s))));
```
So this is using `n_state` which is 2048, `n_kv`, and `n_seqs=2`.
Now, I'd like to detour to see how `n_kv` is actually set. For this we have to
look back into `llm_build_graph`:
```c++
    struct llm_build_context llm(lctx, batch, cb, worst_case);

    const llm_build_cb & cb,
                  bool   worst_case) :
        ...
        n_kv             (worst_case ? kv_self.size : kv_self.n),
        ...
```
Now, if `ctx_params.n_seq_max = 6` and `worst_case=true` then we would have
`n_seqs = 2` and `n_kv = 6`. This would give us the following argument values
for the source and destination views in the above copy:
```
ne0 = n_state * (n_kv - n_seqs)
ne0 =   4096  *    6  - 2
ne0 =   4096  *    4
ne0 =   4096  *    4
ne0 =   16384 

offset = n_seqs * n_state * ggml_element_size(states)
offset =   2    *  4096   *  4
offset =   2    *  4096   *  4
offset =   32768

src: ggml_view_1d(ctx, states, 16384, 32768)
```
And for the destination view we have the same number of elements:
```
ne0 = n_state * (n_kv - n_seqs)
ne0 = 4096    *    6  -  2 
ne0 = 4096    *    4
ne0 = 16384

offset = (kv_head + n_seqs) * n_state * ggml_element_size(s)
       =     0    +  2      *   4096  *    4
       =          2         *   4096  *    4
       = 32768

src : ggml_view_1d(ctx, states, 16384, 32768)
dest: ggml_view_1d(ctx, s     , 16384, 32768)
```
Now, recall that states tensor has been rearranged by `ggml_get_rows` and this
is copying 16384 elements, 4 token states of 4096 each, starting at
offset 32768 which is (2 * 4096 * 4), so we are skipping the first two token
states which is because the current batch has two sequences:
```console
(gdb) up
#1  0x00005555556dcd95 in llm_build_context::build_rwkv6 (this=0x7fffffffd200) at src/llama.cpp:15098
15098	            struct ggml_tensor * token_shift = llm_build_copy_mask_state(ctx0,
(gdb) p batch
$23 = (const llama_ubatch &) @0x7fffffffd470: {equal_seqs = true, n_tokens = 8, n_seq_tokens = 4, n_seqs = 2,
  token = 0x555557483270, embd = 0x0, pos = 0x555555be05a0, n_seq_id = 0x555557483660, seq_id = 0x555556e5c510,
  output = 0x555556e5c5d0 ""}
```
So this is saying copy 16384 elements but skip the token states (2) that are
going take part in the current/upcoming decode processing. And copy these
elements to the original `s` tensor which recall is `kv_self.k_l[il]` also
skipping the first two tensor states. So we will 

Again remember that this is only building up the computation graph and
`kv_self.k_l` might get updates elsewhere and this copying would ensure that
all the token states that are not used for the next processing are preserved.

Next we are creating a 2d view of the states tensor:
```
    // the part of the states that will be used and modified
    return ggml_view_2d(ctx, states, n_state, n_seqs, states->nb[1], 0);
    return ggml_view_2d(ctx, states, 4096, 2, 16384, 0);
```

`kv_self.k_l[i]` is a way of passing information from one layer to the next, and
is not so different from what the KV cache does if we think about it. The KV
cache stores calculations from the previous layer and then makes them available
to the following layers. Here are enabling the state to be passed to the next
layer.

After `llm_build_copy_mask_state` the  `token_shift` tensor will contain the
attention shift and feed-forward shift from the previous layer that correspond
to the sequence/sequences that are currently being processed.

So that was the `token_shift` tensor. Next we have the `wkv_states` tensor:
```c++
            struct ggml_tensor * wkv_states = llm_build_copy_mask_state(ctx0,
                    gf, kv_self.v_l[il], state_copy, state_mask,
                    hparams.n_embd_v_s(), kv_self.size, kv_head, n_kv, n_seqs);

```
This using the same function as above but with different tensors `s` which is
now `kv_self.v_l[0]` and `n_state` is now `hparams.n_embd_v_s()`:
```console
            struct ggml_tensor * wkv_states = llm_build_copy_mask_state(ctx0,
                    gf, kv_self.v_l[il], state_copy, state_mask,
                    hparams.n_embd_v_s(), kv_self.size, kv_head, n_kv, n_seqs);
```
```c++
static struct ggml_tensor * llm_build_copy_mask_state(
        struct ggml_context * ctx,
         struct ggml_cgraph * graph,
         struct ggml_tensor * s,
         struct ggml_tensor * state_copy,
         struct ggml_tensor * state_mask,
                    int32_t   n_state,
                    int32_t   kv_size,
                    int32_t   kv_head,
                    int32_t   n_kv,
                    int32_t   n_seqs) {
    struct ggml_tensor * states = ggml_reshape_2d(ctx, s, n_state, kv_size);
```
Like we did above we can take a look at the tensor `s`:
```console
(gdb) p *s
$104 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU,
buffer = 0x555555cd4f70, ne = {131072, 1, 1, 1}, nb = {4, 524288, 524288, 524288},
op = GGML_OP_NONE, op_params = {0 <repeats 16 times>}, flags = 0,
grad = 0x0, src = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
view_src = 0x0, view_offs = 0, data = 0x7fff35a04020,
name = "cache_v_l0", '\000' <repeats 53 times>, extra = 0x0}
(gdb) p n_state
$105 = 131072
(gdb) p kv_size
$106 = 1
```
And this will create the same operations that we saw earlier.

After this `cur` reshaped from 2048 columns x 512 rows to 1 batch(?), 512 rows
and 2048 columns:
```c++
    cur = ggml_reshape_3d(ctx0, inpL, n_embd, n_seq_tokens, n_seqs);
```
```console
(gdb) p *cur
$121 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0, ne = {2048, 512, 1, 1}, nb = {4, 8192, 4194304,
    4194304}, op = GGML_OP_RESHAPE, op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0, src = {0x55555644c770, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x55555644c770, view_offs = 0, data = 0x0,
  name = " (reshaped)", '\000' <repeats 52 times>, extra = 0x0}
```
And the `token_shift` vector is also reshaped from 4096 1d to 2048 columns, 2 rows:
and 1 batch:
```c++
    token_shift = ggml_reshape_3d(ctx0, token_shift, n_embd, 2, n_seqs);
```
```console
(gdb) p *token_shift
$128 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0, ne = {2048, 2, 1, 1}, nb = {4, 8192, 16384, 16384},
  op = GGML_OP_RESHAPE, op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0, src = {0x55555644d460, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0}, view_src = 0x55555644cbc0, view_offs = 0, data = 0x0,
  name = "node_2 (view) (reshaped)", '\000' <repeats 39 times>, extra = 0x0}
```
Following that we are creating a 3d view of the `token_shift` tensor:
```c++
     struct ggml_tensor * att_shift = ggml_view_3d(ctx0, token_shift, n_embd, 1, n_seqs, token_shift->nb[1], token_shift->nb[2], 0);
     struct ggml_tensor * att_shift = ggml_view_3d(ctx0, token_shift, 2048  , 1,      1,               8192,              16384, 0);
 ```
 So this is basically just a view of the `token_shift` tensor:
 ```console
 (gdb) p *att_shift
$135 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0, ne = {2048, 1, 1, 1}, nb = {4, 8192, 16384, 16384},
  op = GGML_OP_VIEW, op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0, src = {0x55555644e150, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0}, view_src = 0x55555644cbc0, view_offs = 0, data = 0x0,
  name = "node_2 (view) (reshaped) (view)", '\000' <repeats 32 times>, extra = 0x0}
```
Now I think this is for the time mixing part of the RWKV model.


Then we also create a view for the feed-forward (channel mixing?) shift but use
an offset of 2048 (`n_embd`):
```c++
            struct ggml_tensor * ffn_shift = ggml_view_3d(ctx0, token_shift, n_embd, 1, n_seqs, token_shift->nb[1], token_shift->nb[2], n_embd * ggml_element_size(token_shift));
```
Recall that the offset is multipled by the size of the elements in the tensor!

So it might be that we have one tensor of size 4096 which is used to store
both the key and values vectors which are each 2048 in size.

```console
(gdb) p *ffn_shift
$141 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0, ne = {2048, 1, 1, 1}, nb = {4, 8192, 16384, 16384},
  op = GGML_OP_VIEW, op_params = {8192, 0 <repeats 15 times>}, flags = 0, grad = 0x0, src = {0x55555644e150, 0x0, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x55555644cbc0, view_offs = 8192, data = 0x0,
  name = "node_2 (view) (reshaped) (view)", '\000' <repeats 32 times>, extra = 0x0}
```
Then we are going to build the attention norm:
```c++
  struct ggml_tensor * x_norm_att = llm_build_norm(ctx0, cur, hparams, layer->attn_norm, layer->attn_norm_b, LLM_NORM, cb, il);
```
So `layer->attn_norm` is the tensor passed in as the parameter `mw` which is
the is γ:
```console
(gdb) p *layer->attn_norm
$143 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x555556e40520, ne = {2048, 1, 1, 1}, nb = {4, 8192, 8192,
    8192}, op = GGML_OP_NONE, op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0, src = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0}, view_src = 0x0, view_offs = 0, data = 0x7fff46757da0,
  name = "blk.0.attn_norm.weight", '\000' <repeats 41 times>, extra = 0x0}
```
And `layer->attn_norm_b` is the tensor passed in as the parameter `mb` which is
the β (bias):
```console
(gdb) p *layer->attn_norm_b
$144 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x555556e40520, ne = {2048, 1, 1, 1}, nb = {4, 8192, 8192,
    8192}, op = GGML_OP_NONE, op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0, src = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0}, view_src = 0x0, view_offs = 0, data = 0x7fff46759da0, name = "blk.0.attn_norm.bias", '\000' <repeats 43 times>,
  extra = 0x0}
```

```c++
            struct ggml_tensor * x_prev = ggml_concat(
                ctx0,
                att_shift,
                ggml_view_3d(ctx0, x_norm_att, n_embd, n_seq_tokens - 1, n_seqs, x_norm_att->nb[1], x_norm_att->nb[2], 0),
                1
            );
```
So this will concatenate the `att_shift` tensor with a view of the `x_norm_att`
and notice this is `n_seq_tokens - 1` which is 511.
```console
(gdb) p *x_prev->src[1]
$13 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0, ne = {2048, 511, 1, 1}, nb = {4, 8192, 4194304,
    4194304}, op = GGML_OP_VIEW, op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0, src = {0x55555644e5a0, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x55555644e5a0, view_offs = 0, data = 0x0,
  name = " (view)", '\000' <repeats 56 times>, extra = 0x0}
```

This is then passed into the function `llm_build_rwkv6_time_mix`. So reaonably
up until this point we have been creating tensors in preperation for the
time mixing block in the diagram above (I think).
```c++
    cur = ggml_add(ctx0, cur, llm_build_rwkv6_time_mix(lctx, ctx0, layer, x_norm_att, x_prev, &wkv_states));
```
```c++
static struct ggml_tensor * llm_build_rwkv6_time_mix(
        struct llama_context & lctx,
        struct ggml_context * ctx,
        const struct llama_layer * layer,
        struct ggml_tensor * cur,
        struct ggml_tensor * x_prev,
        struct ggml_tensor ** wkv_state) {

    size_t n_embed      = cur->ne[0];
    size_t n_seq_tokens = cur->ne[1];
    size_t n_seqs       = cur->ne[2];

    size_t head_size  = layer->time_mix_first->ne[0];
    size_t head_count = layer->time_mix_first->ne[1];

    size_t n_tokens = n_seqs * n_seq_tokens;
```
```console
(gdb) p n_embed
$19 = 2048
(gdb) p n_seq_tokens
$20 = 512
(gdb) p n_seqs
$21 = 1

(gdb) p head_size
$22 = 64
(gdb) p head_count
$23 = 32
```
`layer->time_mix_first` which looks like this
```console
(gdb) p *layer->time_mix_first
$25 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x555556e40520, ne = {64, 32, 1, 1}, nb = {4, 256, 8192,
    8192}, op = GGML_OP_NONE, op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0, src = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0}, view_src = 0x0, view_offs = 0, data = 0x7fff46a71da0,
  name = "blk.0.time_mix_first.weight", '\000' <repeats 36 times>, extra = 0x0}

(gdb) p n_tokens
$27 = 512
```
The first operation is a element wise subtraction:
```c++
    struct ggml_tensor * sx = ggml_sub(ctx, x_prev, cur);
```
So `cur` would be the current input sequence I think and `x_prev` would be the
previous input sequence. So `sx` would be the difference between the two and
be the change over time. This is then used in in the linear interpolation:
```c++
    struct ggml_tensor * xxx = ggml_add(ctx, ggml_mul(ctx, sx, layer->time_mix_lerp_x), cur);
```
The `time_mix_lerp_x` in the code might be the `µx (mu_x)` mentioned in the
paper which is part of the data-dependent (ddlerp) mechanism.
```
ddlerp□(a,b) = a + (b − a) ⊙ lora□(a +(b − a) ⊙ µx)
```
So together the following lines are calculating the first part of this
which is `a + (b - a) ⊙ µx)` where a is `cur`, b is `x_prev` and `µx` is:
```c++
    struct ggml_tensor * sx = ggml_sub(ctx, x_prev, cur);
    struct ggml_tensor * xxx = ggml_add(ctx, ggml_mul(ctx, sx, layer->time_mix_lerp_x), cur);
```
So `layer->time_mix_lerp_x` is a learnable parameter that controls the initial
mixing of the current and previous inputs.

And recall that `ddlerp` is defined for `r`, `w`, `k` and `v` which are the
```
ddlerp_r(a,b) = a + (b − a) ⊙ lora_r(a +(b − a) ⊙ µx)
ddlerp_w(a,b) = a + (b − a) ⊙ lora_w(a +(b − a) ⊙ µx)
ddlerp_k(a,b) = a + (b − a) ⊙ lora_k(a +(b − a) ⊙ µx)
ddlerp_v(a,b) = a + (b − a) ⊙ lora_v(a +(b − a) ⊙ µx)
```
Notice that we have `a + (b - a) ⊙ µx)` in each of these and we can therefore
reuse the `xxx` tensor for each of these.
I'm not a fan of the name `xxx` and perhaps a different name would be better,
something link `initial_interpolation`, `time_shift_interpolation`. Anyway back
to business.
We have the initial interpolation value calculated and in `xxx`, next we have:
```c++
    xxx = ggml_reshape_4d(
        ctx,
        ggml_tanh(
            ctx,
            ggml_mul_mat(ctx, layer->time_mix_w1, xxx)
        ),
        layer->time_mix_w1->ne[1] / 5, 1, 5, n_tokens
    );
```
The data-dependent linear interpolation has a `lora` function defined like this:
```
lora□(x) = λ□  + tanh(xA□) B□
ddlerp□(a,b) = a + (b − a) ⊙ lora□(a +(b − a) ⊙ µx)

A□ ∈ R D×32
B□ ∈ R 32×D
```
This is implementing part of the lora function from the paper. `x` would be
`xxx` in our case (which is `a + (b-a) ⊙ µx`). So what we are doing it computing
the part that is common for `lora_r`, `lora_w`, `lora_k`, `lora_v`, and `lora_g`.
And `layer->time_mix_w1` is the `A` matrix in the `lora` function. So this is
```
(gdb) p *layer->time_mix_w1
$52 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x555556e40520, ne = {2048, 160, 1, 1}, nb = {4, 8192,
    1310720, 1310720}, op = GGML_OP_NONE, op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0, src = {0x0, 0x0, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0, view_offs = 0, data = 0x7fff4676fda0,
  name = "blk.0.time_mix_w1.weight", '\000' <repeats 39 times>, extra = 0x0}
(gdb) p ggml_n_dims(layer->time_mix_w1)
$53 = 2
```
Now, in the paper the matrix A is defined with the box notation indicating that
there is a specific A matrix (A□ ∈ R D×32) for each of the `r`, `w`, `k`, `v`,
and `g`. But here there is only one A matrix. This is actually computing the
values for all 5 components at once. Notice that is is all surrounded by a
reshape operation:
```
    xxx = ggml_reshape_4d(
        ctx,
        ggml_tanh(
            ctx,
            ggml_mul_mat(ctx, layer->time_mix_w1, xxx)
        ),
        layer->time_mix_w1->ne[1] / 5, 1, 5, n_tokens
    );
```
So we are reshaping the output of the tanh operation which is a tensor and
```console
(gdb) p layer->time_mix_w1->ne[1]
$54 = 160

(gdb) p layer->time_mix_w1->ne[1] / 5
$55 = 32
```
So we will have 32 columns, 1 row, and 5 channels/batches, and 512 tokens.
This a way to optimize the computation and doing it all at once.
So after this xxx will be a 4d tensor
```console
(gdb) p *xxx
$58 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0,
ne = {32, 1, 5, 512}, nb = {4, 128, 128, 640}, op = GGML_OP_RESHAPE,
op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0,
src = {0x55555644f290, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
view_src = 0x55555644f290, view_offs = 0, data = 0x0,
name = " (reshaped)", '\000' <repeats 52 times>, extra = 0x0}
```
One way to think about this is that we have 512 tokens, and for each token
we have 5 groups ('r', 'w', 'k', 'v', 'g') which each are componsed for 1 row
with 32 columns. When going throught this I kind of lost track of the fact that
this we are processing one token at a time (at least at inference time)

Next we have:
```c++
    xxx = ggml_cont(ctx, ggml_permute(ctx, xxx, 0, 1, 3, 2));
```
The `ggml_permute` function is "saying" that we want to use 0 as the first
dimension (same as it currently is), 1 as the second dimension (same as it
currently is), and swap the 2 and 3 dimensions.
```console
(gdb) p ggml_permute(ctx, xxx, 0, 1, 3, 2)
$61 = (ggml_tensor *) 0x55555644f570
(gdb) p *$61
$62 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0, ne = {32, 1, 512, 5}, nb = {4, 128, 640, 128},
  op = GGML_OP_PERMUTE, op_params = {0, 1, 3, 2, 0 <repeats 12 times>}, flags = 0, grad = 0x0, src = {0x55555644f400, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x55555644f290, view_offs = 0, data = 0x0,
  name = " (reshaped) (permuted)", '\000' <repeats 41 times>, extra = 0x0}
```
And following that we have a matrix multipliction::
```c++
    xxx = ggml_mul_mat(
        ctx,
        ggml_reshape_4d(
            ctx,
            layer->time_mix_w2,
            layer->time_mix_w2->ne[0], layer->time_mix_w2->ne[1], 1, 5
        ),
        xxx
    );
```
In a simliar manner to above the tensor `layer->time_mix_w2` also contains the
learned weights for all 5 components. I think this represents the `B` matrix in
paper:
```
lora□(x) = λ□  + tanh(xA□) B□
```
```console
(gdb) p *layer->time_mix_w2
$68 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU,
buffer = 0x555556e40520,
ne = {32, 2048, 5, 1}, nb = {4, 128, 262144, 1310720},
op = GGML_OP_NONE, op_params = {0 <repeats 16 times>}, flags = 0,
grad = 0x0, src = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
view_src = 0x0, view_offs = 0, data = 0x7fff468afda0,
name = "blk.0.time_mix_w2.weight", '\000' <repeats 39 times>, extra = 0x0}
```
So this will be multipled by xxx which is the output of the previous tanh
operation for all 5 components. And in the same way this is performing the
matrix multiplication of B for all components.

And finally after doing all of the above calculations we can compute, or rather
create compute nodes, for the components:
```c++
    struct ggml_tensor *mw = ggml_view_2d(ctx, xxx, n_embed, n_tokens, xxx->nb[1], 0);
    struct ggml_tensor *mk = ggml_view_2d(ctx, xxx, n_embed, n_tokens, xxx->nb[1], n_embed * n_tokens * sizeof(float));
    struct ggml_tensor *mv = ggml_view_2d(ctx, xxx, n_embed, n_tokens, xxx->nb[1], n_embed * n_tokens * 2 * sizeof(float));
    struct ggml_tensor *mr = ggml_view_2d(ctx, xxx, n_embed, n_tokens, xxx->nb[1], n_embed * n_tokens * 3 * sizeof(float));
    struct ggml_tensor *mg = ggml_view_2d(ctx, xxx, n_embed, n_tokens, xxx->nb[1], n_embed * n_tokens * 4 * sizeof(float));
```
And notice that these are views and not new tensors so there is no copying being
done here.

After this we have:
```c++
    struct ggml_tensor * xw = ggml_add(
        ctx,
        ggml_mul(
            ctx,
            ggml_add(ctx, mw, layer->time_mix_lerp_w),
            sx
        ),
        cur
    );
```
Now, this is implementing the last part of the `ddlerp_w` function, and at this
stage `mv` is a view into the lora part of the computation for the w component.
And we also have `sw` which is the difference between the current and previous
input sequences.
So this is the last part of the `ddlerp_w` function, and notice that we first
have an addition of the `mw` and `layer->time_mix_lerp_w`, then multiply this
by `sx` and finally add `cur` to this:
:
```
ddlerp_w(a,b) = a + (b − a) ⊙ (lora_w(a + (b − a) ⊙ µx) + µw)

a                      = cur
(b - a)                = sx
mw                     = lora_w(a + (b − a) ⊙ µx)
layer->time_mix_lerp_w = µw

ddlerp_w(cur,b) = cur + sx ⊙ (mw + layer->time_mix_lerp_w)
```
So this is what the above is computing.
And we will have something similar for `xk`, `xv`, `xr`, and `xg`.

Next, we are calling `llm_build_lora_mm` which will be the `lora_r` and
creating the r tensor. We pass in `xr` from above.:
```c++
    struct ggml_tensor * r = ggml_reshape_4d(ctx,
        llm_build_lora_mm(lctx, ctx, layer->time_mix_receptance, xr),
        head_size,
        1,
        head_count,
        n_tokens);
```
So we are passing in `xr` which is:
```console
(gdb) p *xr
$75 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0,
ne = {2048, 512, 1, 1}, nb = {4, 8192, 4194304, 4194304},
op = GGML_OP_ADD, op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0,
src = {0x5555564510c0, 0x55555644ecd0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
view_src = 0x0, view_offs = 0, data = 0x0, name = '\000' <repeats 63 times>, extra = 0x0}
```
And `layer->time_mix_receptance` I think is the `W_r` matrix:
```
lerp□(xt ,xt−1) W□,
lerp_r(xt ,xt−1) Wr,
```
And in our case we have already computed the ddlerp part which is in the tensor
`xr` so this becomes:
```
xr * W_r
```
We can inspect the `layer->time_mix_receptance` tensor:
```console
(gdb) p *layer->time_mix_receptance
$76 = {type = GGML_TYPE_F16, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x555556e40520,
ne = {2048, 2048, 1, 1}, nb = {2, 4096, 8388608, 8388608},
op = GGML_OP_NONE, op_params = {0 <repeats 16 times>},
flags = 0, grad = 0x0, src = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
view_src = 0x0, view_offs = 0, data = 0x7fff46a73da0, name = "blk.0.time_mix_receptance.weight", '\000' <repeats 31 times>, extra = 0x0}
```
These are passed to `llm_build_lora_mm`:
```c++
// do mat_mul, while optionally apply lora
static struct ggml_tensor * llm_build_lora_mm(
        struct llama_context & lctx,
         struct ggml_context * ctx0,
          struct ggml_tensor * w,
          struct ggml_tensor * cur) {
    struct ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);
```
So we can see that we are multiplying the `W_r` matrix with the `xr` tensor.
In this case there are no lora adapters applied so the res is just returned.

The same is then done for `k` and `v`:
```c++
    struct ggml_tensor * k = ggml_reshape_4d(ctx, llm_build_lora_mm(lctx, ctx, layer->time_mix_key,        xk), 1,         head_size, head_count, n_tokens);
    struct ggml_tensor * v = ggml_reshape_4d(ctx, llm_build_lora_mm(lctx, ctx, layer->time_mix_value,      xv), head_size, 1,         head_count, n_tokens);
```
The shape/sizes of `layer->time_mix_key` and `layer->time_mix_value` are the
same as `layer->time_mix_receptance`.

Recall from the diagram above that th `g` is passed through a `SiLU` activation
function (Sigmoid-weighted Linear Unit):
```
silu(x) = x * sigmoid(x)
```
And notice that before the `g` tensor is passed through the `SiLU` activation
we also multiply by `gt/layer->time_mix_gate`:
```c++
    struct ggml_tensor * g = ggml_silu(
        ctx,
        llm_build_lora_mm(lctx, ctx, layer->time_mix_gate, xg)
    );
```
In Finch we also have `w` which is passed to the `wkv` box in the diagram above.
And in the paper we have:
```
w_t = exp(− exp(d_t))

d_t = lora_d(ddlerp_d(x_t, x_t−1))
```
The following code is computing `w_t` I think but with some optimizations. We
have `d_t` in `xw` which we are multiplying with `layer->time_mix_decay_w1` and
then passing that through a `tanh` activation function:
```c++
    struct ggml_tensor * w = ggml_mul_mat(
        ctx,
        layer->time_mix_decay_w2,
        ggml_tanh(
            ctx,
            ggml_mul_mat(ctx, layer->time_mix_decay_w1, xw)
        )
    );
    w = ggml_add(ctx, w, ggml_reshape_1d(ctx, layer->time_mix_decay, n_embed));
    w = ggml_exp(ctx, ggml_neg(ctx, ggml_exp(ctx, w)));
    w = ggml_reshape_4d(ctx, w, 1, head_size, head_count, n_tokens);
```
So first the tanh operation is created in the computation graph taking the
`xw` tensor as input. Then this is multipled by `layer->time_mix_decay_w1`
(Note that `layer->time_mix_decay_w2` is used in the surrounding multiplication
operation but not in the tanh operation):
```console
(gdb) p *layer->time_mix_decay_w1
$83 = {type = GGML_TYPE_F16, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x555556e40520,
ne = {2048, 64, 1, 1}, nb = {2, 4096, 262144, 262144},
op = GGML_OP_NONE, op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0, src = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
0x0, 0x0, 0x0}, view_src = 0x0, view_offs = 0, data = 0x7fff469f1da0,
name = "blk.0.time_mix_decay_w1.weight", '\000' <repeats 33 times>, extra = 0x0}
```
So we have a two-layer neural network here (the `time_mix_decay_w1` and
`time_mix_decay_w2`) with the tanh activation function in between. And then we
add the `layer->time_mix_decay` bias to the result of the tanh operation.
This might just be something that they found worked well in pratice.

Next we are transposing the `w`, `k`, and `v` tensors:
```c++
    k = ggml_transpose(ctx, k);
    v = ggml_transpose(ctx, v);
    r = ggml_transpose(ctx, r);
```
And then calling the `ggml_rwkv_wkv` function with them:
```c++
    struct ggml_tensor * wkv_output = ggml_rwkv_wkv(ctx, k, v, r, layer->time_mix_first, w, *wkv_state);
```
```c++
struct ggml_tensor * ggml_rwkv_wkv(
        struct ggml_context * ctx,
        struct ggml_tensor * k,
        struct ggml_tensor * v,
        struct ggml_tensor * r,
        struct ggml_tensor * tf,           // time_mix_first
        struct ggml_tensor * td,           // w (time decay)
        struct ggml_tensor * state) {

    const int64_t S = k->ne[0];
    const int64_t H = k->ne[2];
    const int64_t n_tokens = k->ne[3];
    const int64_t n_seqs = state->ne[1];
```
```
(gdb) p S
$88 = 64
(gdb) p H
$89 = 32
(gdb) p n_tokens
$90 = 512
(gdb) p n_seqs
$93 = 1
```
This function contains a lot of asserts which I'm going to skip over. The part
that I'm most interested in is the following:
```c++
    // concat output and new_state
    const int64_t ne[4] = { S * H, n_tokens + S * n_seqs, 1, 1 };
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op   = GGML_OP_RWKV_WKV;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = k;
    result->src[1] = v;
    result->src[2] = r;
    result->src[3] = tf;            // time first (u
    result->src[4] = td;            // time decay
    result->src[5] = state;

    return result;
```
First we are defining the dimensions for a new tensor which will have 4
dimenions:
```
    const int64_t ne[4] = { S * H, n_tokens + S * n_seqs, 1, 1 };
```
```console
(gdb) p ne
$97 = {2048, 576, 1, 1}
```
And then we are creating the tensor and setting the src's for it.


Lets remind ourselves of the `wkv_t` formula in the paper:
```
                                 t-1
wkv_t = diag(u) * K_t^T * v_t +   Σ  diag(w)^t-1-i * K_i^T * v_i 
                                 i=1
```
`K_t` (current) and `K_i` (previous timesteps) are represented by k and likwise
for v. And `u` is represented by `tf` (time first) and `w` by `td` (time decay).
State state is for the summation part of the formula for previous tokens and
recall that all that is happening at this stage is that a compuation graph is
being built and that later when this tensor operation is executed the actual
values will be available to use by it.

This tensor looks like this:
```console
(gdb) p *wkv_output
$106 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0,
ne = {2048, 576, 1, 1}, nb = {4, 8192, 4718592, 4718592},
op = GGML_OP_RWKV_WKV, op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0,
src = {0x555556453060, 0x5555564531d0, 0x555556453340, 0x555556d3a160,
0x555556452ef0, 0x55555644db90, 0x0, 0x0, 0x0, 0x0},
view_src = 0x0, view_offs = 0, data = 0x0, name = '\000' <repeats 63 times>, extra = 0x0}
```
Back in `llm_build_rwkv6_time_mix` this tensor, named `wkv_output`, is then
used to create a view tensor and set the current tensor to this view:
```c++
    cur = ggml_view_1d(ctx, wkv_output, n_embed * n_tokens, 0);
```
```console
(gdb) p *cur
$108 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0,
ne = {1048576, 1, 1, 1}, nb = {4, 4194304, 4194304, 4194304}, op = GGML_OP_VIEW,
op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0,
src = {0x5555564534b0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
view_src = 0x5555564534b0, view_offs = 0, data = 0x0,
name = " (view)", '\000' <repeats 56 times>, extra = 0x0}
```
And we also update the `wkv_state` tensor:
```c++
    *wkv_state = ggml_view_1d(ctx, wkv_output, n_embed * head_size * n_seqs, n_embed * n_tokens * sizeof(float));
```
```console
(gdb) p **wkv_state
$110 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0,
ne = {131072, 1, 1, 1}, nb = {4, 524288, 524288, 524288}, op = GGML_OP_VIEW,
op_params = {4194304, 0 <repeats 15 times>}, flags = 0, grad = 0x0, src = {0x5555564534b0, 0x0, 0x0,
0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x5555564534b0, view_offs = 4194304, data = 0x0,
name = " (view)", '\000' <repeats 56 times>, extra = 0x0}
```
After the wkv operation we have a LayerNorm operation
```
    cur = ggml_reshape_3d(ctx, cur, n_embed / head_count, head_count, n_tokens);
    cur = ggml_norm(ctx, cur, 64e-5f);
```
The reshaping will be to:
```console
(gdb) p *cur
$111 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0, ne = {64, 32, 512, 1}, nb = {4, 256, 8192, 4194304},
  op = GGML_OP_RESHAPE, op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0, src = {0x555556453620, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0}, view_src = 0x5555564534b0, view_offs = 0, data = 0x0,
  name = " (view) (reshaped)", '\000' <repeats 45 times>, extra = 0x0}
```
Next we have a reshape operation what will bring the current tensor back into
the shape 2048 columns and 512 rows:
```c++
    cur = ggml_reshape_2d(ctx, cur, n_embed, n_tokens);
```
Think of this as 512 tokens with their embeddings of size 2048: 
```console
(gdb) p *cur
$112 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0, ne = {2048, 512, 1, 1}, nb = {4, 8192, 4194304,
    4194304}, op = GGML_OP_RESHAPE, op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0, src = {0x555556453a70, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x555556453a70, view_offs = 0, data = 0x0,
  name = " (reshaped)", '\000' <repeats 52 times>, extra = 0x0}
```

The next part I was not able to map back to the paper but might be a custom
normalization operation:
```c++
    cur = ggml_add(ctx, ggml_mul(ctx, cur, layer->time_mix_ln), layer->time_mix_ln_b);
```
Here we are first scaling the current tensor by `layer->time_mix_ln`
(layer normalization) and then adding `layer->time_mix_ln_b`
(layer normalization bias) to it. This is a way to normalize the tensor.
```console
(gdb) p *layer->time_mix_ln
$114 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x555556e40520, ne = {2048, 1, 1, 1}, nb = {4, 8192, 8192,
    8192}, op = GGML_OP_NONE, op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0, src = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0}, view_src = 0x0, view_offs = 0, data = 0x7fff49273da0,
  name = "blk.0.time_mix_ln.weight", '\000' <repeats 39 times>, extra = 0x0}
```
This is something that seems to be common in implementation of papers but won't
be mentioned in the paper itself.

Next we have the multiplication of the current tensor with g, which recall
was passed through the SiLU activation function:
```c++
    cur = ggml_mul(ctx, cur, g);
    cur = llm_build_lora_mm(lctx, ctx, layer->time_mix_output, cur);
```
And then the final element wise multiplication with the `time_mix_output` matrix
which I think is the `W_o` in:
```
o_t = concat( SiLU(g_t) ⊙ LayerNorm(r_t * wkv_t)) W_o
```
The current tensor will then be reshaped and returned:
```c++
    return ggml_reshape_3d(ctx, cur, n_embed, n_seq_tokens, n_seqs);
```
So that lands us back in `build_rwkv6`:
```c++
    ggml_build_forward_expand(gf, cur);
```
This is simply adding the current tensor operation to the graph, so all the
operations that we have done so far will be added into the compute graph.

Next we copyt the updated `wkv_states` and add them to the computation graph:
```c++
            ggml_build_forward_expand(
                gf,
                ggml_cpy(
                    ctx0,
                    wkv_states,
                    ggml_view_1d(
                        ctx0,
                        kv_self.v_l[il],
                        hparams.n_embd_v_s() * n_seqs,
                        hparams.n_embd_v_s() * kv_head * ggml_element_size(kv_self.v_l[il])
                    )
                )
            );
```
```console
(gdb) p *ggml_view_1d(ctx0, kv_self.v_l[il], hparams.n_embd_v_s() * n_seqs, hparams.n_embd_v_s() * kv_head * ggml_element_size(kv_self.v_l[il]))
$119 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0, ne = {131072, 1, 1, 1}, nb = {4, 524288, 524288,
    524288}, op = GGML_OP_VIEW, op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0, src = {0x555556e5f5d0, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x555556e5f5d0, view_offs = 0, data = 0x7fff35a04020,
  name = "cache_v_l0 (view)", '\000' <repeats 46 times>, extra = 0x0}
```

Next we have a normalization for the feed-forward network which is the same
as the channel mixing block in the paper so the lower block in the diagram
above.
```c++
   struct ggml_tensor * x_norm_ffn = llm_build_norm(ctx0, cur, hparams,
       layer->attn_norm_2, layer->attn_norm_2_b, LLM_NORM, cb, il);
```
This is somewhat confusing that the `layer->attn_norm_2` and
`layer->attn_norm_2_b` and "attention" is not really happening in the channel
mixing block (this is dealing with the feed-forward network). Lets move on and
see if this makes more sense later on or if I need to revise this.

Following that we have:
```c++
            x_prev = ggml_concat(
                ctx0,
                ffn_shift,
                ggml_view_3d(ctx0, x_norm_ffn, n_embd, n_seq_tokens - 1, n_seqs, x_norm_ffn->nb[1], x_norm_ffn->nb[2], 0),
                1
            );
            cur = ggml_add(ctx0, cur, llm_build_rwkv6_channel_mix(lctx, ctx0, layer, x_norm_ffn, x_prev));
            ggml_build_forward_expand(gf, cur);
```
So this is concatenating the `ffn_shift` tensor with a view of the `x_norm_ffn`.
Notice that the view is `n_seq_tokens - 1` which is 511 in our case.
So this is creating a 3d view of:
```console
(gdb) p *x_norm_ffn
$125 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0,
ne = {2048, 512, 1, 1}, nb = {4, 8192, 4194304, 4194304},
op = GGML_OP_ADD, op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0,
src = {0x555556454d20, 0x555556d39470, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
view_src = 0x0, view_offs = 0, data = 0x0, name = '\000' <repeats 63 times>,
extra = 0x0}
```
The first dimension will have a size of 2048 (x), the second dimension will have
511. The stride for the first dimension will be 8192 and for the second dimension
it will be 4194304. `ffn_shift` contains the shifted information from the
previous timestep.

And then we are adding the result of the `llm_build_rwkv6_channel_mix` to the
current tensor and adding this to the computation graph.
Now, lets look closer at the `llm_build_rwkv6_channel_mix` function:
```c++
static struct ggml_tensor * llm_build_rwkv6_channel_mix(
        struct llama_context & lctx,
        struct ggml_context * ctx,
        const struct llama_layer * layer,
        struct ggml_tensor * cur,
        struct ggml_tensor * x_prev) {
    struct ggml_tensor * sx = ggml_sub(ctx, x_prev, cur);
    struct ggml_tensor * xk = ggml_add(ctx, ggml_mul(ctx, sx, layer->channel_mix_lerp_k), cur);
    struct ggml_tensor * xr = ggml_add(ctx, ggml_mul(ctx, sx, layer->channel_mix_lerp_r), cur);

    struct ggml_tensor * r = ggml_sigmoid(ctx, llm_build_lora_mm(lctx, ctx, layer->channel_mix_receptance, xr));
    struct ggml_tensor * k = ggml_sqr(
        ctx,
        ggml_relu(
            ctx,
            llm_build_lora_mm(lctx, ctx, layer->channel_mix_key, xk)
        )
    );

    return ggml_mul(ctx, r, llm_build_lora_mm(lctx, ctx, layer->channel_mix_value, k));
}
```
Where cur is the normalization tensor, and `x_prev` is the previous input
sequence and this simlar to what we saw in time mixing block. This `sx` is
then used for the lerp of xk and xr.

Following that we are doing a matrix multiplication of `rx` with the learned
tensor `layer->channel_mix_receptance` and passing that through a sigmoid.

In the channel mixing we have:
```
r_t'  = lerp_r(x_t', x_t_' -1 ) W_r'    // xr
k_t'  = lerp_k(x_t', x_t_' -1 ) W_k'    // xk
v_t'  = ReLU(k')^2 W_v'                 // k
o_t'  = σ(r_t') ⊙ v_t')                 // sigmoid(r) * k
```
Hopefully this makes sense with the above comments but the code is pretty much
following the paper apart from the incorrect "Channel Mixing" block in Figure 1
which caused my a bit of confusion. I've updated the diagram in this document
to reflect the correct nodes/operations of the channel mixing block.

So that with that we are done with the `llm_build_rwkv6_channel_mix` function.
```
    cur = ggml_add(ctx0, cur, llm_build_rwkv6_channel_mix(lctx, ctx0, layer, x_norm_ffn, x_prev));
    ggml_build_forward_expand(gf, cur);
```
So `cur` will be the last operation from the channel mixing functions and then
we are adding this to the computation graph.
Next we have:
```c++
    struct ggml_tensor * last_norm_att = ggml_view_3d(ctx0, x_norm_att,
        n_embd, 1, n_seqs, x_norm_att->nb[1], x_norm_att->nb[2],
        (n_seq_tokens-1)*n_embd*ggml_element_size(x_norm_att));

    struct ggml_tensor * last_norm_ffn = ggml_view_3d(ctx0, x_norm_ffn,
        n_embd, 1, n_seqs, x_norm_ffn->nb[1], x_norm_ffn->nb[2],
        (n_seq_tokens-1)*n_embd*ggml_element_size(x_norm_ffn));

    token_shift = ggml_concat(ctx0, last_norm_att, last_norm_ffn, 1);
```
Now, this is creatings views of the last tokens normalized representations from
both the time mixing and channel mixing blocks. These are then concatenated
together. I think this is part of the passing of information from the previous
layer to the next. So `token_shift` will contain the state of the last token
after it has passed through the time and channel mixing blocks.

Then we are copying the `token_shift` tensor into `kv_self.k_l[il]`:
```c++
            ggml_build_forward_expand(
                gf,
                ggml_cpy(
                    ctx0,
                    ggml_view_1d(ctx0, token_shift, n_embd * n_seqs * 2, 0),
                    ggml_view_1d(ctx0, kv_self.k_l[il], hparams.n_embd_k_s() * n_seqs, hparams.n_embd_k_s() * kv_head * ggml_element_size(kv_self.k_l[il]))
                )
            );
```
So I think on the next iteration the last state of the tokens can be accessed
using `kv_self.k_l[il]`.

Next we have:
```
            if (hparams.rescale_every_n_layers != 0 && (il + 1) % hparams.rescale_every_n_layers == 0) {
                cur = ggml_scale(ctx0, cur, 0.5F);
            }
```
```console
(gdb) p hparams.rescale_every_n_layers
$140 = 6
```
`cur` has the what was returned from the channel mixing block and scaling is
basically multiplying the values in the tensor by 0.5. So this called layer
rescaling or "gradient rescaling" which helps mitigate the problem of exploding
or vanishing gradients during training. It can also help with numerical
stability.  So we are halving the values in the tensor.

Following the scaling we have:
```c++
            cur = lctx.cvec.apply_to(ctx0, cur, il);
```
This is related to control vectors if any are present the apply_to function
would be run for the current tensor. 
```c++
    struct ggml_tensor * apply_to(struct ggml_context * ctx, struct ggml_tensor * cur, int  il) const {
        ggml_tensor * layer_dir = tensor_for(il);
        if (layer_dir != nullptr) {
            cur = ggml_add(ctx, cur, layer_dir);
        }
        return cur;
    }

```
TODO: Make an example with control vectors to see how this works.
After that we set the name of the current tensor and then set inpL to it and
contine with the next layer.
```c++
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
```
So that was how the RWKV layer operations are built up. The rest of the function
looks like this:
```c++
        cur = inpL;
        struct ggml_tensor * inp_out_ids = build_inp_out_ids();
        cur = ggml_reshape_2d(ctx0, cur, n_embd, n_tokens);
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);

        cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, model.output_norm_b, LLM_NORM, cb, -1);
        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);

        cb(cur, "result_output", -1);
        ggml_build_forward_expand(gf, cur);

        return gf;
```

The following is creating a new 1d tensor of size 512 and setting in to the
`lctx.inp_out_ids` tensor:
```c++
    struct ggml_tensor * build_inp_out_ids() {
        lctx.inp_out_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_outputs);
        cb(lctx.inp_out_ids, "inp_out_ids", -1);
        ggml_set_input(lctx.inp_out_ids);
        return lctx.inp_out_ids;
    }
```
This is then reshaped to 2d or 2048 columns and 512 rows.
We also have a normalization layer and then the output layer is added to the
computation graph.

And with that we are done in `llm.build_rwkv6`. I'll probably need to got
through this a few more times but I think I have an initial understanding of how
the RWKV layer is implemented in llama.cpp.

Now the above just went through building the computation graph but we have not
executed any of the nodes yet. Lets set a break point in `ggml_graph_comput`
and see if we can following the execution:
```console
(gdb) br ggml_graph_compute

Breakpoint 5, ggml_graph_compute (cgraph=0x555555cd4198, cplan=0x7fffffffd220) at ggml/src/ggml.c:20091
20091   enum ggml_status ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan) {
    (gdb) bt
#0  ggml_graph_compute (cgraph=0x555555cd4198, cplan=0x7fffffffd220) at ggml/src/ggml.c:20091
#1  0x00005555555c395f in ggml_backend_cpu_graph_compute (backend=0x555555d08ba0, cgraph=0x555555cd4198)
        at ggml/src/ggml-backend.c:817
#2  0x00005555555c27d4 in ggml_backend_graph_compute_async (backend=0x555555d08ba0, cgraph=0x555555cd4198)
            at ggml/src/ggml-backend.c:282
#3  0x00005555555c742e in ggml_backend_sched_compute_splits (sched=0x555555cd5c10) at ggml/src/ggml-backend.c:1806
#4  0x00005555555c806c in ggml_backend_sched_graph_compute_async (sched=0x555555cd5c10, graph=0x55555640b580)
                at ggml/src/ggml-backend.c:1994
#5  0x000055555567fe44 in llama_graph_compute (lctx=..., gf=0x55555640b580, n_threads=4, threadpool=0x0) at src/llama.cpp:16022
#6  0x0000555555680835 in llama_decode_internal (lctx=..., batch_all=...) at src/llama.cpp:16197
#7  0x000055555568ee33 in llama_decode (ctx=0x555555cd03e0, batch=...) at src/llama.cpp:20006
#8  0x000055555556abdb in main (argc=4, argv=0x7fffffffda98) at src/simple-prompt.cpp:130
```

_wip_

```console
(gdb) br ggml_compute_forward_rwkv_wkv_f32
Breakpoint 2 at 0x5555555a9c4f: file ggml/src/ggml.c, line 16963.
```



### WKV Operator
Now in RWKV instead of using the Q, K, and V matrices the formula looks like
this:
```
                 Σ exp(w_t_i+ k_i) . v_i
Att+(W, K, V)_t = -----------------
                 Σ exp(w_t_i+ k_i)

t = current token (current time step of position of the token in the sequence).
i = previous token (previous time step of position of the token in the sequence).
```
Notice that we are still taking a weighted sum of the values, but we are using
weights that are learned during training, and not the Query values. The keys
are still the same as in the original attention mechanism and contain
information about the current token, but the Query is gone. And notice also that
the operation is addition and not multiplication before the exponentiation.

And notice that this we can see the softmax operation in the forumula above:
```
                 Σ exp(w_t_iᵢ+ k_i) 
                 -----------------
                 Σ exp(w_t_iᵢ+ k_i)
```

So `w` is a learned vector and is called a time decay factor which controls how
quickly the influence of previous tokens decays.

So each entry in this vector, which would have the same size as the models
embedding dimensions/channels, would determine how important each feature is
over time:
```
w_t_i = −(t − i) w

t = current token (current time step of position of the token in the sequence).
i = previous token (previous time step of position of the token in the sequence).
w = learned decay vector where each entry is constraied to be non-negative.
```
So for each channel/feature in the embedding there is an entry in the `w`
vector.
```
Token sequence length = 4
Embedding dimension   = 2


Lets say the learned w looks like this (remember that these values must be
non-negative):

   w = [0.2, 0.9]

First token in the sequence (there is not previous token for this entry)
w_0_0 = -(0 - 0) * 0.2 = 0
w_0_1 = -(0 - 0) * 0.9 = 0

w_1_0 = -(1 - 0) * 0.2 = -0.2
w_1_1 = -(1 - 0) * 0.9 = -0.9

w_2_0 = -(2 - 0) * 0.2 = -0.4
w_2_1 = -(2 - 0) * 0.9 = -1.8

w_3_0 = -(3 - 0) * 0.2 = -0.6
w_3_1 = -(3 - 0) * 0.9 = -2.7
```
Now we need to keep in mind that these values will then be added to the
respective key vectors for each token in the sequence.
```
    Σ exp(w_t_i + k_i)
    -----------------
    Σ exp(w_t_i + k_i)
```
So this is summing over all the tokens in the sequence. And notice that we
are adding the `w_t_i` values to the `k_i` values. Lets just make explicit with
the above example:
```
          0           1           2               3
   x = ["Dan",      "loves"    , "ice"      , "cream"]
   k = [ [0.5, -03], [0.7, 0.2], [-0.1, 0.8], [ 0.3, -0.5] ]

   w_t_i + k_i

Learned w vector:
w = [0.2, 0.9]

Key vectors:
k_0 = [0.5, -0.3]  (for "Dan")
k_1 = [0.7, 0.2]   (for "loves")
k_2 = [-0.1, 0.8]  (for "ice")
k_3 = [0.3, -0.5]  (for "cream")

Calculations for w_t_i + k_i:

For t = 0 ("Dan"):
w_0_0 + k_0 = [-(0-0)*0.2 + 0.5, -(0-0)*0.9 - 0.3] = [0.5, -0.3]

For t = 1 ("loves"):
w_1_0 + k_0 = [-(1-0)*0.2 + 0.5, -(1-0)*0.9 - 0.3] = [0.3, -1.2]
w_1_1 + k_1 = [-(1-1)*0.2 + 0.7, -(1-1)*0.9 + 0.2] = [0.7, 0.2]

For t = 2 ("ice"):
w_2_0 + k_0 = [-(2-0)*0.2 + 0.5, -(2-0)*0.9 - 0.3] = [0.1, -2.1]
w_2_1 + k_1 = [-(2-1)*0.2 + 0.7, -(2-1)*0.9 + 0.2] = [0.5, -0.7]
w_2_2 + k_2 = [-(2-2)*0.2 - 0.1, -(2-2)*0.9 + 0.8] = [-0.1, 0.8]

For t = 3 ("cream"):
w_3_0 + k_0 = [-(3-0)*0.2 + 0.5, -(3-0)*0.9 - 0.3] = [-0.1, -3.0]
w_3_1 + k_1 = [-(3-1)*0.2 + 0.7, -(3-1)*0.9 + 0.2] = [0.3, -1.6]
w_3_2 + k_2 = [-(3-2)*0.2 - 0.1, -(3-2)*0.9 + 0.8] = [-0.3, -0.1]
w_3_3 + k_3 = [-(3-3)*0.2 + 0.3, -(3-3)*0.9 - 0.5] = [0.3, -0.5]
```
Notice that we have different decay values for each feature in the embedding.


### Sigmoid
The R vector is passed through the Sigmoid activation function which squashes
the values between 0 and 1. This is important as it controls how much of the
information is retained. So each value in this vector will be passed through
the Sigmoind function.

### LayerNorm x2 (Small Init Embeddings)
This struck me as somewhat odd that there would be two LayerNorm operations
after each other. But this seems like has to do with "Small Init Embeddings"
which is mentioned in section 3.4 of the paper.

A LayerNorm is defined like this:
```
     
        x - μ
y = γ ( ------) + β
          σ

x = input
μ = mean
σ = standard deviation
β = bias (learned)
γ = scale (learned)
```
And having two will mean that both have different learnable parameters.
The embedding values will be normalized twice, with each normalization
potentially emphasizing different aspects of the input due to the separate
learnable parameters.
In llama.cpp there is a function named `llm_build_norm`:
```c++
static struct ggml_tensor * llm_build_norm(
        struct ggml_context * ctx,
         struct ggml_tensor * cur,
        const llama_hparams & hparams,
         struct ggml_tensor * mw,
         struct ggml_tensor * mb,
              llm_norm_type   type,
         const llm_build_cb & cb,
                        int   il) {
```
Where 'mw' is γ and 'mb' is β.


### Time decay vector
How this actually works is that we have a `vector` w which is learned, and it
tells how much the past matters for each dimension.

Each dimension in the w vector represents a feature, also called a channel in 
contexts like image processing. In image processing and image can have multiple
channels, like red, green, and blue for color images. These represent different
types of information (features) of the image. CNN documents/papers would
probably refer to channels as that is one of the main types that they were
designed for (at least initially), that is to process images. In the transformer
architecture this would be called features. 

And the value in each dimension determines how the influence of each feature
decays over time. The idea being that some features might loose their relavance
over time (faster decay), while others might be more important (slower decay).

For example:
```
       +----+
       |0.0 |
       +----+
       |0.9 |
       +----+
       |0.8 |
       +----+
       |-0.7|
       +----+
       |0.6 |
       +----+
```
So w would have to be the same length as the embeddings length in reality.

Then, wt,i is calculated using:
```
wt,i = -(t - i)w
```
The `(t - i)` part is calculating the relative position of the two tokens. It
tells use how far back the the sequence token i is from the current token t.

So, for our example above we would perhaps have something like the following,
and lets say we are currently looking at t=1, i=3 (column).
Recall that t is the current token in the input sequence, and i represents
another token in the input sequence (typically a previous token).
```
(t - 1) = (1 - 3) = -2
```
This tells us how far back in the sequence the token at position i (3) is from
the current token at position `t`. In this case it is 2 tokens ahead which is
why we have a negative number.

```
wt,i = -(t - i)w

wt,i = -(1 - 3)w
wt,i = -(-2)w
wt,i = 2w

       wt,i:
       +----+
       |2.0 |
       +----+
       |1.8 |
       +----+
       |1.6 |
       +----+
       |-1.4|
       +----+
       |1.2 |
       +----+
```
And this wt,i vector would be added to the Kᵢ vector which represents the
decay-adjusted weights for this specific interaction.

The negative sign in front of (t−i) is important because it ensures that the
influence of previous tokens decays as you move forward in the sequence.

The concept of decay here means that the further back a token is in the sequence
(relative to the current position), the less influence it should have. This is
a common assumption in many sequence models, reflecting the idea that recent
information is often more relevant than older information.
The following is an attempt to visualize this:
```
Input sequence represented as words instead of token embeddings:
"Dan" "loves" "ice" "cream"
  1     2       3       4

Table for calculating wt,i, where the t are the rows and i are the columns.

                          i
      | Dan (1) | loves (2) | ice (3) | cream (4) |
---------------------------------------------------
Dan   |  -(1-1) |  -(1-2)   | -(1-3)  |  -(1-4)   |
(1)   |    0    |   -1      |   -2    |    -3     |
---------------------------------------------------
loves |  -(2-1) |  -(2-2)   | -(2-3)  |  -(2-4)   |
(2)   |    -1   |    0      |    1    |    2      |  t
---------------------------------------------------
ice   |  -(3-1) |  -(3-2)   | -(3-3)  |  -(3-4)   |
(3)   |    -2   |    -1     |    0    |     1     |
---------------------------------------------------
cream |  -(4-1) |  -(4-2)   | -(4-3)  |  -(4-4)   |
(4)   |   -3    |    -2     |   -1    |    0      |
---------------------------------------------------
```
So we have a vector w which contains values that determine how important each
feature is over time. And this is used to modulate (changes/controls) the key
vector with the decay-adjusted weights, reflecting how the relationship between
tokens changes based on their relative position and feature decay rates.

In RWKV, we have the following important components:
* R (Receptance) - A vector which is the receiver and integrator of past information.
Similar to the hidden state in an RNN perhaps?
* W (Weight) - A vector containing positional decay information (trained)
* K (Key) - A matrix containing information about the current token.
* V (Value) - A matrix containing information about the current token.

Ealier we mentioned that in the RWKV model they refer to features as channels
which is good to keep in mind when we see component like channel-mixing, so we
can think of it as feature-mixing.

Lets take a look at the formula for the R vector:
```
r_t = W_r * (μ_r ⊙ x_t + (1 - μ_r) ⊙  x_t-1)

Where:
x_t   = current input token at time t
x_-1  = previous input token at time t-1
μ_r   = (mu) scalar variables? that determines how much of the previous token to
        mix in with the current token. This is a learned value.
⊙     = element-wise multiplication.
1-μ_r = scalar value which determines how much of the current token to mix in
        with the previous token.
W_r   = Weight matrix for the R vector.
```
Now, lets just keep in mind that xₜ and xₜ-1 are vectors of token embeddings and
they are the same length the embedding length. And recall that each dimension in
the W vector represents a feature (channel.

### Time-mixing
First lets take a look at what is called time-mixing which is about integrating
information from different time steps which enables the model to effectivly
remember and use past information when making predictions about the current or
future states.
μᵣ is a scalar value that is learned during training, but just a scalar value
and if it is closer too 1, say 0.7 that would mean that the current input has
more influence, and (1 - μᵣ) would then be 0.3 and specifies how much influence
the last token vector has. These are then used to scale the current input vector
and the last input token:
```
(μᵣ⊙ xᵣ + (1 - μᵣ) ⊙  xₜ-1)
```
After scaling the two vectors they are added together which blends the current
token vector with the previous token vector. This is then multiplied by the
learned weight matrix Wᵣ to produce the R vector.

One thing that confused my a little was the usage of element-wise multiplication
symbol ⊙, because if μᵣ is a scalar then that would be the same thing but I
think this is done for clarity and to make it more general.

In the time-mixing stage we also have:
```
k₁ = Wₖ* (μₖ⊙ xᵣ+ (1 - μₖ) ⊙  xₜ-1)
v₁ = Wᵥ* (μᵥ⊙ xᵣ+ (1 - μᵥ) ⊙  xₜ-1)
```
Notice that in the transformer architecture we also have K an V but those were
copies of the input token embeddings. In this case K and V are produced only
by the current input, and the previous input.
These are basically doing the same thing as the R vector, but for the K and V
vectors. So we are mixing in information from the previous token with the
current token, and then we are using the learned weight matrix to produce the
K and V vectors.

```
r₁ = Wᵣ* (μᵣ⊙ xᵣ+ (1 - μᵣ) ⊙  xₜ-1)
k₁ = Wₖ* (μₖ⊙ xᵣ+ (1 - μₖ) ⊙  xₜ-1)
v₁ = Wᵥ* (μᵥ⊙ xᵣ+ (1 - μᵥ) ⊙  xₜ-1)

      ᵢ₌₀
wkv = ----------------------------------------------
      ₜ₋₁
      Σ exp(-(t - 1 - i)w+kᵢ) + exp(u+k)
      ᵢ₌₀

oₜ = Wₒ* (σ(r₁) ⊙  wkvₜ)
```
This can be visualized as:
```
      +------------+
      |   Out      |
      +------------+
          ↑
          |
        +---+
    +---| ⊙ |---+
    |   +---+   |
  +---+    +--------------+
  | σ |    |      WKV     | 
  +---+    +--------------+
    ↑         ↑         ↑
    |         |         |
  +---+     +---+     +---+
  | R |     | K |     | V |
  +---+     +---+     +---+
    |          |          |
    +----------+----------+
               |
            +-----+
            |  μ  |
            +-----+
               ↑         
               |
               |
```


### WKV Operator
In the following note that `w` is a vector and contains values that determine
how important each feature is over time. And this is used to modulate the
key vector with the decay-adjusted weights.

So this is what the WKV operator is doing:

```
      ₜ₋₁
      Σ exp(-(t - 1 - i)w+kᵢ ⊙ vᵢ + exp(u+kₜ) ⊙ vₜ)   
      ᵢ₌₀
wkv = ----------------------------------------------
      ₜ₋₁
      Σ exp(-(t - 1 - i)w+kᵢ) + exp(u+k)
      ᵢ₌₀
```
Now, `t` is the sequence of token embeddings. So above we are summing over all
the tokens in the sequence. So if we had 10 tokens in the sequence we would
get -(10 - 1 - i). And we would get the following values for all tokens:
```
-(10 - 1 - 0) = -9
-(10 - 1 - 1) = -8
-(10 - 1 - 2) = -7
-(10 - 1 - 3) = -6
-(10 - 1 - 4) = -5
-(10 - 1 - 5) = -4
-(10 - 1 - 6) = -3
-(10 - 1 - 7) = -2
-(10 - 1 - 8) = -1
-(10 - 1 - 9) =  0
```
So lets take the first entry where i=0:
```
exp((-9)w + kᵢ ⊙ vᵢ + exp(u+kₜ) ⊙ vₜ)   
```
And `w` is a vector so this will scale each value in the vector by -9. And that
will then be added to the kₜvector.

### Channel-mixing
Now, this is about mixing information from different features (channels) within
a single token. So this is dealing with our features/channels (the different
dimensions in the token embedding vector). 

```
r'₁ = Wᵣ* (μ'ᵣ⊙ xᵣ+ (1 - μ'ᵣ) ⊙  xₜ-1)
k'₁ = Wₖ* (μ'ₖ⊙ xᵣ+ (1 - μ'ₖ) ⊙  xₜ-1)
```
Note that we don't have the V vector here, and I am not sure why that is?

The model performs computations using linear projections of inputs, essentially
transforming the inputs into a space where they can be more effectively analyzed
and combined. 

