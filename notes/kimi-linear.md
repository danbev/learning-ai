## Kimi Linear

```c++
        case LLM_ARCH_KIMI_LINEAR:
            {
                llm = std::make_unique<llm_build_kimi_linear>(*this, params);
            } break;
```
And we can find the implementation in src/models/kimi-linear.cpp.

### Kimi Delta Attention (KDA)
This builds upon [mla](./mla.md) and the "Delta" part is what makes Kimi Linear
special. In older linear attention models, the memory matrix just kept adding
new information until it became a "blurry" mess of data.

KDA uses a Data-Dependent Update (The Delta Rule). Before adding new information
to the memory, it calculates how much of the existing memory is already similar
to the new input.

```
S_t = S_{t-1} + beta_t(v_t - S_{t-1} q_t) k_t

Where:
- S_t is the updated memory/state at time t
- S_{t-1} is the previous memory/state
- beta_t is a scalar (between 0 and 1) which acts like a write enable write enable
         switch. Similar to a forget gate in an LSTM.
- v_t is the value vector at time t (the current input's value representation)
- q_t is the query vector at time t (the current input's query representation)
- k_t is the key vector at time t   (the current input's key representation)
```
So this first multiplying the previous state (s_{t-1}) with the query (q_t) which
is the models prediction. S_{t-1} is everything that the models knows this far,
and q_t is the current query. By multiplying them together, we get a measure of
how much of the current query is already captured in the memory. This is the "similarity" or "overlap"

So we have S_{t-1} which is the current state which is like a key-value lookup
table that has been squashed into a single grid. But how can we "look up"
something in this grid, like we can just use [row][column] right. Instead the
"indices" are directions of the vectors.
```
      S_{t-1}        Q_t
 0  [0  ...   d]     [0]      [0]
    [0  ...   d]     [0]    = [0]
    [0  ...   d]     [0]      [0]
 d  [0  ...   d]     [0]      [0]
```
Lets say we have a d of size 4, and a sequence "The color is Red".
```
k = concept of the Color represented by a vector: [1 0 0 0] 
v = concept of Red represented by a vector      : [5 0 0 0]
```

In linear attention state a "key-value" pair is stored as an outer product of
v X k^T.  So looking at the above equation:
```
S_t = S_{t-1} + beta_t(v_t - S_{t-1} q_t) k_t
```
Lets start with the inner most expression (v_t - S_{t-1} q_t) where we perform
the multiplication of the current state with the query vector:
```

          [5 0 0 0]    [1]
S_{t-1}   [0 0 0 0] x  [0]
          [0 0 0 0]    [0]
          [0 0 0 0]    [0]
```
If we were to ask for the feature of index (0,0) the answer will be 5.
Now, imaging a new sequence "What is the color?" comes in. The query vector for
color could be [1 0 0 0] which is the same as the key vector for color.
```
          [5 0 0 0]  [1]    [5]
          [0 0 0 0]  [0]  = [0]
          [0 0 0 0]  [0]    [0]
          [0 0 0 0]  [0]    [0]
```
And notice that the resulting vector is our representation for "Red", so the
model was able to look it up.
Now, this is where the delta part comes into play. Imaging that the sequence
"The color is now Blue" comes in: 
```
v_t = "Blue" [0 7 0 0]
```
The model will has "Red" as its color (the feature in the state [5 0 0 0]).
The subtraction in (v_t - S_{t-1} q_t):
```
   [0]   [5]   [-5]   // delete Red
   [7] - [0] = [ 7]   // add Blue
   [0]   [0]   [ 0]
   [0]   [0]   [ 0]
```
This is the Delta part.

Beta is a vector of scalar values (0-1) that are often or perhaps always data
dependent. It could be computed as:
```
Beta_t = sigmoid(W_b x_t)
```
Assuming beta is 1 for simplicity, so this gives the model a way to gate, either
write-through (1) or not (0) the update for each feature. So this is an element
wise multiplication:
```
beta = [1 1 1 1] 

  [1]    [-5]     [-5]
  [1] *  [ 7]  =  [ 7]
  [1]    [ 0]     [ 0]
  [1]    [ 0]     [ 0]
```
So this enables selectively updating some features while leaving others unchanged.


And then we compute the outer product with the key vector k_t:
```
(Error) x k_t (color):

   [-5]               [-5 0 0 0]
   [ 7] x [1 0 0 0] = [ 7 0 0 0]
   [ 0]               [ 0 0 0 0]
   [ 0]               [ 0 0 0 0]

x = outer product operator.
```
And the last part is the addition:
S_t = S_{t-1} + Update
```
      [5 0 0 0]   [-5 0 0 0]   [0 0 0 0]
      [0 0 0 0] + [ 7 0 0 0] = [7 0 0 0]
      [0 0 0 0]   [ 0 0 0 0]   [0 0 0 0]
      [0 0 0 0]   [ 0 0 0 0]   [0 0 0 0]
```
Notice that the Red has been cancelled out, and the Blue has been written into
memory.

In Kimi Linear, by multiplying S_{t-1} by q_t, we are doing all of those lookups
at once in a single matrix-vector multiplication. The 2D matrix S has effectively
"pre-summed" all the past information, and q_t simply extracts what is relevant
right now.


And similar to Mamba which as a fixed size state/memory this is also true for
Kimi linear.
So this is like compressing down the entire history of a sequence into a single
matrix and over time this is going to become a "blurry" mess of data. But this
model does not exclusively rely on these types of layers, but it also has MLA
layers interleaved and those layers do have KV-caches that grow over time (but
not as much as traditional attention).

And because we pass along the output from one layer as the input to the next we
are actually feeding the output for the interleaved MLA layers to the KDA layers
and therefor the KDA layers states get updated as well. So in a way the MLA layer
can look back long into the past and then the KDA layer can use that information
to update its state. It might "notice" a specific detail from 500,000 tokens ago
that the KDA layers had started to "blur."

KDA (like Mamba) uses a Parallel Scan or Convolutional Form. During training,
since we already know the whole sentence, we can actually calculate all the S_t
updates for the entire token sequence simultaneously.

### Chunking
So because KDA is recurrent if we have a sequence of 1000 tokens we would need
a loop of 1000 iterations which would be inefficient. Instead we chunk this into
64x64 chunks and process each them sequentialy.
```c++
    ggml_tensor * chunked_causal_mask =
        ggml_tri(ctx0, ggml_fill_inplace(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, CHUNK_SIZE, CHUNK_SIZE), 1.0f),
                    GGML_TRI_TYPE_LOWER);
```
This is creating a new 2d tensor with the size of CHUNK_SIZE x CHUNK_SIZE and
filling it with 1.0f. And that is then passed to ggml_tri which creates a lower
triagular matrix. So the resulting chunked_causal_mask is a 64x64 matrix that
looks like something like this (though not to scale):
```
[ 1  0  0  0 ] Token 0 sees itself
[ 1  1  0  0 ] Token 1 sees token 0 and itself
[ 1  1  1  0 ] Token 2 sees token 0, token 1, and itself
[ 1  1  1  1 ] Token 3 sees token 0, token 1, token 2, and itself

chunk size = 4
```

```c++
    ggml_tensor * chunked_identity = ggml_diag(ctx0, ggml_fill_inplace(ctx0, ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, CHUNK_SIZE), 1.0f));
```
So this is creating a 1d vector of size CHUNK_SIZE (64) and then filling it with
1.0f. This tensor is then passed to ggml_diag which turns this into an Identity
Matrix.
```
[ 1  0  0  0 ]
[ 0  1  0  0 ]
[ 0  0  1  0 ]
[ 0  0  0  1 ]

chunk size = 4
```
We then add these two matrices together:
```c++
    ggml_tensor * chunked_diag_mask = ggml_add(ctx0, chunked_causal_mask, chunked_identity);
```

```
[ 2  0  0  0 ]
[ 1  2  0  0 ]
[ 1  1  2  0 ]
[ 1  1  1  2 ]

chunk size = 4
```
So normally I would only expect to see 1s and 0s in a mask, but here we have 2s
as well. The ones usually indicate that tokens can attent to previous tokens.
In this case it has to do with the delta correction that we discussed above. But
above we were looking at a single token, but in practice we are processing chunks
of tokens at once. So instead of updating he state four times we just perform
one operation and by having having 2 on the diagonal (the tokens that attend
to them selves) we build in the delta correction, the subtraction, into the matrix.

So for token 2 we have:
```
 [1 1 2 0]
1 = history of Token 0
1 = history of Token 1
2 = 1 (history of tokens current self) + 1 (correction factor)
```
So we have one 1 that indicates that this token should attend to itself and then
we have the additional one for the subtraction, just because otherwise a 1 in the
beta vector for this position would cancel out.


For each kda layer we have:
```c++
ggml_tensor * conv_state_all = build_rs(inp_rs, conv_states_all, hparams.n_embd_r(), n_seqs);
```
So the memory for this layer is storing the tail of the previous sequence and
this is used for the convolutions that follow which allow the model to blend with
its immediate neighbors, the 3-4 tokens behind it.
```c++
ggml_tensor * Qcur = causal_conv1d(
    gf,
    ctx0,
    conv_states_all,
    conv_state_all,
    0,                       // qkv
    cur,                     // x
    layer.wq,                // proj_w
    layer.ssm_q_conv,        // conv_w
    d_conv,
    head_dim,
    n_head,
    n_seq_tokens,
    n_seqs,
    n_tokens,
    kv_head);
```
So initially I just thought that this was doing a ggml_conv_1d operation but this
is a function in the same file.

This is handling the the short term memory by mixing in the previous 3 tokens.
which is this is what conv_states_all contains. It contains the previous 3 states
for Q, K and V.
```c++
static ggml_tensor * causal_conv1d(
    ggml_cgraph * gf,
    ggml_context * ctx0,
    ggml_tensor * conv_states_all,
    ggml_tensor * conv_state_all,
    int64_t qkv,
    ggml_tensor * x,
    ggml_tensor * proj_w,
    ggml_tensor * conv_w,
    int64_t d_conv,
    int64_t head_dim,
    int64_t n_head,
    int64_t n_seq_tokens,
    int64_t n_seqs,
    int64_t n_tokens,
    int64_t kv_head) {
```
And above we are passing in 0 as the value for `qkv` which is acting/used like
an index/offset into the state.

So first we createa a view into the conv_states_all tensor which is a 3d tensor
that contains the previous 3 states for Q, K and V and in this case the offset
is  (qkv):
```c++
    ggml_tensor * conv_state_x = ggml_view_3d(ctx0, conv_state_all, d_conv - 1, d_inner, n_seqs,
        (d_conv - 1) * ggml_element_size(conv_state_all),  // nb1: stride between channels
        n_embd_r_total * ggml_element_size(conv_state_all),  // nb2: stride between seqs
        qkv * conv_state_size * ggml_element_size(conv_state_all));
```
```console
(gdb) p conv_state_x->ne
$3 = {3, 4096, 1, 1}
```
```console
(gdb) p x->ne
$2 = {2304, 1, 1, 1}

(gdb) p proj_w->ne
$5 = {2304, 4096, 1, 1}
```
The we project the cur tensor (x) to a higher dimension:
```c++
    ggml_tensor * x_proj = ggml_mul_mat(ctx0, proj_w, x);
```
```console
(gdb) p x_proj->ne
$4 = {4096, 1, 1, 1}
```

Shorlty after we have the ssm_conv operation:
```c++
    ggml_tensor * conv_weight = ggml_reshape_2d(ctx0, conv_w, d_conv, d_inner);

    // Apply conv1d
    // ggml_ssm_conv output: {d_inner, n_seq_tokens, n_seqs}
    ggml_tensor * Xcur = ggml_ssm_conv(ctx0, conv_x, conv_weight);
```
The kernel is conv_weight which is specific to each layer and conv_x is what
the kernel will convolve over. So this is enabling the model to blend the past
3 token featurs together with the current token embedding being processed. This
creates a form of context for the tokens.

Now, this is actually one part of the reason that Kimi Linear does not need
positional encoding like RoPE. Because the kernel is a vector [w_3, w_2, w_1, w_0]
it treats the token that just arrived, t_0, differently than the token that
arrived 3 steps ago, t_3. So the model can distinguish between "Dog bites" and
"Bites dog". And with standard attention there is no such distinction without
positional encoding. But the other part of this is the KDA recurrence which acts
as a sequential chain.

And the above is also done for the K and V tensors as well.

Following that we have the forget gate:
```c++
            // this is a down projection to a latent state
            ggml_tensor * f_a = ggml_mul_mat(ctx0, layer.ssm_f_a, cur);
```
```console
(gdb) p cur->ne
$16 = {2304, 512, 1, 1}
(gdb) p layer.ssm_f_a->ne
$17 = {2304, 128, 1, 1}

(gdb) p f_a->ne
$18 = {128, 512, 1, 1}
```
```c++
            ggml_tensor * g1 = ggml_mul_mat(ctx0, layer.ssm_f_b, f_a);
```
```console
(gdb) p layer.ssm_f_b->ne
$19 = {128, 4096, 1, 1}

(gdb) p g1->ne
$20 = {4096, 512, 1, 1}
```
So the above down projection followed by the up project is like a low-rank
bottleneck that is done to avoid the full large matrix multiplication.

So the g1 (gate) holds the raw instructions for how memory/state should be
updated. But the raw output from the above operations can be any number, positive
negative, or zero. To be useful as a forget gate we need to have them in a stable
usable range.
ssm_dt_b is a learned vector of 4096 numbers that acts as the "baseline forget rate"
and this is applied to the calculated gate (which becomes like an offset from the
base):
```c++
            g1 = ggml_add(ctx0, g1, layer.ssm_dt_b);
            g1 = ggml_softplus(ctx0, g1);
```
Softplus is used because we need stictly positive values for the forget gate. It
is like a forward only update (timestep).
After these operations the g1 is a vector of step sizes, a high value in g1 means
that this channel/feature will update very rapidly (forgetting past information
in favour of immediate input) and a low value means that this channel/feature
will update very slowly, it will hold on to past information for a long time.
Next, we will use the g1 tensor and multiply it by layer.ssm_a
```c++
            ggml_tensor * A = ggml_reshape_3d(ctx0, layer.ssm_a, 1, n_head, 1);
            g1 = ggml_mul(ctx0, g1, A);
```
```console
(gdb) p A->ne
$24 = {1, 32, 1, 1}
(gdb) p g1->ne
$25 = {128, 32, 512, 1}

(gdb) p g1->ne
$26 = {128, 32, 512, 1}
```
layer.ssm_a is a static parameter learned during training and represents a
base forget rate. The previous part we say above is a dynamic value (it changes
for every token).
So head_1 might have a very small value for A which would be a long-term memory
head. It will rememember things for hundres of thousands of tokens.
And head_32 might have a very high value for A which would be a short-term
memory head. It might clears its grid almost immediately and only remember the
last few tokens.
This is a reason that Kimi-linear doesn't need RoPE, because the the "strenght"
of the memory in the 2D grid naturally tells the model how far back in time that
token is from.

Next we have:
```c++
            ggml_tensor * beta = ggml_mul_mat(ctx0, layer.ssm_beta, cur);
```
And this is the "write enable switch" that we discussed above. While the
`f_a -> f_b -> g_1` path controls the "Drain" (forgetting), this line calculates
the Input Gate, the "Tap" that controls how much of the current token actually
gets written into the memory. And this is also data dependent
```console
(gdb) p beta->ne
$27 = {32, 512, 1, 1
```
Recall that we stored the previous tokens in r_l and here we are getting the
states from s_l:
```c++
            ggml_tensor * ssm_states_all = mctx_cur->get_s_l(il);
            ggml_tensor * state = build_rs(inp_rs, ssm_states_all, hparams.n_embd_s(), n_seqs);
            state = ggml_reshape_4d(ctx0, state, head_dim, head_dim, n_head, n_seqs);
```
And depending on if we are processing a single token in the current batch we will
either add model operations for build_kda_autoregressive or build_kda_chunking:
```c++
            // Choose between build_kda_chunking and build_kda_recurrent based on n_tokens
            std::pair<ggml_tensor *, ggml_tensor *> attn_out = n_seq_tokens == 1 ?
                build_kda_autoregressive(Qcur, Kcur, Vcur, g1, beta, state, il) :
                build_kda_chunking(Qcur, Kcur, Vcur, g1, beta, state, chunked_causal_mask, chunked_identity, chunked_diag_mask, il);
```

So chunking is used when we have more than 1 token and because KDA is recurrent
it would need to loop through each token one at a time which would be inefficient.
We don't have this in standard attention because it is not recurrent and can
process the whole sequence at once (it also has RoPE to handle the positions).
But with KDA we have to process the sequence. This is where chunking comes in.
In a standard Transformer, there is no "state" that carries over from token to
token. Every token looks at every previous token simultaneously, so we can
processes a massive NxN attention matrix all at once. But with KDA, we have this
state that gets updated sequentially.

Purely sequential processing (O(N)) is a nightmare for modern GPUs, which want
to do thousands of things at once. Instead of processing 1000 tokens one at a
time we break them into chunks of 64 (CHUNK_SIZE) tokens each. So how can this
be done with a single operation on a chunk?  
This is done by using ggml_solve_tri.

```c++
/*
    This is a ggml implementation of the naive_chunk_kda function of
    https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/naive.py
*/
std::pair<ggml_tensor *, ggml_tensor *> llm_build_kimi_linear::build_kda_chunking(
        ggml_tensor * q,            // Qcur
        ggml_tensor * k,            // Kcur
        ggml_tensor * v,            // Vcur
        ggml_tensor * gk,           // g1 (the forget gate)
        ggml_tensor * beta,         // beta (the write enable switch)
        ggml_tensor * state,        // state (the memory/state that gets updated)
        ggml_tensor * causal_mask,  // chunked_causal_mask
        ggml_tensor * identity,     // chunked_identity
        ggml_tensor * diag_mask,    // chunked_diag_mask
        int           il) {

    const int64_t S_k      = q->ne[0];
    const int64_t H_k      = q->ne[1];
    const int64_t n_tokens = q->ne[2];
    const int64_t n_seqs   = q->ne[3];

    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];
```
```console
(gdb) p S_k
$1 = 128
(gdb) p H_k
$2 = 32
(gdb) p n_tokens
$3 = 512
(gdb) p n_seqs
$4 = 1
(gdb) p S_v
$5 = 128
(gdb) p H_v
$6 = 32

(gdb) p q->ne
$7 = {128, 32, 512, 1}
(gdb) p k->ne
$8 = {128, 32, 512, 1}
(gdb) p v->ne
$9 = {128, 32, 512, 1}
(gdb) p gk->ne
$10 = {128, 32, 512, 1}
(gdb) p beta->ne
$11 = {32, 1, 512, 1}
(gdb) p causal_mask->ne
$12 = {64, 64, 1, 1}
(gdb) p diag_mask->ne
$13 = {64, 64, 1, 1}
```
First we normalize q and k:
```c++
    const bool use_qk_l2norm = true;

    if (use_qk_l2norm) {
        const float eps_norm = hparams.f_norm_rms_eps;

        q = ggml_l2_norm(ctx0, q, eps_norm);
        k = ggml_l2_norm(ctx0, k, eps_norm);
    }
```
Then beta is passed through a sigmoid to ensure that it is between 0 and 1, and
recall that beta is the "write enable switch" that controls how much of the current token
gets written into the state:
```c++
    const float scale = 1.0f / sqrtf(S_v);

    beta = ggml_sigmoid(ctx0, beta);
```
Next, all the following tensors are permuted swapping the second and third
dimensions and then reshaping them into 4D tensors:
```c++
    q  = ggml_cont_4d(ctx0, ggml_permute(ctx0,  q, 0, 2, 1, 3), S_k, n_tokens, H_k, n_seqs);
    k  = ggml_cont_4d(ctx0, ggml_permute(ctx0,  k, 0, 2, 1, 3), S_k, n_tokens, H_k, n_seqs);
    v  = ggml_cont_4d(ctx0, ggml_permute(ctx0,  v, 0, 2, 1, 3), S_v, n_tokens, H_v, n_seqs);
    gk = ggml_cont_4d(ctx0, ggml_permute(ctx0, gk, 0, 2, 1, 3), S_v, n_tokens, H_v, n_seqs);
```
So we first permute the tensor which only manipulates the ne[] and nb[] arrays,
so memory is unchanged but the strides and dimensions are changed.
```console
(gdb) p q->ne
$7 =  {128, 32, 512, 1}
(gdb) p ggml_permute(ctx0, q, 0, 2, 1, 3)->ne
$16 = {128, 512, 32, 1}
```
And because the strides are changes we use ggml_cont_4d which will create a new
tensor and copy and reorder the elements in memory so that they are contiguous.
This is a requirement of the kernel (I think) so that it can efficiently access
the data in memory (coalesced memory access).

And after the ggml_cont_4d operation:
```console
(gdb) p q->ne
$17 = {128, 512, 32, 1}
```
This is because we only have one sequance, n_seqs is 1.
The we do the same with the beta tensor:
```c++
    beta  = ggml_cont(ctx0, ggml_permute(ctx0, beta, 2, 0, 1, 3));
```
Next we reshape the state:
```c++
    state = ggml_reshape_4d(ctx0, state, S_v, S_v, H_v, n_seqs);
```
Next we have padding in case we don't have a multiple of the chunk size (64)
we pad the sequence:
```c++
    // Do padding
    const int64_t chunk_size = CHUNK_SIZE;

    const int64_t pad = (chunk_size - n_tokens % chunk_size) % chunk_size;
    const int64_t n_chunks = (n_tokens + pad) / chunk_size;

    q = ggml_pad(ctx0, q, 0, pad, 0, 0);
    k = ggml_pad(ctx0, k, 0, pad, 0, 0);
    v = ggml_pad(ctx0, v, 0, pad, 0, 0);
    gk = ggml_pad(ctx0, gk, 0, pad, 0, 0);
    beta = ggml_pad(ctx0, beta, 0, pad, 0, 0);
```
```console
(gdb) p (chunk_size - n_tokens % chunk_size) % chunk_size
$22 = 0
(gdb) p 512 / 64
$23 = 8
```
Next we apply the beta gating:
```c++
    ggml_tensor * v_beta = ggml_mul(ctx0, v, beta);
    ggml_tensor * k_beta = ggml_mul(ctx0, k, beta);
```

```c++
    const int64_t HB = H_k * n_seqs;

    q      = ggml_cont_4d(ctx0, q,      S_k, chunk_size, n_chunks, HB);
    k      = ggml_cont_4d(ctx0, k,      S_k, chunk_size, n_chunks, HB);
    k_beta = ggml_cont_4d(ctx0, k_beta, S_k, chunk_size, n_chunks, HB);
    v      = ggml_cont_4d(ctx0, v,      S_v, chunk_size, n_chunks, HB);
    v_beta = ggml_cont_4d(ctx0, v_beta, S_v, chunk_size, n_chunks, HB);

    gk    = ggml_cont_4d(ctx0, gk, S_k, chunk_size, n_chunks, HB);
    beta = ggml_cont_4d(ctx0, beta, 1, chunk_size, n_chunks, HB);

    // switch for cumsum
    gk = ggml_cont_4d(ctx0, ggml_permute(ctx0, gk, 1, 0, 2, 3), chunk_size, S_k, n_chunks, HB);
    cb(gk, "gk", il);
    ggml_tensor * gk_cumsum = ggml_cumsum(ctx0, gk);
    cb(gk_cumsum, "gk_cumsum", il);
```
