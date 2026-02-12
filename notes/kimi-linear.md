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
ggml_tensor * Qcur = causal_conv1d(gf,
    ctx0,
    conv_states_all,
    conv_state_all,
    0,
    cur,
    layer.wq,
    layer.ssm_q_conv,
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
