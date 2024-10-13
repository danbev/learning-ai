### Mixture of Experts (MoE)
In this setup/architecture we have a number of neural networks, the experts,
which are trained together with a gating network.

### Gating Network
The gating network is not a transformer neural network like the experts but it
does take the same input, that is the input embeddings. The difference is that
the output of the gating network is the probability of which experts are most
suitable for the particular input.

Initially during training the weights of the gating network are random and the
backpropagation will adjust the weights that eventually the most suitable
experts are selected for the input (because they reduce the loss the most). So
a gating network might just be a simple linear or non linear layer followed by
a softmax.

And it is trained at the same time as the experts, or at least with the experts
in the system as it needs to calculate the loss of the outputs of the experts so
that its weights can be update accordingly.

So if we have 6 experts the output of the routing layer would be a vector of
six dimensions where each dimension represents the probability of the expert
being selected.

### Transformers and MoE
In transformers the experts are often the feed-forward layers. So we would have
the attention layer and normally followed by a feed-forward layer that operates
on all the tokens (rows of the output matrix from the attention layer), from the
output of the attention layer. This is called a dense feed-forward layer because
the layer handles all the tokens in the sequence.

So lets say we have an input token length (the input sequence as embeddings)
token length 4, embedding dimension 4096, and a hidden dimension of size 11008:
```
     0        4095
     [   ...    ]      // token embedding 1
 H = [   ...    ]      // token embedding 2
     [   ...    ]      // token embedding 3
     [   ...    ]      // token embedding 4
```
Each row of the matrix H is a token embedding. So the feed-forward layer will
operate on one of these at a time. 

So it will take this matrix and expand it from [1, 4096] x [4096, 11008] = [1, 11008],
perform the non-linear operation and then reduce it back to [1, 4096].

Now, 11008 specifies the number of dimensions, and each dimension holds a
value (parameter/weight). The size of this value can be Float32, Float16, or
a quantized value (like 8, 4, 2 bits).
So if we calculate the number of parameters as in 4096x11008 = 45,056,768. And
if we assume that the values are Float32 then each value will be represented by
4 bytes.

And we also have the matrix that reduces the dimensionality back to 4096. So
that is 11008x4096 = 45,056,768. So the total number of parameters in this
feed-forward layer is:
```
4096x11008 = 45,056,768
11008x4096 = 45,056,768
45,056,768 + 45,056,768 = 90,113,536
90,113,536 x 4 = 360,454,144 bytes
360,454,144 / 1024*1024 = 343.75MB
```

I hadn't thought about this before but in the feed-forward layer the tokens in
the sequence are passed through as individual tokens and they don't "interact"
with other tokens in the sequence. So these operations could be different
feed-forward layers which could also be on different machines to distribute the
operations and memory requirements.

And if we recall from the [transformer](./architectures/transformer.md) document
the feed-forward layer consists of an operation that expands the dimensionality of
the input tokens and then performs a non-linear operation on the expanded
tokens, and then reduces their embedding back to the original.

Now, within a transformer architecture we can have a mixture of experts which is
a layer after the attention layer. The first thing in this layer will be the
gating layer which will calculate the probability of each experts, where experts
are just feed-forward networks/layers as described above.

Interestingly these feed-forward layers will become experts on things like
punctuation, verbs, nouns, etc. So a token that is a verb will be routed to the
expert that is an expert on verbs and so on. Compared to a dense feed-forward
layer these feed-forward laysers are sparse in that not all of them will process
the vector of embeddings.

In the following from the Mixtral of Experts paper x would be a vector from the
matrix output of the attention layer:
```
n-1
Σ G(x)ᵢ * Eᵢ(x)
i=0

n = number of experts.
```
G(x) will return a vector with weights, one for each expert and this is often
a sparse vector so most entries will be zero. Eᵢ(x) is the output of the expert
Eᵢ for the input x. Now, this looks like all experts process the input and that
is what the math says but in reality a short-curcuiting mechanism is used so
if the weight for an expert is zero then the expert is not processed.

```
G(x) := softmax(TopK(x * W_g)
```
`W_g` is the weight matrix for the gating network, which is learned during
training. TopK selects the top K scores from the vector `x * W_g`, where K is a
configurable hyperparameter that determines how many experts should be actively
used for each token. These top K scores are then typically passed through a
softmax function to create a probability distribution over the selected
experts.

### Mixture of Experts in llama.cpp
In this section we will be looking at the compute graph that is built up
in llama.cpp for Mixtral 8x7b Model of Experts. I'm not sure I'll be able to
actually run this but it would be enough to inspect the building of the
graphs for get a better understanding of how the model.

```console
lldb ./build/bin/llama-cli -- -m models/mixtral-8x7b-v0.1.Q2_K.gguf --no-warmup -ngl 10 -p "What is LoRA?" -n 10
(lldb) br set -f llama.cpp -l 10594

```
Lets start in `build_llama` and the block for MoE):

So just to recap we first have the standard self-attention which is usually followed
by a feed-forward layer. In the MoE architecture the feed-forward layer is replaced
by a mixture of experts.

So this starts with a normalization layer using RMSNorm which follows the self-attention
layer.
```c++
                // MoE branch
                cur = llm_build_norm(ctx0, ffn_inp, hparams,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, cb, il);
                cb(cur, "ffn_norm", il);
```
We can inspect the shape for the current tensor:
```console
(lldb) p cur->name
(char[64]) "ffn_norm-0"
(lldb) p cur->ne
(int64_t[4])  ([0] = 4096, [1] = 512, [2] = 1, [3] = 1)
```
So we have 512 rows (token embeddings) and 4096 columns/dimensions (embedding dim).
This will be passed into the gate network:
```console
                cur = llm_build_moe_ffn(ctx0, lctx, cur,
                        model.layers[il].ffn_gate_inp,
                        model.layers[il].ffn_up_exps,
                        model.layers[il].ffn_gate_exps,
                        model.layers[il].ffn_down_exps,
                        n_expert, n_expert_used,
                        LLM_FFN_SILU, true,
                        false, 0.0,
                        cb, il);
                cb(cur, "ffn_moe_out", il);
```

```c++
static struct ggml_tensor * llm_build_moe_ffn(
        struct ggml_context * ctx,
       struct llama_context & lctx,
         struct ggml_tensor * cur,
         struct ggml_tensor * gate_inp,
         struct ggml_tensor * up_exps,
         struct ggml_tensor * gate_exps,
         struct ggml_tensor * down_exps,
                    int64_t   n_expert,
                    int64_t   n_expert_used,
            llm_ffn_op_type   type_op,
                       bool   norm_w,
                       bool   scale_w,
                      float   w_scale,
         const llm_build_cb & cb,
                        int   il) {
    int64_t n_embd = cur->ne[0];
    int64_t n_tokens = cur->ne[1];

    ggml_tensor * logits = llm_build_lora_mm(lctx, ctx, gate_inp, cur); // [n_expert, n_tokens]
    cb(logits, "ffn_moe_logits", il);
```
Lets take a look at the `gate_inp` tensor:
```console
(lldb) p gate_inp->ne
(int64_t[4])  ([0] = 4096, [1] = 8, [2] = 1, [3] = 1)

(lldb) p cur->ne
(int64_t[4])  ([0] = 4096, [1] = 512, [2] = 1, [3] = 1)
```
This learned matrix `gate_inp`  will be multiplied by the current tensor above, and recall that
in ggml the second tenors of a matrix multiplication are transposed:
```
    ffn_gate_inp           cur

0  [0 ... 4095]       0   [0 ... 511]     0  [0 ... 511]
      ...                   ...
      ...         x         ...       =
7  [0 ... 4095]                           7  [0 ... 511]
                     4095 [0 ... 511]
```
So this will produce a 8x512 matrix. We have 8 experts and 512 token embeddings which indicate
how suitable each expert is for handling each token.

This tensor will then be passed through a softmax function to produce a probability
distribution over the experts:
```c++
    ggml_tensor * probs = ggml_soft_max(ctx, logits);
    cb(probs, "ffn_moe_probs", il);
```
Then we have the actualy selection of the expert(s):
```c++
    // select experts
    ggml_tensor * selected_experts = ggml_top_k(ctx, probs, n_expert_used);
```
```console
(lldb) p n_expert_used
(int64_t) 2
```
So that will produce a 2x512 matrix:
```console
(lldb) p selected_experts->ne
(int64_t[4])  ([0] = 2, [1] = 512, [2] = 1, [3] = 1)
```
```
 0    [0 .. 1] 
 ...
 
 511  [0 .. 1]
```
So this will have a row for each token and we have two experts that will be used, to the
values will be the index of the experts selected.


Next we have the following callback invocations, and I'm a little curious about the 
first naming of the src tensor setting it to `ffn_moe_argsort` instead of using `ffn_moe_propbs`:
```c++
    cb(selected_experts->src[0], "ffn_moe_argsort", il);
    cb(selected_experts, "ffn_moe_topk", il);
```
I was a little confused about the `src[0]` tensor and the setting of the name to `ffn_moe_argsort`
as I though that it would have the name of the `probs` tensor. But if we look at the `ggml_top_k`
we can see the following:
```c
// ggml_top_k

struct ggml_tensor * ggml_top_k(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   k) {
    GGML_ASSERT(a->ne[0] >= k);

    struct ggml_tensor * result = ggml_argsort(ctx, a, GGML_SORT_ORDER_DESC);

    result = ggml_view_4d(ctx, result,
                k, result->ne[1], result->ne[2], result->ne[3],
                   result->nb[1], result->nb[2], result->nb[3],
                0);

    return result;
}
```
And I think this makes sense as `top_k` will select the top k values and having a sorting
operation makes sense.

Next we will extract the propbabilites from the `probs` tensor using `ggml_get_rows` but 
first there is a reshaping:
```console
(lldb) p probs->ne
(int64_t[4])  ([0] = 8, [1] = 512, [2] = 1, [3] = 1)

(lldb) p ggml_reshape_3d(ctx, probs, 1, n_expert, n_tokens)->ne
(int64_t[4])  ([0] = 1, [1] = 8, [2] = 512, [3] = 1)
```
We can try to visualize this:
```
x = 1, y = 8, z = 512

z0
    y0    [0]
    y1    [0]
    y2    [0]
    y3    [0]
    y4    [0]
    y5    [0]
    y6    [0]
...

z511
    y0    [0]
    y1    [0]
    y2    [0]
    y3    [0]
    y4    [0]
    y5    [0]
    y6    [0]
```
Now, `selected_experts` has a shape of:
```console
(lldb) p selected_experts->ne
(int64_t[4])  ([0] = 2, [1] = 512, [2] = 1, [3] = 1)
```
```
   selected_experts
   0    [0...1]
   ...
   511  [0...1]

```
Now, each row in `selected_experts` will have two columns one for each expert selected for
that token.
For each of the 512 tokens in `selected_experts` this will look up the two values in each row
and use them as indices into the reshaped `probs` tensor to extract the probabilities for the
So for token 0 in `selected_experts` take the two values in that row (lets say the values are 3
and 7), and look those rows up in `reshaped_probs` and extract the values at those rows (only one
row each of 512 dimensions (the z dimension).

```c++
    ggml_tensor * weights = ggml_get_rows(ctx,
            ggml_reshape_3d(ctx, probs, 1, n_expert, n_tokens), selected_experts);
    cb(weights, "ffn_moe_weights", il);
```
The resulting tensor will have the following shape:
```console
(lldb) p weights->ne
(int64_t[4])  ([0] = 1, [1] = 2, [2] = 512, [3] = 1)
```
We can visualize this as follows:
```
x = 1, y = 2, z = 512

z0
    y0    [0]
    y1    [1]
...
z511
    y0    [0]
    y1    [1]
```
So each row is an expert and there is one value for each export per token. Where the value
is probability of the expert being selected for that specific token. This probability is
later used to calculate the weighted sum of the expert outputs. So the output of the expert
that has a higher probability will have a higher weight in the weighted sum.

Next, in the function is the following:
```c++
    if (norm_w) {
        weights = ggml_reshape_2d(ctx, weights, n_expert_used, n_tokens);

        ggml_tensor * weights_sum = ggml_sum_rows(ctx, weights); // [1, n_tokens]
        cb(weights_sum, "ffn_moe_weights_sum", il);

        weights = ggml_div(ctx, weights, weights_sum); // [n_expert_used, n_tokens]
        cb(weights, "ffn_moe_weights_norm", il);

        weights = ggml_reshape_3d(ctx, weights, 1, n_expert_used, n_tokens);
    }
```
And `norm_w` is true in this case to this block will be executed. We can see that the `weights`
are reshaped into a 2d tensor 2x512, which are then summed and the the `weights` tensor is divided
by the sum to normalize the weights. The tensor is then reshaped back to 3d. So this block will simply
normalize the weights for each token.

Next we have the following:
```c++
    cur = ggml_reshape_3d(ctx, cur, n_embd, 1, n_tokens);
```
Notice that is reshaping cur which is currently:
```console
(lldb) p cur->ne
(int64_t[4])  ([0] = 4096, [1] = 512, [2] = 1, [3] = 1)
(lldb) p cur->name
(char[64]) "ffn_norm-0"
(lldb) p n_embd
(int64_t) 4096
(lldb) p n_tokens
(int64_t) 512
```
So this cur will become:
```console
(lldb) p cur->ne
(int64_t[4])  ([0] = 4096, [1] = 1, [2] = 512, [3] = 1)

z0
 [1]  [0 ... 4095]
...

z511
 [1]  [0 ... 4095]
```
Following that we have the operation:
```c++
    ggml_tensor * up = llm_build_lora_mm_id(lctx, ctx, up_exps, cur, selected_experts);
    cb(up, "ffn_moe_up", il);
```
We can inspect `up_exps`:
```console
(lldb) p up_exps->ne
(int64_t[4])  ([0] = 4096, [1] = 14336, [2] = 8, [3] = 1)
y0
  0     [0 ... 4095]
  ...
  14335 [0 ... 4095]

...

y7
  0     [0 ... 4095]
  ...
  14335 [0 ... 4095]
```
Notice that the number of dimensions in y is `14336` which is the same as the number of dimensions
of the inner dimension for the feed forward layer of this model:
```console
llama.feed_forward_length
(lldb) p lctx.model.gguf_kv
(const std::unordered_map<std::string, std::string>) size=22 {
  ...
  [15] = {
    __cc_ = (first = "llama.feed_forward_length", second = "14336")
  }
}
```
So the first tensor argument to `llm_build_lora_mm_id` is the experts tensor matrices. So we have
one 4096x14336 metric for each expert. The second argument is the input, the `cur` tensor which is
```console
(lldb) p cur->ne
(int64_t[4])  ([0] = 4096, [1] = 1, [2] = 512, [3] = 1)
```
And the last argument is the tensor that specifies which expert to use for each token.
This operation will result in a tensor with the following shape:
```console
(lldb) p up->ne
(int64_t[4])  ([0] = 14336, [1] = 2, [2] = 512, [3] = 1)
```
Just note that this up tensor is not used until later but the `cur` tensor is updated so this
operation is performed as this stage (is built).

So this is increasing the dimensionality which is now 14336.
Following that we have another `llm_build_lora_mm_id` operation:
```c++
    ggml_tensor * gate = llm_build_lora_mm_id(lctx, ctx, gate_exps, cur, selected_experts);
    cb(gate, "ffn_moe_gate", il);
```
The gate is what will contain the logits for the two selected experts.
```console
(lldb) p gate->ne
(int64_t[4])  ([0] = 14336, [1] = 2, [2] = 512, [3] = 1)
```
There is a standalone example of the `ggml_mul_mat_id` which is what `llm_build_lora_mm_id`
calls in [mul-mat-id.cpp](./../fundamentals/ggml/src/mul-mat-id.c).

Next we have a SILU operation which recall we would have something similar in a normal
feed-forward layer as well:
```c++
    switch (type_op) {
        case LLM_FFN_SILU:
            {
                gate = ggml_silu(ctx, gate);
                cb(gate, "ffn_moe_silu", il);
            } break;
```
And next is where we use the `up` tensor from above and this is element-wise multiplied
with the gate tensor which was passed through the SILU operation:
```c++
    ggml_tensor * par = ggml_mul(ctx, up, gate);
    cb(par, "ffn_moe_gate_par", il);
```
Notice that this is an element-wise multiplication and not a matrix multiplication.

Next, we perform the down
```c++
    ggml_tensor * experts = llm_build_lora_mm_id(lctx, ctx, down_exps, par, selected_experts);
    cb(experts, "ffn_moe_down", il);
```

This is followed by a elememt-wise multiplication of the weights for each expert:
```c++
    experts = ggml_mul(ctx, experts, weights);
```
This will scale the results according to the weights for each expert. So if one of the experts
has a higher weight then the output of that expert will have a higher weight in the final output.

Next the output of the experts are added:
```c++
    // aggregate experts
    ggml_tensor * moe_out = nullptr;
    for (int i = 0; i < n_expert_used; ++i) {
        ggml_tensor * cur_expert = ggml_view_2d(ctx, experts, n_embd, n_tokens,
                experts->nb[2], i*experts->nb[1]);

        if (i == 0) {
            moe_out = cur_expert;
        } else {
            moe_out = ggml_add(ctx, moe_out, cur_expert);
        }
    }

    if (n_expert_used == 1) {
        // avoid returning a non-contiguous tensor
        moe_out = ggml_cont(ctx, moe_out);
    }

    return moe_out;
```
