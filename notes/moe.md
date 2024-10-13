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

__wip__

```c++
    ggml_tensor * weights = ggml_get_rows(ctx,
            ggml_reshape_3d(ctx, probs, 1, n_expert, n_tokens), selected_experts);
    cb(weights, "ffn_moe_weights", il);
```

