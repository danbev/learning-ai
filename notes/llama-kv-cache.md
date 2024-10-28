o# llama.cpp kv-cache notes

### Overview
I've gone through the theory of Key-Value caching in the transformer architecture
in [llama.md](llama.md). This document is a more detailed look at the
implementation of the key-value cache in the llama.cpp codebase.

### Inference with KV-Cache
Lets set a break point before `llama_decode` and see how this interacts with
the kv-cache.
```console
$ cd fundamentals/llama.cpp
$ make simple_prompt
$ gdb --args simple_prompt
(gdb) br llama_decode_internal
```

In `llama_decode_internal` we find the following:
```c++
        // non-causal masks do not use the KV cache
        if (hparams.causal_attn) {
            llama_kv_cache_update(&lctx);

            // if we have enough unused cells before the current head ->
            //   better to start searching from the beginning of the cache, hoping to fill it
            if (kv_self.head > kv_self.used + 2*n_tokens) {
                kv_self.head = 0;
            }

            if (!llama_kv_cache_find_slot(kv_self, u_batch)) {
                return 1;
            }

            if (!kv_self.recurrent) {
                // a heuristic, to avoid attending the full cache if it is not yet utilized
                // after enough generations, the benefit from this heuristic disappears
                // if we start defragmenting the cache, the benefit from this will be more important
                const uint32_t pad = llama_kv_cache_get_padding(cparams);
                kv_self.n = std::min(kv_self.size, std::max(pad, GGML_PAD(llama_kv_cache_cell_max(kv_self), pad)));
                //kv_self.n = llama_kv_cache_cell_max(kv_self);
            }
        }
``` 
The first call is to `llama_kv_cache_update` which actually does not do anything
is our case. But this checks the `has_shift` of the cache and would perform
apply a shift of the keys if that had been set, which is done by the add/div
functions.
TODO: Incorporate notes from self-extend which I think went through this process
in some detail.

At this stage the cache is empty and head is zero so lets look at find slot:
(in this case `n_tokens` is 6 and cache size is 1024)
```c++
static bool llama_kv_cache_find_slot(
           struct llama_kv_cache & cache,
        const struct llama_batch & batch) { <--- I thought I'd renamed these to ubatch?
    const uint32_t n_tokens = batch.n_tokens;

    if (cache.recurrent) {
        ...
    }

    if (n_tokens > cache.size) {
        LLAMA_LOG_ERROR("%s: n_tokens=%d > cache.size=%d\n", __func__, n_tokens, cache.size);
        return false;
    }

    uint32_t n_tested = 0;
    while (true) {
        if (cache.head + n_tokens > cache.size) {
            n_tested += cache.size - cache.head;
            cache.head = 0;
            continue;
        }

        bool found = true;
        for (uint32_t i = 0; i < n_tokens; i++) {
            if (cache.cells[cache.head + i].pos >= 0) {
                found = false;
                cache.head += i + 1;
                n_tested   += i + 1;
                break;
            }
        }

        if (found) {
            break;
        }

        if (n_tested >= cache.size) {
            //LLAMA_LOG_ERROR("%s: failed to find a slot for %d tokens\n", __func__, n_tokens);
            return false;
        }
    }
```
Lets look what is happening in the for loop, we know that `n_tokens` is 6 so we
will iteratate 0-5 times. There is nothing in the cache yet so all cells will
have positions that are -1 (unused).
```console
(gdb) p cache.cells[cache.head + i]
$85 = {pos = -1, delta = 0, src = 0, seq_id = std::set with 0 elements}
```
So in our cache head is 0 and we are checking the first 6 cells in the cache to
see if their position is greater than 0 (indicating that they in use). `found`
will therefor still be true and we will exit the loop.

Next we iterate over all the sequences.
```c++
    for (uint32_t s = 0; s < n_seqs; s++) {
        for (uint32_t i = 0; i < n_seq_tokens; ++i) {
            uint32_t k = s*n_seq_tokens + i;
            cache.cells[cache.head + k].pos = batch.pos[k];

            for (int32_t j = 0; j < batch.n_seq_id[s]; j++) {
                cache.cells[cache.head + k].seq_id.insert(batch.seq_id[s][j]);
            }
        }
    }

    cache.used += n_tokens;

    return true;
```
We will again iterate over our 6 tokens this time using the ubatch `n_seqs`.
So this will set the position of the cell to the position of the position of
the ubatch, so this cell will the be in use as it will have a pos value that is
not less than 0. And each cell also has a set of sequence ids which identify
the sequences that the token is part of.
This will be done for all the 6 tokens in the ubatch.

Finally we will update the `cache.used` to 6: and then return.
This will return ut to `llama_decode_internal` and we will continue from there.
```c++
            if (!kv_self.recurrent) {
                // a heuristic, to avoid attending the full cache if it is not yet utilized
                // after enough generations, the benefit from this heuristic disappears
                // if we start defragmenting the cache, the benefit from this will be more important
                const uint32_t pad = llama_kv_cache_get_padding(cparams);
                kv_self.n = std::min(kv_self.size, std::max(pad, GGML_PAD(llama_kv_cache_cell_max(kv_self), pad)));
                //kv_self.n = llama_kv_cache_cell_max(kv_self);
            }
```
Now, the padding is different depending on the type of attention in use. If 
flash attenting (FA) is used the padding wil be 256 and otherwise 32.

```console
(gdb) p llama_kv_cache_get_padding(cparams)
$15 = 32
(gdb) p llama_kv_cache_cell_max(kv_self)
$14 = 6

(gdb) p kv_self.n
$17 = 32
```
So this is where `kv_self.n` is set which something that I've been wondering
about.

The next interesting thing that happens with regards to the kv-cache is that
the graph is build:
```c++
        ggml_cgraph * gf = llama_build_graph(lctx, ubatch, false);
```

I'm just noting that in the callback we have the following with is related to
the kv cache but I'm not sure what it is about yet though. 
```c++
    llm_build_cb cb = [&](struct ggml_tensor * cur, const char * name, int il) {
        ...

        if (!lctx.cparams.offload_kqv) {
            if (strcmp(name, "kqv_merged_cont") == 0) {
                // all nodes between the KV store and the attention output are run on the CPU
                ggml_backend_sched_set_tensor_backend(lctx.sched, cur, lctx.backend_cpu);
            }
        }
```
The model I'm using for this example is a llama model so `llm.build_llama` will
be called.
```c++
    struct ggml_cgraph * build_llama() {
        ...

        const int64_t n_embd_head = hparams.n_embd_head_v;
```
```console
(gdb) p n_embd_head
$18 = 128
```

Now, `build_llama` is a method/member of the struct `llm_build_context` which
has a field named `kv_head`:
```console
(gdb) p this.kv_head
(gdb) 0
```
This is very important to and for the first prompt it will be zero and was
something that I overlooked the first time I stepped through the code and it
caused me some confusion. For the next token processed this value will be the
number of that token in the sequence. So we if had 6 tokens in the initital
prompt this would be 6 for the next token to be docoded.

First we have the input layer which will be build using either the tokens in
the ubatch or the embeddings in the ubatch:
```c++
        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);
        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = build_inp_pos();
```
```c++
        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor * KQ_mask = build_inp_KQ_mask();
```
```c++
    struct ggml_tensor * build_inp_KQ_mask(bool causal = true) {
        lctx.inp_KQ_mask = causal
            ? ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_kv,     GGML_PAD(n_tokens, GGML_KQ_MASK_PAD))
            : ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD));
        cb(lctx.inp_KQ_mask, "KQ_mask", -1);
        ggml_set_input(lctx.inp_KQ_mask);

        return flash_attn ? ggml_cast(ctx0, lctx.inp_KQ_mask, GGML_TYPE_F16) : lctx.inp_KQ_mask;
    }
```
In our case this will create a 2d tensor with a dimension of 32 (`n_kv`)
```c++
(gdb) p lctx.inp_KQ_mask->ne
$22 = {32, 32, 1, 1}
```
This mask will be used to mask out earlier tokens in the sequence. And notice
that the comment says that it will be broadcasted to all the heads which is the
reason why it may seem small.
Interesting to see `ggml_cast` which I have not used, and this is making sure
that if flash attention is used the tensor will be cast to f16.

Next all layer operations will be built:
```c++
        const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;
        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;
            ...

            // self-attention
            {
                ...
                struct ggml_tensor * Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, kq_scale, cb, il);
```
Notice here that the roped Key and Query operations are created and then being
passed into `llm_build_kv`. And notice that `kv_head` is passed in which like
we mentioned above will be important for the next tokens.

There are a lot of parameters passed to `llm_build_kv` but if we look through
them we have seen most of them before.
```c++
static struct ggml_tensor * llm_build_kv(
        struct ggml_context * ctx,
       struct llama_context & lctx,
       const llama_kv_cache & kv,
         struct ggml_cgraph * graph,
         struct ggml_tensor * wo,          // model.layers[il].wo
         struct ggml_tensor * wo_b,        // model.layers[il].bo
         struct ggml_tensor * k_cur,
         struct ggml_tensor * v_cur,
         struct ggml_tensor * q_cur,
         struct ggml_tensor * kq_mask,
                    int32_t   n_tokens,
                    int32_t   kv_head,
                    int32_t   n_kv,
                    float     kq_scale,
         const llm_build_cb & cb,
                    int       il) {
    const llama_hparams & hparams = lctx.model.hparams;
    const llama_cparams & cparams = lctx.cparams;

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(graph, q_cur);
    ggml_build_forward_expand(graph, k_cur);
    ggml_build_forward_expand(graph, v_cur);

    llm_build_kv_store(ctx, hparams, cparams, kv, graph, k_cur, v_cur, n_tokens, kv_head, cb, il);

    struct ggml_tensor * cur;

    cur  = llm_build_kqv(ctx, lctx, kv, graph, wo, wo_b, q_cur, kq_mask, n_tokens, n_kv, kq_scale, cb, il);
    cb(cur, "kqv_out", il);

    return cur;
```
I'm showing the complete function as want to point out that first the
`llm_build_kv_store` and then `llm_build_kqv` which is the `QK^T` operation.

What is happening, which I'll go through below, is that the `llm_build_kv_store`
function will copy the current roped key value into the cache (doing this one
head at a time). And then later in `llm_build_kqv` the `k` tensor that will be
used in the attention matrix multiplication will use a view into the layers
cache:
```c++
    struct ggml_tensor * k =
        ggml_view_3d(ctx, kv.k_l[il],
                n_embd_head_k, n_kv, n_head_kv,
                ggml_row_size(kv.k_l[il]->type, n_embd_k_gqa),
                ggml_row_size(kv.k_l[il]->type, n_embd_head_k),
                0);
    cb(k, "k", il);
    ...

        struct ggml_tensor * kqv = ggml_mul_mat(ctx, v, kq);
```
And this will operate on all the values in the cache up to this point.

Lets take a look at a few of the parameters passed to `llm_build_kv_store`:
```console
(gdb) p model.layers[il].wo->ne
$26 = {4096, 4096, 1, 1}
(gdb) p model.layers[il].bo
$27 = (ggml_tensor *) 0x0
(gdb) p Kcur->ne
$28 = {128, 32, 6, 1}
(gdb) p Qcur->ne
$29 = {128, 32, 6, 1}
(gdb) p KQ_mask->ne
$30 = {32, 32, 1, 1}
(gdb) p kv_head
$31 = 0
(gdb) p kq_scale
$32 = 0.0883883461
```

So this is what these matrices looks like for a single layer:
```
           Kcur                     Qcur                 KQ_mask
0   [0     ...     127]      0  [0     ...  127]      0  [0...31]
    .                     .                   .
    .                     .                   .
    .                     .                   .
31  [0     ...     127]      31 [0     ...  127]      31 [0...31]
```
So we have an embedding size of 4096 and we divide this into 32 heads which
means that each head will have an embeddings size of 128.

```c++
    llm_build_kv_store(ctx, hparams, cparams, kv, graph, k_cur, v_cur, n_tokens, kv_head, cb, il);
```

`llm_build_kv_store` also has quite a few parameters but again we have seen most
of them before:
```c++
static void llm_build_kv_store(
        struct ggml_context * ctx,
        const llama_hparams & hparams,
        const llama_cparams & cparams,
       const llama_kv_cache & kv,
         struct ggml_cgraph * graph,
         struct ggml_tensor * k_cur,
         struct ggml_tensor * v_cur,
                    int32_t   n_tokens,
                    int32_t   kv_head,
         const llm_build_cb & cb,
                    int64_t   il) {
    const int64_t n_ctx = cparams.n_ctx;

    const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);
    const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);
```
```console
(lldb) p n_ctx
(const int64_t) 1024
(lldb) p n_embd_k_gqa
(const int64_t) 4096
(lldb) p n_embd_v_gqa
(const int64_t) 4096
```
So we have a context size of 1024 and an embedding dimension size of 4096.
Next we will create a view of the kv cache `k_l` tensor for this layer, and
the number of elements will be `n_tokens * n_embed_k_gqa`, and the offset is
the last argument. The cache is empty in this case.
One note here is that if `kv_head` is a larger value, like 512 and not the size
of the tokens representing the prompt, then this is probably because there is
call to this function as part of `llama_new_context_with_model` and this can can
happen if you set a break point on a line somewhere in this function. Just 
`continue` in gdb or lldb to get past this and break in the function for 
building the operations for the prompt instead..

Next a view of the k matrix is created:
```c++
    struct ggml_tensor * k_cache_view = ggml_view_1d(ctx,
        kv.k_l[il],
        n_tokens*n_embd_k_gqa,
        ggml_row_size(kv.k_l[il]->type, n_embd_k_gqa)*kv_head);
```
We can see here that we are creating a view of the tensor `k_l` for the current
layer and notice that the offset used is taking into account the `kv_head`
value.
So this is creating a new which will be of size (elements) the number of tokens
being processed (6 in this case) times the embeddings size. And the offset
will be 0 in this case because this is the first time we are processing tokens:
```console
(lldb) p ggml_row_size(kv.k_l[il]->type, n_embd_k_gqa)
(size_t) 8192
(lldb) p kv_head
(int32_t) 0
(gdb) p ggml_row_size(kv.k_l[il]->type, n_embd_k_gqa)*kv_head
$69 = 0
```

```console
(lldb) p k_cache_view->ne
(int64_t[4])  ([0] = 24576, [1] = 1, [2] = 1, [3] = 1)
```
So in this case the will produce a view of the tensor and the view will span
the first 24576 (`n_tokens * n_embd_k_gqa` which is 6 * 4096 in this case)
elements.

Now, just to make this clear I'll also show what this would look like when
decoding the next token.
```console
(gdb) p kv_head
$75 = 6
(gdb) p n_tokens
$76 = 1
(gdb) p n_embd_k_qga
No symbol "n_embd_k_qga" in current context.
(gdb) p n_embd_k_gqa
$77 = 4096
(gdb) p n_embd_k_gqa * 1
$78 = 4096

(gdb) p ggml_row_size(kv.k_l[il]->type, n_embd_k_gqa)*kv_head
$79 = 49152
```
Notice that the offset is now 49152 which is the size of the previous view and
we can visualize this as follows:
```
                                                   offset
     0   [0   ...        4095]  (prompt token 1)   0
     1   [0   ...        4095]  (prompt token 2)   8192
     2   [0   ...        4095]  (prompt token 3)   16384
     3   [0   ...        4095]  (prompt token 4)   24576
     4   [0   ...        4095]  (prompt token 5)   32768 
     5   [0   ...        4095]  (prompt token 6)   40960 
     6   [0   ...        4095]  (next token 1)     49152
     ...
     ...
     ...
  1023   [0   ...        4095]
```
At this point we have created an operation to create a view with the correct
offset.

Next we create a tensor operation that will copy the current tokens roped
key value into this slot of the cache:
Then a copy operation will be created for copying `k_cur` to the `k_cache_view`:
```c++
    ggml_build_forward_expand(graph, ggml_cpy(ctx, k_cur, k_cache_view));
```
So this is how the new tokens roped key value is added to the cache. We have a
cache which can store 1024 tokens, each having an embedding dimension of 4096,
and `kv_head` is used to create the offset into the `k_l` tensor for each 
layer.

These tensors have different dimensions but the same number of elelements:
```console
(gdb) p ggml_n_dims(k_cur)
$16 = 3
(gdb) p ggml_n_dims(k_cache_view)
$17 = 1

(gdb) p k_cache_view->ne
$13 = {24576, 1, 1, 1}
(gdb) p k_cur->ne
$14 = {128, 32, 6, 1}
(gdb) p 128 * 32 * 6
$15 = 24576
```
So this is creating a copy operation to copy a head of the key tensor into
the k cache view. And we have 6 tokens in this batch so there will be 6 of these
which is indicated by the `z` axis below:
```
z_0

  0   [0     ...     127]
  .
  .
  .
  31  [0     ...     127]

.
.
.

z_5

  0   [0     ...     127]
  .
  .
  .
  31  [0     ...     127]

```

Now, `k_cur` was passed in and is the tensor representing the roped key value for
this token. So my understanding is that the cache is empty in this case and this
it taking the roped key value and copying it into the cache. 

This is only processing the current batch which is the prompt in our case so
there is nothing in the cache at this point. But if we had already processed
some tokens this would just be adding to the currently processed token to the
key cache. So we are adding a column to the key matrix for each token we
process.

Then we also create a view for the value matrix:
```c++
    struct ggml_tensor * v_cache_view = nullptr;
    if (cparams.flash_attn) {
        v_cache_view = ggml_view_1d(ctx, kv.v_l[il],
            n_tokens*n_embd_v_gqa, ggml_row_size(kv.v_l[il]->type, n_embd_v_gqa) * kv_head);
    } else {
        // note: the V cache is transposed when not using flash attention
        v_cache_view = ggml_view_2d(ctx, kv.v_l[il], n_tokens, n_embd_v_gqa,
                (  n_ctx)*ggml_element_size(kv.v_l[il]),
                (kv_head)*ggml_element_size(kv.v_l[il]));

        v_cur = ggml_transpose(ctx, v_cur);
    }
    ggml_build_forward_expand(graph, ggml_cpy(ctx, v_cur, v_cache_view));
```
```console
(gdb) p v_cache_view->ne
$31 = {6, 4096, 1, 1}
```
And this is then transposed and also copied into the value cache view.
So the above will have populated the cache with the key and value for one
single layer. The next time we process a token which belongs to the same
sequence this will add a column to the key cache matrix and a row to the 
value cache matrix.

This is the last thing to happen in `llm_build_kv_store` and we will be back in
`llm_build_kv`:
```c++
    llm_build_kv_store(ctx, hparams, cparams, kv, graph, k_cur, v_cur, n_tokens, kv_head, cb, il);
```

```c++
static struct ggml_tensor * llm_build_kqv(
        struct ggml_context * ctx,
       struct llama_context & lctx,
       const llama_kv_cache & kv,
         struct ggml_cgraph * graph,
         struct ggml_tensor * wo,
         struct ggml_tensor * wo_b,
         struct ggml_tensor * q_cur,
         struct ggml_tensor * kq_mask,
                    int32_t   n_tokens,
                    int32_t   n_kv,
                    float     kq_scale,
         const llm_build_cb & cb,
                    int       il) {

    const llama_model   & model   = lctx.model;
    const llama_hparams & hparams = lctx.model.hparams;
    const llama_cparams & cparams = lctx.cparams;

    const int64_t n_ctx         = cparams.n_ctx;
    const int64_t n_head        = hparams.n_head(il);
    const int64_t n_head_kv     = hparams.n_head_kv(il);
    const int64_t n_embd_head_k = hparams.n_embd_head_k;
    const int64_t n_embd_k_gqa  = hparams.n_embd_k_gqa(il);
    const int64_t n_embd_head_v = hparams.n_embd_head_v;
    const int64_t n_embd_v_gqa  = hparams.n_embd_v_gqa(il);

    struct ggml_tensor * q = ggml_permute(ctx, q_cur, 0, 2, 1, 3);
```
So this will swap the second and third dimensions of `q_cur`:
```console
(lldb) p q_cur->ne
(int64_t[4])  ([0] = 128, [1] = 32, [2] = 6, [3] = 1)

(lldb) p q->ne
(int64_t[4])  ([0] = 128, [1] = 6, [2] = 32, [3] = 1)
```
So this will be restructured to something like this:
```
q matrix:

z0
          x-axis ->
  0   [0  ...   127]          y-axis
      .                         ↓
      .
  5   [0  ...   127]

.
.
.

z31
  0   [0  ...   127]
      .
      .
  5   [0  ...   127]
```
So we have 32 heads, and each one matrix with 6 rows each with 128 dimensions is
what I'm trying to convey here.

Next something similar is done for the Key matrix:
```c++
    struct ggml_tensor * k =
        ggml_view_3d(ctx, kv.k_l[il],
                n_embd_head_k, n_kv, n_head_kv,
                ggml_row_size(kv.k_l[il]->type, n_embd_k_gqa),
                ggml_row_size(kv.k_l[il]->type, n_embd_head_k),
                0);
    cb(k, "k", il);
```
Notice that this is using `kv.k_l[il]` which is the the tensor of the cache for
this layer. So when the k q multiplication is done below it will be using this view
of the cache.

But notice that the dimensions are different:
```
z0
  0   [0  ...   127]
      .
      .
      .
      .
  31  [0  ...   127]
.  
.
.

z31
  0   [0  ...   127]
      .
      .
      .
      .
  31  [0  ...   127]

```
```console
(lldb) p k->ne
(int64_t[4])  ([0] = 128, [1] = 32, [2] = 32, [3] = 1)
```
Next there is a block for flash attention: TODO: try out flash attention.

```c++
    if (cparams.flash_attn) {

    } else {
        struct ggml_tensor * kq = ggml_mul_mat(ctx, k, q);
        cb(kq, "kq", il);
```
Next is the actual QK^T operation is created:
```console
struct ggml_tensor * kq = ggml_mul_mat(ctx, k, q);

(lldb) p k->ne
(int64_t[4])  ([0] = 128, [1] = 32, [2] = 32, [3] = 1)
(lldb) p q->ne
(int64_t[4])  ([0] = 128, [1] = 6, [2] = 32, [3] = 1)
```
We can visualize this as we are performing 32 separate 2d  multiplications:
```
      k matrix
z0
  0   [0  ...   127]
      .
      .
      .
      .
  31  [0  ...   127]

       q matrix
z0
          x-axis ->
  0   [0  ...   127]          y-axis
      .                         ↓
      .                       
  5   [0  ...   127]          

```
We need to keep in mind that ggml will transpose the second tensor so this becomes:
```
      k matrix                   q matrix

  0   [0  ...   127]        0    [0  ...  5]     0  [0 ... 31]
      .                          .                  .
      .                x         .            =     .
      .                          .               5  [0 ... 31]
  31  [0  ...   127]             .

                           127   [0  ...  5]
```
And this will enable the multiplication to work.

Next we have:
```c++
        if (model.arch == LLM_ARCH_PHI2 || model.arch == LLM_ARCH_PHI3 ||
            model.arch == LLM_ARCH_GPTNEOX || model.arch == LLM_ARCH_QWEN2 ||
            model.arch == LLM_ARCH_NEMOTRON || model.arch == LLM_ARCH_CHATGLM) {
            // for this arch, we need to perform the KQ multiplication with F32 precision, otherwise we get NaNs
            // ref: https://github.com/ggerganov/llama.cpp/pull/4490#issuecomment-1859055847
            ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        }
```
This is not the case for this session but something that might be good to be aware of.

Next there is a special case for GROK:
```c++
        if (model.arch == LLM_ARCH_GROK) {
            // need to do the following:
            // multiply by attn_output_multiplyer of 0.08838834764831845
            // and then :
            // kq = 30 * tanh(kq / 30)
            // before the softmax below

            //try from phi2
            //ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

            kq = ggml_tanh(ctx, ggml_scale(ctx, kq, 0.08838834764831845f/30.0f));
            kq = ggml_scale(ctx, kq, 30);
        }
```

That is pretty much it for `llama_build_graph` related to the kv cache.

Now, lets take a look at `llama_set_inputs` to see if the kv cache is used
there.
```c++
    if (lctx.inp_KQ_mask || lctx.inp_KQ_mask_swa) {
        // NOTE: hparams.causal_attn indicates the model is capable of generation and uses the kv cache.
        if (cparams.causal_attn && !lctx.is_encoding) {
            const int64_t n_kv         = kv_self.n;
            const int64_t n_tokens     = ubatch.n_tokens;
            const int64_t n_seq_tokens = ubatch.n_seq_tokens;
            const int64_t n_seqs       = ubatch.n_seqs;


            float * data     = nullptr;
            float * data_swa = nullptr;

            if (lctx.inp_KQ_mask) {
                GGML_ASSERT(ggml_backend_buffer_is_host(lctx.inp_KQ_mask->buffer));
                data = (float *) lctx.inp_KQ_mask->data;
            }
```

Next we have the following for loop which is setting the mask (lctx.inp_KQ_mask)
for a single head (h):
```c++
            for (int h = 0; h < 1; ++h) {
                for (int s = 0; s < n_seqs; ++s) {
                    const llama_seq_id seq_id = ubatch.seq_id[s][0];

                    for (int j = 0; j < n_seq_tokens; ++j) {
                        const llama_pos pos = ubatch.pos[s*n_seq_tokens + j];

                        for (int i = 0; i < n_kv; ++i) {
                            float f;
                            if (!kv_self.cells[i].has_seq_id(seq_id) || kv_self.cells[i].pos > pos) {
                                f = -INFINITY;
                            } else {
                                if (hparams.use_alibi) {
                                    f = -std::abs(kv_self.cells[i].pos - pos);
                                } else {
                                    f = 0.0f;
                                }
                            }

                            if (data) {
                                data[h*(n_kv*n_tokens) + s*(n_kv*n_seq_tokens) + j*n_kv + i] = f;
                            }

                            // may need to cut off old tokens for sliding window
                            if (data_swa) {
                                if (pos - kv_self.cells[i].pos >= (int32_t)hparams.n_swa) {
                                    f = -INFINITY;
                                }
                                data_swa[h*(n_kv*n_tokens) + s*(n_kv*n_seq_tokens) + j*n_kv + i] = f;
                            }
                        }
                    }
                }

                if (data) {
                    for (int i = n_tokens; i < GGML_PAD(n_tokens, GGML_KQ_MASK_PAD); ++i) {
                        for (int j = 0; j < n_kv; ++j) {
                            data[h*(n_kv*n_tokens) + i*n_kv + j] = -INFINITY;
                        }
                    }
                }

                if (data_swa) {
                    for (int i = n_tokens; i < GGML_PAD(n_tokens, GGML_KQ_MASK_PAD); ++i) {
                        for (int j = 0; j < n_kv; ++j) {
                            data_swa[h*(n_kv*n_tokens) + i*n_kv + j] = -INFINITY;
                        }
                    }
                }
            }
```
Since h will always be zero we can simplify this a little for readability:
```c++
                for (int s = 0; s < n_seqs; ++s) {
                    const llama_seq_id seq_id = ubatch.seq_id[s][0];

                    for (int j = 0; j < n_seq_tokens; ++j) {
                        const llama_pos pos = ubatch.pos[s*n_seq_tokens + j];

                        for (int i = 0; i < n_kv; ++i) {
                            float f;
                            if (!kv_self.cells[i].has_seq_id(seq_id) || kv_self.cells[i].pos > pos) {
                                f = -INFINITY;
                            } else {
                                if (hparams.use_alibi) {
                                    f = -std::abs(kv_self.cells[i].pos - pos);
                                } else {
                                    f = 0.0f;
                                }
                            }

                            if (data) {
                                data[s * (n_kv*n_seq_tokens) + j*n_kv + i] = f;
                            }

                            // may need to cut off old tokens for sliding window
                            if (data_swa) {
                                if (pos - kv_self.cells[i].pos >= (int32_t)hparams.n_swa) {
                                    f = -INFINITY;
                                }
                                data_swa[s * (n_kv*n_seq_tokens) + j*n_kv + i] = f;
                            }
                        }
                    }
                }
```
The mask (for a single head) is a square matrix:
```console
(gdb) p lctx.inp_KQ_mask->ne
$23 = {32, 32, 1, 1}
```
The above will iterate over all the tokens in the ubatch (`n_seq` above which is
6 in this case). And each token can potentially be part of multiple sequences
which is that `n_seq_tokens` is specifying so we iterate over all of them. Then
we iterator over all the entries in the kv cache (32 here even though we only
have 6 tokens due to the padding we saw earlier). For each entry in the cache
the code will check if the current cache cell does not have the current tokens
sequence id, or if the current cells position is greater than the current tokens
position.

This is what the cache cells looks like at this point:
```
cell_0    {pos = 0, delta = 0, src = -1, tail = -1, seq_id = std::set with 1 element = {[0] = 0}}
cell_1    {pos = 1, delta = 0, src = -1, tail = -1, seq_id = std::set with 1 element = {[0] = 0}}
cell_2    {pos = 2, delta = 0, src = -1, tail = -1, seq_id = std::set with 1 element = {[0] = 0}}
cell_3    {pos = 3, delta = 0, src = -1, tail = -1, seq_id = std::set with 1 element = {[0] = 0}}
cell_4    {pos = 4, delta = 0, src = -1, tail = -1, seq_id = std::set with 1 element = {[0] = 0}}
cell_5    {pos = 5, delta = 0, src = -1, tail = -1, seq_id = std::set with 1 element = {[0] = 0}}
cell_6    {pos = -1, delta = 0, src = -1, tail = -1, seq_id = std::set with 0 elements}
.
.
.
cell_1023 {pos = -1, delta = 0, src = -1, tail = -1, seq_id = std::set with 0 elements}
```
So for the first entry the else clause will be executed and the value of f will
be 0.0f. And recall that `data` is a pointer to the tensor `lctx.inp_KQ_mask`'s
data.
```c++
    data[s * (n_kv * n_seq_tokens) + j * n_kv + i] = f;
```
This first iteration s is 0, j is 0, and i is 0:
```
    data[0] = 0.0f;
```
For the next iteration (of 32) the value of f will be -INFINITY:
```
    data[1] = -INFINITY;
```
And this will continue until all 32 entries have been processed for the first
token in the sequence. 
```console
      ↓
      0    1    2    3    4    5    6   ...        31
    +----+----+----+----+----+----+----+----+----+----+
    |0.0f|-inf|-inf|-inf|-inf|-inf|-inf|  ...    |    |
    +----+----+----+----+----+----+----+----+----+----+
    | 0  | 0  | 0  | 0  | 0  |  0 |  0 | 0  |   ...   |
    +----+----+----+----+----+----+----+----+----+----+
    | 0  | 0  | 0  | 0  | 0  |  0 |  0 | 0  |   ...   |
    +----+----+----+----+----+----+----+----+----+----+
    | 0  | 0  | 0  | 0  | 0  |  0 |  0 | 0  |   ...   |
    +----+----+----+----+----+----+----+----+----+----+
    | 0  | 0  | 0  | 0  | 0  |  0 |  0 | 0  |   ...   |
    +----+----+----+----+----+----+----+----+----+----+
    | 0  | 0  | 0  | 0  | 0  |  0 |  0 | 0  |   ...   |
    +----+----+----+----+----+----+----+----+----+----+
```
Notice that the values in this matrix are initially zero.

Then we do the same for the next token in the sequence (s=1).
```c++
    data[1 * (n_kv * n_seq_tokens) + j * n_kv + i] = f;
    data[1 * (n_kv * n_seq_tokens)] = f;
    data[1 * (32 * 1)] = f;
    data[1 * (32)] = f;
    data[(32)] = f;
```
```console
           ↓
      0    1    2    3    4    5    6   ...        31
    +----+----+----+----+----+----+----+----+---------+
    |0.0f|-inf|-inf|-inf|-inf|-inf|-inf|-inf|   ...   |
    +----+----+----+----+----+----+----+----+---------+
    |0.0f|0.0f|-inf|-inf|-inf|-inf|-inf|-inf|   ...   |
    +----+----+----+----+----+----+----+----+----+----+
    | 0  | 0  | 0  | 0  | 0  |  0 |  0 | 0  |   ...   |
    +----+----+----+----+----+----+----+----+----+----+
    | 0  | 0  | 0  | 0  | 0  |  0 |  0 | 0  |   ...   |
    +----+----+----+----+----+----+----+----+----+----+
    | 0  | 0  | 0  | 0  | 0  |  0 |  0 | 0  |   ...   |
    +----+----+----+----+----+----+----+----+----+----+
    | 0  | 0  | 0  | 0  | 0  |  0 |  0 | 0  |   ...   |
    +----+----+----+----+----+----+----+----+----+----+
    | 0  | 0  | 0  | 0  | 0  |  0 |  0 | 0  |   ...   |
    +----+----+----+----+----+----+----+----+----+----+
```
Then we do the same for the next token in the sequence (s=2).
```c++
    data[2 * (n_kv * n_seq_tokens) + j * n_kv + i] = f;
    data[2 * (n_kv * n_seq_tokens)] = f;
    data[2 * ( * 1)] = f;
    data[2 * (32)] = f;
    data[64] = f;
```
```console
                ↓
      0    1    2    3    4    5    6   ...        31
    +----+----+----+----+----+----+----+----+---------+
    |0.0f|-inf|-inf|-inf|-inf|-inf|-inf|-inf|   ...   |
    +----+----+----+----+----+----+----+----+---------+
    |0.0f|0.0f|-inf|-inf|-inf|-inf|-inf|-inf|   ...   |
    +----+----+----+----+----+----+----+----+----+----+
    |0.0f|0.0f|0.0f|-inf|-inf|-inf|-inf|-inf|   ...   |
    +----+----+----+----+----+----+----+----+----+----+
    | 0  | 0  | 0  | 0  | 0  |  0 |  0 | 0  |   ...   |
    +----+----+----+----+----+----+----+----+----+----+
    | 0  | 0  | 0  | 0  | 0  |  0 |  0 | 0  |   ...   |
    +----+----+----+----+----+----+----+----+----+----+
    | 0  | 0  | 0  | 0  | 0  |  0 |  0 | 0  |   ...   |
    +----+----+----+----+----+----+----+----+----+----+
    | 0  | 0  | 0  | 0  | 0  |  0 |  0 | 0  |   ...   |
    +----+----+----+----+----+----+----+----+----+----+
```
And this continues until all the `ubatch.n_seq` tokens have been processed which
is 6 in this case:
```console
      0    1    2    3    4    5    6   ...        31
    +----+----+----+----+----+----+----+----+---------+
0   |0.0f|-inf|-inf|-inf|-inf|-inf|-inf|-inf|   ...   |
    +----+----+----+----+----+----+----+----+---------+
1   |0.0f|0.0f|-inf|-inf|-inf|-inf|-inf|-inf|   ...   |
    +----+----+----+----+----+----+----+----+----+----+
2   |0.0f|0.0f|0.0f|-inf|-inf|-inf|-inf|-inf|   ...   |
    +----+----+----+----+----+----+----+----+----+----+
3   |0.0f|0.0f|0.0f|0.0f|-inf|-inf|-inf|-inf|   ...   |
    +----+----+----+----+----+----+----+----+----+----+
4   |0.0f|0.0f|0.0f|0.0f|0.0f|-inf|-inf|-inf|   ...   |
    +----+----+----+----+----+----+----+----+----+----+
5   |0.0f|0.0f|0.0f|0.0f|0.0f|0.0f|-inf|-inf|   ...   |
    +----+----+----+----+----+----+----+----+----+----+
6   | 0  | 0  | 0  | 0  | 0  |  0 |  0 | 0  |   ...   |
    +----+----+----+----+----+----+----+----+----+----+
7   | 0  | 0  | 0  | 0  | 0  |  0 |  0 | 0  |   ...   |
    +----+----+----+----+----+----+----+----+----+----+
    .
    .
    .
    +----+----+----+----+----+----+----+----+----+----+
31  | 0  | 0  | 0  | 0  | 0  |  0 |  0 | 0  |   ...   |
    +----+----+----+----+----+----+----+----+----+----+
```

Then we have the following loop:
```c++
                if (data) {
                    for (int i = n_tokens; i < GGML_PAD(n_tokens, GGML_KQ_MASK_PAD); ++i) {
                        for (int j = 0; j < n_kv; ++j) {
                            data[h*(n_kv*n_tokens) + i*n_kv + j] = -INFINITY;
                        }
                    }
                }
```
The `GGML_PAD` macro is defined as:
```c++
#define GGML_KQ_MASK_PAD 32
#define GGML_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))

GGML_PAD(n_tokens, GGML_KQ_MASK_PAD);
GGML_PAD(6, 32);
GGML_PAD(6, 32) (((6) + (32) - 1) & ~((32) - 1))
= 32
```
This will round up to the nearest multiple of 32 which is 32 in this case. So
in our case this will iterate of the last 26 entries in the mask and set them
to -INFINITY. Recall that each token has a mask and this will fill each of
these padding token masks with -INFINITY. So there will be 26 tokens with a mask
that look something like this:
```console
      0    1    2    3    4    5    6   ...        31
    +----+----+----+----+----+----+----+----+---------+
0   |-inf|-inf|-inf|-inf|-inf|-inf|-inf|-inf|   ...   |
    +----+----+----+----+----+----+----+----+---------+
1   |-inf|-inf|-inf|-inf|-inf|-inf|-inf|-inf|   ...   |
    +----+----+----+----+----+----+----+----+----+----+
2   |-inf|-inf|-inf|-inf|-inf|-inf|-inf|-inf|   ...   |
    +----+----+----+----+----+----+----+----+----+----+
3   |-inf|-inf|-inf|-inf|-inf|-inf|-inf|-inf|   ...   |
    +----+----+----+----+----+----+----+----+----+----+
4   |-inf|-inf|-inf|-inf|-inf|-inf|-inf|-inf|   ...   |
    +----+----+----+----+----+----+----+----+----+----+
5   |-inf|-inf|-inf|-inf|-inf|-inf|-inf|-inf|   ...   |
    +----+----+----+----+----+----+----+----+----+----+
6   |-inf|-inf|-inf|-inf|-inf|-inf|-inf|-inf|   ...   |
    +----+----+----+----+----+----+----+----+----+----+
7   |-inf|-inf|-inf|-inf|-inf|-inf|-inf|-inf|   ...   |
    +----+----+----+----+----+----+----+----+----+----+
    .
    .
    .
    +----+----+----+----+----+----+----+----+----+----+
31  |-inf|-inf|-inf|-inf|-inf|-inf|-inf|-inf|   ...   |
    +----+----+----+----+----+----+----+----+----+----+
```
After this we will continue in `llama_set_inputs` but there is nothing more
releated to the kv cache in this function.

So to recap this after the QK^T operation is performed the result will be
copied into the layers `kv.k_l` tensor. And similar for the value cache but
that the operation is:
```c++
        struct ggml_tensor * kqv = ggml_mul_mat(ctx, v, kq);
        cb(kqv, "kqv", il);
```

The next thing that happens is the graph is computed, which will perform the
operations that have been built up in the graphs.
Following that the cache head is updated in `
```c++
        // update the kv ring buffer
        {
            kv_self.head += n_tokens;

            // Ensure kv cache head points to a valid index.
            if (kv_self.head >= kv_self.size) {
                kv_self.head = 0;
            }
        }
```
So for the next decode `kv_self.head` will be 6 and the offset we say above for
the key and values cache will use 6 when calculating the offset to store the 
roped k and value cache entried for the next token.

_wip_



### `kv_self`
A `llama_context` contains a member named `kv_self` (self as in self attention)
which is of type `llama_kv_cache`. This struct is defined in `llama.cpp`:
```c++
struct llama_context {
    ...
    // key + value cache for the self attention
    struct llama_kv_cache kv_self;
```
So every `llama_context` will have a key-value cache for self attention.

And the `llama_kv_cache` struct is defined as follows:
```c++
struct llama_kv_cache {
    bool has_shift = false;
    bool do_defrag = false;
    bool do_copy   = false;
    bool recurrent = false; // with recurrent state models, a cell can hold the state for more than one past token
    bool v_trans   = true;  // the value tensor is transposed
    uint32_t head = 0;
    uint32_t size = 0;
    uint32_t used = 0; // used cells (i.e. at least one seq_id)

    // computed before each graph build
    uint32_t n = 0;

    ggml_type type_k = GGML_TYPE_F16;
    ggml_type type_v = GGML_TYPE_F16;

    std::vector<llama_kv_cell> cells;

    std::vector<struct ggml_tensor *> k_l; // per layer
    std::vector<struct ggml_tensor *> v_l;

    std::vector<struct ggml_context *> ctxs;
    std::vector<ggml_backend_buffer_t> bufs;
```

Recall that there is a KV-Cache per layer in the transformer architecture. 
And notice that there is a vector of `ggml_tensor` pointer
for key and one for the value per layers. So for each layer there is a tensor
which we will see later is a 1d tensor, or just a list of values. And each layer
has a `ggml_context` and also a `ggml_backend_buffer_t`.

So when a `llama_context` is created the `kv_self` will also be created using
default initialization (so just default values will be assigned to it).
```c++
struct llama_context {
    llama_context(const llama_model & model) : model(model), t_start_us(model.t_start_us), t_load_us(model.t_load_us) {}
    ...
}
```

```console
$ gdb --args simple_prompt
(gdb) br llama_new_context_with_model
(gdb) r
(gdb) 
16368	    llama_context * ctx = new llama_context(*model);
(gdb) n
(gdb) p ctx.kv_self 
$1 = {has_shift = false, do_defrag = false, do_copy = false, recurrent = false,
v_trans = true, head = 0, size = 0, used = 0, n = 0,
type_k = GGML_TYPE_F16,
type_v = GGML_TYPE_F16,
cells = std::vector of length 0, capacity 0, 
k_l = std::vector of length 0, capacity 0,
v_l = std::vector of length 0, capacity 0,
ctxs = std::vector of length 0, capacity 0, 
bufs = std::vector of length 0, capacity 0}
```
So after the construction of a `llama_context` the `kv_self` member is
initialized to default values, there has not been any explicit assignements to
any of the members of `kv_self`.

Further down in `llama_new_context_with_model` `kv_self` is initialized with:
```c++
        if (!llama_kv_cache_init(ctx->kv_self, ctx, type_k, type_v, kv_size, cparams.offload_kqv)) {
            LLAMA_LOG_ERROR("%s: llama_kv_cache_init() failed for self-attention cache\n", __func__);
            llama_free(ctx);
            return nullptr;
        }
```
So we are passing in `ctx->kv_self` which will have a local name of `cache`
below:
```
static bool llama_kv_cache_init(
             struct llama_kv_cache & cache,
               const llama_context * ctx,
                         ggml_type   type_k,
                         ggml_type   type_v,
                          uint32_t   kv_size,
                              bool   offload) {

    const int64_t  n_layer = hparams.n_layer;

    cache.has_shift = false;

    cache.recurrent = llama_model_is_recurrent(&model);
    cache.v_trans   = !cache.recurrent && !cparams.flash_attn;

    cache.head = 0;
    cache.size = kv_size;
    cache.used = 0;

    cache.type_k = type_k;
    cache.type_v = type_v;

    cache.cells.clear();
    cache.cells.resize(kv_size);
```
The `kv_size` is the passed in and will be the size of the computation param
`n_ctx` unless the model supports Mamba.
```console
(gdb) p n_layer
$3 = 32

(gdb) p kv_size
$12 = 1024

(gdb) p cache.v_trans
$4 = true

(gdb) p cache.size
$5 = 1024

(gdb) p cache.type_v
$9 = GGML_TYPE_F16

(gdb) p cache.cells.size()
$10 = 1024
```
So we can see that we have 1024 cells in this cache.

Next, a map of `ggml_backend_buffer_type_t` and a count of the different types
of backend buffers for the kv cache. In our case `offload` is true (comes from
cparams.offload_kqv) so that is the path that will be taken. But also notice
that if this was not the case then the default buffer type would be
`llama_default_buffer_type_cpu(true)` would be set as the key and the found 32.

But for the offload case we iterate over the number of layers and count the
number of different buffer types used:
```c++
    // count used buffer types
    std::map<ggml_backend_buffer_type_t, int> buft_layer_count;
    if (offload) {
        for (int64_t i = 0; i < n_layer; ++i) {
            buft_layer_count[model.buft_layer[i].buft]++;
        }
    } else {
        buft_layer_count[llama_default_buffer_type_cpu(true)] = n_layer;
    }
```
So the model struct has a field `buft_layer` which is a vector of `llama_buft`:
```console
(gdb) ptype model.buft_layer
type = std::vector<llama_model::layer_buft>

(gdb) p model.buft_layer.size()
$14 = 32
```
This vector is populated by `llama_load_tensors`.

After this the map looks like this:
```console
(gdb) p buft_layer_count
$25 = std::map with 1 element = {[0x555555978400 <ggml_backend_cpu_buffer_type>] = 32}
```
Next for each of the entries in the `buft_layer_count` map we create a ggml
context for each buffer type, and in this case there is only one element, which
has a count of 32:
```c++
    // create a context for each buffer type
    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    for (auto & it : buft_layer_count) {
        int n_layers = it.second;
        struct ggml_init_params params = {
            /*.mem_size   =*/ 2u*n_layers*ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            LLAMA_LOG_ERROR("%s: failed to allocate context for kv cache\n", __func__);
            return false;
        }
        ctx_map[it.first] = ctx;
        cache.ctxs.push_back(ctx);
    }
```
So after this there will be one entry in `cache.ctxs`

Next the vectors of key and value vectors will be reserved for the capacity of
the number of layers in the model, the following will happen in `llama_kv_cache_init`:
```c++
    cache.k_l.reserve(n_layer);
    cache.v_l.reserve(n_layer);
```
And these vectors store elements of type `ggml_tensor`:
```console
(lldb) p n_layer
(const int64_t) 32

(gdb) ptype cache.k_l
type = std::vector<ggml_tensor*>
```
So each of these vectors will be able to hold 32 `ggml_tensor` pointers, one
for each layer. 

Next, these vectors are populated with the following:
```
    for (int i = 0; i < (int) n_layer; i++) {
        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(i) + hparams.n_embd_k_s();
        const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(i) + hparams.n_embd_v_s();

        struct ggml_context * ctx = offload ? ctx_map.at(model.buft_layer[i].buft) : cache.ctxs.front();
        ggml_tensor * k = ggml_new_tensor_1d(ctx, type_k, n_embd_k_gqa*kv_size);
        ggml_tensor * v = ggml_new_tensor_1d(ctx, type_v, n_embd_v_gqa*kv_size);
        ggml_format_name(k, "cache_k_l%d", i);
        ggml_format_name(v, "cache_v_l%d", i);
        cache.k_l.push_back(k);
        cache.v_l.push_back(v);
    }
```
So we have this function `n_embd_k_gqa` which returnes the number of embedding
dimensions for the Key matrix for grouped query attention (qga). Notice that we
are passing in the layer which sounds like there can be different embedding
sizes for different layers.
```c++
    uint32_t n_embd_k_gqa(uint32_t il = 0) const { // dimension of key embeddings across all k-v heads
        const uint32_t n_head_kv = this->n_head_kv(il);

        return n_embd_head_k * n_head_kv;
    }
```
And `n_embd_head_k` is the number of embeddings in the key matrix for each head:
```console
    uint32_t n_head_kv(uint32_t il = 0) const {
        if (il < n_layer) {
            return n_head_kv_arr[il];
        }

        GGML_ABORT("fatal error");
    }
```
`n_head_kv_arr` is a fixed size array, the size is of `LLAMA_MAX_LAYERS` which is
currently 512:
```c++
#define LLAMA_MAX_LAYERS  512

    std::array<uint32_t, LLAMA_MAX_LAYERS> n_head_arr;
    std::array<uint32_t, LLAMA_MAX_LAYERS> n_head_kv_arr;
    std::array<uint32_t, LLAMA_MAX_LAYERS> n_ff_arr;
```
Notice that there other per-layer parameters, `n_head_arr` is a value per layer
for the number of heads, and then we also have `n_ff_arr` which is the  number
of feed-forward/MLP layers too. The values are loaded from the model in
`llm_load_hparams`:
```c++
    // zero-out the per-layer hparams
    std::fill(hparams.n_head_arr.begin(),    hparams.n_head_arr.end(),    0);
    std::fill(hparams.n_head_kv_arr.begin(), hparams.n_head_kv_arr.end(), 0);
    std::fill(hparams.n_ff_arr.begin(),      hparams.n_ff_arr.end(),      0);

    ml.get_key_or_arr(LLM_KV_FEED_FORWARD_LENGTH,  hparams.n_ff_arr,   hparams.n_layer);
    ml.get_key_or_arr(LLM_KV_ATTENTION_HEAD_COUNT, hparams.n_head_arr, hparams.n_layer);

    // n_head_kv is optional, default to n_head
    hparams.n_head_kv_arr = hparams.n_head_arr;

    ml.get_key_or_arr(LLM_KV_ATTENTION_HEAD_COUNT_KV, hparams.n_head_kv_arr, hparams.n_layer, false);
```
So the model can indeed have a different number of heads for each layer, but in
our case we have the same number of heads for each layer.

```console
(lldb) p this
(const llama_hparams *) 0x0000000132812228

(gdb) p n_head_kv_arr.size()
$23 = 512
(gdb) p n_head_kv_arr[il]
$25 = 32
```
So the dimensions for these tensors will be:
```c++
(gdb) p n_embd_k_gqa
$35 = 4096
(gdb) p n_embd_v_gqa
$36 = 4096
```
And the size of the cache in this case is `kv_size`, so the above will create
a 1d tensor of size `n_embd_k_gqa*kv_size (4096*1024)` for the key and the value
tensors. And `hparams_n_embd_k_s` is zero in this case as this is only used for 
recursive models I think which is not the case here. So the embedding dimension
for each layer is 4096.

Just to recap where we are:
```c++
    for (int i = 0; i < (int) n_layer; i++) {
        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(i) + hparams.n_embd_k_s();
        const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(i) + hparams.n_embd_v_s();

        struct ggml_context * ctx = offload ? ctx_map.at(model.buft_layer[i].buft) : cache.ctxs.front();
        ggml_tensor * k = ggml_new_tensor_1d(ctx, type_k, n_embd_k_gqa*kv_size);
        ggml_tensor * v = ggml_new_tensor_1d(ctx, type_v, n_embd_v_gqa*kv_size);
        ggml_format_name(k, "cache_k_l%d", i);
        ggml_format_name(v, "cache_v_l%d", i);
        cache.k_l.push_back(k);
        cache.v_l.push_back(v);
    }
```

So each of the k tensors will be of size of the embeddings dimensions plus the
number of items that can be stored in the cache.
```console
(gdb) p n_embd_k_gqa 
$39 = 4096
(gdb) p kv_size
$40 = 1024
```
And we can take a look at the shape of `k`:
```console
(gdb) p *k
$3 = {type = GGML_TYPE_F16, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0,
ne = {4194304, 1, 1, 1}, nb = {2, 8388608, 8388608, 8388608}, 
op = GGML_OP_NONE, op_params = {0 <repeats 16 times>},
flags = 0,
grad = 0x0,
src = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, 
perf_runs = 0, perf_cycles = 0, perf_time_us = 0,
view_src = 0x0,
view_offs = 0,
data = 0x0,
name = '\000' <repeats 63 times>, 
extra = 0x0, padding = "\000\000\000\000\000\000\000"}
```
So this is a 1d tensor:
```console
   [0                                                        4194304]
```
And these tensors will then be added to the `cache.k_l` and `cache.v_l` vectors:
```c++
        cache.k_l.push_back(k);
        cache.v_l.push_back(v);
```
So each layer will have a key tensor which has 4194304 elements which we can
think of as being 1024 rows (one entry for each item in the cache, each with
4096 dimensions in each. One row for each entry in the cache. 4096 is the
embedding dimenesion size.
```
     0   [0   ...        4095]
     ...
     ...
     ...
  1023   [0   ...        4095]

```
And recall that this will be done for each `n_layer`s in the model.

Again, recall that the `ctx_map` only contains one entry.
```c++
    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    for (auto it : ctx_map) {
        ggml_backend_buffer_type_t buft = it.first;
        ggml_context * ctx = it.second;
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!buf) {
            LLAMA_LOG_ERROR("%s: failed to allocate buffer for kv cache\n", __func__);
            return false;
        }
        ggml_backend_buffer_clear(buf, 0);
        cache.bufs.push_back(buf);
    }
```
Now, `buft` describes the buffer type and we can inspect it like this:
```console
(gdb) ptype buft
type = struct ggml_backend_buffer_type {
    ggml_backend_buffer_type_i iface;
    ggml_backend_buffer_type_context_t context;
} *

(gdb) ptype buft.iface
type = struct ggml_backend_buffer_type_i {
    const char *(*get_name)(ggml_backend_buffer_type_t);
    ggml_backend_buffer_t (*alloc_buffer)(ggml_backend_buffer_type_t, size_t);
    size_t (*get_alignment)(ggml_backend_buffer_type_t);
    size_t (*get_max_size)(ggml_backend_buffer_type_t);
    size_t (*get_alloc_size)(ggml_backend_buffer_type_t, const ggml_tensor *);
    _Bool (*is_host)(ggml_backend_buffer_type_t);
}

(gdb) p buft.iface.get_name(buft)
$67 = 0x555555882d22 "CPU"

(gdb) p buft.iface.is_host(buft)
$70 = true
```
Notice that `buf` is of type `ggml_backend_buffer_t` which is different from
`buft` which is of type `ggml_backend_buffer_type_t`, at least I have some
trouble keeping these apart as the names are somewhat similar. I try to think
that one it is a description of a backend buffer type and the other is an actual
buffer. So in the following we are passing in the buffer type to allocate a new
buffer of that type:
```c++
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
```
```c++
ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors_from_buft(struct ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_get_no_alloc(ctx) == true);

    size_t alignment = ggml_backend_buft_get_alignment(buft);
    size_t max_size = ggml_backend_buft_get_max_size(buft);

    ggml_backend_buffer_t * buffers = NULL;
    size_t n_buffers = 0;
```
```console
gdb) p alignment 
$71 = 32
(gdb) p max_size
$72 = 18446744073709551615
```

```c++
    ggml_backend_buffer_t * buffers = NULL;
    size_t n_buffers = 0;

    size_t cur_buf_size = 0;
    struct ggml_tensor * first = ggml_get_first_tensor(ctx);
```
`first` will be `cache_k_l0`:
```console
(gdb) p *first
$74 = {type = GGML_TYPE_F16, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0, ne = {4194304, 1, 1, 1}, nb = {2, 8388608, 8388608, 
    8388608}, op = GGML_OP_NONE, op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0, src = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 
    0x0, 0x0, 0x0, 0x0}, perf_runs = 0, perf_cycles = 0, perf_time_us = 0, view_src = 0x0, view_offs = 0, data = 0x0, 
  name = "cache_k_l0", '\000' <repeats 53 times>, extra = 0x0, padding = "\000\000\000\000\000\000\000"}
```

Then the following loop will iterate over the tensors in the passed in context
and calculate the size for the buffer:
```c++
    for (struct ggml_tensor * t = first; t != NULL; t = ggml_get_next_tensor(ctx, t)) {
        size_t this_size = 0;
        if (t->data == NULL && t->view_src == NULL) {
            this_size = GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
        }

        if (this_size > max_size) {
            ...
        }

        if ((cur_buf_size + this_size) > max_size) {
            // allocate tensors in the current buffer
            if (!alloc_tensor_range(ctx, first, t, buft, cur_buf_size, &buffers, &n_buffers)) {
                return NULL;
            }
            first = t;
            cur_buf_size = this_size;
        } else {
            cur_buf_size += this_size;
        }
    }
```
The second time throught the loop `t` will be:
```console
(gdb) p *t
$76 = {type = GGML_TYPE_F16, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0, ne = {4194304, 1, 1, 1}, nb = {2, 8388608, 8388608, 
    8388608}, op = GGML_OP_NONE, op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0, src = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 
    0x0, 0x0, 0x0, 0x0}, perf_runs = 0, perf_cycles = 0, perf_time_us = 0, view_src = 0x0, view_offs = 0, data = 0x0, 
  name = "cache_v_l0", '\000' <repeats 53 times>, extra = 0x0, padding = "\000\000\000\000\000\000\000"}
```
When the loop has completed the following wil be run:
```c++
    if (cur_buf_size > 0) {
        if (!alloc_tensor_range(ctx, first, NULL, buft, cur_buf_size, &buffers, &n_buffers)) {
            return NULL;
        }
    }
```
We can find `alloc_tensor_range` in the `ggml-alloc.c`::
```c
static bool alloc_tensor_range(struct ggml_context * ctx,
        struct ggml_tensor * first, struct ggml_tensor * last,
        ggml_backend_buffer_type_t buft, size_t size,
        ggml_backend_buffer_t ** buffers, size_t * n_buffers) {
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, size);
```
And in `ggml-backend.c` we have:
```c
GGML_CALL ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    return buft->iface.alloc_buffer(buft, size);
}
```
Which in our case will call the following function:
```c
GGML_CALL static ggml_backend_buffer_t ggml_backend_cpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    size += TENSOR_ALIGNMENT;   // malloc may return an address that is not aligned
    void * data = malloc(size); // TODO: use GGML_ALIGNED_MALLOC (move to ggml-impl.h)
    if (data == NULL) {
        fprintf(stderr, "%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }

    return ggml_backend_buffer_init(buft, cpu_backend_buffer_i, data, size);
}
```
Now, malloc will take the size (in bytes), so that will be:
```console
(gdb) p size / 1024.0/1024.0
$82 = 512.00003051757812
```
So 512MB will be allocated for the buffer. If we were using a different backend
for example CUDA this would instead be the function
`ggml_backend_cuda_buffer_type_alloc_buffer` in llama.cpp/ggml-backend.cu which
does `ggml_cuda_device_malloc` which allocates memory on the device instead.

The last call in the function is:
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

Following this code there is some logging of the memory usage:
```c++

        {
            size_t memory_size_k = 0;
            size_t memory_size_v = 0;

            for (auto & k : ctx->kv_self.k_l) {
                memory_size_k += ggml_nbytes(k);
            }

            for (auto & v : ctx->kv_self.v_l) {
                memory_size_v += ggml_nbytes(v);
            }

            LLAMA_LOG_INFO("%s: KV self size  = %7.2f MiB, K (%s): %7.2f MiB, V (%s): %7.2f MiB\n", __func__,
                (float)(memory_size_k + memory_size_v) / (1024.0f * 1024.0f),
                ggml_type_name(type_k), (float)memory_size_k / (1024.0f * 1024.0f),
                ggml_type_name(type_v), (float)memory_size_v / (1024.0f * 1024.0f));
        }
```
Next we have `llama_output_reserve` which is is called in a number of places,
so what does it actually do?
```c
            // resized during inference when a batch uses more outputs
            if (llama_output_reserve(*ctx, params.n_seq_max) < params.n_seq_max) {
                LLAMA_LOG_ERROR("%s: failed to reserve initial output buffer\n", __func__);
                llama_free(ctx);
                return nullptr;
            }
```
In this case the `n_seq_max` is 1:
```console
(gdb) p params.n_seq_max
$14 = 1
```
```c
            ctx->buf_compute_meta.resize(
                ggml_tensor_overhead()*LLAMA_MAX_NODES + 
                ggml_graph_overhead_custom(LLAMA_MAX_NODES, false));
            ctx->sched = ggml_backend_sched_new(ctx->backends.data(),
                backend_buft.data(),
                ctx->backends.size(),
                LLAMA_MAX_NODES,
                pipeline_parallel);
```
What is a GGML Backend Schuduler?   TODO: Look into this.


