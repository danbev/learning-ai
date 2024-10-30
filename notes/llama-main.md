## llama-cli (examples/main/main.cpp)
This page will go through and explain how examples/main/main.cpp in llama.cpp
works.

### Debugging
First build with debugging symbols enabled:
```console
$ make -j8 llama-cli LLAMA_DEBUG=1 GGML_CUDA=1
```

Then run the debugger:
```console
$ gdb --args ./llama-cli
$ gdb --args ./llama-cli -m models/gemma-2-9b-it.gguf -dkvc -ngl 15 -p "Dan loves icecream"
```
Lets start where the model and the context have been created:
```cpp
    std::tie(model, ctx) = llama_init_from_gpt_params(params);
```
(gdb) br main.cpp:207
```

#### prompt tokenization
The prompt will be tokenized using the following:
```c
    embd_inp = ::llama_tokenize(ctx, prompt, true, true);
```
It can be nice to inspect the tokens which can be done like this:
```console
(gdb) p embd_inp
$9 = std::vector of length 5, capacity 20 = {2, 7022, 16147, 8357, 35081}
```
And we can show the tokens as strings using:
```console
(gdb) p ctx.model.vocab.id_to_token[2]
$10 = {text = "<bos>", score = -1000, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) p ctx.model.vocab.id_to_token[7022]
$11 = {text = "Dan", score = -1000, attr = LLAMA_TOKEN_ATTR_NORMAL}
```

This following had me confused for a while and I opened a question eventually
to ask about it:
```cpp
    if ((int) embd_inp.size() > n_ctx - 4) {
        LOG_TEE("%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
        return 1;
    }
```
The response was the following:
```
The goal is to leave at least some context for the generation because if the
prompt fills the entire context then we can't generate new tokens.
```



LongRope is a technique that allows the model to handle sequences longer than
but this is done as a fine tuning step after the model has been trained.
Self-Extend does not require fine tuning.


### ...
The kv-cache is updated by `llama_decode_internal`:
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
Now, if there has been some update to the kv-cache, like setting the `has_shift`
flag or the `do_copy`the `llama_kv_cache_update` will perform updates. For the
initial prompt this will not be the case. So this would not do anything.
The `kv_self.head` and `kv_self.used` will also be 0 at this point.
Next we have `llama_kv_cache_find_slot` which will find a slot for the tokens
```c++
static bool llama_kv_cache_find_slot(
           struct llama_kv_cache & cache,
        const struct llama_batch & batch) {
    const uint32_t n_tokens = batch.n_tokens;
    // ignore the recurrent if clause for now.

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

    for (uint32_t i = 0; i < n_tokens; i++) {
        cache.cells[cache.head + i].pos = batch.pos[i];

        for (int32_t j = 0; j < batch.n_seq_id[i]; j++) {
            cache.cells[cache.head + i].seq_id.insert(batch.seq_id[i][j]);
        }
    }

    cache.used += n_tokens;

    return true;
```
Notice that the if statement in the while true loop checking that the number
of tokens will fit in the cache. If not, the head will be reset to 0 and the
loop will continue. And notice that `n_tested` is updated to the size of the
cache minus the head.

```console
(gdb) p cache.head
$9 = 0
(gdb) p cache.head + n_tokens
$10 = 2
(gdb) p cache.size
$11 = 8192
```c++
        bool found = true;
        for (uint32_t i = 0; i < n_tokens; i++) {
            if (cache.cells[cache.head + i].pos >= 0) {
                found = false;
                cache.head += i + 1;
                n_tested   += i + 1;
                break;
            }
        }
```
So the if statement looping over the number of tokens in the batch and checking
if the position in that cell is greater than or equal to 0 which means that is
not empty (-1).

This is not the case so we will break out of the loop.
So this is really checking that the cells at the current head are empty.

So `found` will still be true in this case and we will break out of the loop.
Next, we will iterate over all the tokens in the batch. 
```c++
    for (uint32_t i = 0; i < n_tokens; i++) {
        cache.cells[cache.head + i].pos = batch.pos[i];

        for (int32_t j = 0; j < batch.n_seq_id[i]; j++) {
            cache.cells[cache.head + i].seq_id.insert(batch.seq_id[i][j]);
        }
    }
```
Also, notice that the positions are the positions as they are in the batch, so
there is nothing related to self-extend here!

And update the position and the sequence id for each token in the batch.
After that cache.used will be updated and then we return true:
```c++
    cache.used += n_tokens;
```

Back in `llama_decode_internal` we have:
```c++

            if (!kv_self.recurrent) {
                // a heuristic, to avoid attending the full cache if it is not yet utilized
                // after enough generations, the benefit from this heuristic disappears
                // if we start defragmenting the cache, the benefit from this will be more important
                const uint32_t pad = llama_kv_cache_get_padding(cparams);
                kv_self.n = std::min(kv_self.size, std::max(pad, GGML_PAD(llama_kv_cache_cell_max(kv_self), pad)));
            }
```
```c++
static uint32_t llama_kv_cache_get_padding(const struct llama_cparams & cparams) {
    // the FA kernels require padding to avoid extra runtime boundary checks
    return cparams.flash_attn ? 256u : 32u;
}
```
```console
(gdb) p pad
$18 = 32
(gdb) p llama_kv_cache_cell_max(kv_self)
$19 = 2
(gdb) p kv_self.size
$20 = 8192
(gdb) p kv_self.n
$21 = 32
```

After the ggml compute graph has been built and computed we end up in:
```c++
        ggml_cgraph * gf = llama_build_graph(lctx, u_batch, false);

        llama_graph_compute(lctx, gf, n_threads);

        // update the kv ring buffer
        {
            kv_self.head += n_tokens;

            // Ensure kv cache head points to a valid index.
            if (kv_self.head >= kv_self.size) {
                kv_self.head = 0;
            }
        }
```
`llama_graph_compute` will build the computation graph
Both the Query and the Key cached will be roped:
```c++
                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
```

```console
(gdb) p *Qcur
$11 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0, ne = {2048, 2, 1, 1}, nb = {4, 8192, 
    16384, 16384}, op = GGML_OP_MUL_MAT, op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0, src = {
    0x55555bd630e0, 0x7fffcf51e8a0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0, view_offs = 0, 
  data = 0x0, name = "Qcur-0", '\000' <repeats 57 times>, extra = 0x0}
```
In this case there batch only contains two tokens as this is the warmup decode
but that does not matter. We can see that we have something like:
```
   0                                          2047
0  [...........................................]
1  [...........................................]
```
```console
(gdb) p n_embd_head
$12 = 64
(gdb) p n_head
$13 = 32
(gdb) p n_tokens
$14 = 2

(gdb) p *ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens)
$16 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0, ne = {64, 32, 2, 1}, nb = {4, 256, 
    8192, 16384}, op = GGML_OP_RESHAPE, op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0, src = {
    0x7fffcf51ea10, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x7fffcf51ea10, view_offs = 0, 
  data = 0x0, name = "Qcur-0 (reshaped)", '\000' <repeats 46 times>, extra = 0x0}
```
```
   0           64 
0  [...........]
         .
         .        /
         .       /
31 [...........]/
                0
32*64 = 2048
```
So this is setting up the computation graph and the above reshaped tensor will
later be updated with values./

```console
(gdb) p *inp_pos
$18 = {type = GGML_TYPE_I32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0, ne = {2, 1, 1, 1}, nb = {4, 8, 8, 
    8}, op = GGML_OP_NONE, op_params = {0 <repeats 16 times>}, flags = 1, grad = 0x0, src = {0x0, 0x0, 0x0, 0x0, 
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0, view_offs = 0, data = 0x0, 
  name = "inp_pos", '\000' <repeats 56 times>, extra = 0x0}
```

```c++
                cur = llm_build_kv(ctx0, model, hparams, cparams, kv_self, gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
```


From `build_gemma2`:
```c++
                cur = llm_build_kv(ctx0,
                                   model,
                                   hparams,
                                   cparams,
                                   kv_self,
                                   gf,
                                   model.layers[il].wo,
                                   NULL,
                                   Kcur,
                                   Vcur,
                                   Qcur,
                                   KQ_mask_l,
                                   n_tokens,
                                   kv_head,
                                   n_kv,
                                   1.0f,
                                   cb,
                                   il);
```
```c++
static struct ggml_tensor * llm_build_kv(
        struct ggml_context * ctx,
          const llama_model & model,
        const llama_hparams & hparams,
        const llama_cparams & cparams,
       const llama_kv_cache & kv,
         struct ggml_cgraph * graph,
         struct ggml_tensor * wo,
         struct ggml_tensor * wo_b,
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

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(graph, q_cur);
    ggml_build_forward_expand(graph, k_cur);
    ggml_build_forward_expand(graph, v_cur);

    llm_build_kv_store(ctx, hparams, cparams, kv, graph, k_cur, v_cur, n_tokens, kv_head, cb, il);

    struct ggml_tensor * cur;

    cur  = llm_build_kqv(ctx, model, hparams, cparams, kv, graph, wo, wo_b,
            q_cur, kq_mask, n_tokens, n_kv, kq_scale, cb, il);
    cb(cur, "kqv_out", il);

    return cur;
}
```
What does `ggml_build_forward_expand` do?
This will basically go through the passed in tensor and add it to the graph         
and then visit its parents (the src[]). These will then be added to the         
cgraph and will have been also added to the hashset.
I don't quite understand the part about reordering and reducing the number of
splits in the graph.


```c
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



    struct ggml_tensor * k_cache_view = ggml_view_1d(ctx, kv.k_l[il], n_tokens*n_embd_k_gqa,
            (ggml_row_size(kv.k_l[il]->type, n_embd_k_gqa))*kv_head);
    cb(k_cache_view, "k_cache_view", il);
```
So the above is setting up the nodes/leafs in the computation graph.

Back in `llama_decode_internal` we then have the following:
```c
        ggml_backend_sched_alloc_graph(lctx.sched, gf);

        llama_set_inputs(lctx, u_batch);

        llama_graph_compute(lctx, gf, n_threads);
```
Lets take a closer look at `llama_set_inputs`:
```c
static void llama_set_inputs(llama_context & lctx, const llama_batch & batch) {
    const auto & hparams = lctx.model.hparams;
    const auto & cparams = lctx.cparams;
    const auto & kv_self = lctx.kv_self;

    if (batch.token) {
        const int64_t n_tokens = batch.n_tokens;

        ggml_backend_tensor_set(lctx.inp_tokens, batch.token, 0, n_tokens*ggml_element_size(lctx.inp_tokens));
    }
```
So the tensor that will be updates is `lctx.inp_tokens` and the data will be
from the batch.token and the size will be the number of tokens times the size
of the elements in the tensor. The 0 is the offset.
```console
(gdb) p ggml_element_size(lctx.inp_tokens)
$56 = 4
(gdb) p n_tokens
$57 = 5
```

```c
GGML_CALL void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf != NULL && "tensor buffer not set");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    if (!size) {
        return;
    }

    buf->iface.set_tensor(buf, tensor, data, offset, size);
}
```
So this will just end up in a memcpy:
```c
GGML_CALL static void ggml_backend_cpu_buffer_set_tensor(
    ggml_backend_buffer_t buffer,
    struct ggml_tensor * tensor,
    const void * data,
    size_t offset,
    size_t size) {

    memcpy((char *)tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}
```
So that took care of the token values, that is the ids (token ids) of the
tokens.

Next we will do something similar for the positions of the tokens in the batch:
(still in `llama_set_inputs`);
```c
    if (batch.pos && lctx.inp_pos) {
        const int64_t n_tokens = batch.n_tokens;

        ggml_backend_tensor_set(lctx.inp_pos, batch.pos, 0, n_tokens*ggml_element_size(lctx.inp_pos));
    }
```
```console
(gdb) p batch.pos[0]
$78 = 0
(gdb) p batch.pos[1]
$79 = 1
(gdb) p batch.pos[2]
$80 = 2
(gdb) p batch.pos[3]
$81 = 3
(gdb) p batch.pos[4]
$82 = 4
(gdb) p batch.pos[5]
$83 = 6648929
(gdb) p ggml_element_size(lctx.inp_pos)
$84 = 4
```

Next we have (note that `is_encoding` would be true is the model was and
encoder-decoder model like T5): 
```c++
    if (lctx.inp_KQ_mask) {
        // NOTE: hparams.causal_attn indicates the model is capable of generation and uses the kv cache.
        if (cparams.causal_attn && !lctx.is_encoding) {
            const int64_t n_kv     = kv_self.n;
            const int64_t n_tokens = batch.n_tokens;

            GGML_ASSERT(ggml_backend_buffer_is_host(lctx.inp_KQ_mask->buffer));

            float * data     = (float *) lctx.inp_KQ_mask->data;
            float * data_swa = nullptr;

            if (lctx.inp_KQ_mask_swa) {
                data_swa = (float *) lctx.inp_KQ_mask_swa->data;
            }

            // For causal attention, use only the previous KV cells
            // of the correct sequence for each token of the batch.
            // It's assumed that if a token in the batch has multiple sequences, they are equivalent.
            for (int h = 0; h < 1; ++h) {
                for (int j = 0; j < n_tokens; ++j) {
                    const llama_pos    pos    = batch.pos[j];
                    const llama_seq_id seq_id = batch.seq_id[j][0];

                    for (int i = 0; i < n_kv; ++i) {
                        float f;
                        if (!lctx.kv_self.cells[i].has_seq_id(seq_id) || lctx.kv_self.cells[i].pos > pos) {
                            f = -INFINITY;
                        } else {
                            if (hparams.use_alibi) {
                                f = -fabs(lctx.kv_self.cells[i].pos - pos);
                            } else {
                                f = 0.0f;
                            }
                        }
                        data[h*(n_kv*n_tokens) + j*n_kv + i] = f;

                        // may need to cut off old tokens for sliding window
                        if (data_swa) {
                            if (pos - lctx.kv_self.cells[i].pos >= (int32_t)hparams.n_swa) {
                                f = -INFINITY;
                            }
                            data_swa[h*(n_kv*n_tokens) + j*n_kv + i] = f;
                        }
                    }
                }

                for (int i = n_tokens; i < GGML_PAD(n_tokens, GGML_KQ_MASK_PAD); ++i) {
                    for (int j = 0; j < n_kv; ++j) {
                        data[h*(n_kv*n_tokens) + i*n_kv + j] = -INFINITY;
                    }
                }
            }
```
The for loop with the `h` index looks a little odd to me. This index will be
inialized to 0 and then the loop will run once. This value, 0, is also used
in a few calculations in the code which could be remove as they will always be
zero. But lets think about what is happening here. The inner for loop is going
to iterate over all the tokens in the batch and then for each token it will
iterate over the number of kv_self.n which in this case is 32. 'f' will be 0.0f
in our case and then the inp_KQ_mask will be updated with that value:
```c++
                        data[h*(n_kv*n_tokens) + j*n_kv + i] = f;
```
But notice that `h*(n_kv*n_tokens)` will always be 0 and could possibly be
removed.
The next time through the loop i will be 1 and this will cause and the current
pos is 0, so the first if statement will be entered and f set to -INFINITY. And
this makes sense if we think about it. For the first token is should not attend
to any tokens ahead of it. So the next value in inp_KQ_mask will be -INFINITY.
And this will happen for all values up to n_kv (32).
This will build up a mask tensor matrix that looks likes something like this:
```
   0                                           31
   +----+-----+-----+---------------------------+
   | 0  |~inf |~inf | ...                       |
   | 0  |  0  |~inf |                           |
   |                                            |
   |                                            |
   |                                            |
   |                                            |
   |                                            |
   |                                            |
   |                                            |
   +--------------------------------------------+
 31
```
After that and having gone through and creating the mask for the tokens in the
batch there might be more slots in the mask matrix that need to be filled which
is why the following will start at n_tokens and for each them set the values
to ~inf:
```++
                for (int i = n_tokens; i < GGML_PAD(n_tokens, GGML_KQ_MASK_PAD); ++i) {
                    for (int j = 0; j < n_kv; ++j) {
                        data[h*(n_kv*n_tokens) + i*n_kv + j] = -INFINITY;
                    }
                }
```
And in our case that is what llama_set_inputs does.


```console
(gdb) p n_kv
$95 = 32
```

```c
    if (lctx.inp_KQ_mask) {
        // NOTE: hparams.causal_attn indicates the model is capable of generation and uses the kv cache.
        if (cparams.causal_attn) {
            const int64_t n_kv     = kv_self.n;
            const int64_t n_tokens = batch.n_tokens;

            GGML_ASSERT(ggml_backend_buffer_is_host(lctx.inp_KQ_mask->buffer));

            float * data     = (float *) lctx.inp_KQ_mask->data;
            float * data_swa = nullptr;

            if (lctx.inp_KQ_mask_swa) {
                data_swa = (float *) lctx.inp_KQ_mask_swa->data;
            }


            for (int h = 0; h < 1; ++h) {
                for (int j = 0; j < n_tokens; ++j) {
                    const llama_pos    pos    = batch.pos[j];
                    const llama_seq_id seq_id = batch.seq_id[j][0];

                    for (int i = 0; i < n_kv; ++i) {
                        float f;
                        if (!lctx.kv_self.cells[i].has_seq_id(seq_id) || lctx.kv_self.cells[i].pos > pos) {
                            f = -INFINITY;
                        } else {
                            if (hparams.use_alibi) {
                                f = -fabs(lctx.kv_self.cells[i].pos - pos);
                            } else {
                                f = 0.0f;
                            }
                        }
                        data[h*(n_kv*n_tokens) + j*n_kv + i] = f;

                        // may need to cut off old tokens for sliding window
                        if (data_swa) {
                            if (pos - lctx.kv_self.cells[i].pos >= (int32_t)hparams.n_swa) {
                                f = -INFINITY;
                            }
                            data_swa[h*(n_kv*n_tokens) + j*n_kv + i] = f;
                        }
                    }
                }

                for (int i = n_tokens; i < GGML_PAD(n_tokens, GGML_KQ_MASK_PAD); ++i) {
                    for (int j = 0; j < n_kv; ++j) {
                        data[h*(n_kv*n_tokens) + i*n_kv + j] = -INFINITY;
                    }
                }
            }
```
Now, this is going through all the tokens in the batch, and then for each token
going though the 32 entries in the kv cache ( only 32?). If the current cells
does not have the same `seq_id` as the current token, or if the current cell is
occupied then f wil be set to -INFINITY. Otherwise it will be set to 0.0f.
What is happening on this line:
```console
                        data[h*(n_kv*n_tokens) + j*n_kv + i] = f;
```
For the first iteration this will set the `inp_KQ_mask` tensor value to 0.0f.
The second time through the inner "kv" loop we will check the next cell but this
time the cell's pos will be greater that the current pos, which is 0, so this
time f will be set to -INFINITY (masked out). And this makes sense that for the
first token it should only attent to itself and not the tokens ahead/infront off
it.
After that all the tokens from `n_tokens` to the end will be set to -INFINITY
and therefor masked out.

Back in `llama_decode_internal` we are now ready to compute the graph:

```c
        llama_set_inputs(lctx, u_batch);

        llama_graph_compute(lctx, gf, n_threads);
```

In llama_decode_internal we have the following function which comes before
llama_kv_cache_find_slot:
```c++

        // non-causal masks do not use the KV cache
        if (hparams.causal_attn) {
            llama_kv_cache_update(&lctx);
```
```c++
void llama_kv_cache_update(struct llama_context * ctx) {
    llama_kv_cache_update_internal(*ctx);
}

static void llama_kv_cache_update_internal(struct llama_context & lctx) {
    bool need_reserve = false;

    // apply K-shift if needed
    if (lctx.model.hparams.rope_type != LLAMA_ROPE_TYPE_NONE && lctx.kv_self.has_shift) {
        {
            ggml_backend_sched_reset(lctx.sched);

            ggml_cgraph * gf = llama_build_graph_k_shift(lctx);

            ggml_backend_sched_alloc_graph(lctx.sched, gf);

            llama_set_k_shift(lctx);

            llama_graph_compute(lctx, gf, lctx.cparams.n_threads);

            need_reserve = true;
        }

        {
            auto & kv_self = lctx.kv_self;

            kv_self.has_shift = false;

            for (uint32_t i = 0; i < kv_self.size; ++i) {
                kv_self.cells[i].delta = 0;
            }
        }
    }
```
```c++
static struct ggml_cgraph * llama_build_graph_k_shift(llama_context & lctx) {
    llama_batch dummy;
    dummy.n_tokens = 0;

    llm_build_cb cb = [&](struct ggml_tensor * , const char * , int ) { };

    struct llm_build_context llm(lctx, dummy, cb, false);

    llm.init();

    struct ggml_cgraph * result = llm.build_k_shift();

    llm.free();

    return result;
}

    struct ggml_cgraph * build_k_shift() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, LLAMA_MAX_NODES, false);

        GGML_ASSERT(kv_self.size == n_ctx);

        lctx.inp_K_shift = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_ctx);
        cb(lctx.inp_K_shift, "K_shift", -1);
        ggml_set_input(lctx.inp_K_shift);

        for (int il = 0; il < n_layer; ++il) {
            const int64_t n_head_kv = hparams.n_head_kv(il);
            const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);
            struct ggml_tensor * rope_factors = build_rope_factors(il);
            struct ggml_tensor * tmp =
                // we rotate only the first n_rot dimensions
                ggml_rope_ext_inplace(ctx0,
                        ggml_view_3d(ctx0, kv_self.k_l[il],
                            n_embd_head_k, n_head_kv, n_ctx,
                            ggml_row_size(kv_self.k_l[il]->type, n_embd_head_k),
                            ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa),
                            0),
                        lctx.inp_K_shift, rope_factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);

            cb(tmp, "K_shifted", il);
            ggml_build_forward_expand(gf, tmp);
        }

        return gf;
    }

    struct ggml_tensor * build_rope_factors(int il) {
        // choose long/short freq factors based on the context size
        const auto n_ctx_pre_seq = cparams.n_ctx / cparams.n_seq_max;

        if (n_ctx_pre_seq > hparams.n_ctx_orig_yarn) {
            return model.layers[il].rope_long;
        }

        return model.layers[il].rope_short;
    }
```
Is this `build_rope_factors` an impl. of LongRope?

```console
(gdb) p *ggml_view_3d(ctx0, kv_self.k_l[il],n_embd_head_k, n_head_kv, n_ctx, ggml_row_size(kv_self.k_l[il]->type, n_embd_head_k), ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa), 0)
$79 = {type = GGML_TYPE_F16, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0, ne = {64, 4, 8000, 1}, nb = {2, 128, 
    512, 4096000}, op = GGML_OP_VIEW, op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0, src = {
    0x55555bd47850, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x55555bd47850, view_offs = 0, 
  data = 0x7fffa08d2020, name = "cache_k_l0 (view)", '\000' <repeats 46 times>, extra = 0x0}
(gdb) p kv_self.k_l[il]
$80 = (ggml_tensor *) 0x55555bd47850
(gdb) p *kv_self.k_l[il]
$81 = {type = GGML_TYPE_F16, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x55555bd39490, ne = {2048000, 1, 1, 1}, 
  nb = {2, 4096000, 4096000, 4096000}, op = GGML_OP_NONE, op_params = {0 <repeats 16 times>}, flags = 0, 
  grad = 0x0, src = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0, view_offs = 0, 
  data = 0x7fffa08d2020, name = "cache_k_l0", '\000' <repeats 53 times>, extra = 0x0}
```

After that we will have a call to `llama_set_k_shift`:
```c++
            ggml_cgraph * gf = llama_build_graph_k_shift(lctx);

            ggml_backend_sched_alloc_graph(lctx.sched, gf);

            llama_set_k_shift(lctx);
```

```c++
static void llama_set_k_shift(llama_context & lctx) {
    const int64_t kv_size = lctx.kv_self.size;

    assert(ggml_backend_buffer_is_host(lctx.inp_K_shift->buffer));

    int32_t * data = (int32_t *) lctx.inp_K_shift->data;

    for (int i = 0; i < kv_size; ++i) {
        data[i] = lctx.kv_self.cells[i].delta;
    }
}
```
Notice that this is getting the data member from the inp_K_shift tensor and
and then iterating through number of cache elements. And it is using the delta
that we updated ealier in the `ga_n` block!So I think this is how the deltas are
used.
TODO: take a closer look at how inp_K_shift is used in the computation
graph. So I actually missed this when going through the code above but this
tensor is used here:
```c++
            struct ggml_tensor * tmp =
                // we rotate only the first n_rot dimensions
                ggml_rope_ext_inplace(ctx0,
                        ggml_view_3d(ctx0, kv_self.k_l[il],
                            n_embd_head_k, n_head_kv, n_ctx,
                            ggml_row_size(kv_self.k_l[il]->type, n_embd_head_k),
                            ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa),
                            0),
                        lctx.inp_K_shift, rope_factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
```
The first tensor passed to `ggml_rope_ext_inplace` is the tensor to be rotated
the second is the tensor containing the positions. This will be set as src1 for
this operation (remember that this is only setting up the computation graphs and
that the actual operation is performed later during the forward pass.

Lets set a break point in `ggml_compute_forward_rope_f32` to see how the b
tensor above is used.

```console
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    ...

    const int32_t * pos = (const int32_t *) src1->data;

    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            const int64_t p = pos[i2];
```
So the above is looping over 
```console
(gdb) p src0.ne[3]
$109 = 1
```
And the looping over `src0.ne[2]` which is 512.
```console
(gdb) p *src0
$105 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x55555bd3a0a0, ne = {64, 32, 512, 1}, 
  nb = {4, 256, 8192, 4194304}, op = GGML_OP_RESHAPE, op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0, 
  src = {0x7fffcf51ea10, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x7fffcf51ea10, view_offs = 0, 
  data = 0x7fff80cd1820, name = "Qcur-0 (reshaped)", '\000' <repeats 46 times>, extra = 0x0}
```

```console
(gdb) p *src1
$115 = {type = GGML_TYPE_I32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x55555bd3a0a0, ne = {512, 1, 1, 1}, 
  nb = {4, 2048, 2048, 2048}, op = GGML_OP_NONE, op_params = {0 <repeats 16 times>}, flags = 1, grad = 0x0, 
  src = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0, view_offs = 0, 
  data = 0x7fff7f530820, name = "inp_pos", '\000' <repeats 56 times>, extra = 0x0}
```



### `ggml_rope_ext`
```c
                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);
```
Lets start by focusing on the second argument which is `a` and this would be
the tensor that the rotation should be applied to. This tensor is first
reshaped to a 3D tensor:
```console
(gdb) p *Qcur 
$2 = {type = GGML_TYPE_F32,
backend = GGML_BACKEND_TYPE_CPU, 
buffer = 0x0,
ne = {4096, 512, 1, 1},
nb = {4, 16384, 8388608, 8388608},
op = GGML_OP_MUL_MAT, op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0, 
src = {0x555558423910, 0x7fffcf51e8a0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0, view_offs = 0, 
data = 0x0, name = "Qcur-0", '\000' <repeats 57 times>, extra = 0x0}
```
``` 
                    QCur
     0                                         4095
   0 +-------------------------------------------+
     |                                           |
     |                                           |
     |                                           |
     |                                           |
     |                                           |
     |                                           |
     |                                           |
     |                                           |
     |                                           |
 511 +-------------------------------------------+
```
Lets look at the reshaping:
```c
    ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens)
```
```console
(gdb) p n_embd_head
$8 = 128
(gdb) p n_head
$9 = 32
(gdb) p n_tokens
$10 = 512
```
So that becomes:
```c
    ggml_reshape_3d(ctx0, Qcur, 128, 32, 512)
```
And notice what we have split the dimensions which were 4096 into 128x32 (4096)
```
         /--------------------------+ 0
        /                          /
       /                          /
     0/                   127    /
  0  +---------------------+    /
     |                     |   /
     |                     |  /
     |                     | /
     |                     |/
  32 +---------------------+ 511
```
So we are reshaping Qcur to the above dimensions before calling rope.
The signagure for `ggml_rope_ext` is:
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
So Qcur is a, b is `inp_pos`. `c` is null.


```c
static void ggml_compute_forward_rope_f32(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst,
        const bool forward) {
    ...
    const float theta_scale = powf(freq_base, -2.0f/n_dims);
```
So, we can see here that the `freq_base` is used to calculate the `theta_scale`
and notice that this the same as specified in the vanilla RoPE paper where
we take 10000^(-2/d). And we can see what `n_dims` is used for. 
```c
    const int32_t * pos = (const int32_t *) src1->data;
```
And here we can see that the tensor `b` is the position tensor which makes
sense as it's dimension matches the embedding dimension (512 in this case).

