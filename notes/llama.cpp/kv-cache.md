### Key Value Cache
So the key-value cache is a tensor that is stored on the backend which we can
think of like a large pre-allocated block of memory (tensor).

So lets start where the kv-cache is created, which happens when we create a new
llama-context. The llama-context constructor has the following call:
```c++
    // init the memory module
    if (!hparams.vocab_only) {
        llama_memory_params params_mem = {
            /*.type_k   =*/ params.type_k,
            /*.type_v   =*/ params.type_v,
            /*.swa_full =*/ params.swa_full,
        };

        memory.reset(model.create_memory(params_mem, cparams));
    }
```
So a model can have a create_memory function that creates a memory module. Not
all models need this, for example an embedding model does not need a kv-cache.
This function can be found in llama-model.cpp:
```c++
llama_memory_i * llama_model::create_memory(const llama_memory_params & params, const llama_cparams & cparams) const {
    llama_memory_i * res;

    switch (arch) {
        case LLM_ARCH_BERT:
        case LLM_ARCH_JINA_BERT_V2:
        case LLM_ARCH_JINA_BERT_V3:
        case LLM_ARCH_NOMIC_BERT:
        case LLM_ARCH_NOMIC_BERT_MOE:
        case LLM_ARCH_NEO_BERT:
        case LLM_ARCH_WAVTOKENIZER_DEC:
        case LLM_ARCH_MODERN_BERT:
        case LLM_ARCH_GEMMA_EMBEDDING:
        case LLM_ARCH_DREAM:
        case LLM_ARCH_LLADA:
        case LLM_ARCH_LLADA_MOE:
        case LLM_ARCH_RND1:
            {
                res = nullptr;
            } break;
        default:
            {
            ...
                        res = new llama_kv_cache(
                                *this,
                                params.type_k,
                                params.type_v,
                                !cparams.flash_attn,
                                cparams.offload_kqv,
                                cparams.kv_unified,
                                cparams.n_ctx_seq,
                                cparams.n_seq_max,
                                1,
                                hparams.n_swa,
                                hparams.swa_type,
                                nullptr,
                                nullptr);
                    }
```
```console
(gdb) p arch
$4 = LLM_ARCH_QWEN2
```

```c++
llama_kv_cache::llama_kv_cache(
        const llama_model & model,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   offload,
                     bool   unified,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
    const layer_filter_cb & filter,
    const  layer_reuse_cb & reuse) :
    model(model), hparams(model.hparams), v_trans(v_trans),
    n_seq_max(n_seq_max), n_stream(unified ? 1 : n_seq_max), n_pad(n_pad), n_swa(n_swa), swa_type(swa_type) {
    ...

    for (uint32_t il = 0; il < hparams.n_layer; il++) {
        if (!hparams.has_kv(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: does not have KV cache\n", __func__, il);
            continue;
        }
        ...
        const uint32_t n_embd_k_gqa =            hparams.n_embd_k_gqa(il);
        const uint32_t n_embd_v_gqa = !v_trans ? hparams.n_embd_v_gqa(il) : hparams.n_embd_v_gqa_max();

        const char * dev_name = "CPU";

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

        if (offload) {
            auto * dev = model.dev_layer(il);
            buft = ggml_backend_dev_buffer_type(dev);

            dev_name = ggml_backend_dev_name(dev);
        }

        ggml_context * ctx = ctx_for_buft(buft);

        ggml_tensor * k = ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream);
        ggml_tensor * v = ggml_new_tensor_3d(ctx, type_v, n_embd_v_gqa, kv_size, n_stream);

        ggml_format_name(k, "cache_k_l%d", il);
        ggml_format_name(v, "cache_v_l%d", il);
```

```console
(gdb) p kv_size
$11 = 32768

(gdb) p k->ne
$9 = {128, 32768, 1, 1}

(gdb) p v->ne
$10 = {128, 32768, 1, 1}

(gdb) up
(gdb) p cparams.n_ctx_seq
$13 = 32768
```
The `n_ctx_seq` parameter is the maximum context length for a sequence which we
use to create the k and v tensors.

Notice that the last dimension is `n_stream` which in the current case is just 1,
but if we had multiple sequences in the batch then there would be a stream for
each of them. Notice that n_streams depends on the `unified` parameter:
```c++
    n_seq_max(n_seq_max), n_stream(unified ? 1 : n_seq_max), n_pad(n_pad), n_swa(n_swa), swa_type(swa_type) {
        ...
        ggml_tensor * k = ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream);
```
So if unified is true then n_stream is always 1, and other wise it will be equals
to the maximum number of sequences in the batch.

So lets say we have unified is false and we have two sequences:
```console
K tensor shape: [128, 32768, 2]

s0
   0    [0  ... 127]
   1    [0  ... 127]
   2    [0  ... 127]
   ...
   32767 [0  ... 127]

s1
   0    [0  ... 127]
   1    [0  ... 127]
   2    [0  ... 127]
   ...
   32767 [0  ... 127]
```
Each sequence has its own 32768 positions reserved and sequence 0 always uses
stream 0 and sequence 1 always uses stream 1.

And if unified was true, and we still have two sequences, then we would have:
```console
K tensor shape: [128, 32768, 1]

   0    [0  ... 127]
   1    [0  ... 127]
   2    [0  ... 127]
   ...
   32767 [0  ... 127]
```
Both sequences share the same 32768 positions. Since both sequences share the
same we need a way to distinguish between them which is what slot_info does.

I'm not sure where to put this but it is somewhat related and this is about
what happens when the a cache tensor is full.
Imaginge we have a cache of size 4 and we have the following state:
```console
Physical Slot   Token    Logical Pos   Vector Content (Simplified)
Slot 0          "The"    Pos 0         Vector v rotated by 0∘
Slot 1          "cat"    Pos 1         Vector v rotated by 10∘
Slot 2          "sat"    Pos 2         Vector v rotated by 20∘
Slot 3          "on"     Pos 3         Vector v rotated by 30∘
```
When we decode a new token we don't have room in the cache. And lets say the next
token is "mat" (well the embeddings for "mat" that is). What we want is to evict
the first token "The" and insert "mat" in its place. But inserting and moving
is a costly operation. Nothing is moved, but the sequence should now be:
```console
tokens   : "cat sat on mat"
positions:   0   1   2  3
```
But the rotations for the vectors not correct now. For example "cat" was previously
at position 1 and had a rotation of 10∘, but now it is at position 0 and should
have a rotation of 0. We can first rotate all the vectors back by one step:
```console
Slot 0: "The" -> -10 (degrees)
Slot 1: "cat" -> 0   (degrees) Looks like the Pos 0 now.
Slot 2: "sat" -> 10  (degrees) Looks like the Pos 1 now.
Slot 3: "on"  -> 20  (degrees) Looks like the Pos 2 now.
```
We the calculate the vector for "mat" and rotate it for Pos 3 (30 degrees) and
overwrite Slot 0:
```console
Physical Slot   Token    Logical Pos   Vector Content (New state)
Slot 0          "mat"    Pos 3         Vector v rotated by 30∘
Slot 1          "cat"    Pos 0         Vector v rotated by 0∘
Slot 2          "sat"    Pos 1         Vector v rotated by 10∘
Slot 3          "on"     Pos 2         Vector v rotated by 20∘
```
The attention operation does not care about the physical order of the slots, it
just looks at the cache, it sees:
1. A token at Slot 1 that mathematically claims to be at Pos 0.
2. A token at Slot 2 that mathematically claims to be at Pos 1.
3. A token at Slot 3 that mathematically claims to be at Pos 2.
4. A token at Slot 0 that mathematically claims to be at Pos 3.

Notice that unified means that all sequences share the same k and v tensors.
But what does this mean in practice? Well first save some memory as we only have
half the number of values to store (in the case of two sequences).

Next we have the following:
```c++
        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;

        for (uint32_t s = 0; s < n_stream; ++s) {
            k_stream.push_back(ggml_view_2d(ctx, k, n_embd_k_gqa, kv_size, k->nb[1], s*k->nb[2]));
            v_stream.push_back(ggml_view_2d(ctx, v, n_embd_v_gqa, kv_size, v->nb[1], s*v->nb[2]));
        }
```
Notice that this is creating a vector of tensor views for each stream, and that
s is used as the offset into the 3rd dimension of the k and v tensors.

Next the map_layers_ids updated to insert (if the key does not exist it will
be created) the layer id and map it to the index in the layers vector:
```c++
        map_layer_ids[il] = layers.size();

        layers.push_back({ il, k, v, k_stream, v_stream, });
```
And the layers is vector of llama_kv_cache::kv_layer.
```c++
    struct kv_layer {
        // layer index in the model
        // note: can be different from the layer index in the KV cache
        uint32_t il;

        ggml_tensor * k;
        ggml_tensor * v;

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;
    };
```
So to recap, this is where the k and v tensors are created which will be done for
each layer. After all layers have been processed we have:
```c++
    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf;
        if (model.hparams.no_alloc) {
            buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                t->buffer = buf; // set dummy buffer for KV cache so that the backend scheduler won't try to allocate it
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft); // real buffer
        }
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for kv cache");
        }
        ...
```
This is where the actual memory for the kv-cache tensors is allocated on the
backend by allocating all the tensors in the context associated with the given
buffer type.

So that was how the kv-cache tensors are created and allocated and keep in mind
that there is one for each later in the model.

Next, when the model graph is build there are inputs created for the kv-cache.
For qwen2 which is the model that we are using:
```console
(gdb) p model.arch
$21 = LLM_ARCH_QWEN2
```
The function `llm_build_qwen2::llm_build_qwen2` looks like this (can be found in
src/models/qwen2.cpp):
```c++
llm_build_qwen2::llm_build_qwen2(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    ...
    auto * inp_attn = build_attn_inp_kv();
```

```c++
llm_graph_input_attn_kv * llm_graph_context::build_attn_inp_kv() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = build_attn_inp_kv_impl(ctx0, ubatch, hparams, cparams, mctx_cur);

    return (llm_graph_input_attn_kv *) res->add_input(std::move(inp));
}
```

```c++
static std::unique_ptr<llm_graph_input_attn_kv> build_attn_inp_kv_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_hparams & hparams,
    const llama_cparams & cparams,
    const llama_kv_cache_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_attn_kv>(hparams, cparams, mctx_cur);

    {
        GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_iswa for SWA");

        const auto n_kv     = mctx_cur->get_n_kv();
        const auto n_tokens = ubatch.n_tokens;
        const auto n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;

        inp->self_k_idxs = mctx_cur->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_cur->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_kv, n_tokens/n_stream, 1, n_stream);
        ggml_set_input(inp->self_kq_mask);

        inp->self_kq_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->self_kq_mask, GGML_TYPE_F16) : inp->self_kq_mask;
    }

    return inp;
}
```
So we can see that a unique_pointer that wraps `llm_graph_input_attn_kv` is
created and this will call the constructor of `llm_graph_input_attn_kv` which
just sets the passed in member fields.

```c++
ggml_tensor * llama_kv_cache_context::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_k_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    ggml_tensor * k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);

    ggml_set_input(k_idxs);

    return k_idxs;
}
```
So the above is just creating a 1d tensor with the same number of elements as
there are tokens in the current ubatch. And this tensor is marked a an input tensor.
And we do the same for the v indices tensor.

Next we create the kq mask tensor:
```c++
        inp->self_kq_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_kv, n_tokens/n_stream, 1, n_stream);
        ggml_set_input(inp->self_kq_mask);
```
Finally we a converted (cnv) version of the kq mask tensor if flash attention
is enabled:
```c++
        inp->self_kq_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->self_kq_mask, GGML_TYPE_F16) : inp->self_kq_mask;
```
This will then return to build_attn_inp_kv where the input is added to the
lla_graph_result:
```c++
llm_graph_input_attn_kv * llm_graph_context::build_attn_inp_kv() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = build_attn_inp_kv_impl(ctx0, ubatch, hparams, cparams, mctx_cur);

    return (llm_graph_input_attn_kv *) res->add_input(std::move(inp));
}
```
Now, later when process_ubatch is called it will call set_input on all the inputs
which includes our kv-cache input:
```c++
void llm_graph_input_attn_kv::set_input(const llama_ubatch * ubatch) {
    mctx->set_input_k_idxs(self_k_idxs, ubatch);
    mctx->set_input_v_idxs(self_v_idxs, ubatch);

    mctx->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
}
```
```c++
void llama_kv_cache_context::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_k_idxs(dst, ubatch, sinfos[i_cur]);
}
```
```console
$9 = (const llama_ubatch *) 0x555556d66f10
(gdb) p *ubatch
$10 = {b_equal_seqs = 0, n_tokens = 36, n_seq_tokens = 1, n_seqs = 36, n_seqs_unq = 1, n_pos = 1, token = 0x555556d66fd0,
  embd = 0x0, pos = 0x555556d6aef0, n_seq_id = 0x555556d6ce60, seq_id = 0x555555f00810, seq_id_unq = 0x5555584b65a0,
  seq_idx = 0x55555635c040, output = 0x5555584b6230 "", data = std::shared_ptr<llama_ubatch::data_t> (use count 3, weak count 0) = {
    get() = 0x5555563f4930}}
(gdb) p i_cur
$7 = 0
(gdb) p sinfos[0]
$8 = {s0 = 0, s1 = 0, strm = std::vector of length 1, capacity 1 = {0}, idxs = std::vector of length 1, capacity 1 = {
    std::vector of length 36, capacity 36 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
      24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35}}}

```
```c++
void llama_kv_cache::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;

    int64_t * data = (int64_t *) dst->data;  // dst is self_k_idxs

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        // base offset for this stream
        const int64_t offs = sinfo.strm[s]*get_size(); // get_size() is the kv cache size (32768 in this case)

        for (uint32_t i = 0; i < sinfo.size(); ++i) {
            data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
            // stream_id * cache_size + slot_index
        }
    }
}
```
So what we are doing here is that we are populating the self_k_idxs tensor with
the indices that tells the attention operation where in the kv-cache read/write
values. Recall that the kv-cache is the large pre-allocated tensor that holds all
the K for a layer in the model, assuming a unified cache here:
```console
K tensor shape: [128, 32768, 1]

   0    [0  ... 127]
   1    [0  ... 127]
   2    [0  ... 127]
   ...
   32767 [0  ... 127]
```
Now, there can be multiple sequences in the batch and they are all stored in the
same K tensor above. What we are calculating above is the actual real indices
into the K tensor for each token in the ubatch. If we had multiple sequences
than and lets say we have a context size of 512, the first sequence would use
0-511 and the second sequence would use 512-1023 and so on.

_wip_
TODO: continue with set_input_kq_mask explanation...


### llm_graph_input_attn_kv
```c++
class llm_graph_input_attn_kv : public llm_graph_input_i {
public:
    ...

    ggml_tensor * self_k_idxs = nullptr; // I64 [n_batch]
    ggml_tensor * self_v_idxs = nullptr; // I64 [n_batch] or [n_batch*n_embd_v_gqa]

    ggml_tensor * self_kq_mask     = nullptr; // F32 [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_cnv = nullptr; //     [n_kv, n_batch/n_stream, 1, n_stream]

}:
```
`self_k_idxs` stores indices telling the attention operation where in the KV
cache to read/write K values for this batch.

`self_v_idxs` stores indices telling the attention operation where in the KV
cache to read/write V values for this batch.

Lets take look at how the K values are set:
```c++
void llm_graph_input_attn_kv::set_input(const llama_ubatch * ubatch) {
    mctx->set_input_k_idxs(self_k_idxs, ubatch);
    mctx->set_input_v_idxs(self_v_idxs, ubatch);

    mctx->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
}
```
This will call:

```c++
void llama_kv_cache_context::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_k_idxs(dst, ubatch, sinfos[i_cur]);
}
```
Notice that `i_cur` is the index of the cur ubatch being processed.
This delegates  to the kv-cache itself:
```c++
void llama_kv_cache::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const int64_t offs = sinfo.strm[s]*get_size();

        for (uint32_t i = 0; i < sinfo.size(); ++i) {
            data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
        }
    }
}
```
