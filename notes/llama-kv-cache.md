## llama.cpp kv-cache notes

### Overview
I've gone through the theory of Key-Value caching in the transformer architecture
in [llama.md](llama.md). This document is a more detailed look at the
implementation of the key-value cache in the llama.cpp codebase.

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

Recall that there is a KV-Cache per layer in the transformer architecture. And
the cache is storing the output of the QK computation, and the output of the
value computation. And notice that there is a vector of `ggml_tensor` pointer
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

Now, lets create a watch point on `kv_self` so we can trace the interactions:
```console
(gdb) watch ctx.kv_self 
Watchpoint 2: ctx.kv_self
```

Further down in the code, the `kv_self` is initialized with:
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

    ...
    const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa() + hparams.n_embd_k_s();
}
```
So the number/size of grouped query attention embeddings for the keys matrix
will be:
```c++
    uint32_t n_embd_k_gqa() const { // dimension of key embeddings across all k-v heads
        return n_embd_head_k * n_head_kv;
    }
```

The actual sizes below will depend on the model used. The following are for
`llama-2-7b.Q4_0.gguf`:
```console
(gdb) p hparams.n_embd_head_k 
$5 = 128
(gdb) p hparams.n_head
$6 = 32
(gdb) p hparams.n_head_kv 
$7 = 32
(gdb) p n_layer
$8 = 32
```
The first update the cache is the following:
```c++
    cache.has_shift = false;

    // TODO: find a nicer way to add other recurrent model architectures
    cache.recurrent = model.arch == LLM_ARCH_MAMBA;
    cache.v_trans   = !cparams.flash_attn;
```

```console
(gdb) p model.arch
$10 = LLM_ARCH_LLAMA
(gdb) p cparams.flash_attn
$11 = false
```
Next we have:
```c++
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
(gdb) p kv_size
$12 = 1024
(gdb) p cache.cells.size()
$16 = 0
(gdb) f
#0  llama_kv_cache_init (cache=..., ctx=0x555555a4d530, type_k=GGML_TYPE_F16, type_v=GGML_TYPE_F16, kv_size=1024, offload=true)
    at llama.cpp:2662
2662	    cache.cells.resize(kv_size);
(gdb) n
2664	    if (cache.recurrent) {
(gdb) p cache.cells.size()
$17 = 1024
```
So we can see that we have 1024 cells in this cache.

Next, a map of `ggml_backend_buffer_type_t` and a count of the different types
of backend buffers for the kv cache. In our case `offload` is true so that is
the path that will be taken. But also notice that if this was not the case then
the default buffer type would be `llama_default_buffer_type_cpu(true)` would be
set as the key and the found 32.
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
After this the map looks like this:
```console
(gdb) p buft_layer_count
$25 = std::map with 1 element = {[0x555555978400 <ggml_backend_cpu_buffer_type>] = 32}
```
Next for each of the entries in the `buft_layer_count` map we create a context
for each buffer type, and in this case there is only one element, which has a
count of 32:
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
the number of layers in the model:
```c++
    cache.k_l.reserve(n_layer);
    cache.v_l.reserve(n_layer);
```
```console
(gdb) ptype cache.k_l
type = std::vector<ggml_tensor*>
```
So each of these vectors will be able to hold 32 `ggml_tensor` pointers, one
for each layer.
Next, these vectors are populated with the following:
```
    for (int i = 0; i < (int) n_layer; i++) {
        struct ggml_context * ctx = offload ? ctx_map.at(model.buft_layer[i].buft) : cache.ctxs.front();
        ggml_tensor * k = ggml_new_tensor_1d(ctx, type_k, n_embd_k_gqa*kv_size);
        ggml_tensor * v = ggml_new_tensor_1d(ctx, type_v, n_embd_v_gqa*kv_size);
        ggml_format_name(k, "cache_k_l%d", i);
        ggml_format_name(v, "cache_v_l%d", i);
        cache.k_l.push_back(k);
        cache.v_l.push_back(v);
    }
```
So each of the k tensors will be of size `n_embd_k_gqa*kv_size`:
```console
(gdb) p n_embd_k_gqa 
$39 = 4096
(gdb) p kv_size
$40 = 1024
```
And we can take a look at `k`:
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

After this we will return to `llama_new_context_with_model`:
```c++
        if (!llama_kv_cache_init(ctx->kv_self, ctx, type_k, type_v, kv_size, cparams.offload_kqv)) {
            LLAMA_LOG_ERROR("%s: llama_kv_cache_init() failed for self-attention cache\n", __func__);
            llama_free(ctx);
            return nullptr;
        }
```
```c
static bool llama_kv_cache_init(
             struct llama_kv_cache & cache,
               const llama_context * ctx,
                         ggml_type   type_k,
                         ggml_type   type_v,
                          uint32_t   kv_size,
                              bool   offload) {
    const llama_model & model = ctx->model;
    const llama_cparams & cparams = ctx->cparams;

    const struct llama_hparams & hparams = model.hparams;

    const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa() + hparams.n_embd_k_s();
    const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa() + hparams.n_embd_v_s();
```
Now the number of embedding in the key matrix for grouped query attention (gqa)
is:
```console
(gdb) p hparams.n_embd_k_gqa()
$5 = 2048
(gdb) p hparams.n_embd_k_s()
$8 = 0
```
And `n_embd_k_s` is the number of embeddings in the rolling state
embeddings, which is 0 in our case, this is used for SSM (like Mamba).
```c
    uint32_t n_embd_k_s() const { // dimension of the rolling state embeddings
        // corresponds to Mamba's conv_states size
        // TODO: maybe support other convolution strides than 1
        // NOTE: since the first column of the conv_state is shifted out each time, it's not actually needed
        return (ssm_d_conv > 0 ? ssm_d_conv - 1 : 0) * ssm_d_inner;
    }
```
Next we have a few more local variables:
```c
    const int64_t  n_layer      = hparams.n_layer;
    cache.has_shift = false;

    cache.recurrent = model.arch == LLM_ARCH_MAMBA;
    cache.v_trans   = !cparams.flash_attn;

    cache.head = 0;
    cache.size = kv_size;
    cache.used = 0;

    cache.type_k = type_k;
    cache.type_v = type_v;

    cache.cells.clear();
    cache.cells.resize(kv_size);
```
A little further down we have the following
```c
    cache.k_l.reserve(n_layer);
    cache.v_l.reserve(n_layer);

    for (int i = 0; i < (int) n_layer; i++) {
        struct ggml_context * ctx = offload ? ctx_map.at(model.buft_layer[i].buft) : cache.ctxs.front();
        ggml_tensor * k = ggml_new_tensor_1d(ctx, type_k, n_embd_k_gqa*kv_size);
        ggml_tensor * v = ggml_new_tensor_1d(ctx, type_v, n_embd_v_gqa*kv_size);
        ggml_format_name(k, "cache_k_l%d", i);
        ggml_format_name(v, "cache_v_l%d", i);
        cache.k_l.push_back(k);
        cache.v_l.push_back(v);
    }
```
For each layer in the current model we will create a new tensor for the keys
with the size of 
```console
(gdb) p n_embd_k_gqa
$5 = 2048
(gdb) p kv_size
$7 = 8192
```
`kv_size` is the number of tokens that the cache can hold, and each of these has
an embedding size of 2048. Think of this as there being 8192 rows and 2048
columns, but this is only a 1d list. 
This tensors in the context will then be created in the backend and this
function will return.

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
What is a GGML Backend Schuduler?   


Lets set a break point before `llama_decode` and see how this interacts with
the kv-cache. In `llama_decode_internal` we find the following:
```c
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
functions( TODO: updated this with details).




```c++

static bool llama_kv_cache_find_slot(
           struct llama_kv_cache & cache,
        const struct llama_batch & batch) {
    const uint32_t n_tokens = batch.n_tokens;

    if (cache.recurrent) {
        ...
    }

    while (true) {
        if (cache.head + n_tokens > cache.size) {
            ...
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
        ...
    }
```
Lets look what is happening in the for look, we know that `n_tokens` is 6 so we
will iteratate 0-5 times.
```console
(gdb) p cache.cells[cache.head + i]
$85 = {pos = -1, delta = 0, src = 0, seq_id = std::set with 0 elements}
```
So in our cache head is 0 and we are checking the first 6 cells in the cache to
see if their position is greater than 0. `found` will therefor still be true
and we will exit the loop:
```c++
    for (uint32_t i = 0; i < n_tokens; i++) {
        cache.cells[cache.head + i].pos = batch.pos[i];

        for (int32_t j = 0; j < batch.n_seq_id[i]; j++) {
            cache.cells[cache.head + i].seq_id.insert(batch.seq_id[i][j]);
        }
    }

    cache.used += n_tokens;

    return true;
```
We will again iterate over our 6 tokens.
```console
(gdb) p cache.cells[0]
$97 = {pos = 0, delta = 0, src = 0, seq_id = std::set with 0 elements}
```
We then also update the same cells sequence ids by looking at the number of
sequence ids the batch has which is one in our case:
```console
(gdb) p batch.n_seq_id[0]
$99 = 1
(gdb) p batch.seq_id[0][0]
$101 = 0
```
So we will set 
```console
(gdb) p cache.cells[0]
$102 = {pos = 0, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
```
So a cache cell contains the position of a token, and the sequence id(s).
Note that cache.head is not modified here. So the next iteration we will have
1=1 and the same will happen for that cache cell entry.
```console
(gdb) p cache.cells[0]
$109 = {pos = 0, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
(gdb) p cache.cells[1]
$110 = {pos = 1, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
(gdb) p cache.cells[2]
$111 = {pos = 2, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
(gdb) p cache.cells[3]
$112 = {pos = 3, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
(gdb) p cache.cells[4]
$113 = {pos = 4, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
(gdb) p cache.cells[5]
$114 = {pos = 5, delta = 0, src = 0, seq_id = std::set with 1 element = {[0] = 0}}
```
Finally we will update the `cache.used` to 6: and then return.


The kv-cache is first updated in by the call to `llama_decode` which calls
`llama_decode_internal` which calls `llama_kv_cache_find_slot` which will
update the cache

```console
665	   if (llama_decode(ctx, llama_batch_get_one(&embd[i], n_eval, n_past, 0))) {
(gdb) p ctx.kv_self.used
$47 = 0
(gdb) tbreak llama_decode
Temporary breakpoint 6 at 0x555555666b3d: file src/llama.cpp, line 19321.
(gdb) continue 
```
```c++
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
```
Before we enter the `llama_kv_cache_find_slot` we can inspect the cache:
```console
(gdb) p cache.used
$48 = 0
```

```
    for (uint32_t i = 0; i < n_tokens; i++) {
        cache.cells[cache.head + i].pos = batch.pos[i];

        for (int32_t j = 0; j < batch.n_seq_id[i]; j++) {
            cache.cells[cache.head + i].seq_id.insert(batch.seq_id[i][j]);
        }
    }

    cache.used += n_tokens;
```

TODO: look closer as `llama_set_inputs` and how the kv-cache is used there.
