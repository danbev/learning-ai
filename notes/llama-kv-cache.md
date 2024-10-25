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
the number of layers in the model:
```c++
    cache.k_l.reserve(n_layer);
    cache.v_l.reserve(n_layer);
```
And these vectors store elements of type `ggml_tensor`:
```console
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
dimensions for the Key matrix for grouped query attention. Notice that we are
passing in the layer which sounds like there can be different embedding sizes
for different layers.
```c++
    uint32_t n_embd_k_gqa(uint32_t il = 0) const { // dimension of key embeddings across all k-v heads
        const uint32_t n_head_kv = this->n_head_kv(il);

        return n_embd_head_k * n_head_kv;
    }

    uint32_t n_head_kv(uint32_t il = 0) const {
        if (il < n_layer) {
            return n_head_kv_arr[il];
        }

        GGML_ABORT("fatal error");
    }
```
```console
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
a 1d tensor of size `n_embd_k_gqa*kv_size` (4096*1024) for the key and the value
tensors.

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
And these tensors will then be added to the `cache.k_l` and `cache.v_l` vectors:
```c++
        cache.k_l.push_back(k);
        cache.v_l.push_back(v);
```
So each layer will have a key tensor which has 4194304 elements which we can
think of as beeing 1024 rows with 4096 dimensions in each. One row for each
entry in the cache. 4096 is the embedding dimenesion size.

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
functions. TODO: Incorporate notes from self-extend which I think went through
this process in some detail.
And this stage the cache is empty and head is zero so lets look at find slot:
(in this case n_tokens is 6 and cache size is 1024)
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

_wip_ Figuring out what parts to keep in the above and perhaps replace with
that follows:

Let's take a look at how this is implemented in llama.cpp. I'll be using
simple-prompt to demonstrate this.
```console
$ gdb --args ./simple-prompt
(gdb) br simple-prompt.cpp:34
(gdb) r
(gdb) f
#1  0x0000000000408fab in main (argc=1, argv=0x7fffffffd198) at src/simple-prompt.cpp:34
34	    llama_context * ctx = llama_new_context_with_model(model, ctx_params);
(gdb) s
gdb) l
8736	        return nullptr;
8737	    }
8738	
8739	    llama_context * ctx = new llama_context(*model);
```

```c
        if (!llama_kv_cache_init(ctx->kv_self, ctx, type_k, type_v, kv_size, cparams.offload_kqv)) {
            LLAMA_LOG_ERROR("%s: llama_kv_cache_init() failed for self-attention cache\n", __func__);
            llama_free(ctx);
            return nullptr;
        }
```
```console
(gdb) p type_k
$11 = GGML_TYPE_F16
(gdb) p type_v
$12 = GGML_TYPE_F16
(gdb) p kv_size
$13 = 1024
(gdb) p cparams.offload_kqv 
$14 = true
(gdb) p ctx->kv_self
$15 = {has_shift = false, do_defrag = false, do_copy = false, recurrent = false,
v_trans = true, head = 0, size = 0, used = 0, n = 0,
type_k = GGML_TYPE_F16,
type_v = GGML_TYPE_F16,
cells = std::vector of length 0, capacity 0,
k_l = std::vector of length 0, capacity 0,
v_l = std::vector of length 0, capacity 0,
ctxs = std::vector of length 0, capacity 0, 
bufs = std::vector of length 0, capacity 0}
```
```c++
static bool llama_kv_cache_init(
             struct llama_kv_cache & cache,
               const llama_context * ctx,
                         ggml_type   type_k,
                         ggml_type   type_v,
                          uint32_t   kv_size,
                              bool   offload) {
    ...
    const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa() + hparams.n_embd_k_s();
    const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa() + hparams.n_embd_v_s();
    const int64_t  n_layer      = hparams.n_layer;
```
```console
(gdb) p n_embd_k_gqa 
$19 = 5120
(gdb) p n_embd_v_gqa 
$20 = 5120
(gdb) p n_layer
$21 = 40
```

```c++
    cache.head = 0;
    cache.size = kv_size;
    cache.used = 0;

    cache.type_k = type_k;
    cache.type_v = type_v;

    cache.cells.clear();
    cache.cells.resize(kv_size);
```
After the resize the cache cells  will contain 1024 cells:
```console
(gdb) p cache.cells.size()
$28 = 1024
```
```c++
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



At this stage the kv_self is uninitialized. We can inspect this struct using:
```console
gdb) ptype ctx.kv_self
type = struct llama_kv_cache {
    bool has_shift;
    bool do_defrag;
    bool do_copy;
    bool recurrent;
    uint32_t head;
    uint32_t size;
    uint32_t used;
    uint32_t n;
    ggml_type type_k;
    ggml_type type_v;
    std::vector<llama_kv_cell> cells;
    std::vector<ggml_tensor*> k_l;
    std::vector<ggml_tensor*> v_l;
    std::vector<ggml_context*> ctxs;
    std::vector<ggml_backend_buffer*> bufs;
  public:
    size_t total_size(void) const;
    ~llama_kv_cache(void);
}
```
We can see we have two vectors of  pointers to ggml_tensor. These are the key
and value matrices for each layer.

A little further down we have:
```console
(gdb) s
8789 if (!llama_kv_cache_init(ctx->model.hparams, ctx->kv_self, memory_type, cparams.n_ctx, model->n_gpu_layers)) {
Notice that we are passing in `ctx->kv_self`, and the cparams.n_ctx which is
the context length set to 1024 in this case.
```console
(gdb) s
1518	static bool llama_kv_cache_init(
(gdb) l
1519	        const struct llama_hparams & hparams,
1520	             struct llama_kv_cache & cache,
1521	                         ggml_type   wtype,
1522	                          uint32_t   n_ctx,
1523	                               int   n_gpu_layers) {
1524	    const uint32_t n_embd  = hparams.n_embd_gqa();
1525	    const uint32_t n_layer = hparams.n_layer;
1526	
1527	    const int64_t n_mem      = n_layer*n_ctx;
1528	    const int64_t n_elements = n_embd*n_mem;
```
The first line had be asking what `gqa` is and I think this stands for
grouped query attention. 
```console
(gdb) s
1532	    cache.head = 0;
(gdb) s
1533	    cache.size = n_ctx;
(gdb) s
1534	    cache.used = 0;
(gdb) s
1536	    cache.cells.clear();
(gdb) p cache
$20 = (llama_kv_cache &) @0xc624c8: {has_shift = false, head = 0, size = 1024, used = 0, n = 0, 
  cells = std::vector of length 0, capacity 0, k = 0x0, v = 0x0, ctx = 0x0, buf = {data = 0x0, size = 0, 
    fallback = false}}

1536	    cache.cells.clear();
(gdb) n
1537	    cache.cells.resize(n_ctx);
(gdb) n
(gdb) p cache
$21 = (llama_kv_cache &) @0xc624c8: {has_shift = false, head = 0, size = 1024, used = 0, n = 0, 
  cells = std::vector of length 1024, capacity 1024 = {{pos = -1, delta = 0, seq_id = std::set with 0 elements}, {
      pos = -1, delta = 0, seq_id = std::set with 0 elements}, {pos = -1, delta = 0, 
      seq_id = std::set with 0 elements}, {pos = -1, delta = 0, seq_id = std::set with 0 elements}, {pos = -1, 
      ...
1544	    params.mem_buffer = cache.buf.data;
(gdb) s
1545	    params.no_alloc   = false;
(gdb) s
1547	    cache.ctx = ggml_init(params);
```
So we can see here that we are going to initialize context for ggml. I did not
notice that `ggml_context *ctx` was a member of `llama_kv_cache`.
Next we are going to create a one dimensional tensor of GGML_TYPE_F16 (half
precision float) with 209715200 elements.
```console
1554	    cache.k = ggml_new_tensor_1d(cache.ctx, wtype, n_elements);
(gdb) p n_elements
$31 = 209715200
(gdb) p wtype
$32 = GGML_TYPE_F16
```
Hmm, the size of the tensor don't make sense to me yet. The 1d tensor is like
a list of number and it's size is 209715200. And the type of these slots is
F16 so that would be 2 bytes per slot, so 16 bytes per slot.
```console
llama_new_context_with_model: kv self size  =  800.00 MiB
```

```console
gdb) s
8804	            ctx->logits.reserve(hparams.n_vocab);
(gdb) p hparams.n_vocab
$40 = 32000
```
That is pretty much it for the intialization of the llama_context. This will
return us to simple_prompt.cpp:
```console
4	    llama_context * ctx = llama_new_context_with_model(model, ctx_params);
35	    if (ctx == NULL) {
36	        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
37	        return 1;
38	    }
```
Now, lets look what happens when decode is called and how this interacts with
the key-value cache.
```console
(gdb) br simple-prompt.cpp:115
Breakpoint 2 at 0x40949e: file src/simple-prompt.cpp, line 115.
(gdb) c
Continuing.
batch.n_tokens: 6
batch.tokens: [1, 1724, 338, 4309, 4717, 29973, ]
prompt: What is LoRA?
Breakpoint 2, main (argc=1, argv=0x7fffffffd198) at src/simple-prompt.cpp:115
115	    if (llama_decode(ctx, batch) != 0) {
```
And the batch looks like this:
```console
(gdb) p batch
$54 = {n_tokens = 6, token = 0xc63980, embd = 0x0, pos = 0x8a86f0, n_seq_id = 0x8ab7b0, seq_id = 0x8abfc0, 
  logits = 0x8a9790 "", all_pos_0 = 0, all_pos_1 = 0, all_seq_id = 0}
(gdb) s
5678	    const auto n_batch = cparams.n_batch;
(gdb) s
(gdb) p n_batch
$56 = 512
```
So, `n_batch` is the maximum number of tokens that can be in a single batch, and
`n_tokens` is the number of tokens in the current batch.
```console
(gdb) n
5682	    int n_threads = n_tokens == 1 ? cparams.n_threads : cparams.n_threads_batch;
```
I found this a little interesting and because I've always called decode with
a number of tokens, never a single token. But thinking back the example with
the key-value cache and how it would pass in a single token as the input but
the key and value matrices would contain all the previous tokens.
```console
(gdb) n
5695	    auto & kv_self = lctx.kv_self;

5734	    // if we have enough unused cells before the current head ->
5735	    //   better to start searching from the beginning of the cache, hoping to fill it
5736	    if (kv_self.head > kv_self.used + 2*n_tokens) {
5737	        kv_self.head = 0;
5738	    }
5739	
5740	    if (!llama_kv_cache_find_slot(kv_self, batch)) {
```
```console
1584	// find an empty slot of size "n_tokens" in the cache
1585	// updates the cache head
1586	// Note: On success, it's important that cache.head points
1587	// to the first cell of the slot.
1588	static bool llama_kv_cache_find_slot(
1589	           struct llama_kv_cache & cache,
1590	        const struct llama_batch & batch) {
1591	    const uint32_t n_ctx    = cache.size;
1592	    const uint32_t n_tokens = batch.n_tokens;
1593	
1594	    if (n_tokens > n_ctx) {
1595	        LLAMA_LOG_ERROR("%s: n_tokens=%d > n_ctx=%d\n", __func__, n_tokens, n_ctx);
1596	        return false;
1597	    }
```
In this case we have the max context size in token of 1024 and the number of
tokens in the batch is 6:
```console
(gdb) p n_ctx
$69 = 1024
(gdb) p n_tokens
$70 = 6
```
The number of tokens in the batch cannot exceed the max context size.
```console
(gdb) l
1599	    uint32_t n_tested = 0;
1600	
1601	    while (true) {
1602	        if (cache.head + n_tokens > n_ctx) {
1603	            n_tested += n_ctx - cache.head;
1604	            cache.head = 0;
1605	            continue;
1606	        }
```
So we are going to loop and the first thing we do is check if the head plus the
number of tokens in the batch exceed the max number of tokens allowed.  If this
is the case then `n_tested` is incremented with the max context size minus the
cache head.

Lets pretend that we have a head that is 1020 and the number of tokens is 6 and
n_ctx is 1024. Then 1020+6=1026 and 1026 > 1024. And n_tested will become
1024-1020=4. And the head will be set to 0. And then the loop will continue but
this time head will be zero. And the if statement will compare 6 > 1024 which
is false and skip the body of the if statement.
```console
1608	        bool found = true;
1609	        for (uint32_t i = 0; i < n_tokens; i++) {
1610	            if (cache.cells[cache.head + i].pos >= 0) {
1611	                found = false;
1612	                cache.head += i + 1;
(gdb) l
1613	                n_tested   += i + 1;
1614	                break;
1615	            }
1616	        }
```
So we are going to loop over all the 6 tokens in the batch.
```console
(gdb) p i
$83 = 0
(gdb) p cache.head
$84 = 0

(gdb) p cache.cells[cache.head + i].pos
$85 = -1
```
cache.cells is a vector of size 1024, the max number of tokens allowed.
```console
(gdb) p lctx.kv_self.cells.size()
$24 = 1024
```

If an entry in this vector is currently not set, its position is -1. So in our
case the cells checked will not be greater than zero and the if block will not
be executed. So this is making sure that from the current head there are
n_tokens number of slots available.

That will lead us to the following code:
```console
1628	    for (uint32_t i = 0; i < n_tokens; i++) {
1629	        cache.cells[cache.head + i].pos = batch.pos[i];
1630	
1631	        for (int32_t j = 0; j < batch.n_seq_id[i]; j++) {
1632	            cache.cells[cache.head + i].seq_id.insert(batch.seq_id[i][j]);
1633	        }
1634	    }
1635	
1636	    cache.used += n_tokens;
1637	
1638	    return true;
```
And it makes sence that we again will loop over the 6 tokens in the batch and
now add them to the cells. 
```console
gdb) p cache.cells
$106 = std::vector of length 1024, capacity 1024 = {
{pos = 0, delta = 0, seq_id = std::set with 1 element = {[0] = 0}},
{pos = 1, delta = 0, seq_id = std::set with 1 element = {[0] = 0}},
{pos = 2, delta = 0, seq_id = std::set with 1 element = {[0] = 0}},
{pos = 3, delta = 0, seq_id = std::set with 1 element = {[0] = 0}},
{pos = 4, delta = 0, seq_id = std::set with 1 element = {[0] = 0}},
{pos = 5, delta = 0, seq_id = std::set with 1 element = {[0] = 0}},
```
That is it for finding a slot in the key-value cache.
```console
(gdb) s
5747	    kv_self.n = std::min(
                (int32_t) cparams.n_ctx,
                std::max(32, GGML_PAD(llama_kv_cache_cell_max(kv_self), 32))
            );
```
A cell is considered in use if its position is greater than or equal to zero and
it's sequence id is not empty. So that should return 6 in our case:
```console
(gdb) p llama_kv_cache_cell_max(kv_self)
$108 = 6
(gdb) p cparams.n_ctx
$109 = 1024
(gdb) p kv_self.n
$110 = 32
```
What is kv_self.n?  


