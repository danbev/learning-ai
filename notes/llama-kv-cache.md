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
for key layer and one for the value layers. So for each layer there is a tensor
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
So the number/size of grouped query attention embeddings for the keys matrix will
be
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
`n_ctx` unless the model supports MAMBA.
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
for each buffer type, and in this case there is only one element, which has and
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

