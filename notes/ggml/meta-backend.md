## Meta backend
This a virtual device that aggregates multiple physical devices, like multiple
GPUs, or CPU plus GPU combos into a single logical device.

And recall that ggml_backend_dev_t, a device represents a hardware device. And
a ggml_backend_t (backend) is an active instance or context created from a device
that executes operations.

So a virtual device combines multiple physical devices and can be used to split
a model across multiple GPUs, for tensor parallelism, etc.

### Example
Build the example:
```console
$ cd fundamentals/ggml
$ make ggml-init-cuda
$ make meta-backend
```

### Walkthrough
Run in gdb:
```console
$ lldb bin/meta-backend
(lldb) br set -f meta-backend.cpp -l 44
Breakpoint 1: where = meta-backend`main + 180 at meta-backend.cpp:44:34, address = 0x0000000100000c4c
(lldb) r
Process 52088 stopped
* thread #1, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
    frame #0: 0x0000000100000c4c meta-backend`main(argc=1, argv=0x000000016fdfee50) at meta-backend.cpp:44:34
   41  	    }
   42
   43  	    //ggml_backend_dev_t devs[] = {cpu_dev, cuda_dev};
-> 44  	    ggml_backend_dev_t devs[] = {cpu_dev};
   45  	    ggml_backend_dev_t meta_dev = ggml_backend_meta_device(devs, 1, get_split_state, NULL);
   46  	    printf("Meta device created: %s\n", ggml_backend_dev_name(meta_dev));
   47
```
In ggml-meta-backend.cpp we find
```c++
ggml_backend_dev_t ggml_backend_meta_device(ggml_backend_dev_t * devs,
    size_t n_devs,
    ggml_backend_meta_get_split_state_t get_split_state,
    void * get_split_state_ud) {
```
So this function takes one or more devices, and a callback and user data for
the callback.
Following that we have the creation of a vector of unique pointers to
ggml_backend_meta_device_context:
```c++
    static std::vector<std::unique_ptr<ggml_backend_meta_device_context>> ctxs;
```
Note that this is static.
So this is a type that is new and it holds the devices and the callback plus
callback user data:
```c++
struct ggml_backend_meta_device_context {
    std::vector<ggml_backend_dev_t>     simple_devs;
    ggml_backend_meta_get_split_state_t get_split_state;
    void *                              get_split_state_ud;

    std::string name;
    std::string description;
    ...
```
After that we have:
```c++
    static std::map<ggml_backend_meta_device_context, struct ggml_backend_device> meta_devs;
```
Again note that this is also static. This acts like a cache which we will see
futher down.

Then we will iterate over all the passed in devices and add them to the simple_devs
vector:
```c++
    std::vector<ggml_backend_dev_t> simple_devs;
    simple_devs.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        simple_devs.push_back(devs[i]);
    }
    ggml_backend_meta_device_context ctx(simple_devs, get_split_state, get_split_state_ud);
```
Next, we check the cache to see if a context a context like the one passed in
has already been created device for it:
```c++
    {
        auto it = meta_devs.find(ctx);
        if (it != meta_devs.end()) {
            return &it->second;
        }
    }
```
Next the local context that was created previously, which is a local variable
so we are creating a copy of this on the heap:
```c++
    ctxs.push_back(std::make_unique<ggml_backend_meta_device_context>(ctx));
```
And recall that ctxs is static so it has a lifetime of the entire program.
And after that we have the last part of the function and this is where we create
a ggml_backend_device:
```c++
    struct ggml_backend_device meta_dev = {
        /*iface  =*/ ggml_backend_meta_device_iface,
        /*reg    =*/ nullptr,
        /*ctx    =*/ ctxs.back().get(),
    };

    auto result = meta_devs.emplace(*ctxs.back(), meta_dev);
    return &result.first->second;
```
```c++
static const ggml_backend_device_i ggml_backend_meta_device_iface = {
    /* .get_name             = */ ggml_backend_meta_device_get_name,
    /* .get_description      = */ ggml_backend_meta_device_get_description,
    /* .get_memory           = */ ggml_backend_meta_device_get_memory,
    /* .get_type             = */ ggml_backend_meta_device_get_type,
    /* .get_props            = */ ggml_backend_meta_device_get_props,
    /* .init_backend         = */ ggml_backend_meta_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_meta_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_meta_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_meta_device_supports_op,
    /* .supports_buft        = */ ggml_backend_meta_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};
```
So that was the end of ggml_backend_meta_device and we are back in main:
```c++
    ggml_backend_t meta_backend = ggml_backend_dev_init(meta_dev, NULL);
```
This will delegate to the device interface above so the init_backend function
registered will be called:
```c++
static ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_meta_context * backend_ctx = new ggml_backend_meta_context(dev, params);

    ggml_backend_t backend = new struct ggml_backend;
    backend->guid    = ggml_backend_meta_guid();
    backend->iface   = ggml_backend_meta_i;
    backend->device  = dev;
    backend->context = backend_ctx;
    return backend;
}
```
Don't confuse the ggml_backend_meta_context with ggml_backend_meta_device_context
this is different and not something we've looked at before:
```c++
struct ggml_backend_meta_context {
    struct cgraph_config {
        ggml_cgraph * cgraph_main = nullptr;
        int           offset      = 0; // Node offset vs. original graph

        std::vector<ggml_cgraph *> cgraphs_aux;
    };
    struct backend_config {
        ggml_backend_t backend;

        std::vector<cgraph_config>           cgraphs;
        std::vector<ggml_tensor *>           nodes;
        std::vector<ggml_backend_buffer_ptr> bufs;

        backend_config(ggml_backend_t backend, const size_t n_reduce_steps) : backend(backend) {
            bufs.resize(n_reduce_steps);
        }
    };
    std::string                 name;
    std::vector<backend_config> backend_configs;
    ggml_context_ptr            ctx;
    std::vector<ggml_cgraph *>  cgraphs_aux;
    std::vector<ggml_tensor *>  nodes_aux;
    size_t                      n_reduce_steps;
    int                         max_nnodes    = 0;
    size_t                      max_tmp_size  = 0;
    size_t                      max_subgraphs = 0;
    size_t                      n_subgraphs   = 0;
    uint64_t                    uid           = 0;

    void *                               comm_ctx       = nullptr;
    ggml_backend_comm_allreduce_tensor_t comm_allreduce = nullptr;
```
The All-Reduce is about how results are communicated between the backends of a
meta backend. For example, say we have 4 GPUs and we parts of an operations on
each of them then we might have something like this:
```console
Local merge part:
GPU0 sends its output to CPU1
GPU2 sends its output to GPU3

GPU1 performs an addition, so GPU1 will have the result of GPU0 and GPU1.
GPU3 performs an addition, so GPU3 will have the result of GPU2 and GPU3.

Final merge part:
GPU1 sends its sum to GPU3
GPU3 adds them together, so GPU3 now has the total result. 

All part (distribution):
GPU3 sends the total back to GPU1. So both now have the total.
GPU1 sends the total back to GPU0.
GPU3 sends the total back to GPU2.
```
And after that a new ggml_backend is created and returned.
Back in main we create two tensors and a addtion operation and add them to the
graph. And after that we have:
```c++
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, meta_backend);
```
In ggml/src/ggml-alloc.c we can find the following:
```c++
ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors_from_buft(struct ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    size_t nbytes_total = 0;
    if (ggml_backend_buft_is_meta(buft)) {
        return ggml_backend_meta_alloc_ctx_tensors_from_buft(ctx, buft);
    }
    return ggml_backend_alloc_ctx_tensors_from_buft_impl(ctx, buft, &nbytes_total, /*no_alloc =*/ false);
}
```
So there is a special case for meta backends. Let's look at that function:
```c++
struct ggml_backend_buffer * ggml_backend_meta_alloc_ctx_tensors_from_buft(struct ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024*1024, // FIXME
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ggml_backend_meta_buffer_context * meta_buf_ctx = new ggml_backend_meta_buffer_context();
    meta_buf_ctx->buf_configs.reserve(n_simple_bufts);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        meta_buf_ctx->buf_configs.emplace_back(ggml_init(params), nullptr);
    }

    ggml_backend_buffer_t meta_buf = ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, meta_buf_ctx, 0);
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        t->buffer = meta_buf;
        ggml_backend_meta_buffer_init_tensor(meta_buf, t);
        t->data = (void *) 0x2000000000000000; // FIXME
    }
    for (size_t i = 0; i < n_simple_bufts; i++) {
        meta_buf_ctx->buf_configs[i].buf = ggml_backend_alloc_ctx_tensors_from_buft(
            meta_buf_ctx->buf_configs[i].ctx, ggml_backend_meta_buft_simple_buft(buft, i));
        meta_buf->size = std::max(meta_buf->size, ggml_backend_buffer_get_size(meta_buf_ctx->buf_configs[i].buf));
    }
    return meta_buf;
}
```
```c++
struct ggml_backend_meta_buffer_context {
    static constexpr size_t nbtc = GGML_TENSOR_SIZE - sizeof(ggml_tensor::padding);

    std::map<std::pair<const ggml_tensor *, bool>, std::pair<ggml_backend_meta_split_state, char[nbtc]>> split_state_cache;
    std::map<          const ggml_tensor *,        std::vector<ggml_tensor *>>                           simple_tensors;

    struct buffer_config {
        ggml_context          * ctx;
        ggml_backend_buffer_t   buf;

        buffer_config(ggml_context * ctx, ggml_backend_buffer_t buf) : ctx(ctx), buf(buf) {}
    };
    std::vector<buffer_config> buf_configs;

    int debug;

    ggml_backend_meta_buffer_context() {
        const char * GGML_META_DEBUG = getenv("GGML_META_DEBUG");
        debug = GGML_META_DEBUG ? atoi(GGML_META_DEBUG) : 0;
    }
};
```
We can see that this holds a vector of ggml_context's, one for each backend which
are created in the previous function.

Next we have the following which just created a ggml_backend_buffer_t for the
meta backend:
```c++
    ggml_backend_buffer_t meta_buf = ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, meta_buf_ctx, 0);
```
Next we will iterate over all the tensors in the context and initalize them,
setting their buffer to the meta buffer:
```c++
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        t->buffer = meta_buf;
        ggml_backend_meta_buffer_init_tensor(meta_buf, t);
        t->data = (void *) 0x2000000000000000; // FIXME
    }
```

```c++
static enum ggml_status ggml_backend_meta_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    const size_t n_simple_bufs = ggml_backend_meta_buffer_n_bufs(buffer);

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ true);
    GGML_ASSERT(ggml_nelements(tensor) == 0 || split_state.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
    GGML_ASSERT(split_state.n_segments <= 16);

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ true);
```
The above call will call the 
```c++
static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(const struct ggml_tensor * tensor, bool assume_sync) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    ...
    if (it == buf_ctx->split_state_cache.end()) {
        buf_ctx->split_state_cache[key].first = calculate_split_state();
```
Where calculate_split_state() is a lambda:
```c++
    auto calculate_split_state = [&]() -> ggml_backend_meta_split_state {
        if (ggml_nelements(tensor) == 0) {
            return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, 1};
        }

        if (ggml_backend_buffer_get_usage(tensor->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE && tensor->view_src == nullptr) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
            const ggml_backend_meta_device_context * dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
            ggml_backend_meta_split_state ret = dev_ctx->get_split_state(tensor, dev_ctx->get_split_state_ud);
```
And this is calling our callback function:
```c++
ggml_backend_meta_split_state get_split_state(const struct ggml_tensor * tensor, void * user_data) {
    ggml_backend_meta_split_state state;

    // Replicate the tensor on all devices.
    state.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
    state.n_segments = 1;
    return state;
}
```
So for each tensor this function will be called to determine how to split the
tensor.
```c++
    enum ggml_backend_meta_split_axis {
        // tensor split by tensor dimensions:
        GGML_BACKEND_SPLIT_AXIS_0 = 0,
        GGML_BACKEND_SPLIT_AXIS_1 = 1,
        GGML_BACKEND_SPLIT_AXIS_2 = 2,
        GGML_BACKEND_SPLIT_AXIS_3 = 3,

        GGML_BACKEND_SPLIT_AXIS_MIRRORED = 10, // all values on all backends
        GGML_BACKEND_SPLIT_AXIS_PARTIAL  = 11, // each backend has a partial sum

        // for internal bookkeeping only:
        GGML_BACKEND_SPLIT_AXIS_NONE    = 98,
        GGML_BACKEND_SPLIT_AXIS_UNKNOWN = 99,
    };
```

```console
(gdb) p tensor->name
$7 = "a", '\000' <repeats 62 times>
```

```c++
    if (it == buf_ctx->split_state_cache.end()) {
        buf_ctx->split_state_cache[key].first = calculate_split_state();
        memcpy(buf_ctx->split_state_cache[key].second, tensor, sizeof(buf_ctx->split_state_cache[key].second));
```
```console
(gdb) p key
$16 = {first = 0x7fffcbbff060, second = true}

(gdb) ptype buf_ctx->split_state_cache[key].second
type = char [328]
```
So the above is copying the tensor data into the cache, but only the first 328
bytes of the tensor. This is the size of the ggml_tensor struct.
This is storing a snapshot of the tensor in the cache.



__wip__
