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
```
So we can see that a new ggml_backend_meta_buffer_context is created for this
meta backend. This type looks like this following
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
So an instance of this type has a vector of buffer_config s and an debug
member. The other types are static. The

Next, we are going to iterate over all the tensors in the context and set its
buffer to the meta backend buffer type, and the data pointer to a dummy/place
hold value: 
```c++
    // This is just creating a ggml_backend_buffer_t struct.
    ggml_backend_buffer_t meta_buf = ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, meta_buf_ctx, 0);

    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        t->buffer = meta_buf;
    }
```
Remember in a meta backend the tensor actual data will be distributed amoung
potentially many actual real backends. But there are a number of places in the
code where tensor->data is checked for NULL so we need some value here.
And we also set the buffer type to be the meta-data backend buffer type which
contains the following functions:
```c++
static const ggml_backend_buffer_i ggml_backend_meta_buffer_iface = {
    /* .free_buffer     = */ ggml_backend_meta_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_meta_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_meta_buffer_init_tensor,
    /* .memset_tensor   = */ nullptr, // TODO implement
    /* .set_tensor      = */ ggml_backend_meta_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_meta_buffer_get_tensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_meta_buffer_clear,
    /* .reset           = */ ggml_backend_meta_buffer_reset,
};

```
So when for example `set_tensor` this is would be where the splitting (if any
would be done I think. I'll look into this when we get to that point in the
debugging session).
Then we have a call to initialize the meta backend tensor:
```c++
        ggml_backend_meta_buffer_init_tensor(meta_buf, t);
```

```c++
static enum ggml_status ggml_backend_meta_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    const size_t n_simple_bufs = ggml_backend_meta_buffer_n_bufs(buffer);

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ true);
    GGML_ASSERT(ggml_nelements(tensor) == 0 || split_state.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
    GGML_ASSERT(split_state.n_segments <= 16);
```

```c++
static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(const struct ggml_tensor * tensor, bool assume_sync) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
```
This function has a number of lambdas that handle different types of operations.
```c++
    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ true);
```
The above call will call the 
```c++
static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(const struct ggml_tensor * tensor, bool assume_sync) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    ...

    const std::pair key = std::make_pair(tensor, assume_sync);
    auto it = buf_ctx->split_state_cache.find(key);
    if (it != buf_ctx->split_state_cache.end() &&
        memcmp(it->second.second, (const char *) tensor, sizeof(it->second.second)) != 0) {
        buf_ctx->split_state_cache.clear();
        it = buf_ctx->split_state_cache.end();
    }
```
Here we are creating a key (hashmap key) for this tensor. And then using it to
see if it is in the split_state_cache. If we look back at the static split_state_cache
we had:
```c++
std::map<std::pair<const ggml_tensor *, bool>,
         std::pair<ggml_backend_meta_split_state, char[nbtc]>> split_state_cache;
```
So both the key and the value are std::pairs. The key pair is the tensor pointer
and a bool that indicates async flag. If async is true the backend knows it needs
to perform a All-Reduce operation to synchronize the partial sum. So we can
ask the cache for the split state for a tensor given a specific sync assumption.

The value contains the actual split state as its first member, and then has a
char array as it's second member. This is used as a snapshot of the ggml_tensor
struct and is used to check if the tensor has changed (cache invalidation). This
is what is done above if the iterator is not equal to the end, that is there is
state for this key, the memcmp (compare) will check the second member against
the current tensor to check this. If they are not the same it clears the entire
cache (simplest/safest solution to handle dependencies), and sets it to to then
end so that it will be added again.

Next, we populate the cache if the current tensor does not have a split state
in the cache:
```c++
    if (it == buf_ctx->split_state_cache.end()) {
        buf_ctx->split_state_cache[key].first = calculate_split_state();
        memcpy(buf_ctx->split_state_cache[key].second, tensor, sizeof(buf_ctx->split_state_cache[key].second));
```
Where calculate_split_state() is a lambda in ggml_backend_meta_get_split_state:
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
    (void)user_data;
    ggml_backend_meta_split_state state;

    printf("get_split_state for tensor '%s' (op: %s)\n", tensor->name, ggml_op_name(tensor->op));

    state.axis = GGML_BACKEND_SPLIT_AXIS_0;
    state.n_segments = 1;

    state.ne[0] = tensor->ne[0] / 2;                   // First half for CPU
    state.ne[1] = tensor->ne[0] - (tensor->ne[0] / 2); // Second half for Metal

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
Now, similar to a ggml tensor which has a number of element array which specifies
the number of elements per dimention, the split state also has a ne[] array:
```
    struct ggml_backend_meta_split_state {
        enum ggml_backend_meta_split_axis axis;

        int64_t  ne[16*GGML_BACKEND_META_MAX_DEVICES];
        uint32_t n_segments;
    };
```
In this case split->ne[0], the index is the index of the backend which we
specified when creating this meta backend:
```c++
    ggml_backend_dev_t devs[] = {cpu_dev, mtl_dev};
    ggml_backend_dev_t meta_dev = ggml_backend_meta_device(devs, 2, get_split_state, NULL);
```
And the value here is the number of number of elements from the current tensor
that will be that will be allocated to devs[0].

And just to orient ourselves a bit here is the backtrace from the debugger:
```console
(gdb) bt
#0  operator() (__closure=0x7fffffffb050) at /home/danbev/work/ai/learning-ai/fundamentals/ggml/ggml/src/ggml-backend-meta.cpp:764
#1  0x00005555555a2382 in ggml_backend_meta_get_split_state (tensor=0x7fffcbbff060, assume_sync=true)
    at /home/danbev/work/ai/learning-ai/fundamentals/ggml/ggml/src/ggml-backend-meta.cpp:1032
#2  0x00005555555a2cbc in ggml_backend_meta_buffer_init_tensor (buffer=0x55555ce6cbe0, tensor=0x7fffcbbff060)
    at /home/danbev/work/ai/learning-ai/fundamentals/ggml/ggml/src/ggml-backend-meta.cpp:1090
#3  0x00005555555a54a4 in ggml_backend_meta_alloc_ctx_tensors_from_buft (ctx=0x55555ce1eda0, buft=0x55555ce40658)
    at /home/danbev/work/ai/learning-ai/fundamentals/ggml/ggml/src/ggml-backend-meta.cpp:1437
#4  0x000055555559095d in ggml_backend_alloc_ctx_tensors_from_buft (ctx=0x55555ce1eda0, buft=0x55555ce40658)
    at /home/danbev/work/ai/learning-ai/fundamentals/ggml/ggml/src/ggml-alloc.c:1241
#5  0x00005555555909c0 in ggml_backend_alloc_ctx_tensors (ctx=0x55555ce1eda0, backend=0x55555ce53590)
    at /home/danbev/work/ai/learning-ai/fundamentals/ggml/ggml/src/ggml-alloc.c:1247
#6  0x0000555555568126 in main (argc=1, argv=0x7fffffffd6a8) at src/meta-backend.cpp:81
```
So for the source tensors and the operation the following block is called:
```c++
    auto calculate_split_state = [&]() -> ggml_backend_meta_split_state {
        if (ggml_nelements(tensor) == 0) {
            return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, 1};
        }
        if (ggml_backend_buffer_get_usage(tensor->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE && tensor->view_src == nullptr) {
            ...

            return ret;
        }
```
And this is the block that calls our callback function. This returnes early but
for a view this rest of the lambda is executed:.
```c++
        std::vector<ggml_backend_meta_split_state> src_ss(GGML_MAX_SRC, {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, 1});
        for (size_t i = 0; i < GGML_MAX_SRC; i++) {
            if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                src_ss[i] = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, 1};
                continue;
            }
            src_ss[i] = ggml_backend_meta_get_split_state(tensor->src[i], /*assume_sync =*/ true);
            GGML_ASSERT(src_ss[i].axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        }
```
So this is creating a new std::vector which will hold source split states with a
size of 10 which is the max number of source tensors, and all 10 entries will be
initialied with  axis of GGML_BACKEND_SPLIT_AXIS_NONE, ne of {0}, and
n_segments 1.
Then we iterate over the max nr of sources (10), and if the source tensor is nullptr
of the is the same as the current tensor this sets the axis to unknown.

And recall that a view doesn't own its data, it just has an shape and the data
is part of another tensor but we will have to have a split for a view. So we
will recursively call ggml_backend_meta_get_split_state with the views source
tensor and assign that to entry in src_ss.

After all sources have been processed we have the following switch statement:
```c++
        ggml_backend_meta_split_state split_state;
        switch (tensor->op) {
            ...
            case GGML_OP_VIEW: {
                split_state = handle_view(src_ss);
            } break;
            ...
```
And handle_view is a lambda defined ealier in this function:
```console
    auto handle_view = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (ggml_is_contiguous(tensor) && ggml_is_contiguous(tensor->src[0])) {
            return handle_reshape(src_ss);
        }
```
And handle_view looks like this:
```c++
    auto handle_reshape = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            ...
```
So, src_ss is a vector of 10 elements which is the max number of source tensors
that an operation tensor can have. And just to recap, the current tensor is a
result reshape tensor (from our view tensor that is). It was a source which is
the "real" tensors and what we are doing it trying to determine if/how the
current tensor needs to be split and how
In this case we are simply using the same split state as same split state as
the source (in the case of mirror and partial that is).

```c++
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1:
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3: {

                // if the the source was split along its last dimension
                if (src_ss[0].axis == ggml_n_dims(tensor->src[0]) - 1) {
                    return {ggml_backend_meta_split_axis(ggml_n_dims(tensor) - 1), {0}, 1};
                }
```
We know that it was one of the above cases , either split 0, 1, 2, or 3 dimension.


And after the lambda returnes we have:
```c++
        if (split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS) {
            bool first_src_split_by_axis = true;
```
Just a not about the split_stat.axis and how it is used in the code base. It
intentionally gives the first 4 values of the enum the 0-3, and the rest are
assigned higher values which enables us to use the enum in statements like the
above. 
```console
(gdb) p split_state.axis
$23 = GGML_BACKEND_SPLIT_AXIS_MIRRORED
(gdb) p (int)split_state.axis
$24 = 10
```

The returned state split will be set in the cache:
```c++
    if (it == buf_ctx->split_state_cache.end()) {
        buf_ctx->split_state_cache[key].first = calculate_split_state();
        memcpy(buf_ctx->split_state_cache[key].second, tensor, sizeof(buf_ctx->split_state_cache[key].second));
```
And then the rest of the function is mainly for debugging, and the
ggml_backend_meta_split_state is returned.
```console
(gdb) 
ggml_backend_meta_buffer_init_tensor (buffer=0x55555ce6cbe0, tensor=0x7fffcbbff4b0)
    at /home/danbev/work/ai/learning-ai/fundamentals/ggml/ggml/src/ggml-backend-meta.cpp:1091
1091	    GGML_ASSERT(ggml_nelements(tensor) == 0 || split_state.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
```
```c++
static enum ggml_status ggml_backend_meta_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    const size_t n_simple_bufs = ggml_backend_meta_buffer_n_bufs(buffer);

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ true);
    GGML_ASSERT(ggml_nelements(tensor) == 0 || split_state.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
    GGML_ASSERT(split_state.n_segments <= 16);

    int split_dim = split_state.axis;
    int64_t ne[GGML_MAX_DIMS];
    size_t  nb[GGML_MAX_DIMS];
    for (size_t k = 0; k < GGML_MAX_DIMS; k++) {
        ne[k] = tensor->ne[k];
        nb[k] = tensor->nb[k];
    }
```
The above loop is just making a local copy of the number of elements and number
of bytes of the operation tensor so that the original tensor values are not modified
later.

Next, a vector of ggml_tensor pointers are created named simple_tensors which
are the kind of tensors that we are used to see normally. They have their own
ne, nb, type, data pointer, and a buffer pointing to memory managed by a specific
real backend. But a simple tensor might not contains all of the tensors data
which is a difference.
```c++
    std::vector<ggml_tensor *> simple_tensors;
    simple_tensors.reserve(n_simple_bufs);

    for (size_t j = 0; j < n_simple_bufs; j++) {
        ggml_context          * simple_ctx = buf_ctx->buf_configs[j].ctx;
        ggml_backend_buffer_t   simple_buf = buf_ctx->buf_configs[j].buf;
        ...
        ggml_tensor * t_ij = ggml_new_tensor(simple_ctx, tensor->type, GGML_MAX_DIMS, ne);
```
So here we can see the actual tensor is created.
And then we copy the rest of the tensor properties:
```c++
        t_ij->op = tensor->op;
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            t_ij->nb[i] = nb[i];
        }
        t_ij->flags = tensor->flags;
        memcpy(t_ij->op_params, tensor->op_params, sizeof(tensor->op_params));
        ggml_set_name(t_ij, tensor->name);
        t_ij->buffer = simple_buf;
        t_ij->view_src = tensor->view_src;
        t_ij->view_offs = tensor->view_offs;
```
And since our tensor is a view we will entry this block and set the tensors
view_src:
```c++
        if (t_ij->view_src != nullptr && ggml_backend_buffer_is_meta(t_ij->view_src->buffer)) {
            t_ij->view_src = ggml_backend_meta_buffer_simple_tensor(tensor->view_src, j);
```
Lets take a closer look at ggml_backend_meta_buffer_simple_tensor:
```c++
static struct ggml_tensor * ggml_backend_meta_buffer_simple_tensor(const struct ggml_tensor * tensor, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    GGML_ASSERT(index < buf_ctx->buf_configs.size());

    auto it = buf_ctx->simple_tensors.find(tensor);
    if (it == buf_ctx->simple_tensors.end()) {
        return nullptr;
    }
    return it->second[index];
}
```
We saw the simple_tensors field ealier but we did to actually look at what it
was in detail.
```c++
struct ggml_backend_meta_buffer_context {
    static constexpr size_t nbtc = GGML_TENSOR_SIZE - sizeof(ggml_tensor::padding);

    std::map<std::pair<const ggml_tensor *, bool>, std::pair<ggml_backend_meta_split_state, char[nbtc]>> split_state_cache;
    std::map<          const ggml_tensor *,        std::vector<ggml_tensor *>>                           simple_tensors;
```
So a backend buffer context has a map of tensor pointer to a vector of tensor
pointers. So the function above uses the tensor as the key and then the index
to get the tensor of interest.
So before the following call the current simple_tensor as a view_src that points
to the meta source tensor. But this will look up the simple tensor and set it:
```c++
            t_ij->view_src = ggml_backend_meta_buffer_simple_tensor(tensor->view_src, j);
```
And if we have a view_src then we update the data pointer to point to that
source_views data. And if not then it uses the the tensors data (not a view):
```c++
        if (t_ij->view_src != nullptr) {
            t_ij->data = (char *) t_ij->view_src->data + t_ij->view_offs;
        } else if (simple_buf != nullptr) {
            t_ij->data = (char *) ggml_backend_buffer_get_base(simple_buf)
                + size_t(tensor->data) - size_t(ggml_backend_buffer_get_base(buffer));
        }
```

```c++
    for (size_t i = 0; i < n_simple_bufts; i++) {
        meta_buf_ctx->buf_configs[i].buf = ggml_backend_alloc_ctx_tensors_from_buft(
            meta_buf_ctx->buf_configs[i].ctx, ggml_backend_meta_buft_simple_buft(buft, i));
        meta_buf->size = std::max(meta_buf->size, ggml_backend_buffer_get_size(meta_buf_ctx->buf_configs[i].buf));
    }
```
But this time we are not passing in a backend buffer context but the context
for the simple tensors. So this is where the actual physical memory on each
simple backend will be created.

So that is pretty much the last thing that happends for
ggml_backend_alloc_ctx_tensors.

The next interaction happens when we call:
```c++
    ggml_backend_tensor_set(a, a_data, 0, sizeof(a_data));
```
```c++
void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor(buf, tensor, data, offset, size);
}
```
```console
(gdb) p *buf
$48 = {iface = {free_buffer = 0x55555559e0c7 <ggml_backend_meta_buffer_free_buffer(ggml_backend_buffer_t)>, 
    get_base = 0x5555555a2abe <ggml_backend_meta_buffer_get_base(ggml_backend_buffer_t)>, 
    init_tensor = 0x5555555a2ad6 <ggml_backend_meta_buffer_init_tensor(ggml_backend_buffer_t, ggml_tensor*)>, memset_tensor = 0x0, 
    set_tensor = 0x5555555a3704 <ggml_backend_meta_buffer_set_tensor(ggml_backend_buffer_t, ggml_tensor*, void const*, size_t, size_t)>, 
    get_tensor = 0x5555555a4413 <ggml_backend_meta_buffer_get_tensor(ggml_backend_buffer_t, ggml_tensor const*, void*, size_t, size_t)>, set_tensor_2d = 0x0, get_tensor_2d = 0x0, cpy_tensor = 0x0, 
    clear = 0x5555555a4f02 <ggml_backend_meta_buffer_clear(ggml_backend_buffer_t, uint8_t)>, 
    reset = 0x5555555a4f6a <ggml_backend_meta_buffer_reset(ggml_backend_buffer_t)>}, buft = 0x55555ce40658, 
  context = 0x55555ce2fa80, size = 384, usage = GGML_BACKEND_BUFFER_USAGE_ANY}

(gdb) p buf.iface
$49 = {free_buffer = 0x55555559e0c7 <ggml_backend_meta_buffer_free_buffer(ggml_backend_buffer_t)>, 
  get_base = 0x5555555a2abe <ggml_backend_meta_buffer_get_base(ggml_backend_buffer_t)>, 
  init_tensor = 0x5555555a2ad6 <ggml_backend_meta_buffer_init_tensor(ggml_backend_buffer_t, ggml_tensor*)>, memset_tensor = 0x0, 
  set_tensor = 0x5555555a3704 <ggml_backend_meta_buffer_set_tensor(ggml_backend_buffer_t, ggml_tensor*, void const*, size_t, size_t)>, 
  get_tensor = 0x5555555a4413 <ggml_backend_meta_buffer_get_tensor(ggml_backend_buffer_t, ggml_tensor const*, void*, size_t, size_t)>, 
  set_tensor_2d = 0x0, get_tensor_2d = 0x0, cpy_tensor = 0x0, 
  clear = 0x5555555a4f02 <ggml_backend_meta_buffer_clear(ggml_backend_buffer_t, uint8_t)>, 
  reset = 0x5555555a4f6a <ggml_backend_meta_buffer_reset(ggml_backend_buffer_t)>}
```
So we can see that when we call `buf->iface.set_tensor` this will be calling
into the meta bachend buffer.
```c++
static void ggml_backend_meta_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
```
This is again calling ggml_backend_meta_get_split_state, but this time with
assume_sync set to false, so the data transfer my be async.

```c++
    switch (split_state.axis) {
        ...

        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ggml_backend_tensor_set(simple_tensor, data, offset, size);
            }
        } break;
```
And in our case this will iterate over our two backend meta buffers, get the
simple tensor for the current tensor, and the index is the backend in question.
And then we perform a normal ggml_backend_tensor_set call, well this was also
that but now the simple_tensor is used which has a different backend buffer
so this will get dispatched to the actual backend.

__wip__

