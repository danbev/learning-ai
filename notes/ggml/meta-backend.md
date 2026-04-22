## Meta backend
TODO: add intro from Linux NUC


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

__wip__
