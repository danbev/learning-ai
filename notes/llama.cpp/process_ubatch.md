### process_ubatch
This is what actually processes a micro batch (ubatch) within a decode call. In
short the batch that a user provides is split into micro batches that fit into
the model context and then it is passed to llama_contex::process_ubatch:
```c++
llm_graph_result * llama_context::process_ubatch(const llama_ubatch & ubatch, llm_graph_type gtype, llama_memory_context_i * mctx, ggml_status & ret) {
    ...
    auto * res = gf_res_prev.get();
    auto * gf  = res->get_gf();

    const auto gparams = graph_params(res, ubatch, mctx, gtype);
    if (!graph_reuse_disable && res->can_reuse(gparams)) {
        //LLAMA_LOG_DEBUG("%s: reusing previous graph\n", __func__);

        n_reused++;
    } else {
        res->reset();
}
```
The previous graph result is stored and it is retreived here and then used to
figure out if any parameters have changed that would cause the graph to be
rebuilt.

Now, I'm mostly interested in when a new graph is processed:
```c++
        ggml_backend_sched_reset(sched.get());
        ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);

        gf = model.build_graph(gparams);

        if (!ggml_backend_sched_alloc_graph(sched.get(), gf)) {
            LLAMA_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            ret = GGML_STATUS_ALLOC_FAILED;
            return nullptr;
        }
```
So the scheduler is reset and the graph is built from the model and then the 
graph is allocated by the scheduler.
```c++
bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {

    sched->cur_copy = sched->next_copy; // something to do with pipline parallelism
    sched->next_copy = (sched->next_copy + 1) % sched->n_copies; // round robin over copies

    ggml_backend_sched_split_graph(sched, graph);

    if (!ggml_backend_sched_alloc_splits(sched)) {
        return false;
    }

    sched->is_alloc = true;

    return true;
}
```
cur_copy and next_copy are related to the pipelining implementation. They manage
a set of rotating buffers to enable the overlapping of computation and data
transfers, which is crucial for achieving high performance in heterogeneous
computing environments (like CPU + GPU). More on this later.

So we then have ggml_backend_sched_split_graph.
```c++
void ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    // reset splits
    sched->n_splits = 0;
    sched->n_graph_inputs = 0;
    sched->is_reset = false;

    struct ggml_init_params params = {
        /* .mem_size =   */ sched->context_buffer_size,
        /* .mem_buffer = */ sched->context_buffer,
        /* .no_alloc =   */ true
    };

    ggml_free(sched->ctx);

    sched->ctx = ggml_init(params);
    if (sched->ctx == NULL) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }
```
So we can see that a new ggml_context (memory area for tensor and graph meta
data) is created for the scheduler. This is a no_alloc context, so no memory
allocation will be done in this context. This is just for tracking the graph
```console
(gdb) p params.mem_size / (1024 * 1024)
$7 = 19

(gdb) p graph->n_leafs
$8 = 76

(gdb) p graph->n_nodes
$9 = 242
```

The we have pass 1 of the scheduler:
```c++
    // pass 1: assign backends to ops with pre-allocated inputs
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        // Note that the scheduler was reset so all leaf backend ids are -1 at first
        int * leaf_backend_id = &tensor_backend_id(leaf);
        // do not overwrite user assignments
        if (*leaf_backend_id == -1) {
            *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
        }
    }
```
tensor_backend_id is a macro which is defined like this:
```c++
#define hash_id(tensor) ggml_hash_find_or_insert(&sched->hash_set, tensor)
#define tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
```
So this would be equivalent to:
```c++
```console
(gdb) p ggml_hash_find((const ggml_hash_set*)&(sched->hash_set), leaf)
$10 = 688
```
So we have a hash set:
```console
(gdb) ptype sched->hash_set
type = struct ggml_hash_set {
    size_t size;
    ggml_bitset_t *used;
    ggml_tensor **keys;
}
```
The size is that the number of elements for the used bitset, which is used to
keep tracke of which slots in the keys array are occupied.
The keys array holds pointers to the tensors that have been assigned to this
scheduler. Notice that it holds pointers and these pointers point to the hash
value tensor backend ids array.

```console
(gdb) p h
$44 = 887
(gdb) p ggml_bitset_get(hash_set->used, 887)
$43 = true
(gdb) f
#0  ggml_hash_find_or_insert (hash_set=0x555556916cf0, key=0x555556a73cd0)
    at /home/danbev/work/ai/llama.cpp-debug/ggml/src/ggml-impl.h:306
306	            hash_set->keys[i] = key;

(gdb) p hash_set->keys[887]
$45 = (ggml_tensor *) 0x555556a73cd0
(gdb) p key
$46 = (ggml_tensor *) 0x555556a73cd0
```
So the hash set has a bitset for each slot which can we can use the hash of
the node to query if it is set. And we can get to the tensor using the keys if
the bitset is set.

So at this point the tensor has been added to the hash_set, the bitset for the
hash of the tensor is set and the entry in keys points to the tensor. But we
still don't have a backend id for this tensor yet:
```c++
(gdb) p sched->hv_tensor_backend_ids[688]
$14 = -1
```
So the pointer to the tensor, leaf in this case, is hashed to get an index into
the hv_tensor_backend_ids (hash value tensor backend ids) array, which holds the
backend ids for each tensor.

```c++
struct ggml_backend_sched {
    ...
    // hash map of the nodes in the graph
    struct ggml_hash_set  hash_set;
    int                 * hv_tensor_backend_ids; // [hash_set.size]
    struct ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]
```
Notice that hv_tensor_backend_ids is an array of ints, and this is the backend
id for a tensor. So we can quickly lookup a tensor using:
```
1. Use the tensor pointer to find its index in the hash set, it returns i lets say.
2. So we know that hash_set->keys[i] is the our tensors pointer.
3. To get the backend id for this tensor we can do hv_tensor_backend_ids[i].
```
Which is exactly what the macro does.

So the backend id has not been assigned so we will call ggml_backend_sched_backend_id_from_cur.
This function is used multiple times so it is worth looking at in detail:
```c++
static int ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, struct ggml_tensor * tensor) {
    int cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
    if (cur_backend_id != -1) {
        SET_CAUSE(tensor, "1.dst");
        return cur_backend_id;
    }
    ...
```
Notice that we are passing the same tensor for both the tensor and op parameters:
And the tensor (leaf) in this case is:
```console
(gdb) p *leaf
$2 = {type = GGML_TYPE_Q4_0, buffer = 0x555555f122d0, ne = {288, 288, 1, 1}, nb = {18, 162, 46656, 46656},
op = GGML_OP_NONE, op_params = {0 <repeats 16 times>}, flags = 0, src = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0, view_offs = 0, data = 0x7fffc10ca040,
name = "blk.0.attn_q.weight", '\000' <repeats 44 times>, 
extra = 0x7ffff738b508 <ggml_repack_get_optimal_repack_type(ggml_tensor const*)::q4_0_8x8_q8_0>, padding = "\000\000\000\000\000\000\000"}
```
We can see that this tensor has a buffer which is happends when the model is
loaded.
Next the available backends will be iterated over to find a backend that both
support this type of buffer and the operation as well:
```c++
    // find highest prio backend that supports the buffer type and the op
    for (int i = 0; i < sched->n_backends; i++) {
        if (ggml_backend_supports_buft(sched->backends[i], buffer->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            return i;
        }
    }

    return -1;
}
```
The first thing that happens is that we get the buffer from the tensor, if this
tensor is a view then we use the the view source buffer, otherwise we use the
tensor buffer.

The SET_CAUSE macro has to be enabled explicitly by setting the following
value to 1:
```c++
#if 0
#define GGML_SCHED_MAX_SPLITS_DEBUG 4096
static char causes[GGML_DEFAULT_GRAPH_SIZE*16 + GGML_SCHED_MAX_SPLITS_DEBUG*GGML_SCHED_MAX_SPLIT_INPUTS][128]; // debug only
#define SET_CAUSE(node, ...) sprintf(causes[hash_id(node)], __VA_ARGS__)
#define GET_CAUSE(node) causes[hash_id(node)]
#else
#define SET_CAUSE(node, ...)
#define GET_CAUSE(node) ""
#endif
```
This provides information about the reason why a particular backend was chosen
for a and inforation about each pass.
```
- 1.dst chosen because tensor already lives in a buffer whose backend supports the op (direct).
- 1.vsrc chosen from view_src’s buffer.
- Hard abort if preallocated but unsupported; no cause stored.
- 1.inp assigned because it’s an input tensor (fallback to last backend/CPU).
- 1.off offload to higher-priority backend that wants the op while weight is on host.
- 1.wgtN picked the weight’s backend (N is src index).
- 2.sup extended an existing backend assignment while expanding forward/backward because backend supports the op.
- 3.best picked backend with most supported inputs for an unassigned node.
- 3.upg upgraded to a higher-priority backend with same buffer type that supports all inputs.
- 4.vsrc inherited backend from view source in pass 4.
- 4.cur inherited backend from the current node for remaining sources.
- 4.cpy this tensor is a created copy for a split/pipeline copy.
- usr set manually by ggml_backend_sched_set_tensor_backend.
```

_wip_


After that we set the inputs for the graph, and then compute the graph:
```c++
    // set the input data for the input tensors
    {
        //const auto t_start_us = ggml_time_us();

        res->set_inputs(&ubatch);

        //LLAMA_LOG_INFO("graph set inputs time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);
    }

    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
```
