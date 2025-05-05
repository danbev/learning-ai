### GGML Backend Sched

### llama.cpp llama_new_context_with_model
In `llama_new_context_with_model` we have the following:
```c++
            // initialize scheduler with the worst-case graph
            uint32_t n_seqs = 1; // TODO: worst-case number of sequences
            uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);
            llama_token token = llama_token_bos(&ctx->model); // not actually used by llama_build_graph, but required to choose between token and embedding inputs graph

            llama_ubatch ubatch_pp = { true, n_tokens, n_tokens / n_seqs, n_seqs, &token, nullptr, nullptr, nullptr, nullptr, nullptr};
            ggml_cgraph * gf_pp = llama_build_graph(*ctx, ubatch_pp, true);

            // reserve pp graph first so that buffers are only allocated once
            ggml_backend_sched_reserve(ctx->sched.get(), gf_pp);
            int n_splits_pp = ggml_backend_sched_get_n_splits(ctx->sched.get());
            int n_nodes_pp = ggml_graph_n_nodes(gf_pp);

            // reserve with tg graph to get the number of splits and nodes
            llama_ubatch ubatch_tg = { true, 1, 1, n_seqs, &token, nullptr, nullptr, nullptr, nullptr, nullptr};
            ggml_cgraph * gf_tg = llama_build_graph(*ctx, ubatch_tg, true);
            ggml_backend_sched_reserve(ctx->sched.get(), gf_tg);
            int n_splits_tg = ggml_backend_sched_get_n_splits(ctx->sched.get());
            int n_nodes_tg = ggml_graph_n_nodes(gf_tg);

            // reserve again with pp graph to avoid ggml-alloc reallocations during inference
            gf_pp = llama_build_graph(*ctx, ubatch_pp, true);
            if (!ggml_backend_sched_reserve(ctx->sched.get(), gf_pp)) {
                LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
                llama_free(ctx);
                return nullptr;
            }
```
Where `pp` stands for prefill prompt (I think) which is the initial prompt and
can contain multiple tokens, and `tg` stands for token generation which is
usually a single token at a time. 

So a compute graph is first built for the prefill prompt and it uses the maximum
number of tokens (worst case) to know how much memory will be required for that
case should it happen.

So first we have the planning (hash nodes) is done by
`ggml_gallocr_alloc_graph_impl` (and note that the hash set in galloc is reset
each time `ggml_gallocr_reserve_n` is called). This will go through all the
nodes and leafs and figure out/plan the most optimal way to allocate the tensors.
This information will be stored in galloc's hash set and then used to set the
`node_alloc` instances:
```c++
            struct hash_node * hn = ggml_gallocr_hash_get(galloc, node);
            node_alloc->dst.buffer_id = hn->buffer_id;
            node_alloc->dst.offset    = hn->offset;
            node_alloc->dst.size_max  = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], node);
```
And notice that this is also setting the size of the tensor using the nodes
backend that was assign to it during planning. Lets take a look at this:
```console
(gdb) p node->name
$90 = "inp_embd", '\000' <repeats 55 times>
(gdb) p node->ne
$93 = {4096, 512, 1, 1}
(gdb) p node->op
$94 = GGML_OP_GET_ROWS
(gdb) p *hn
$89 = {n_children = 0, n_views = 0, buffer_id = 0, offset = 8394752, allocated = false}

(gdb) p ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], node)
$91 = 8388608
```
Notice that the number of tokens is `512` which matches the worse case. And this
will be done for all tensors in the graph. 
So after the reservation has been completed we can inspect the node_alloc:
```console
(gdb) p ctx->sched.get().galloc.node_allocs[0].dst
$106 = {buffer_id = 0, offset = 8394752, size_max = 8388608}
```
Notice that the size of the tensor is set to 8388608 which is the same as the
and the offset is 839475.

So that is after the reservation for the prefill prompt. Now, what happens when
we use the token generation (tg) prompt and perform a reserve on that
computation graph?  
This is the code that will do this:
```c++
            // reserve with tg graph to get the number of splits and nodes
            llama_ubatch ubatch_tg = { true, 1, 1, n_seqs, &token, nullptr, nullptr, nullptr, nullptr, nullptr};
            ggml_cgraph * gf_tg = llama_build_graph(*ctx, ubatch_tg, true);
            ggml_backend_sched_reserve(ctx->sched.get(), gf_tg);
            int n_splits_tg = ggml_backend_sched_get_n_splits(ctx->sched.get());
            int n_nodes_tg = ggml_graph_n_nodes(gf_tg);
```
And this might be pointing out the obvious but the number of nodes and leafs
are the same here for these computation graphs which one might expect::
```console
(gdb) p gf_pp->n_nodes
$10 = 1030
(gdb) p gf_pp->n_leafs 
$11 = 359
(gdb) p gf_tg->n_nodes 
$12 = 1030
(gdb) p gf_tg->n_leafs 
$13 = 359
```

And, `ggml_gallocr_reserve_n` will reset:
```c++
    size_t min_hash_size = graph->n_nodes + graph->n_leafs;
    // add 25% margin to avoid hash collisions
    // Same as (min_hash_size = min_hash_size + (min_hash_size / 4))
    min_hash_size += min_hash_size / 4;

    // initialize hash table
    if (galloc->hash_set.size < min_hash_size) {
        ggml_hash_set_free(&galloc->hash_set);
        galloc->hash_set = ggml_hash_set_new(min_hash_size);
        GGML_ASSERT(galloc->hash_set.keys != NULL);

        free(galloc->hash_values);
        galloc->hash_values = malloc(sizeof(struct hash_node) * galloc->hash_set.size);
        GGML_ASSERT(galloc->hash_values != NULL);
    }
```
So the above is comparing the schedulers current galloc hash set size and if it
is smaller than the minimum hash size of the passed in graph then it will free
the hash set and create a new one with the new size.

In our case the hash set will not be reset.

And the reset the dynamic tensor allocator:
```c++
    // reset allocators
    for (int i = 0; i < galloc->n_buffers; i++) {
        ggml_dyn_tallocr_reset(galloc->buf_tallocs[i]);
    }
```
And then perform the planning phase (setting gallocs hash nodes etc):
```c++
    // allocate in hash table
    ggml_gallocr_alloc_graph_impl(galloc, graph, node_buffer_ids, leaf_buffer_ids);
```
The above will first reset the galloc hash set and then go throught the node
planning stage creating all the hash nodes.
```c++
    // set the node_allocs from the hash table
    if (galloc->n_nodes < graph->n_nodes) {
        free(galloc->node_allocs);
        galloc->node_allocs = calloc(graph->n_nodes, sizeof(struct node_alloc));
        GGML_ASSERT(galloc->node_allocs != NULL);
    }
```
In our case the number of nodes is the same so the node_allocs will not be
reset.

And then we have the same code as before:
```c++
            struct hash_node * hn = ggml_gallocr_hash_get(galloc, node);
            node_alloc->dst.buffer_id = hn->buffer_id;
            node_alloc->dst.offset    = hn->offset;
            node_alloc->dst.size_max  = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], node);
```
But this time the node token size is different
```console
(gdb) p *node
$113 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0,
ne = {4096, 1, 1, 1}, nb = {4, 16384, 16384, 16384}, 
op = GGML_OP_GET_ROWS, op_params = {0 <repeats 16 times>}, flags = 0, src = {0x555555b64320, 0x7ffff4a40980, 0x0, 0x0, 0x0, 0x0, 
0x0, 0x0, 0x0, 0x0}, view_src = 0x0, view_offs = 0, data = 0x0,
name = "inp_embd", '\000' <repeats 55 times>, extra = 0x0, 
padding = "\000\000\000\000\000\000\000"}

(gdb) p *hn
$114 = {n_children = 0, n_views = 0, buffer_id = 0, offset = 524384, allocated = false}
(gdb) p hn->offset
$115 = 524384
(gdb) p ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], node)
$116 = 16384
```
So in this case we have a different offset and a different size for this tensor.

Then the splits an number of nodes for the token generation is retrieved:
```console
(gdb) p n_splits_tg
$118 = 1
(gdb) p n_nodes_tg
$119 = 1030
```
If we inspect the
```console
(gdb) p ggml_hash_find((const ggml_hash_set*)&(ctx->sched.get()->galloc.hash_set), gf_pp->nodes[0])
$34 = 1852
(gdb) p ggml_hash_find((const ggml_hash_set*)&(ctx->sched.get()->galloc.hash_set), gf_tg->nodes[0])
$35 = 1852
```
If we then compare the `node_alloc` for this node with the prefill prompt we
can see that the prefill prompt (the first one below) had a much larger max
size:
```console
(gdb) p ctx->sched.get().galloc.node_allocs[0].dst
$106 = {buffer_id = 0, offset = 8394752, size_max = 8388608}

(gdb) p ctx->sched.get()->galloc.node_allocs[0].dst
$38 = {buffer_id = 0, offset = 524384, size_max = 16384}
```
So at this point, after the token generation prompt has been reserved the
`max_size` is not set the the smaller size of 16384.
Now, if we don't reserve this pp graph again then the `max_size` will be set to
16384 (for this node that is). When we later call
`ggml_backend_sched_alloc_graph(lctx.sched.get(), gf)` in `llama_decode_impl`
there will be a check 
```c++
bool ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, struct ggml_cgraph * graph) {
    if (ggml_gallocr_needs_realloc(galloc, graph)) {
        ...
```
```c++
static bool ggml_gallocr_node_needs_realloc(ggml_gallocr_t galloc, struct ggml_tensor * node, struct tensor_alloc * talloc) {
    ...

    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct node_alloc * node_alloc = &galloc->node_allocs[i];

        if (!ggml_gallocr_node_needs_realloc(galloc, node, &node_alloc->dst)) {
            ...
```
So this is getting the node from the graph, and then the node_alloc from galloc:
```console
(gdb) p *node
$42 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0,
ne = {4096, 7, 1, 1}, nb = {4, 16384, 114688, 114688},
op = GGML_OP_GET_ROWS, op_params = {0 <repeats 16 times>}, flags = 0,
src = {0x555555b64320, 0x7ffff4a40980, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
view_src = 0x0, view_offs = 0, data = 0x0, name = "inp_embd", '\000' <repeats 55 times>,
extra = 0x0, padding = "\000\000\000\000\000\000\000"}

(gdb) p node_alloc.dst
$45 = {buffer_id = 0, offset = 8394752, size_max = 8388608}
```
And then calling `ggml_gallocr_node_needs_realloc`:
```c++
static bool ggml_gallocr_node_needs_realloc(ggml_gallocr_t galloc, struct ggml_tensor * node, struct tensor_alloc * talloc) {
    size_t node_size = 0;
    if (!node->data && !node->view_src) {
        GGML_ASSERT(talloc->buffer_id >= 0); // prevent segfault when misusing the API
        node_size = ggml_backend_buft_get_alloc_size(galloc->bufts[talloc->buffer_id], node);
    }
    return talloc->size_max >= node_size;
}
```
```console
(gdb) p talloc->size_max
$47 = 8388608
(gdb) p node_size
$46 = 114688
(gdb) p talloc->size_max >= node_size
$48 = 1

(gdb) p (bool) $48
$7 = true
```
Since we re-reserved with the prefill prompt graph the `max_size` is set to
8388608 and the node size is 114688. But if we did not re-reserve the prefill
we would have the following situation:
```console
(gdb) p (bool) (talloc->size_max >= node_size)
$11 = false
```
```c++
static bool ggml_gallocr_needs_realloc(ggml_gallocr_t galloc, struct ggml_cgraph * graph) {
    ...
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct node_alloc * node_alloc = &galloc->node_allocs[i];

        if (!ggml_gallocr_node_needs_realloc(galloc, node, &node_alloc->dst)) {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: node %s is not valid\n", __func__, node->name);
#endif
            return true;
        }
```
So this will indeed return true because of the negative sign. And this will
then return true which will cause a reserve:
```c++
bool ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, struct ggml_cgraph * graph) {
    if (ggml_gallocr_needs_realloc(galloc, graph)) {
        if (galloc->n_buffers == 1) {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: reallocating buffers automatically\n", __func__);
#endif
            if (!ggml_gallocr_reserve(galloc, graph)) {
                return false;
            }
        } else {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: cannot reallocate multi buffer graph automatically, call reserve\n", __func__);
#endif
            return false;
        }
    }
```
Our previous reserve was done during the creation of the context and part of
the startup and not a inference time.

So this is why there is another reserve for the prefill prompt graph so that the
sizes are correct for the first prompt as before this the sizes are those for
the token generation. And this is done here instead of in the `llama_decode_impl`
which happens at inference time.
```c++
            // reserve again with pp graph to avoid ggml-alloc reallocations during inference
            gf_pp = llama_build_graph(*ctx, ubatch_pp, true);
            if (!ggml_backend_sched_reserve(ctx->sched.get(), gf_pp)) {
                LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
                llama_free(ctx);
                return nullptr;
            }
```
Why does the prefill prompt computation graph need to be built again?  

The function `ggml_backed_sched_reserve` is also called in the following
function:
```c++
static void llama_kv_cache_update_impl(struct llama_context & lctx) {
    bool need_reserve = false;

    if (lctx.kv_self.has_shift) {
        if (!llama_kv_cache_can_shift(&lctx)) {
            GGML_ABORT("The current context does not support K-shift");
        }

        // apply K-shift if needed
        if (lctx.model.hparams.rope_type != LLAMA_ROPE_TYPE_NONE) {
            ggml_backend_sched_reset(lctx.sched.get());

            ggml_cgraph * gf = llama_build_graph_k_shift(lctx);

            ggml_backend_sched_alloc_graph(lctx.sched.get(), gf);

            llama_set_k_shift(lctx);

            llama_graph_compute(lctx, gf, lctx.cparams.n_threads, lctx.threadpool);

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

    // defragment the KV cache if needed
    if (lctx.kv_self.do_defrag) {
        llama_kv_cache_defrag_impl(lctx);

        need_reserve = true;

        lctx.kv_self.do_defrag = false;
    }

    // reserve a worst case graph again
    if (need_reserve) {
        // TODO: extract to a function
        // build worst-case graph
        uint32_t n_seqs = 1; // TODO: worst-case number of sequences
        uint32_t n_tokens = std::min(lctx.cparams.n_ctx, lctx.cparams.n_ubatch);
        llama_token token = llama_token_bos(&lctx.model); // not actually used by llama_build_graph, but required to choose between token and embedding inputs graph
        llama_ubatch ubatch = { true, n_tokens, n_tokens / n_seqs, n_seqs, &token, nullptr, nullptr, nullptr, nullptr, nullptr};
        ggml_cgraph * gf = llama_build_graph(lctx, ubatch, true);

        // initialize scheduler with the worst-case graph
        ggml_backend_sched_reset(lctx.sched.get());
        if (!ggml_backend_sched_reserve(lctx.sched.get(), gf)) {
            LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
        }
    }
}
```
Now, if a k-shift is needed then the scheduler is reset and a new graph is built
and the scheduler is allocated with the new graph which is then computed. But
this would remove the allocations as there will be a difference in the nodes
and the allocations. This is also the case for `llama_kv_cache_defrag_impl`.
I think this might be the reason why there is another worst case graph
reservation done to "reset" this back to the ealier reservation state. And this
is using the max prompt size as in this case the sequence length would be at
the max because the kv cache needed shifting or defragmenting (again I think).

### hash sets
What confused me initially has that when stepping through the code I found
myself mixing the `hash_set` that the `ggml_backend_sched` struct has and the
one that `ggml_gallocr` has.

The scheduler struct has a hash set
```c++
struct ggml_backend_sched {
    ...

    // hash map of the nodes in the graph
    struct ggml_hash_set  hash_set;
    int                 * hv_tensor_backend_ids; // [hash_set.size]
    struct ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]
```
So if the hash set size is 8209, then `hv_tensor_backend_ids` will be an array
of 8209 integers and `hv_tensor_copies` will be an array of 8209 pointers to
pointers to `ggml_tensor`.

Lets expore this a little. We can take a node from the graph, in this case
I'm using the graph for the prefill prompt `llama_new_context_with_model` in
llama.cpp:
```console
(gdb) p *gf_pp->nodes[0]
$51 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0, ne = {4096, 512, 1, 1}, nb = {4, 16384, 8388608, 
    8388608}, op = GGML_OP_GET_ROWS, op_params = {0 <repeats 16 times>}, flags = 0, src = {0x555555b64320, 0x7ffff4a40980, 0x0, 0x0, 
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0, view_offs = 0, data = 0x0,
    name = "inp_embd", '\000' <repeats 55 times>, 
    extra = 0x0, padding = "\000\000\000\000\000\000\000"}
```
We can look up this node using:
```console
(gdb) p ggml_hash_find((const ggml_hash_set*)&(ctx->sched.get()->hash_set), gf_pp->nodes[0])
$70 = 2660

(gdb) p ctx->sched.get().hv_tensor_backend_ids[$70]
$71 = -1
```
Now, this was done after a reserve, so the tensors have not been allocated to
backend yet, hence the -1. 

So this hash set is about accessing the backend for a tensor/node, and also
the copies (but that I not something that I've looked into yet).

The hash set in `ggml_gallocr` looks like this:
```c++
struct ggml_gallocr {
    ggml_backend_buffer_type_t * bufts; // [n_buffers]
    ggml_backend_buffer_t * buffers; // [n_buffers]
    struct ggml_dyn_tallocr ** buf_tallocs; // [n_buffers]
    int n_buffers;

    struct ggml_hash_set hash_set;
    struct hash_node * hash_values; // [hash_set.size]

    struct node_alloc * node_allocs; // [n_nodes]
    int n_nodes;

    struct leaf_alloc * leaf_allocs; // [n_leafs]
    int n_leafs;
};
```

```console
(gdb) p ctx->sched.get().galloc->hash_set
$74 = {size = 2053, used = 0x5555557b7350, keys = 0x555555831150}

(gdb) p ggml_hash_find((const ggml_hash_set*)&(ctx->sched.get()->galloc.hash_set), gf_pp->nodes[0])
$75 = 572

(gdb) p ctx->sched.get()->galloc.hash_values[$75]
$76 = {n_children = 0, n_views = 0, buffer_id = 0, offset = 8394752, allocated = false}
```
An instance of the `hash_node` struct is created for each tensor and holds
allocation releated metadata and is used for graph execution planning.
It provides information about where the tensor should be allocated (which buffer
and offset), if the tensor has children, and if it is allocated or not. But this
is still about _planning_. This information can be gathered during planning if
all the leafs and nodes are iterated over it would be possible to see when a
node/leaf is no longer used anymore and then reuse the buffer and offset for
another tensor.


In llama.cpp we have the following code in `llama_new_context_with_model`:
```c++
ctx->sched.reset(ggml_backend_sched_new(backend_ptrs.data(),
                                        backend_buft.data(),
                                        backend_ptrs.size(),
                                        max_nodes,
                                        pipeline_parallel));
```
```console
(gdb) p max_nodes
$26 = 8192
(gdb) p backend_ptrs.size()
$27 = 1
```
```c++
ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_buffer_type_t * bufts,
        int n_backends,
        size_t graph_size,
        bool parallel) {
    ...

    struct ggml_backend_sched * sched = (ggml_backend_sched *) calloc(1, sizeof(struct ggml_backend_sched));

    const char * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;
    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;

    sched->hash_set    = ggml_hash_set_new(graph_size);
```

```c++
struct ggml_backend_sched {
    ...
    struct ggml_hash_set  hash_set;

```
And recall that `ggml_hash_set` is defined as:
```c++
struct ggml_hash_set {
    size_t size;
    ggml_bitset_t * used;       // whether or not the keys are in use i.e. set
    struct ggml_tensor ** keys; // actual tensors in the set, keys[i] is only defined if ggml_bitset_get(used, i)
};
```
Now, this hash set can be used to quickly determine if a tensor in the keys has
been set by using the bitset.
The size of the hash set will a a prime number greater than or equal to the
size of the graph.

```c++
    sched->hv_tensor_backend_ids = (int *) malloc(sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
```
Notice that this is allocating memory for the number of tensors in the hash set
which in this case is 8209 (closest prime number) and the type of the elements
of `hv_tensor_backend_ids` which is `int * hv_tensor_backend_ids`.
```console
(gdb) p sched->hash_set.size * 4
$34 = 32836
```
```c++
    sched->hv_tensor_buffers     = (ggml_backend_buffer_type_t *) malloc(sched->hash_set.size * sizeof(sched->hv_tensor_buffers[0]));
```c++
    sched->hv_tensor_copies      = (ggml_tensor **) malloc(sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));
```
```c++
    const size_t ggml_sched_max_splits = graph_size; // at most there is one split for each node in the graph
    const size_t nodes_size = graph_size + ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2;
```
```console
(gdb) p nodes_size
$44 = 172032
```

