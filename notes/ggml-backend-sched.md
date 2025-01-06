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
we use the token generation prompt and perform a reserve on that computation
graph?  
Well, `ggml_gallocr_reserve_n` will reset the 
```c++
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
So in this case we have a different offset and a different size for the tensor.

Then the splits an number of nodes for the token generation is retrieved:
```console
(gdb) p n_splits_tg
$118 = 1
(gdb) p n_nodes_tg
$119 = 1030
```
Following that we will again reserve the prefill prompt graph so that the
sizes are correct for the first prompt as before this the sizes are those for
the token generation.
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

