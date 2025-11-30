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

So pass1 assigns backends to the leaf and nodes.
So just to recap, in pass 1 we iterated over all the leafs and all the nodes in the graph and
and try to assign backends to them from the current tensor.

A backend is only assigned if the tensor already has a buffer/view buffer, is flagged as graph input
(defaults to CPU), or an op’s weight buffer dictates a backend. If none of those conditions apply—common
for unallocated intermediates with no input flag or weight hint— the backend stays -1, and pass 2 is
what starts filling those gaps.

pass 2:
So we will again iterate over all the nodes in the graph.
```c++
    // expand gpu down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }

            int * node_backend_id = &tensor_backend_id(node);
            // if a backend was assinged in pass 1
            if (*node_backend_id != -1) {
                // Don't propagate cpu backend (lowest prio backend)
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    // propagete the current nodes backend as the current backend id
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                // If the current node does NOT have a backend id assigned we try to assign the current backend
                // id to it
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
```
So we are going through all the nodes, and if the current node has a backend id (it is not -1)
and we skip the CPU backend (the lowest priority backend), which breaks the propagation chain.
But otherwise this enables nodes that come after each other to have the same backend id assigned to them.

So a node that was assigned a backend in the first pass, will not be changed. But nodes that
were not assigned a backend might be assigned to the backend "chain" if supported

“?” means unassigned, “G” is a GPU backend, “C” is CPU. Arrows show propagation.
```
Initial after pass 1 (example):

idx: 0 1 2 3 4 5 6 7
     ? ? G ? C ? G ?

Forward scan, GPU-only:

- Start cur = -1
- idx2 has G → cur = G
- idx3 is ? → supports? yes → set to G
- idx4 is C → CPU resets chain → cur = -1
- idx5 is ? with cur = -1 → stays ?
- idx6 is G → cur = G
- idx7 is ? → supports? yes → set to G

After forward:

idx: 0 1 2 3 4 5 6 7
     ? ? G G C ? G G

Backward scan, GPU-only:

- Start cur = -1
- idx7 is G → cur = G
- idx6 is G → cur = G
- idx5 is ? → supports? yes → set to G
- idx4 is C → reset cur = -1
- idx3 is G → cur = G
- idx2 is G → cur = G
- idx1 is ? → supports? yes → set to G
- idx0 is ? → supports? yes → set to G

After backward:

idx: 0 1 2 3 4 5 6 7
   G G G G C G G G
```
If any ? doesn’t support G, it stays ?. The subsequent “rest” scans do the same
but allow CPU to propagate too, filling remaining ? where supported.

We will then again iterate over all the nodes in the graph:
```c++
    // expand rest down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }

            int * node_backend_id = &tensor_backend_id(node);
            // if the current node has a backend id assigned
            if (*node_backend_id != -1) {
                // set the current backend id to the current nodes backend id
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                // if the current node does not have a backend id assigned we try to assign the current backend
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
```

```c++
static void ggml_backend_sched_set_if_supported(ggml_backend_sched_t sched,
    struct ggml_tensor * node, int cur_backend_id, int * node_backend_id) {

    if (ggml_backend_supports_op(sched->backends[cur_backend_id], node)) {
        *node_backend_id = cur_backend_id;
        SET_CAUSE(node, "2.sup");
    }
}
```

Pass 3:
We will again iterate over all the nodes in the graph:
```c++
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        // skip view nodes
        if (ggml_is_view_op(node->op)) {
            continue;
        }

        int * node_backend_id = &tensor_backend_id(node);
        // If the current node does not yet have an assigned backend
        if (*node_backend_id == -1) {

            int n_supported_best = -1;
            // Iterate over all the backends
            for (int b = 0; b < sched->n_backends; b++) {

                // if the backend supports the current node op
                if (ggml_backend_supports_op(sched->backends[b], node)) {
                    int n_supported = 0;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        // if the op does not have any sources we continue
                        if (src == NULL) {
                            continue;
                        }

                        if ((tensor_backend_id(src) != -1 ||
                             tensor_backend_id(src->view_src) != -1) &&
                             ggml_backend_sched_buffer_supported(sched, src, b)) {
                            n_supported++;
                        }
                    }
                    // Keep track of the best supported backend
                    if (n_supported > n_supported_best) {
                        n_supported_best = n_supported;
                        // update the current nodes backend id (might be updated again later in the loop)
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.best");
                    }
                }
            }
        } else {
            // assigned node: upgrade to higher prio backend if possible
            for (int b = 0; b < *node_backend_id; b++) {
                if (sched->bufts[b] == sched->bufts[*node_backend_id] && ggml_backend_supports_op(sched->backends[b], node)) {
                    bool supported = true;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }

                        if (!ggml_backend_sched_buffer_supported(sched, src, b)) {
                            supported = false;
                            break;
                        }
                    }
                    if (supported) {
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.upg");
                        break;
                    }
                }
            }
        }
```
So the motivation here is to assign nodes to backens that are used the most which
avoids having to copy data between backends.

This brings us to pass 4.

Pass 4:
```c++
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * cur_backend_id = &tensor_backend_id(node);

        // if the current node is a view and has a backend assigned then we 
        // set the current backend id.
        if (node->view_src != NULL && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(node->view_src);
            SET_CAUSE(node, "4.vsrc");
        }

        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }

            int * src_backend_id = &tensor_backend_id(src);
            if (*src_backend_id == -1) {
                if (src->view_src != NULL) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    SET_CAUSE(src, "4.vsrc");
                } else {
                    *src_backend_id = *cur_backend_id;
                    SET_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        for (int b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
            ggml_backend_sched_set_if_supported(sched, node, b, cur_backend_id);
        }
        GGML_ASSERT(*cur_backend_id != -1);
    }
```

Before pass 5 lets take a look at hv_tensor_copies which is a field in the
scheduler struct:
```c++
    struct ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]
```
And the following macro:
```c++
#define tensor_id_copy(id, backend_id, copy_id)
  sched->hv_tensor_copies[ (id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
```

```
Dimension 1: tensor ID (hash_set.size different tensors)
Dimension 2: backend ID (n_backends)
Dimension 3: copy number (n_copies)

(lldb) p sched->hash_set.size
(size_t) 4099                  // possible tensor ids

(lldb) p sched->n_backends
(int) 3                        // backend ids 0, 1, 2

(lldb) p sched->n_copies
(int) 1                        // copy ids 0

tensor_id
     backend
0    [0, 1, 2]       ← 3 backend copies of tensor 0
1    [0, 1, 2]       ← 3 backend copies of tensor 1
2    [0, 1, 2]       ← 3 backend copies of tensor 2
...
4098 [0, 1, 2]       ← 3 backend copies of tensor 4098

Flattened layout:
[t0_b0, t0_b1, t0_b2, t1_b0, t1_b1, t1_b2, t2_b0, t2_b1, t2_b2, ...]

Index calculation:
index = tensor_id * (n_backends * n_copies) + backend_id * n_copies + copy_id

Examples:
index = 0  * (3 * 1) + 0 * 1 + 0 = 0   ← tensor 0, backend 0, copy 0
inext = 1  * (3 * 1) + 1 * 1 + 0 = 1   ← tensor 0, backend 1, copy 0
```
This macro will be used in pass 5.

Pass 5:
This is where we actually split and create copies if they are required.
```c++
    // pass 5: split graph, find tensors that need to be copied
    {
        int i_split = 0;
        struct ggml_backend_sched_split * split = &sched->splits[0];
        // find the backend of the first split, skipping view ops
        int i = 0;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (!ggml_is_view_op(node->op)) {
                split->backend_id = tensor_backend_id(node);
                break;
            }
        }
        split->i_start = 0;
        split->n_inputs = 0;
```
Notice that we get a pointer to the first split which is of type ggml_backend_sched_split:
```c++
struct ggml_backend_sched_split {
    int backend_id;            // Which backend this split runs on
    int i_start;               // Start index in graph->nodes
    int i_end;                 // End index in graph->nodes
    struct ggml_tensor * inputs[GGML_SCHED_MAX_SPLIT_INPUTS];  // Input tensors needed
    int n_inputs;              // Number of inputs
    struct ggml_cgraph graph;  // View into the main graph
};
```
So each split has a `backend_id` where this split will be run. The `i_start` and
`i_end` are the start and end indices into `graph->nodes`.

The `inputs` array is an array of tensors that are needed by this split and will
need to be copied:
```console
// A graph that runs like this:
Node1(GPU) -> Node2(GPU) -> Node3(CPU) -> Node4(CPU)

// It might get split into:
Split1: {
    backend_id: GPU_BACKEND,
    i_start: 0,
    i_end: 2,         // Covers Node1 and Node2
    inputs: [],       // Empty because all inputs are on GPU
    n_inputs: 0
}

Split2: {
    backend_id: CPU_BACKEND,
    i_start: 2,
    i_end: 4,         // Covers Node3 and Node4
    inputs: [Node2],  // Needs Node2's output from GPU
    n_inputs: 1
}
```

```c++
        int cur_backend_id = split->backend_id;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];

            if (ggml_is_view_op(node->op)) {
                continue;
            }

            const int node_backend_id = tensor_backend_id(node);

            // check if we should start a new split based on the sources of the current node
            bool need_new_split = false;
            // If the current nodes backend is the same as the current split backend but it has inputs
            if (node_backend_id == cur_backend_id && split->n_inputs > 0) {
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    struct ggml_tensor * src = node->src[j];
                    if (src == NULL) {
                        continue;
                    }

                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                        int src_backend_id = tensor_backend_id(src);
                        if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                            need_new_split = true;
                            break;
                        }
                    }
                    // check if the split has too many inputs
                    // FIXME: count the number of inputs instead of only checking when full
                    if (split->n_inputs == GGML_SCHED_MAX_SPLIT_INPUTS) {
                        const size_t id = hash_id(src);
                        int src_backend_id = sched->hv_tensor_backend_ids[id];
                        bool supported = ggml_backend_sched_buffer_supported(sched, src, cur_backend_id);
                        if (src_backend_id != cur_backend_id && tensor_id_copy(id, cur_backend_id, 0) == NULL && !supported) {
                            need_new_split = true;
                            break;
                        }
                    }
                }
            }
```
The checks above and later below will break out of the current split if one of
them is true. 

Next we have the following check:
```c++
            // if the current node backend id is different from the current split backend id
            if (node_backend_id != cur_backend_id || need_new_split) {
                // we create a new split
                split->i_end = i;
                i_split++;

                // ensure we have enough capacity for the split
                if (i_split >= sched->splits_capacity) {
                    sched->splits_capacity *= 2;
                    sched->splits = (ggml_backend_sched_split *)
                        realloc(sched->splits, sched->splits_capacity * sizeof(struct ggml_backend_sched_split));
                    GGML_ASSERT(sched->splits != NULL);
                }

                split = &sched->splits[i_split];
                split->backend_id = node_backend_id;
                split->i_start = i;
                split->n_inputs = 0;
                cur_backend_id = node_backend_id;
            }
```

```c++
            // find inputs that are not on the same backend
            for (int j = 0; j < GGML_MAX_SRC; j++) {

                // get the input tensor and check if it is NULL
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }

                size_t src_id = hash_id(src);
                const int src_backend_id = sched->hv_tensor_backend_ids[src_id];
                GGML_ASSERT(src_backend_id != -1); // all inputs should be assigned by now

                // if the source tensor is marked as an input tensor and we have/need multiple copies
                if (src->flags & GGML_TENSOR_FLAG_INPUT && sched->n_copies > 1) {

                    // check if we have already created copies for this tensor/backend (see below)
                    if (tensor_id_copy(src_id, src_backend_id, 0) == NULL) {
                        // If we have not copied it then we get the backend for this source tensor
                        ggml_backend_t backend = sched->backends[src_backend_id];
                        
                        // for each numberof copies we need, again this is related to pipelining
                        for (int c = 0; c < sched->n_copies; c++) {
                            // This is just a tensor but it has the same name as a macro which can
                            // be a little confusing at first. Perhaps it would be clearer to
                            // initialize this to nullptr to make it clear?
                            struct ggml_tensor * tensor_copy;

                            // First can use the original tensor (one of the pipelines can use it)
                            if (c == sched->cur_copy) {
                                tensor_copy = src; // use the original tensor as the current copy
                            } else {
                                // But for others pipelines we need to create a copy of the tensor
                                tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                                ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            }

                            // This sets the copied tensor as both input and
                            // output to prevent ggml-alloc from overwriting the tensor
                            // But notice that the actual check here is not required, we will
                            // only ever enter this block if n_copies > 1
                            if (sched->n_copies > 1) {
                                ggml_set_input(tensor_copy);
                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }

                            // assign the copy to the tensor id copy array
                            tensor_id_copy(src_id, src_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }

                        // update the number of inputs
                        int n_graph_inputs = sched->n_graph_inputs++;
                        GGML_ASSERT(n_graph_inputs < GGML_SCHED_MAX_SPLIT_INPUTS);
                        // add the source tensor to teh graph inputs
                        sched->graph_inputs[n_graph_inputs] = src;
                    }
                }
```
And here we see the first usage of tensor_id_copy:
```console
(lldb) p src_id
(size_t) 3912
(lldb) p src_backend_id
(const int) 1

(lldb) p  sched->hv_tensor_copies[ src_id * sched->n_backends * sched->n_copies + (src_backend_id) * sched->n_copies + 0]
(ggml_tensor *) nullptr
```
This will then be set in the above loop using
```c++
tensor_id_copy(src_id, src_backend_id, c) = tensor_copy;
```
Keep in mind that we are actually not copying anything at this point, we are just
wiring up the tensor object and recording where copied will live later.

Next, we have:
```c++
                // if the input tensor is on a different backend than the current split and
                // the backend cannot support the source tensor buffer
                // and the backend does not support the source tensor buffer (for example a CPU backend
                // cannot read a GPU buffer directly)
                if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                    // check if we have already created copies for this tensor/backend
                    if (tensor_id_copy(src_id, cur_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];

                        // create the number of copies we need.
                        for (int c = 0; c < sched->n_copies; c++) {
                            // So we duplicated the source tensor.
                            struct ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);

                            // set it as input and output so that it does not get optimized away as
                            // tensors that are not used as input or outputs can be reused.
                            if (sched->n_copies > 1) {
                                ggml_set_input(tensor_copy);
                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }

                            // set the copy in the tensor id copy array. 
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        // update the inputs for this split
                        int n_inputs = split->n_inputs++;
                        GGML_ASSERT(n_inputs < GGML_SCHED_MAX_SPLIT_INPUTS);
                        split->inputs[n_inputs] = src;
                    }
                    // Note that the following will re-write the graph! It is updating
                    // the current nodes input src[j] to point to the copy which is now
                    // on a different backend. So prior to this the src[j] pointed to the original
                    // tensor which is on a different backend. It will now instead point to the copy
                    // which is on the current split backend. The backend does not contain the data
                    // yet, that happens later.
                    node->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
                }
            }
        } // end of for loop over graph nodes

        // update the current splits end index
        split->i_end = graph->n_nodes;
        sched->n_splits = i_split + 1;
    }

    if (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }
```
So we have now gone through the enitre graph, and sched->node_backend_ids and
sched->leaf_backend_ids contain the current arrays we just computed.
```c++
    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        int * tmp = sched->node_backend_ids;
        // set the previous backend ids to the current ones, which will be updated
        // in the next section.
        sched->node_backend_ids = sched->prev_node_backend_ids;
        // store the current backend ids as the previous ones
        sched->prev_node_backend_ids = tmp;

        tmp = sched->leaf_backend_ids;
        sched->leaf_backend_ids = sched->prev_leaf_backend_ids;
        sched->prev_leaf_backend_ids = tmp;
    }
```

```c++
    int graph_size = std::max(graph->n_nodes, graph->n_leafs) + sched->n_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*sched->n_copies;
    if (sched->graph.size < graph_size) {
        sched->graph.size = graph_size;
        sched->graph.nodes = (ggml_tensor **) realloc(sched->graph.nodes, graph_size * sizeof(struct ggml_tensor *));
        sched->graph.leafs = (ggml_tensor **) realloc(sched->graph.leafs, graph_size * sizeof(struct ggml_tensor *));
        GGML_ASSERT(sched->graph.nodes != NULL);
        GGML_ASSERT(sched->graph.leafs != NULL);
    }
    // reset the graph node and leaf counts as we will be going through the graph
    // again.
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;
```
```c++
    struct ggml_cgraph * graph_copy = &sched->graph;

    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];
        // create a slice/view of the graph for this split
        split->graph = ggml_graph_view(graph, split->i_start, split->i_end);

        // Optimize this split of the graph. This needs to happen before we make graph_copy,
        // so they are in sync.
        ggml_backend_graph_optimize(sched->backends[split->backend_id], &split->graph);
```
```c++
static void ggml_backend_graph_optimize(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    if (backend->iface.graph_optimize != NULL) {
        backend->iface.graph_optimize(backend, cgraph);
    }
}
```
I've not come accross ggml_optimize before so lets take a look at a concrete 
implementation. This give each backend a chance to optimize the computation graph
specifically to the backend in question.
This can be things like fusing separate ggml operations into single kernels. For
example if we have multiple separate ggml operations like add, mul, norm in our
code, these might actually be fused together into a single kernel.
```c++
struct ggml_tensor * normalized = ggml_rms_norm(ctx, x, eps);
struct ggml_tensor * scaled = ggml_mul(ctx, normalized, weight);
struct ggml_tensor * result = ggml_add(ctx, scaled, bias);
```
Metal sees:
```
rms_norm → mul → add
```
And can fuse into single kernel that does all three operations in one pass.
* This means the intermediate results can be kept in registers and not written to
VRAM.
* Fewer kernel launches which reduces CPU overhead.

This is actually one of the reasons why ggml separates graph construction from
graph execution so that this kind of optimization can be done.

Looking at the Metal backend:
```c++
void ggml_graph_optimize(ggml_cgraph * gf) {
    constexpr int MAX_FUSE = 16;

    const int n = gf->n_nodes;

    enum ggml_op ops[MAX_FUSE];

    std::vector<node_info> nodes;
    nodes.reserve(gf->n_nodes);

    // fuse nodes:
    // we don't want to make reorders that break fusing, so we first pack all fusable tensors
    //   and perform the reorder over the fused nodes. after the reorder is done, we unfuse
    for (int i = 0; i < n; i++) {
        node_info node = {
            /*.node =*/ gf->nodes[i],
            /*.fused =*/ {},
        };

        // fuse only ops that start with these operations
        // can be expanded when needed
        if (node.op() == GGML_OP_ADD ||
            node.op() == GGML_OP_NORM ||
            node.op() == GGML_OP_RMS_NORM) {
            ops[0] = node.op();

            int f = i + 1;
            while (f < n && f < i + MAX_FUSE) {
                // conservatively allow fusing only these ops
                // can be expanded when needed
                if (gf->nodes[f]->op != GGML_OP_ADD &&
                    gf->nodes[f]->op != GGML_OP_MUL &&
                    gf->nodes[f]->op != GGML_OP_NORM &&
                    gf->nodes[f]->op != GGML_OP_RMS_NORM) {
                    break;
                }
                ops[f - i] = gf->nodes[f]->op;
                f++;
            }

            f -= i;
            for (; f > 1; f--) {
                if (ggml_can_fuse(gf, i, ops, f)) {
                    break;
                }
            }

            // add the fused tensors into the node info so we can unfuse them later
            for (int k = 1; k < f; k++) {
                ++i;

                // the .dst() becomes the last fused tensor
                node.add_fused(gf->nodes[i]);
            }
        }

        nodes.push_back(std::move(node));
    }

#if 1
    // reorder to improve concurrency
    const auto order = ggml_metal_graph_optimize_reorder(nodes);
#else
    std::vector<int> order(nodes.size());
    for (size_t i = 0; i < nodes.size(); i++) {
        order[i] = i;
    }
#endif

    // unfuse
    {
        int j = 0;
        for (const auto i : order) {
            const auto & node = nodes[i];

            gf->nodes[j++] = node.node;

            for (auto * fused : node.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }
}
```
Again, we are not actually copying tensors or fusing anything at this stage, but
the graph is arranged so that that backend can later recognize patterns that it
can fuse together.

After ggml_backend_graph_optimize. 
```c++
        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        for (int j = 0; j < split->n_inputs; j++) {
            assert(graph_copy->size > (graph_copy->n_nodes + 1));

            // get the first input tensor for this split
            struct ggml_tensor * input = split->inputs[j];
            const size_t input_id = hash_id(input);
            struct ggml_tensor * input_cpy = tensor_id_copy(input_id, split->backend_id, sched->cur_copy);

            // add a dependency to the input source so that it is not freed before the copy is done
            // Recall that we above updated the node src[j] to point to the copy. This
            // copy tensor node is not part of the split graph and could be freed as it is
            // seen as unused otherwise. So we create a view tensor to the original input tensor
            // and set it as the source of the view. This creates a dependency in the graph
            // so that that ggml-alloc will not free the original input tensor before the copy is done.
            struct ggml_tensor * input_dep = ggml_view_tensor(sched->ctx, input);
            input_dep->src[0] = input;

            // Recall that node_backend_ids is an array of ints that stores the backend ids
            // for nodes in the graph. We are setting index of the current position of 
            // graph_copy->n_nodes to the backend id of the input tensor.
            sched->node_backend_ids[graph_copy->n_nodes] = sched->hv_tensor_backend_ids[input_id];
            graph_copy->nodes[graph_copy->n_nodes++] = input_dep;
            // Notice the increment of n_nodes above.

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
            // Add the input copy to the graph copy
            graph_copy->nodes[graph_copy->n_nodes++] = input_cpy;
        }

        for (int j = split->i_start; j < split->i_end; j++) {
            assert(graph_copy->size > graph_copy->n_nodes);
            sched->node_backend_ids[graph_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];
        }
}
```

_wip_

```console
(lldb) expr sched->backends[0]->iface.get_name(sched->backends[0])
(const char *) $1 = 0x00000001004bdc7b "Metal"
(lldb) expr sched->backends[1]->iface.get_name(sched->backends[1])
(const char *) $2 = 0x000000010008f46a "BLAS"
(lldb) expr sched->backends[2]->iface.get_name(sched->backends[2])
(const char *) $3 = 0x000000010039bbae "CPU"
```

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
