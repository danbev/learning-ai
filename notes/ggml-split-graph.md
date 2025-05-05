## GGML Split Graph
So this is about splitting a computation graph into subgraphs that can be
computed on the same backend device.

An instance of `ggml_backend_sched` has a field which is an array of
`ggml_backend_sched_split` structs:
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

For example:
```console
// If we have a graph with 6 nodes:
graph->nodes = [Node0, Node1, Node2, Node3, Node4, Node5]

// And it gets split into 2 splits:

splits[0] = {
    i_start: 0,
    i_end: 3,
    backend_id: GPU_BACKEND   // First 3 nodes run on GPU
}
splits[1] = {
    i_start: 3,
    i_end: 6,
    backend_id: CPU_BACKEND   // Last 3 nodes run on CPU
}
```
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
So this function which is quite long is about setting up this information in
the sched struct:
```c++
(gdb) ptype *sched
type = struct ggml_backend_sched {
    bool is_reset;
    bool is_alloc;
    int n_backends;
    ggml_backend_t backends[16];
    ggml_backend_buffer_type_t bufts[16];
    ggml_gallocr_t galloc;
    ggml_hash_set hash_set;
    int *hv_tensor_backend_ids;
    ggml_tensor **hv_tensor_copies;
    int *node_backend_ids;
    int *leaf_backend_ids;
    int *prev_node_backend_ids;
    int *prev_leaf_backend_ids;
    ggml_cgraph graph;
    ggml_backend_sched_split *splits;
    int n_splits;
    int splits_capacity;
    int n_copies;
    int cur_copy;
    ggml_backend_event_t events[16][4];
    ggml_tensor *graph_inputs[10];
    int n_graph_inputs;
    ggml_context *ctx;
    ggml_backend_sched_eval_callback callback_eval;
    void *callback_eval_user_data;
    char *context_buffer;
    size_t context_buffer_size;
    int debug;
}
```

First, a new ggml_context will be created for the scheduler.
```c++
// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
static void ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
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
The graph I'm using here is very simple, and it consists of a multiplication
operation (mul = a * b). So we have one operation node and two leaf nodes:
```console
(gdb) p *graph
$9 = {size = 2048, n_nodes = 1, n_leafs = 2, nodes = 0x7ffff6600500, grads = 0x0, grad_accs = 0x0, leafs = 0x7ffff6604500,
  visited_hash_set = {size = 4099, used = 0x7ffff6610518, keys = 0x7ffff6608500}, order = GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT}

(gdb) p ggml_graph_print(g1)
=== GRAPH ===
n_nodes = 1
 -   0: [     1,     1,     1]              MUL  
n_leafs = 2
 -   0: [     1,     1]     NONE                a
 -   1: [     1,     1]     NONE                b
========================================
```
Notice that we start with 0 splits. So now, lets look at the first pass in this
function.

#### First pass
```c++
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        int * leaf_backend_id = &tensor_backend_id(leaf);
        // do not overwrite user assignments
        if (*leaf_backend_id == -1) {
            *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
        }
    }
```
Notice the use of the `tensor_backend_id` macro. This macro is defined as
follows:
```c++
#define hash_id(tensor) ggml_hash_find_or_insert(&sched->hash_set, tensor)
#define tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
```
This would be the same as the following command:
```console
(gdb) p sched->hv_tensor_backend_ids[ggml_hash(leaf) % sched->hash_set->size]
$8 = -1
```
This is iterating over all the leafs in the graph and if the leaf does not have
a backend assigned to it yet it will assign a backend to it using the function
`ggml_backend_sched_backend_id_from_cur`.

So, -1 means that this tensor does not have a backend assigned to it yet.
```console
(gdb) p *leaf_backend_id
$10 = -1
```
So this will call the function `ggml_backend_sched_backend_id_from_cur`.
```c++
// returns the backend that should be used for the node based on the current locations
static int ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, struct ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    int cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
    if (cur_backend_id != -1) {
        SET_CAUSE(tensor, "1.dst");
        return cur_backend_id;
    }
    ...
    // graph input
    if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        SET_CAUSE(tensor, "1.inp");
        return cur_backend_id;
    }
```
Since this is `a` an input tensor it will be assigned to the last backend which
as the comment says is assumed to be the CPU.
The same process will happen for the tensor `b`.

After that the nodes will also be visited:
```c++
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * node_backend_id = &tensor_backend_id(node);
        // do not overwrite user assignments
        if (*node_backend_id == -1) {
            *node_backend_id = ggml_backend_sched_backend_id_from_cur(sched, node);
        }
    }
```
This is doing the same thing as above but for the nodes. But in the
ggml_backend_sched_backend_id_from_cur function, this is not an input tensor
so it will not take the same path as above. The following block will iterate
over the sources of the `mul` operation.
```c++
    // operations with weights are preferably run on the same backend as the weights
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        const struct ggml_tensor * src = tensor->src[i];
        if (src == NULL) {
            continue;
        }
        // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
        // not an ideal solution
        if (tensor->op != GGML_OP_ROPE && src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            int src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
            // check if a backend with higher prio wants to offload the op
            if (src_backend_id == sched->n_backends - 1) {
                for (int b = 0; b < src_backend_id; b++) {
                    if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                        SET_CAUSE(tensor, "1.off");
                        return b;
                    }
                }
            }
            SET_CAUSE(tensor, "1.wgt%d", i);
            return src_backend_id;
        }
    }

    return -1;
```
But since the source buffers are null, -1 will be returned.
```console
(gdb) p tensor->src[0]->buffer
$21 = (ggml_backend_buffer *) 0x0
(gdb) p tensor->src[1]->buffer
$22 = (ggml_backend_buffer *) 0x0
```

#### Second pass

```c++
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
```
So we are again getting the node_backend_id for the current node:
```console
(gdb) p sched->hv_tensor_backend_ids[ggml_hash(node) % sched->hash_set->size]
$24 = -1
```
So actually nothing will be done in the above for loop.
The next block we have is the following, which notice will iterate over the
nodes in reverse order (staring from the last node):
```c++
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
```
Notice that this code is actually identical to the previous blocks. The
reason for first doing an iteration from the first to the last node is to
identify a node that has a backend assigned it will assign that backend to
subsequent nodes. The backward iteration will do something simlar but from the
other end, and assign backends to the previous nodes (if they don't have one
assiged). Something like this:
```
Initial state:
Node1(unassigned) -> Node2(unassigned) -> Node3(GPU) -> Node4(unassigned) -> Node5(GPU)

After forward pass:
Node1(unassigned) -> Node2(unassigned) -> Node3(GPU) -> Node4(GPU) -> Node5(GPU)

After backward pass:
Node1(GPU) -> Node2(GPU) -> Node3(GPU) -> Node4(GPU) -> Node5(GPU)
```
Notice that the above code skips nodes that have been assigned the CPU backend
so this is about grouping GPU tensor/nodes(operation nodes) together.
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
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
```
This is doing pretty much the same as the previous blocks but with the difference
that the CPU backend is not skipped. So any unassigned nodes will be assigned to
the CPU backend. Notice the function calls ggml_backend_sched_set_if_supported so
not all nodes will be assigned to the CPU backend, they can still have -1 as
their backend id after this pass.

#### Pass 3
Like mentioned above there might still be nodes that have not been assigned a
backend id. 

```c++
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        int * node_backend_id = &tensor_backend_id(node);
        if (*node_backend_id == -1) {
            // unassigned node: find the backend with the most supported inputs
            int n_supported_best = -1;
            for (int b = 0; b < sched->n_backends; b++) {
                if (ggml_backend_supports_op(sched->backends[b], node)) {
                    int n_supported = 0;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && ggml_backend_sched_buffer_supported(sched, src, b)) {
                            n_supported++;
                        }
                    }
                    if (n_supported > n_supported_best) {
                        n_supported_best = n_supported;
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
    }
```
So in the case where a node is assigned to a backend the code will iterate over
all the backends before the current backend id (ones with higher priority). And
notice that there is check to make sure that the buffer types are the same type.
Backends for GPUs like CUDA and ROCm might use the same buffer type for their
GPU memory for example. And CPU and BLAS might use the same buffer type for CPU
based memory.
```
Backend          Buffer Type        Physical Memory
CUDA GPU     ->  CUDA_MEMORY    ->  NVIDIA GPU Memory
ROCm GPU     ->  ROCM_MEMORY    ->  AMD GPU Memory
CPU          ->  HOST_MEMORY    ->  System RAM
BLAS         ->  HOST_MEMORY    ->  System RAM
```
So the above code is making sure that the buffer types are the same between the
higher priority backend and the current backend, and the operation must also be
supported by the higher priority backend.

#### Pass 4

```c++
    // pass 4: assign backends to remaining src from dst and view_src
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * cur_backend_id = &tensor_backend_id(node);
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
    }
```
A view tensor will be assingned to the same backend as its source tensor. If the
source is not a view then assign it to the same backend as the current node.

#### Pass 5

```c++
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
Notice that `i` is outside of the loop and used after the loop. So we can see
that this starts by extracting the first split from the splits array and then 
loops over all the nodes in the graph. And for each it will check if the
operation is not a view operation and in that case will set the current split's
backend_id to the backend id of the current node. This is `1` in this case
which is the CPU backend.

The number of splits are initially 16, that is the array of splits is allocated
in `ggml_backend_sched_new` with an initial capacity of 16:
```c++
    const int initial_splits_capacity = 16;
```

Next it will set the split `i_start` index and `n_inputs` to 0. And notice that
it will continue iterating over the nodes in the graph (not from the beginning
but from `i` above.
```c++
        int cur_backend_id = split->backend_id;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];

            if (ggml_is_view_op(node->op)) {
                continue;
            }

            const int node_backend_id = tensor_backend_id(node);

            assert(node_backend_id != -1); // all nodes should be assigned by now

            // check if we should start a new split based on the sources of the current node
            bool need_new_split = false;
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

            if (node_backend_id != cur_backend_id || need_new_split) {
                split->i_end = i;
                i_split++;
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
The above will check if the current nodes backend id is the same as the current
splits backend id. If they are the same we just continue but if they differ
then `need_new_split` will be set to true. This will then be used in the last
if block. Notice that the split's array can grow if needed. This is how a split
is created.

Next the sources of the current node are iterated over.
```c++
            // find inputs that are not on the same backend
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }

                size_t src_id = hash_id(src);
                const int src_backend_id = sched->hv_tensor_backend_ids[src_id];
                assert(src_backend_id != -1); // all inputs should be assigned by now

                if (src->flags & GGML_TENSOR_FLAG_INPUT && sched->n_copies > 1) {
                    if (tensor_id_copy(src_id, src_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[src_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy;
                            if (c == sched->cur_copy) {
                                tensor_copy = src; // use the original tensor as the current copy
                            } else {
                                tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                                ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            }
                            if (sched->n_copies > 1) {
                                ggml_set_input(tensor_copy);
                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, src_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_graph_inputs = sched->n_graph_inputs++;
                        GGML_ASSERT(n_graph_inputs < GGML_SCHED_MAX_SPLIT_INPUTS);
                        sched->graph_inputs[n_graph_inputs] = src;
                    }
                }
```
The above is looking up the src tensors backend id using the scheduler's
hash values backend map (`hv_tensor_backend_ids`).
```console
(gdb) p src->name
$29 = "l_a", '\000' <repeats 60 times>
(gdb) p src->flags
$27 = 1
(gdb) p src->flags & GGML_TENSOR_FLAG_INPUT
$28 = 1
(gdb) p sched->n_copies
$30 = 1
```
Notice that `n_copies` is 1 so the above block will not be executed in this
case. `n_copies` is used to determine how many copies of a tensor needs to
exist, for example if when the same input tensor needs to exist on multiple
backends simultaneously for computation purposes. `n_copies` is set in
`ggml_backend_sched_new` and in this case parallel is false so `n_copies` is:
```c++
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;

#ifndef GGML_SCHED_MAX_COPIES
#define GGML_SCHED_MAX_COPIES 4
#endif
```
```c++
    if (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }
```

```console
    backend name   number of inputs
              ↓    ↓
## SPLIT #0: CPU # 0 inputs:
node #  0 (       MUL):                  l_r (   0K) [  CPU         ]:                  l_a (   0K) [  CPU         ]                  l_b (   0K) [  CPU         ]
```
In this case there are 0 inputs and I found the `:` after the inputs a little
misleading the first time looking at this.

```c++
    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        int * tmp = sched->node_backend_ids;
        sched->node_backend_ids = sched->prev_node_backend_ids;
        sched->prev_node_backend_ids = tmp;

        tmp = sched->leaf_backend_ids;
        sched->leaf_backend_ids = sched->prev_leaf_backend_ids;
        sched->prev_leaf_backend_ids = tmp;
    }
```
Notice that this is reusing the memory of the previous backend ids by swapping
the pointers.

```c++
    int graph_size = std::max(graph->n_nodes, graph->n_leafs) + sched->n_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*sched->n_copies;
    if (sched->graph.size < graph_size) {
        sched->graph.size = graph_size;
        sched->graph.nodes = (ggml_tensor **) realloc(sched->graph.nodes, graph_size * sizeof(struct ggml_tensor *));
        sched->graph.leafs = (ggml_tensor **) realloc(sched->graph.leafs, graph_size * sizeof(struct ggml_tensor *));
        GGML_ASSERT(sched->graph.nodes != NULL);
        GGML_ASSERT(sched->graph.leafs != NULL);
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;
```
```c++
    struct ggml_cgraph * graph_copy = &sched->graph;
    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];
        split->graph = ggml_graph_view(graph, split->i_start, split->i_end);

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        for (int j = 0; j < split->n_inputs; j++) {
            assert(graph_copy->size > (graph_copy->n_nodes + 1));

            struct ggml_tensor * input = split->inputs[j];
            const size_t input_id = hash_id(input);
            struct ggml_tensor * input_cpy = tensor_id_copy(input_id, split->backend_id, sched->cur_copy);

            // add a dependency to the input source so that it is not freed before the copy is done
            struct ggml_tensor * input_dep = ggml_view_tensor(sched->ctx, input);
            input_dep->src[0] = input;
            sched->node_backend_ids[graph_copy->n_nodes] = sched->hv_tensor_backend_ids[input_id];
            graph_copy->nodes[graph_copy->n_nodes++] = input_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
            graph_copy->nodes[graph_copy->n_nodes++] = input_cpy;
        }

        for (int j = split->i_start; j < split->i_end; j++) {
            assert(graph_copy->size > graph_copy->n_nodes);
            sched->node_backend_ids[graph_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];
        }
    }
```
```c++
    if (sched->n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        for (int i = 0; i < sched->n_graph_inputs; i++) {
            struct ggml_tensor * input = sched->graph_inputs[i];
            size_t id = hash_id(input);
            int backend_id = tensor_backend_id(input);
            for (int c = 0; c < sched->n_copies; c++) {
                struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                assert(graph_copy->size > graph_copy->n_leafs);
                graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
            }
        }

        for (int i = 0; i < sched->n_splits; i++) {
            struct ggml_backend_sched_split * split = &sched->splits[i];
            int backend_id = split->backend_id;
            for (int j = 0; j < split->n_inputs; j++) {
                struct ggml_tensor * input = split->inputs[j];
                size_t id = hash_id(input);
                for (int c = 0; c < sched->n_copies; c++) {
                    struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                    sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                    assert(graph_copy->size > graph_copy->n_leafs);
                    graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
                }
            }
        }
    }
```
```c++
    // add leafs from the original graph
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        sched->leaf_backend_ids[graph_copy->n_leafs] = tensor_backend_id(leaf);
        assert(graph_copy->size > graph_copy->n_leafs);
        graph_copy->leafs[graph_copy->n_leafs++] = leaf;
    }
}
```


```c++
    struct ggml_cgraph * graph_copy = &sched->graph;

    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];
        split->graph = ggml_graph_view(graph, split->i_start, split->i_end);

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        for (int j = 0; j < split->n_inputs; j++) {
            assert(graph_copy->size > (graph_copy->n_nodes + 1));

            struct ggml_tensor * input = split->inputs[j];
            const size_t input_id = hash_id(input);
            struct ggml_tensor * input_cpy = tensor_id_copy(input_id, split->backend_id, sched->cur_copy);

            // add a dependency to the input source so that it is not freed before the copy is done
            struct ggml_tensor * input_dep = ggml_view_tensor(sched->ctx, input);
            input_dep->src[0] = input;
            sched->node_backend_ids[graph_copy->n_nodes] = sched->hv_tensor_backend_ids[input_id];
            graph_copy->nodes[graph_copy->n_nodes++] = input_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
            graph_copy->nodes[graph_copy->n_nodes++] = input_cpy;
        }

        for (int j = split->i_start; j < split->i_end; j++) {
            assert(graph_copy->size > graph_copy->n_nodes);
            sched->node_backend_ids[graph_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];
        }
    }
```
I've not come accross a graph view before but it is what it sounds like, it uses
the same nodes as the original graph but only has access to a subset of the nodes.
A graph view is 
```c++
struct ggml_cgraph ggml_graph_view(struct ggml_cgraph * cgraph0, int i0, int i1) {
    struct ggml_cgraph cgraph = {
        /*.size             =*/ 0,
        /*.n_nodes          =*/ i1 - i0,
        /*.n_leafs          =*/ 0,
        /*.nodes            =*/ cgraph0->nodes + i0,
        /*.grads            =*/ NULL, // gradients would need visited_hash_set
        /*.grad_accs        =*/ NULL,
        /*.leafs            =*/ NULL,
        /*.visited_hash_set =*/ { 0, NULL, NULL },
        /*.order            =*/ cgraph0->order,
    };

    return cgraph;
}
```
So this will then call:
```c++
    if (!ggml_backend_sched_alloc_splits(sched)) {
        return false;
    }
```
In this case everything is run on the CPU backend to the backend ids will not
change:
```c++
static bool ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    bool backend_ids_changed = false;
    for (int i = 0; i < sched->graph.n_nodes; i++) {
        if (sched->node_backend_ids[i] != sched->prev_node_backend_ids[i] &&
            sched->bufts[sched->node_backend_ids[i]] != sched->bufts[sched->prev_node_backend_ids[i]]) {
            backend_ids_changed = true;
            break;
        }
    }
    if (!backend_ids_changed) {
        for (int i = 0; i < sched->graph.n_leafs; i++) {
            if (sched->leaf_backend_ids[i] != sched->prev_leaf_backend_ids[i] &&
                sched->bufts[sched->leaf_backend_ids[i]] != sched->bufts[sched->prev_leaf_backend_ids[i]]) {
                backend_ids_changed = true;
                break;
            }
        }
    }

    // allocate graph
    if (backend_ids_changed || !ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
```

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
In this case the graph will need to be reallocated because the number of nodes
in the graph has changed:
```console
(gdb) s
ggml_gallocr_needs_realloc (galloc=0x5555557040a0, graph=0x5555556f9088)
    at /home/danbev/work/ai/learning-ai/fundamentals/ggml/ggml/src/ggml-alloc.c:822
822	    if (galloc->n_nodes != graph->n_nodes) {
(gdb) p galloc->n_nodes
$47 = 1
(gdb) p graph->n_nodes
$48 = 2
```
So we will call `ggml_gallocr_reserve` and pass in galloc and the graph.
So this is the same galloc as we used initialy I think:
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

    // reset allocators
    for (int i = 0; i < galloc->n_buffers; i++) {
        ggml_dyn_tallocr_reset(galloc->buf_tallocs[i]);
    }
```
```console
(gdb) p min_hash_size
$51 = 6
(gdb) p galloc->hash_set.size
$52 = 3
```
So the hash_set will be freed and a new one will be created with a size of 6.
The hash values will also be freed and a new one will be created with the same
size as the hash set.
The dynamic tensor allocators will also be reset.

```c++
    // allocate in hash table
    ggml_gallocr_alloc_graph_impl(galloc, graph, node_buffer_ids, leaf_buffer_ids);
```

```c++
        // allocate tensor from the buffer
        struct ggml_dyn_tallocr * alloc = galloc->buf_tallocs[buffer_id];
        ggml_backend_buffer_type_t buft = galloc->bufts[buffer_id];
        size_t size = ggml_backend_buft_get_alloc_size(buft, node);
        size_t offset = ggml_dyn_tallocr_alloc(alloc, size, node);
        hn->buffer_id = buffer_id;
        hn->offset = offset;
        return;
```
So  hash node's will be created and set. 
