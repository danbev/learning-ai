## ggml_gallocr_reserve
This function will be called by [ggml_backend_sched_alloc_splits](./ggml-split-graph.md).
```c++
bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT((int)sched->hash_set.size >= graph->n_nodes + graph->n_leafs);

    ggml_backend_sched_split_graph(sched, graph);


    if (!ggml_backend_sched_alloc_splits(sched)) {  <-------
        return false;
    }

    sched->is_alloc = true;

    return true;
}
```
And if we look in `ggml_backend_sched_alloc_splits`
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
```
So this will iterate over all the nodes in the graph. Notice the struct
`ggml_backend_sched` has two arrays with previous backend node ids and the
previous leaf ids. This above is checking if any of the node's have been
assigned to a different backend and if so setting `backend_ids_changed` to true
and breaking out of the loop.

```c++
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
        // the re-allocation may cause the split inputs to be moved to a different address
        ggml_backend_sched_synchronize(sched);
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_changed);
#endif
        ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids);
        if (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            return false;
        }
    }

    return true;
}
```
In this case (this debugging session) the backend_ids have not changed but
the check in `ggml_gallocr_needs_realloc` is comparing the following values:
```console
(gdb) p galloc->n_nodes
$7 = 0
(gdb) p graph->n_nodes
$8 = 1
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
    ...
```
This is why we can see the following debug message when using a debug build:
```console
ggml_gallocr_needs_realloc: graph has different number of nodes
```
This feels somewhat misleading as the graph has not changed and perhaps this
should only be printed if galloc->n_nodes is not zero.

And the above is clause will be entered and the following will be printed:
```console
ggml_gallocr_alloc_graph: reallocating buffers automatically
```
So `ggml_gallocr_reserve` will be called.

```c++
bool ggml_gallocr_reserve(ggml_gallocr_t galloc, struct ggml_cgraph *graph) {
    return ggml_gallocr_reserve_n(galloc, graph, NULL, NULL);
}
```
Which brings us to the following function:
```c++
bool ggml_gallocr_reserve_n(ggml_gallocr_t galloc, struct ggml_cgraph * graph, const int * node_buffer_ids, const int * leaf_buffer_ids) {
    size_t min_hash_size = graph->n_nodes + graph->n_leafs;
    // add 25% margin to avoid hash collisions
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

    // reset allocators
    for (int i = 0; i < galloc->n_buffers; i++) {
        ggml_dyn_tallocr_reset(galloc->buf_tallocs[i]);
    }

    // allocate in hash table
    ggml_gallocr_alloc_graph_impl(galloc, graph, node_buffer_ids, leaf_buffer_ids);
    ...
}
```
```c++
static void ggml_gallocr_alloc_graph_impl(ggml_gallocr_t galloc, struct ggml_cgraph * graph, const int * node_buffer_ids, const int * leaf_buffer_ids) {
    // clear hash tables
    ggml_hash_set_reset(&galloc->hash_set);
    memset(galloc->hash_values, 0, sizeof(struct hash_node) * galloc->hash_set.size);

    // allocate leafs
    // these may be tensors that the application is not using in the graph, but may still want to allocate for other purposes
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        ggml_gallocr_allocate_node(galloc, leaf, get_node_buffer_id(leaf_buffer_ids, i));
    }
    ...
```
So in out case the first leaf is:
```console
(gdb) p *leaf
$11 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0, ne = {1, 1, 1, 1}, nb = {4, 4, 4, 4}, op = GGML_OP_NONE, 
  op_params = {0 <repeats 16 times>}, flags = 1, src = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0, 
  view_offs = 0, data = 0x0, name = "a", '\000' <repeats 62 times>, extra = 0x0, padding = "\000\000\000\000\000\000\000"}
```

```c++
static void ggml_gallocr_allocate_node(ggml_gallocr_t galloc, struct ggml_tensor * node, int buffer_id) {
    GGML_ASSERT(buffer_id >= 0);
    struct hash_node * hn = ggml_gallocr_hash_get(galloc, node);

    if (!ggml_gallocr_is_allocated(galloc, node) && !ggml_is_view(node)) {
        hn->allocated = true;
        assert(hn->offset == 0);
```
So this is just checking if the node has already been allocated and if not it
will set the allocated flag of the `hash_node` to true. This struct looks like
this:
```console
(gdb) ptype hn
type = struct hash_node {
    int n_children;
    int n_views;
    int buffer_id;
    size_t offset;   // offset within the buffer
    _Bool allocated;
} *
```
So we should be able to get the `hash_node` for a node using the following:
```console
(gdb) p ggml_hash(t) % galloc->hash_set->size
$18 = 1

(gdb) p galloc->hash_values[ggml_hash(t) % galloc->hash_set->size]
$17 = {n_children = 0, n_views = 0, buffer_id = 0, offset = 0, allocated = false}
```

```c++

        // try to reuse a parent's buffer (inplace)
        if (ggml_op_can_inplace(node->op)) {
```
Next, it will check if the operation can be done inplace:
```c++
static bool ggml_op_can_inplace(enum ggml_op op) {
    switch (op) {
        case GGML_OP_SCALE:
        case GGML_OP_DIAG_MASK_ZERO:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_ADD:
        case GGML_OP_ADD1:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_UNARY:
        case GGML_OP_ROPE:
        case GGML_OP_RMS_NORM:
        case GGML_OP_SOFT_MAX:
            return true;
        default:
            return false;
    }
}
```
The current node is not an operations so this will not be entered this time
but later for the `mul` node it should be.
```c++
        // allocate tensor from the buffer
        struct ggml_dyn_tallocr * alloc = galloc->buf_tallocs[buffer_id];
        ggml_backend_buffer_type_t buft = galloc->bufts[buffer_id];
        size_t size = ggml_backend_buft_get_alloc_size(buft, node);
        size_t offset = ggml_dyn_tallocr_alloc(alloc, size, node);
        hn->buffer_id = buffer_id;
        hn->offset = offset;
```
Now, this is setting the `hash_node`'s buffer id to the buffer id, which was
passed in to this function as an argument (in this session is is 0). And it
will get an offset into this buffer:
```console
(gdb) p *hn
$23 = {n_children = 0, n_views = 0, buffer_id = 0, offset = 0, allocated = true}
```
And for the next leaf which is tensor b:
```console
(gdb) p *hn
$26 = {n_children = 0, n_views = 0, buffer_id = 0, offset = 32, allocated = true}
```
So we have the two tensors a and be allocated in the same buffer (0) and a
starts at offset 0 and b at offset 32.

Just to recap that the `galloc` struct has a field named  `buf_tallocs` which is
an array with then number of elements that there are buffers (`n_buffers`):
```console
(gdb) p *galloc.buf_tallocs[0]
$28 = {alignment = 32, n_free_blocks = 1, free_blocks = {{offset = 0, size = 9223372036854775807}, {
      offset = 0, size = 0} <repeats 255 times>}, max_size = 0}

(gdb) p galloc.buf_tallocs[0]->n_free_blocks
$29 = 1

(gdb) p galloc.buf_tallocs[0]->free_blocks[0]
$30 = {offset = 0, size = 9223372036854775807}
```
So we one free block (`n_free_blocks`) and it starts at offset 0 and has a size
9223372036854775807
```
block (variable name in code above):

   0                                                                  9223372036854775807
   +---------------------------------------------------------------------+
   |                         Free Block                                  |
   +---------------------------------------------------------------------+
   ↑                                                                     ↑
   offset                                                               size
```
And we retrieved the backend buffer type for the buffer id, and also calculate
the size in bytes that the node needs which is 4 in this case. With this we will
call `ggml_dyn_tallocr_alloc`. The offset returned will be 32 for this node and
the next free block will be:
```console
(gdb) p galloc.buf_tallocs[0].free_blocks[0]
$44 = {offset = 32, size = 9223372036854775775}
```
```
block:

   0       31                                                       SIZE_MAX
   +---------------------------------------------------------------------+
   |        |                 Free block                                 |
   +---------------------------------------------------------------------+
            ↑                                                     ↑
          offset                                                 size
                                                            (SIZE_MAX - 32)
```
We can check these values using:
```console
(gdb) p galloc->hash_values[ggml_hash(graph->leafs[0]) % galloc->hash_set->size]
$47 = {n_children = 0, n_views = 0, buffer_id = 0, offset = 0, allocated = true}

(gdb) p galloc->hash_values[ggml_hash(graph->leafs[1]) % galloc->hash_set->size]
$48 = {n_children = 0, n_views = 0, buffer_id = 0, offset = 32, allocated = true}
```

Next, we will iterate over the nodes:
```c++
    // count number of children and views
    // allocate other graph inputs and leafs first to avoid overwriting them
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];

        // TODO: better way to add external dependencies
        // GGML_OP_NONE does not appear normally in the graph nodes, but is used by ggml-backend to add dependencies to
        // control when some tensors are allocated and freed. in this case, the dependencies are in `src`, but the node
        // itself is never used and should not be considered a dependency
        if (ggml_is_view(node) && node->op != GGML_OP_NONE) {
            struct ggml_tensor * view_src = node->view_src;
            ggml_gallocr_hash_get(galloc, view_src)->n_views += 1;
        }

        if (node->flags & GGML_TENSOR_FLAG_INPUT) {
            ggml_gallocr_allocate_node(galloc, graph->nodes[i], get_node_buffer_id(node_buffer_ids, i));
        }

        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }

            ggml_gallocr_hash_get(galloc, src)->n_children += 1;

            // allocate explicit inputs
            if (src->flags & GGML_TENSOR_FLAG_INPUT) {
                ggml_gallocr_allocate_node(galloc, src, get_node_buffer_id(node_buffer_ids, i));
            }
        }
    }
```
So this will iterate over all the nodes in the graph and for each src update
the `n_children` of the src tensor `hash_node`. And notice that the next check
is checking if the src tensors is an input tensor and not if the current node is.

So this will call `ggml_gallocr_allocate_node` similar to what we did for the
leafs above. And again notice that this is passing in src, that is tensor `a`
and not the `mul` tensor. But did we not already allocate this tensor?   Yes,
and there is a check in for this in `ggml_gallocr_allocate_node`:
```c++
    if (!ggml_gallocr_is_allocated(galloc, node) && !ggml_is_view(node)) {
        hn->allocated = true;
        assert(hn->offset == 0);
```

Next tensors (not inputs or leafs are allocated, so operations?):
```c++
    // allocate tensors
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int buffer_id = get_node_buffer_id(node_buffer_ids, i);

        // allocate parents (only leafs need to be allocated at this point)
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * parent = node->src[j];
            if (parent == NULL) {
                continue;
            }
            ggml_gallocr_allocate_node(galloc, parent, buffer_id);
        }

        // allocate node
        ggml_gallocr_allocate_node(galloc, node, buffer_id);
```
In this case parent will be tensor a:
```console
(gdb) p parent->name
$37 = "a", '\000' <repeats 62 times>
```
And we will again call `ggml_gallocr_allocate_node` for this tensor and like
before it has already been allocated.

Now, we will allocate the mul tensor:
```console
```c++
        // allocate node
        ggml_gallocr_allocate_node(galloc, node, buffer_id);
```
Now, in this case the operation can be made inplace:
```console
(gdb) p ggml_op_can_inplace(node->op)
$42 = true
```

```c++
        if (ggml_op_can_inplace(node->op)) {
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                struct ggml_tensor * parent = node->src[i];
                if (parent == NULL) {
                    continue;
                }

                // if the node's data is external, then we cannot re-use it
                if (!ggml_gallocr_is_own(galloc, parent)) {
                    AT_PRINTF("not reusing parent %s for %s as %p is external\n", parent->name, node->name, parent->data);
                    continue;
                }

                // outputs cannot be reused
                if (parent->flags & GGML_TENSOR_FLAG_OUTPUT || (parent->view_src != NULL && parent->view_src->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                    AT_PRINTF("not reusing parent %s for %s as it is an output\n", parent->name, node->name);
                    continue;
                }

                if (!ggml_are_same_layout(node, parent)) {
                    AT_PRINTF("not reusing parent %s for %s as layouts are different\n", parent->name, node->name);
                    continue;
                }

                struct hash_node * p_hn = ggml_gallocr_hash_get(galloc, parent);
                if (p_hn->n_children == 1 && p_hn->n_views == 0) {
                    if (ggml_is_view(parent)) {
                        struct ggml_tensor * view_src = parent->view_src;
                        struct hash_node * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                        if (view_src_hn->n_views == 1 && view_src_hn->n_children == 0 && view_src->data == parent->data) {
                            AT_PRINTF("reusing view parent %s (%s) for %s\n", parent->name, view_src->name, node->name);
                            assert(view_src_hn->offset == p_hn->offset);
                            hn->buffer_id = p_hn->buffer_id;
                            hn->offset = p_hn->offset;
                            p_hn->allocated = false; // avoid freeing the parent
                            view_src_hn->allocated = false;
                            return;
                        }
                    } else {
                        AT_PRINTF("reusing parent %s for %s\n", parent->name, node->name);
                        hn->buffer_id = p_hn->buffer_id;
                        hn->offset = p_hn->offset;
                        p_hn->allocated = false; // avoid freeing the parent
                        return;
                    }
                }
            }
        }
```
In our case `a` has one child, `n_children` is 1 and no views, `n_views` is 0,
and it is not a view. Now, this is interesting what happnes next:
```c++
                        hn->buffer_id = p_hn->buffer_id;
                        hn->offset = p_hn->offset;
                        p_hn->allocated = false; // avoid freeing the parent
```
`hn` is the `hash_node` for the mul tensor, and it will have its buffer id set
to the one that `a` has, and also the same offset that `a` has. _So this is how
tensors in a graph can be reused_. The mul tensor will now use `a`'s buffer id
and offset.

The `hash_node` struct is created for each tensor and hold allocation releated
metadata and is used for graph execution planning. It provides information about
where the tensor is allocated (which buffer and offset), if it has children
and if it is allocated or not. But this is still about planning I think and
the other struct we will see later hold similar information but for a different
stage so it might be good to think of it this way. Like even during planning
if all the leafs and nodes are iterated over it would be possible to see when
a node/leaf is no longer used anymore and then reuse the buffer and offset for
another tensor, just like we saw above.

Back in `ggml_gallocr_alloc_graph_impl` we then have the following which is
just for logging:
```c++
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * parent = node->src[j];
            if (parent == NULL) {
                continue;
            }
            AT_PRINTF("%s", parent->name);
            if (j < GGML_MAX_SRC - 1 && node->src[j + 1] != NULL) {
                AT_PRINTF(", ");
            }
        }
        AT_PRINTF("\n");
```
So we are still iterating over the nodes and we only have one operation node
which is mul, and again it's parents will be iterated over:
```c++
        // update parents
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * parent = node->src[j];
            if (parent == NULL) {
                continue;
            }
            struct hash_node * p_hn = ggml_gallocr_hash_get(galloc, parent);
            p_hn->n_children -= 1;

            AT_PRINTF("parent %s: %d children, %d views, allocated: %d\n",
                parent->name, p_hn->n_children, p_hn->n_views, p_hn->allocated);

            if (p_hn->n_children == 0 && p_hn->n_views == 0) {
                if (ggml_is_view(parent)) {
                    struct ggml_tensor * view_src = parent->view_src;
                    struct hash_node * view_src_hn = ggml_gallocr_hash_get(galloc, view_src);
                    view_src_hn->n_views -= 1;
                    AT_PRINTF("view_src %s: %d children, %d views\n",
                        view_src->name, view_src_hn->n_children, view_src_hn->n_views);
                    if (view_src_hn->n_views == 0 && view_src_hn->n_children == 0 && view_src_hn->allocated) {
                        ggml_gallocr_free_node(galloc, view_src);
                    }
                }
                else if (p_hn->allocated) {
                    ggml_gallocr_free_node(galloc, parent);
                }
            }
            AT_PRINTF("\n");
        }
```
This time, the parents `hash_node` will have its `n_children` decremented which
will bring the count down to zero. It is not a view so that is skipped and since
we set `p_hn->allocated=false` ealier this will not be freed which is good as
it is reused by the mul tensor.
But for the `b` tensor it will be freed as it has not been reused.

This will then return back to `ggml_gallocr_reserve_n`:
```c++
    // set the node_allocs from the hash table
    if (galloc->n_nodes < graph->n_nodes) {
        free(galloc->node_allocs);
        galloc->node_allocs = calloc(graph->n_nodes, sizeof(struct node_alloc));
        GGML_ASSERT(galloc->node_allocs != NULL);
    }
```
So `node_allocs` is another member of the `galloc` struct and is an array of
`node_alloc` structs and the size if `galloc->n_nodes`:
```console
(gdb) ptype *galloc->node_allocs
type = struct node_alloc {
    struct tensor_alloc dst;
    struct tensor_alloc src[10];
}

(gdb) ptype galloc->node_allocs[0].dst
type = struct tensor_alloc {
    int buffer_id;
    size_t offset;
    size_t size_max;
}

(gdb) ptype galloc->leaf_allocs[0]
type = struct leaf_alloc {
    struct tensor_alloc leaf;
}
```
Now, we can see that this struct alos stores a buffer id and an offset, just
like the `hash_node` struct did and this had me confused for a while. But notice
that we have the addition of the `size_max` field which represents the size
allocated for the tensor.


So the following will iterate over all the nodes in the graph.
The `mul` node is not a view and does not have any data so the else block will
be executed.
```c++
    galloc->n_nodes = graph->n_nodes;
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct node_alloc * node_alloc = &galloc->node_allocs[i];
        if (node->view_src || node->data) {
            node_alloc->dst.buffer_id = -1;
            node_alloc->dst.offset = SIZE_MAX;
            node_alloc->dst.size_max = 0;
        } else {
            struct hash_node * hn = ggml_gallocr_hash_get(galloc, node);
            node_alloc->dst.buffer_id = hn->buffer_id;
            node_alloc->dst.offset    = hn->offset;
            node_alloc->dst.size_max  = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], node);
        }
        ...
```
Notice that the `hash_node` for this node is retrieved (so from the "planning")
and it will set the `buffer_id` and `offset` to the values that were of
`node_alloc`: 
```console
(gdb) p *hn
$66 = {n_children = 0, n_views = 0, buffer_id = 0, offset = 0, allocated = true}
```
And recall that these are the values for the leaf tensor `a` as it was resued
by above. But we are also setting the `size_max` field which is the size which
is this case will be `ggml_nbytes(node)`.

Following that we will iterate of the nodes sources:
```c++
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (!src || src->view_src || src->data) {
                node_alloc->src[j].buffer_id = -1;
                node_alloc->src[j].offset = SIZE_MAX;
                node_alloc->src[j].size_max = 0;
            } else {
                struct hash_node * hn = ggml_gallocr_hash_get(galloc, src);
                node_alloc->src[j].buffer_id = hn->buffer_id;
                node_alloc->src[j].offset   = hn->offset;
                node_alloc->src[j].size_max = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], src);
            }
        }
    }
```
And the above will set the sources for the `node_alloc` to the source hash_node
information for the leafs a and b.
```console
(gdb) p src->name
$76 = "a", '\000' <repeats 62 times>

(gdb) p *ggml_gallocr_hash_get(galloc, src)
$75 = {n_children = 0, n_views = 0, buffer_id = 0, offset = 0, allocated = false}
```
Remember that when we reused a for mul we only set a's hash_node's `allocated`
to false, it is still in the hash table.
```console
(gdb) p node_alloc->src[0]
$77 = {buffer_id = 0, offset = 0, size_max = 4}
```
So the source for this `node_alloc` can be found in buffer 0, at offset 0 and
has a size of 4 bytes. And the same for `b`:
```console
(gdb) p node_alloc->src[1]
$80 = {buffer_id = 0, offset = 32, size_max = 4}
```
The rest of the sources will have `buffer_id` set to -1 and `offset` to
`SIZE_MAX` and `size_max` to 0.

After that we will do something similar but for leafs:
```c++
    if (galloc->n_leafs < graph->n_leafs) {
        free(galloc->leaf_allocs);
        galloc->leaf_allocs = calloc(graph->n_leafs, sizeof(galloc->leaf_allocs[0]));
        GGML_ASSERT(galloc->leaf_allocs != NULL);
    }
    galloc->n_leafs = graph->n_leafs;
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        struct hash_node * hn = ggml_gallocr_hash_get(galloc, leaf);
        if (leaf->view_src || leaf->data) {
            galloc->leaf_allocs[i].leaf.buffer_id = -1;
            galloc->leaf_allocs[i].leaf.offset = SIZE_MAX;
            galloc->leaf_allocs[i].leaf.size_max = 0;
        } else {
            galloc->leaf_allocs[i].leaf.buffer_id = hn->buffer_id;
            galloc->leaf_allocs[i].leaf.offset = hn->offset;
            galloc->leaf_allocs[i].leaf.size_max = ggml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], leaf);
        }
    }
```
```console
(gdb) p galloc->leaf_allocs[0]
$84 = {leaf = {buffer_id = 0, offset = 0, size_max = 4}}
(gdb) p galloc->leaf_allocs[1]
$85 = {leaf = {buffer_id = 0, offset = 32, size_max = 4}}
```

Next we will iterate over all the buffers (just 1 in this case)
```c++
    // reallocate buffers if needed
    for (int i = 0; i < galloc->n_buffers; i++) {
        // if the buffer type is used multiple times, we reuse the same buffer
        for (int j = 0; j < i; j++) {
            if (galloc->buf_tallocs[j] == galloc->buf_tallocs[i]) {
                galloc->buffers[i] = galloc->buffers[j];
                break;
            }
        }

        size_t cur_size = galloc->buffers[i] ? ggml_backend_buffer_get_size(galloc->buffers[i]) : 0;
        size_t new_size = ggml_dyn_tallocr_max_size(galloc->buf_tallocs[i]);

        // even if there are no tensors allocated in this buffer, we still need to allocate it to initialize views
        if (new_size > cur_size || galloc->buffers[i] == NULL) {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: reallocating %s buffer from size %.02f MiB to %.02f MiB\n", __func__, ggml_backend_buft_name(galloc->bufts[i]), cur_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
#endif

            ggml_backend_buffer_free(galloc->buffers[i]);
            galloc->buffers[i] = ggml_backend_buft_alloc_buffer(galloc->bufts[i], new_size);
            if (galloc->buffers[i] == NULL) {
                GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(galloc->bufts[i]), new_size);
                return false;
            }
            ggml_backend_buffer_set_usage(galloc->buffers[i], GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        }
    }
```
The `new_size` will be 64:
```console
(gdb) p *galloc->buf_tallocs[0]
$90 = {alignment = 32, n_free_blocks = 1, free_blocks = {{offset = 32, size = 9223372036854775775},
    {offset = 0, size = 0} <repeats 255 times>}, max_size = 64}
```
And this will be passed to `ggml_backend_buft_alloc_buffer` which will allocate
a buffer of this size for the specific backend buffer type:
```c++
ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    if (size == 0) {
        // return a dummy buffer for zero-sized allocations
        return ggml_backend_buffer_init(buft, {}, NULL, 0);
    }

    return buft->iface.alloc_buffer(buft, size);
}
```
And this will end up in (ggml-backend.cpp):
```c++
static ggml_backend_buffer_t ggml_backend_cpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);

    if (data == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_cpu_buffer_i, data, size);
}
```
And then the usage of the buffer is set to `GGML_BACKEND_BUFFER_USAGE_COMPUTE`.
```console
(gdb) p *galloc->buffers[i]
$95 = {iface = {
    free_buffer = 0x55555558b5a5 <ggml_backend_cpu_buffer_free_buffer(ggml_backend_buffer_t)>,
    get_base = 0x55555558b56b <ggml_backend_cpu_buffer_get_base(ggml_backend_buffer_t)>,
    init_tensor = 0x0,
    memset_tensor = 0x55555558b5d3 <ggml_backend_cpu_buffer_memset_tensor(ggml_backend_buffer_t, ggml_tensor*, uint8_t, size_t, size_t)>,
    set_tensor = 0x55555558b61b <ggml_backend_cpu_buffer_set_tensor(ggml_backend_buffer_t, ggml_tensor*, void const*, size_t, size_t)>,
    get_tensor = 0x55555558b664 <ggml_backend_cpu_buffer_get_tensor(ggml_backend_buffer_t, ggml_tensor const*, void*, size_t, size_t)>,
    cpy_tensor = 0x55555558b6ad <ggml_backend_cpu_buffer_cpy_tensor(ggml_backend_buffer_t, ggml_tensor const*, ggml_tensor*)>,
    clear = 0x55555558b717 <ggml_backend_cpu_buffer_clear(ggml_backend_buffer_t, uint8_t)>,
    reset = 0x0},
  buft = 0x555555661680 <ggml_backend_cpu_buffer_type::ggml_backend_cpu_buffer_type>,
  context = 0x555555705540, size = 64, usage = GGML_BACKEND_BUFFER_USAGE_COMPUTE}
```
So this is how the actual memory buffer is allocated and is what the
`node_alloc` instances above point into.
