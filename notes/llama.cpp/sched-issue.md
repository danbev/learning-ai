## Backend Scheduling Issue (maybe)
This is an issue I've run into when trying to get a multi-modal vision model
to work with the new Vision API.

My goal with this was to get some experience with the new Vision API and also
with a multi-modal model that uses cross-attention, like Llama 3.2 Vision, so
that I can hopefully contribute to this part of the project in the future.

To get something working I looked at Ollama's support for Llama 3.2 Vision
Instruct and the model the [provide](https://ollama.com/x/llama3.2-vision).
They have two models, one for the language model and one for the vision encoder.
In our case I made the assumption that we only want one model so that that is
what I opted for.

I wanted to follow the new Vision API and the Llava example that was provided
in https://github.com/ggerganov/llama.cpp/pull/9687. So I used the same image to
try to reproduce the same/simliar output.

### The Issue
While developing/debugging the model I added a number of tensors that are copies
of tensors used in the computation graph so that I could inspect their output
if the original tensor gets resued by the backend schdular, which I think is
something that it can do with tensors that are part of the graph. So this is a
way to inspect the output of a tensor which might get reused by the backend
scheduler.

So I added tensors like this:
```c++
    struct ggml_tensor * inp_raw_copy = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, image_size_width, image_size_height, n_channels, n_tiles);
    ggml_set_input(inp_raw_copy);
    ggml_set_name(inp_raw_copy, "inp_raw_copy");
    ggml_build_forward_expand(gf, inp_raw_copy);
```
Now, running the [example] with this code will produce a pretty resonable
output:
```console
The image shows a close-up of the Eiffel Tower in Paris, France. The tower is
made of metal and has a dark gray color. It is shaped like a square with four
sides, and it has a flat top. The background of the image is a light gray color.
```
This might not be perfect but at least the image is described and the vision
encoder produces something that the language model can also work with.

Now, if I comment out the code above, the output will be different. The output
will be something like:
```console
"The image shows a gray background..."
```

I initially thought it was because the image patch embeddings were not
being generated correctly, but when I've checked the output image patch
embeddings (uncommenting the code in `encode_image_with_ca_vision`) using:
```console
$ sha256sum image_patch_embeddings.bin
319cc0572866e3b165d5f59dc3e5709b87ec503ff6af10e3cd487d21e2ad17ab  image_patch_embeddings.bin
```
The image patch embeddings are the same with this code commmented out or not,
so it does not seem like removing this tensor has an effect on the image patch
embeddings (the vision encoder).

I also noticed that if I increase the number of layers that I offload to the GPU
this also effect the output. For example, if I change the number of layers from
30 to 36 the I will also see the output above with the "gray background".

It seems to me like if I make a change to the computation graph of the vision
model this can have an effect on the language model which I was not expecting
(not saying it is wrong as I'm unfamiliar the inner workings of the backend
scheduler). Almost like the graph are shared but I was thinking that they would
not be after calling `ggml_backend_sched_reset(ctx.sched)`.

Does anyone recognize this issue, or have any ideas where I should start looking
to try to figure this out?

The [example] contains the steps to convert the model, quantize it, and also run
it.

[example]: https://github.com/danbev/llama.cpp/tree/vision-api-mllama-example/examples/simple-vision-mllama#simple-vision-mllama-example

I opened this [discussion](https://github.com/ggerganov/llama.cpp/discussions/10780)
and got some helpful comments by slaren:
```
That's likely to cause the issue. The tensor data is stored in buffers allocated
by ggml_backend_sched, and it is reused for every graph evaluation. Effectively,
tensors allocated in the graph by ggml_backend_sched, become invalidated when
allocating another graph. By adding dummy tensors, you probably made the compute
buffer for the vision part large enough so that the output tensor is allocated
beyond the space that the next graph needs, but you cannot rely on this.
```
My initial thought was that since the vision model will copy the image patch
embeddings after it has computed the graph then it would not matter if there
tensors were added to the computation graph. But I think what slaren is saying
if I understood the comments correctly the issue is that the language model
and the vision model are both using the same `ggml_backend_sched`.
What will happen is that when the context is first created by calling
`llama_new_context_with_model` a new `ggml_backend_sched`:
```c++
            ctx->sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, pipeline_parallel));
```
So that will have created a new `ggml_backend_sched` for the language model.

Next there will be a micro batch (ubatch):
```c++
            // initialize scheduler with the worst-case graph
            uint32_t n_seqs = 1; // TODO: worst-case number of sequences
            uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);
            llama_token token = llama_token_bos(&ctx->model); // not actually used by llama_build_graph, but required to choose between token and embedding inputs graph

            llama_ubatch ubatch_pp = { true, n_tokens, n_tokens / n_seqs, n_seqs, &token, nullptr, 0, nullptr, nullptr, nullptr, nullptr};
```
With the number of tokens being the minimum of the context size and the micro
ubatch param:
```console
(gdb) p cparams.n_ctx
$7 = 4096
(gdb) p cparams.n_ubatch
$8 = 512
(gdb) p token
$10 = 128000
```
So the number of tokens will be 512. This micro batch will then be used to build
a graph for a prefill/prompt (pp, like the initial prompt): 
```c++
            ggml_cgraph * gf_pp = llama_build_graph(*ctx, ubatch_pp, true);
```
Now this will build a graph for the language model calling `build_mllama`.
Next, using this graph we will reserve the graph.
```c++

            // reserve pp graph first so that buffers are only allocated once
            ggml_backend_sched_reserve(ctx->sched.get(), gf_pp);
```
So this will first generate the splits for the graph, then use the graph to
optimize the memory usage, and then create leaf and node allocations for the
tensors of the graph (but the tensors will not have gotten their buffer or
data pointers updated yet as this is only a reservation).

For example:
```console
gdb) p *gf_pp->nodes[0]
$24 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0, ne = {4096, 512, 1, 1}, nb = {4, 16384, 8388608,
    8388608}, op = GGML_OP_GET_ROWS, op_params = {0 <repeats 16 times>}, flags = 0, src = {0x55555a3e3820, 0x555556b08a50, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0, view_offs = 0, data = 0x0, name = "inp_embd", '\000' <repeats 55 times>,
  extra = 0x0, padding = "\000\000\000\000\000\000\000"}

(gdb) p ctx->sched.get()->galloc->node_allocs[0]
$23 = {
  dst = {buffer_id = 1, offset = 8394752, size_max = 8388608},
  src = {
        {buffer_id = -1, offset = 18446744073709551615, size_max = 0},
        {buffer_id = 1, offset = 0, size_max = 2048},
        {buffer_id = -1, offset = 18446744073709551615, size_max = 0},
        {buffer_id = -1, offset = 18446744073709551615, size_max = 0},
        {buffer_id = -1, offset = 18446744073709551615, size_max = 0},
        {buffer_id = -1, offset = 18446744073709551615, size_max = 0},
        {buffer_id = -1, offset = 18446744073709551615, size_max = 0},
        {buffer_id = -1, offset = 18446744073709551615, size_max = 0},
        {buffer_id = -1, offset = 18446744073709551615, size_max = 0},
        {buffer_id = -1, offset = 18446744073709551615, size_max = 0}
   }
}
(gdb) p ctx->sched.get()->galloc->hash_values[ggml_hash(gf_pp->nodes[0]) % ctx->sched.get()->galloc->hash_set->size]
$30 = {n_children = 0, n_views = 0, buffer_id = 1, offset = 8394752, allocated = false}
```
Next we have:
```c++
            int n_splits_pp = ggml_backend_sched_get_n_splits(ctx->sched.get());
            int n_nodes_pp = ggml_graph_n_nodes(gf_pp);
```
So we are doing this to get the number of splits and the number of nodes that
for a prefill/prompt graph.
```console
(gdb) p n_splits_pp
$31 = 93
(gdb) p n_nodes_pp
$32 = 1030
```
Next we to the same thing for a token generation (tg, where the ubatch n_tokens
is 1):
```c++
            // reserve with tg graph to get the number of splits and nodes
            llama_ubatch ubatch_tg = { true, 1, 1, n_seqs, &token, nullptr, 0, nullptr, nullptr, nullptr, nullptr};
            ggml_cgraph * gf_tg = llama_build_graph(*ctx, ubatch_tg, true);
            ggml_backend_sched_reserve(ctx->sched.get(), gf_tg);
            int n_splits_tg = ggml_backend_sched_get_n_splits(ctx->sched.get());
            int n_nodes_tg = ggml_graph_n_nodes(gf_tg);
```
Then we re-reserve the prefill/prompt graph, so when the prompt is decoded 
the graph will already be reserved (node_allocs and leaf_allocs will have been
populated and ready for use )
```console
            // reserve again with pp graph to avoid ggml-alloc reallocations during inference
            gf_pp = llama_build_graph(*ctx, ubatch_pp, true);
            if (!ggml_backend_sched_reserve(ctx->sched.get(), gf_pp)) {
                LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
                llama_free(ctx);
                return nullptr;
            }
```
Later when `llama_decode_internal` is called it will perform the following:
```c++
        ggml_backend_sched_reset(lctx.sched.get());
        ggml_backend_sched_set_eval_callback(lctx.sched.get(), lctx.cparams.cb_eval, lctx.cparams.cb_eval_user_data);

        ggml_cgraph * gf = llama_build_graph(lctx, ubatch, false);

        ...
        ggml_backend_sched_alloc_graph(lctx.sched.get(), gf);

        llama_set_inputs(lctx, ubatch);

        const auto compute_status = llama_graph_compute(lctx, gf, n_threads, threadpool);
```
The call to `ggml_backend_sched_alloc_graph` will use the information in
sched-galloc to initalize the tensors so that their buffer and data pointers
are set.

Now, in the case of the vision model what will happen is that using the same
sched:
```c++
    // initialize vision context
    if (model->has_vision) {
        switch (model->arch) {
            case LLM_ARCH_MLLAMA:
                {
                    ctx->ca_vision.model = &model->ca_vision;
                    ctx->ca_vision.sched = ctx->sched.get();
                    const size_t max_nodes = llama_model_max_nodes(*model);
                    ctx->ca_vision.buf_compute_meta.resize(ggml_tensor_overhead()*max_nodes + ggml_graph_overhead_custom(max_nodes, false));
                }
                break;
            default:
                    ctx->clip.model = &model->clip;
                    ctx->clip.sched = ctx->sched.get();
                    const size_t max_nodes = llama_model_max_nodes(*model);
                    ctx->clip.buf_compute_meta.resize(ggml_tensor_overhead()*max_nodes + ggml_graph_overhead_custom(max_nodes, false));
                break;
        }
    }
```
In examples/simple-vision-mllama/simple-vision.cpp we have:
```c++
    if (llama_encode_vision(ctx, img_batch) != 0) {
        LOG("%s: llama_encode_vision() failed\n", __func__);
        return 1;
    }
```
This will call the following function in llama.cpp:
```c++
int32_t llama_encode_vision(struct llama_context * ctx, llama_batch_img batch) {
    if (ctx->ca_vision.sched != NULL) {
        return ca_llama_encode_vision_internal(ctx->ca_vision, &batch);
    } else {
        return llama_encode_vision_internal(ctx->clip, &batch);
    }
}
```
And in llama-vision.cpp we have:
```c++
int32_t ca_llama_encode_vision_internal(ca_context & ctx, llama_batch_img * batch) {
    ...
        std::vector<float> output_single;
        int32_t status = encode_image_with_ca_vision(ctx, *batch->imgs[i], output_single);
        if (status != 0) {
            return status;
        }
        ...
}
```
```c++
static int32_t encode_image_with_ca_vision(ca_context & ctx,
        llama_img img, std::vector<float> & output) {
    ...
    static ggml_cgraph * gf = mllama_image_build_graph(&ctx, img_batch);
    ggml_backend_sched_reset(ctx.sched);
    bool ok = ggml_backend_sched_alloc_graph(ctx.sched, gf);
    if (!ok) {
        LLAMA_LOG_ERROR("failed to alloc memory for graph\n");
        return -1;
    }
    ...
```
Noow, notice that this is building the graph for the vision model, and then
using the same sched to allocate the graph.
```console
(gdb) p *graph
$34 = {size = 2048, n_nodes = 1394, n_leafs = 518, nodes = 0x55555a4ac1b0, grads = 0x0, grad_accs = 0x0, leafs = 0x55555a4b01b0,
  visited_hash_set = {size = 4099, used = 0x55555a4bc1c8, keys = 0x55555a4b41b0}, order = GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT}
```
So first the graph is split,  all the leafs and notes will be assigned backend
id (simplifed but I've gone over this in detail in another document). And then
`ggml_backend_sched_alloc_graph` will be called to allocate the graph.
In this case the backend id will have changed and this will cause a re-reservation
to occur. This will reset reset the tensor allocators.
```c++
    // reset allocators
    for (int i = 0; i < galloc->n_buffers; i++) {
        ggml_dyn_tallocr_reset(galloc->buf_tallocs[i]);
    }
```
```c++
    // allocate in hash table
    ggml_gallocr_alloc_graph_impl(galloc, graph, node_buffer_ids, leaf_buffer_ids);
```
This will create the node and leaf allocs which will then be used to assign
the tensors with the optimal backend buffers and their data pointers into these
bufffers using the information from the node and leaf allocs.

```c++
    // set the node_allocs from the hash table
    if (galloc->n_nodes < graph->n_nodes) {
        free(galloc->node_allocs);
        galloc->node_allocs = calloc(graph->n_nodes, sizeof(struct node_alloc));
        GGML_ASSERT(galloc->node_allocs != NULL);
    }
```
In this case the above block will be executed as the number of nodes in the
graph is greater than the number of nodes in the galloc:
```console
(gdb) p galloc->n_nodes
$39 = 1252
(gdb) p graph->n_nodes
$40 = 1904
```
This is because the language model has fewer nodes. So this will free the
node_allocs and then allocate new ones for the vision model with the new size
which is now 1904.
```c++
    galloc->n_nodes = graph->n_nodes;
```
```console
(gdb) p galloc->n_nodes
$41 = 1904
```
Next the nodes in the graph will be iterated over.
```c++
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
Back in `ggml_gallocr_reserve` we then call `ggml_gallocr_alloc_graph`:
```c++
        if (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            return false;
        }
```
This will return false in our case 
```c++
bool ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, struct ggml_cgraph * graph) {
    ...
    // reset buffers
    for (int i = 0; i < galloc->n_buffers; i++) {
        if (galloc->buffers[i] != NULL) {
            ggml_backend_buffer_reset(galloc->buffers[i]);
        }
    }

    // allocate the graph tensors from the previous assignments
    // leafs
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        struct leaf_alloc * leaf_alloc = &galloc->leaf_allocs[i];
        ggml_gallocr_init_tensor(galloc, leaf, &leaf_alloc->leaf);
    }
    // nodes
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct node_alloc * node_alloc = &galloc->node_allocs[i];
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            ggml_gallocr_init_tensor(galloc, src, &node_alloc->src[j]);
        }
        ggml_gallocr_init_tensor(galloc, node, &node_alloc->dst);
    }

    return true;
}
```
So this all looks reaonable to me (even though I don't think we should be
using the same sched for the language model and the vision model).

Now, if I remove on of the tensors that I added to be able to inspect tensors
that might be resued and go through this again.



