### CUDA backend notes
This document contain notes on the CUDA backend implementation in ggml.

### CUDA compute
This section is going to take a look at CUDA computation.
```console
(gdb) bt
#0  ggml_cuda_compute_forward (ctx=..., dst=0x555556a29fe0)
    at /home/danbev/work/ai/llama.cpp-debug/ggml/src/ggml-cuda/ggml-cuda.cu:2701
#1  0x00007fffec44297f in evaluate_and_capture_cuda_graph (cuda_ctx=0x5555569181d0, cgraph=0x555556adf168,
    graph_evaluated_or_captured=@0x7fffffff88e8: false, use_cuda_graph=@0x7fffffff88da: false,
    cuda_graph_update_required=@0x7fffffff88db: true) at /home/danbev/work/ai/llama.cpp-debug/ggml/src/ggml-cuda/ggml-cuda.cu:3641

#2  0x00007fffec44323c in ggml_backend_cuda_graph_compute (backend=0x555556886180, cgraph=0x555556adf168)
    at /home/danbev/work/ai/llama.cpp-debug/ggml/src/ggml-cuda/ggml-cuda.cu:3759

    ------------------ CUDA layers starts here ------------------

#3  0x00007ffff7eef049 in ggml_backend_graph_compute_async (backend=0x555556886180, cgraph=0x555556adf168)
    at /home/danbev/work/ai/llama.cpp-debug/ggml/src/ggml-backend.cpp:359

#4  0x00007ffff7ef4168 in ggml_backend_sched_compute_splits (sched=0x555556ad7870)
    at /home/danbev/work/ai/llama.cpp-debug/ggml/src/ggml-backend.cpp:1575

#5  0x00007ffff7ef4ff6 in ggml_backend_sched_graph_compute_async (sched=0x555556ad7870, graph=0x555556a0b4f0)
    at /home/danbev/work/ai/llama.cpp-debug/ggml/src/ggml-backend.cpp:1784

#6  0x00007ffff7a5a86a in llama_context::graph_compute (this=0x555556b05840, gf=0x555556a0b4f0, batched=true)
    at /home/danbev/work/ai/llama.cpp-debug/src/llama-context.cpp:1976

#7  0x00007ffff7a55e54 in llama_context::process_ubatch (this=0x555556b05840, ubatch=..., gtype=LLM_GRAPH_TYPE_DECODER,
    mctx=0x555556ae0680, ret=@0x7fffffffcb04: 65535) at /home/danbev/work/ai/llama.cpp-debug/src/llama-context.cpp:1033

#8  0x00007ffff7a57d4e in llama_context::decode (this=0x555556b05840, batch_inp=...)
    at /home/danbev/work/ai/llama.cpp-debug/src/llama-context.cpp:1438
```
So lets start at the first function call of the CUDA backend:
```c++
static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    ggml_cuda_set_device(cuda_ctx->device);

#ifdef USE_CUDA_GRAPH
    static const bool disable_cuda_graphs_due_to_env = (getenv("GGML_CUDA_DISABLE_GRAPHS") != nullptr);

    // Objects required for CUDA Graph
    if (cuda_ctx->cuda_graph == nullptr) {
        cuda_ctx->cuda_graph.reset(new ggml_cuda_graph());
    }
```
```console
(gdb) p *cuda_ctx->cuda_graph
$4 = {graph = 0x0, instance = 0x0, num_nodes = 0, nodes = std::vector of length 0, capacity 0,
  params = std::vector of length 0, capacity 0, disable_due_to_gpu_arch = false, disable_due_to_too_many_updates = false,
  disable_due_to_failed_graph_capture = false, number_consecutive_updates = 0,
  ggml_graph_properties = std::vector of length 0, capacity 0}
```
```c++

    bool use_cuda_graph = true;
    bool cuda_graph_update_required = false;

    // Notice that this is checking the graph field of cuda_ctx->cuda_graph
    if (cuda_ctx->cuda_graph->graph == nullptr) {
        // ggml_cuda_info() will initialize the device info if not already done (only done once)
        // CUDA graphs are supported on Ampere and above
        if (ggml_cuda_info().devices[cuda_ctx->device].cc < GGML_CUDA_CC_AMPERE) {
            cuda_ctx->cuda_graph->disable_due_to_gpu_arch = true;
            GGML_LOG_DEBUG("%s: disabling CUDA graphs due to GPU architecture\n", __func__);
        }
    }
```
```c++
    // Disable CUDA graphs in presence of env var, old GPU, use-case which is changing too rapidly,
    // or previous graph capture failure.
    // Also disable for multi-gpu for now. TO DO investigate
    if (disable_cuda_graphs_due_to_env
        || cuda_ctx->cuda_graph->disable_due_to_gpu_arch
        || cuda_ctx->cuda_graph->disable_due_to_too_many_updates
        || cuda_ctx->cuda_graph->disable_due_to_failed_graph_capture) {
        use_cuda_graph = false;
    }
```

```c++
    if (use_cuda_graph) {
        cuda_graph_update_required = is_cuda_graph_update_required(cuda_ctx, cgraph);
```
So this is one question I had. When do cuda graphs need to be updated.
First just note that a cuda_ctx->cuda_graph has a ggml_graph_properties member
which is a vector of ggml_graph_node_properties structs. So a single
ggml_cuda_graph can have many ggml_graph_properties, so one for each node/tensor
in the graph.
```console
(gdb) ptype cuda_ctx->cuda_graph->ggml_graph_properties[0]
type = struct ggml_graph_node_properties {
    void *node_address;   // data field of the ggml tensor/node
    ggml_op node_op;
    int64_t ne[4];
    size_t nb[4];
    void *src_address[10];
    int32_t op_params[16];
}
```

```c++
static bool is_cuda_graph_update_required(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph) {

    bool cuda_graph_update_required = false;

    // First usage of CUDA graph
    if (cuda_ctx->cuda_graph->instance == nullptr) {
        cuda_graph_update_required = true;
    }

    // Check if the graph size has changed. Notice that this is using the properties
    // that we discussed above to match the size of the nodes in the ggml compute graph
    // with the number of nodes that the graph had when it was last captured.
    if (cuda_ctx->cuda_graph->ggml_graph_properties.size() != (size_t)cgraph->n_nodes) {
        cuda_graph_update_required = true;
        cuda_ctx->cuda_graph->ggml_graph_properties.resize(cgraph->n_nodes);
    }

    // Loop over nodes in GGML graph to determine if CUDA graph update is required
    // and store properties to allow this comparison for the next token
    for (int i = 0; i < cgraph->n_nodes; i++) {
        bool has_matching_properties = true;
        if (!cuda_graph_update_required) {
            has_matching_properties = ggml_graph_node_has_matching_properties(cgraph->nodes[i], &cuda_ctx->cuda_graph->ggml_graph_properties[i]);
        }
        if (!has_matching_properties) {
            cuda_graph_update_required = true;
        }
        // Notice that this updates the properties regardless of whether an
        // update is required or not
        set_ggml_graph_node_properties(cgraph->nodes[i], &cuda_ctx->cuda_graph->ggml_graph_properties[i]);
    }

    return cuda_graph_update_required;
}
```
* First time (instance == nullptr)
* GGML Compute Graph size changed (different number of nodes)
* Any tensor/node property changed:
    * Tensor dimensions (ne[])
    * Tensor strides (nb[])
    * Memory addresses (node->data, src addresses)
    * Operation type
    * Operation parameters

Next we have a check that the graph is compatible with CUDA graphs.
```c++
        use_cuda_graph = check_node_graph_compatibility(cgraph, use_cuda_graph);
```
```c++
#ifdef USE_CUDA_GRAPH
static bool check_node_graph_compatibility(ggml_cgraph * cgraph, bool use_cuda_graph) {

    // Loop over nodes in GGML graph to obtain info needed for CUDA graph

    const std::string gemma3n_per_layer_proj_src0_name = "inp_per_layer_selected";
    const std::string gemma3n_per_layer_proj_src1_name = "per_layer_proj";
    const std::string ffn_moe_gate_bias_prefix = "ffn_moe_gate_biased";
    const std::string ffn_moe_up_bias_prefix = "ffn_moe_up_biased";
    const std::string ffn_moe_down_bias_prefix = "ffn_moe_down_biased";
    const std::string nemotron_h_block_out_prefix = "nemotron_h_block_out";
    const std::string mamba2_y_add_d_prefix = "mamba2_y_add_d";

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if (ggml_is_empty(node) ||
            node->op == GGML_OP_RESHAPE ||
            node->op == GGML_OP_TRANSPOSE ||
            node->op == GGML_OP_VIEW ||
            node->op == GGML_OP_PERMUTE ||
            node->op == GGML_OP_NONE) {
            continue;
        }

        // Tensors that are split accorss multiple GPUs are not supported by CUDA graphs.
        // CUDA graphs capture operations on a single stream/device only.
        if (node->src[0] && node->src[0]->buffer && ggml_backend_buft_is_cuda_split(node->src[0]->buffer->buft)) {
            use_cuda_graph = false; // Split buffers are not supported by CUDA graph capture
            GGML_LOG_DEBUG("%s: disabling CUDA graphs due to split buffer\n", __func__);
        }

        // MUL_MAT_ID is for MoE models expert selection.
        if (node->op == GGML_OP_MUL_MAT_ID && node->ne[2] != 1) {
            use_cuda_graph = false; // This node type is not supported by CUDA graph capture
            GGML_LOG_DEBUG("%s: disabling CUDA graphs due to unsupported node type\n", __func__);
        }
```
There are a few conditions and a few explicitely named tensors that are exceptions
and have been proved to work:
```c++
        if (node->op == GGML_OP_ADD &&
            node->src[1] && node->src[1]->ne[1] > 1 &&

            (node->src[0] ? node->src[0]->name != gemma3n_per_layer_proj_src0_name : true) &&
            (node->src[1] ? node->src[1]->name != gemma3n_per_layer_proj_src1_name : true) &&

            strncmp(node->name, ffn_moe_gate_bias_prefix.c_str(),   ffn_moe_gate_bias_prefix.size())    != 0 &&
            strncmp(node->name, ffn_moe_up_bias_prefix.c_str(),     ffn_moe_up_bias_prefix.size())      != 0 &&
            strncmp(node->name, ffn_moe_down_bias_prefix.c_str(),   ffn_moe_down_bias_prefix.size())    != 0 &&
            strncmp(node->name, nemotron_h_block_out_prefix.c_str(),nemotron_h_block_out_prefix.size()) != 0 &&
            strncmp(node->name, mamba2_y_add_d_prefix.c_str(),      mamba2_y_add_d_prefix.size())       != 0) {

            // disable CUDA graphs for batch size > 1 for now while excluding the matrix-matrix addition
            // as part of Gemma3n's `project_per_layer_input` operation by means of matching node names. See
            // https://github.com/ggml-org/llama.cpp/blob/f9a31eea06a859e34cecb88b4d020c7f03d86cc4/src/llama-model.cpp#L10199-L10241 and
            // https://github.com/huggingface/transformers/blob/bda75b4011239d065de84aa3e744b67ebfa7b245/src/transformers/models/gemma3n/modeling_gemma3n.py#L1773,
            // Generally, changes in batch size or context size can cause changes to the grid size of some kernels.
            use_cuda_graph = false;
            GGML_LOG_DEBUG("%s: disabling CUDA graphs due to batch size > 1 [%s] [%ld %ld %ld %ld]\n", __func__, node->name, node->ne[0], node->ne[1], node->ne[2], node->ne[3]);
        }

        if (!use_cuda_graph) {
            break;
        }
    }

    return use_cuda_graph;
}
```
So this will return us back to ggml_backend_cuda_graph_compute:
```c++
        // Disable CUDA graphs (from the next token) if the use-case is demanding too many consecutive graph updates.
        if (use_cuda_graph && cuda_graph_update_required) {
            cuda_ctx->cuda_graph->number_consecutive_updates++;
        } else {
            cuda_ctx->cuda_graph->number_consecutive_updates = 0;
        }

        if (cuda_ctx->cuda_graph->number_consecutive_updates >= 4) {
            cuda_ctx->cuda_graph->disable_due_to_too_many_updates = true;
            GGML_LOG_DEBUG("%s: disabling CUDA graphs due to too many consecutive updates\n", __func__);
        }
    }
```
Now, CUDA graphs are only of use when the graph is stable andn can be resued
many time. If the graph keeps changing then it might "cost" more that it saves.
There is a cost of capturing the graph (first execution), recapturing or updating
and also some memory overhead for the graph structure. The above is an adaptive
strategy to disable CUDA graphs if there are too many consecutive updates.

After that we start the CUDA graph capture if needed:
```c++
    if (use_cuda_graph && cuda_graph_update_required) {
        // Start CUDA graph capture
        {
            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            ggml_cuda_lock_counter.fetch_add(1, std::memory_order_relaxed);
        }

        CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));
    }
```
Finally we have call to evaluate_and_capture_cuda_graph():
```c++
    bool graph_evaluated_or_captured = false;

    evaluate_and_capture_cuda_graph(cuda_ctx, cgraph, graph_evaluated_or_captured, use_cuda_graph, cuda_graph_update_required);

    return GGML_STATUS_SUCCESS;
}
```

_wip_
