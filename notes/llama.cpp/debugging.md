## Debugging/troubleshooting
Debugging and troubleshooting tips for llama.cpp and ggml.

### Graph execution
It can sometimes be misleading to name tensors and then after the compution try
to see what the values were for each tensor in the graph. This is because the
graph scheduler can sometimes optimize the graph and resuse some of the tensors
for different operations. What an be done instead is create a copy of the
tensor we are interested in and then inspect it. For example:
```c++
        struct ggml_tensor * mul_copy = ggml_mul(ctx0, cur, layer.norm_in_w);
        ggml_format_name(mul, "layernorm_mul_copy-%ld", il);
        ggml_build_forward_expand(gf, mul_copy);
```
And then in some other part of the code we can inspect the tensor:
```c++
    {
        struct ggml_tensor* next_out = ggml_graph_get_tensor(gf, "layernorm_mul_copy-0");
        float next_buffer[10];
        ggml_backend_t next_bk = ggml_backend_sched_get_tensor_backend(ctx.sched, next_out);
        printf("Backend type: %s\n", ggml_backend_name(next_bk));
        printf("Tensor type: %s\n", ggml_type_name(next_out->type));
        ggml_backend_tensor_get_async(next_bk, next_out, next_buffer, 0, sizeof(next_buffer));
        ggml_backend_sched_synchronize(ctx.sched);
        for (int i = 0; i < 10; i++) {
            printf("layernorm_mul_copy-0[%d] = %f (isnan=%d)\n", i, next_buffer[i], std::isnan(next_buffer[i]));
        }
    }
```


### Storing and comparing tenors
I've run into situations where I print out say the first 10 value of a tensor
but it can be tricky sometime that the actual type of the data are getting
printed correctly. Instead we can save the tensor to a file and then compare
it with a known good tensor. The use case I had was I was trying to compare
a model that worked in ollama but did not work in llama.cpp:
```c++
void save_tensor_binary(const char* filename, struct ggml_tensor* tensor, ggml_backend_sched_t sched) {
    size_t nbytes = ggml_nbytes(tensor);
    std::vector<uint8_t> buf(nbytes);

    // Get backend for tensor
    ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);

    // Get raw tensor data
    ggml_backend_tensor_get_async(backend, tensor, buf.data(), 0, nbytes);
    ggml_backend_sched_synchronize(sched);  // Make sure async operation completes

    // Write to binary file
    std::ofstream outfile(filename, std::ios::out | std::ios::binary);
    if (!outfile) {
        fprintf(stderr, "Failed to open file %s for writing\n", filename);
        return;
    }

    outfile.write(reinterpret_cast<const char*>(buf.data()), nbytes);
    outfile.close();
}
```
The we can store tensors using:
```c++
save_tensor_binary("llama_inp_raw.bin", tensor, ctx.sched); 
```

And with two files we can use a tool like cmp:
```console

```

### Printing tensor weights
When printing tensors it is important to do this after the backends have been
scheduled/reserved. This can be done by placing the following code in 
llama-context.cpp, in process_ubatch after the graphs has been created and
reserved:
```c++
    {
        struct ggml_tensor * tensor = model.layers[0].ssm_a;
        if (tensor != nullptr) {
            float buffer[10];
            ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched.get(), tensor);
            printf("Backend type: %s\n", ggml_backend_name(backend));
            printf("Tensor type: %s\n", ggml_type_name(tensor->type));
            ggml_backend_tensor_get_async(backend, tensor, buffer, 0, sizeof(buffer));
            ggml_backend_sched_synchronize(sched.get());
            for (int i = 0; i < 10; i++) {
                printf("%s[%d] = %f\n", tensor->name, i, buffer[i]);
            }
        }
    }
```
This can be placed after set_inputs if you want to see the input tensors.

### Executing one layer at a time
This can be useful when we want to isolate a layer and inspect and compare the
logits it output with an original model.

```c++
    ggml_build_forward_expand(gf, inp->get_attn()->get_k_idxs());
    ggml_build_forward_expand(gf, inp->get_attn()->get_v_idxs());
    ggml_build_forward_expand(gf, inp->get_attn()->get_kq_mask());

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    //for (int il = 0; il < n_layer; ++il) {
    for (int il = 0; il < 1; ++il) {
        struct ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        if (hparams.is_recurrent(il)) {
            // ssm layer //
            cur = build_mamba2_layer(inp->get_recr(), cur, model, ubatch, il);
        } else if (hparams.n_ff(il) == 0) {
            // attention layer //
            cur = build_attention_layer(cur, inp->get_attn(), model, n_embd_head, il);
        } else {
            cur = build_ffn_layer(cur, model, il);
        }

        //if (il == n_layer - 1 && inp_out_ids) {
        if (il == 0) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // add residual
        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "h_block_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
```
Adding this here as it can be helpful for debugging.
