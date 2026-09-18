## Qwen3 VL notes
This document contains notes about the Qwen3 VL model.

### n_deepstack_layers
So I came accross this in the Qwen3 VL model and wanted to understand what it
it about. In the [config.json](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/blob/main/config.json#L41):
we have:
```json
"vision_config": {
    "deepstack_visual_indexes": [
      5,
      11,
      17
    ],
```
This is used by the vision encoder, hence it is in the `vision_config` element.
This specifies a number of layers that are extracted and stored to later be used
in the target model, the text model.

The the gguf this will be a metadata field:
```console
     27: [BOOL]     |       24 | clip.vision.is_deepstack_layers = [False, False, False, False, False, True, ...]
 ```

Each ViT (Vision Transformer) depth taps a different granularity:
* ViT layer 5  → local texture, edges, fine-grained spatial details
* ViT layer 11 → mid-level patterns, object parts
* ViT layer 17 → higher-level but still pre-final features

The ideas here is to capture these different levels of visual features and then
make them available to the text model so that it can have more context, not just
the file projection which is usually the case.

These layers are extracted in tools/mtmd/models/qwen3vl.cpp:
```c++
    ggml_tensor * deepstack_features = nullptr;
    const int merge_factor = hparams.n_merge > 0 ? hparams.n_merge * hparams.n_merge : 4; // default 2x2=4 for qwen3vl
    ...
        if (layer.has_deepstack()) {
            ggml_tensor * feat = ggml_reshape_3d(ctx0, cur, n_embd * merge_factor, n_pos / merge_factor, batch_size);
            feat = build_norm(feat, layer.deepstack_norm_w, layer.deepstack_norm_b, norm_t, eps, il);
            feat = build_ffn(feat,
                layer.deepstack_fc1_w, layer.deepstack_fc1_b,
                nullptr, nullptr,
                layer.deepstack_fc2_w, layer.deepstack_fc2_b,
                ffn_op_type::FFN_GELU, il);

            if(!deepstack_features) {
                deepstack_features = feat;
            } else {
                // concat along the feature dimension
                deepstack_features = ggml_concat(ctx0, deepstack_features, feat, 0);
            }
        }
```
This might look like it is doing a lot more than just saving the layers but recall
that the final layer is also projected into the main models vector space and
this need to happen for these layers output as well. And they have weights that
have been trained to do this for these specific layers.

And this information is injected into the text model as early possible so that
all/most of the text model can see and reason about these features/information.
This happens in qwen3vl.cpp and:
```c++
        if (il < (int) n_deepstack_layers) {
            ggml_tensor * ds = ggml_view_2d(ctx0, res->t_inp_embd, n_embd, n_tokens, res->t_inp_embd->nb[1], (il + 1) * n_embd * sizeof(float));
            cur = ggml_add(ctx0, cur, ds);
            cb(cur, "deepstack_out", il);
        }
```
Notice that this is injecting the embeddings from the vision encoder into the
first 3 layers (using the example of 3 deepstack layers from above).
