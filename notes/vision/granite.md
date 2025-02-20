## Granite Vision
This document contains notes related to the multimodal Granite model.

This model has both a text (language model) and a vision (image encoder)
component.

### Convert Granite Vision model to GGUF
First we need to clone the model:
```console
$ git clone https://huggingface.co/ibm-granite/granite-vision-3.1-2b-preview
```

Similar to llava 1.6 this model contains both the vision and language model and
we need to extract them so that we have one vision model and one language model
to run them using `llama-llava-cli`. This is what we use the surgery script for:
```console
(venv) $ python examples/llava/llava_surgery_v2.py -C -m /home/danbev/work/ai/models/granite-vision-3.1-2b-preview
...
Done! All vision tower tensors are removed from the model files and stored in llava.clip file.
...
Done!
Now you can convert /home/danbev/work/ai/models/granite-vision-3.1-2b-preview to a regular LLaMA GGUF file.
Also, use /home/danbev/work/ai/models/granite-vision-3.1-2b-preview/llava.projector to prepare a llava-encoder.gguf file.
```
So this will create a `llava.clip` which is the vision encoder model tensors
and a `llava.projector` file which contains the projection layers (which projects
the image patch embeddings to the language model token embeddings space).

```console
$ mkdir -p vit
$ cp ${GRANITE_PATH}/llava.clip vit/pytorch_model.bin
$ cp ${GRANITE_PATH}/llava.projector vit/
$ cp ${GRANITE_PATH}/config.json vit/
```
I had to manually add the fields `layer_norm_eps` and `hidden_act` to vision
config in vit/config.json:
```json
    "vision_config": {
        "hidden_size": 1152,
        "image_size": 384,
        "intermediate_size": 4304,
        "model_type": "siglip_vision_model",
        "num_attention_heads": 16,
        "num_hidden_layers": 27,
        "patch_size": 14,
        "layer_norm_eps": 1e-05,
        "hidden_act": "silu"
    }
```

So we now want to convert the project to a GGUF model that we can use with
`llama-llava-cli`:
```console
(venv) python ./examples/llava/convert_image_encoder_to_gguf.py -m vit --llava-projector vit/llava.projector --output-dir vit --clip-model-is-siglip

Done. Output file: vit/mmproj-model-f16.gguf
```

And the we need to convert the language model to gguf:
```console
(venv) python ./examples/convert_legacy_llama.py /home/danbev/work/ai/models/granite-vision-3.1-2b-preview --skip-unknown

INFO:convert:Loading model file /home/danbev/work/ai/models/granite-vision-3.1-2b-preview/model-00001-of-00002.safetensors
INFO:convert:Loading model file /home/danbev/work/ai/models/granite-vision-3.1-2b-preview/model-00001-of-00002.safetensors
INFO:convert:Loading model file /home/danbev/work/ai/models/granite-vision-3.1-2b-preview/model-00002-of-00002.safetensors
Traceback (most recent call last):
  File "/home/danbev/work/ai/llama.cpp-debug/./examples/convert_legacy_llama.py", line 1462, in <module>
    main()
  File "/home/danbev/work/ai/llama.cpp-debug/./examples/convert_legacy_llama.py", line 1371, in main
    model_plus = load_some_model(dir_model)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/danbev/work/ai/llama.cpp-debug/./examples/convert_legacy_llama.py", line 1219, in load_some_model
    model_plus = merge_multifile_models(models_plus)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/danbev/work/ai/llama.cpp-debug/./examples/convert_legacy_llama.py", line 521, in merge_multifile_models
    model = merge_sharded([mp.model for mp in models_plus])
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/danbev/work/ai/llama.cpp-debug/./examples/convert_legacy_llama.py", line 500, in merge_sharded
    return {name: convert(name) for name in names}
                  ^^^^^^^^^^^^^
  File "/home/danbev/work/ai/llama.cpp-debug/./examples/convert_legacy_llama.py", line 475, in convert
    lazy_tensors = [model[name] for model in models]
                    ~~~~~^^^^^^
KeyError: 'image_newline'
```
There is a note about this in the README.md and that if an occurs we can try
using the following script to convert the model:
```python
import os
import transformers
model_path = "/home/danbev/work/ai/models/granite-vision-3.1-2b-preview"
llm_export_path = "models/granite-vision-3.1-2b"

tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
model = transformers.AutoModelForImageTextToText.from_pretrained(model_path)

tokenizer.save_pretrained(llm_export_path)
model.language_model.save_pretrained(llm_export_path)
```
So lets try that:
```console
(venv) $ python convert-granite.py
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 13.20it/s]
```
So this will have produced a directory with the language model files:
```console
(venv) $ ls models/granite-vision-3.1-2b/
added_tokens.json       merges.txt                        model-00003-of-00003.safetensors  tokenizer_config.json
config.json             model-00001-of-00003.safetensors  model.safetensors.index.json      tokenizer.json
generation_config.json  model-00002-of-00003.safetensors  special_tokens_map.json           vocab.json
```
We can now use the `convert_hf_to_gguf.py` script to convert the language model
to gguf format:
```console
(venv) python ./convert_hf_to_gguf.py models/granite-vision-3.1-2b/ --outfile models/granite-vision-3.1-2b-f16.gguf --outtype f16
```
And optionally we can quantize the model to a lower precision.
```console
/build/bin/llama-quantize models/granite-vision-3.1-2b-f16.gguf models/granite-vision-3.1-2b-Q4_K.gguf Q4_K
```

And then we can run the model using `llama-llava-cli`:
```console
./llama-llava-cli -m models/granite-vision-3.1-2b-Q4_K.gguf --mmproj vit/mmproj-model-f16.gguf --image some-image.jpg -c 4096 -ngl 20
```

But there might be somethin wrong with the steps above I think. Looking at
the console ouput I can see:
```console
clip_model_load: model name:   vit
clip_model_load: description:  image encoder for LLaVA
clip_model_load: GGUF version: 3
clip_model_load: alignment:    32
clip_model_load: n_tensors:    451
clip_model_load: n_kv:         20
clip_model_load: ftype:        f16

clip_model_load: loaded meta data with 20 key-value pairs and 451 tensors from vit/mmproj-model-f16.gguf
clip_model_load: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
clip_model_load: - kv   0:                       general.architecture str              = clip
clip_model_load: - kv   1:                      clip.has_text_encoder bool             = false
clip_model_load: - kv   2:                    clip.has_vision_encoder bool             = true
clip_model_load: - kv   3:                   clip.has_llava_projector bool             = true
clip_model_load: - kv   4:                          general.file_type u32              = 1
clip_model_load: - kv   5:                               general.name str              = vit
clip_model_load: - kv   6:                        general.description str              = image encoder for LLaVA
clip_model_load: - kv   7:                        clip.projector_type str              = mlp
clip_model_load: - kv   8:                     clip.vision.image_size u32              = 384
clip_model_load: - kv   9:                     clip.vision.patch_size u32              = 14
clip_model_load: - kv  10:               clip.vision.embedding_length u32              = 1152
clip_model_load: - kv  11:            clip.vision.feed_forward_length u32              = 4304
clip_model_load: - kv  12:                 clip.vision.projection_dim u32              = 0
clip_model_load: - kv  13:           clip.vision.attention.head_count u32              = 16
clip_model_load: - kv  14:   clip.vision.attention.layer_norm_epsilon f32              = 0.000010
clip_model_load: - kv  15:                    clip.vision.block_count u32              = 27
clip_model_load: - kv  16:                  clip.vision.feature_layer arr[i32,4]       = [4, 8, 16, 27]
clip_model_load: - kv  17:                     clip.vision.image_mean arr[f32,3]       = [0.481455, 0.457828, 0.408211]
clip_model_load: - kv  18:                      clip.vision.image_std arr[f32,3]       = [0.268630, 0.261303, 0.275777]
clip_model_load: - kv  19:                              clip.use_gelu bool             = false
clip_model_load: - type  f32:  282 tensors
clip_model_load: - type  f16:  169 tensors
clip_model_load: CLIP using CPU backend
key clip.use_silu not found in file
clip_model_load: text_encoder:   0
clip_model_load: vision_encoder: 1
clip_model_load: llava_projector:  1
clip_model_load: minicpmv_projector:  0
clip_model_load: glm_projector:  0
clip_model_load: model size:     851.17 MB
clip_model_load: metadata size:  0.16 MB
clip_model_load: params backend buffer size =  851.17 MB (451 tensors)
key clip.vision.image_grid_pinpoints not found in file
key clip.vision.mm_patch_merge_type not found in file
key clip.vision.image_crop_resolution not found in file
clip_model_load: compute allocated memory: 59.76 MB
```
And `clip_image_batch_encode` in `clip.cpp` seems to take a very long time to
complete `ggml_backend_graph_compute(ctx->backend, gf)`.
'm not sure if there is something I messed up above or if it could be something
with configuration or options.
