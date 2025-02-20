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
I had to manually add the following fields to the vision config:
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
There is a note about this in the README.md and the an error occurs we can try
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
```console
(venv) $ python convert-granite.py
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 13.20it/s]
```
So this will have produces a director with the language model files:
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

```console
./llama-llava-cli -m models/granite-vision-3.1-2b-f16.gguf --mmproj vit/mmproj-model-f16.gguf --image some-image.jpg -c 4096
```
