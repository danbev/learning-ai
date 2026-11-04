### NVIDIA Model Optimizer (modelopt)
This is a example that explores how quantizing a model using the NVIDIA Model
Optimizer works and the configuration files it generates. This model can then
be used to convert it to ggml, allowing me to step through a real small example
and understand the conversion process better.

So ModelOpt is an optimization framework/toolkit, just like gptqmodel,
llm-compressor etc, that can convert to various formats like NVFP4, FP8, INT4 and
other formats.

Model repos: https://github.com/NVIDIA/Model-Optimizer


### Install
```console
(venv) pip install "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu130
(venv) printf "torch==2.14.1\ntriton==3.8.0\n" > constraints.txt
(venv) pip install "nvidia-modelopt[hf]" datasets -c constraints.txt
(venv) pip install -e ~/work/llama.cpp/gguf-py

```

### Quantize a model
```console
(venv) spark $ python src/quantize_modelopt.py
...
Writing model shards: 100%|████████████████████████████████████████████████| 1/1 [00:03<00:00,  3.87s/it]
Copied 0 source checkpoint file(s) to models/Qwen2.5-0.5B-Instruct-nvfp4: []

quantization_config:
{
  "config_groups": {
    "group_0": {
      "input_activations": {
        "dynamic": false,
        "num_bits": 4,
        "type": "float",
        "group_size": 16
      },
      "weights": {
        "dynamic": false,
        "num_bits": 4,
        "type": "float",
        "group_size": 16
      },
      "targets": [
        "Linear"
      ]
    }
  },
  "ignore": [
    "lm_head",
    "model.embed_tokens"
  ],
  "quant_algo": "NVFP4",
  "group_size": 16,
  "producer": {
    "name": "modelopt",
    "version": "0.48.0"
  },
  "quant_method": "modelopt"
}

layer 0 down_proj tensors:
  model.layers.0.mlp.down_proj.input_scale           F32      []
  model.layers.0.mlp.down_proj.weight                U8       [896, 2432]
  model.layers.0.mlp.down_proj.weight_scale          F8_E4M3  [896, 304]
  model.layers.0.mlp.down_proj.weight_scale_2        F32      []
```
