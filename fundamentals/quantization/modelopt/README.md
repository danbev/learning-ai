### NVIDIA Model Optimizer (modelopt)
This is a example that explores how quantizing a model using the NVIDIA Model
Optimizer works and the configuration files it generates. This model can then
be used to convert it to ggml, allowing me to step through a real small example
and understand the conversion process better.

So ModelOpt is an optimization framework/toolkit, just like gptqmodel,
llm-compressor etc, that can convert to various formats like NVFP4, FP8, INT4 and
other formats.

Model repos: https://github.com/NVIDIA/Model-Optimizer

### Download orginal model
```console
(env) $ hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir models/Qwen2.5-0.5B-Instruct
```

### Install
```console
(venv) pip install "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu130
(venv) printf "torch==2.14.1\ntriton==3.8.0\n" > constraints.txt
(venv) pip install "nvidia-modelopt[hf]" datasets -c constraints.txt
(venv) pip install -e ~/work/llama.cpp/gguf-py

```

### Quantize a model
```console
(venv) spark $ python src/quantize_modelopt.py --qformat nvfp4_mixed
...
```
I'm using the `nvfp4_fp8_mixed` format, so that hf_quant_config.json will contain
a `quantization_layers` dict as if the quant formats are not mixed there will be
one such dict in the config.

### Processing
All nn.Linear with QuantLinear which adds 3 TensorQuantizers input_quantizer
which is applied to x before matmul, weight_quantizer which is applied to W, and
output_quantizer which is applied to the result.

The quantization configuraiton is printed in the script, and this works as rules
I think, so first we have more general rules. 
```console
{
  "quant_cfg": [
    {
      "quantizer_name": "*",     // all are initially disabled
      "enable": false
    },
    {
      "quantizer_name": "*weight_quantizer",
      "cfg": {
        "num_bits": [
          2,                      // exponents bits  (E2M1)
          1                       // mantissa bits   (E2M1)
        ],
        "effective_bits": 4.5,
        "block_sizes": {
          "-1": 16,
          "type": "dynamic",
          "scale_bits": [
            4,
            3
          ]
        }
      }
    },
    {
      "quantizer_name": "*input_quantizer",
      "cfg": {
        "num_bits": [
          2,
          1
        ],
        "effective_bits": 4.5,
        "block_sizes": {
          "-1": 16,
          "type": "dynamic",
          "scale_bits": [
            4,
            3
          ]
        }
      }
    },
    ...
```
So a layer in the model would have had TensorQuantizers added to it, for example
```console
model.layers.0.self_attn_q_proj.weight_quantizer
model.layers.0.self_attn_q_proj.input_quantizer
model.layers.0.mlp.down_proj.input_quantizer
lm_head.weight_quantizer
```
So all *weight_quantizers and *input_quantizers will get the NVFP format.
