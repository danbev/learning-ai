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
(venv) spark $ python src/quantize_modelopt.py --qformat nvfp4_fp8_mixed
...
```
I'm using the `nvfp4_fp8_mixed` format, so that hf_quant_config.json will contain
a `quantization_layers` dict as if the quant formats are not mixed there will be
one such dict in the config.
