### Devstral 2 Model conversion
This model can be found in this Hugging Face model repository:  
https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512


### Downloading the model files
```console
(venv) $ cd ~/models
(venv) $ export HF_TOKEN=<token>
(venv) $ hf download mistralai/Devstral-Small-2-24B-Instruct-2512 --local-dir Devstral-Small-2-24B-Instruct-2512
```

### Python dependencies
```console
(venv) $ pip install sentence-transformers==5.2.0
(venv) $ pip install transformers==5.0.0rc1
(venv) $ pip install triton
(venv) $ pip install accelerate
(venv) $ pip install mistral-common
```

### Run the original model
Lets run the original model to get some output logits to compare visually against
later.
```console
(venv) $ cd examples/model-conversion
(venv) $ CUDA_VISIBLE_DEVICES="" make causal-run-original-model

Logits shape: torch.Size([1, 6, 131072])
Last token logits shape: (131072,)
Vocab size: 131072
First 10 logits: [-6.03125 -6.03125   0.25195312 -6.03125 -6.03125 -6.03125 -6.03125  -6.03125 -6.03125 -3.34375]
Last 10 logits:  [-5.03125 -3.140625 -3.75       -6.59375 -4.875   -5.5625  -3.671875 -4.625   -5.96875 -5.6875 ]
Top 5 predictions:
  Token 1010 ('\n'): 6.687500
  Token 19607 (' Brian'): 5.968750
  Token 7044 (' Alex'): 5.875000
  Token 6959 (' David'): 5.562500
  Token 15051 (' Chris'): 5.562500
Saved bin logits to: data/pytorch-Devstral-Small-2-24B-Instruct-2512.bin
Saved txt logist to: data/pytorch-Devstral-Small-2-24B-Instruct-2512.txt
```

### Model Conversion
```console
(venv) $ export MODEL_PATH=~/models/Devstral-Small-2-24B-Instruct-2512
(venv) $ python convert_hf_to_gguf.py --mistral-format --outfile models/ --outtype=bf16 ${MODEL_PATH}
(venv) $ python convert_hf_to_gguf.py --mistral-format --outfile models/ --mmproj ${MODEL_PATH}
```

```console
(venv) $ export CONVERTED_MODEL=/home/dbevenius/work/llama.cpp-staging/models/Devstral-Small-2-24B-Instruct-2512.gguf
```

### Run the converted model
```console
(venv) $ make causal-run-converted_model
...
Input prompt: "Hello, my name is"
Tokenized prompt (6 tokens): <s> (1)Hello (22177), (1044) my (2036) name (2564) is (1395)
Vocab size: 131072
Saving data to data/llamacpp-Devstral-Small-2-24B-Instruct-2512.bin
First 10 logits: -6.039290 -6.039550 0.213944 -6.042861 -6.039347 -6.043080 -6.039300 -6.039615 -6.039384 -3.323704
Last 10 logits:  -4.996169 -3.164495 -3.793222 -6.617929 -4.842769 -5.592521 -3.666847 -4.626977 -5.998837 -5.684978
```

### Model verification
```console
(venv) $ env CUDA_VISIBLE_DEVICES="" make causal-run-original-model 
(venv) $ CUDA_VISIBLE_DEVICES="" make causal-run-original-model 
Loading model and tokenizer using AutoTokenizer: /home/dbevenius/models/Devstral-Small-2-24B-Instruct-2512
Model type:        mistral3
Vocab size:        131072
Hidden size:       5120
Number of layers:  40
BOS token id:      1
EOS token id:      2
Using FP8 quantized models requires a GPU or XPU, we will default to dequantizing the model to bf16 since no GPU or XPU is available
Loading weights: 100%|█| 585/585 [00:09<00:00, 63.28it/s, Materializing param=model.vision_tower.transformer.layers.23.ffn_norm.weight
Model class: Mistral3ForConditionalGeneration
Input tokens: tensor([[    1, 22177,  1044,  2036,  2564,  1395]])
Input text: 'Hello, my name is'
Tokenized: ['<s>', 'Hello', ',', 'Ġmy', 'Ġname', 'Ġis']
Logits shape: torch.Size([1, 6, 131072])
Last token logits shape: (131072,)
Vocab size: 131072
First 10 logits: [-6.03125    -6.03125     0.25195312 -6.03125    -6.03125    -6.03125
 -6.03125    -6.03125    -6.03125    -3.34375   ]
Last 10 logits: [-5.03125  -3.140625 -3.75     -6.59375  -4.875    -5.5625   -3.671875
 -4.625    -5.96875  -5.6875  ]
Top 5 predictions:
  Token 1010 ('\n'): 6.687500
  Token 19607 (' Brian'): 5.968750
  Token 7044 (' Alex'): 5.875000
  Token 6959 (' David'): 5.562500
  Token 15051 (' Chris'): 5.562500
Saved bin logits to: data/pytorch-Devstral-Small-2-24B-Instruct-2512.bin
Saved txt logist to: data/pytorch-Devstral-Small-2-24B-Instruct-2512.txt
(venv) $ CUDA_VISIBLE_DEVICES="" make causal-verify-logits 
Loading model and tokenizer using AutoTokenizer: /home/dbevenius/models/Devstral-Small-2-24B-Instruct-2512
Model type:        mistral3
Vocab size:        131072
Hidden size:       5120
Number of layers:  40
BOS token id:      1
EOS token id:      2
Using FP8 quantized models requires a GPU or XPU, we will default to dequantizing the model to bf16 since no GPU or XPU is available
Loading weights: 100%|█| 585/585 [00:09<00:00, 64.16it/s, Materializing param=model.vision_tower.transformer.layers.23.ffn_norm.weight
Model class: Mistral3ForConditionalGeneration
Input tokens: tensor([[    1, 22177,  1044,  2036,  2564,  1395]])
Input text: 'Hello, my name is'
Tokenized: ['<s>', 'Hello', ',', 'Ġmy', 'Ġname', 'Ġis']
Logits shape: torch.Size([1, 6, 131072])
Last token logits shape: (131072,)
Vocab size: 131072
First 10 logits: [-6.03125    -6.03125     0.25195312 -6.03125    -6.03125    -6.03125
 -6.03125    -6.03125    -6.03125    -3.34375   ]
Last 10 logits: [-5.03125  -3.140625 -3.75     -6.59375  -4.875    -5.5625   -3.671875
 -4.625    -5.96875  -5.6875  ]
Top 5 predictions:
  Token 1010 ('\n'): 6.687500
  Token 19607 (' Brian'): 5.968750
  Token 7044 (' Alex'): 5.875000
  Token 6959 (' David'): 5.562500
  Token 15051 (' Chris'): 5.562500
Saved bin logits to: data/pytorch-Devstral-Small-2-24B-Instruct-2512.bin
Saved txt logist to: data/pytorch-Devstral-Small-2-24B-Instruct-2512.txt


Input prompt: "Hello, my name is"
Tokenized prompt (6 tokens): <s> (1)Hello (22177), (1044) my (2036) name (2564) is (1395)
Vocab size: 131072
Saving data to data/llamacpp-Devstral-Small-2-24B-Instruct-2512.bin
First 10 logits: -6.039290 -6.039550  0.213944 -6.042861 -6.039347 -6.043080 -6.039300 -6.039615 -6.039384 -3.323704 
Last 10 logits:  -4.996169 -3.164495 -3.793222 -6.617929 -4.842769 -5.592521 -3.666847 -4.626977 -5.998837 -5.684978 

Data saved to data/llamacpp-Devstral-Small-2-24B-Instruct-2512.bin
Data saved to data/llamacpp-Devstral-Small-2-24B-Instruct-2512.txt
Using converted model: Devstral-Small-2-24B-Instruct-2512
Checked all required files were found. Proceeding...

🔍 GGML Model Validation for model  Devstral-Small-2-24B-Instruct-2512
========================================
PyTorch logits  : data/pytorch-Devstral-Small-2-24B-Instruct-2512.bin
llama.cpp logits: data/llamacpp-Devstral-Small-2-24B-Instruct-2512.bin

Top 10 PyTorch logits:   [6.6875    5.96875   5.875     5.5625    5.5625    5.53125   5.375    5.3125   5.3125    5.3125  ]
Top 10 llama.cpp logits: [6.7031956 6.032293  5.8454757 5.5685377 5.5511737 5.5213685 5.382902 5.316436 5.2942443 5.281416]

Max absolute difference: 0.1408
✅ OK: Lightweight model check successful!
       Ok to proceed with NMSE check...
Model name: Devstral-Small-2-24B-Instruct-2512
PyTorch logits file: data/pytorch-Devstral-Small-2-24B-Instruct-2512.bin
llama.cpp logits file: data/llamacpp-Devstral-Small-2-24B-Instruct-2512.bin
📊 NMSE Check for Model Comparison
==================================================
Reference (ground truth): data/pytorch-Devstral-Small-2-24B-Instruct-2512.bin
Test (to evaluate):       data/llamacpp-Devstral-Small-2-24B-Instruct-2512.bin

Loading reference logits...
  Shape: (131072,), Type: float32
Loading test logits...
  Shape: (131072,), Type: float32

✅ Shapes match: (131072,)

📈 METRICS
==============================
MSE (Mean Squared Error):     8.512747e-04
Reference Variance:           2.652392e+00
NMSE:                         3.209460e-04
Max Absolute Error:           0.140754
Mean Absolute Error:          0.023267
NMSE (dB):                    -34.94 dB

🎯 INTERPRETATION
==============================
👍 Very good match

📋 GUIDANCE
==============================
✅ EXCELLENT: Your GGML conversion is working very well!
   The differences are negligible for practical use.

📚 NMSE BENCHMARKS
==============================
< 1e-6:  Essentially identical
< 1e-4:  Excellent (typical for good conversions)
< 1e-3:  Very good
< 1e-2:  Good (acceptable for most use cases)
< 0.1:   Acceptable (may need verification)
> 1.0:   Poor (worse than random)

✅ RESULT: PASS (NMSE = 3.21e-04)
```
