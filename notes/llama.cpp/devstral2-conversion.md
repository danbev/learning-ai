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

### Devstral 2 123B Model conversion
```console
(venv) $ hf download mistralai/Devstral-2-123B-Instruct-2512 --local-dir Devstral-2-123B-Instruct-2512
```

#### Model Conversion
```console
#!/usr/bin/env bash
set -e

ulimit -n 65536

# Required versions of Python packages:
# pip install sentence-transformers==5.2.0
# pip install transformers==5.0.0rc1
# pip install mistral-common
# pip install triton
# pip install accelerate

### Path to the original model
export MODEL_PATH=${MODEL_PATH:-~/models/Devstral-2-123B-Instruct-2512}
printf "Original model path: %s\n" "${MODEL_PATH}"

### Converted model output directory
models_dir=$PWD/models
printf "Converted model directory: %s\n" "${models_dir}"

source venv/bin/activate

export CONVERTED_MODEL=/home/dbevenius/work/llama.cpp-staging/models/Devstral-2-123B-Instruct-2512.gguf

### Convert the original model to GGUF format
python convert_hf_to_gguf.py --verbose --mistral-format --outfile ${CONVERTED_MODEL} --outtype=bf16 ${MODEL_PATH}

pushd examples/model-conversion

### Verify the converted model
#make causal-verify-logits

### Quantize the converted model
make causal-quantize-Q8_0

deactivate

popd
```

#### Verify logits
```console
(venv) $ export CONVERTED_MODEL=/home/dbevenius/work/llama.cpp-staging/models/Devstral-2-123B-Instruct-2512.gguf
(venv) $ make causal-verify-logits
Loading model and tokenizer using AutoTokenizer: /home/dbevenius/models/Devstral-2-123B-Instruct-2512
Model type:        ministral3
Vocab size:        131072
Hidden size:       12288
Number of layers:  88
BOS token id:      1
EOS token id:      2
Using FP8 quantized models requires a GPU or XPU, we will default to dequantizing the model to bf16 since no GPU or XPU is available
Loading weights: 100%|███████████████████████████████████████| 795/795 [03:48<00:00,  3.47it/s, Materializing param=model.norm.weight]
Some parameters are on the meta device because they were offloaded to the cpu and disk.
Model class: Ministral3ForCausalLM
Input tokens: tensor([[    1, 22177,  1044,  2036,  2564,  1395]])
Input text: 'Hello, my name is'
Tokenized: ['<s>', 'Hello', ',', 'Ġmy', 'Ġname', 'Ġis']
Logits shape: torch.Size([1, 6, 131072])
Last token logits shape: (131072,)
Vocab size: 131072
First 10 logits: [-5.        -6.65625    1.1640625 -5.25      -5.40625   -5.21875
 -5.09375   -5.40625   -5.09375   -5.21875  ]
Last 10 logits: [-5.4375  -4.5     -2.59375 -6.125   -5.1875  -5.875   -4.625   -5.46875
 -5.59375 -5.5625 ]
Top 5 predictions:
  Token 98746 (' Brenda'): 7.875000
  Token 1267 ('\n\n'): 7.812500
  Token 5939 (' Dr'): 7.593750
  Token 3986 (' John'): 7.375000
  Token 6959 (' David'): 7.250000
Saved bin logits to: data/pytorch-Devstral-2-123B-Instruct-2512.bin
Saved txt logist to: data/pytorch-Devstral-2-123B-Instruct-2512.txt

llama_model_loader: loaded meta data with 42 key-value pairs and 795 tensors from /home/dbevenius/work/llama.cpp-staging/models/Devstral-2-123B-Instruct-2512.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Devstral 2 123B Instruct 2512
llama_model_loader: - kv   3:                            general.version str              = 2512
llama_model_loader: - kv   4:                           general.finetune str              = Instruct
llama_model_loader: - kv   5:                           general.basename str              = Devstral-2
llama_model_loader: - kv   6:                         general.size_label str              = 123B
llama_model_loader: - kv   7:                            general.license str              = other
llama_model_loader: - kv   8:                               general.tags arr[str,1]       = ["mistral-common"]
llama_model_loader: - kv   9:                          llama.block_count u32              = 88
llama_model_loader: - kv  10:                       llama.context_length u32              = 262144
llama_model_loader: - kv  11:                     llama.embedding_length u32              = 12288
llama_model_loader: - kv  12:                  llama.feed_forward_length u32              = 28672
llama_model_loader: - kv  13:                 llama.attention.head_count u32              = 96
llama_model_loader: - kv  14:              llama.attention.head_count_kv u32              = 8
llama_model_loader: - kv  15:                       llama.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  16:     llama.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  17:                 llama.attention.key_length u32              = 128
llama_model_loader: - kv  18:               llama.attention.value_length u32              = 128
llama_model_loader: - kv  19:                          general.file_type u32              = 32
llama_model_loader: - kv  20:                 llama.rope.dimension_count u32              = 128
llama_model_loader: - kv  21:                    llama.rope.scaling.type str              = yarn
llama_model_loader: - kv  22:                  llama.rope.scaling.factor f32              = 64.000000
llama_model_loader: - kv  23:          llama.rope.scaling.yarn_beta_fast f32              = 4.000000
llama_model_loader: - kv  24:          llama.rope.scaling.yarn_beta_slow f32              = 1.000000
llama_model_loader: - kv  25:     llama.rope.scaling.yarn_log_multiplier f32              = 1.000000
llama_model_loader: - kv  26: llama.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  27:               general.quantization_version u32              = 2
llama_model_loader: - kv  28:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  29:                         tokenizer.ggml.pre str              = tekken
llama_model_loader: - kv  30:                      tokenizer.ggml.merges arr[str,269443]  = ["Ġ Ġ", "Ġ t", "e r", "i n", "Ġ �...
llama_model_loader: - kv  31:                tokenizer.ggml.bos_token_id u32              = 1
llama_model_loader: - kv  32:                tokenizer.ggml.eos_token_id u32              = 2
llama_model_loader: - kv  33:            tokenizer.ggml.unknown_token_id u32              = 0
llama_model_loader: - kv  34:            tokenizer.ggml.padding_token_id u32              = 11
llama_model_loader: - kv  35:                      tokenizer.ggml.tokens arr[str,131072]  = ["<unk>", "<s>", "</s>", "[INST]", "[...
llama_model_loader: - kv  36:                      tokenizer.ggml.scores arr[i32,131072]  = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
llama_model_loader: - kv  37:                  tokenizer.ggml.token_type arr[i32,131072]  = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, ...
llama_model_loader: - kv  38:                           llama.vocab_size u32              = 131072
llama_model_loader: - kv  39:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  40:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  41:                    tokenizer.chat_template str              = {#- Default system message if no syst...
llama_model_loader: - type  f32:  177 tensors
llama_model_loader: - type bf16:  618 tensors
print_info: file format = GGUF V3 (latest)
print_info: file type   = BF16
print_info: file size   = 232.88 GiB (16.00 BPW)
Input prompt: "Hello, my name is"
Tokenized prompt (6 tokens): <s> (1)Hello (22177), (1044) my (2036) name (2564) is (1395)
Vocab size: 131072
Saving data to data/llamacpp-Devstral-2-123B-Instruct-2512.bin
First 10 logits: -4.979530 -6.676220  1.161673 -5.248273 -5.435472 -5.205020 -5.096416 -5.410293 -5.099162 -5.201845
Last 10 logits:  -5.496621 -4.489549 -2.593567 -6.141421 -5.175629 -5.888896 -4.596966 -5.429083 -5.639769 -5.575043

Data saved to data/llamacpp-Devstral-2-123B-Instruct-2512.bin
Data saved to data/llamacpp-Devstral-2-123B-Instruct-2512.txt
Using converted model: Devstral-2-123B-Instruct-2512
Checked all required files were found. Proceeding...

🔍 GGML Model Validation for model  Devstral-2-123B-Instruct-2512
========================================
PyTorch logits  : data/pytorch-Devstral-2-123B-Instruct-2512.bin
llama.cpp logits: data/llamacpp-Devstral-2-123B-Instruct-2512.bin

Top 10 PyTorch logits:   [7.875     7.8125    7.59375   7.375     7.25      7.125     7.0625    7.0625    6.9375    6.84375]
Top 10 llama.cpp logits: [7.9227424 7.8393626 7.6513405 7.3801007 7.2741103 7.0872426 7.0714746 7.051253  6.9258447 6.8819084]
Max absolute difference: 0.1232
✅ OK: Lightweight model check successful!
       Ok to proceed with NMSE check...

Model name: Devstral-2-123B-Instruct-2512
PyTorch logits file: data/pytorch-Devstral-2-123B-Instruct-2512.bin
llama.cpp logits file: data/llamacpp-Devstral-2-123B-Instruct-2512.bin

📊 NMSE Check for Model Comparison
==================================================
Reference (ground truth): data/pytorch-Devstral-2-123B-Instruct-2512.bin
Test (to evaluate):       data/llamacpp-Devstral-2-123B-Instruct-2512.bin

Loading reference logits...
  Shape: (131072,), Type: float32
Loading test logits...
  Shape: (131072,), Type: float32

✅ Shapes match: (131072,)

📈 METRICS
==============================
MSE (Mean Squared Error):     5.817258e-04
Reference Variance:           3.842987e+00
NMSE:                         1.513734e-04
Max Absolute Error:           0.123181
Mean Absolute Error:          0.018988
NMSE (dB):                    -38.20 dB

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

✅ RESULT: PASS (NMSE = 1.51e-04)
```

#### Model upload
This model will be added to an existing collection so we need the collection
"slug" which can be retrieved using:
```python
$ cat get-collection-slug.py
from huggingface_hub import get_collection

# This will verify if the slug is valid and return the collection object
collection = get_collection("ggml-org/devstral-2")
print(collection.slug)
```
```console
(venv) $ python get-collection-slug.py
ggml-org/devstral-2-694421075a7abbae56a8188d
```
Upload script:
```console
#!/bin/bash

set -e

namespace="ggml-org"
collection_name="Devstral 2"
base_model_name="Devstral-2-123B-Instruct-2512"
local_model_name="Devstral-2-123B-Instruct-2512"
original_model_ns="mistralai"

model_path=../../models

pushd examples/model-conversion

CAUSAL_COLLECTION_SLUG="ggml-org/devstral-2-694421075a7abbae56a8188d"

### Create Model repository
causal_model_name="${base_model_name}"
original_base_model="${original_model_ns}/${causal_model_name}"
make hf-create-model "MODEL_NAME=${causal_model_name}" "NAMESPACE=${namespace}" "ORIGINAL_BASE_MODEL=${original_base_model}"

### Upload Q8_0 model
causal_model_path="${model_path}/${local_model_name}-Q8_0.gguf"
echo ${causal_model_path}
make hf-upload-gguf-to-model "MODEL_PATH=${causal_model_path}" "REPO_ID=${namespace}/${causal_model_name}-GGUF" "NAME_IN_REPO=${base_model_name}-Q8_0.gguf"

make hf-add-model-to-collection "COLLECTION=${CAUSAL_COLLECTION_SLUG}" "MODEL=${namespace}/${causal_model_name}-GGUF"

popd
```
