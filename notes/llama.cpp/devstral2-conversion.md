### Devstral 2 Model conversion
This model can be found in this Hugging Face model repository:  
https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512


### Downloading the model files
```console
(venv) $ cd ~/models
(venv) $ export HF_TOKEN=<token>
(venv) $ hf download mistralai/Devstral-Small-2-24B-Instruct-2512 --local-dir Devstral-Small-2-24B-Instruct-2512
```

### Run the original model
Lets run the original model to get some output logits to compare visually against
later.
```console
(venv) $ cd examples/model-conversion
(venv) $ make causual-run-original_model
Input tokens: tensor([[22177,  1044,  2036,  2564,  1395]], device='cuda:0')
Input text: 'Hello, my name is'
Tokenized: ['Hello', ',', 'Ġmy', 'Ġname', 'Ġis']

Logits shape: torch.Size([1, 5, 131072])
Last token logits shape: (131072,)
Vocab size: 131072
First 10 logits: [-7.5    -7.5      2.      -7.5     -7.5     -7.5     -7.5     -7.5     -7.5     -7.09375]
Last 10 logits: [-4.90625 -4.90625 -6.40625 -8.4375  -6.78125 -6.15625 -5.0625  -7.09375 -5.375   -6.3125 ]
Top 5 predictions:
  Token 1278 (' the'): 5.968750
  Token 1032 (' '): 5.468750
  Token 1766 (' ['): 5.125000
  Token 2129 (' “'): 5.062500
  Token 1261 (' a'): 5.000000
Saved bin logits to: data/pytorch-Devstral-Small-2-24B-Instruct-2512.bin
Saved txt logist to: data/pytorch-Devstral-Small-2-24B-Instruct-2512.txt
```

But if I run this using the mistral transformers package I get different logits
and a different number of tokens (the BOS token is added):
```console
(venv) $ python run-devstral.py
Unrecognized keys in `rope_parameters` for 'rope_type'='yarn': {'max_position_embeddings'}
/home/dbevenius/work/llama.cpp-staging/venv/lib/python3.12/site-packages/torch/cuda/__init__.py:285: UserWarning:
    Found GPU0 NVIDIA GB10 which is of cuda capability 12.1.
    Minimum and Maximum cuda capability supported by this version of PyTorch is
    (8.0) - (12.0)

  warnings.warn(
Loading weights: 100%|█| 1145/1145 [02:51<00:00,  6.68it/s, Materializing param=model.vision_tower.transformer.layers.23.ffn_norm.weig
Input tokens: tensor([[    1, 22177,  1044,  2036,  2564,  1395]], device='cuda:0')
Input text: 'Hello, my name is'
Tokenized: ['<s>', 'Hello', ',', ' my', ' name', ' is']
Processing chunk with tokens 0 to 512
Logits shape: torch.Size([1, 6, 131072])
Last token logits shape: (131072,)
Vocab size: 131072
First 10 logits: [-4.78125  -4.78125 -1.25   -4.78125 -4.78125 -4.78125 -4.78125 -4.78125 -4.78125  -6.5     ]
Last 10 logits:  [-4.03125  -3.46875 -4.5625 -5.625    -4.5    -4.0625  -3.53125 -5.5     -1.921875 -4.46875 ]
Top 5 predictions:
  Token 1605 (' not'): 9.812500
  Token 2036 (' my'): 8.437500
  Token 1261 (' a'): 8.312500
  Token 1395 (' is'): 7.468750
  Token 1278 (' the'): 7.031250
```
And notice that the model actually does add a BOS token at the start of the input!

This is the python script used above:
```python
import torch
from transformers import (
    Mistral3ForConditionalGeneration,
    MistralCommonBackend,
)
import numpy as np

model_path = "/home/dbevenius/models/Devstral-Small-2-24B-Instruct-2512"

tokenizer = MistralCommonBackend.from_pretrained(model_path, trust_remote_code=True)
model = Mistral3ForConditionalGeneration.from_pretrained(model_path, device_map="auto", trust_remote_code=True)

device = next(model.parameters()).device
prompt = "Hello, my name is"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

print(f"Input tokens: {input_ids}")
print(f"Input text: {repr(prompt)}")
print(f"Tokenized: {tokenizer.convert_ids_to_tokens(input_ids[0])}")

batch_size = 512

with torch.no_grad():
    past = None
    outputs = None
    for i in range(0, input_ids.size(1), batch_size):
        print(f"Processing chunk with tokens {i} to {i + batch_size}")
        chunk = input_ids[:, i:i + batch_size]
        outputs = model(chunk.to(model.device), past_key_values=past, use_cache=True)
        past = outputs.past_key_values

    logits = outputs.logits # type: ignore

    # Extract logits for the last token (next token prediction)
    last_logits = logits[0, -1, :].float().cpu().numpy()

    print(f"Logits shape: {logits.shape}")
    print(f"Last token logits shape: {last_logits.shape}")
    print(f"Vocab size: {len(last_logits)}")

    # Print some sample logits for quick verification
    print(f"First 10 logits: {last_logits[:10]}")
    print(f"Last 10 logits: {last_logits[-10:]}")

    # Show top 5 predicted tokens
    top_indices = np.argsort(last_logits)[-5:][::-1]
    print("Top 5 predictions:")
    for idx in top_indices:
        token = tokenizer.decode([idx])
        print(f"  Token {idx} ({repr(token)}): {last_logits[idx]:.6f}")
```
I've update our conversion script to match the above and I can get the same
logits with it now:
```console
Model class: Mistral3ForConditionalGeneration
Input tokens: tensor([[    1, 22177,  1044,  2036,  2564,  1395]], device='cuda:0')
Input text: 'Hello, my name is'
Tokenized: ['<s>', 'Hello', ',', ' my', ' name', ' is']
Processing chunk with tokens 0 to 512
Logits shape: torch.Size([1, 6, 131072])
Last token logits shape: (131072,)
Vocab size: 131072
First 10 logits: [-4.78125 -4.78125 -1.25   -4.78125 -4.78125 -4.78125 -4.78125 -4.78125 -4.78125  -6.5    ]
Last 10 logits:  [-4.03125 -3.46875 -4.5625 -5.625   -4.5     -4.0625  -3.53125 -5.5     -1.921875 -4.46875 ]
Top 5 predictions:
  Token 1605 (' not'): 9.812500
  Token 2036 (' my'): 8.437500
  Token 1261 (' a'): 8.312500
  Token 1395 (' is'): 7.468750
  Token 1278 (' the'): 7.031250
```
So this matches the original model output now and should be good for verifying
the converted model later.

### Model Conversion setup
The following package versions are required for the conversion:
```console
(venv) $ pip install sentence-transformers==5.2.0
(venv) $ pip install transformers==5.0.0rc0
(venv) $ pip install mistral-common
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
Last 10 logits: -4.996169 -3.164495 -3.793222 -6.617929 -4.842769 -5.592521 -3.666847 -4.626977 -5.998837 -5.684978
```

### Model verification
```console
(venv) $ make causal-verify-logits 
Loading model and tokenizer using AutoTokenizer: /home/dbevenius/models/Devstral-Small-2-24B-Instruct-2512
The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is ignored.
You are using a model of type mistral3 to instantiate a model of type mistral. This is not supported for all configurations of models and can yield errors.
Model type:        mistral3
Vocab size:        32000
Hidden size:       4096
Number of layers:  32
BOS token id:      1
EOS token id:      2
Unrecognized keys in `rope_parameters` for 'rope_type'='yarn': {'max_position_embeddings'}
/home/dbevenius/work/llama.cpp-staging/venv/lib/python3.12/site-packages/torch/cuda/__init__.py:285: UserWarning: 
    Found GPU0 NVIDIA GB10 which is of cuda capability 12.1.
    Minimum and Maximum cuda capability supported by this version of PyTorch is
    (8.0) - (12.0)
    
  warnings.warn(
Loading weights: 100%|█| 1145/1145 [03:29<00:00,  5.47it/s, Materializing param=model.vision_tower.transformer.layers.23.ffn_norm.weig
Model class: Mistral3ForConditionalGeneration
Input tokens: tensor([[    1, 22177,  1044,  2036,  2564,  1395]], device='cuda:0')
Input text: 'Hello, my name is'
Tokenized: ['<s>', 'Hello', ',', ' my', ' name', ' is']
Processing chunk with tokens 0 to 512
Logits shape: torch.Size([1, 6, 131072])
Last token logits shape: (131072,)
Vocab size: 131072
First 10 logits: [-4.78125 -4.78125 -1.25    -4.78125 -4.78125 -4.78125 -4.78125 -4.78125
 -4.78125 -6.5    ]
Last 10 logits: [-4.03125  -3.46875  -4.5625   -5.625    -4.5      -4.0625   -3.53125
 -5.5      -1.921875 -4.46875 ]
Top 5 predictions:
  Token 1605 (' not'): 9.812500
  Token 2036 (' my'): 8.437500
  Token 1261 (' a'): 8.312500
  Token 1395 (' is'): 7.468750
  Token 1278 (' the'): 7.031250
Saved bin logits to: data/pytorch-Devstral-Small-2-24B-Instruct-2512.bin
Saved txt logist to: data/pytorch-Devstral-Small-2-24B-Instruct-2512.txt
llama_model_load_from_file_impl: using device CUDA0 (NVIDIA GB10) (000f:01:00.0) - 118548 MiB free
llama_model_loader: loaded meta data with 48 key-value pairs and 363 tensors from /home/dbevenius/work/llama.cpp-staging/models/Devstral-Small-2-24B-Instruct-2512.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = mistral3
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Devstral Small 2 24B Instruct 2512
llama_model_loader: - kv   3:                            general.version str              = 2512
llama_model_loader: - kv   4:                           general.finetune str              = Instruct
llama_model_loader: - kv   5:                           general.basename str              = Devstral-Small-2
llama_model_loader: - kv   6:                         general.size_label str              = 24B
llama_model_loader: - kv   7:                            general.license str              = apache-2.0
llama_model_loader: - kv   8:                   general.base_model.count u32              = 1
llama_model_loader: - kv   9:                  general.base_model.0.name str              = Mistral Small 3.1 24B Base 2503
llama_model_loader: - kv  10:               general.base_model.0.version str              = 2503
llama_model_loader: - kv  11:          general.base_model.0.organization str              = Mistralai
llama_model_loader: - kv  12:              general.base_model.0.repo_url str              = https://huggingface.co/mistralai/Mist...
llama_model_loader: - kv  13:                               general.tags arr[str,1]       = ["mistral-common"]
llama_model_loader: - kv  14:                       mistral3.block_count u32              = 40
llama_model_loader: - kv  15:                    mistral3.context_length u32              = 393216
llama_model_loader: - kv  16:                  mistral3.embedding_length u32              = 5120
llama_model_loader: - kv  17:               mistral3.feed_forward_length u32              = 32768
llama_model_loader: - kv  18:              mistral3.attention.head_count u32              = 32
llama_model_loader: - kv  19:           mistral3.attention.head_count_kv u32              = 8
llama_model_loader: - kv  20:                    mistral3.rope.freq_base f32              = 100000000.000000
llama_model_loader: - kv  21:  mistral3.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  22:              mistral3.attention.key_length u32              = 128
llama_model_loader: - kv  23:            mistral3.attention.value_length u32              = 128
llama_model_loader: - kv  24:                          general.file_type u32              = 32
llama_model_loader: - kv  25:              mistral3.rope.dimension_count u32              = 128
llama_model_loader: - kv  26:                 mistral3.rope.scaling.type str              = yarn
llama_model_loader: - kv  27:               mistral3.rope.scaling.factor f32              = 48.000000
llama_model_loader: - kv  28:       mistral3.rope.scaling.yarn_beta_fast f32              = 32.000000
llama_model_loader: - kv  29:       mistral3.rope.scaling.yarn_beta_slow f32              = 1.000000
llama_model_loader: - kv  30:  mistral3.rope.scaling.yarn_log_multiplier f32              = 1.000000
llama_model_loader: - kv  31: mistral3.rope.scaling.original_context_length u32              = 8192
llama_model_loader: - kv  32:       mistral3.attention.temperature_scale f32              = 0.100000
llama_model_loader: - kv  33:               general.quantization_version u32              = 2
llama_model_loader: - kv  34:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  35:                         tokenizer.ggml.pre str              = tekken
llama_model_loader: - kv  36:                      tokenizer.ggml.merges arr[str,269443]  = ["Ġ Ġ", "Ġ t", "e r", "i n", "Ġ �...
llama_model_loader: - kv  37:                tokenizer.ggml.bos_token_id u32              = 1
llama_model_loader: - kv  38:                tokenizer.ggml.eos_token_id u32              = 2
llama_model_loader: - kv  39:            tokenizer.ggml.unknown_token_id u32              = 0
llama_model_loader: - kv  40:            tokenizer.ggml.padding_token_id u32              = 11
llama_model_loader: - kv  41:                      tokenizer.ggml.tokens arr[str,131072]  = ["<unk>", "<s>", "</s>", "[INST]", "[...
llama_model_loader: - kv  42:                      tokenizer.ggml.scores arr[i32,131072]  = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
llama_model_loader: - kv  43:                  tokenizer.ggml.token_type arr[i32,131072]  = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, ...
llama_model_loader: - kv  44:                        mistral3.vocab_size u32              = 131072
llama_model_loader: - kv  45:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  46:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  47:                    tokenizer.chat_template str              = {#- Default system message if no syst...
llama_model_loader: - type  f32:   81 tensors
llama_model_loader: - type bf16:  282 tensors
print_info: file format = GGUF V3 (latest)
print_info: file type   = BF16
print_info: file size   = 43.91 GiB (16.00 BPW) 
init_tokenizer: initializing tokenizer for type 2
load: special tokens cache size = 1000
load: token to piece cache size = 0.8498 MB
print_info: arch             = mistral3
print_info: vocab_only       = 0
print_info: no_alloc         = 0
print_info: n_ctx_train      = 393216
print_info: n_embd           = 5120
print_info: n_embd_inp       = 5120
print_info: n_layer          = 40
print_info: n_head           = 32
print_info: n_head_kv        = 8
print_info: n_rot            = 128
print_info: n_swa            = 0
print_info: is_swa_any       = 0
print_info: n_embd_head_k    = 128
print_info: n_embd_head_v    = 128
print_info: n_gqa            = 4
print_info: n_embd_k_gqa     = 1024
print_info: n_embd_v_gqa     = 1024
print_info: f_norm_eps       = 0.0e+00
print_info: f_norm_rms_eps   = 1.0e-05
print_info: f_clamp_kqv      = 0.0e+00
print_info: f_max_alibi_bias = 0.0e+00
print_info: f_logit_scale    = 0.0e+00
print_info: f_attn_scale     = 0.0e+00
print_info: n_ff             = 32768
print_info: n_expert         = 0
print_info: n_expert_used    = 0
print_info: n_expert_groups  = 0
print_info: n_group_used     = 0
print_info: causal attn      = 1
print_info: pooling type     = 0
print_info: rope type        = 0
print_info: rope scaling     = yarn
print_info: freq_base_train  = 100000000.0
print_info: freq_scale_train = 0.0208333
print_info: n_ctx_orig_yarn  = 8192
print_info: rope_yarn_log_mul= 1.0000
print_info: rope_finetuned   = unknown
print_info: model type       = 14B
print_info: model params     = 23.57 B
print_info: general.name     = Devstral Small 2 24B Instruct 2512
print_info: vocab type       = BPE
print_info: n_vocab          = 131072
print_info: n_merges         = 269443
print_info: BOS token        = 1 '<s>'
print_info: EOS token        = 2 '</s>'
print_info: UNK token        = 0 '<unk>'
print_info: PAD token        = 11 '<pad>'
print_info: LF token         = 1010 'Ċ'
print_info: EOG token        = 2 '</s>'
print_info: max token length = 150
load_tensors: loading model tensors, this can take a while... (mmap = true)
ggml_backend_cuda_get_available_uma_memory: final available_memory_kb: 121329656
load_tensors: offloading 0 repeating layers to GPU
load_tensors: offloaded 0/41 layers to GPU
load_tensors:   CPU_Mapped model buffer size = 44961.58 MiB
.................................................................................................
Model name: Devstral-Small-2-24B-Instruct-2512
llama_context: constructing llama_context
llama_context: setting new yarn_attn_factor = 1.0000 (mscale == 1.0, mscale_all_dim = 1.0)
llama_context: n_seq_max     = 1
llama_context: n_ctx         = 256
llama_context: n_ctx_seq     = 256
llama_context: n_batch       = 6
llama_context: n_ubatch      = 6
llama_context: causal_attn   = 1
llama_context: flash_attn    = auto
llama_context: kv_unified    = false
llama_context: freq_base     = 100000000.0
llama_context: freq_scale    = 0.0208333
llama_context: n_ctx_seq (256) < n_ctx_train (393216) -- the full capacity of the model will not be utilized
set_abort_callback: call
llama_context:        CPU  output buffer size =     0.50 MiB
llama_kv_cache: layer  39: dev = CPU
llama_kv_cache:        CPU KV buffer size =    40.00 MiB
llama_kv_cache: size =   40.00 MiB (   256 cells,  40 layers,  1/1 seqs), K (f16):   20.00 MiB, V (f16):   20.00 MiB
llama_context: enumerating backends
llama_context: backend_ptrs.size() = 2
llama_context: max_nodes = 2904
llama_context: reserving full memory module
llama_context: worst-case: n_tokens = 6, n_seqs = 1, n_outputs = 1
graph_reserve: reserving a graph for ubatch with n_tokens =    1, n_seqs =  1, n_outputs =    1
llama_context: Flash Attention was auto, set to enabled
graph_reserve: reserving a graph for ubatch with n_tokens =    6, n_seqs =  1, n_outputs =    6
graph_reserve: reserving a graph for ubatch with n_tokens =    1, n_seqs =  1, n_outputs =    1
graph_reserve: reserving a graph for ubatch with n_tokens =    6, n_seqs =  1, n_outputs =    6
llama_context:  CUDA_Host compute buffer size =     3.12 MiB
llama_context: graph nodes  = 1287
llama_context: graph splits = 1

Input prompt: "Hello, my name is"
Tokenized prompt (6 tokens): <s> (1)Hello (22177), (1044) my (2036) name (2564) is (1395)
Vocab size: 131072
Saving data to data/llamacpp-Devstral-Small-2-24B-Instruct-2512.bin
First 10 logits: -6.039290 -6.039550 0.213944 -6.042861 -6.039347 -6.043080 -6.039300 -6.039615 -6.039384 -3.323704 
Last 10 logits: -4.996169 -3.164495 -3.793222 -6.617929 -4.842769 -5.592521 -3.666847 -4.626977 -5.998837 -5.684978 

Data saved to data/llamacpp-Devstral-Small-2-24B-Instruct-2512.bin
Data saved to data/llamacpp-Devstral-Small-2-24B-Instruct-2512.txt

Using converted model: Devstral-Small-2-24B-Instruct-2512
Checked all required files were found. Proceeding...

🔍 GGML Model Validation for model  Devstral-Small-2-24B-Instruct-2512
========================================
PyTorch logits  : data/pytorch-Devstral-Small-2-24B-Instruct-2512.bin
llama.cpp logits: data/llamacpp-Devstral-Small-2-24B-Instruct-2512.bin

Top 10 PyTorch logits:   [9.8125    8.4375   8.3125    7.46875   7.03125   6.65625   6.59375  6.59375  6.59375   6.34375 ]
Top 10 llama.cpp logits: [6.7031956 6.032293 5.8454757 5.5685377 5.5511737 5.5213685 5.382902 5.316436 5.2942443 5.281416]

Max absolute difference: 11.0893
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
MSE (Mean Squared Error):     7.065036e+00
Reference Variance:           1.903586e+00
NMSE:                         3.711434e+00
Max Absolute Error:           11.089338
Mean Absolute Error:          2.278391
NMSE (dB):                    5.70 dB

🎯 INTERPRETATION
==============================
❌ Very poor match (worse than noise)

📋 GUIDANCE
==============================
❌ PROBLEMATIC: Large differences detected.
   Check your conversion process for potential issues.
   Verify you're using the same model weights.

📚 NMSE BENCHMARKS
==============================
< 1e-6:  Essentially identical
< 1e-4:  Excellent (typical for good conversions)
< 1e-3:  Very good
< 1e-2:  Good (acceptable for most use cases)
< 0.1:   Acceptable (may need verification)
> 1.0:   Poor (worse than random)

❌ RESULT: NEEDS REVIEW (NMSE = 3.71e+00)
make: *** [Makefile:63: causal-verify-logits] Error 1
```

_wip_
