### Gemma 3-270m bos token issue
This is an issue that we discovered after converting Gemma 3-270m to GGUF format.
and running these models.

### Behavior of base model (non instruction tuned)
When running the Gemma 3-270m model, it was observed that the model was
generating double BOS tokens.
```console
(venv) $ ./build/bin/llama-cli -m models/gemma-3-270m.gguf -p "hello" -n 20 -sp
llama_model_loader: loaded meta data with 33 key-value pairs and 236 tensors from models/gemma-3-270m.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = gemma3
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Gemma 3 270m
llama_model_loader: - kv   3:                           general.basename str              = gemma-3
llama_model_loader: - kv   4:                         general.size_label str              = 270M
llama_model_loader: - kv   5:                            general.license str              = gemma
llama_model_loader: - kv   6:                               general.tags arr[str,4]       = ["gemma3", "gemma", "google", "text-g...
llama_model_loader: - kv   7:                      gemma3.context_length u32              = 32768
llama_model_loader: - kv   8:                    gemma3.embedding_length u32              = 640
llama_model_loader: - kv   9:                         gemma3.block_count u32              = 18
llama_model_loader: - kv  10:                 gemma3.feed_forward_length u32              = 2048
llama_model_loader: - kv  11:                gemma3.attention.head_count u32              = 4
llama_model_loader: - kv  12:    gemma3.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  13:                gemma3.attention.key_length u32              = 256
llama_model_loader: - kv  14:              gemma3.attention.value_length u32              = 256
llama_model_loader: - kv  15:                          general.file_type u32              = 1
llama_model_loader: - kv  16:                      gemma3.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  17:            gemma3.attention.sliding_window u32              = 512
llama_model_loader: - kv  18:             gemma3.attention.head_count_kv u32              = 1
llama_model_loader: - kv  19:               general.quantization_version u32              = 2
llama_model_loader: - kv  20:                       tokenizer.ggml.model str              = llama
llama_model_loader: - kv  21:                         tokenizer.ggml.pre str              = default
llama_model_loader: - kv  22:                      tokenizer.ggml.tokens arr[str,262144]  = ["<pad>", "<eos>", "<bos>", "<unk>", ...
llama_model_loader: - kv  23:                      tokenizer.ggml.scores arr[f32,262144]  = [-1000.000000, -1000.000000, -1000.00...
llama_model_loader: - kv  24:                  tokenizer.ggml.token_type arr[i32,262144]  = [3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3, ...
llama_model_loader: - kv  25:                tokenizer.ggml.bos_token_id u32              = 2
llama_model_loader: - kv  26:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  27:            tokenizer.ggml.unknown_token_id u32              = 3
llama_model_loader: - kv  28:            tokenizer.ggml.padding_token_id u32              = 0
llama_model_loader: - kv  29:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  30:               tokenizer.ggml.add_sep_token bool             = false
llama_model_loader: - kv  31:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  32:            tokenizer.ggml.add_space_prefix bool             = false

...

<bos>hello. I'm using a new version of the game. I have been playing it for a while
```

The pretrained/base model has following in `tokenizer_config.json`:
```console
{
  "add_bos_token": true,
  ```
}
```

### Behavior of instruction tuned model
```console
(venv) $ ./build/bin/llama-cli -m models/gemma-3-270m-it.gguf -p "What is the capital of France?" -n 20 -sp -no-cnv
...
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = gemma3
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Gemma 3 270m It
llama_model_loader: - kv   3:                           general.finetune str              = it
llama_model_loader: - kv   4:                           general.basename str              = gemma-3
llama_model_loader: - kv   5:                         general.size_label str              = 270M
llama_model_loader: - kv   6:                            general.license str              = gemma
llama_model_loader: - kv   7:                   general.base_model.count u32              = 1
llama_model_loader: - kv   8:                  general.base_model.0.name str              = Gemma 3 270m
llama_model_loader: - kv   9:          general.base_model.0.organization str              = Google
llama_model_loader: - kv  10:              general.base_model.0.repo_url str              = https://huggingface.co/google/gemma-3...
llama_model_loader: - kv  11:                               general.tags arr[str,4]       = ["gemma3", "gemma", "google", "text-g...
llama_model_loader: - kv  12:                      gemma3.context_length u32              = 32768
llama_model_loader: - kv  13:                    gemma3.embedding_length u32              = 640
llama_model_loader: - kv  14:                         gemma3.block_count u32              = 18
llama_model_loader: - kv  15:                 gemma3.feed_forward_length u32              = 2048
llama_model_loader: - kv  16:                gemma3.attention.head_count u32              = 4
llama_model_loader: - kv  17:    gemma3.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  18:                gemma3.attention.key_length u32              = 256
llama_model_loader: - kv  19:              gemma3.attention.value_length u32              = 256
llama_model_loader: - kv  20:                          general.file_type u32              = 1
llama_model_loader: - kv  21:                      gemma3.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  22:            gemma3.attention.sliding_window u32              = 512
llama_model_loader: - kv  23:             gemma3.attention.head_count_kv u32              = 1
llama_model_loader: - kv  24:               general.quantization_version u32              = 2
llama_model_loader: - kv  25:                       tokenizer.ggml.model str              = llama
llama_model_loader: - kv  26:                         tokenizer.ggml.pre str              = default
llama_model_loader: - kv  27:                      tokenizer.ggml.tokens arr[str,262144]  = ["<pad>", "<eos>", "<bos>", "<unk>", ...
llama_model_loader: - kv  28:                      tokenizer.ggml.scores arr[f32,262144]  = [-1000.000000, -1000.000000, -1000.00...
llama_model_loader: - kv  29:                  tokenizer.ggml.token_type arr[i32,262144]  = [3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3, ...
llama_model_loader: - kv  30:                tokenizer.ggml.bos_token_id u32              = 2
llama_model_loader: - kv  31:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  32:            tokenizer.ggml.unknown_token_id u32              = 3
llama_model_loader: - kv  33:            tokenizer.ggml.padding_token_id u32              = 0
llama_model_loader: - kv  34:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  35:               tokenizer.ggml.add_sep_token bool             = false
llama_model_loader: - kv  36:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  37:                    tokenizer.chat_template str              = {{ bos_token }}\n{%- if messages[0]['r...
llama_model_loader: - kv  38:            tokenizer.ggml.add_space_prefix bool             = false
...

<bos>What is the capital of France?

A) Paris
B) London
C) Rome
D) Berlin

**Answer:**
```
The instruction tuned model has the following in `tokenizer_config.json`:
```console
{
  "add_bos_token": true,
  ...
}
```
