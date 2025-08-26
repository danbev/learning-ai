## Quantization

## QAT (Quantization Aware Training) quantization
When quantizing a QAT model it might also be good to make sure that the
embeddings weights are also quantized to say Q8_0. By default the `token_embd`
weights migth be left in float32/float16 precision. For example, if we look at
the following model:
```console
(venv) $ ./gguf-py/gguf/scripts/gguf_dump.py ~/Downloads/gemma-3-27b-it-q4_0.gguf 
INFO:gguf-dump:* Loading: /home/danbev/Downloads/gemma-3-27b-it-q4_0.gguf
* File is LITTLE endian, script is running on a LITTLE endian host.
* Dumping 42 key/value pair(s)
      1: UINT32     |        1 | GGUF.version = 3
      2: UINT64     |        1 | GGUF.tensor_count = 808
      3: UINT64     |        1 | GGUF.kv_count = 39
      4: STRING     |        1 | general.architecture = 'gemma3'
      5: UINT32     |        1 | gemma3.context_length = 131072
      6: UINT32     |        1 | gemma3.block_count = 62
      7: UINT32     |        1 | gemma3.embedding_length = 5376
      8: UINT32     |        1 | gemma3.feed_forward_length = 21504
      9: UINT32     |        1 | gemma3.attention.head_count = 32
     10: UINT32     |        1 | gemma3.attention.head_count_kv = 16
     11: UINT32     |        1 | gemma3.attention.key_length = 128
     12: UINT32     |        1 | gemma3.attention.value_length = 128
     13: FLOAT32    |        1 | gemma3.attention.layer_norm_rms_epsilon = 9.999999974752427e-07
     14: STRING     |        1 | gemma3.rope.scaling.type = 'linear'
     15: FLOAT32    |        1 | gemma3.rope.scaling.factor = 8.0
     16: FLOAT32    |        1 | gemma3.rope.freq_base = 1000000.0
     17: UINT32     |        1 | gemma3.attention.sliding_window = 1024
     18: STRING     |        1 | tokenizer.ggml.model = 'llama'
     19: UINT32     |        1 | tokenizer.ggml.bos_token_id = 2
     20: UINT32     |        1 | tokenizer.ggml.eos_token_id = 1
     21: UINT32     |        1 | tokenizer.ggml.padding_token_id = 0
     22: UINT32     |        1 | tokenizer.ggml.unknown_token_id = 3
     23: [STRING]   |   262144 | tokenizer.ggml.tokens = ['<pad>', '<eos>', '<bos>', '<unk>', '<mask>', '[multimodal]', ...]
     24: [FLOAT32]  |   262144 | tokenizer.ggml.scores = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...]
     25: [INT32]    |   262144 | tokenizer.ggml.token_type = [3, 3, 3, 2, 1, 1, ...]
     26: UINT32     |        1 | general.quantization_version = 2
     27: UINT32     |        1 | general.file_type = 2
     28: STRING     |        1 | tokenizer.chat_template = "{{ bos_token }} {%- if messages[0]['role'] == 'system' -%..."
     29: UINT32     |        1 | gemma3.mm.tokens_per_image = 256
     30: UINT32     |        1 | gemma3.vision.attention.head_count = 16
     31: FLOAT32    |        1 | gemma3.vision.attention.layer_norm_epsilon = 9.999999974752427e-07
     32: UINT32     |        1 | gemma3.vision.block_count = 27
     33: UINT32     |        1 | gemma3.vision.embedding_length = 1152
     34: UINT32     |        1 | gemma3.vision.feed_forward_length = 4304
     35: UINT32     |        1 | gemma3.vision.image_size = 896
     36: UINT32     |        1 | gemma3.vision.num_channels = 3
     37: UINT32     |        1 | gemma3.vision.patch_size = 14
     38: BOOL       |        1 | tokenizer.ggml.add_bos_token = True
     39: BOOL       |        1 | tokenizer.ggml.add_eos_token = False
     40: BOOL       |        1 | tokenizer.ggml.add_padding_token = False
     41: BOOL       |        1 | tokenizer.ggml.add_unknown_token = False
     42: STRING     |        1 | tokenizer.ggml.pre = 'default'
* Dumping 808 tensor(s)
      1:       5376 |  5376,     1,     1,     1 | F32     | output_norm.weight
      2: 1409286144 |  5376, 262144,    1,     1 | F16     | token_embd.weight
      3:   11010048 |  5376,  2048,     1,     1 | Q4_0    | blk.0.attn_k.weight
      4:        128 |   128,     1,     1,     1 | F32     | blk.0.attn_k_norm.weight
      5:       5376 |  5376,     1,     1,     1 | F32     | blk.0.attn_norm.weight
      6:   22020096 |  4096,  5376,     1,     1 | Q4_0    | blk.0.attn_output.weight
      7:   22020096 |  5376,  4096,     1,     1 | Q4_0    | blk.0.attn_q.weight
      8:        128 |   128,     1,     1,     1 | F32     | blk.0.attn_q_norm.weight
      9:   11010048 |  5376,  2048,     1,     1 | Q4_0    | blk.0.attn_v.weight
     10:  115605504 | 21504,  5376,     1,     1 | Q4_0    | blk.0.ffn_down.weight
     11:  115605504 |  5376, 21504,     1,     1 | Q4_0    | blk.0.ffn_gate.weight
     12:       5376 |  5376,     1,     1,     1 | F32     | blk.0.ffn_norm.weight
     13:  115605504 |  5376, 21504,     1,     1 | Q4_0    | blk.0.ffn_up.weight
     14:       5376 |  5376,     1,     1,     1 | F32     | blk.0.post_attention_norm.weight
     15:       5376 |  5376,     1,     1,     1 | F32     | blk.0.post_ffw_norm.weight
```
When we quantize the model using `llama-quantize` there is the option to specify
the data type of the token embedding weights using `--token-embedding-typ`:
```console
(venv) $ ./build/bin/llama-quantize --help
  ...
  --token-embedding-type ggml_type: use this ggml_type for the token embeddings tensor
  ...
```

So this models original weights are in bf16 so we frist convert to that:
```console
(venv) $ export MODEL_PATH=~/work/ai/models/gemma-3-27b-it-qat-q4_0-unquantized/
(venv) $ make causal-convert-model-bf16
```
```console
(venv) $ make causal-inspect-converted-model
INFO:gguf-dump:* Loading: /home/danbev/work/ai/llama.cpp/models/gemma-3-27b-it-qat-q4_0-unquantized.gguf
* File is LITTLE endian, script is running on a LITTLE endian host.
* Dumping 44 key/value pair(s)
      1: UINT32     |        1 | GGUF.version = 3
      2: UINT64     |        1 | GGUF.tensor_count = 808
      3: UINT64     |        1 | GGUF.kv_count = 41
      4: STRING     |        1 | general.architecture = 'gemma3'
      5: STRING     |        1 | general.type = 'model'
      6: STRING     |        1 | general.name = 'Gemma 3 27b It Qat Q4_0 Unquantized'
      7: STRING     |        1 | general.finetune = 'it-qat-unquantized'
      8: STRING     |        1 | general.basename = 'gemma-3'
      9: STRING     |        1 | general.size_label = '27B'
     10: STRING     |        1 | general.license = 'gemma'
     11: UINT32     |        1 | general.base_model.count = 1
     12: STRING     |        1 | general.base_model.0.name = 'Gemma 3 27b It'
     13: STRING     |        1 | general.base_model.0.organization = 'Google'
     14: STRING     |        1 | general.base_model.0.repo_url = 'https://huggingface.co/google/gemma-3-27b-it'
     15: [STRING]   |        4 | general.tags = ['gemma3', 'gemma', 'google', 'image-text-to-text']
     16: UINT32     |        1 | gemma3.context_length = 131072
     17: UINT32     |        1 | gemma3.embedding_length = 5376
     18: UINT32     |        1 | gemma3.block_count = 62
     19: UINT32     |        1 | gemma3.feed_forward_length = 21504
     20: UINT32     |        1 | gemma3.attention.head_count = 32
     21: FLOAT32    |        1 | gemma3.attention.layer_norm_rms_epsilon = 9.999999974752427e-07
     22: UINT32     |        1 | gemma3.attention.key_length = 128
     23: UINT32     |        1 | gemma3.attention.value_length = 128
     24: UINT32     |        1 | general.file_type = 32
     25: FLOAT32    |        1 | gemma3.rope.freq_base = 1000000.0
     26: UINT32     |        1 | gemma3.attention.sliding_window = 1024
     27: UINT32     |        1 | gemma3.attention.head_count_kv = 16
     28: STRING     |        1 | gemma3.rope.scaling.type = 'linear'
     29: FLOAT32    |        1 | gemma3.rope.scaling.factor = 8.0
     30: UINT32     |        1 | general.quantization_version = 2
     31: STRING     |        1 | tokenizer.ggml.model = 'llama'
     32: STRING     |        1 | tokenizer.ggml.pre = 'default'
     33: [STRING]   |   262208 | tokenizer.ggml.tokens = ['<pad>', '<eos>', '<bos>', '<unk>', '<mask>', '[multimodal]', ...]
     34: [FLOAT32]  |   262208 | tokenizer.ggml.scores = [-1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, ...]
     35: [INT32]    |   262208 | tokenizer.ggml.token_type = [3, 3, 3, 3, 3, 4, ...]
     36: UINT32     |        1 | tokenizer.ggml.bos_token_id = 2
     37: UINT32     |        1 | tokenizer.ggml.eos_token_id = 1
     38: UINT32     |        1 | tokenizer.ggml.unknown_token_id = 3
     39: UINT32     |        1 | tokenizer.ggml.padding_token_id = 0
     40: BOOL       |        1 | tokenizer.ggml.add_bos_token = True
     41: BOOL       |        1 | tokenizer.ggml.add_sep_token = False
     42: BOOL       |        1 | tokenizer.ggml.add_eos_token = False
     43: STRING     |        1 | tokenizer.chat_template = "{{ bos_token }}\n{%- if messages[0]['role'] == 'system' -%..."
     44: BOOL       |        1 | tokenizer.ggml.add_space_prefix = False
* Dumping 808 tensor(s)
      1: 1409630208 |  5376, 262208,     1,     1 | BF16    | token_embd.weight
      2:       5376 |  5376,     1,     1,     1 | F32     | blk.0.attn_norm.weight
      3:  115605504 | 21504,  5376,     1,     1 | BF16    | blk.0.ffn_down.weight
      4:  115605504 |  5376, 21504,     1,     1 | BF16    | blk.0.ffn_gate.weight
      5:  115605504 |  5376, 21504,     1,     1 | BF16    | blk.0.ffn_up.weight
      6:       5376 |  5376,     1,     1,     1 | F32     | blk.0.post_attention_norm.weight
      7:       5376 |  5376,     1,     1,     1 | F32     | blk.0.post_ffw_norm.weight
      8:       5376 |  5376,     1,     1,     1 | F32     | blk.0.ffn_norm.weight
      9:        128 |   128,     1,     1,     1 | F32     | blk.0.attn_k_norm.weight
     10:   11010048 |  5376,  2048,     1,     1 | BF16    | blk.0.attn_k.weight
     11:   22020096 |  4096,  5376,     1,     1 | BF16    | blk.0.attn_output.weight
     12:        128 |   128,     1,     1,     1 | F32     | blk.0.attn_q_norm.weight
     13:   22020096 |  5376,  4096,     1,     1 | BF16    | blk.0.attn_q.weight
     14:   11010048 |  5376,  2048,     1,     1 | BF16    | blk.0.attn_v.weight
     ...
```
So this models has been trained with QAT and specifically Q4_0 quantization in
mind, so we can quantize it using the following command:
```console
(venv) $ make causal-quantize-qat-Q4_0
```

This will produce a model which by defaults will have the token embedding weights
as Q8_0 quantized. We can verify this by inspecting the model:
```console
(venv) $ ../../gguf-py/gguf/scripts/gguf_dump.py ../../models/gemma-3-27b-it-qat-q4_0-unquantized-Q4_0.gguf
INFO:gguf-dump:* Loading: ../../models/gemma-3-27b-it-qat-q4_0-unquantized-Q4_0.gguf
* File is LITTLE endian, script is running on a LITTLE endian host.
* Dumping 44 key/value pair(s)
      1: UINT32     |        1 | GGUF.version = 3
      2: UINT64     |        1 | GGUF.tensor_count = 808
      3: UINT64     |        1 | GGUF.kv_count = 41
      4: STRING     |        1 | general.architecture = 'gemma3'
      5: STRING     |        1 | general.type = 'model'
      6: STRING     |        1 | general.name = 'Gemma 3 27b It Qat Q4_0 Unquantized'
      7: STRING     |        1 | general.finetune = 'it-qat-unquantized'
      8: STRING     |        1 | general.basename = 'gemma-3'
      9: STRING     |        1 | general.size_label = '27B'
     10: STRING     |        1 | general.license = 'gemma'
     11: UINT32     |        1 | general.base_model.count = 1
     12: STRING     |        1 | general.base_model.0.name = 'Gemma 3 27b It'
     13: STRING     |        1 | general.base_model.0.organization = 'Google'
     14: STRING     |        1 | general.base_model.0.repo_url = 'https://huggingface.co/google/gemma-3-27b-it'
     15: [STRING]   |        4 | general.tags = ['gemma3', 'gemma', 'google', 'image-text-to-text']
     16: UINT32     |        1 | gemma3.context_length = 131072
     17: UINT32     |        1 | gemma3.embedding_length = 5376
     18: UINT32     |        1 | gemma3.block_count = 62
     19: UINT32     |        1 | gemma3.feed_forward_length = 21504
     20: UINT32     |        1 | gemma3.attention.head_count = 32
     21: FLOAT32    |        1 | gemma3.attention.layer_norm_rms_epsilon = 9.999999974752427e-07
     22: UINT32     |        1 | gemma3.attention.key_length = 128
     23: UINT32     |        1 | gemma3.attention.value_length = 128
     24: FLOAT32    |        1 | gemma3.rope.freq_base = 1000000.0
     25: UINT32     |        1 | gemma3.attention.sliding_window = 1024
     26: UINT32     |        1 | gemma3.attention.head_count_kv = 16
     27: STRING     |        1 | gemma3.rope.scaling.type = 'linear'
     28: FLOAT32    |        1 | gemma3.rope.scaling.factor = 8.0
     29: STRING     |        1 | tokenizer.ggml.model = 'llama'
     30: STRING     |        1 | tokenizer.ggml.pre = 'default'
     31: [STRING]   |   262208 | tokenizer.ggml.tokens = ['<pad>', '<eos>', '<bos>', '<unk>', '<mask>', '[multimodal]', ...]
     32: [FLOAT32]  |   262208 | tokenizer.ggml.scores = [-1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, ...]
     33: [INT32]    |   262208 | tokenizer.ggml.token_type = [3, 3, 3, 3, 3, 4, ...]
     34: UINT32     |        1 | tokenizer.ggml.bos_token_id = 2
     35: UINT32     |        1 | tokenizer.ggml.eos_token_id = 1
     36: UINT32     |        1 | tokenizer.ggml.unknown_token_id = 3
     37: UINT32     |        1 | tokenizer.ggml.padding_token_id = 0
     38: BOOL       |        1 | tokenizer.ggml.add_bos_token = True
     39: BOOL       |        1 | tokenizer.ggml.add_sep_token = False
     40: BOOL       |        1 | tokenizer.ggml.add_eos_token = False
     41: STRING     |        1 | tokenizer.chat_template = "{{ bos_token }}\n{%- if messages[0]['role'] == 'system' -%..."
     42: BOOL       |        1 | tokenizer.ggml.add_space_prefix = False
     43: UINT32     |        1 | general.quantization_version = 2
     44: UINT32     |        1 | general.file_type = 2
* Dumping 808 tensor(s)
      1:       5376 |  5376,     1,     1,     1 | F32     | output_norm.weight
      2: 1409630208 |  5376, 262208,    1,     1 | Q8_0    | token_embd.weight
      3:   11010048 |  5376,  2048,     1,     1 | Q4_0    | blk.0.attn_k.weight
      4:        128 |   128,     1,     1,     1 | F32     | blk.0.attn_k_norm.weight
      5:       5376 |  5376,     1,     1,     1 | F32     | blk.0.attn_norm.weight
      6:   22020096 |  4096,  5376,     1,     1 | Q4_0    | blk.0.attn_output.weight
      7:   22020096 |  5376,  4096,     1,     1 | Q4_0    | blk.0.attn_q.weight
      8:        128 |   128,     1,     1,     1 | F32     | blk.0.attn_q_norm.weight
      9:   11010048 |  5376,  2048,     1,     1 | Q4_0    | blk.0.attn_v.weight
     10:  115605504 | 21504,  5376,     1,     1 | Q4_0    | blk.0.ffn_down.weight
     11:  115605504 |  5376, 21504,     1,     1 | Q4_0    | blk.0.ffn_gate.weight
     12:       5376 |  5376,     1,     1,     1 | F32     | blk.0.ffn_norm.weight
     13:  115605504 |  5376, 21504,     1,     1 | Q4_0    | blk.0.ffn_up.weight
     14:       5376 |  5376,     1,     1,     1 | F32     | blk.0.post_attention_norm.weight
     15:       5376 |  5376,     1,     1,     1 | F32     | blk.0.post_ffw_norm.weight
     ...
```
The difference in between this and the previous model is that the token embedding
```console
(venv) $ ls -lh ../../models/gemma-3-27b-it-qat-q4_0-unquantized-Q4_0.gguf
-rw-rw-r-- 1 danbev danbev 15G Aug 26 06:29 ../../models/gemma-3-27b-it-qat-q4_0-unquantized-Q4_0.gguf
(venv) $ ls -lh ~/Downloads/gemma-3-27b-it-q4_0.gguf
-rw-rw-r-- 1 danbev danbev 17G Aug 25 20:05 /home/danbev/Downloads/gemma-3-27b-it-q4_0.gguf
```

It is not clear to me why Q8_0 is chosen and not Q4_0, but it might be that we
want to keep the precision of the token embeddings higher than the rest of the
model. And recall that this is specifically for QAT models.

