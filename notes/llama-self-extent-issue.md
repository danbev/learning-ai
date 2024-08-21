## llama.cpp self-extent issue

Size of initial prompt:
```console
./llama-tokenize -m /home/danbev/.cache/lm-studio/models/TheBloke/TinyLlama-1.1B-1T-OpenOrca-GGUF/tinyllama-1.1b-1t-openorca.Q2_K.gguf -f self-extend.txt --show-count

Total number of tokens: 7038
```
Model information:
```console
$ ./inspect-model.sh ~/.cache/lm-studio/models/TheBloke/TinyLlama-1.1B-1T-OpenOrca-GGUF/tinyllama-1.1b-1t-openorca.Q2_K.gguf 
INFO:gguf-dump:* Loading: /home/danbev/.cache/lm-studio/models/TheBloke/TinyLlama-1.1B-1T-OpenOrca-GGUF/tinyllama-1.1b-1t-openorca.Q2_K.gguf
* File is LITTLE endian, script is running on a LITTLE endian host.
* Dumping 22 key/value pair(s)
      1: UINT32     |        1 | GGUF.version = 2
      2: UINT64     |        1 | GGUF.tensor_count = 201
      3: UINT64     |        1 | GGUF.kv_count = 19
      4: STRING     |        1 | general.architecture = 'llama'
      5: STRING     |        1 | general.name = 'jeff31415_tinyllama-1.1b-1t-openorca'
      6: UINT32     |        1 | llama.context_length = 2048
      7: UINT32     |        1 | llama.embedding_length = 2048
      8: UINT32     |        1 | llama.block_count = 22
      9: UINT32     |        1 | llama.feed_forward_length = 5632
     10: UINT32     |        1 | llama.rope.dimension_count = 64
     11: UINT32     |        1 | llama.attention.head_count = 32
     12: UINT32     |        1 | llama.attention.head_count_kv = 4
     13: FLOAT32    |        1 | llama.attention.layer_norm_rms_epsilon = 9.999999747378752e-06
     14: UINT32     |        1 | general.file_type = 10
     15: STRING     |        1 | tokenizer.ggml.model = 'llama'
     16: [STRING]   |    32000 | tokenizer.ggml.tokens
     17: [FLOAT32]  |    32000 | tokenizer.ggml.scores
     18: [INT32]    |    32000 | tokenizer.ggml.token_type
     19: UINT32     |        1 | tokenizer.ggml.bos_token_id = 1
     20: UINT32     |        1 | tokenizer.ggml.eos_token_id = 2
     21: UINT32     |        1 | tokenizer.ggml.unknown_token_id = 0
     22: UINT32     |        1 | general.quantization_version = 2
```
So we have an input prompt that is is longer than the context size that this
model was trained on. If we try starting llama-cli with this prompt we get
the following warning and error:
```console

```
