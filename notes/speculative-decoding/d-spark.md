### D-Spark
This was part of Deepseek V4 series and is a speculative decoding implementation.

Paper:  
https://github.com/deepseek-ai/DeepSpec/blob/main/DSpark_paper.pdf

The speculative decoding uses a draft model. So this is not anything new, we have
seen multiple variants of speculative decoding with draft models in this directory
before. So what is different with d-spark?  

D-Spark is similar (an extension) to [d-flash](./d-flash.md) in that it does a
block of decoding in parallel, compared to Eagle which does autoregressive
decoding (sequential one after the other).

There is an issue with D-Flash which is inherent to the parallel processing which
is when we do parallel decoding we are predicting the complete "sentence" in one
go. There is no way for any token to know about the token prior to it or after
it as they get generated at the same time. 
A token generated at position 3 has no idea what token was just generated at
position 2. This is called lack of "inter-token dependencies" and creates an issue
called 'multi-modal collision'. This has nothing to do with multi-modal models
but it about the 'mode' or 'path' that the sequence of tokens could follow.
A multi-modal collision happens because the parallel drafter is evaluating all
of these possible paths simultaneously without coordinating between the token
positions.

Because the model predicts each position independently and averages out all the
possible past tokens rather than locking into a single specific path, it outputs
the incoherent, mixed-up combination "of problem". The different valid paths
(modes) literally collide with one another. This results in what the paper calls
"inconsistent suffix combinations," which causes the quality of the draft tokens
to rapidly decay the further down the sequence you go (called suffix decay I
think).

### Parallel block and sequential block
So lets say we have an initial prompt, this will be processed by the target
model and it will produce predicted next token.
```
Target model:

prompt -> target model -> predicted next token (anchor token)

Parallel block:
                                                        -> base_logit_0, hidden_state_0
[anchor token + mask + mask + mask] -> parallel block   -> base_logit_1, hidden_state_1
                                                        -> base_logit_2, hidden_state_2
                                                        -> base_logit_3, hidden_state_3
```
Notice that the parallel block produces logits (raw scores for the whole language
vocabulary, and the hidden states (the last layer of the tranformer for each
token). The logits will be used in the sequential block, next in the diagram,
and the hidden states are used after the sequential block.

```console
Sequential block:

anchor token
    ↓
sequential block  ← base_logit_0
    ↓
transitioned logit_0  ➔  [ Sample Draft Token 0 ]  (Saved for final output)
     +-----------------------------+
     ↓             
sequential block  ← base_logit_1
     ↓             
transitioned logit_1  ➔  [ Sample Draft Token 1 ]  (Saved for final output)
     +-----------------------------+
     ↓             
sequential block  ← base_logit_2
     ↓             
transitioned logit_2  ➔  [ Sample Draft Token 2 ]  (Saved for final output)
     +-----------------------------+
     ↓             
sequential block  ← base_logit_3
     ↓             
transitioned logit_3  ➔  [ Sample Draft Token 3 ]  (Saved for final output)

Output:
Tokens: [Draft Token 0, Draft Token 1, Draft Token 2, Draft Token 3]

[Draft Token 0, Draft Token 1, Draft Token 2, Draft Token 3]
     ↓               ↓             ↓            ↓
                    Confidence Head
     ↓               ↓             ↓            ↓
[Conf Score 0,  Conf Score 1,  Conf Score 2,  Conf Score 3 ]


Tokens: [Draft Token 0, Draft Token 1, Draft Token 2, Draft Token 3]
Scores: [Conf Score 0,  Conf Score 1,  Conf Score 2,  Conf Score 3 ]
```
What D-Spark does is it adds a sequential layer after the parallel block. So the
parallel block will output logits and also the hidden states for the predicted
tokens. And recall that the hidden states could just be the last layer of the
transformer, and the logits are the output of running the last hidden state
through the lm-head projection, going from the internal hidden space to the
token vocabulary score.

### Confidence head
It takes two pieces of information, the draft token and the conf score and
stitches them together into a single vector, multiplies that vector by its
learned matrix, and then passes the result through a sigmoid function to squash
the final number into a probability between 0 and 1
This matrix has learned how to accurately score the quality of its own guesses.
While this learned matrix is great at ranking which tokens are good and which
are bad, its absolute probability numbers are usually too high

### Transisition bias
So a transisition in probablitiy and statistics simply describes the likeihood
of moving from one specific state to another. An example of this is a Markov chain
where the next state depends only on the current state. In DSpark the default
sequential block is a Markov head, which looks at the token we just generated,
which is the anchor token for from the target model, or a logits vector from
the paralllel block. It calculates the probability of transistioning to the
next logical token.

Recall that the parallel block produces "base logits" (raw vocabulary scores)
independently, meaning those scores have no idea what the preceding tokens are.
The sequential stage supplements these independent base logits by mathematically
adding the transition bias directly to them.

Lets say the parallel block produces logits for "of course" and "no problem" and
that it give both "course", and "problem" a high score. And lets say that the
sequential block choose "of" for position 1. The Markov head will look up 'of'
in a small embedding table and generates a transistion bias vector, which is a
set of adjustment numbers, and adds it to the base logits. It literally adds a
positive number to the score for "course" (boosting it) and adds a negative
number to the score for "problem" (suppressing it.

### Nemotron 3.5 Lightning DSpark issue
I'm looking into this using NVIDIA Nemotron 3.5 Lightning and running into the
following error:
```console
0.02.977.124 E llama_model_load: error loading model: check_tensor_dims: tensor 'conf_proj.weight' not found
0.02.977.147 E llama_model_load_from_file_impl: failed to load model
```
```console
(gdb) catch throw
(gdb) r
0.01.014.304 D create_tensor: loading tensor token_embd.weight
0.01.014.380 D create_tensor: loading tensor markov_w1.weight
0.01.014.486 D create_tensor: loading tensor markov_w2.weight
0.01.014.498 D create_tensor: loading tensor conf_proj.weight

Thread 1 "llama-server" hit Catchpoint 1 (exception thrown), 0x0000fffff37da8c0 in __cxa_throw ()
   from /lib/aarch64-linux-gnu/libstdc++.so.6
(gdb) up
#1  0x0000fffff1bc6444 [PAC] in llama_model_loader::check_tensor_dims (this=0xffffffff4d30,
    name="conf_proj.weight", ne=std::vector of length 2, capacity 2 = {...}, required=true, allow_reshape=false)
    at /home/danbev/work/llama.cpp/src/llama-model-loader.cpp:871
871	        throw std::runtime_error(format("%s: tensor '%s' not found", __func__, name.c_str()));

(gdb) up
#2  0x0000fffff1bc86f0 in llama_model_loader::create_tensor (this=0xffffffff4d20, hparams=..., 
    buft_list_cpu=0xaaaaabbcaa78, buft_list_input=0xaaaaabbcaa78, buft_list_output=0xaaaaaf313dc8, 
    buft_list_layer=0x0, tn=..., ne=std::initializer_list of length 2 = {...}, flags=0)
    at /home/danbev/work/llama.cpp/src/llama-model-loader.cpp:1272
1272	    const struct ggml_tensor * cur = check_tensor_dims(tn.str(), ne, !(flags & TENSOR_NOT_REQUIRED), flags & TENSOR_ALLOW_RESHAPE);

```
Lets take a look at where this loading is coming from:
```console
(gdb) bt
#0  0x0000fffff37da8c0 in __cxa_throw () from /lib/aarch64-linux-gnu/libstdc++.so.6
#1  0x0000fffff1bc6444 [PAC] in llama_model_loader::check_tensor_dims (this=0xffffffff4d20,
    name="conf_proj.weight", ne=std::vector of length 2, capacity 2 = {...}, required=true, allow_reshape=false)
    at /home/danbev/work/llama.cpp/src/llama-model-loader.cpp:871
#2  0x0000fffff1bc86f0 in llama_model_loader::create_tensor (this=0xffffffff4d20, hparams=...,
    buft_list_cpu=0xaaaaabbcaa78, buft_list_input=0xaaaaabbcaa78, buft_list_output=0xaaaaaf313dc8,
    buft_list_layer=0x0, tn=..., ne=std::initializer_list of length 2 = {...}, flags=0)
    at /home/danbev/work/llama.cpp/src/llama-model-loader.cpp:1272
#3  0x0000fffff1c03dac in llama_model_base::create_tensor (this=0xaaaaabf1c090, ml=..., tn=...,
    ne=std::initializer_list of length 2 = {...}, flags=0) at /home/danbev/work/llama.cpp/src/llama-model.cpp:1698
#4  0x0000fffff1c09140 in llama_model_base::create_tensor (this=0xaaaaabf1c090, tn=...,
    ne=std::initializer_list of length 2 = {...}, flags=0) at /home/danbev/work/llama.cpp/src/llama-model.cpp:2911
#5  0x0000fffff1d1fb98 in llama_model_dflash::load_arch_tensors (this=0xaaaaabf1c090)
    at /home/danbev/work/llama.cpp/src/models/dflash.cpp:94
#6  0x0000fffff1c01684 in llama_model_base::load_tensors (this=0xaaaaabf1c090, ml=...)
    at /home/danbev/work/llama.cpp/src/llama-model.cpp:1388
```
Lets looks at frame 5:
```console
(gdb) f 5
(gdb) f 5
#5  0x0000fffff1d1fb98 in llama_model_dflash::load_arch_tensors (this=0xaaaaabf1c090)
    at /home/danbev/work/llama.cpp/src/models/dflash.cpp:94
94	        dspark_conf_proj   = create_tensor(tn(LLM_TENSOR_DSPARK_CONF_PROJ, "weight"), { n_embd + dspark_markov_rank, 1 }, 0);
```

The dspark draft model that I converted (which might be incorrectly converted but
this is what I'm trying to figure out), looke like this:
```console
(venv) spark $ gguf-dump ../convert/upload-NVIDIA_Nemotron_3.5_Lightning_30B_A3B/dspark-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4.gguf
INFO:gguf-dump:* Loading: ../convert/upload-NVIDIA_Nemotron_3.5_Lightning_30B_A3B/dspark-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4.gguf
* File is LITTLE endian, script is running on a LITTLE endian host.
* Dumping 45 key/value pair(s)
      1: UINT32     |        1 | GGUF.version = 3
      2: UINT64     |        1 | GGUF.tensor_count = 97
      3: UINT64     |        1 | GGUF.kv_count = 42
      4: STRING     |        1 | general.architecture = 'dflash'
      5: STRING     |        1 | general.type = 'model'
      6: STRING     |        1 | general.name = 'NVIDIA-Nemotron-3.5-Lightning-30B-A3B'
      7: STRING     |        1 | general.size_label = '615M'
      8: STRING     |        1 | general.license = 'other'
      9: STRING     |        1 | general.license.name = 'openmdw-1.1'
     10: STRING     |        1 | general.license.link = 'https://openmdw.ai/license/1-1/'
     11: UINT32     |        1 | general.base_model.count = 2
     12: STRING     |        1 | general.base_model.0.name = 'NVIDIA Nemotron 3.5 Lightning 30B A3B BF16'
     13: STRING     |        1 | general.base_model.0.organization = 'Nvidia'
     14: STRING     |        1 | general.base_model.0.repo_url = 'https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightni...'
     15: STRING     |        1 | general.base_model.1.name = 'NVIDIA Nemotron 3.5 Lightning 30B A3B NVFP4'
     16: STRING     |        1 | general.base_model.1.organization = 'Nvidia'
     17: STRING     |        1 | general.base_model.1.repo_url = 'https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightni...'
     18: [STRING]   |        7 | general.tags = ['nvidia', 'ModelOpt', 'Nemotron-3.5-Lightning', 'latent-moe', 'mtp', 'DSpark', ...]
     19: UINT32     |        1 | dflash.block_count = 6
     20: UINT32     |        1 | dflash.context_length = 1048576
     21: UINT32     |        1 | dflash.embedding_length = 2688
     22: UINT32     |        1 | dflash.feed_forward_length = 6144
     23: UINT32     |        1 | dflash.attention.head_count = 32
     24: UINT32     |        1 | dflash.attention.head_count_kv = 2
     25: FLOAT32    |        1 | dflash.rope.freq_base = 10000.0
     26: FLOAT32    |        1 | dflash.attention.layer_norm_rms_epsilon = 9.999999974752427e-07
     27: UINT32     |        1 | dflash.attention.key_length = 128
     28: UINT32     |        1 | dflash.attention.value_length = 128
     29: UINT32     |        1 | general.file_type = 39
     30: UINT32     |        1 | dflash.block_size = 8
     31: [INT32]    |        6 | dflash.target_layers = [2, 6, 20, 30, 42, 52]
     32: UINT32     |        1 | general.quantization_version = 2
     33: STRING     |        1 | tokenizer.ggml.model = 'gpt2'
     34: STRING     |        1 | tokenizer.ggml.pre = 'pixtral'
     35: [STRING]   |   131072 | tokenizer.ggml.tokens = ['<unk>', '<s>', '</s>', '[INST]', '[/INST]', '[AVAILABLE_TOOLS]', ...]
     36: [INT32]    |   131072 | tokenizer.ggml.token_type = [3, 3, 3, 3, 3, 3, ...]
     37: [STRING]   |   269443 | tokenizer.ggml.merges = ['Ġ Ġ', 'Ġ t', 'e r', 'i n', 'Ġ ĠĠĠ', 'ĠĠ ĠĠ', ...]
     38: UINT32     |        1 | tokenizer.ggml.bos_token_id = 1
     39: UINT32     |        1 | tokenizer.ggml.eos_token_id = 11
     40: UINT32     |        1 | tokenizer.ggml.unknown_token_id = 0
     41: UINT32     |        1 | tokenizer.ggml.padding_token_id = 11
     42: BOOL       |        1 | tokenizer.ggml.add_bos_token = False
     43: BOOL       |        1 | tokenizer.ggml.add_eos_token = False
     44: STRING     |        1 | tokenizer.chat_template = '{% macro render_extra_keys(json_dict, handled_keys) %}\n  ...'
     45: UINT32     |        1 | tokenizer.ggml.mask_token_id = 990
* Dumping 97 tensor(s)
      1:   43352064 | 16128,  2688,     1,     1 | NVFP4   | fc.weight
      2:          1 |     1,     1,     1,     1 | F32     | fc.scale
      3:   16515072 |  6144,  2688,     1,     1 | NVFP4   | blk.0.ffn_down.weight
      4:          1 |     1,     1,     1,     1 | F32     | blk.0.ffn_down.scale
      5:   16515072 |  2688,  6144,     1,     1 | NVFP4   | blk.0.ffn_gate.weight
      6:          1 |     1,     1,     1,     1 | F32     | blk.0.ffn_gate.scale
      7:   16515072 |  2688,  6144,     1,     1 | NVFP4   | blk.0.ffn_up.weight
      8:          1 |     1,     1,     1,     1 | F32     | blk.0.ffn_up.scale
      9:   16515072 |  6144,  2688,     1,     1 | NVFP4   | blk.1.ffn_down.weight
     10:          1 |     1,     1,     1,     1 | F32     | blk.1.ffn_down.scale
     11:   16515072 |  2688,  6144,     1,     1 | NVFP4   | blk.1.ffn_gate.weight
     12:          1 |     1,     1,     1,     1 | F32     | blk.1.ffn_gate.scale
     13:   16515072 |  2688,  6144,     1,     1 | NVFP4   | blk.1.ffn_up.weight
     14:          1 |     1,     1,     1,     1 | F32     | blk.1.ffn_up.scale
     15:   16515072 |  6144,  2688,     1,     1 | NVFP4   | blk.2.ffn_down.weight
     16:          1 |     1,     1,     1,     1 | F32     | blk.2.ffn_down.scale
     17:   16515072 |  2688,  6144,     1,     1 | NVFP4   | blk.2.ffn_gate.weight
     18:          1 |     1,     1,     1,     1 | F32     | blk.2.ffn_gate.scale
     19:   16515072 |  2688,  6144,     1,     1 | NVFP4   | blk.2.ffn_up.weight
     20:          1 |     1,     1,     1,     1 | F32     | blk.2.ffn_up.scale
     21:   16515072 |  6144,  2688,     1,     1 | NVFP4   | blk.3.ffn_down.weight
     22:          1 |     1,     1,     1,     1 | F32     | blk.3.ffn_down.scale
     23:   16515072 |  2688,  6144,     1,     1 | NVFP4   | blk.3.ffn_gate.weight
     24:          1 |     1,     1,     1,     1 | F32     | blk.3.ffn_gate.scale
     25:   16515072 |  2688,  6144,     1,     1 | NVFP4   | blk.3.ffn_up.weight
     26:          1 |     1,     1,     1,     1 | F32     | blk.3.ffn_up.scale
     27:   16515072 |  6144,  2688,     1,     1 | NVFP4   | blk.4.ffn_down.weight
     28:          1 |     1,     1,     1,     1 | F32     | blk.4.ffn_down.scale
     29:   16515072 |  2688,  6144,     1,     1 | NVFP4   | blk.4.ffn_gate.weight
     30:          1 |     1,     1,     1,     1 | F32     | blk.4.ffn_gate.scale
     31:   16515072 |  2688,  6144,     1,     1 | NVFP4   | blk.4.ffn_up.weight
     32:          1 |     1,     1,     1,     1 | F32     | blk.4.ffn_up.scale
     33:   16515072 |  6144,  2688,     1,     1 | NVFP4   | blk.5.ffn_down.weight
     34:          1 |     1,     1,     1,     1 | F32     | blk.5.ffn_down.scale
     35:   16515072 |  2688,  6144,     1,     1 | NVFP4   | blk.5.ffn_gate.weight
     36:          1 |     1,     1,     1,     1 | F32     | blk.5.ffn_gate.scale
     37:   16515072 |  2688,  6144,     1,     1 | NVFP4   | blk.5.ffn_up.weight
     38:          1 |     1,     1,     1,     1 | F32     | blk.5.ffn_up.scale
     39:   67108864 |   512,131072,     1,     1 | NVFP4   | markov_w2.weight
     40:          1 |     1,     1,     1,     1 | F32     | markov_w2.scale
     41:       2688 |  2688,     1,     1,     1 | F32     | enc.output_norm.weight
     42:       2688 |  2688,     1,     1,     1 | F32     | blk.0.attn_norm.weight
     43:       2688 |  2688,     1,     1,     1 | F32     | blk.0.ffn_norm.weight
     44:         32 |    32,     1,     1,     1 | F32     | blk.0.attn_sinks
     45:        128 |   128,     1,     1,     1 | F32     | blk.0.attn_k_norm.weight
     46:     688128 |  2688,   256,     1,     1 | BF16    | blk.0.attn_k.weight
     47:   11010048 |  4096,  2688,     1,     1 | BF16    | blk.0.attn_output.weight
     48:        128 |   128,     1,     1,     1 | F32     | blk.0.attn_q_norm.weight
     49:   11010048 |  2688,  4096,     1,     1 | BF16    | blk.0.attn_q.weight
     50:     688128 |  2688,   256,     1,     1 | BF16    | blk.0.attn_v.weight
     51:       2688 |  2688,     1,     1,     1 | F32     | blk.1.attn_norm.weight
     52:       2688 |  2688,     1,     1,     1 | F32     | blk.1.ffn_norm.weight
     53:         32 |    32,     1,     1,     1 | F32     | blk.1.attn_sinks
     54:        128 |   128,     1,     1,     1 | F32     | blk.1.attn_k_norm.weight
     55:     688128 |  2688,   256,     1,     1 | BF16    | blk.1.attn_k.weight
     56:   11010048 |  4096,  2688,     1,     1 | BF16    | blk.1.attn_output.weight
     57:        128 |   128,     1,     1,     1 | F32     | blk.1.attn_q_norm.weight
     58:   11010048 |  2688,  4096,     1,     1 | BF16    | blk.1.attn_q.weight
     59:     688128 |  2688,   256,     1,     1 | BF16    | blk.1.attn_v.weight
     60:       2688 |  2688,     1,     1,     1 | F32     | blk.2.attn_norm.weight
     61:       2688 |  2688,     1,     1,     1 | F32     | blk.2.ffn_norm.weight
     62:         32 |    32,     1,     1,     1 | F32     | blk.2.attn_sinks
     63:        128 |   128,     1,     1,     1 | F32     | blk.2.attn_k_norm.weight
     64:     688128 |  2688,   256,     1,     1 | BF16    | blk.2.attn_k.weight
     65:   11010048 |  4096,  2688,     1,     1 | BF16    | blk.2.attn_output.weight
     66:        128 |   128,     1,     1,     1 | F32     | blk.2.attn_q_norm.weight
     67:   11010048 |  2688,  4096,     1,     1 | BF16    | blk.2.attn_q.weight
     68:     688128 |  2688,   256,     1,     1 | BF16    | blk.2.attn_v.weight
     69:       2688 |  2688,     1,     1,     1 | F32     | blk.3.attn_norm.weight
     70:       2688 |  2688,     1,     1,     1 | F32     | blk.3.ffn_norm.weight
     71:         32 |    32,     1,     1,     1 | F32     | blk.3.attn_sinks
     72:        128 |   128,     1,     1,     1 | F32     | blk.3.attn_k_norm.weight
     73:     688128 |  2688,   256,     1,     1 | BF16    | blk.3.attn_k.weight
     74:   11010048 |  4096,  2688,     1,     1 | BF16    | blk.3.attn_output.weight
     75:        128 |   128,     1,     1,     1 | F32     | blk.3.attn_q_norm.weight
     76:   11010048 |  2688,  4096,     1,     1 | BF16    | blk.3.attn_q.weight
     77:     688128 |  2688,   256,     1,     1 | BF16    | blk.3.attn_v.weight
     78:       2688 |  2688,     1,     1,     1 | F32     | blk.4.attn_norm.weight
     79:       2688 |  2688,     1,     1,     1 | F32     | blk.4.ffn_norm.weight
     80:         32 |    32,     1,     1,     1 | F32     | blk.4.attn_sinks
     81:        128 |   128,     1,     1,     1 | F32     | blk.4.attn_k_norm.weight
     82:     688128 |  2688,   256,     1,     1 | BF16    | blk.4.attn_k.weight
     83:   11010048 |  4096,  2688,     1,     1 | BF16    | blk.4.attn_output.weight
     84:        128 |   128,     1,     1,     1 | F32     | blk.4.attn_q_norm.weight
     85:   11010048 |  2688,  4096,     1,     1 | BF16    | blk.4.attn_q.weight
     86:     688128 |  2688,   256,     1,     1 | BF16    | blk.4.attn_v.weight
     87:       2688 |  2688,     1,     1,     1 | F32     | blk.5.attn_norm.weight
     88:       2688 |  2688,     1,     1,     1 | F32     | blk.5.ffn_norm.weight
     89:         32 |    32,     1,     1,     1 | F32     | blk.5.attn_sinks
     90:        128 |   128,     1,     1,     1 | F32     | blk.5.attn_k_norm.weight
     91:     688128 |  2688,   256,     1,     1 | BF16    | blk.5.attn_k.weight
     92:   11010048 |  4096,  2688,     1,     1 | BF16    | blk.5.attn_output.weight
     93:        128 |   128,     1,     1,     1 | F32     | blk.5.attn_q_norm.weight
     94:   11010048 |  2688,  4096,     1,     1 | BF16    | blk.5.attn_q.weight
     95:     688128 |  2688,   256,     1,     1 | BF16    | blk.5.attn_v.weight
     96:   67108864 |   512,131072,     1,     1 | BF16    | markov_w1.weight
     97:       2688 |  2688,     1,     1,     1 | F32     | output_norm.weight
```

_wip_
