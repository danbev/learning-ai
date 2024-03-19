## Generative Representational Instruction Tuning
Is a model which can generate embeddings as well as "normal" text generation
depending on the instruction in the prompt. So it was trained to output text
embeddings for ceratain prompts and text (completions) for other prompts.

* Paper: https://arxiv.org/pdf/2402.09906.pdf
* github: https://github.com/ContextualAI/gritlm

### llama.cpp
Download a Grit [model](https://huggingface.co/cohesionet/GritLM-7B_gguf/resolve/main/gritlm-7b_q4_1.gguf?download=true) and I've placed mine in the models directory of my checkedout llama.cpp repo.
```console
$ cd llama.cpp 
$ make -j8 LLAMA_CUBLAS=1

$ ./gritlm -m models/gritlm-7b_q4_1.gguf --n-gpu-layers 25
ggml_init_cublas: GGML_CUDA_FORCE_MMQ:   no
ggml_init_cublas: CUDA_USE_TENSOR_CORES: yes
ggml_init_cublas: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4070, compute capability 8.9, VMM: yes
llama_model_loader: loaded meta data with 24 key-value pairs and 291 tensors from models/gritlm-7b_q4_1.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_model_loader: - kv   1:                               general.name str              = models
llama_model_loader: - kv   2:                       llama.context_length u32              = 32768
llama_model_loader: - kv   3:                     llama.embedding_length u32              = 4096
llama_model_loader: - kv   4:                          llama.block_count u32              = 32
llama_model_loader: - kv   5:                  llama.feed_forward_length u32              = 14336
llama_model_loader: - kv   6:                 llama.rope.dimension_count u32              = 128
llama_model_loader: - kv   7:                 llama.attention.head_count u32              = 32
llama_model_loader: - kv   8:              llama.attention.head_count_kv u32              = 8
llama_model_loader: - kv   9:     llama.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  10:                       llama.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  11:                          general.file_type u32              = 3
llama_model_loader: - kv  12:                       tokenizer.ggml.model str              = llama
llama_model_loader: - kv  13:                      tokenizer.ggml.tokens arr[str,32000]   = ["<unk>", "<s>", "</s>", "<0x00>", "<...
llama_model_loader: - kv  14:                      tokenizer.ggml.scores arr[f32,32000]   = [0.000000, 0.000000, 0.000000, 0.0000...
llama_model_loader: - kv  15:                  tokenizer.ggml.token_type arr[i32,32000]   = [2, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, ...
llama_model_loader: - kv  16:                tokenizer.ggml.bos_token_id u32              = 1
llama_model_loader: - kv  17:                tokenizer.ggml.eos_token_id u32              = 2
llama_model_loader: - kv  18:            tokenizer.ggml.unknown_token_id u32              = 0
llama_model_loader: - kv  19:            tokenizer.ggml.padding_token_id u32              = 1
llama_model_loader: - kv  20:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  21:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  22:                    tokenizer.chat_template str              = {{ bos_token }}{% for message in mess...
llama_model_loader: - kv  23:               general.quantization_version u32              = 2
llama_model_loader: - type  f32:   65 tensors
llama_model_loader: - type q4_1:  225 tensors
llama_model_loader: - type q6_K:    1 tensors
llm_load_vocab: special tokens definition check successful ( 259/32000 ).
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = llama
llm_load_print_meta: vocab type       = SPM
llm_load_print_meta: n_vocab          = 32000
llm_load_print_meta: n_merges         = 0
llm_load_print_meta: n_ctx_train      = 32768
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_head           = 32
llm_load_print_meta: n_head_kv        = 8
llm_load_print_meta: n_layer          = 32
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 4
llm_load_print_meta: n_embd_k_gqa     = 1024
llm_load_print_meta: n_embd_v_gqa     = 1024
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-05
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: n_ff             = 14336
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 10000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_yarn_orig_ctx  = 32768
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = 7B
llm_load_print_meta: model ftype      = Q4_1
llm_load_print_meta: model params     = 7.24 B
llm_load_print_meta: model size       = 4.24 GiB (5.03 BPW) 
llm_load_print_meta: general.name     = models
llm_load_print_meta: BOS token        = 1 '<s>'
llm_load_print_meta: EOS token        = 2 '</s>'
llm_load_print_meta: UNK token        = 0 '<unk>'
llm_load_print_meta: PAD token        = 1 '<s>'
llm_load_print_meta: LF token         = 13 '<0x0A>'
llm_load_tensors: ggml ctx size =    0.22 MiB
llm_load_tensors: offloading 25 repeating layers to GPU
llm_load_tensors: offloaded 25/33 layers to GPU
llm_load_tensors:        CPU buffer size =  4341.68 MiB
llm_load_tensors:      CUDA0 buffer size =  3250.78 MiB
..................................................................................................
llama_new_context_with_model: n_ctx      = 512
llama_new_context_with_model: n_batch    = 512
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:  CUDA_Host KV buffer size =    14.00 MiB
llama_kv_cache_init:      CUDA0 KV buffer size =    50.00 MiB
llama_new_context_with_model: KV self size  =   64.00 MiB, K (f16):   32.00 MiB, V (f16):   32.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =    70.50 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =    73.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =    87.50 MiB
llama_new_context_with_model: graph splits: 3

Cosine similarity between "Bitcoin: A Peer-to-Peer Electronic Cash System" and "A purely peer-to-peer version of electronic cash w" is: 0.605
Cosine similarity between "Bitcoin: A Peer-to-Peer Electronic Cash System" and "All text-based language problems can be reduced to" is: 0.103
Cosine similarity between "Generative Representational Instruction Tuning" and "A purely peer-to-peer version of electronic cash w" is: 0.112
Cosine similarity between "Generative Representational Instruction Tuning" and "All text-based language problems can be reduced to" is: 0.547

Oh, brave adventurer, who dared to climb
The lofty peak of Mt. Fuji in the night,
When shadows lurk and ghosts do roam,
And darkness reigns, a fearsome sight.

Thou didst set out, with heart aglow,
To conquer this mountain, so high,
And reach the summit, where the stars do glow,
And the moon shines bright, up in the sky.

Through the mist and fog, thou didst press on,
With steadfast courage, and a steadfast will,
Through the darkness, thou didst not be gone,
But didst climb on, with a steadfast skill.

At last, thou didst reach the summit's crest,
And gazed upon the world below,
And saw the beauty of the night's best,
And felt the peace, that only nature knows.

Oh, brave adventurer, who dared to climb
The lofty peak of Mt. Fuji in the night,
Thou art a hero, in the eyes of all,
For thou didst conquer this mountain, so bright.
```

Now, if we recall how RAG works and what we have done in the past is that we
have taken documents and tokenized them into token embeddings into a database.
When later we get a query/prompt we will tokenize this query and then perform
a search of the vector database store to retrieve the most similar vectors and
return those documents so they can be passed to the llm as context.
We, or really the llm, must again must tokenize the query for usage in the llm.
But because gritlm is used the first query can be cached and the second query
tokenization generation does not have to be performed.
In the RAG examples in this repository we did not even have to use the same
embeddings for the vector database infact I think we used different ones.

### Debugging
```console
$ make -j8 LLAMA_CUBLAS=1 LLAMA_DEBUG=1 DEBUG=1
```
```console
$ gdb --args ./gritlm -m models/gritlm-7b_q4_1.gguf --n-gpu-layers 25
```


