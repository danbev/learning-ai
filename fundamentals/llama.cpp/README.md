## LLama.ccp exploration/example project
This project is an exploration of the LLama.cpp library. The goal it to have
small isolated examples that can be run and debugged in isolation to help
understand how the library works.

#### Initial setup
To update the submodule run:
```console
$ git submodule update --recursive --remote
```

Updating to the latest upstream llama.cpp:
```console
$ git submodule update --remote --merge
```

### Debugging
The examples in this project can be build with debug symbols enabled allowing
for exploration of the llama.cpp, and ggml.cpp libraries. For example:
```console
$ gdb --args ./simple-prompt 
Reading symbols from ./simple-prompt...

(gdb) br simple-prompt.cpp:7
Breakpoint 1 at 0x410bff: file src/simple-prompt.cpp, line 7.

(gdb) r
Starting program: /home/danielbevenius/work/ai/learning-ai/fundamentals/llama.cpp/simple-prompt 
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib64/libthread_db.so.1".

Breakpoint 1, main (argc=1, argv=0x7fffffffc118) at src/simple-prompt.cpp:7
warning: Source file is more recent than executable.
7	    gpt_params params;
```

### ctags
To generate ctags for the project run:
```console
$ ctags -R --languages=C++ --c++-kinds=+p --fields=+iaS --extra=+q .
```

#### Configuration
This project uses a git submodule to include the LLama.cpp library. To
add the submodule run (only the first time and I'm just adding this for
documenation purposes):
```console
$ git submodule add https://github.com/ggerganov/llama.cpp llama.cpp
$ git submodule update --init --recursive
```

### Finetuning
This section describes how to build and configure the finetuning example.

First, the model needs to be downloads:
```console
$ make download-llama-model
```
And the we also need to download the text that the model will be train on:
```console
$ make download-shakespeare
```
After that we can fine tune the model:
```console
$ make finetune-llama-2-7b.Q4_0-model 
./finetune \
        --model-base models/llama-2-7b.Q4_0.gguf \
        --checkpoint-in chk-llama-2-7b.Q4_0-shakespeare-LATEST.gguf \
        --checkpoint-out chk-llama-2-7b.Q4_0-shakespeare-ITERATION.gguf \
        --lora-out lora-llama-2-7b.Q4_0-shakespeare-ITERATION.bin \
        --train-data "shakespeare.txt" \
        --save-every 10 \
        --threads 6 --adam-iter 30 --batch 4 --ctx 64 \
        --use-checkpointing
main: seed: 1703670912
main: model base = 'models/llama-2-7b.Q4_0.gguf'
llama_model_loader: loaded meta data with 19 key-value pairs and 291 tensors from models/llama-2-7b.Q4_0.gguf (version GGUF V2)
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_model_loader: - kv   1:                               general.name str              = LLaMA v2
llama_model_loader: - kv   2:                       llama.context_length u32              = 4096
llama_model_loader: - kv   3:                     llama.embedding_length u32              = 4096
llama_model_loader: - kv   4:                          llama.block_count u32              = 32
llama_model_loader: - kv   5:                  llama.feed_forward_length u32              = 11008
llama_model_loader: - kv   6:                 llama.rope.dimension_count u32              = 128
llama_model_loader: - kv   7:                 llama.attention.head_count u32              = 32
llama_model_loader: - kv   8:              llama.attention.head_count_kv u32              = 32
llama_model_loader: - kv   9:     llama.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  10:                          general.file_type u32              = 2
llama_model_loader: - kv  11:                       tokenizer.ggml.model str              = llama
llama_model_loader: - kv  12:                      tokenizer.ggml.tokens arr[str,32000]   = ["<unk>", "<s>", "</s>", "<0x00>", "<...
llama_model_loader: - kv  13:                      tokenizer.ggml.scores arr[f32,32000]   = [0.000000, 0.000000, 0.000000, 0.0000...
llama_model_loader: - kv  14:                  tokenizer.ggml.token_type arr[i32,32000]   = [2, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, ...
llama_model_loader: - kv  15:                tokenizer.ggml.bos_token_id u32              = 1
llama_model_loader: - kv  16:                tokenizer.ggml.eos_token_id u32              = 2
llama_model_loader: - kv  17:            tokenizer.ggml.unknown_token_id u32              = 0
llama_model_loader: - kv  18:               general.quantization_version u32              = 2
llama_model_loader: - type  f32:   65 tensors
llama_model_loader: - type q4_0:  225 tensors
llama_model_loader: - type q6_K:    1 tensors
llm_load_vocab: special tokens definition check successful ( 259/32000 ).
llm_load_print_meta: format           = GGUF V2
llm_load_print_meta: arch             = llama
llm_load_print_meta: vocab type       = SPM
llm_load_print_meta: n_vocab          = 32000
llm_load_print_meta: n_merges         = 0
llm_load_print_meta: n_ctx_train      = 4096
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_head           = 32
llm_load_print_meta: n_head_kv        = 32
llm_load_print_meta: n_layer          = 32
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_gqa            = 1
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-05
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: n_ff             = 11008
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 10000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_yarn_orig_ctx  = 4096
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: model type       = 7B
llm_load_print_meta: model ftype      = Q4_0
llm_load_print_meta: model params     = 6.74 B
llm_load_print_meta: model size       = 3.56 GiB (4.54 BPW) 
llm_load_print_meta: general.name     = LLaMA v2
llm_load_print_meta: BOS token        = 1 '<s>'
llm_load_print_meta: EOS token        = 2 '</s>'
llm_load_print_meta: UNK token        = 0 '<unk>'
llm_load_print_meta: LF token         = 13 '<0x0A>'
llm_load_tensors: ggml ctx size       =    0.11 MiB
llm_load_tensors: system memory used  = 3647.98 MiB
..................................................................................................
llama_new_context_with_model: n_ctx      = 512
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 1
llama_new_context_with_model: KV self size  =  256.00 MiB, K (f16):  128.00 MiB, V (f16):  128.00 MiB
llama_build_graph: non-view tensors processed: 676/676
llama_new_context_with_model: compute buffer total size = 73.69 MiB
main: init model
print_params: n_vocab:   32000
print_params: n_ctx:     64
print_params: n_embd:    4096
print_params: n_ff:      11008
print_params: n_head:    32
print_params: n_head_kv: 32
print_params: n_layer:   32
print_params: norm_rms_eps          : 0.000010
print_params: rope_freq_base        : 10000.000000
print_params: rope_freq_scale       : 1.000000
print_lora_params: n_rank_attention_norm : 1
print_lora_params: n_rank_wq             : 4
print_lora_params: n_rank_wk             : 4
print_lora_params: n_rank_wv             : 4
print_lora_params: n_rank_wo             : 4
print_lora_params: n_rank_ffn_norm       : 1
print_lora_params: n_rank_w1             : 4
print_lora_params: n_rank_w2             : 4
print_lora_params: n_rank_w3             : 4
print_lora_params: n_rank_tok_embeddings : 4
print_lora_params: n_rank_norm           : 1
print_lora_params: n_rank_output         : 4
main: total train_iterations 0
main: seen train_samples     0
main: seen train_tokens      0
main: completed train_epochs 0
main: lora_size = 84863776 bytes (80.9 MB)
main: opt_size  = 126593008 bytes (120.7 MB)
main: opt iter 0
main: input_size = 32769056 bytes (31.3 MB)
main: compute_size = 4353442144 bytes (4151.8 MB)
main: evaluation order = RIGHT_TO_LEFT
main: tokenize training data
tokenize_file: total number of samples: 27520
main: number of training tokens: 27584
main: number of unique tokens: 3069
main: train data seems to have changed. restarting shuffled epoch.
main: begin training
main: work_size = 768376 bytes (0.7 MB)
train_opt_callback: iter=     0 sample=1/27520 sched=0.000000 loss=0.000000 |->
train_opt_callback: iter=     1 sample=5/27520 sched=0.010000 loss=2.823248 dt=00:04:46 eta=02:18:26 |->
train_opt_callback: iter=     2 sample=9/27520 sched=0.020000 loss=3.138579 dt=00:05:12 eta=02:26:03 |>
train_opt_callback: iter=     3 sample=13/27520 sched=0.030000 loss=2.933344 dt=00:05:31 eta=02:29:11 |>
train_opt_callback: iter=     4 sample=17/27520 sched=0.040000 loss=2.046678 dt=00:05:40 eta=02:27:28 |--------->
train_opt_callback: iter=     5 sample=21/27520 sched=0.050000 loss=2.475715 dt=00:05:43 eta=02:23:17 |---->
train_opt_callback: iter=     6 sample=25/27520 sched=0.060000 loss=2.484926 dt=00:05:46 eta=02:18:47 |---->
:train_opt_callback: iter=     7 sample=29/27520 sched=0.070000 loss=2.131633 dt=00:05:53 eta=02:15:21 |-------->
```
