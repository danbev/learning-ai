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

#### CUDA support
If a GPU is available and the CUDA toolkit is installed llama.cpp can be built
with cuBLAS support using the following commands:
```console
$ make clean-llama
$ make llama-cuda
```
For details about installing and configuring CUDA on Fedora 39 see
[egpu.md](../../notes/egpu.md).


To build an example that uses the GPU run:
```console
$ source cuda-env.sh
$ make simple-prompt-cuda
```

You can monitor the gpu usage using:
```console
$ make monitor-gpu
```
This uses the `nvidia-smi` command to monitor the GPU usage and can be closed
using CTRL+C.

### Finetuning
This section describes how to build and configure the finetuning example.

First, the model needs a base-model that will be fine-turned on a specific
task to be downloads:
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
gtrain_opt_callback: iter=     8 sample=33/27520 sched=0.080000 loss=3.031846 dt=00:05:48 eta=02:07:56 |>
train_opt_callback: iter=     9 sample=37/27520 sched=0.090000 loss=2.695413 dt=00:05:48 eta=02:02:08 |-->
save_checkpoint_lora_file: saving to chk-llama-2-7b.Q4_0-shakespeare-10.gguf
save_checkpoint_lora_file: saving to chk-llama-2-7b.Q4_0-shakespeare-LATEST.gguf
save_as_llama_lora: saving to lora-llama-2-7b.Q4_0-shakespeare-10.bin
save_as_llama_lora: saving to lora-llama-2-7b.Q4_0-shakespeare-LATEST.bin
train_opt_callback: iter=    10 sample=41/27520 sched=0.100000 loss=2.781403 dt=00:05:47 eta=01:55:55 |->
train_opt_callback: iter=    11 sample=45/27520 sched=0.110000 loss=2.144866 dt=00:05:58 eta=01:53:24 |-------->
train_opt_callback: iter=    12 sample=49/27520 sched=0.120000 loss=2.300252 dt=00:05:54 eta=01:46:19 |------>
^Ptrain_opt_callback: iter=    13 sample=53/27520 sched=0.130000 loss=2.814412 dt=00:05:54 eta=01:40:23 |->
train_opt_callback: iter=    14 sample=57/27520 sched=0.140000 loss=2.255224 dt=00:05:45 eta=01:32:09 |------->
train_opt_callback: iter=    15 sample=61/27520 sched=0.150000 loss=2.278017 dt=00:05:41 eta=01:25:24 |------>
train_opt_callback: iter=    16 sample=65/27520 sched=0.160000 loss=2.077713 dt=00:05:37 eta=01:18:47 |-------->
train_opt_callback: iter=    17 sample=69/27520 sched=0.170000 loss=2.336711 dt=00:05:38 eta=01:13:18 |------>
train_opt_callback: iter=    18 sample=73/27520 sched=0.180000 loss=2.242626 dt=00:05:39 eta=01:07:57 |------->
train_opt_callback: iter=    19 sample=77/27520 sched=0.190000 loss=2.361472 dt=00:05:40 eta=01:02:21 |------>
save_checkpoint_lora_file: saving to chk-llama-2-7b.Q4_0-shakespeare-20.gguf
save_checkpoint_lora_file: saving to chk-llama-2-7b.Q4_0-shakespeare-LATEST.gguf
save_as_llama_lora: saving to lora-llama-2-7b.Q4_0-shakespeare-20.bin
save_as_llama_lora: saving to lora-llama-2-7b.Q4_0-shakespeare-LATEST.bin
train_opt_callback: iter=    20 sample=81/27520 sched=0.200000 loss=2.328380 dt=00:05:39 eta=00:56:37 |------>
train_opt_callback: iter=    21 sample=85/27520 sched=0.210000 loss=2.167545 dt=00:05:38 eta=00:50:47 |-------->
train_opt_callback: iter=    22 sample=89/27520 sched=0.220000 loss=2.301173 dt=00:05:39 eta=00:45:13 |------>
train_opt_callback: iter=    23 sample=93/27520 sched=0.230000 loss=2.635886 dt=00:05:37 eta=00:39:24 |--->
train_opt_callback: iter=    24 sample=97/27520 sched=0.240000 loss=2.472879 dt=00:05:41 eta=00:34:10 |----->
train_opt_callback: iter=    25 sample=101/27520 sched=0.250000 loss=2.402841 dt=00:05:40 eta=00:28:22 |----->
train_opt_callback: iter=    26 sample=105/27520 sched=0.260000 loss=1.968541 dt=00:05:36 eta=00:22:25 |---------->
train_opt_callback: iter=    27 sample=109/27520 sched=0.270000 loss=2.039970 dt=00:05:37 eta=00:16:53 |--------->
train_opt_callback: iter=    28 sample=113/27520 sched=0.280000 loss=2.072679 dt=00:05:38 eta=00:11:17 |--------->
train_opt_callback: iter=    29 sample=117/27520 sched=0.290000 loss=1.905186 dt=00:05:51 eta=00:05:51 |---------->
save_checkpoint_lora_file: saving to chk-llama-2-7b.Q4_0-shakespeare-30.gguf
save_checkpoint_lora_file: saving to chk-llama-2-7b.Q4_0-shakespeare-LATEST.gguf
save_as_llama_lora: saving to lora-llama-2-7b.Q4_0-shakespeare-30.bin
save_as_llama_lora: saving to lora-llama-2-7b.Q4_0-shakespeare-LATEST.bin
train_opt_callback: iter=    30 sample=121/27520 sched=0.300000 loss=1.846082 dt=00:06:16 eta=0.0ms |----------->
main: total training time: 02:57:51
```
So that took about 3 hours to run on my machine. And the base model I used was
quantized which meant that I ran into a warning when using the model:
```console
$ ./llama.cpp/main -m models/llama-2-7b.Q4_0.gguf \
   --lora lora-llama-2-7b.Q4_0-shakespeare-LATEST.bin \
	-p "Love's fire heats water"

llama_apply_lora_from_file_internal: warning: using a lora adapter with a
quantized model may result in poor quality, use a f16 or f32 base model
with --lora-base

sampling: 
	repeat_last_n = 64, repeat_penalty = 1.100, frequency_penalty = 0.000, presence_penalty = 0.000
	top_k = 40, tfs_z = 1.000, top_p = 0.950, min_p = 0.050, typical_p = 1.000, temp = 0.800
	mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000
sampling order: 
CFG -> Penalties -> top_k -> tfs_z -> typical_p -> top_p -> min_p -> temp 
generate: n_ctx = 512, n_batch = 512, n_predict = -1, n_keep = 0

Love's fire heats water into steam and drives the turbine.nahmong
```
So this is not very accurate and I was hoping for a completion like:
```
Love's fire heats water, water cools not love.
```
I tried using a larger base model, like a Q8, but the time estimated to was
around 15 hours. I think I'll need a GPU for this and one option might be to use
colab but I'm also going to try this using an external GPU (eGPU).
