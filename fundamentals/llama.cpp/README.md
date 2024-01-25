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
with cuBLAS (CUDA Basic Linear Algebra Subroutine) support using the following
commands:
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

### GPU finetuning
With the GPU I've got I re-ran the above training, but also updated the base
model to be a Q8 model:
```console
$ make finetune LLAMA_CUBLAS=1
$ make finetune-llama-2-7b.Q8_0-model
~/work/ai/llama.cpp/finetune \
        --model-base models/llama-2-7b.Q8_0.gguf \
        --checkpoint-in chk-llama-2-7b.Q8_0-shakespeare-LATEST.gguf \
        --checkpoint-out chk-llama-2-7b.Q8_0-shakespeare-ITERATION.gguf \
        --lora-out lora-llama-2-7b.Q8_0-shakespeare-ITERATION.bin \
        --train-data "data/shakespeare.txt" \
        --save-every 10 \
        --threads 6 --adam-iter 30 --batch 4 --ctx 64 \
        --use-checkpointing
main: seed: 1704957459
main: model base = 'models/llama-2-7b.Q8_0.gguf'

ggml_init_cublas: GGML_CUDA_FORCE_MMQ:   no
ggml_init_cublas: CUDA_USE_TENSOR_CORES: yes
ggml_init_cublas: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4070, compute capability 8.9, VMM: yes

llama_model_loader: loaded meta data with 19 key-value pairs and 291 tensors from models/llama-2-7b.Q8_0.gguf (version GGUF V2)
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
llama_model_loader: - kv  10:                          general.file_type u32              = 7
llama_model_loader: - kv  11:                       tokenizer.ggml.model str              = llama
llama_model_loader: - kv  12:                      tokenizer.ggml.tokens arr[str,32000]   = ["<unk>", "<s>", "</s>", "<0x00>", "<...
llama_model_loader: - kv  13:                      tokenizer.ggml.scores arr[f32,32000]   = [0.000000, 0.000000, 0.000000, 0.0000...
llama_model_loader: - kv  14:                  tokenizer.ggml.token_type arr[i32,32000]   = [2, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, ...
llama_model_loader: - kv  15:                tokenizer.ggml.bos_token_id u32              = 1
llama_model_loader: - kv  16:                tokenizer.ggml.eos_token_id u32              = 2
llama_model_loader: - kv  17:            tokenizer.ggml.unknown_token_id u32              = 0
llama_model_loader: - kv  18:               general.quantization_version u32              = 2
llama_model_loader: - type  f32:   65 tensors
llama_model_loader: - type q8_0:  226 tensors
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
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 1
llm_load_print_meta: n_embd_k_gqa     = 4096
llm_load_print_meta: n_embd_v_gqa     = 4096
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
llm_load_print_meta: model ftype      = Q8_0
llm_load_print_meta: model params     = 6.74 B
llm_load_print_meta: model size       = 6.67 GiB (8.50 BPW) 
llm_load_print_meta: general.name     = LLaMA v2
llm_load_print_meta: BOS token        = 1 '<s>'
llm_load_print_meta: EOS token        = 2 '</s>'
llm_load_print_meta: UNK token        = 0 '<unk>'
llm_load_print_meta: LF token         = 13 '<0x0A>'
llm_load_tensors: ggml ctx size       =    0.11 MiB

llm_load_tensors: using CUDA for GPU acceleration
llm_load_tensors: system memory used  = 6828.75 MiB
llm_load_tensors: offloading 0 repeating layers to GPU
llm_load_tensors: offloaded 0/33 layers to GPU

....................................................................................................
llama_new_context_with_model: n_ctx      = 512
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 1
llama_new_context_with_model: KV self size  =  256.00 MiB, K (f16):  128.00 MiB, V (f16):  128.00 MiB
llama_build_graph: non-view tensors processed: 676/676
llama_new_context_with_model: compute buffer total size = 73.69 MiB
main: init model
print_params: n_vocab               : 32000
print_params: n_ctx                 : 64
print_params: n_embd                : 4096
print_params: n_ff                  : 11008
print_params: n_head                : 32
print_params: n_head_kv             : 32
print_params: n_layer               : 32
print_params: norm_rms_eps          : 0.000010
print_params: rope_freq_base        : 10000.000000
print_params: rope_freq_scale       : 1.000000
print_lora_params: n_rank_attention_norm : 1
print_lora_params: n_rank_wq             : 4
print_lora_params: n_rank_wk             : 4
print_lora_params: n_rank_wv             : 4
print_lora_params: n_rank_wo             : 4
print_lora_params: n_rank_ffn_norm       : 1
print_lora_params: n_rank_ffn_gate       : 4
print_lora_params: n_rank_ffn_down       : 4
print_lora_params: n_rank_ffn_up         : 4
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
train_opt_callback: iter=     1 sample=5/27520 sched=0.010000 loss=2.388987 dt=00:04:22 eta=02:07:01 |->
train_opt_callback: iter=     2 sample=9/27520 sched=0.020000 loss=1.724298 dt=00:04:58 eta=02:19:09 |-------->
train_opt_callback: iter=     3 sample=13/27520 sched=0.030000 loss=2.180474 dt=00:05:04 eta=02:17:01 |--->
train_opt_callback: iter=     4 sample=17/27520 sched=0.040000 loss=2.438480 dt=00:05:12 eta=02:15:36 |->
train_opt_callback: iter=     5 sample=21/27520 sched=0.050000 loss=2.484093 dt=00:05:04 eta=02:06:44 |>
train_opt_callback: iter=     6 sample=25/27520 sched=0.060000 loss=2.695781 dt=00:05:09 eta=02:03:40 |>
train_opt_callback: iter=     7 sample=29/27520 sched=0.070000 loss=2.991009 dt=00:05:07 eta=01:57:51 |>
train_opt_callback: iter=     8 sample=33/27520 sched=0.080000 loss=2.280556 dt=00:05:09 eta=01:53:27 |-->
train_opt_callback: iter=     9 sample=37/27520 sched=0.090000 loss=2.405420 dt=00:05:13 eta=01:49:33 |->
save_checkpoint_lora_file: saving to chk-llama-2-7b.Q8_0-shakespeare-10.gguf
save_checkpoint_lora_file: saving to chk-llama-2-7b.Q8_0-shakespeare-LATEST.gguf
save_as_llama_lora: saving to lora-llama-2-7b.Q8_0-shakespeare-10.bin
save_as_llama_lora: saving to lora-llama-2-7b.Q8_0-shakespeare-LATEST.bin
train_opt_callback: iter=    10 sample=41/27520 sched=0.100000 loss=1.686964 dt=00:05:19 eta=01:46:26 |-------->
train_opt_callback: iter=    11 sample=45/27520 sched=0.110000 loss=2.034604 dt=00:05:18 eta=01:40:43 |----->
train_opt_callback: iter=    12 sample=49/27520 sched=0.120000 loss=2.004366 dt=00:05:14 eta=01:34:29 |----->
train_opt_callback: iter=    13 sample=53/27520 sched=0.130000 loss=2.060003 dt=00:05:15 eta=01:29:26 |---->
train_opt_callback: iter=    14 sample=57/27520 sched=0.140000 loss=2.358844 dt=00:05:11 eta=01:23:06 |->
train_opt_callback: iter=    15 sample=61/27520 sched=0.150000 loss=2.637806 dt=00:05:15 eta=01:18:45 |>
train_opt_callback: iter=    16 sample=65/27520 sched=0.160000 loss=2.313774 dt=00:05:20 eta=01:14:48 |-->
train_opt_callback: iter=    17 sample=69/27520 sched=0.170000 loss=1.786586 dt=00:05:06 eta=01:06:29 |------->
train_opt_callback: iter=    18 sample=73/27520 sched=0.180000 loss=1.707973 dt=00:04:58 eta=00:59:45 |-------->
train_opt_callback: iter=    19 sample=77/27520 sched=0.190000 loss=1.879471 dt=00:04:59 eta=00:54:49 |------>
save_checkpoint_lora_file: saving to chk-llama-2-7b.Q8_0-shakespeare-20.gguf
save_checkpoint_lora_file: saving to chk-llama-2-7b.Q8_0-shakespeare-LATEST.gguf
save_as_llama_lora: saving to lora-llama-2-7b.Q8_0-shakespeare-20.bin
save_as_llama_lora: saving to lora-llama-2-7b.Q8_0-shakespeare-LATEST.bin
train_opt_callback: iter=    20 sample=81/27520 sched=0.200000 loss=1.793221 dt=00:04:55 eta=00:49:19 |------->
train_opt_callback: iter=    21 sample=85/27520 sched=0.210000 loss=2.316586 dt=00:04:58 eta=00:44:43 |-->
train_opt_callback: iter=    22 sample=89/27520 sched=0.220000 loss=1.700760 dt=00:04:56 eta=00:39:29 |-------->
train_opt_callback: iter=    23 sample=93/27520 sched=0.230000 loss=2.253810 dt=00:04:57 eta=00:34:40 |-->
train_opt_callback: iter=    24 sample=97/27520 sched=0.240000 loss=1.886706 dt=00:04:55 eta=00:29:33 |------>
train_opt_callback: iter=    25 sample=101/27520 sched=0.250000 loss=2.514717 dt=00:04:55 eta=00:24:38 |>
train_opt_callback: iter=    26 sample=105/27520 sched=0.260000 loss=2.133458 dt=00:04:54 eta=00:19:39 |---->
train_opt_callback: iter=    27 sample=109/27520 sched=0.270000 loss=1.720507 dt=00:04:54 eta=00:14:44 |-------->
train_opt_callback: iter=    28 sample=113/27520 sched=0.280000 loss=1.918458 dt=00:04:54 eta=00:09:49 |------>
train_opt_callback: iter=    29 sample=117/27520 sched=0.290000 loss=1.626976 dt=00:04:54 eta=00:04:54 |--------->
save_checkpoint_lora_file: saving to chk-llama-2-7b.Q8_0-shakespeare-30.gguf
save_checkpoint_lora_file: saving to chk-llama-2-7b.Q8_0-shakespeare-LATEST.gguf
save_as_llama_lora: saving to lora-llama-2-7b.Q8_0-shakespeare-30.bin
save_as_llama_lora: saving to lora-llama-2-7b.Q8_0-shakespeare-LATEST.bin
train_opt_callback: iter=    30 sample=121/27520 sched=0.300000 loss=1.318394 dt=00:04:54 eta=0.0ms |------------>
main: total training time: 02:36:41
```
I've separated some of the CUDA related output in the above log and I'm not sure
why it's not using the tensor cores for layer:
```console
llm_load_tensors: using CUDA for GPU acceleration
llm_load_tensors: system memory used  = 6828.75 MiB
llm_load_tensors: offloading 0 repeating layers to GPU
llm_load_tensors: offloaded 0/33 layers to GPU
```
I'm going to look into this and see if I can figure out why it's not using the
tensor cores.

There is a command line option to specify the number of layers to offload to the
GPU:
```console
  -ngl N, --n-gpu-layers N   Number of model layers to offload to GPU (default 0)
```
Since we had 0/33 layers in the output above lets try offloading 33 layers:
```
--n-gpu-layers 15 \
```

What I needed to do was download open_llama_3b_v2, and convert it to a GGUF.
Then using this model I could run the finetuning using the GPU:
```console
$ make download-open_llama-3b-v2
$ make convert-open_llama-model
$ make finetune-open_llama-model-cuda 
~/work/ai/llama.cpp/finetune \
        --model-base models/open_llama-2-7b.gguf \
        --checkpoint-in chk-open_llama-shakespeare-LATEST.gguf \
        --checkpoint-out chk-open_llama-shakespeare-ITERATION.gguf \
        --lora-out lora-open_llama-shakespeare-ITERATION.bin \
        --train-data "data/shakespeare.txt" \
        --save-every 10 \
        --threads 6 --adam-iter 30 --batch 4 --ctx 64 \
        --use-checkpointing \
        --n-gpu-layers 33
main: seed: 1704981027
main: model base = 'models/open_llama-2-7b.gguf'
ggml_init_cublas: GGML_CUDA_FORCE_MMQ:   no
ggml_init_cublas: CUDA_USE_TENSOR_CORES: yes
ggml_init_cublas: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4070, compute capability 8.9, VMM: yes
llama_model_loader: loaded meta data with 20 key-value pairs and 237 tensors from models/open_llama-2-7b.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_model_loader: - kv   1:                               general.name str              = .
llama_model_loader: - kv   2:                       llama.context_length u32              = 2048
llama_model_loader: - kv   3:                     llama.embedding_length u32              = 3200
llama_model_loader: - kv   4:                          llama.block_count u32              = 26
llama_model_loader: - kv   5:                  llama.feed_forward_length u32              = 8640
llama_model_loader: - kv   6:                 llama.rope.dimension_count u32              = 100
llama_model_loader: - kv   7:                 llama.attention.head_count u32              = 32
llama_model_loader: - kv   8:              llama.attention.head_count_kv u32              = 32
llama_model_loader: - kv   9:     llama.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  10:                          general.file_type u32              = 1
llama_model_loader: - kv  11:                       tokenizer.ggml.model str              = llama
llama_model_loader: - kv  12:                      tokenizer.ggml.tokens arr[str,32000]   = ["<unk>", "<s>", "</s>", "<0x00>", "<...
llama_model_loader: - kv  13:                      tokenizer.ggml.scores arr[f32,32000]   = [0.000000, 0.000000, 0.000000, 0.0000...
llama_model_loader: - kv  14:                  tokenizer.ggml.token_type arr[i32,32000]   = [2, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, ...
llama_model_loader: - kv  15:                tokenizer.ggml.bos_token_id u32              = 1
llama_model_loader: - kv  16:                tokenizer.ggml.eos_token_id u32              = 2
llama_model_loader: - kv  17:            tokenizer.ggml.padding_token_id u32              = 0
llama_model_loader: - kv  18:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  19:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - type  f32:   53 tensors
llama_model_loader: - type  f16:  184 tensors
llm_load_vocab: special tokens definition check successful ( 259/32000 ).
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = llama
llm_load_print_meta: vocab type       = SPM
llm_load_print_meta: n_vocab          = 32000
llm_load_print_meta: n_merges         = 0
llm_load_print_meta: n_ctx_train      = 2048
llm_load_print_meta: n_embd           = 3200
llm_load_print_meta: n_head           = 32
llm_load_print_meta: n_head_kv        = 32
llm_load_print_meta: n_layer          = 26
llm_load_print_meta: n_rot            = 100
llm_load_print_meta: n_embd_head_k    = 100
llm_load_print_meta: n_embd_head_v    = 100
llm_load_print_meta: n_gqa            = 1
llm_load_print_meta: n_embd_k_gqa     = 3200
llm_load_print_meta: n_embd_v_gqa     = 3200
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: n_ff             = 8640
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 10000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_yarn_orig_ctx  = 2048
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: model type       = 3B
llm_load_print_meta: model ftype      = F16
llm_load_print_meta: model params     = 3.43 B
llm_load_print_meta: model size       = 6.38 GiB (16.00 BPW) 
llm_load_print_meta: general.name     = .
llm_load_print_meta: BOS token        = 1 '<s>'
llm_load_print_meta: EOS token        = 2 '</s>'
llm_load_print_meta: UNK token        = 0 '<unk>'
llm_load_print_meta: PAD token        = 0 '<unk>'
llm_load_print_meta: LF token         = 13 '<0x0A>'
llm_load_tensors: ggml ctx size       =    0.09 MiB
llm_load_tensors: using CUDA for GPU acceleration
llm_load_tensors: system memory used  =  195.40 MiB
llm_load_tensors: VRAM used           = 6340.49 MiB
llm_load_tensors: offloading 26 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 27/27 layers to GPU
.................................................................................................
llama_new_context_with_model: n_ctx      = 512
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init: VRAM kv self = 162.50 MB
llama_new_context_with_model: KV self size  =  162.50 MiB, K (f16):   81.25 MiB, V (f16):   81.25 MiB
llama_build_graph: non-view tensors processed: 550/550
llama_new_context_with_model: compute buffer total size = 71.94 MiB
llama_new_context_with_model: VRAM scratch buffer: 68.75 MiB
llama_new_context_with_model: total VRAM used: 6571.74 MiB (model: 6340.49 MiB, context: 231.25 MiB)
main: init model
print_params: n_vocab               : 32000
print_params: n_ctx                 : 64
print_params: n_embd                : 3200
print_params: n_ff                  : 8640
print_params: n_head                : 32
print_params: n_head_kv             : 32
print_params: n_layer               : 26
print_params: norm_rms_eps          : 0.000001
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
main: lora_size = 54844064 bytes (52.3 MB)
main: opt_size  = 81694048 bytes (77.9 MB)
main: opt iter 0
main: input_size = 32769056 bytes (31.3 MB)
main: compute_size = 2785784672 bytes (2656.7 MB)
main: evaluation order = RIGHT_TO_LEFT
main: tokenize training data
tokenize_file: total number of samples: 26766
main: number of training tokens: 26830
main: number of unique tokens: 3320
main: train data seems to have changed. restarting shuffled epoch.
main: begin training
main: work_size = 768376 bytes (0.7 MB)
train_opt_callback: iter=     0 sample=1/26766 sched=0.000000 loss=0.000000 |->
train_opt_callback: iter=     1 sample=5/26766 sched=0.010000 loss=3.246417 dt=00:01:49 eta=00:52:41 |->
train_opt_callback: iter=     2 sample=9/26766 sched=0.020000 loss=4.157176 dt=00:02:00 eta=00:56:22 |>
train_opt_callback: iter=     3 sample=13/26766 sched=0.030000 loss=4.014138 dt=00:02:07 eta=00:57:21 |>
train_opt_callback: iter=     4 sample=17/26766 sched=0.040000 loss=3.897578 dt=00:02:09 eta=00:56:16 |>
train_opt_callback: iter=     5 sample=21/26766 sched=0.050000 loss=4.125562 dt=00:02:11 eta=00:54:50 |>
train_opt_callback: iter=     6 sample=25/26766 sched=0.060000 loss=4.075722 dt=00:02:09 eta=00:51:43 |>
train_opt_callback: iter=     7 sample=29/26766 sched=0.070000 loss=4.134147 dt=00:02:09 eta=00:49:45 |>
train_opt_callback: iter=     8 sample=33/26766 sched=0.080000 loss=4.040177 dt=00:02:08 eta=00:46:57 |>
train_opt_callback: iter=     9 sample=37/26766 sched=0.090000 loss=4.008081 dt=00:02:10 eta=00:45:35 |>
save_checkpoint_lora_file: saving to chk-open_llama-shakespeare-10.gguf
save_checkpoint_lora_file: saving to chk-open_llama-shakespeare-LATEST.gguf
save_as_llama_lora: saving to lora-open_llama-shakespeare-10.bin
save_as_llama_lora: saving to lora-open_llama-shakespeare-LATEST.bin
train_opt_callback: iter=    10 sample=41/26766 sched=0.100000 loss=3.735419 dt=00:02:11 eta=00:43:57 |>
train_opt_callback: iter=    11 sample=45/26766 sched=0.110000 loss=3.912213 dt=00:02:10 eta=00:41:21 |>
train_opt_callback: iter=    12 sample=49/26766 sched=0.120000 loss=3.502368 dt=00:02:08 eta=00:38:28 |>
train_opt_callback: iter=    13 sample=53/26766 sched=0.130000 loss=4.167959 dt=00:02:12 eta=00:37:36 |>
train_opt_callback: iter=    14 sample=57/26766 sched=0.140000 loss=3.665470 dt=00:02:12 eta=00:35:16 |>
train_opt_callback: iter=    15 sample=61/26766 sched=0.150000 loss=3.757030 dt=00:02:11 eta=00:32:49 |>
train_opt_callback: iter=    16 sample=65/26766 sched=0.160000 loss=3.851915 dt=00:02:08 eta=00:30:01 |>
train_opt_callback: iter=    17 sample=69/26766 sched=0.170000 loss=3.795099 dt=00:02:09 eta=00:28:07 |>
train_opt_callback: iter=    18 sample=73/26766 sched=0.180000 loss=3.775026 dt=00:02:11 eta=00:26:21 |>
train_opt_callback: iter=    19 sample=77/26766 sched=0.190000 loss=3.533836 dt=00:02:08 eta=00:23:30 |>
save_checkpoint_lora_file: saving to chk-open_llama-shakespeare-20.gguf
save_checkpoint_lora_file: saving to chk-open_llama-shakespeare-LATEST.gguf
save_as_llama_lora: saving to lora-open_llama-shakespeare-20.bin
save_as_llama_lora: saving to lora-open_llama-shakespeare-LATEST.bin
train_opt_callback: iter=    20 sample=81/26766 sched=0.200000 loss=3.507541 dt=00:02:08 eta=00:21:20 |>
train_opt_callback: iter=    21 sample=85/26766 sched=0.210000 loss=3.096025 dt=00:02:09 eta=00:19:25 |--->
train_opt_callback: iter=    22 sample=89/26766 sched=0.220000 loss=3.861896 dt=00:02:09 eta=00:17:17 |>
train_opt_callback: iter=    23 sample=93/26766 sched=0.230000 loss=2.967174 dt=00:02:11 eta=00:15:21 |---->
train_opt_callback: iter=    24 sample=97/26766 sched=0.240000 loss=3.490952 dt=00:02:12 eta=00:13:13 |>
train_opt_callback: iter=    25 sample=101/26766 sched=0.250000 loss=3.548366 dt=00:02:10 eta=00:10:54 |>
train_opt_callback: iter=    26 sample=105/26766 sched=0.260000 loss=2.873124 dt=00:02:11 eta=00:08:45 |----->
train_opt_callback: iter=    27 sample=109/26766 sched=0.270000 loss=3.133437 dt=00:02:12 eta=00:06:37 |-->
train_opt_callback: iter=    28 sample=113/26766 sched=0.280000 loss=3.350684 dt=00:02:13 eta=00:04:26 |>
train_opt_callback: iter=    29 sample=117/26766 sched=0.290000 loss=3.184809 dt=00:02:19 eta=00:02:19 |-->
save_checkpoint_lora_file: saving to chk-open_llama-shakespeare-30.gguf
save_checkpoint_lora_file: saving to chk-open_llama-shakespeare-LATEST.gguf
save_as_llama_lora: saving to lora-open_llama-shakespeare-30.bin
save_as_llama_lora: saving to lora-open_llama-shakespeare-LATEST.bin
train_opt_callback: iter=    30 sample=121/26766 sched=0.300000 loss=3.426814 dt=00:02:22 eta=0.0ms |>
main: total training time: 01:07:48
```
This looks much better, instead of almost 3 hours it took a little over an hour
(but also keep in mind that my last attempt without a GPU was using a quantized
model so this is not a far comparison) to fine-tune.

Now, lets try to use this to predict the next word in a sentence. First lets
see what the base model alone predicts:
```console
$ make finetune-predict
./llama.cpp/main -m models/open_llama-2-7b.gguf \
-n 30 \
        --n-gpu-layers 27 \
-p "Love's fire heats water"
Log start
main: build = 1795 (8f900ab)
main: built with gcc (Spack GCC) 12.1.0 for x86_64-pc-linux-gnu
main: seed  = 1705037165
ggml_init_cublas: GGML_CUDA_FORCE_MMQ:   no
ggml_init_cublas: CUDA_USE_TENSOR_CORES: yes
ggml_init_cublas: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4070, compute capability 8.9, VMM: yes
llama_model_loader: loaded meta data with 20 key-value pairs and 237 tensors from models/open_llama-2-7b.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_model_loader: - kv   1:                               general.name str              = .
llama_model_loader: - kv   2:                       llama.context_length u32              = 2048
llama_model_loader: - kv   3:                     llama.embedding_length u32              = 3200
llama_model_loader: - kv   4:                          llama.block_count u32              = 26
llama_model_loader: - kv   5:                  llama.feed_forward_length u32              = 8640
llama_model_loader: - kv   6:                 llama.rope.dimension_count u32              = 100
llama_model_loader: - kv   7:                 llama.attention.head_count u32              = 32
llama_model_loader: - kv   8:              llama.attention.head_count_kv u32              = 32
llama_model_loader: - kv   9:     llama.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  10:                          general.file_type u32              = 1
llama_model_loader: - kv  11:                       tokenizer.ggml.model str              = llama
llama_model_loader: - kv  12:                      tokenizer.ggml.tokens arr[str,32000]   = ["<unk>", "<s>", "</s>", "<0x00>", "<...
llama_model_loader: - kv  13:                      tokenizer.ggml.scores arr[f32,32000]   = [0.000000, 0.000000, 0.000000, 0.0000...
llama_model_loader: - kv  14:                  tokenizer.ggml.token_type arr[i32,32000]   = [2, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, ...
llama_model_loader: - kv  15:                tokenizer.ggml.bos_token_id u32              = 1
llama_model_loader: - kv  16:                tokenizer.ggml.eos_token_id u32              = 2
llama_model_loader: - kv  17:            tokenizer.ggml.padding_token_id u32              = 0
llama_model_loader: - kv  18:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  19:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - type  f32:   53 tensors
llama_model_loader: - type  f16:  184 tensors
llm_load_vocab: special tokens definition check successful ( 259/32000 ).
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = llama
llm_load_print_meta: vocab type       = SPM
llm_load_print_meta: n_vocab          = 32000
llm_load_print_meta: n_merges         = 0
llm_load_print_meta: n_ctx_train      = 2048
llm_load_print_meta: n_embd           = 3200
llm_load_print_meta: n_head           = 32
llm_load_print_meta: n_head_kv        = 32
llm_load_print_meta: n_layer          = 26
llm_load_print_meta: n_rot            = 100
llm_load_print_meta: n_embd_head_k    = 100
llm_load_print_meta: n_embd_head_v    = 100
llm_load_print_meta: n_gqa            = 1
llm_load_print_meta: n_embd_k_gqa     = 3200
llm_load_print_meta: n_embd_v_gqa     = 3200
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: n_ff             = 8640
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 10000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_yarn_orig_ctx  = 2048
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: model type       = 3B
llm_load_print_meta: model ftype      = F16
llm_load_print_meta: model params     = 3.43 B
llm_load_print_meta: model size       = 6.38 GiB (16.00 BPW) 
llm_load_print_meta: general.name     = .
llm_load_print_meta: BOS token        = 1 '<s>'
llm_load_print_meta: EOS token        = 2 '</s>'
llm_load_print_meta: UNK token        = 0 '<unk>'
llm_load_print_meta: PAD token        = 0 '<unk>'
llm_load_print_meta: LF token         = 13 '<0x0A>'
llm_load_tensors: ggml ctx size       =    0.09 MiB
llm_load_tensors: using CUDA for GPU acceleration
llm_load_tensors: system memory used  =  195.40 MiB
llm_load_tensors: VRAM used           = 6340.49 MiB
llm_load_tensors: offloading 26 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 27/27 layers to GPU
.................................................................................................
llama_new_context_with_model: n_ctx      = 512
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init: VRAM kv self = 162.50 MB
llama_new_context_with_model: KV self size  =  162.50 MiB, K (f16):   81.25 MiB, V (f16):   81.25 MiB
llama_build_graph: non-view tensors processed: 550/550
llama_new_context_with_model: compute buffer total size = 71.94 MiB
llama_new_context_with_model: VRAM scratch buffer: 68.75 MiB
llama_new_context_with_model: total VRAM used: 6571.74 MiB (model: 6340.49 MiB, context: 231.25 MiB)

system_info: n_threads = 6 / 12 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | FMA = 1 | NEON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | 
sampling: 
	repeat_last_n = 64, repeat_penalty = 1.100, frequency_penalty = 0.000, presence_penalty = 0.000
	top_k = 40, tfs_z = 1.000, top_p = 0.950, min_p = 0.050, typical_p = 1.000, temp = 0.800
	mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000
sampling order: 
CFG -> Penalties -> top_k -> tfs_z -> typical_p -> top_p -> min_p -> temp 
generate: n_ctx = 512, n_batch = 512, n_predict = 30, n_keep = 0


 Love's fire heats water to make steam
Published: 26 February, 2009, 18:54
TAGS: Technology.
```
And now using the finetune lora model:
```console
$ make finetune-predict-lora
./llama.cpp/main -m models/open_llama-2-7b.gguf \
        --lora lora-open_llama-shakespeare-LATEST.bin \
-n 30 \
-p "Love's fire heats water"
Log start
main: build = 1795 (8f900ab)
main: built with gcc (Spack GCC) 12.1.0 for x86_64-pc-linux-gnu
main: seed  = 1705037014
ggml_init_cublas: GGML_CUDA_FORCE_MMQ:   no
ggml_init_cublas: CUDA_USE_TENSOR_CORES: yes
ggml_init_cublas: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4070, compute capability 8.9, VMM: yes
llama_model_loader: loaded meta data with 20 key-value pairs and 237 tensors from models/open_llama-2-7b.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_model_loader: - kv   1:                               general.name str              = .
llama_model_loader: - kv   2:                       llama.context_length u32              = 2048
llama_model_loader: - kv   3:                     llama.embedding_length u32              = 3200
llama_model_loader: - kv   4:                          llama.block_count u32              = 26
llama_model_loader: - kv   5:                  llama.feed_forward_length u32              = 8640
llama_model_loader: - kv   6:                 llama.rope.dimension_count u32              = 100
llama_model_loader: - kv   7:                 llama.attention.head_count u32              = 32
llama_model_loader: - kv   8:              llama.attention.head_count_kv u32              = 32
llama_model_loader: - kv   9:     llama.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  10:                          general.file_type u32              = 1
llama_model_loader: - kv  11:                       tokenizer.ggml.model str              = llama
llama_model_loader: - kv  12:                      tokenizer.ggml.tokens arr[str,32000]   = ["<unk>", "<s>", "</s>", "<0x00>", "<...
llama_model_loader: - kv  13:                      tokenizer.ggml.scores arr[f32,32000]   = [0.000000, 0.000000, 0.000000, 0.0000...
llama_model_loader: - kv  14:                  tokenizer.ggml.token_type arr[i32,32000]   = [2, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, ...
llama_model_loader: - kv  15:                tokenizer.ggml.bos_token_id u32              = 1
llama_model_loader: - kv  16:                tokenizer.ggml.eos_token_id u32              = 2
llama_model_loader: - kv  17:            tokenizer.ggml.padding_token_id u32              = 0
llama_model_loader: - kv  18:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  19:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - type  f32:   53 tensors
llama_model_loader: - type  f16:  184 tensors
llm_load_vocab: special tokens definition check successful ( 259/32000 ).
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = llama
llm_load_print_meta: vocab type       = SPM
llm_load_print_meta: n_vocab          = 32000
llm_load_print_meta: n_merges         = 0
llm_load_print_meta: n_ctx_train      = 2048
llm_load_print_meta: n_embd           = 3200
llm_load_print_meta: n_head           = 32
llm_load_print_meta: n_head_kv        = 32
llm_load_print_meta: n_layer          = 26
llm_load_print_meta: n_rot            = 100
llm_load_print_meta: n_embd_head_k    = 100
llm_load_print_meta: n_embd_head_v    = 100
llm_load_print_meta: n_gqa            = 1
llm_load_print_meta: n_embd_k_gqa     = 3200
llm_load_print_meta: n_embd_v_gqa     = 3200
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: n_ff             = 8640
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 10000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_yarn_orig_ctx  = 2048
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: model type       = 3B
llm_load_print_meta: model ftype      = F16
llm_load_print_meta: model params     = 3.43 B
llm_load_print_meta: model size       = 6.38 GiB (16.00 BPW) 
llm_load_print_meta: general.name     = .
llm_load_print_meta: BOS token        = 1 '<s>'
llm_load_print_meta: EOS token        = 2 '</s>'
llm_load_print_meta: UNK token        = 0 '<unk>'
llm_load_print_meta: PAD token        = 0 '<unk>'
llm_load_print_meta: LF token         = 13 '<0x0A>'
llm_load_tensors: ggml ctx size       =    0.09 MiB
llm_load_tensors: using CUDA for GPU acceleration
llm_load_tensors: system memory used  = 6535.89 MiB
llm_load_tensors: offloading 0 repeating layers to GPU
llm_load_tensors: offloaded 0/27 layers to GPU
.................................................................................................
llama_new_context_with_model: n_ctx      = 512
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 1
llama_new_context_with_model: KV self size  =  162.50 MiB, K (f16):   81.25 MiB, V (f16):   81.25 MiB
llama_build_graph: non-view tensors processed: 550/550
llama_new_context_with_model: compute buffer total size = 71.94 MiB
llama_apply_lora_from_file_internal: applying lora adapter from 'lora-open_llama-shakespeare-LATEST.bin' - please wait ...
llama_apply_lora_from_file_internal: r = 4, alpha = 4, scaling = 1.00
llama_apply_lora_from_file_internal: allocating 1172 MB for lora temporary buffer
........................................................... done (7095.45 ms)

system_info: n_threads = 6 / 12 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | FMA = 1 | NEON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | 
sampling: 
	repeat_last_n = 64, repeat_penalty = 1.100, frequency_penalty = 0.000, presence_penalty = 0.000
	top_k = 40, tfs_z = 1.000, top_p = 0.950, min_p = 0.050, typical_p = 1.000, temp = 0.800
	mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000
sampling order: 
CFG -> Penalties -> top_k -> tfs_z -> typical_p -> top_p -> min_p -> temp 
generate: n_ctx = 512, n_batch = 512, n_predict = 30, n_keep = 0


 Love's fire heats water in the well,
As 'tis thy love I think I never saw.
O, let my heart have life, if it be so
```
That does not make much sense to me but does sound more like shakespeare than
the output when using only the base model.

The lora adapter layer can be merged with the base model to create a new model:
```console
$ make merge-lora-adapter-with-base-model
```
And then we can run the inference using only that model and not the lora adapter
file:
```console
$ make finetune-predict-lora-merged-model
```
This also allows all the layers to be offloaded to the GPU.

_wip_


### GPU and LoRA
I ran into the following error when trying to offload layers to the GPU when
using a lora adapter.
```console
$ make finetune-predict-lora
...
llama_apply_lora_from_file_internal: applying lora adapter from 'lora-open_llama-shakespeare-LATEST.bin' - please wait ...
llama_apply_lora_from_file_internal: r = 4, alpha = 4, scaling = 1.00
llama_apply_lora_from_file_internal: allocating 1172 MB for lora temporary buffer
llama_model_apply_lora_from_file: failed to apply lora adapter: llama_apply_lora_from_file_internal: error: the simultaneous use of LoRAs and GPU acceleration is only supported for f16 models. dest_t->type: 0
llama_init_from_gpt_params: error: failed to apply lora adapter
main: error: unable to load model
make: *** [Makefile:116: finetune-predict-lora] Error 1
```
So the final layer of the lora process is f32 but it needs to be of type f16. We
need to convert this in some way if we want to be able to use the GPU.

I found this
[discussion](https://github.com/ggerganov/llama.cpp/discussions/4317) which is
related to this issue.


### Finetuning with chat interactions
In this case I want to finetune a basemodel with chat interactions. For this
I should use a basemodel that has been trained for chat and it should therefore
recognize the format of the chat. The actual format of the chat will need to
the same as the basemodel uses.

Lets try https://huggingface.co/meta-llama/Llama-2-7b-chat-hf:
(we need to specify our HuggingFace username and an access token)
```console
$ make checkout-llama-2-7b-chat-hf
```
Then we convert this model to a GGUF model:
```console
$ make convert-llama-2.7b-chat-model
...
Loading model: Llama-2-7b-chat-hf
Traceback (most recent call last):
  File "/home/danielbevenius/work/ai/learning-ai/fundamentals/llama.cpp/llama.cpp/convert-hf-to-gguf.py", line 1268, in <module>
    main()
  File "/home/danielbevenius/work/ai/learning-ai/fundamentals/llama.cpp/llama.cpp/convert-hf-to-gguf.py", line 1249, in main
    model_instance = model_class(dir_model, ftype_map[args.outtype], fname_out, args.bigendian)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/danielbevenius/work/ai/learning-ai/fundamentals/llama.cpp/llama.cpp/convert-hf-to-gguf.py", line 57, in __init__
    self.model_arch = self._get_model_architecture()
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/danielbevenius/work/ai/learning-ai/fundamentals/llama.cpp/llama.cpp/convert-hf-to-gguf.py", line 246, in _get_model_architecture
    raise NotImplementedError(f'Architecture "{arch}" not supported!')
NotImplementedError: Architecture "LlamaForCausalLM" not supported!
make: *** [Makefile:85: convert-llama-2.7b-chat-model] Error 1
```
If we look in [config.json](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/blob/main/config.json)
we find:
```json
{
  "_name_or_path": "meta-llama/Llama-2-7b-chat-hf",
  "architectures": [
    "LlamaForCausalLM"
  ],
  ...
}
```
This architecture is not supported by the convert-hf-to-gguf.py script. Lets
try adding it:
```console
index b133f3b4..9c60b84c 100755
--- a/convert-hf-to-gguf.py
+++ b/convert-hf-to-gguf.py
@@ -242,6 +242,8 @@ class Model:
             return gguf.MODEL_ARCH.PHI2
         if arch == "PlamoForCausalLM":
             return gguf.MODEL_ARCH.PLAMO
+        if arch == "LlamaForCausalLM":
+            return gguf.MODEL_ARCH.LLAMA
 
         raise NotImplementedError(f'Architecture "{arch}" not supported!')
```
Rerunning the above command:
```console
$ make convert-llama-2.7b-chat-model
...

Model successfully exported to 'models/llama-2-7b-chat.gguf'
Now lets see if we can use this as a base model for LoRA training:
```console
$ make finetune-llama-model-cuda 
./finetune \
        --model-base models/llama-2-7b-chat.gguf \
        --checkpoint-in chk-training-LATEST.gguf \
        --checkpoint-out chk-training-ITERATION.gguf \
        --lora-out lora-training-ITERATION.gguf \
        --train-data "data/assistent-training.txt" \
        --save-every 10 \
        --threads 6 --adam-iter 30 --batch 4 --ctx 64 \
        --use-checkpointing \
        --n-gpu-layers 33
main: seed: 1705476230
main: model base = 'models/llama-2-7b-chat.gguf'
ggml_init_cublas: GGML_CUDA_FORCE_MMQ:   no
ggml_init_cublas: CUDA_USE_TENSOR_CORES: yes
ggml_init_cublas: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4070, compute capability 8.9, VMM: yes
llama_model_loader: loaded meta data with 20 key-value pairs and 323 tensors from models/llama-2-7b-chat.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_model_loader: - kv   1:                               general.name str              = Llama-2-7b-chat-hf
llama_model_loader: - kv   2:                          llama.block_count u32              = 32
llama_model_loader: - kv   3:                       llama.context_length u32              = 4096
llama_model_loader: - kv   4:                     llama.embedding_length u32              = 4096
llama_model_loader: - kv   5:                  llama.feed_forward_length u32              = 11008
llama_model_loader: - kv   6:                 llama.attention.head_count u32              = 32
llama_model_loader: - kv   7:              llama.attention.head_count_kv u32              = 32
llama_model_loader: - kv   8:     llama.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv   9:                llama.use_parallel_residual bool             = true
llama_model_loader: - kv  10:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  11:                      tokenizer.ggml.tokens arr[str,32000]   = ["<unk>", "<s>", "</s>", "<0x00>", "<...
llama_model_loader: - kv  12:                  tokenizer.ggml.token_type arr[i32,32000]   = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  13:                      tokenizer.ggml.merges arr[str,61249]   = ["▁ t", "e r", "i n", "▁ a", "e n...
llama_model_loader: - kv  14:                tokenizer.ggml.bos_token_id u32              = 1
llama_model_loader: - kv  15:                tokenizer.ggml.eos_token_id u32              = 2
llama_model_loader: - kv  16:            tokenizer.ggml.unknown_token_id u32              = 0
llama_model_loader: - kv  17:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  18:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  19:                    tokenizer.chat_template str              = {% if messages[0]['role'] == 'system'...
llama_model_loader: - type  f32:  323 tensors
llama_model_load: error loading model: ERROR: byte not found in vocab
llama_load_model_from_file: failed to load model
make: *** [Makefile:100: finetune-llama-model-cuda] Segmentation fault (core dumped)
```
Hmm, it looks like this is using BPE and not SPM (found this out by looking at 
the code where this error is thrown) and perhaps that is the issue.
So, when I added the LlamaForCausalLM architecture I did missed adding this
arch to from_model_architecture which is done for the other archs. Adding this
and just specifying the model as:
```python
class LlamaModel(Model):
    def set_vocab(self):
        self._set_vocab_sentencepiece()
```
So the changes would then look like this:
```console
diff --git a/convert-hf-to-gguf.py b/convert-hf-to-gguf.py
index b133f3b4..ddfabf0e 100755
--- a/convert-hf-to-gguf.py
+++ b/convert-hf-to-gguf.py
@@ -197,6 +197,8 @@ class Model:
             return Phi2Model
         if model_architecture == "PlamoForCausalLM":
             return PlamoModel
+        if model_architecture == "LlamaForCausalLM":
+            return LlamaModel
         return Model
 
     def _is_model_safetensors(self) -> bool:
@@ -242,6 +244,8 @@ class Model:
             return gguf.MODEL_ARCH.PHI2
         if arch == "PlamoForCausalLM":
             return gguf.MODEL_ARCH.PLAMO
+        if arch == "LlamaForCausalLM":
+            return gguf.MODEL_ARCH.LLAMA
 
         raise NotImplementedError(f'Architecture "{arch}" not supported!')
 
@@ -888,6 +892,10 @@ class MixtralModel(Model):
     def set_vocab(self):
         self._set_vocab_sentencepiece()
 
+class LlamaModel(Model):
+    def set_vocab(self):
```
So lets convert the model with these changes and see if the error is gone now:
```console
llm_load_print_meta: general.name     = Llama-2-7b-chat-hf
llm_load_print_meta: BOS token        = 1 '<s>'
llm_load_print_meta: EOS token        = 2 '</s>'
llm_load_print_meta: UNK token        = 0 '<unk>'
llm_load_print_meta: LF token         = 13 '<0x0A>'
llm_load_tensors: ggml ctx size =    0.25 MiB
llama_model_load: error loading model: done_getting_tensors: wrong number of tensors; expected 323, got 291
llama_load_model_from_file: failed to load model
make: *** [Makefile:100: finetune-llama-model-cuda] Segmentation fault (core dumped)
```
So there are fewer tensors in the model than expected, and specifically
323-291=32. This sounds like there is a layer that is missing and the number of
attentions heads I know are 32. I wonder if these are some missing from the
conversion. Actually the original model has 291 layers and it is the converted
model that has 32 layers extra.

```
self.hparams: {
'_name_or_path': 'meta-llama/Llama-2-7b-chat-hf',
'architectures': ['LlamaForCausalLM'],
'bos_token_id': 1,
'eos_token_id': 2,
'hidden_act': 'silu',
'hidden_size': 4096, 
'initializer_range': 0.02,
'intermediate_size': 11008, 
'max_position_embeddings': 4096, 
'model_type': 'llama', 

'num_attention_heads': 32, 
'num_hidden_layers': 32, 
'num_key_value_heads': 32, 

'pretraining_tp': 1, 
'rms_norm_eps': 1e-05, 
'rope_scaling': None, 
'tie_word_embeddings': False, 
'torch_dtype': 'float16',
'transformers_version': '4.32.0.dev0', 
'use_cache': True, 
'vocab_size': 32000}
```

I've downloaded the original model from meta-llama/Llama-2-7b-chat and converted
it to gguf and then quantized it to F16. This is a little more than my GPU can
handle as it only has 12GB of VRAM. But if I reduce the number of layers that
are to be offloaded to the GPU, setting it to 25, then it will start training.
Note sure how long this will take but it will probably take longer as not all
layers can be on the GPU.

TODO: Would it be possible to quantize the model to Q8_0 which I would then
be able to run on my GPU? In the README.md of the finetune example the example
they show is using an open-llama open-llama-3b-v2-q8_0.gguf model. But in that
example it does not mention Cuda and it was when using Cude that I ran into
an issue and note when running without Cude.

_wip_

This section is for looking into supporting HuggingFace Llama models and
converting them to GGUF models. 
If we look llama-2-7b-chat/params.json we find:
```json
{
  "dim": 4096,
  "multiple_of": 256, 
  "n_heads": 32, 
  "n_layers": 32, 
  "norm_eps": 1e-06, 
  "vocab_size": 32000}
}
```
I updated vocab_size as it was initially `-1`. 
So we have the n_heads, n_layers and notice that we have a multiple_of.


### Finetuning user/assistent training data formatting
I ran into a case where I had specified training data in the current format:
```
<s>[INST] What is RHSA-1820:1234? [/INST] RHSA-1820:1234 is a Red Hat Security Advisory that deals with a division by zero error in the Bajja library. </s>
```
And the when running the finetuning I was specifying the following command line
options:
```console
finetune-model:
	./finetune \
        --model-base ${MODEL} \
        --checkpoint-in chk-${TYPE}-training-LATEST.gguf \
        --checkpoint-out chk-${TYPE}-training-ITERATION.gguf \
        --lora-out lora-${TYPE}-training-ITERATION.gguf \
        --train-data "${TRAIN_DATA}" \
        --save-every 10 \
        --threads 6 \
       	--adam-iter 30 \
        --batch 4 \
        --use-checkpointing \
       	--ctx 80 \
        --sample-start '<s>' \
        ${CUDA_GPU_LAYERS}
```
Notice that I was specifying `<s>` as the sample-start character which is
perfectly fine, but the problem with this is I was not setting
`--include-sample-start` which means that the `<s>` character would be included
in the samples that llama.cpp will tokenize. We can see this by using the
[tokenize_file.cc](./src/tokenize_file.cc) program:
```console
$ make tokenize-file 
...
sample: '[INST] What is RHSA-1820:2010? [/INST] RHSA-1820:2010 is a Red Hat Security Advisory addressing an insecure default configuration in the India server tool. </s>
work. </s>
```
If we instead set `--include-sample-start`, just setting the argument include
tokenize_file.cc, to true then the `<s>` character will also be included in the
sample data:
```
sample: '<s>[INST] What is RHSA-1820:1234? [/INST] RHSA-1820:1234 is a Red Hat Security Advisory that deals with a division by zero error in the Bajja library. </s>
'
```
So with those changes and running the finetuning again I'm still not getting
the results I'm expecting. I'm getting the following output:
```console
$ make predict-llama-lora-merged-model
...
Can you show me a summary of RHSA-1820:1234? [end of text]
```
I'm not sure if this could be due to the fact that I only had like 10 samples
in the finetuning data set. I'm going to increase this to 100 and see if that
makes a difference. Increasing the number of samples can be done using ChatGPT
or using copilot in the samples file directly.

Regarding the formatting of the training data I found the following in the
[blog](https://mlabonne.github.io/blog/posts/Fine_Tune_Your_Own_Llama_2_Model_in_a_Colab_Notebook.html)
and this is the format that I'm following:
```
'<s>[INST] What lottery will it be entered into? [/INST] Hi there! It looks like I need some more context before I can answer your question. Could you please specify what lottery you are referring to and/or what is being entered into it? </s>'
```

Still with more sample I'm getting the same result.
Actually, I'm also seeing responses like this:
```console
Can you show me a summary of RHSA-1820:1234? [/INST] RHSA-1820 is a Red Hat Security Advisory for a privilege escalation issue in the kernel. </s>
 [end of text]
```
Hmm, I think my prompt also needs to be updated. That is I've been using text
completion previously and not a chat prompt. So perhaps I need to also format
the prompt in the same way. For example:
```console
	-p "<s>[INST] Can you show me a summary of RHSA-1820:1234? [/INST]"
```
One thing to note about the output is that by default it will include the prompt
and it includes the `[INST], and [/INST]` tokens which confused me a little but
this can be suppress by setting `--no-display-prompt`:
```console
$ make predict-llama-lora-merged-model 
./llama.cpp/main -m llama-lora-merged-model.gguf \
        -n 80 \
        --n-gpu-layers 27 \
        --no-display-prompt \
        --log-disable \
        -p "<s>[INST] Can you show me a summary of RHSA-1820:1234? [/INST]"
ggml_init_cublas: GGML_CUDA_FORCE_MMQ:   no
ggml_init_cublas: CUDA_USE_TENSOR_CORES: yes
ggml_init_cublas: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4070, compute capability 8.9, VMM: yes
 RHSA-1820:1234 is an Red Hat Security Advisory addressing a buffer overflow in the Red Hat Enterprise Linux kernel that could lead to a privilege escalation vulnerability. </s>
```
Notice the `</s>` token at the end of the output. I'm not sure if this is
expected or if it is something that is caused by the training data. If I ask
the model to predict some other question we can compare the difference:
```console
$ make predict-llama-lora-merged-model 
./llama.cpp/main -m llama-lora-merged-model.gguf \
        -n 80 \
        --n-gpu-layers 27 \
        --no-display-prompt \
        --log-disable \
        -p "<s>[INST] What is the capital of Sweden? [/INST]"
ggml_init_cublas: GGML_CUDA_FORCE_MMQ:   no
ggml_init_cublas: CUDA_USE_TENSOR_CORES: yes
ggml_init_cublas: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4070, compute capability 8.9, VMM: yes
  The capital of Sweden is Stockholm.
```
Notice that there is no `</s>` token at the end of the output. 
I wonder if this could be caused by the fact that I'm using the `<s>` token as
the start token instead of a separate character?

I'm going to try to use a separate character as the start token and see if that
like `###`:
```console
$ make predict-llama-lora-merged-model 
./llama.cpp/main -m llama-lora-merged-model.gguf \
        -n 200 \
        --n-gpu-layers 27 \
        --no-display-prompt \
        --log-disable \
        --threads 6 \
        --ctx-size 512 \
        -p "<s>[INST] Can you show me a summary of 2 things to do Stockholm? [/INST]"
ggml_init_cublas: GGML_CUDA_FORCE_MMQ:   no
ggml_init_cublas: CUDA_USE_TENSOR_CORES: yes
ggml_init_cublas: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4070, compute capability 8.9, VMM: yes
  Of course! Here are two things to do in Stockholm:

1. Visit the Vasa Museum: The Vasa Museum is one of Sweden'$
```
Now, if I use the base model with the same prompt I get this result:
```console
$ make predict-llama-lora-merged-model 
./llama.cpp/main -m models/llama-2-7b-chat.gguf \
        -n 200 \
        --n-gpu-layers 27 \
        --no-display-prompt \
        --log-disable \
        --threads 6 \
        --ctx-size 512 \
        -p "<s>[INST] Can you show me a summary of 2 things to do Stockholm? [/INST]"
ggml_init_cublas: GGML_CUDA_FORCE_MMQ:   no
ggml_init_cublas: CUDA_USE_TENSOR_CORES: yes
ggml_init_cublas: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4070, compute capability 8.9, VMM: yes
  Of course! Here are two things to do in Stockholm, Sweden:

1. Visit the Vasa Museum: The Vasa Museum is one of Stockholm's most popular attractions and is located on the waterfront. The museum features the world's only preserved 17th-century ship, the Vasa, which sank on its maiden voyage in 1628. The ship has been restored and visitors can explore its grand halls, cabins, and cannons.
2. Explore the Old Town (Gamla Stan): Stockholm's Old Town is a charming neighborhood filled with cobblestone streets, medieval buildings, and historic landmarks like Storkyrkan (the Church of St. Nicholas) and the Riddarholmen Church. Visitors can also explore the many boutiques, restaurants, and cafes in the area or take a boat tour of the city's canals
```
It seems like merging with the base model is not working as expected.
I've updated the training data with new example and more varied as I thought
one issue might be that the samples I had were too similar.
The result after training are now something like this:
```
$ make predict-llama-lora-merged-model 
./llama.cpp/main -m llama-lora-merged-model.gguf \
        -n 100 \
        --n-gpu-layers 27 \
        --no-display-prompt \
        --log-disable \
        --threads 6 \
        --ctx-size 512 \
        -p "<s>[INST] Can you show me a summary of RHSA-2024:0102? [/INST]"
ggml_init_cublas: GGML_CUDA_FORCE_MMQ:   no
ggml_init_cublas: CUDA_USE_TENSOR_CORES: yes
ggml_init_cublas: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4070, compute capability 8.9, VMM: yes
  Red Hat Security Advisory #2024:0102 is a critical fix for RHSA-2023:1178, which was released on December 15th.
```
