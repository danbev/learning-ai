## Langchain examples


### Streamlit

```console
(langch) $ streamlit run src/agent-example.py 
Traceback (most recent call last):
  File "/home/danielbevenius/work/ai/learning-ai/langchain/langch/bin/streamlit", line 8, in <module>
    sys.exit(main())
             ^^^^^^
  File "/home/danielbevenius/work/ai/learning-ai/langchain/langch/lib64/python3.11/site-packages/click/core.py", line 1157, in __call__
    return self.main(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/danielbevenius/work/ai/learning-ai/langchain/langch/lib64/python3.11/site-packages/click/core.py", line 1078, in main
    rv = self.invoke(ctx)
         ^^^^^^^^^^^^^^^^
  File "/home/danielbevenius/work/ai/learning-ai/langchain/langch/lib64/python3.11/site-packages/click/core.py", line 1688, in invoke
    return _process_result(sub_ctx.command.invoke(sub_ctx))
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/danielbevenius/work/ai/learning-ai/langchain/langch/lib64/python3.11/site-packages/click/core.py", line 1434, in invoke
    return ctx.invoke(self.callback, **ctx.params)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/danielbevenius/work/ai/learning-ai/langchain/langch/lib64/python3.11/site-packages/click/core.py", line 783, in invoke
    return __callback(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/danielbevenius/work/ai/learning-ai/langchain/langch/lib64/python3.11/site-packages/streamlit/web/cli.py", line 233, in main_run
    _main_run(target, args, flag_options=kwargs)
  File "/home/danielbevenius/work/ai/learning-ai/langchain/langch/lib64/python3.11/site-packages/streamlit/web/cli.py", line 269, in _main_run
    bootstrap.run(file, command_line, args, flag_options)
  File "/home/danielbevenius/work/ai/learning-ai/langchain/langch/lib64/python3.11/site-packages/streamlit/web/bootstrap.py", line 411, in run
    _install_pages_watcher(main_script_path)
  File "/home/danielbevenius/work/ai/learning-ai/langchain/langch/lib64/python3.11/site-packages/streamlit/web/bootstrap.py", line 386, in _install_pages_watcher
    watch_dir(
  File "/home/danielbevenius/work/ai/learning-ai/langchain/langch/lib64/python3.11/site-packages/streamlit/watcher/path_watcher.py", line 153, in watch_dir
    return _watch_path(
           ^^^^^^^^^^^^
  File "/home/danielbevenius/work/ai/learning-ai/langchain/langch/lib64/python3.11/site-packages/streamlit/watcher/path_watcher.py", line 128, in _watch_path
    watcher_class(
  File "/home/danielbevenius/work/ai/learning-ai/langchain/langch/lib64/python3.11/site-packages/streamlit/watcher/event_based_path_watcher.py", line 92, in __init__
    path_watcher.watch_path(
  File "/home/danielbevenius/work/ai/learning-ai/langchain/langch/lib64/python3.11/site-packages/streamlit/watcher/event_based_path_watcher.py", line 170, in watch_path
    folder_handler.watch = self._observer.schedule(
                           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/danielbevenius/work/ai/learning-ai/langchain/langch/lib64/python3.11/site-packages/watchdog/observers/api.py", line 301, in schedule
    emitter.start()
  File "/home/danielbevenius/work/ai/learning-ai/langchain/langch/lib64/python3.11/site-packages/watchdog/utils/__init__.py", line 92, in start
    self.on_thread_start()
  File "/home/danielbevenius/work/ai/learning-ai/langchain/langch/lib64/python3.11/site-packages/watchdog/observers/inotify.py", line 119, in on_thread_start
    self._inotify = InotifyBuffer(path, self.watch.is_recursive)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/danielbevenius/work/ai/learning-ai/langchain/langch/lib64/python3.11/site-packages/watchdog/observers/inotify_buffer.py", line 37, in __init__
    self._inotify = Inotify(path, recursive)
                    ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/danielbevenius/work/ai/learning-ai/langchain/langch/lib64/python3.11/site-packages/watchdog/observers/inotify_c.py", line 179, in __init__
    self._add_dir_watch(path, recursive, event_mask)
  File "/home/danielbevenius/work/ai/learning-ai/langchain/langch/lib64/python3.11/site-packages/watchdog/observers/inotify_c.py", line 395, in _add_dir_watch
    self._add_watch(path, mask)
  File "/home/danielbevenius/work/ai/learning-ai/langchain/langch/lib64/python3.11/site-packages/watchdog/observers/inotify_c.py", line 416, in _add_watch
    Inotify._raise_error()
  File "/home/danielbevenius/work/ai/learning-ai/langchain/langch/lib64/python3.11/site-packages/watchdog/observers/inotify_c.py", line 428, in _raise_error
    raise OSError(errno.ENOSPC, "inotify watch limit reached")
OSError: [Errno 28] inotify watch limit reached
(langch) $ streamlit run args --server.fileWatcherType none src/agent-example.py 
Usage: streamlit run [OPTIONS] TARGET [ARGS]...
Try 'streamlit run --help' for help.

Error: Streamlit requires raw Python (.py) files, but the provided file has no extension.
For more information, please see https://docs.streamlit.io
```
What worked for me was to set fileWatcherType to poll in the config.toml file:

```toml 
[server]
fileWatcherType = "poll"
```
I was then able to run a streamlit application using:
```
(langch) $ streamlit run src/agent-example.py 

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.0.66:8501
```

### llama-local.py example
This is an example of using llama.cpp to run LLama2 modules locally. For this
to work we need to pip install `llama_cpp_python` and we also need to have
a model to work with. I used the following model:
```console
$ wget -P models https://huggingface.co/localmodels/Llama-2-7B-Chat-ggml/resolve/main/llama-2-7b-chat.ggmlv3.q2_K.bin
```

llama.cpp has been updated to handle GGUF format instead of GGML format which
means that we need to convert the older GGML format to GGUF format:
```console
(langch) $ ~/work/ai/llama.cpp/convert-llama-ggml-to-gguf.py --input models/llama-2-7b-chat.ggmlv3.q2_K.bin  --output models/llama-2-7b-chat.gguf.q2_K.bin
* Using config: Namespace(input=PosixPath('models/llama-2-7b-chat.ggmlv3.q2_K.bin'), output=PosixPath('models/llama-2-7b-chat.gguf.q2_K.bin'), name=None, desc=None, gqa=1, eps='5.0e-06', context_length=2048, model_metadata_dir=None, vocab_dir=None, vocabtype='spm')

=== WARNING === Be aware that this conversion script is best-effort. Use a native GGUF model if possible. === WARNING ===

- Note: If converting LLaMA2, specifying "--eps 1e-5" is required. 70B models also need "--gqa 8".
* Scanning GGML input file
* File format: GGJTv3 with ftype MOSTLY_Q2_K
* GGML model hyperparameters: <Hyperparameters: n_vocab=32000, n_embd=4096, n_mult=256, n_head=32, n_layer=32, n_rot=128, n_ff=11008, ftype=MOSTLY_Q2_K>

=== WARNING === Special tokens may not be converted correctly. Use --model-metadata-dir if possible === WARNING ===

* Preparing to save GGUF file
* Adding model parameters and KV items
* Adding 32000 vocab item(s)
* Adding 291 tensor(s)
    gguf: write header
    gguf: write metadata
    gguf: write tensors
* Successful completion. Output saved to: models/llama-2-7b-chat.gguf.q2_K.bin
```
After this we should be good to go and be able to run the example:
```console
(langch) $ python src/llama-local.py 
llama_model_loader: loaded meta data with 19 key-value pairs and 291 tensors from models/llama-2-7b-chat.gguf.q2_K.bin (version GGUF V2 (latest))
llama_model_loader: - tensor    0:                token_embd.weight q2_K     [  4096, 32000,     1,     1 ]
llama_model_loader: - tensor    1:               output_norm.weight f32      [  4096,     1,     1,     1 ]
...
llama_model_loader: - tensor  290:           blk.31.ffn_norm.weight f32      [  4096,     1,     1,     1 ]
llama_model_loader: - kv   0:                       general.architecture str     
llama_model_loader: - kv   1:                               general.name str     
llama_model_loader: - kv   2:                        general.description str     
llama_model_loader: - kv   3:                          general.file_type u32     
llama_model_loader: - kv   4:                       llama.context_length u32     
llama_model_loader: - kv   5:                     llama.embedding_length u32     
llama_model_loader: - kv   6:                          llama.block_count u32     
llama_model_loader: - kv   7:                  llama.feed_forward_length u32     
llama_model_loader: - kv   8:                 llama.rope.dimension_count u32     
llama_model_loader: - kv   9:                 llama.attention.head_count u32     
llama_model_loader: - kv  10:              llama.attention.head_count_kv u32     
llama_model_loader: - kv  11:     llama.attention.layer_norm_rms_epsilon f32     
llama_model_loader: - kv  12:                       tokenizer.ggml.model str     
llama_model_loader: - kv  13:                      tokenizer.ggml.tokens arr     
llama_model_loader: - kv  14:                      tokenizer.ggml.scores arr     
llama_model_loader: - kv  15:                  tokenizer.ggml.token_type arr     
llama_model_loader: - kv  16:            tokenizer.ggml.unknown_token_id u32     
llama_model_loader: - kv  17:                tokenizer.ggml.bos_token_id u32     
llama_model_loader: - kv  18:                tokenizer.ggml.eos_token_id u32     
llama_model_loader: - type  f32:   65 tensors
llama_model_loader: - type q2_K:  129 tensors
llama_model_loader: - type q4_K:   96 tensors
llama_model_loader: - type q6_K:    1 tensors
llm_load_print_meta: format           = GGUF V2 (latest)
llm_load_print_meta: arch             = llama
llm_load_print_meta: vocab type       = SPM
llm_load_print_meta: n_vocab          = 32000
llm_load_print_meta: n_merges         = 0
llm_load_print_meta: n_ctx_train      = 2048
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_head           = 32
llm_load_print_meta: n_head_kv        = 32
llm_load_print_meta: n_layer          = 32
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_gqa            = 1
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 5.0e-06
llm_load_print_meta: n_ff             = 11008
llm_load_print_meta: freq_base_train  = 10000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: model type       = 7B
llm_load_print_meta: model ftype      = mostly Q2_K
llm_load_print_meta: model params     = 6.74 B
llm_load_print_meta: model size       = 2.67 GiB (3.40 BPW) 
llm_load_print_meta: general.name   = llama-2-7b-chat.ggmlv3.q2_K.bin
llm_load_print_meta: BOS token = 1 '<s>'
llm_load_print_meta: EOS token = 2 '</s>'
llm_load_print_meta: UNK token = 0 '<unk>'
llm_load_print_meta: LF token  = 13 '<0x0A>'
llm_load_tensors: ggml ctx size =    0.09 MB
llm_load_tensors: mem required  = 2733.66 MB
.................................................................................................
llama_new_context_with_model: n_ctx      = 512
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 1
llama_new_context_with_model: kv self size  =  256.00 MB
llama_new_context_with_model: compute buffer total size = 6.98 MB
AVX = 1 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | FMA = 1 | NEON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | 
LlamaCpp
Params: {'model_path': 'models/llama-2-7b-chat.gguf.q2_K.bin', 'suffix': None, 'max_tokens': 2000, 'temperature': 0.0, 'top_p': 1.0, 'logprobs': None, 'echo': False, 'stop_sequences': [], 'repeat_penalty': 1.1, 'top_k': 40}

Answer: Austin Danger Powers is a fictional character and a parody of the James Bond character. He is depicted as an over-the-top, campy, and flamboyant secret agent who is always ready to save the world from danger. He has a catchphrase "Danger! Oh, Danger!" and is known for his elaborate outfits and gadgets.
llama_print_timings:        load time =   474.45 ms
llama_print_timings:      sample time =    42.89 ms /    86 runs   (    0.50 ms per token,  2005.04 tokens per second)
llama_print_timings: prompt eval time =  1266.48 ms /    22 tokens (   57.57 ms per token,    17.37 tokens per second)
llama_print_timings:        eval time = 14069.34 ms /    85 runs   (  165.52 ms per token,     6.04 tokens per second)
llama_print_timings:       total time = 15576.92 ms
```
