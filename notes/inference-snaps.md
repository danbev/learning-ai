## Canonical Inference snaps:

## Installation
```console
$ sudo snap install gemma3+llamacpp --beta
```
```console
$ sudo gemma3 use-engine cpu
Need to install the following components:
- model-4b-it-q4-0-gguf (2.9GiB)
- mmproj-f16-4b-gguf (790.1MiB)

Do you want to continue? [y/n] y

Engine changed to "cpu".

Run "snap restart gemma3.server" to use the new engine.
```

### Usage
```console
$ gemma3 chat
Using server at http://localhost:8328/v1
Type your prompt, then ENTER to submit. CTRL-C to quit.
```
```console
$ gemma3 status
engine: cpu
services:
    server: active
endpoints:
    openai: http://localhost:8328/v1
```
And we can access the webui using: http://localhost:8328

### logs
```console
$ sudo snap logs gemma3 
2026-02-18T10:54:35+01:00 gemma3.server[140011]: slot update_slots: id  2 | task 37 | n_tokens = 4, memory_seq_rm [4, end)
2026-02-18T10:54:35+01:00 gemma3.server[140011]: slot update_slots: id  2 | task 37 | prompt processing progress, n_tokens = 17, batch.n_tokens = 13, progress = 1.000000
2026-02-18T10:54:35+01:00 gemma3.server[140011]: slot update_slots: id  2 | task 37 | prompt done, n_tokens = 17, batch.n_tokens = 13
2026-02-18T10:54:38+01:00 gemma3.server[140011]: slot print_timing: id  2 | task 37 |
2026-02-18T10:54:38+01:00 gemma3.server[140011]: prompt eval time =     263.10 ms /    13 tokens (   20.24 ms per token,    49.41 tokens per second)
2026-02-18T10:54:38+01:00 gemma3.server[140011]:        eval time =    2719.40 ms /    31 tokens (   87.72 ms per token,    11.40 tokens per second)
2026-02-18T10:54:38+01:00 gemma3.server[140011]:       total time =    2982.49 ms /    44 tokens
2026-02-18T10:54:38+01:00 gemma3.server[140011]: slot      release: id  2 | task 37 | stop processing: n_tokens = 47, truncated = 0
2026-02-18T10:54:38+01:00 gemma3.server[140011]: srv  update_slots: all slots are idle
2026-02-18T10:54:38+01:00 gemma3.server[140011]: srv  log_server_r: request: POST /v1/chat/completions 127.0.0.1 200
```

### Stop
```console
$ sudo snap stop gemma3 
Stopped.
```

### llama.cpp
```console
$ ls /snap/gemma3/components/163/llamacpp/bin/
 libggml-cpu-alderlake.so        libggml-cpu-sse42.so   LICENSE-linenoise     llama-imatrix        llama-qwen2vl-cli
 libggml-cpu-haswell.so          libggml-cpu-x64.so     llama-batched-bench   llama-llava-cli      llama-run
 libggml-cpu-icelake.so         'libggml-cuda*'         llama-bench           llama-minicpmv-cli   llama-server
 libggml-cpu-sandybridge.so      LICENSE-curl           llama-cli             llama-mtmd-cli       llama-tokenize
 libggml-cpu-sapphirerapids.so   LICENSE-httplib        llama-gemma3-cli      llama-perplexity     llama-tts
 libggml-cpu-skylakex.so         LICENSE-jsonhpp        llama-gguf-split      llama-quantize
```
