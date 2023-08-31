## llm example

### Download the model
To run the example, you need to download the model. The model is too large to
include in the repository, so you need to download it from the Hugging Face.

```console
$ wget -P models https://huggingface.co/rustformers/redpajama-3b-ggml/resolve/main/RedPajama-INCITE-Base-3B-v1-q4_0.bin
```

With the model file downloaded we can then try out the example:
```console
$ cargo r
    Finished dev [unoptimized + debuginfo] target(s) in 0.10s
     Running `target/debug/llm-example`
Loaded hyperparameters
ggml ctx size = 1492.71 MB

Loaded tensor 8/388
Loaded tensor 16/388
Loaded tensor 24/388
Loaded tensor 32/388
Loaded tensor 40/388
Loaded tensor 48/388
Loaded tensor 56/388
Loaded tensor 64/388
Loaded tensor 72/388
Loaded tensor 80/388
Loaded tensor 88/388
Loaded tensor 96/388
Loaded tensor 104/388
Loaded tensor 112/388
Loaded tensor 120/388
Loaded tensor 128/388
Loaded tensor 136/388
Loaded tensor 144/388
Loaded tensor 152/388
Loaded tensor 160/388
Loaded tensor 168/388
Loaded tensor 176/388
Loaded tensor 184/388
Loaded tensor 192/388
Loaded tensor 200/388
Loaded tensor 208/388
Loaded tensor 216/388
Loaded tensor 224/388
Loaded tensor 232/388
Loaded tensor 240/388
Loaded tensor 248/388
Loaded tensor 256/388
Loaded tensor 264/388
Loaded tensor 272/388
Loaded tensor 280/388
Loaded tensor 288/388
Loaded tensor 296/388
Loaded tensor 304/388
Loaded tensor 312/388
Loaded tensor 320/388
Loaded tensor 328/388
Loaded tensor 336/388
Loaded tensor 344/388
Loaded tensor 352/388
Loaded tensor 360/388
Loaded tensor 368/388
Loaded tensor 376/388
Loaded tensor 384/388
Loading of model complete
Model size = 1493.11 MB / num tensors = 388
Model fully loaded! Elapsed: 1183ms
<|padding|>Rust is a cool programming language because and other results about as the result was sent by

Inference stats:
feed_prompt_duration: 15094ms
prompt_tokens: 9
predict_duration: 32253ms
predict_tokens: 19
per_token_duration: 1697.526ms
```
