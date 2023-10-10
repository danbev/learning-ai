## llm-chain example using llama driver


### Setup
This example requires that a llm model is downloaded and placed in the models
dirctory:
```console
$ wget -P models https://huggingface.co/localmodels/Llama-2-7B-Chat-ggml/resolve/main/llama-2-7b-chat.ggmlv3.q4_0.bin
```

### Running
```console
$ cargo r
    Finished dev [unoptimized + debuginfo] target(s) in 0.09s
     Running `target/debug/llm-chains-llama-example`
llama.cpp: loading model from ./models/llama-2-7b-chat.ggmlv3.q4_0.bin
llama_model_load_internal: format     = ggjt v3 (latest)
llama_model_load_internal: n_vocab    = 32000
llama_model_load_internal: n_ctx      = 512
llama_model_load_internal: n_embd     = 4096
llama_model_load_internal: n_mult     = 256
llama_model_load_internal: n_head     = 32
llama_model_load_internal: n_layer    = 32
llama_model_load_internal: n_rot      = 128
llama_model_load_internal: ftype      = 2 (mostly Q4_0)
llama_model_load_internal: n_ff       = 11008
llama_model_load_internal: n_parts    = 1
llama_model_load_internal: model size = 7B
llama_model_load_internal: ggml ctx size =    0.07 MB
llama_model_load_internal: mem required  = 5407.71 MB (+ 1026.00 MB per state)
.
llama_init_from_file: kv self size  =  256.00 MB
Query: How are you?
 I hope you're doing well. Hinweis: This is a common greeting in German, used to acknowledge someone's presence or to ask how they are feeling. It is a polite and friendly way to start a conversation.
```
