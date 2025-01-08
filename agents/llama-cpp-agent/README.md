## Llama.cpp Agent
This is an exploration into AI Agents for llama.cpp. The goals is to gain some
hands-on experience with AI agents and understand how they can be built.

_This is very much a exploration at this point and these ideas might be valid in practice_

### Overview

The idea is to enable agent to work with llama.cpp and be able to run the
locally. The tools that an Agent uses will be defined using the Web Assembly
Component Model and an interface defined in Web Assembly Interface Types (WIT).

The motivation for choosing this is that using the Web Assembly Component Model,
we can define an interface for the tools that an agent uses and then implement
the tool in any language that supports Web Assembly Component Model. This
includes Rust, Python, JavaScript.

Another motivation for using this is that using WASM we can run the agent in a
sandboxed environment and hence the agent can be trusted to run on the
user's machine or perhaps in a server environment. Perhaps this could enable
agents to be deployed in server environments but in a safe way for companies
that want to offer this feature as a service.

So the idea is that a tools interface be defined in WIT and then tool
implementors would use wit-bindgen (or similar tools) to generate the interface
in their language of choice. These would then be compiled to WASM and the agent
would would load and use the WASM tools need to accomplish its tasks.

## Tools
Tools are what agents use to accomplish their tasks. These tools are defined
using the Web Interface Types (WIT) and implemented as Web Assembly Component
Models.


### Building Tools
The following shows an example of building the Echo tool which just
returns/echos the input it recieves:
```console
$ make echo-tool
```
This will produce a `tools/echo/target/wasm32-wasip1/debug/echo_tool.wasm` which
is a normal wasm (not a web assembly component model module that is).

Then we create a component model module from the wasm file:
```console
$ make echo-component
```
This will produce `components/echo-tool-component.wasm` which is a web assembly
component model module.

This can then be used with the `tool-runner` which is really just for testing
the component standalone:
```console
$ make run-echo-tool
cd tool-runner && cargo build
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.08s
cd tools/echo && cargo build --target wasm32-wasip1
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.01s
wasm-tools component new tools/echo/target/wasm32-wasip1/debug/echo_tool.wasm \
    --adapt wit-lib/wasi_snapshot_preview1.reactor.wasm \
    -o components/echo-tool-component.wasm
cd tool-runner && cargo run -- -c ../components/echo-tool-component.wasm --value "Hello"
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.07s
     Running `target/debug/tool-runner -c ../components/echo-tool-component.wasm --value Hello`
Component path: "../components/echo-tool-component.wasm"
Tool metadata:
  Name: Echo
  Description: Echos the passed in value
  Version: 0.1.0
  Parameters:
    - value: Value to be echoed (string)

Executing tool...
[Success] Tool output: Hello
```
There is also a print tool but this was mainly to make sure that wasi is working
and that it is possible to print to the console from the wasm module.

### Agent

### Running the agent
```console
$ make run-agent
```
<details><summary>example output</summary>

```console
cd agent && cargo run -- -m ../models/Phi-3-mini-4k-instruct-q4.gguf -p "Please echo back 'Something to echo'"
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.09s
     Running `target/debug/main -m ../models/Phi-3-mini-4k-instruct-q4.gguf -p 'Please echo back '\''Something to echo'\'''`
llama_model_loader: loaded meta data with 24 key-value pairs and 195 tensors from ../models/Phi-3-mini-4k-instruct-q4.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = phi3
llama_model_loader: - kv   1:                               general.name str              = Phi3
llama_model_loader: - kv   2:                        phi3.context_length u32              = 4096
llama_model_loader: - kv   3:                      phi3.embedding_length u32              = 3072
llama_model_loader: - kv   4:                   phi3.feed_forward_length u32              = 8192
llama_model_loader: - kv   5:                           phi3.block_count u32              = 32
llama_model_loader: - kv   6:                  phi3.attention.head_count u32              = 32
llama_model_loader: - kv   7:               phi3.attention.head_count_kv u32              = 32
llama_model_loader: - kv   8:      phi3.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv   9:                  phi3.rope.dimension_count u32              = 96
llama_model_loader: - kv  10:                          general.file_type u32              = 15
llama_model_loader: - kv  11:                       tokenizer.ggml.model str              = llama
llama_model_loader: - kv  12:                         tokenizer.ggml.pre str              = default
llama_model_loader: - kv  13:                      tokenizer.ggml.tokens arr[str,32064]   = ["<unk>", "<s>", "</s>", "<0x00>", "<...
llama_model_loader: - kv  14:                      tokenizer.ggml.scores arr[f32,32064]   = [0.000000, 0.000000, 0.000000, 0.0000...
llama_model_loader: - kv  15:                  tokenizer.ggml.token_type arr[i32,32064]   = [2, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, ...
llama_model_loader: - kv  16:                tokenizer.ggml.bos_token_id u32              = 1
llama_model_loader: - kv  17:                tokenizer.ggml.eos_token_id u32              = 32000
llama_model_loader: - kv  18:            tokenizer.ggml.unknown_token_id u32              = 0
llama_model_loader: - kv  19:            tokenizer.ggml.padding_token_id u32              = 32000
llama_model_loader: - kv  20:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  21:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  22:                    tokenizer.chat_template str              = {{ bos_token }}{% for message in mess...
llama_model_loader: - kv  23:               general.quantization_version u32              = 2
llama_model_loader: - type  f32:   65 tensors
llama_model_loader: - type q4_K:   81 tensors
llama_model_loader: - type q5_K:   32 tensors
llama_model_loader: - type q6_K:   17 tensors
llm_load_vocab: control-looking token:  32007 '<|end|>' was not control-type; this is probably a bug in the model. its type will be overridden
llm_load_vocab: control-looking token:  32000 '<|endoftext|>' was not control-type; this is probably a bug in the model. its type will be overridden
llm_load_vocab: control token:      2 '</s>' is not marked as EOG
llm_load_vocab: control token:      1 '<s>' is not marked as EOG
llm_load_vocab: special tokens cache size = 67
llm_load_vocab: token to piece cache size = 0.1690 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = phi3
llm_load_print_meta: vocab type       = SPM
llm_load_print_meta: n_vocab          = 32064
llm_load_print_meta: n_merges         = 0
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 4096
llm_load_print_meta: n_embd           = 3072
llm_load_print_meta: n_layer          = 32
llm_load_print_meta: n_head           = 32
llm_load_print_meta: n_head_kv        = 32
llm_load_print_meta: n_rot            = 96
llm_load_print_meta: n_swa            = 2047
llm_load_print_meta: n_embd_head_k    = 96
llm_load_print_meta: n_embd_head_v    = 96
llm_load_print_meta: n_gqa            = 1
llm_load_print_meta: n_embd_k_gqa     = 3072
llm_load_print_meta: n_embd_v_gqa     = 3072
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-05
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 8192
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 2
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 10000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 4096
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: ssm_dt_b_c_rms   = 0
llm_load_print_meta: model type       = 3B
llm_load_print_meta: model ftype      = Q4_K - Medium
llm_load_print_meta: model params     = 3.82 B
llm_load_print_meta: model size       = 2.23 GiB (5.01 BPW)
llm_load_print_meta: general.name     = Phi3
llm_load_print_meta: BOS token        = 1 '<s>'
llm_load_print_meta: EOS token        = 32000 '<|endoftext|>'
llm_load_print_meta: EOT token        = 32007 '<|end|>'
llm_load_print_meta: UNK token        = 0 '<unk>'
llm_load_print_meta: PAD token        = 32000 '<|endoftext|>'
llm_load_print_meta: LF token         = 13 '<0x0A>'
llm_load_print_meta: EOG token        = 32000 '<|endoftext|>'
llm_load_print_meta: EOG token        = 32007 '<|end|>'
llm_load_print_meta: max token length = 48
llm_load_tensors: tensor 'token_embd.weight' (q4_K) (and 194 others) cannot be used with preferred buffer type CPU_AARCH64, using CPU instead
llm_load_tensors:   CPU_Mapped model buffer size =  2281.66 MiB
...........................................................................................
llama_new_context_with_model: n_seq_max     = 1
llama_new_context_with_model: n_ctx         = 2048
llama_new_context_with_model: n_ctx_per_seq = 2048
llama_new_context_with_model: n_batch       = 2048
llama_new_context_with_model: n_ubatch      = 512
llama_new_context_with_model: flash_attn    = 0
llama_new_context_with_model: freq_base     = 10000.0
llama_new_context_with_model: freq_scale    = 1
llama_new_context_with_model: n_ctx_per_seq (2048) < n_ctx_train (4096) -- the full capacity of the model will not be utilized
llama_kv_cache_init:        CPU KV buffer size =   768.00 MiB
llama_new_context_with_model: KV self size  =  768.00 MiB, K (f16):  384.00 MiB, V (f16):  384.00 MiB
llama_new_context_with_model:        CPU  output buffer size =     0.12 MiB
llama_new_context_with_model:        CPU compute buffer size =   168.01 MiB
llama_new_context_with_model: graph nodes  = 1286
llama_new_context_with_model: graph splits = 1
Prompt: <|user|>
 You are a helpful AI assistant. You have access to an Echo tool. When asked to echo something, respond ONLY with the exact tool command format and include the complete text to be echoed.

 Example interaction:
 User: Please echo back 'hello'
 Assistant: USE_TOOL: Echo, value=hello

 Available tool:
 Echo - Echoes back the input text
 Usage: USE_TOOL: Echo, value=<text to echo>
 Note: Make sure to include the complete text after 'value='

 Please echo back 'Something to echo' <|end|>
<|assistant|>
 USE_TOOL: Echo, value=Something
Executing tool: Echo, params: 1
  - value: Something
Execution result: Something
Agent response: Something
```

</details>

#### Download model
There is no particular model that is needed for this agent, however is needs
to be an instruction trained model.
```console
$ make download-phi-mini-instruct 
```

### Setup/Configuration
```console
$ rustup target add wasm32-wasip1
```

```console
$ cargo install wasm-tools
```

```console
$ cargo install wac-cli
```
