### Gemma 3-270m bos token issue
This is an issue that we discovered after converting Gemma 3-270m to GGUF format.
and running these models.

### Behavior of base model (non instruction tuned)
When running the Gemma 3-270m model, it was observed that the model was
generating double BOS tokens.
```console
(venv) $ ./build/bin/llama-cli -m models/gemma-3-270m.gguf -p "hello" -n 20 -sp
llama_model_loader: loaded meta data with 33 key-value pairs and 236 tensors from models/gemma-3-270m.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = gemma3
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Gemma 3 270m
llama_model_loader: - kv   3:                           general.basename str              = gemma-3
llama_model_loader: - kv   4:                         general.size_label str              = 270M
llama_model_loader: - kv   5:                            general.license str              = gemma
llama_model_loader: - kv   6:                               general.tags arr[str,4]       = ["gemma3", "gemma", "google", "text-g...
llama_model_loader: - kv   7:                      gemma3.context_length u32              = 32768
llama_model_loader: - kv   8:                    gemma3.embedding_length u32              = 640
llama_model_loader: - kv   9:                         gemma3.block_count u32              = 18
llama_model_loader: - kv  10:                 gemma3.feed_forward_length u32              = 2048
llama_model_loader: - kv  11:                gemma3.attention.head_count u32              = 4
llama_model_loader: - kv  12:    gemma3.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  13:                gemma3.attention.key_length u32              = 256
llama_model_loader: - kv  14:              gemma3.attention.value_length u32              = 256
llama_model_loader: - kv  15:                          general.file_type u32              = 1
llama_model_loader: - kv  16:                      gemma3.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  17:            gemma3.attention.sliding_window u32              = 512
llama_model_loader: - kv  18:             gemma3.attention.head_count_kv u32              = 1
llama_model_loader: - kv  19:               general.quantization_version u32              = 2
llama_model_loader: - kv  20:                       tokenizer.ggml.model str              = llama
llama_model_loader: - kv  21:                         tokenizer.ggml.pre str              = default
llama_model_loader: - kv  22:                      tokenizer.ggml.tokens arr[str,262144]  = ["<pad>", "<eos>", "<bos>", "<unk>", ...
llama_model_loader: - kv  23:                      tokenizer.ggml.scores arr[f32,262144]  = [-1000.000000, -1000.000000, -1000.00...
llama_model_loader: - kv  24:                  tokenizer.ggml.token_type arr[i32,262144]  = [3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3, ...
llama_model_loader: - kv  25:                tokenizer.ggml.bos_token_id u32              = 2
llama_model_loader: - kv  26:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  27:            tokenizer.ggml.unknown_token_id u32              = 3
llama_model_loader: - kv  28:            tokenizer.ggml.padding_token_id u32              = 0
llama_model_loader: - kv  29:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  30:               tokenizer.ggml.add_sep_token bool             = false
llama_model_loader: - kv  31:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  32:            tokenizer.ggml.add_space_prefix bool             = false

...

<bos>hello. I'm using a new version of the game. I have been playing it for a while
```

The pretrained/base model has following in `tokenizer_config.json`:
```console
{
  "add_bos_token": true,
  ```
}
```
The pretrained/basemodels learn during training that a `<bos` token means
this is the start of a new prompt/sentence. Without this the model will not
know when to start.

### Behavior of instruction tuned model
```console
(venv) $ ./build/bin/llama-cli -m models/gemma-3-270m-it.gguf -p "What is the capital of France?" -n 20 -sp -no-cnv
...
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = gemma3
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Gemma 3 270m It
llama_model_loader: - kv   3:                           general.finetune str              = it
llama_model_loader: - kv   4:                           general.basename str              = gemma-3
llama_model_loader: - kv   5:                         general.size_label str              = 270M
llama_model_loader: - kv   6:                            general.license str              = gemma
llama_model_loader: - kv   7:                   general.base_model.count u32              = 1
llama_model_loader: - kv   8:                  general.base_model.0.name str              = Gemma 3 270m
llama_model_loader: - kv   9:          general.base_model.0.organization str              = Google
llama_model_loader: - kv  10:              general.base_model.0.repo_url str              = https://huggingface.co/google/gemma-3...
llama_model_loader: - kv  11:                               general.tags arr[str,4]       = ["gemma3", "gemma", "google", "text-g...
llama_model_loader: - kv  12:                      gemma3.context_length u32              = 32768
llama_model_loader: - kv  13:                    gemma3.embedding_length u32              = 640
llama_model_loader: - kv  14:                         gemma3.block_count u32              = 18
llama_model_loader: - kv  15:                 gemma3.feed_forward_length u32              = 2048
llama_model_loader: - kv  16:                gemma3.attention.head_count u32              = 4
llama_model_loader: - kv  17:    gemma3.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  18:                gemma3.attention.key_length u32              = 256
llama_model_loader: - kv  19:              gemma3.attention.value_length u32              = 256
llama_model_loader: - kv  20:                          general.file_type u32              = 1
llama_model_loader: - kv  21:                      gemma3.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  22:            gemma3.attention.sliding_window u32              = 512
llama_model_loader: - kv  23:             gemma3.attention.head_count_kv u32              = 1
llama_model_loader: - kv  24:               general.quantization_version u32              = 2
llama_model_loader: - kv  25:                       tokenizer.ggml.model str              = llama
llama_model_loader: - kv  26:                         tokenizer.ggml.pre str              = default
llama_model_loader: - kv  27:                      tokenizer.ggml.tokens arr[str,262144]  = ["<pad>", "<eos>", "<bos>", "<unk>", ...
llama_model_loader: - kv  28:                      tokenizer.ggml.scores arr[f32,262144]  = [-1000.000000, -1000.000000, -1000.00...
llama_model_loader: - kv  29:                  tokenizer.ggml.token_type arr[i32,262144]  = [3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3, ...
llama_model_loader: - kv  30:                tokenizer.ggml.bos_token_id u32              = 2
llama_model_loader: - kv  31:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  32:            tokenizer.ggml.unknown_token_id u32              = 3
llama_model_loader: - kv  33:            tokenizer.ggml.padding_token_id u32              = 0
llama_model_loader: - kv  34:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  35:               tokenizer.ggml.add_sep_token bool             = false
llama_model_loader: - kv  36:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  37:                    tokenizer.chat_template str              = {{ bos_token }}\n{%- if messages[0]['r...
llama_model_loader: - kv  38:            tokenizer.ggml.add_space_prefix bool             = false
...

<bos>What is the capital of France?

A) Paris
B) London
C) Rome
D) Berlin

**Answer:**
```
The instruction tuned model has the following in `tokenizer_config.json`:
```console
{
  "add_bos_token": true,
  ...
}
```
Now, the instruction tuned model have chat/conversation templates and their
training data contains explicit role markers like `<start_of_turn>`. This already
tells the model that this is the beginning of a user message and there is no
need for a `<bos>` token.

So the above is not correct output for `llama-cli`.

And running with `--jinja` flag, the output is:
```console
(venv) $ build/bin/llama-cli -m models/gemma-3-270m-it.gguf -c 0 -fa --jinja -p "Test" --verbose-prompt
...
tokenize: Added a BOS token to the prompt as specified by the model but the prompt also starts with a BOS token. So now the final prompt starts with 2 BOS tokens. Are you sure this is what you want?
main: prompt: 'Test'
main: number of tokens in prompt = 11
     2 -> '<bos>'
     2 -> '<bos>'
   105 -> '<start_of_turn>'
  2364 -> 'user'
   107 -> '
'
  3694 -> 'Test'
   106 -> '<end_of_turn>'
   107 -> '
'
   105 -> '<start_of_turn>'
  4368 -> 'model'
   107 -> '
'

main: interactive mode on.
sampler seed: 1414592202
sampler params:
	repeat_last_n = 64, repeat_penalty = 1.000, frequency_penalty = 0.000, presence_penalty = 0.000
	dry_multiplier = 0.000, dry_base = 1.750, dry_allowed_length = 2, dry_penalty_last_n = 32768
	top_k = 40, top_p = 0.950, min_p = 0.050, xtc_probability = 0.000, xtc_threshold = 0.100, typical_p = 1.000, top_n_sigma = -1.000, temp = 0.800
	mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000
sampler chain: logits -> logit-bias -> penalties -> dry -> top-n-sigma -> top-k -> typical -> top-p -> min-p -> xtc -> temp-ext -> dist
generate: n_ctx = 32768, n_batch = 2048, n_predict = -1, n_keep = 0

== Running in interactive mode. ==
 - Press Ctrl+C to interject at any time.
 - Press Return to return control to the AI.
 - To return control without starting a new line, end your input with '/'.
 - If you want to submit another line, end your input with '\'.
 - Not using system message. To change it, set a different value via -sys PROMPT

user
Test
model
Okay, I understand. I'm ready to help! I'm
>
```

### Workaround
The workaround for this was to add the following to `convert_hf_to_gguf.py`:
```python
class Gemma3Model(TextModel):
    model_arch = gguf.MODEL_ARCH.GEMMA3
    norm_shift = 1.0  # Gemma3RMSNorm adds 1.0 to the norm value

    def set_vocab(self):
        self._set_vocab_sentencepiece()

        self.gguf_writer.add_add_space_prefix(False)

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model)
        if tokenizer.chat_template is None:
            self.gguf_writer.add_add_bos_token(True)
        else:
            self.gguf_writer.add_add_bos_token(False)

```
With this change the bos token is added for the base model but not for the
instruction tuned model. But here is still an issue with the `--jinja` flag.

### Issue with `--jinja` flag
Even with the workaround in place the following output is generated when the
`--jinja` flag is used with the instruction tuned model:
```console
(venv) $ build/bin/llama-cli -m models/gemma-3-270m-it.gguf -c 0 -fa --jinja -p "Test" --verbose-prompt
...

main: prompt: 'Test'
main: number of tokens in prompt = 10
     2 -> '<bos>'
   105 -> '<start_of_turn>'
  2364 -> 'user'
   107 -> '
'
  3694 -> 'Test'
   106 -> '<end_of_turn>'
   107 -> '
'
   105 -> '<start_of_turn>'
  4368 -> 'model'
   107 -> '
'
```
Before the workaround there where two `<bos>` tokens in the prompt.

```console
(venv) $ gdb --args build/bin/llama-cli -m models/gemma-3-270m-it.gguf -c 0 -fa --jinja -p "Test" --verbose-prompt
(gdb) br main.cpp:153
(gdb) r
Thread 1 "llama-cli" hit Breakpoint 1, main (argc=10, argv=0x7fffffffd5f8) at /home/danbev/work/ai/llama.cpp/tools/main/main.cpp:153
warning: Source file is more recent than executable.
153	    auto chat_templates = common_chat_templates_init(model, params.chat_template);

```
```c++
common_chat_templates_ptr common_chat_templates_init(
    const struct llama_model * model,
    const std::string & chat_template_override,
    const std::string & bos_token_override,
    const std::string & eos_token_override)
{
    std::string default_template_src;
    std::string template_tool_use_src;

    bool has_explicit_template = !chat_template_override.empty();
    if (chat_template_override.empty()) {
        GGML_ASSERT(model != nullptr);
        const auto * str = llama_model_chat_template(model, /* name */ nullptr);
```
So this is getting the chat template from the model.
```console
(gdb) set print elements 0
(gdb) p (char*) str
$5 = 0x55555672ae50 "{{ bos_token }}\n{%- if messages[0]['role'] == 'system' -%}\n    {%- if messages[0]['content'] is string -%}\n        {%- set first_user_prefix = messages[0]['content'] + '\n\n' -%}\n    {%- else -%}\n        {%- set first_user_prefix = messages[0]['content'][0]['text'] + '\n\n' -%}\n    {%- endif -%}\n    {%- set loop_messages = messages[1:] -%}\n{%- else -%}\n    {%- set first_user_prefix = \"\" -%}\n    {%- set loop_messages = messages -%}\n{%- endif -%}\n{%- for message in loop_messages -%}\n    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}\n        {{ raise_exception(\"Conversation roles must alternate user/assistant/user/assistant/...\") }}\n    {%- endif -%}\n    {%- if (message['role'] == 'assistant') -%}\n        {%- set role = \"model\" -%}\n    {%- else -%}\n        {%- set role = message['role'] -%}\n    {%- endif -%}\n    {{ '<start_of_turn>' + role + '\n' + (first_user_prefix if loop.first else \"\") }}\n    {%- if message['content'] is string -%}\n        {{ message['content'] | trim }}\n    {%- elif message['content'] is iterable -%}\n        {%- for item in message['content'] -%}\n", ' ' <repeats 12 times>, "{%- if item['type'] == 'image' -%}\n", ' ' <repeats 16 times>, "{{ '<start_of_image>' }}\n", ' ' <repeats 12 times>, "{%- elif item['type'] == 'text' -%}\n", ' ' <repeats 16 times>, "{{ item['text'] | trim }}\n", ' ' <repeats 12 times>, "{%- endif -%}\n        {%- endfor -%}\n    {%- else -%}\n        {{ raise_exception(\"Invalid content type\") }}\n    {%- endif -%}\n    {{ '<end_of_turn>\n' }}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n    {{'<start_of_turn>model\n'}}\n{%- endif -%}\n"
```
Notice that this has the `bos_token` is in chat template which is as it should
be. It should be in control of over the `bos_token`. The problem we had before
was that because the model had `add_bos_token` set to `true`, it would also
insert a `<bos>` token at the start of the prompt, resultin in the two `<bos>`
tokens.

