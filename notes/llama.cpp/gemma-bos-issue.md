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
(venv) $ ./build/bin/llama-cli -m models/gemma-3-270m-it.gguf -p "What is the capital of France?" -n 20 -sp
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
need for a `<bos>` token (apart for the one in the template, more on this later).

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
The issue with `--jinja` flag is that 2 `<bos>` tokens are added instead of only
one.
(venv) $ build/bin/llama-cli -m models/gemma-3-270m-it.gguf -c 0 -fa --jinja -p "Test" --verbose-prompt
...
Add output...
```

In chat.cpp we have the following:
```c++
    minja::chat_template_options tmpl_opts;
    // To avoid double BOS / EOS tokens, we're manually removing begining / trailing tokens
    // instead of using `chat_template_options.use_bos_token = false`, since these tokens
    // may be needed inside the template / between messages too.
    auto result = tmpl.apply(tmpl_inputs, tmpl_opts);
    if (inputs.add_bos && string_starts_with(result, tmpl.bos_token())) {
        result = result.substr(tmpl.bos_token().size());
    }
    if (inputs.add_eos && string_ends_with(result, tmpl.eos_token())) {
        result = result.substr(0, result.size() - tmpl.eos_token().size());
    }
    return result;
```

```console
(gdb) p result
$4 = "<bos><start_of_turn>user\nYou are a helpful assistant\n\nHello<end_of_turn>\n<start_of_turn>model\nHi there<end_of_turn>\n<start_of_turn>user\nHow are you?<end_of_turn>\n<start_of_turn>model\n"
(gdb) p inputs.add_bos
$5 = true
gdb) p string_starts_with(result, tmpl.bos_token())
$6 = true
```
And this does remove the bos token:
```console
(gdb) p result
$8 = "<start_of_turn>user\nYou are a helpful assistant\n\nHello<end_of_turn>\n<start_of_turn>model\nHi there<end_of_turn>\n<start_of_turn>user\nHow are you?<end_of_turn>\n<start_of_turn>model\n"
```
So we first have:
```c++
    auto chat_templates = common_chat_templates_init(model, params.chat_template);
```
```c++
common_chat_templates_ptr common_chat_templates_init(
    const struct llama_model * model,
    const std::string & chat_template_override,
    const std::string & bos_token_override,
    const std::string & eos_token_override)
{
    ...

    bool add_bos = false;
    bool add_eos = false;
    if (model) {
        const auto * vocab = llama_model_get_vocab(model);
        const auto get_token = [&](llama_token token, const char * name, const char * jinja_variable_name) {
            if (token == LLAMA_TOKEN_NULL) {
                if (default_template_src.find(jinja_variable_name) != std::string::npos
                    || template_tool_use_src.find(jinja_variable_name) != std::string::npos) {
                    LOG_WRN("common_chat_templates_init: warning: vocab does not have a %s token, jinja template won't work as intended.\n", name);
                }
                return std::string();
            }
            return common_token_to_piece(vocab, token, true);
        };
        token_bos = get_token(llama_vocab_bos(vocab), "BOS", "bos_token");
        token_eos = get_token(llama_vocab_eos(vocab), "EOS", "eos_token");
        add_bos = llama_vocab_get_add_bos(vocab);
        add_eos = llama_vocab_get_add_eos(vocab);
    }
}
```

```console
(gdb) until 586
common_chat_templates_init (model=0x555555dced00, chat_template_override="", bos_token_override="", eos_token_override="")
    at /home/danbev/work/ai/llama.cpp/common/chat.cpp:586
586	        add_bos = llama_vocab_get_add_bos(vocab);
(gdb) n
587	        add_eos = llama_vocab_get_add_eos(vocab);
(gdb) p add_bos
$28 = true
```

Now, back in main.cpp we later have the following:
```c++
    const bool add_bos = llama_vocab_get_add_bos(vocab) && !params.use_jinja;
```
Now, here we have `params.use_jinja` set to `true`, so this is `false` so this
means that the removal of the `<bos>` token in the chat template will not happen:
```c++
293             if (!params.system_prompt.empty() || !params.prompt.empty()) {
294                 common_chat_templates_inputs inputs;
295                 inputs.use_jinja = g_params->use_jinja;
296                 inputs.messages = chat_msgs;
297                 inputs.add_generation_prompt = !params.prompt.empty();
298
299                 prompt = common_chat_templates_apply(chat_templates.get(), inputs).prompt;
300             }
```
But notice that `inputs.add_bos` is not set so the default will be false. Perhaps
this should instead be set to `add_bos` from above?
```console
diff --git a/tools/main/main.cpp b/tools/main/main.cpp
index dc776f59e..04379201e 100644
--- a/tools/main/main.cpp
+++ b/tools/main/main.cpp
@@ -255,7 +255,7 @@ int main(int argc, char ** argv) {
         }
     }

-    const bool add_bos = llama_vocab_get_add_bos(vocab) && !params.use_jinja;
+    const bool add_bos = llama_vocab_get_add_bos(vocab);
     if (!llama_model_has_encoder(model)) {
         GGML_ASSERT(!llama_vocab_get_add_eos(vocab));
     }
@@ -294,6 +294,7 @@ int main(int argc, char ** argv) {
                 common_chat_templates_inputs inputs;
                 inputs.use_jinja = g_params->use_jinja;
                 inputs.messages = chat_msgs;
+                inputs.add_bos = add_bos;
                 inputs.add_generation_prompt = !params.prompt.empty();

                 prompt = common_chat_templates_apply(chat_templates.get(), inputs).prompt;

```
