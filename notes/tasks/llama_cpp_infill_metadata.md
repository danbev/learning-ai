## Infill Metadata task
This task is about adding metadata for special [infill](../infill.md) tokens.
Currently, the infill tokens are specified in
[llama.cpp](https://github.com/ggerganov/llama.cpp/blob/4bd0f93e4ab4fe6682e7d0241c1bdec1397e954a/llama.cpp#L2058-L2062): 
```c++
    id special_prefix_id = 32007;
    id special_middle_id = 32009;
    id special_suffix_id = 32008;
    id special_eot_id    = 32010;
```
These are the token ids that CodeLlama uses, but other models that support
infill might not use the sames ids. This was discovered when trying to use
[CodeGemma](https://huggingface.co/google/codegemma-7b-it/blob/main/tokenizer_config.json#L541-L564)
which uses the following ids:
```json
    "67": {
      "content": "<|fim_prefix|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": false
    },
    "68": {
      "content": "<|fim_middle|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": false
    },
    "69": {
      "content": "<|fim_suffix|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": false
    },
```
I've tried setting the above ids to these instead to see that it works with
CodeGemma, but and it kind of works, but it's not perfect:
```
id special_prefix_id = 67;
id special_middle_id = 68;
id special_suffix_id = 69;
id special_eot_id    = 70;
```

This task is about adding metadata to the infill tokens so that different models
can specify their own token ids and things will still work.


### Implementation
TODO:
