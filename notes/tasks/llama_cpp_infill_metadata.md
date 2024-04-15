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
CodeGemma, which it did once I just the correct model (I initially used the
model that was trained for instruction following which had `it` in it's name
when I should have been using the model that was trained for code completion
which does not have `it` in it's name):
```
id special_prefix_id = 67;
id special_middle_id = 68;
id special_suffix_id = 69;
id special_eot_id    = 70;
```

This task is about adding metadata to the infill tokens so that different models
can specify their own token ids and things will still work.


### Implementation
Looking in [gguf-py/constants.py](https://github.com/ggerganov/llama.cpp/blob/4bd0f93e4ab4fe6682e7d0241c1bdec1397e954a/gguf-py/gguf/constants.py#L73) there is a Tokenizer class and it contains
constants for `BOS_ID`, `EOS_ID`, `PAD_ID`, `UNK_ID`, etc. Perhaps a start would
be to add these infill tokens to this class.

What we want is to add new properties to the `Tokenizer` class in `constants.py`
which can be used to add key-value pairs to the GGUF model file. For example:
```python
    class Tokenizer:
        ...
        PREFIX_ID        = "tokenizer.ggml.prefix_token_id"
        MIDDLE_ID        = "tokenizer.ggml.middle_token_id"
        SUFFIX_ID        = "tokenizer.ggml.suffix_token_id"
        EOT_ID           = "tokenizer.ggml.eot_token_id"
```
These will then have to be added to the generated GGUF file.
TODO: How do we add these to the GGUF file?

The other special tokens can be inspected in a gguf file using the
`gguf-dump.py`:
```console
(venv3) $ gguf-py/scripts/gguf-dump.py models/codegemma-7b-f16.gguf
* Loading: models/codegemma-7b-f16.gguf
* File is LITTLE endian, script is running on a LITTLE endian host.

* Dumping 24 key/value pair(s)
      1: UINT32     |        1 | GGUF.version = 3
      2: UINT64     |        1 | GGUF.tensor_count = 254
      3: UINT64     |        1 | GGUF.kv_count = 21
      4: STRING     |        1 | general.architecture = 'gemma'
      5: STRING     |        1 | general.name = 'codegemma-7b'
      6: UINT32     |        1 | gemma.context_length = 8192
      7: UINT32     |        1 | gemma.block_count = 28
      8: UINT32     |        1 | gemma.embedding_length = 3072
      9: UINT32     |        1 | gemma.feed_forward_length = 24576
     10: UINT32     |        1 | gemma.attention.head_count = 16
     11: UINT32     |        1 | gemma.attention.head_count_kv = 16
     12: UINT32     |        1 | gemma.attention.key_length = 256
     13: UINT32     |        1 | gemma.attention.value_length = 256
     14: FLOAT32    |        1 | gemma.attention.layer_norm_rms_epsilon = 9.999999974752427e-07
     15: STRING     |        1 | tokenizer.ggml.model = 'llama'
     16: UINT32     |        1 | tokenizer.ggml.bos_token_id = 2
     17: UINT32     |        1 | tokenizer.ggml.eos_token_id = 1
     18: UINT32     |        1 | tokenizer.ggml.padding_token_id = 0
     19: UINT32     |        1 | tokenizer.ggml.unknown_token_id = 3
     20: [STRING]   |   256128 | tokenizer.ggml.tokens
     21: [FLOAT32]  |   256128 | tokenizer.ggml.scores
     22: [INT32]    |   256128 | tokenizer.ggml.token_type
     23: UINT32     |        1 | general.quantization_version = 2
     24: UINT32     |        1 | general.file_type = 1
```
So we want to have these new properties added to the GGUF file.

This will then enable us to add them to the `llm_kv` enum in `llama.cpp`:
```c++
enum llm_kv {
    ...
    LLM_KV_TOKENIZER_PREFIX_ID,
    LLM_KV_TOKENIZER_SUFFIX_ID,
    LLM_KV_TOKENIZER_MIDDLE_ID,
    LLM_KV_TOKENIZER_EOT_ID,
};
```
```c++
static const std::map<llm_kv, const char *> LLM_KV_NAMES = {
    { LLM_KV_GENERAL_ARCHITECTURE,          "general.architecture"                  },
    { LLM_KV_GENERAL_QUANTIZATION_VERSION,  "general.quantization_version"          },
    { LLM_KV_GENERAL_ALIGNMENT,             "general.alignment"                     },
    { LLM_KV_GENERAL_NAME,                  "general.name"                          },
    { LLM_KV_GENERAL_AUTHOR,                "general.author"                        },
    { LLM_KV_GENERAL_VERSION,               "general.version"                       },
    { LLM_KV_GENERAL_URL,                   "general.url"                           },
    { LLM_KV_GENERAL_DESCRIPTION,           "general.description"                   },
    { LLM_KV_GENERAL_LICENSE,               "general.license"                       },
    { LLM_KV_GENERAL_SOURCE_URL,            "general.source.url"                    },
    { LLM_KV_GENERAL_SOURCE_HF_REPO,        "general.source.huggingface.repository" },

    { LLM_KV_VOCAB_SIZE,                    "%s.vocab_size"            },
    { LLM_KV_CONTEXT_LENGTH,                "%s.context_length"        },
    { LLM_KV_EMBEDDING_LENGTH,              "%s.embedding_length"      },
    { LLM_KV_BLOCK_COUNT,                   "%s.block_count"           },
    { LLM_KV_FEED_FORWARD_LENGTH,           "%s.feed_forward_length"   },
    { LLM_KV_USE_PARALLEL_RESIDUAL,         "%s.use_parallel_residual" },
    { LLM_KV_TENSOR_DATA_LAYOUT,            "%s.tensor_data_layout"    },
    { LLM_KV_EXPERT_COUNT,                  "%s.expert_count"          },
    { LLM_KV_EXPERT_USED_COUNT,             "%s.expert_used_count"     },
    { LLM_KV_POOLING_TYPE ,                 "%s.pooling_type"          },
    { LLM_KV_LOGIT_SCALE,                   "%s.logit_scale"           },

    { LLM_KV_TOKENIZER_BOS_ID,              "tokenizer.ggml.bos_token_id"       },
    { LLM_KV_TOKENIZER_EOS_ID,              "tokenizer.ggml.eos_token_id"       },
    { LLM_KV_TOKENIZER_UNK_ID,              "tokenizer.ggml.unknown_token_id"   },
    { LLM_KV_TOKENIZER_SEP_ID,              "tokenizer.ggml.seperator_token_id" },
    { LLM_KV_TOKENIZER_PAD_ID,              "tokenizer.ggml.padding_token_id"   },
    { LLM_KV_TOKENIZER_CLS_ID,              "tokenizer.ggml.cls_token_id"       },
    { LLM_KV_TOKENIZER_MASK_ID,             "tokenizer.ggml.mask_token_id"      },
    { LLM_KV_TOKENIZER_ADD_BOS,             "tokenizer.ggml.add_bos_token"      },
    { LLM_KV_TOKENIZER_ADD_EOS,             "tokenizer.ggml.add_eos_token"      },
    { LLM_KV_TOKENIZER_ADD_PREFIX,          "tokenizer.ggml.add_space_prefix"   },
    { LLM_KV_TOKENIZER_HF_JSON,             "tokenizer.huggingface.json"        },
    { LLM_KV_TOKENIZER_RWKV,                "tokenizer.rwkv.world"              },

    { LLM_KV_TOKENIZER_PREFIX_ID,           "tokenizer.ggml.prefix_token_id"    },
    { LLM_KV_TOKENIZER_SUFFIX_ID,           "tokenizer.ggml.suffix_token_id"    },
    { LLM_KV_TOKENIZER_MIDDLE_ID,           "tokenizer.ggml.middle_token_id"    },
    { LLM_KV_TOKENIZER_EOT_ID,              "tokenizer.ggml.eot_token_id"       },
};
```
And when the vocabulary is loaded in `llm_load_vocab` we can add these special
tokens types:
```c++
        const std::vector<std::pair<enum llm_kv, int32_t &>> special_token_types = {
            { LLM_KV_TOKENIZER_BOS_ID,    vocab.special_bos_id  },
            { LLM_KV_TOKENIZER_EOS_ID,    vocab.special_eos_id  },
            { LLM_KV_TOKENIZER_UNK_ID,    vocab.special_unk_id  },
            { LLM_KV_TOKENIZER_SEP_ID,    vocab.special_sep_id  },
            { LLM_KV_TOKENIZER_PAD_ID,    vocab.special_pad_id  },
            { LLM_KV_TOKENIZER_CLS_ID,    vocab.special_cls_id  },
            { LLM_KV_TOKENIZER_MASK_ID,   vocab.special_mask_id },
            { LLM_KV_TOKENIZER_PREFIX_ID, vocab.special_prefix_id },
            { LLM_KV_TOKENIZER_SUFFIX_ID, vocab.special_suffix_id },
            { LLM_KV_TOKENIZER_MIDDLE_ID, vocab.special_middle_id },
            { LLM_KV_TOKENIZER_EOT_ID,    vocab.special_eot_id },
        };
```

For CodeGemma, the infill tokens are in the `added_tokens` array in
tokenizer.json. `added_tokens` are tokens that have been added to the tokenizer
beyond the standard vocabulary.
```console
(venv3) $ cat ../codegemma-7b/tokenizer.json | jq '.added_tokens[67:71]'
[
  {
    "id": 67,
    "content": "<|fim_prefix|>",
    "single_word": false,
    "lstrip": false,
    "rstrip": false,
    "normalized": false,
    "special": false
  },
  {
    "id": 68,
    "content": "<|fim_middle|>",
    "single_word": false,
    "lstrip": false,
    "rstrip": false,
    "normalized": false,
    "special": false
  },
  {
    "id": 69,
    "content": "<|fim_suffix|>",
    "single_word": false,
    "lstrip": false,
    "rstrip": false,
    "normalized": false,
    "special": false
  },
  {
    "id": 70,
    "content": "<|file_separator|>",
    "single_word": false,
    "lstrip": false,
    "rstrip": false,
    "normalized": false,
    "special": false
  }
]
```
And we can check that the vocabulary also contains these tokens:
```console
(venv3) $ cat ../codegemma-7b/tokenizer.json | jq '.model.vocab["<|fim_prefix|>"]'
67
```
One thing I noticed is that CodeLlama does not have any entried related to
Fill-In-the-Middle (FIM)/Infill tokens in it's `add_tokens` array:
```
(venv3) $ cat ../codellama-13b/tokenizer.json | jq '.added_tokens'
[
  {
    "id": 0,
    "content": "<unk>",
    "single_word": false,
    "lstrip": false,
    "rstrip": false,
    "normalized": false,
    "special": true
  },
  {
    "id": 1,
    "content": "<s>",
    "single_word": false,
    "lstrip": false,
    "rstrip": false,
    "normalized": false,
    "special": true
  },
  {
    "id": 2,
    "content": "</s>",
    "single_word": false,
    "lstrip": false,
    "rstrip": false,
    "normalized": false,
    "special": true
  }
]
```
```console
(venv3) $ cat ../CodeLlama-7b-hf/tokenizer.json | jq '.model.vocab["▁<PRE>"]'
32007
(venv3) $ cat ../CodeLlama-7b-hf/tokenizer.json | jq '.model.vocab["▁<SUF>"]'
32008
(venv3) $ cat ../CodeLlama-7b-hf/tokenizer.json | jq '.model.vocab["▁<MID>"]'
32009
(venv3) $ cat ../CodeLlama-7b-hf/tokenizer.json | jq '.model.vocab["▁<EOT>"]'
32010
```

The `tokenizer.json` file is loaded by `vocab.py` which has a `SpecialVocab`
class which looks like this:
```python
class SpecialVocab:
    merges: list[str]
    add_special_token: dict[str, bool]
    special_token_ids: dict[str, int]
    chat_template: str | None

    def __init__(
        self, path: str | os.PathLike[str], load_merges: bool = False,
        special_token_types: tuple[str, ...] | None = None,
        n_vocab: int | None = None,
    ):
        self.special_token_ids = {}
        self.add_special_token = {}
        self.n_vocab = n_vocab
        self.load_merges = load_merges
        self.merges = []
        self.chat_template = None
        if special_token_types is not None:
            self.special_token_types = special_token_types
        else:
            self.special_token_types = ('bos', 'eos', 'unk', 'sep', 'pad', 'cls', 'mask',
                                        'prefix', 'suffix', 'middle', 'eot')
        self._load(Path(path))
```
Notice that `special_token_types` is a parameter of the constructor and is
defined as a tuple of strings (any number), and recall that a tuple is immutable
and also more performant than a list when iterating and accessing element. 

So we can either passing in a list of special token types when creating a new
instance of `SpecialVocab` or we can use the default values which shown above.
Next `_load` will be called:
```python
    def _load(self, path: Path) -> None:
        self._try_load_from_tokenizer_json(path)
        self._try_load_from_config_json(path)
        if self.load_merges and not self.merges:
            self._try_load_merges_txt(path)
```
```
    def _try_load_from_tokenizer_json(self, path: Path) -> bool:
        tokenizer_file = path / 'tokenizer.json'
        if tokenizer_file.is_file():
            with open(tokenizer_file, encoding = 'utf-8') as f:
                tokenizer = json.load(f)
            if self.load_merges:
                merges = tokenizer.get('model', {}).get('merges')
                if isinstance(merges, list) and merges and isinstance(merges[0], str):
                    self.merges = merges
            added_tokens = tokenizer.get('added_tokens', {})
        else:
            added_tokens = {}
```
Notice that `added_tokens` will now contain the infill special tokens in our
case. But note that this is not the case for CodeLlama. 

Now, that is the loading/construction of the `SpecialVocab` class. When this
class's `add_to_gguf` method is called 
```python
    def add_to_gguf(self, gw: GGUFWriter, quiet: bool = False) -> None:
        ...
        for typ, tokid in self.special_token_ids.items():
            id_handler: Callable[[int], None] | None = getattr(gw, f'add_{typ}_token_id', None)
            if id_handler is None:
                print(
                    f'gguf: WARNING: No handler for special token type {typ} with id {tokid} - skipping',
                    file = sys.stderr,
                )
                continue
            if not quiet:
                print(f'gguf: Setting special token type {typ} to {tokid}')
            id_handler(tokid)

        for typ, value in self.add_special_token.items():
            add_handler: Callable[[bool], None] | None = getattr(gw, f'add_add_{typ}_token', None)
            if add_handler is None:
                print(
                    f'gguf: WARNING: No handler for add_{typ}_token with value {value} - skipping',
                    file = sys.stderr,
                )
                continue
            if not quiet:
                print(f'gguf: Setting add_{typ}_token to {value}')
            add_handler(value)
```
So the first loop will iterate over the special token ids and call the functions
in `gguf_writer.py` to set the token ids like this one for the `bos` token:
```python
    def add_bos_token_id(self, id: int) -> None:
        self.add_uint32(Keys.Tokenizer.BOS_ID, id)
```
So adding method like the above for the infill tokens should be enough to have
them added to the GGUF file provided we also specify the tokens that we want
in the constructor of SpecialVocab.

Now, this model also has a `tokenizer_config.json` file and it also has entries
for the infill tokens:
```console
(venv3) $ cat ../codegemma-7b-it/tokenizer_config.json | jq '.added_tokens_decoder."67"'
{
  "content": "<|fim_prefix|>",
  "lstrip": false,
  "normalized": false,
  "rstrip": false,
  "single_word": false,
  "special": false
}
```
TODO: sort out the usage of usage `tokenizer_config.json`. It only looks like
the `chat_template` element is used from this file so I'm ignoring it for this
specific task.

Lets see if we can convert the model to a GGUF model (see instructions in
[Testing/Verification](#testingverification)) before running this command the
first time:
```console
./convert-hf-to-gguf.py --outtype f16 --outfile models/codegemma-7b-it-f16.gguf ~/work/ai/codegemma-7b-it
...
Model successfully exported to 'models/codegemma-7b-it-f16.gguf'
```
Now, lets inspect the generated GGUF model:
```console
* Loading: models/codegemma-7b-it-f16.gguf
* File is LITTLE endian, script is running on a LITTLE endian host.

* Dumping 26 key/value pair(s)
      1: UINT32     |        1 | GGUF.version = 3
      2: UINT64     |        1 | GGUF.tensor_count = 254
      3: UINT64     |        1 | GGUF.kv_count = 23
      4: STRING     |        1 | general.architecture = 'gemma'
      5: STRING     |        1 | general.name = 'codegemma-7b-it'
      6: UINT32     |        1 | gemma.context_length = 8192
      7: UINT32     |        1 | gemma.embedding_length = 3072
      8: UINT32     |        1 | gemma.block_count = 28
      9: UINT32     |        1 | gemma.feed_forward_length = 24576
     10: UINT32     |        1 | gemma.attention.head_count = 16
     11: UINT32     |        1 | gemma.attention.head_count_kv = 16
     12: FLOAT32    |        1 | gemma.attention.layer_norm_rms_epsilon = 9.999999974752427e-07
     13: UINT32     |        1 | gemma.attention.key_length = 256
     14: UINT32     |        1 | gemma.attention.value_length = 256
     15: UINT32     |        1 | general.file_type = 1
     16: STRING     |        1 | tokenizer.ggml.model = 'llama'
     17: [STRING]   |   256000 | tokenizer.ggml.tokens
     18: [FLOAT32]  |   256000 | tokenizer.ggml.scores
     19: [INT32]    |   256000 | tokenizer.ggml.token_type
     20: UINT32     |        1 | tokenizer.ggml.bos_token_id = 2
     21: UINT32     |        1 | tokenizer.ggml.eos_token_id = 1
     22: UINT32     |        1 | tokenizer.ggml.unknown_token_id = 3
     23: UINT32     |        1 | tokenizer.ggml.padding_token_id = 0
     24: BOOL       |        1 | tokenizer.ggml.add_bos_token = True
     25: BOOL       |        1 | tokenizer.ggml.add_eos_token = False
     26: STRING     |        1 | tokenizer.chat_template = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ ra"
```
Hmm, that did not work out as expected.
Ah, we also need to make sure that these special tokens are added when the
model is converted. This can be done in `convert-hf-to-gguf.py`:
```python
@Model.register("GemmaForCausalLM")
class GemmaModel(Model):
    model_arch = gguf.MODEL_ARCH.GEMMA

    def set_vocab(self):
        self._set_vocab_sentencepiece()

        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=False,
                                          special_token_types = ['prefix', 'suffix', 'middle', 'eot'])
        special_vocab._set_special_token("prefix", 67)
        special_vocab._set_special_token("suffix", 68)
        special_vocab._set_special_token("middle", 69)
        special_vocab._set_special_token("eot", 70)
        special_vocab.add_to_gguf(self.gguf_writer)
```
This may not be the correct/best way to do this but I'm trying to get this
working and will then go back and clean this up. With that change and re-running
the the conversion the generated model has the special token key-value fields.

Re-running the conversion and inspecting the generated GGUF model:
```console
(venv3) $ gguf-py/scripts/gguf-dump.py models/codegemma-7b-f16.gguf
* Loading: models/codegemma-7b-f16.gguf
* File is LITTLE endian, script is running on a LITTLE endian host.

* Dumping 29 key/value pair(s)
      1: UINT32     |        1 | GGUF.version = 3
      2: UINT64     |        1 | GGUF.tensor_count = 254
      3: UINT64     |        1 | GGUF.kv_count = 26
      4: STRING     |        1 | general.architecture = 'gemma'
      5: STRING     |        1 | general.name = 'codegemma-7b'
      6: UINT32     |        1 | gemma.context_length = 8192
      7: UINT32     |        1 | gemma.embedding_length = 3072
      8: UINT32     |        1 | gemma.block_count = 28
      9: UINT32     |        1 | gemma.feed_forward_length = 24576
     10: UINT32     |        1 | gemma.attention.head_count = 16
     11: UINT32     |        1 | gemma.attention.head_count_kv = 16
     12: FLOAT32    |        1 | gemma.attention.layer_norm_rms_epsilon = 9.999999974752427e-07
     13: UINT32     |        1 | gemma.attention.key_length = 256
     14: UINT32     |        1 | gemma.attention.value_length = 256
     15: UINT32     |        1 | general.file_type = 1
     16: STRING     |        1 | tokenizer.ggml.model = 'llama'
     17: [STRING]   |   256000 | tokenizer.ggml.tokens
     18: [FLOAT32]  |   256000 | tokenizer.ggml.scores
     19: [INT32]    |   256000 | tokenizer.ggml.token_type
     20: UINT32     |        1 | tokenizer.ggml.bos_token_id = 2
     21: UINT32     |        1 | tokenizer.ggml.eos_token_id = 1
     22: UINT32     |        1 | tokenizer.ggml.unknown_token_id = 3
     23: UINT32     |        1 | tokenizer.ggml.padding_token_id = 0
     24: BOOL       |        1 | tokenizer.ggml.add_bos_token = True
     25: BOOL       |        1 | tokenizer.ggml.add_eos_token = False
     26: UINT32     |        1 | tokenizer.ggml.prefix_token_id = 67
     27: UINT32     |        1 | tokenizer.ggml.suffix_token_id = 68
     28: UINT32     |        1 | tokenizer.ggml.middle_token_id = 69
     29: UINT32     |        1 | tokenizer.ggml.eot_token_id = 70
```
And we can see that the FIM/infill tokens ids are in the GGUF model.

Now we can run the `infill` program and see if this works using the generated
model:
```console
$ ./infill -t 10 -ngl 0 -m models/codegemma-7b-f16.gguf -c 4096 --temp 0.7 --repeat_penalty 1.1 -n 20 --in-prefix "def helloworld():\n    print(\"hell" --in-suffix "\n   print(\"goodbye world\")\n    "

#####  Infill mode  #####

<|fim_prefix|> def helloworld():\n    print("hell<|fim_middle|> \n   print("goodbye world")\n    <|fim_suffix|>
Traceback (most recent call last):

  File "<ipython-input>", line 2<|file_separator|>
```
This does not look very good at all and I need to look into this. The good thing
to note is that the correct special tokens specific to CodeGemma are being used.

Lets also try converting CodeLlama to a GGUF model and see if that works as
well after adding the special tokens to the `convert-hf-to-gguf.py` script:
```console
$ ./convert-hf-to-gguf.py --outtype f16 --outfile models/codellama-7b-hf-f16.gguf ~/work/ai/CodeLlama-7b-hf
```
And we can inspect the generated GGUF model:
```console
(venv3) $ gguf-py/scripts/gguf-dump.py models/codellama-7b-hf-f16.gguf 
* Loading: models/codellama-7b-hf-f16.gguf
* File is LITTLE endian, script is running on a LITTLE endian host.

* Dumping 29 key/value pair(s)
      1: UINT32     |        1 | GGUF.version = 3
      2: UINT64     |        1 | GGUF.tensor_count = 291
      3: UINT64     |        1 | GGUF.kv_count = 26
      4: STRING     |        1 | general.architecture = 'llama'
      5: STRING     |        1 | general.name = 'CodeLlama-7b-hf'
      6: UINT32     |        1 | llama.block_count = 32
      7: UINT32     |        1 | llama.context_length = 16384
      8: UINT32     |        1 | llama.embedding_length = 4096
      9: UINT32     |        1 | llama.feed_forward_length = 11008
     10: UINT32     |        1 | llama.attention.head_count = 32
     11: UINT32     |        1 | llama.attention.head_count_kv = 32
     12: FLOAT32    |        1 | llama.rope.freq_base = 1000000.0
     13: FLOAT32    |        1 | llama.attention.layer_norm_rms_epsilon = 9.999999747378752e-06
     14: UINT32     |        1 | general.file_type = 1
     15: UINT32     |        1 | llama.vocab_size = 32016
     16: UINT32     |        1 | llama.rope.dimension_count = 128
     17: STRING     |        1 | tokenizer.ggml.model = 'llama'
     18: [STRING]   |    32016 | tokenizer.ggml.tokens
     19: [FLOAT32]  |    32016 | tokenizer.ggml.scores
     20: [INT32]    |    32016 | tokenizer.ggml.token_type
     21: UINT32     |        1 | tokenizer.ggml.bos_token_id = 1
     22: UINT32     |        1 | tokenizer.ggml.eos_token_id = 2
     23: UINT32     |        1 | tokenizer.ggml.unknown_token_id = 0
     24: BOOL       |        1 | tokenizer.ggml.add_bos_token = True
     25: BOOL       |        1 | tokenizer.ggml.add_eos_token = False
     26: UINT32     |        1 | tokenizer.ggml.prefix_token_id = 32007
     27: UINT32     |        1 | tokenizer.ggml.suffix_token_id = 32008
     28: UINT32     |        1 | tokenizer.ggml.middle_token_id = 32009
     29: UINT32     |        1 | tokenizer.ggml.eot_token_id = 32010
```
And then run it:
```console
(venv3) $ ./infill -t 10 -ngl 0 -m models/codellama-7b-hf-f16.gguf -c 4096 --temp 0.7 --repeat_penalty 1.1 -n 20 --in-prefix "def helloworld():\n    print(\"hell" --in-suffix "\n   print(\"goodbye world\")\n    "

#####  Infill mode  #####

 <PRE> def helloworld():\n    print("hell <SUF> \n   print("goodbye world")\n     <MID>o world")\n  <EOT>
```

There are also changes required to llama.cpp. Currently, the special tokens
are specified with default values in llama_vocab:
```c++
struct llama_vocab {
    id special_prefix_id = 32007;
    id special_middle_id = 32009;
    id special_suffix_id = 32008;
    id special_eot_id    = 32010;
```
If we set these to `-1` for example the existing gguf models will not work since
they have not been converted with the updates in this task (they won't have the
key-values for the infill special tokens that is). Perhaps we could reach out
to the maintainers of those models and see if they would be willing to
re-convert and publish updates?

With those changes made, can we still use the existing models?  
Lets try CodeLlama:
```console
./infill -t 10 -ngl 0 -m models/codellama-13b.Q5_K_S.gguf -c 4096 --temp 0.7 --repeat_penalty 1.1 -n 20 --in-prefix "def helloworld():\n    print(\"hell" --in-suffix "\n   print(\"goodbye world\")\n    "
...
#####  Infill mode  #####

 <PRE> def helloworld():\n    print("hell <SUF> \n   print("goodbye world")\n     <MID>o world")',
        'def goodbyeworld(): <EOT>
```
And CodeGemma:
```console
./infill -t 10 -ngl 0 -m ~/Downloads/codegemma-7b-f16.gguf -c 4096 --temp 0.7 --repeat_penalty 1.1 -n 20 --in-prefix "def helloworld():\n    print(\"hell" --in-suffix "\n   print(\"goodbye world\")\n    "
...

#####  Infill mode  #####

<|fim_prefix|> def helloworld():\n    print("hell<|fim_suffix|> \n   print("goodbye world")\n    <|fim_middle|>o World!")<|file_separator|>
```

#### Questions
* Should the special tokens values really be extracted from the model files
instead of being hardcoded in the `convert-hf-to-gguf.py` script like this?

### Testing/Verification
To be able to test this while developing we will need to have a model that
supports infill, like CodeLlama or CodeGemma. Lets use CodeGemma and the first
step is to checkout the model from huggingface. This will be a non-gguf model
since I'm assuming that the metadata is added to the gguf model during the
conversion.

You'll need to use a HuggingFace [access token](https://huggingface.co/settings/tokens)
instead of a password in the following command with your HuggingFace username:
```console
$ pushd ~/work/ai
$ git clone https://huggingface.co/google/codegemma-7b
```

And we should install `gguf.py` in editable mode:
```console
$ cd ~/work/ai/llama.cpp/gguf-py
$ source venv/bin/activate
$ pip install -e .
```
With that setup we should be able to run `convert-hf-to-gguf.py`:
```console
./convert-hf-to-gguf.py --outtype f16 --outfile models/codegemma-7b-f16.gguf ~/work/ai/codegemma-7b
```

We also need to try out `CodeLlama` to make sure that it works as well.
```console
$ pushd ~/work/ai
$ git clone https://huggingface.co/codellama/CodeLlama-7b-hf/
```

```console
$ ./convert-hf-to-gguf.py --outtype f16 --outfile models/codellama-7b-hf-f16.gguf ~/work/ai/CodeLlama-7b-hf
```

The work resulted in the following PRs:
https://github.com/ggerganov/llama.cpp/pull/6689

