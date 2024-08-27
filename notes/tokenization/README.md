## Tokenization
The tokenization process involves splitting the input text into tokens, which
are then looked up in the vocabulary to get the token IDs.

Models include a vocabulary file, which is a list of all the tokens in the
model. There might be a configuration file in addition to this that specifies
the type of tokenizing that the model uses, like Byte-Pair Encoding (BPE),
WordPiece, SentencePiece, or Unigram, etc.

#### Tokenization notes
The following notes are individual walkthroughs of the tokenization process for
different tokenization types in llama.cpp:

* [Byte Pair Encoding (BPE)](./bpe.md)
* [WordPiece](./wordpiece.md) TODO
* [SentencePiece](./sentencepiece.md)
* [Unigram](./unigram.md) TODO

### Tokenization in llama.cpp
Llama.cpp supports the following types of tokenization:
```c
    enum llama_vocab_type {
        LLAMA_VOCAB_TYPE_NONE = 0, // For models without vocab
        LLAMA_VOCAB_TYPE_SPM  = 1, // LLaMA tokenizer based on byte-level BPE with byte fallback
        LLAMA_VOCAB_TYPE_BPE  = 2, // GPT-2 tokenizer based on byte-level BPE
        LLAMA_VOCAB_TYPE_WPM  = 3, // BERT tokenizer based on WordPiece
        LLAMA_VOCAB_TYPE_UGM  = 4, // T5 tokenizer based on Unigram
    };
```

The tokenization in llama.cpp is exposed through the llama.h header and has
the following interface:
```c++

    LLAMA_API int32_t llama_tokenize(
        const struct llama_model * model,
                      const char * text,
                         int32_t   text_len,
                     llama_token * tokens,
                         int32_t   n_tokens_max,
                            bool   add_special,
                            bool   parse_special);

    LLAMA_API int32_t llama_detokenize(
        const struct llama_model * model,
               const llama_token * tokens,
                         int32_t   n_tokens,
                            char * text,
                         int32_t   text_len_max,
                            bool   remove_special,
                            bool   unparse_special);
```
When we call `llama_tokenize` this call will land in `llama.cpp`
```c++
int32_t llama_tokenize(
    const struct llama_model * model,
                  const char * text,
                     int32_t   text_len,
                 llama_token * tokens,
                     int32_t   n_tokens_max,
                        bool   add_special,
                        bool   parse_special) {
    return llama_tokenize_impl(model->vocab, text, text_len, tokens,
                               n_tokens_max, add_special, parse_special);
}
```
And `llama_tokenize_impl` is defined in `llama-vocab.cpp`:
```c++
std::vector<llama_vocab::id> llama_tokenize_internal(const llama_vocab & vocab,
    std::string raw_text, bool add_special, bool parse_special) {
    std::vector<llama_vocab::id> output;
    std::forward_list<fragment_buffer_variant> fragment_buffer;

    if (!raw_text.empty()) {
        fragment_buffer.emplace_front(raw_text, 0, raw_text.length());
        tokenizer_st_partition(vocab, fragment_buffer, parse_special);
    }

    switch (vocab.type) {
        ...
    }
```
For details about the `fragment_buffer` and `tokenizer_st_partition` see 
the sections below.


#### fragment_buffer_variant
In llama-vocab.cpp have the following enum and struct:
```c++
typedef enum FRAGMENT_BUFFER_VARIANT_TYPE {
    FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN,
    FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT
} FRAGMENT_BUFFER_VARIANT_TYPE;

struct fragment_buffer_variant {
    fragment_buffer_variant(llama_vocab::id _token)
    :
        type(FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN),
        token(_token),
        raw_text(_dummy),
        offset(0),
        length(0) {}

    fragment_buffer_variant(const std::string & _raw_text, int64_t _offset, int64_t _length)
    :
        type(FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT),
        token((llama_vocab::id) - 1),
        raw_text(_raw_text),
        offset(_offset),
        length(_length){
            GGML_ASSERT(_offset >= 0);
            GGML_ASSERT(_length >= 1);
            GGML_ASSERT(offset + length <= raw_text.length());
        }

    const FRAGMENT_BUFFER_VARIANT_TYPE type;
    const llama_vocab::id token;
    const std::string _dummy;
    const std::string & raw_text;
    const uint64_t offset;
    const uint64_t length;
};
```
This struct can either hold a string with a offset and length or a token id.
The `_dummy` token is only used for the token variant because the `raw_text`
member is const.

For example, is we call `llama_tokenize_internal` with the string
"What is LoRA?" we will have the following fragment_buffer:
```console
(gdb) p fragment_buffer
$7 = std::forward_list = {[0] = {
type = FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT,
token = -1,
_dummy = "",
raw_text = "What is LoRA?", 
offset = 0,
length = 13}}
```
The `FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN` is used to represent special tokens
like `<s>`, `<unk>`, `</s>`. See `tokenizer_st_partition` below for an example
of this.

#### tokenizer_st_partition
If the model has `cache_special_tokens` `tokenizer_st_partition` will iterate
over them.
```c++
static void tokenizer_st_partition(const llama_vocab & vocab,
    std::forward_list<fragment_buffer_variant> & buffer, bool parse_special) {
    // for each special token
    for (const llama_vocab::id special_id : vocab.cache_special_tokens) {
        const auto & data = vocab.id_to_token[special_id];
        const auto & special_token = data.text;
        ...
```
So this will iterate over all the special tokens of the model in question,
which is llama-2-7b.Q4_0.gguf.

For example this migth be:
```console
(gdb) p vocab.cache_special_tokens 
$9 = std::vector of length 3, capacity 4 = {0, 2, 1}
(gdb) p special_id
$10 = 0
(gdb) p vocab.id_to_token[0]
$11 = {text = "<unk>", score = 0, attr = LLAMA_TOKEN_ATTR_UNKNOWN}
(gdb) p vocab.special_unk_id 
$12 = 0
(gdb) p data.text
$14 = "<unk>"
```
So `special_token` will be "<unk>". Then we will iterate over all the fragments
which is is only one of in our case:
```c++
        // for each text fragment
        std::forward_list<fragment_buffer_variant>::iterator it = buffer.begin();
        while (it != buffer.end()) {
            auto & fragment = (*it);
```
```console
(gdb) p fragment
$22 = (fragment_buffer_variant &) @0x555555bff918: {type = FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT, token = -1, 
  _dummy = "", raw_text = "What is LoRA?", offset = 0, length = 13}
```
And we can see that the type matches the following is statement so this block
will be executed:
```c++
            // if a fragment is text ( not yet processed )
            if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                auto & raw_text = fragment.raw_text;

```
The we will search the `raw_text` for the "<unk>" token:
```c++
                    auto match = raw_text.find(special_token, raw_text_base_offset);
```
Lets update the input to to be instead to see what happens when there is
match:
```console
(gdb) p raw_text
$29 = "<s>What is LoRA?</s>"
```
```c++
                    // if match is further than base offset
                    //  then we have some text to the left of it
                    if (match > raw_text_base_offset) {
                        ...

                        if (left_reminder_length > 0) {
                            buffer.emplace_after(it, raw_text, left_reminder_offset, left_reminder_length);
                            it++;
                        }
```
So we have our `raw_string` and we are looking for the "</s>>" token in it.
```
0             16
↓              ↓
<s>What is LoRA?</s>
```
So the above `emplace_after` call will create a new `fragment_buffer_variant`
using the same `raw_text` but with an offset of 0 and a length of 16.
```console
(gdb) p buffer
$43 = std::forward_list = {
[0] = {type = FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT, token = -1, _dummy = "", 
    raw_text = "<s>What is LoRA?</s>",
    offset = 0, length = 20},
[1] = {type = FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT, token = -1, _dummy = "",
    raw_text = "<s>What is LoRA?</s>", 
    offset = 0, length = 16}}
```
After that another `fragment_buffer_variant` is created but this time it will
use the token constructor:
```c++
                    buffer.emplace_after(it, special_id);
```
And after this the buffer will look like this:
```console
(gdb) p buffer
$46 = std::forward_list = {
[0] = {type = FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT, token = -1, _dummy = "", 
    raw_text = "<s>What is LoRA?</s>", offset = 0, length = 20},
[1] = {type = FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT, token = -1, _dummy = "",
    raw_text = "<s>What is LoRA?</s>", offset = 0, length = 16},
[2] = {type = FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN, token = 2, _dummy = "", 
    raw_text = "", offset = 0, length = 0}}
```

```c++
                        if (source == 0) {
                            buffer.erase_after(buffer.before_begin());
                        } else {
                            buffer.erase_after(std::next(buffer.begin(), (source-1)));
                        }
```
`before_begin()` is a function provided by the `std::forward_list` and it
returns an iterator to the element before the first element in the list. And
`erase_after` will remove the element after the iterator passed to it. So this
removes the first element in the buffer and after this call the buffer will look
like this:
```console
(gdb) p buffer
$48 = std::forward_list = {
[0] = {type = FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT, token = -1, _dummy = "", 
    raw_text = "<s>What is LoRA?</s>", offset = 0, length = 16},
[1] = {type = FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN, token = 2, _dummy = "", raw_text = "", offset = 0, length = 0}}
```
Then the loop will continue with the next special character which in our
case will be `<s>` which is special token 1. This will be added directly after
the first element in the buffer:
```console
(gdb) p buffer
(gdb) p buffer
$52 = std::forward_list = {
[0] = {type = FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT, token = -1, _dummy = "", 
    raw_text = "<s>What is LoRA?</s>", offset = 0, length = 16},
[1] = {type = FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN, token = 1, _dummy = "", raw_text = "", offset = 0, length = 0}, 
[2] = {type = FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN, token = 2, _dummy = "", raw_text = "", offset = 0, length = 0}}
```
And we are doing to do something similar to what we did above but this time with
the text that is to the right of this special token:
```c++
                            buffer.emplace_after(it, raw_text, right_reminder_offset, right_reminder_length);
```
```console
(gdb) p right_reminder_offset
$53 = 3
(gdb) p right_reminder_length

gdb) p buffer
$55 = std::forward_list = {
[0] = {type = FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT, token = -1, _dummy = "", raw_text = "<s>What is LoRA?</s>", offset = 0, length = 16},
[1] = {type = FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN, token = 1, _dummy = "", raw_text = "", offset = 0, length = 0}, 
[2] = {type = FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT, token = -1, _dummy = "", raw_text = "<s>What is LoRA?</s>", offset = 3, length = 13},
[3] = {type = FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN, token = 2, _dummy = "", raw_text = "", offset = 0, length = 0}}
```
And again we need to remove the first element in the buffer to adjust it for
the special tokens that have been inserted. This is done the same way as above
by removing the first element. The buffer then looks like this:
```console
(gdb) p buffer
$56 = std::forward_list = {
[0] = {type = FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN, token = 1, _dummy = "", raw_text = "", offset = 0, length = 0},
[1] = {type = FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT, token = -1, _dummy = "", raw_text = "<s>What is LoRA?</s>", offset = 3, length = 13},
[2] = {type = FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN, token = 2, _dummy = "", raw_text = "", offset = 0, length = 0}}
```
