## Unigram Tokenization
Is also a subword tokenizer like SPM and BPE. The name unigram comes from that
this algorithm treats each character in isolation, it does not group pairs of
words or characters together.

What models use unigram?  
T5 uses unigram tokenization. So lets download a T5 model and how this works
in llama.cpp (flan-t5-small.Q4_K_M.gguf).

### llama.cpp UGM (Unigram Tokenization)
Lite SPM and BPE will will end up in the function `llama_tokenize_internal` and
this time in the case for `LLAMA_VOCAB_TYPE_UGM`.
```c++
        case LLAMA_VOCAB_TYPE_UGM:
            {
                llm_tokenizer_ugm tokenizer(vocab);

                if (add_special && vocab.tokenizer_add_bos != 0) {
                    GGML_ASSERT(vocab.special_bos_id != -1);
                    output.push_back(vocab.special_bos_id);
                }

                for (const auto & fragment : fragment_buffer) {
                    if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                        auto raw_text = fragment.raw_text.substr(fragment.offset, fragment.length);
                        tokenizer.tokenize(raw_text, output);
                    } else { // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
                        output.push_back(fragment.token);
                    }
                }

                if (add_special && vocab.tokenizer_add_bos != 0 && output.size() >= 2 && output[1] == vocab.special_bos_id) {
                    LLAMA_LOG_WARN(
                        "%s: Added a BOS token to the prompt as specified by the model but the prompt "
                        "also starts with a BOS token. So now the final prompt starts with 2 BOS tokens. "
                        "Are you sure this is what you want?\n", __FUNCTION__);
                }

                if (add_special && vocab.tokenizer_add_eos == 1) {
                    GGML_ASSERT(vocab.special_eos_id != -1);
                    output.push_back(vocab.special_eos_id);
                }
            } break;
```

```c++
    llm_tokenizer_ugm(const llama_vocab & vocab) : vocab(vocab) {
        if (vocab.precompiled_charsmap.size() > 0) {
            size_t charsmap_offset = 0;

            // First four bytes of precompiled_charsmap contains length of binary
            // blob containing XOR-compressed compact double array (XCDA) entries
            uint32_t xcda_blob_size = *(const uint32_t *) &vocab.precompiled_charsmap[0];
            charsmap_offset += sizeof(xcda_blob_size);
            if (xcda_blob_size + charsmap_offset >= vocab.precompiled_charsmap.size()) {
                throw std::runtime_error("Index out of array bounds in precompiled charsmap!");
            }
```
Now, `precompiled_charsmap` is a data structure that maps characters to
potential token beginnings in the model's vocabulary.
This can work by creating a mapping from each unique starting character to a
list of tokens that begin with it. When processing input text, use the current
character to lookup potential tokens in the charsmap which narrow downs the
search. For example:
```
Vocabulary: ["un", "uni", "unique", "que", "quick"]

Charsmap:
'u': [0, 1, 2]  # Indices for "un", "uni", "unique"
'q': [3, 4]     # Indices for "que", "quick"
```
Looking up or searching for a string sequence was also something that the Trie
data structure is good at. And a way to implement a Trie is using the Double
Array Trie, and also the compressed Double Array Trie. So this is what the I
think is used in unigram.

So like the comment says we first extract size of the compressed double array
and also add the size to `charsmap_offset`:
```console
(gdb) p sizeof(xcda_blob_size)
$18 = 4
(gdb) p charsmap_offset
$19 = 4
```
At this point charsmap_offset is 4 this will be used to index into the
precompiled_charsmap to get the compressed double array:
```
            xcda_array = (const uint32_t *) &vocab.precompiled_charsmap[charsmap_offset];
```
So this is a pointer to the compressed double array. I've written about this
type of [Trie](../trie.md) before. So each entry in that array is a 32-bit
unsigned integer. And the number of entries in the array will be:
```console
gdb) p xcda_blob_size / sizeof(uint32_t)
$21 = 44288
```
That will be stored in `xcda_array_size`. Following that `charsmap_offset` will
be incremented by the size of the compressed double array:
```c++
            charsmap_offset += xcda_blob_size;
```
I'm not sure what these prefix replacements are about yet but hopefully this
will become clear when we look at the tokenize function later:
```c++
            // Remaining bytes of precompiled charsmap contain null-terminated
            // replacement strings for prefixes matched by the XCDA.
            prefix_replacements = &vocab.precompiled_charsmap[charsmap_offset];
            prefix_replacements_size = vocab.precompiled_charsmap.size() - charsmap_offset;
```
Next we iterate over all the token ids (32128):
```c++
        for (unsigned int id = 0; id < vocab.id_to_token.size(); ++id) {
            const auto &token_data = vocab.id_to_token[id];

            if (llama_is_normal_token(vocab, id)) {
                min_score = std::min<float>(min_score, token_data.score);
                max_score = std::max<float>(max_score, token_data.score);
            }

            if (llama_is_normal_token(vocab, id) ||
                llama_is_user_defined_token(vocab, id) ||
                llama_is_unused_token(vocab, id)) {
                token_matcher.insert(token_data.text.data(), token_data.text.size(), id);
            }

            if (llama_is_user_defined_token(vocab, id)) {
                user_defined_token_matcher.insert(token_data.text.data(), token_data.text.size());
            }
        }

        unknown_token_score = min_score - unknown_token_score_penalty;
    }
```
```console
(gdb) p token_data
$26 = (const llama_vocab::token_data &) @0x7ffff76c6010: {text = "<pad>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
```
Then we update the min and max score for the current token if it was less than
/greater than the current min/max (which are member of the struct).
Then is the token is a normal/user defined or unused token we insert the token
into the `token_matcher` which is of type `naive_trie`.

_wip_
