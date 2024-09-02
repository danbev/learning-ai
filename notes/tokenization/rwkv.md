## Receptance-Weighted Key-Value tokenizer

### llama.cpp RWKV tokenizer
To try this out we need a RWKV model in GGUF format which can be generated
with the following commands:
```console
$ cd fundamentals/llama.cpp
$ make checkout-rwkv-model
$ make convert-rwkv-model
```

The tokenize example can be run using:
```console
$ make run-rwkv-tokenize
```
And we can debug on Linux using:
```console
$ gdb --args ./tokenize models/v6-Finch-1B6-HF.gguf
(gdb) br llama-vocab.cpp:1511 if raw_text.compare("ÅWhat is LoRA?") == 0
```
```c++
        case LLAMA_VOCAB_TYPE_RWKV:
            {
                for (const auto & fragment : fragment_buffer) {
                    if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                        auto raw_text = fragment.raw_text.substr(fragment.offset, fragment.length);

                        llm_tokenizer_rwkv tokenizer(vocab);
                        tokenizer.tokenize(raw_text, output);
                    }
                }
            }
```
The constructor of `llm_tokenizer_rwkv` takes a `llm_vocab` and will build a
Trie using the tokens in the vocabulary.
The vocabulary size is:
```console
(gdb) p vocab.id_to_token.size()
$3 = 65536
```
```c++
    llm_tokenizer_rwkv(const llama_vocab & vocab): vocab(vocab) {
        // RWKV supports arbitrary byte tokens, but the vocab struct only supports string tokens.
        // For now, we decode the vocab here into the lookup we'll use for tokenization.

        // build trie
        for (unsigned int id = 0; id < vocab.id_to_token.size(); ++id) {
            const auto & token = vocab.id_to_token[id];
            const auto data = llama_unescape_rwkv_token(token.text);
            token_matcher.insert((const char *) data.data(), data.size(), id);
        }
    }
```
This unscapes the tokens in the vocabulary and inserts them into the trie.
After this we have tokenize which is passed in an empty
`std::vector<llama_vocab::id>`.

The tokenize function will try to match 
```c++
    void tokenize(const std::string & text, std::vector<llama_vocab::id> & output) {
        uint32_t position = 0;

        while (position < text.size()) {
            const struct naive_trie * node = token_matcher.traverse(text[position]);
            if (node == NULL) {
                // no matching token found, add unknown token
                output.push_back(vocab.special_unk_id);
                position += 1;
                continue;
            }

            // traverse the trie to find the longest matching token
            uint32_t token_id = 0;
            uint32_t token_length = 0;
            while (node != NULL) {
                if (node->has_value) {
                    token_id = node->value;
                    token_length = position + 1;
                }
                node = node->traverse(text[++position]);
            }

            // add the longest matching token
            output.push_back(token_id);
            position = token_length;
        }
```
Recall that our input text is:
```console
(gdb) p text
$14 = "ÅWhat is LoRA?"
```
And the the first character will be searched for in the Trie:
```console
(gdb) p/x text[position]
$13 = (const __gnu_cxx::__alloc_traits<std::allocator<char>, char>::value_type &) @0x7fffffffd240: 0xc3
```
This will be found in the Trie and hence is in the vocabulary:
```console
(gdb) p node->value
$16 = 196

(gdb) p vocab.id_to_token[196]
$15 = {text = "\\xc3", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
```
And notice that it will continue searching the children of the current node
to see if a longer prefix can be matched.
In this case there is a longer token:
```console
(gdb) p vocab.id_to_token[2467]
$18 = {text = "\\xc3\\x85", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
```
This is the token `Å` which is the longest prefix that can be matched. So this
token id will be added to the output vector.
This will continue and `What` is a also a token in the vocabulary and will be
added to the output vector:
```console
(gdb) p vocab.id_to_token[24326]
$25 = {text = "What", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
```
The final output vector will look like this:
```console
(gdb) finish
b) p output
$27 = std::vector of length 6, capacity 8 = {2467, 24326, 4600, 3991, 1393, 64}
```
So this was a very simple tokenization process I we compare it to the others.
