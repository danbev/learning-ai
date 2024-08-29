## Byte-Pair Encoding (BPE)
This is a subword tokenization algorithm.

### Training Process
During the training process we take a corpus of text and split it into
individual characters. So this will give us a text file of characters.

1) We then iterate over all of these characters and count the frequency adjacent
pairs.

2) The most frequent pair is merged into a single token. This merge operation is
recorded as a rule in a file, typically named `merges.txt`, where each line
represents a rule for transforming a pair of tokens into a single token.

This token is then added to the vocabulary and the pairs in the text are
replaced by this new token.
This process is repeated until we reach the desired vocabulary size which is a
fixed size variable.

Example:
Training corpus: "Hello World!"

The initial vocabulary would be:
```
H, e, l, o, W, r, d, !
```
Where each character is a separate token. Notice that this is a set.

So BPE starts with a vocabulary of individual characters and iteratively merges
the most frequent pairs of adjacent tokens. During training, BPE creates a list
of merge rules, each representing a pair of tokens to be merged. The "rank" in
BPE refers to the order or priority of these merge rules. The most frequent
pairs get the lowest ranks (highest priority).

#### llama.cpp Byte Pair Encoding
This section will step through [tokenize.cpp](../fundamentals/llama.cpp/tokenize.cpp)
which be built and started using the following commands:
```console
$ cd ../fundamentals/llama.cpp/
$ make tokenize
$ gdb --args ./tokenize
(gdb) br tokenize.cpp:58
Breakpoint 1 at 0x168cf: file src/tokenize.cpp, line 58.
```
The model we choose will need to use BPE as the tokenizer and for this debugging
session I'm using `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`.

Now, in `llama_tokenize_internal` we have a switch statement on the
`vocab.type` which in this case is:
```console
(gdb) p vocab.type
$57 = LLAMA_VOCAB_TYPE_BPE
```

```c++
        case LLAMA_VOCAB_TYPE_BPE:
            {
                llm_tokenizer_bpe tokenizer(vocab);

                if (add_special) {
                    tokenizer.append_bos(output);
                }
```
So we can see that the first thing that happens is that a new `llm_tokenizer_bpe`
is created with the `vocab` as an argument.

`llm_tokenizer_bpe` has the following member fields:
```
    const llama_vocab & vocab;

    std::vector<std::string> regex_exprs;

    std::vector<llm_symbol> symbols;
    std::vector<llm_symbol> symbols_final;

    llm_bigram_bpe::queue work_queue;
```
Is also has a constructor which sets the `regex_exprs` field differently
depending on the `vocab.type_pre`:
```console
(gdb) p vocab.type_pre
$2 = LLAMA_VOCAB_PRE_TYPE_LLAMA3
```
Some models have specific preprocessing steps that they have performed on their
tokenization training data and which should also be applied to the input text
that is to be tokenized. This is the purpose of the `type_pre` field in the 
`llama_vocab` struct.
```c++
    llm_tokenizer_bpe(const llama_vocab & vocab): vocab(vocab) {
        GGML_ASSERT(vocab.type == LLAMA_VOCAB_TYPE_BPE);
        switch (vocab.type_pre) {
            case LLAMA_VOCAB_PRE_TYPE_LLAMA3:
                regex_exprs = {
                    // original regex from tokenizer.json
                    //"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",

                    // adapted: https://github.com/ggerganov/llama.cpp/pull/6920#issuecomment-2080233989
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
```
This is what the regex looks like in tokenizer.json:
```
"normalizer": null,
  "pre_tokenizer": {
    "type": "Sequence",
    "pretokenizers": [
      {
        "type": "Split",
        "pattern": {
          "Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
        },
        "behavior": "Isolated",
        "invert": false
      },
      {
        "type": "ByteLevel",
        "add_prefix_space": false,
        "trim_offsets": true,
        "use_regex": false
      }
    ]
```
Now, when we looked at SPM we did not have any of this preprocessing. This is
because SPM treats the input as a sequence of bytes and does not have any
special handling for characters or numbers. It does not make any assumptions
about word boundries or special characters and hence does not need any
pre-processing. But for BPE there is this handling and some of the models have
specific preprocessing steps that they have performed on their tokenization.
For example, take the string "New York" which might be tokenized as:
```
["New", " ", "York"]
```
But what we want is a single token for "New York" and this is what the
Another reason is that BPE is case sensitive which can increase the vocabulary
size. So we might want to lowercase all the text before tokenization.
Also punctuation can cause problems in BPE if not handled correctly. As an
example if we have 'dog', 'dog!', 'dog?' all be single separate tokens this
would be suboptimal. So we might want to remove punctuation before tokenization.
So this regex acts like merging rules for the input text before tokenization.

```
"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",

p{L} = any kind of letter from any language
p{N} = any kind of numeric character in any script
\\s  = any whitespace character
\\S  = any non-whitespace character
```
To play around with this regex there python script
[llama3-pre-regex.py](../../fundamentals/tokenization/src/llama3-pre-regex.py).

Next, if `add_special` is true 
```c++
                if (add_special) {
                    tokenizer.append_bos(output);
                }

                for (const auto & fragment : fragment_buffer) {
                    if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                        auto raw_text = fragment.raw_text.substr(fragment.offset, fragment.length);

#ifdef PRETOKENIZERDEBUG
                        LLAMA_LOG_WARN("TT: (%ld %ld %ld) '%s'\n", raw_text.length(), fragment.offset, fragment.length, raw_text.c_str());
#endif
                        tokenizer.tokenize(raw_text, output);
                    } else { // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
                        tokenizer.append(fragment.token, output);
                    }
            }
```
Similar to the SPM we will iterate of the `fragment_buffer` and call tokenize.
```c++
    void tokenize(const std::string & text, std::vector<llama_vocab::id> & output) {
        int final_prev_index = -1;

        const auto word_collection = unicode_regex_split(text, regex_exprs);
```
Notice that this is where the `regex_exprs` is used. This will return a vector
of strings where each entry is a token according to the rules in the regex.
We are then iterating through this vector:
```c++
        for (auto & word : word_collection) {
            work_queue = llm_bigram_bpe::queue();
            symbols.clear();

            int index = 0;
            size_t offset = 0;

            if (vocab.tokenizer_ignore_merges && vocab.token_to_id.find(word) != vocab.token_to_id.end()) {
                symbols.emplace_back(llm_symbol{-1, -1, word.c_str(), word.size()});
                offset = word.size();
            }
```
Notice how the above is checking `tokenizer_ignore_merges` is set and if the
word is in the vocabulary. If so we add the the symbol for it to theh symbols
vector.

After that we are going to create an `llm_symbol` for this word and add it
to the symbols vector:
```c++
            while (offset < word.size()) {
                llm_symbol sym;
                size_t char_len = std::min(word.size() - offset, (size_t) unicode_len_utf8(word[offset]));
                sym.text = word.c_str() + offset;
                sym.n = char_len;
                offset += sym.n;
                sym.prev = index - 1;
                sym.next = offset == word.size() ? -1 : index + 1;
                index++;
                symbols.emplace_back(sym);
            }
```
We then iterate over all the symbols added above and add bigrams for each pair
of symbols:
```c++
            for (size_t i = 1; i < symbols.size(); ++i) {
                add_new_bigram(i - 1, i);
            }
```

So, we called this only the integers -1 and 1 (the first time),
so left will be -1 and right 1.
``c++
    void add_new_bigram(int left, int right) {
        std::string left_token  = std::string(symbols[left].text,  symbols[left].n);
        std::string right_token = std::string(symbols[right].text, symbols[right].n);

        int rank_found = -1;

        rank_found = vocab.find_bpe_rank(left_token, right_token);

        if (rank_found < 0) {
            return;
        }

        llm_bigram_bpe bigram;

        bigram.left  = left;
        bigram.right = right;
        bigram.text  = left_token + right_token;
        bigram.size  = left_token.size() + right_token.size();
        bigram.rank  = rank_found;

        work_queue.push(bigram);
    }

```
Next it will extract the strings for each of these utf8 characters.

Next we are checking if this pair of tokens (bigram) exists in our BPE
vocabulary.
The rank represents the priority or frequency of this bigram in the BPE merges.
```c
int llama_vocab::find_bpe_rank(const std::string & token_left, const std::string & token_right) const {
    GGML_ASSERT(token_left.find(' ')   == std::string::npos);
    GGML_ASSERT(token_left.find('\n')  == std::string::npos);
    GGML_ASSERT(token_right.find(' ')  == std::string::npos);
    GGML_ASSERT(token_right.find('\n') == std::string::npos);

    auto it = bpe_ranks.find(std::make_pair(token_left, token_right));
    if (it == bpe_ranks.end()) {
        return -1;
    }

    return it->second;
}
```
BPE starts with a vocabulary of individual characters and iteratively merges
the most frequent pairs of adjacent tokens.
During training, BPE creates a list of merge rules, each representing a pair
of tokens to be merged.
The "rank" in BPE refers to the order or priority of these merge rules.
The most frequent pairs get the lowest ranks (highest priority).

This function is crucial in determining whether a given pair of tokens should
be merged and, if so, with what priority.  It directly implements the concept
of BPE ranks, allowing the tokenizer to apply merge rules in the correct order.

`bpe_ranks` is a map of pairs of string as the key and an rank as the value:
 ```
struct llama_vocab {
    ...
    std::map<std::pair<std::string, std::string>, int> bpe_ranks;
    ...
}
```
This is populated in llama.cpp by by `llm_load_vocab`:
```c++  
        } else if (tokenizer_model == "gpt2") {
            vocab.type = LLAMA_VOCAB_TYPE_BPE;

            // read bpe merges and populate bpe ranks
            const int merges_keyidx = gguf_find_key(ctx, kv(LLM_KV_TOKENIZER_MERGES).c_str());
            if (merges_keyidx == -1) {
                throw std::runtime_error("cannot find tokenizer merges in model file\n");
            }

            const int n_merges = gguf_get_arr_n(ctx, merges_keyidx);
            for (int i = 0; i < n_merges; i++) {
                const std::string word = gguf_get_arr_str(ctx, merges_keyidx, i);
                GGML_ASSERT(unicode_cpts_from_utf8(word).size() > 0);

                std::string first;
                std::string second;

                const size_t pos = word.find(' ', 1);

                if (pos != std::string::npos) {
                    first  = word.substr(0, pos);
                    second = word.substr(pos + 1);
                }

                vocab.bpe_ranks.emplace(std::make_pair(first, second), i);
            }
```
Back in `add_new_bigram` 
```c++
        llm_bigram_bpe bigram;

        bigram.left  = left;
        bigram.right = right;
        bigram.text  = left_token + right_token;
        bigram.size  = left_token.size() + right_token.size();
        bigram.rank  = rank_found;

        work_queue.push(bigram);
```
Recall that `left` and `right` are indexes no strings.
```c++
    llm_bigram_bpe::queue work_queue;

    using queue_storage = std::vector<llm_bigram_bpe>;
    using queue = std::priority_queue<llm_bigram_bpe, queue_storage, comparator>;

    llm_bigram_bpe::queue work_queue;
```
So the priority queue will contain pairs (bigrams) with the highest rank
(lowest value) first.
Back in tokenize we then have:
```c++
            // build token(s)
            while (!work_queue.empty()) {
                auto bigram = work_queue.top();
                work_queue.pop();
                ...
            }
```
Recall that `top()` will return a `const_reference` to the top most element, but
will not remove it (the function is const):
```c++
const_reference top() const;
void pop();
```
And note that `pop()` is void. 

Notice that the following will create a new llm_bigram_bpe by using the copy
constructor of llm_bigram_bpe:
```c++
                auto bigram = work_queue.top();
```
So a new `llm_bigram_bpe` struct will be created on the stack and the all the
members copied. Now for `left` and `right` these are just integers so they are
trivial to copy. The same for `rank` and `size`. But for `text` a new string is
allocated and all characters copied.

```
struct llm_bigram_bpe {
    struct comparator {
        bool operator()(const llm_bigram_bpe & l, const llm_bigram_bpe & r) const {
            return l.rank > r.rank || (l.rank == r.rank && l.left > r.left);
        }
    };

    using queue_storage = std::vector<llm_bigram_bpe>;
    using queue = std::priority_queue<llm_bigram_bpe, queue_storage, comparator>;
    llm_symbol::index left;
    llm_symbol::index right;
    std::string text;
    int rank;
    size_t size;
};
```
We can avoid this memory copy of the string using:
```c++
    auto bigram = std::move(const_cast<llm_bigram_bpe&>(work_queue.top()));
```

Then we go through the `work_queue` until it is empty:
```c++
            while (!work_queue.empty()) {
                auto bigram = work_queue.pop_move();

                auto & left_symbol = symbols[bigram.left];
                auto & right_symbol = symbols[bigram.right];

                if (left_symbol.n == 0 || right_symbol.n == 0) {
                    continue;
                }
                std::string left_token = std::string(left_symbol.text, left_symbol.n);
                std::string right_token = std::string(right_symbol.text, right_symbol.n);
                if (left_token + right_token != bigram.text) {
                    continue;  // Skip this bigram if it's outdated
                }

                // merge the right sym into the left one
                left_symbol.n += right_symbol.n;
                right_symbol.n = 0;

                // remove the right sym from the chain
                left_symbol.next = right_symbol.next;
                if (right_symbol.next >= 0) {
                    symbols[right_symbol.next].prev = bigram.left;
                }

                add_new_bigram(left_symbol.prev, bigram.left);  // left side of current symbol
                add_new_bigram(bigram.left, left_symbol.next);  // right side of current symbol
            }
```


We will merge the right symbol into the left one and then remove the
right symbol from the linked list:
```
[A] -> [B] -> [C] -> [D] -> [E]

[A] -> [B] -> [C] -> [D] -> [E]
       ^     ^
       |     |
    left   right

[A] -> [BC] -> [D] -> [E]
```

Now, after we have merged the left and right into the left symbol, that merge
might be able to generate new merges that were not there before this merge.
So we look to the left of the newly merged symbol and try to find any bigram
pair and if so add it to the work queue. And the same for the right which my now
be possible to make a bigram. This is the above two lines of code are doing.

```c++
add_new_bigram(left_symbol.prev, bigram.left);  // left side of current symbol
```
This adds a bigram for the symbol before BC and BC itself.
In our example, this would be A-BC.

```c++
add_new_bigram(bigram.left, left_symbol.next);  // right side of current symbol
```
This adds a bigram for BC and the symbol after it.
In our example, this would be BC-D.

[A] -> [BC] -> [D] -> [E]

And two new bigrams have been added to the work queue for future consideration:
```c++
A-BC
BC-D
```

Before:
[A] -> [B] -> [C] -> [D] -> [E]
       ^     ^
       |     |
    left   right

After:
[A] -> [BC] -> [D] -> [E]
  ^     ^  ^    ^
  |     |  |    |
  +-----+  +----+
  New bigrams added
```

After that (and still in the loop of the current word) we add the symbols
to the `symbols_final` vector which maintains the correct order of tokens:
```c++
            // add the finished tokens to the final list keeping correct order for next and prev
            for (auto & sym : symbols) {
                if (sym.n > 0) {
                    sym.prev = final_prev_index;
                    sym.next = -1;
                    if (final_prev_index != -1) {
                        symbols_final[final_prev_index].next = symbols_final.size();
                    }
                    symbols_final.emplace_back(sym);
                    final_prev_index = symbols_final.size() - 1;
                }
            }
```
```console
(gdb) br llama-vocab.cpp:567 if raw_text.compare("What is LoRA?") == 0
```
With all the symbols in `symbols_final` we set symbols to it and then iterate
over all the symbols:
```
        symbols = symbols_final;

        if (!symbols.empty()) {
            for (int i = 0; i != -1; i = symbols[i].next) {
                auto & symbol = symbols[i];
                if (symbol.n == 0) {
                    continue;
                }

                const std::string str = std::string(symbol.text, symbol.n);
                const auto token = vocab.token_to_id.find(str);

                if (token == vocab.token_to_id.end()) {
                    for (auto j = str.begin(); j != str.end(); ++j) {
                        std::string byte_str(1, *j);
                        auto token_multibyte = vocab.token_to_id.find(byte_str);
                        if (token_multibyte != vocab.token_to_id.end()) {
                            output.push_back(token_multibyte->second);
                        }
                    }
                } else {
                    output.push_back((*token).second);
                }
            }
        }
```
We get the token as a string and then look it up in the vocabulary. If the token
is in the vocabulary the id will be added to the output vector. If not we will
try to lookup each byte of the token in the vocabulary and add its id to the
output.

Back in `llama_tokenize_internal`:
```c++
                if (add_special) {
                    tokenizer.append_eos(output);
                    tokenizer.check_double_bos_eos(output);
                }
```
