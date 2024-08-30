## Unigram Tokenization
Is also a subword tokenizer like SPM and BPE. The name unigram comes from that
this algorithm treats each character in isolation, it does not group pairs of
words or characters together.

What models use unigram?  
T5 uses unigram tokenization. So lets download a T5 model and how this works
in llama.cpp (flan-t5-small.Q4_K_M.gguf).

### llama.cpp UGM (Unigram Tokenization)
We can start a session with gdb and set a break point `llama_tokenize_internal`
and specify that it break only for a specific raw_text:
```console
(gdb) break llama_tokenize_internal if raw_text.compare("What is LoRA?") == 0
```
If we don't do this there is call to tokenize in `llm_load_tokenizer` which
will be hit before our break point which can cause some confusion at first.

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
Then is the token is a normal, or user-defined, or unused token we insert the
token into the `token_matcher` which is of type `naive_trie`.

Back in the `llama_tokenize_internal` function we can see that the
`llm_tokenizer_ugm` will later call the `tokenize` function:
```c++
                        tokenizer.tokenize(raw_text, output);
```

```c++
    void tokenize(const std::string & text, std::vector<llama_vocab::id> & output) {
        // get current size of output (for reversal later)
        size_t output_size = output.size();

        // normalize the input first
        std::string normalized;
        normalize(text, &normalized);
        size_t input_len = normalized.size();
```
So we will first normalize the input text.
```c++
    void normalize(const std::string& input, std::string * normalized) {
        normalized->clear();
        normalized->reserve(input.size() * 3);

        const std::string space = vocab.tokenizer_escape_whitespaces ? escaped_space : " ";
```
`escaped_space` is defined in the struct as:
```c++
    // escaped space symbol - U+2581 (Lower One Eighth Block)
    const std::string escaped_space = "\xE2\x96\x81";
```
In this case it is:
```console
gdb) p space
$32 = "▁"
```
Then name comes from that it is a block character that fills the lower 1/8th of
a character cell.

Following that we have:
```c++
        size_t input_len = input.size();

        for (size_t input_offset = 0; input_offset < input_len; ) {
            auto norm_res = normalize_prefix(input, input_offset);
```
Now, lets take a closer look at `normalize_prefix`:
```c++
    struct normalization_result normalize_prefix(const std::string & input, size_t input_offset) {
        if (input_offset == input.size()) {
            return { &input[input_offset], 0, 0 };
        }

        // if input prefix matches some user-defined token return this token as normalization result
        auto user_defined_token_match = user_defined_token_matcher.get_longest_prefix(
            &input[input_offset], input.size() - input_offset);
```
TODO: How are user-defined tokens defined?
To get an overview of the `llm_tokenizer_ugm` struct we can use the `ptype`:
```console
(gdb) ptype llm_tokenizer_ugm
type = struct llm_tokenizer_ugm {
  private:
    const llama_vocab &vocab;
    const std::string escaped_space;
    const char *prefix_replacements;
    size_t prefix_replacements_size;
    const uint32_t *xcda_array;
    size_t xcda_array_size;
    naive_trie user_defined_token_matcher;
    float min_score;
    float max_score;
    float unknown_token_score_penalty;
    float unknown_token_score;
    naive_trie token_matcher;

  public:
    llm_tokenizer_ugm(const llama_vocab &);
    void tokenize(const std::string &, std::vector<int> &);
  private:
    void normalize(const std::string &, std::string *);
    llm_tokenizer_ugm::normalization_result normalize_prefix(const std::string &, size_t);
}
```
So we have one Trie for user-defined tokens and one for normal tokens.
In our case this is an empty Trie:
```console
(gdb) p user_defined_token_matcher
$34 = {children = std::map with 0 elements, has_value = false, value = 0}

(gdb) p user_defined_token_match
$51 = {first = 0x7fffffffd240 "What is LoRA?", second = 0}
```
Next, if the user-defined token match is greater than 0 we return the match:
```c++
        if (user_defined_token_match.second > 0) {
            return { &input[input_offset], user_defined_token_match.second, user_defined_token_match.second };
        }
```
Next we have: 
```
        size_t longest_prefix_length = 0;
        size_t longest_prefix_offset = 0;

        if (xcda_array_size > 0) {
            struct xcda_array_view xcda_view(xcda_array, xcda_array_size);
```
The `xcda_array_view` is a struct that wraps the compressed double array and
and has functions to access the elements using `get_base`, `get_lcheck`,
`get_value`, and `get_node`.
```c++
            // Find the longest normalized sequence matching the input prefix by walking
            // the XOR-compressed compact double array (XCDA) starting from the root node
            // We find the index of the next node by calculating BASE[s] ^ c where s is
            // the index of the previous node and c is a numerical character value
            uint32_t node_index = 0;

            // get BASE of the root node
            node_index = xcda_view.get_base(node_index);

            for (size_t prefix_offset = input_offset; prefix_offset < input.size(); prefix_offset++) {
                unsigned char c = input[prefix_offset];
                if (c == 0) {
                    break;
                }
                node_index ^= c;
                // if value of LCHECK is not c it means that this is not a child of
                // the previous node, so we stop matching
                if (xcda_view.get_lcheck(node_index) != c) {
                    break;
                }
                bool is_leaf = xcda_view.get_leaf(node_index);
                // get BASE of the current node
                node_index ^= xcda_view.get_base(node_index);
                // if LEAF of the current node is true, it means that its BASE points to the node
                // containing index of replacement sequence for currently matched input prefix
                if (is_leaf)
                {
                    longest_prefix_length = prefix_offset - input_offset + 1;
                    // get index of replacement sequence for currently matched input prefix
                    longest_prefix_offset = xcda_view.get_value(node_index);
                }
            }
```
To orient ourselves a little, the above for loop will iterate 
```console
(gdb) p input_size()
No symbol "input_size" in current context.
(gdb) p input.size()
$55 = 13
(gdb) p input_offset
$56 = 0

(gdb) p prefix_offset
$58 = 0
(gdb) p c
$60 = 87 'W'

```
So `c` will be 'W', and we get the base value for this node by XORing the
using the value of 'c'.
```c++
                node_index ^= c;
```
```console
(gdb) p node_index
$61 = 118
```
Next we verify that this 'path' was indeed inserted before but if not we break
out of the loop.
```c++
                if (xcda_view.get_lcheck(node_index) != c) {
                    break;
                }
```
Next we check if the current node is a leaf node, which is a node that marks
the end of a complete string (as opposed to prefixes) in the Trie.
```c++
                bool is_leaf = xcda_view.get_leaf(node_index);
```
In our case 'W' should note be a leaf node:
```console
(gdb) p is_leaf
$62 = false
```
Next we get the base for 'C', and we use `is_leaf` but since it is false we
don't enter this block (yet):
```
                node_index ^= xcda_view.get_base(node_index);

                // if LEAF of the current node is true, it means that its BASE points to the node
                // containing index of replacement sequence for currently matched input prefix
                if (is_leaf)
                {
                    longest_prefix_length = prefix_offset - input_offset + 1;
                    // get index of replacement sequence for currently matched input prefix
                    longest_prefix_offset = xcda_view.get_value(node_index);
                }
```
That will complete the first iteration of the for loop and at this point we have
handled 'W' of 'What is LoRA?'.
Next we have 'h':
```console
(gdb) p c
$64 = 104 'h'
(gdb) p node_index
$65 = 554

(gdb) p xcda_view.get_lcheck(node_index) != c
$67 = true
```
So notice that we will break out of the loop here since the `lcheck` value is
not equal to 'h'. This is because the 'h' is not a child of 'W' in the Trie.
```
                // if value of LCHECK is not c it means that this is not a child of
                // the previous node, so we stop matching
                if (xcda_view.get_lcheck(node_index) != c) {
                    break;
                }
```
Following this we have the following if statement and in our case
`longest_prefix_length` will be 0:
```
        if (longest_prefix_length > 0) {
            // we have a match, so return the replacement sequence
            if (longest_prefix_offset >= prefix_replacements_size) {
                throw std::runtime_error("Index out of array bounds in precompiled charsmap!");
            }
            const char * prefix_replacement = &prefix_replacements[longest_prefix_offset];
            return { prefix_replacement, strlen(prefix_replacement), longest_prefix_length };
        } else {
            // check if the input prefix contains a valid sequence of UTF-8 code units
            try {
                // if yes, return this sequence unmodified
                size_t prefix_offset = input_offset;
                unicode_cpt_from_utf8(input, prefix_offset);
                return { &input[input_offset], prefix_offset - input_offset, prefix_offset - input_offset };
            } catch (std::invalid_argument & /*ex*/) {
                // if no, consume 1 byte and return U+FFFD - REPLACEMENT CHARACTER
                return { "\xEF\xBF\xBD", 3, 1 };
            }
        }
```
Notice that `unicode_cpt_from_utf8` is called with the input string which is 
a `const std::string&` parameter and cannot be updated in the function. But the
`prefix_offset` is a reference to the original `input_offset` and will be
updated in the function. For 'W' this is just one byte in utf-8 so the offset
should be incremented by 1. This using 'aggregate initialization' or
'braced initialization' to return a struct of type `normalization_result`:
```c++
                return { &input[input_offset], prefix_offset - input_offset, prefix_offset - input_offset };
```
```c++
    // helper structure for returning normalization results
    struct normalization_result {
        const char * normalized;
        size_t normalized_len;
        size_t consumed_input;
    };
```
So this will return to the normalize function (recall that we were in
`normalize_prefix`):
```
(gdb) p norm_res
$78 = {normalized = 0x7fffffffd240 "What is LoRA?", normalized_len = 1, consumed_input = 1}
```
```c++
            for (size_t i = 0; i < norm_res.normalized_len; i++) {
                char c = norm_res.normalized[i];
                if (c != ' ') {
                    if (!processing_non_ws) {
                        processing_non_ws = true;
                        if ((shall_prepend_space && !is_space_prepended) || shall_merge_spaces) {
                            normalized->append(space);
                            is_space_prepended = true;
                        }
                    }
                    normalized->push_back(c);
                } else {
                    if (processing_non_ws) {
                        processing_non_ws = false;
                    }
                    if (!shall_merge_spaces) {
                        normalized->append(space);
                    }
                }
            }
```
So in this case we only have one entry in the normalized string which is 'W'.
`processing_non_ws` is a boolean that is set to true if we are processing a
non-whitespace character.  In this case we will append a "space" to the
normalized string (that was passed into this function and currently empty):
```console
(gdb) p *normalized
$83 = "▁"
```
And then we will add the 'W' to the normalized string:
```console
(gdb) p *normalized
$84 = "▁W"
```
So the normalization in this case is to replace spaces with the `escaped_space`.
So when we come to the next word 'is' we will also replace the space:
```console
(gdb) p *normalized
$103 = "▁What▁i"
```
The complete normalized string will be:
```console
(gdb) p normalized
$118 = "▁What▁is▁LoRA?"
```
So this will land us back in tokenize.
```c++
        size_t input_len = normalized.size();
        if (input_len == 0) {
            return;
        }
```
```
    struct best_tokenization {
        llama_token token_id;
        size_t input_offset;
        float score_sum;
    };
```
```console
(gdb) p input_len
$120 = 20
(gdb) p normalized
$121 = "▁What▁is▁LoRA?"
```
```c++
        // initialize score_sum to -FLT_MAX so it will be always lower than sums of token scores
        std::vector<struct best_tokenization> tokenization_results(input_len + 1, {vocab.special_unk_id, 0, -FLT_MAX});
```
Notice that this is creating a vector with a size of 21 and each element is a
struct of type `best_tokenization` with the values `{vocab.special_unk_id, 0,
-FLT_MAX}`. So the vector will look like this:
```console
(gdb) p tokenization_results
$122 = std::vector of length 21, capacity 21 = {
{token_id = 2, input_offset = 0, score_sum = -3.40282347e+38},
{token_id = 2, input_offset = 0, score_sum = -3.40282347e+38},
{token_id = 2, input_offset = 0, score_sum = -3.40282347e+38},
{token_id = 2, input_offset = 0, score_sum = -3.40282347e+38},
{token_id = 2, input_offset = 0, score_sum = -3.40282347e+38},
{token_id = 2, input_offset = 0, score_sum = -3.40282347e+38},
{token_id = 2, input_offset = 0, score_sum = -3.40282347e+38},
{token_id = 2, input_offset = 0, score_sum = -3.40282347e+38},
{token_id = 2, input_offset = 0, score_sum = -3.40282347e+38},
{token_id = 2, input_offset = 0, score_sum = -3.40282347e+38},
{token_id = 2, input_offset = 0, score_sum = -3.40282347e+38},
{token_id = 2, input_offset = 0, score_sum = -3.40282347e+38},
{token_id = 2, input_offset = 0, score_sum = -3.40282347e+38},
{token_id = 2, input_offset = 0, score_sum = -3.40282347e+38},
{token_id = 2, input_offset = 0, score_sum = -3.40282347e+38},
{token_id = 2, input_offset = 0, score_sum = -3.40282347e+38},
{token_id = 2, input_offset = 0, score_sum = -3.40282347e+38},
{token_id = 2, input_offset = 0, score_sum = -3.40282347e+38},
{token_id = 2, input_offset = 0, score_sum = -3.40282347e+38},
{token_id = 2, input_offset = 0, score_sum = -3.40282347e+38},
{token_id = 2, input_offset = 0, score_sum = -3.40282347e+38}}
```

Next we are doing to iterate over all the input characters which is now 20:
```c++
        // at the beginning tokenization score is zero
        tokenization_results[0] = { vocab.special_unk_id, 0, 0 };

        for (size_t input_offset = 0; input_offset < input_len;) {
            size_t prefix_offset = input_offset;
            // calculate how many code units are in the currently processed UTF code point
            size_t n_utf8_code_units = std::min<size_t>(unicode_len_utf8(normalized[input_offset]), input_len - input_offset);

            // traverse the token matcher trie to find a matching token
            bool single_codepoint_token_found = false;
            const struct best_tokenization & current_best = tokenization_results[input_offset];

            struct naive_trie * node  = token_matcher.traverse(normalized[prefix_offset++]);
```
So just to recap the normalized input looks like this:
```console
(gdb) p normalized
$128 = "▁What▁is▁LoRA?"
```
And we are going to go through this character by character. Now, the first
character it will look up is '▁' which is a special character and is what we
replacd spaces with (beginning of words). And this is an utf8 character that
is 3 codepoints long:
```console
(gdb) p n_utf8_code_units
$129 = 3
```
```console
(gdb) p current_best
$130 = (const llm_tokenizer_ugm::best_tokenization &) @0x555555aabf20: {token_id = 2, input_offset = 0, score_sum = 0}
```

We will then traverse the trie until we find a leaf node. In our case this
will be when we reach the end of the '▁' , that is the 3 character.
```c++
            while (prefix_offset <= input_len && node != NULL) {
                // check if we found valid token in prefix
                if (node->has_value) {
                    // check if it corresponds to the whole UTF code point
                    if (prefix_offset - input_offset == n_utf8_code_units) {
                        single_codepoint_token_found = true;
                    }
                    llama_token token_id = node->value;
                    const auto & token_data = vocab.id_to_token[token_id];

                    // we set the user-defined token scores to 0 to make them more likely to be selected
                    // (normal token scores are log probabilities, so they are negative)
                    // score type is double here to make tokenization results exactly
                    // the same as in the HF tokenizer using SentencePiece
                    const double token_score = llama_is_user_defined_token(vocab, token_id) ? 0.0 : token_data.score;
                    const double challenger_score = current_best.score_sum + token_score;
                    struct best_tokenization & current_champ = tokenization_results[prefix_offset];
                    if (challenger_score > current_champ.score_sum) {
                        struct best_tokenization challenger = { token_id, input_offset, (float) challenger_score };
                        current_champ = challenger;
                    }
                }
                node = node->traverse(normalized[prefix_offset++]);
            }
```
We can inspect the value of the node:
```console
(gdb) p vocab.id_to_token[node->value]
$145 = {text = "▁", score = -2.01229286, attr = LLAMA_TOKEN_ATTR_NORMAL}
```
We setting the 'challenger_score' to the sum of the current best score and the
token score:
```c++
                    const double challenger_score = current_best.score_sum + token_score;
```
```console
(gdb) p current_best.score_sum + token_data.score
$148 = -2.01229286
```
And we are then setting `current_champ` to entry 3 in the `tokenization_results`
```c++
                    struct best_tokenization & current_champ = tokenization_results[prefix_offset];
```
And if the challenger score is greater than the current champ score we update
the current champ:
```c++
                    if (challenger_score > current_champ.score_sum) {
                        struct best_tokenization challenger = { token_id, input_offset, (float) challenger_score };
                        current_champ = challenger;
                    }
```
And notice that we are dealing with references so this will update the entry
in the `tokenization_results` vector:
```console
(gdb) p tokenization_results[prefix_offset]
$156 = {token_id = 3, input_offset = 0, score_sum = -2.01229286}
```
Just to remind myself how what is happening here. We are iterating over all the
codepoints of the normalized string `▁What▁is▁LoRA?` which recall has a length
of 20 which is what `input_len` is below.
```c++
        size_t input_len = normalized.size();

        for (size_t input_offset = 0; input_offset < input_len;) {
            ...
            while (prefix_offset <= input_len && node != NULL) {
                ...
            }

            // move to the next UTF code point
            input_offset += n_utf8_code_units;
        }
```
In fact lets set a breakpoint here before this loop as I get the feeling I
might have to step through this a few times to understand it.
```console
(gdb) br llama-vocab.cpp:838 if normalized.compare("▁What▁is▁LoRA?") == 0
```
We are going through the normalized string, character by character.
So the normalized string is 20 codepoints as we have unicode characters what 
where added as part of the normalization process. So this will loop from 0 to up
20. But notice that `input_offset` gets incremented by the last line of the loop
and is incremented by the number of codepoints in the current utf-8 character.

```
"▁What▁is▁LoRA?"
 ↓
"▁"   n_utf8_code_units=3
=
```
Then we will access the the current character in the normalized string (notice
that the post increment operator is used here so that happens afterwards):
```
struct naive_trie * node  = token_matcher.traverse(normalized[prefix_offset++]);
```
So we are getting the first byte from normalized which is the first byte of the
unicode character '▁':
So '▁' is represented as 3 bytes in utf-8. So the first byte is `0xe2` which
```
(gdb) x/3xb &normalized[0]
0x555555bcaab0:	0xe2	0x96	0x81
```

```c++
    struct naive_trie * traverse(const char c) {
        auto res = children.find(c);
        if (res != children.end()) {
            return &res->second;
        } else {
            return NULL;
        }
    }
```
```console
(gdb) p c
$208 = -30 '\342'
```
And we are going to lookup this character in the map of `naive_trie`. Now, if
the is an entry in the map this means that there is a path in the trie for this
character and we return that node.

So at this point `prefix_offset=1` and `input_len=20` and node is not NULL.
Next we will enter the while loop:
```c++
            while (prefix_offset <= input_len && node != NULL) {
```
Now, if a node has a value this means that we have found a token in the trie
(which is called a terminal node). The value is the token id. So node is
non-null in this case but it does not have a value so we will search this node
for the next char in the 3 bytes string '▁', which is '0x96':
```c++
        node = node->traverse(normalized[prefix_offset++]);
```
And this will once again increment the `prefix_offset` which will become 2 after
this line. Then we will again iterate the while loop and the same thing will
happen as this node is not a terminal node. So we will search/lookup the
third byte of the utf-8 character '▁' which is '0x81':
```c++
        node = node->traverse(normalized[prefix_offset++]);
```
And again increment the `prefix_offset` which will become 3. And then we will
iterate the while loop but this this time the node has a value:
```console
(gdb) p node->value
$239 = 3
```
The value is the token id which is can inspect using:
```console
(gdb) p vocab.id_to_token[3]
$240 = {text = "▁", score = -2.01229286, attr = LLAMA_TOKEN_ATTR_NORMAL}
```

Lets take a closer look at this if block:
```c++
    if (node->has_value) {
        // check if it corresponds to the whole UTF code point
        if (prefix_offset - input_offset == n_utf8_code_units) {
            single_codepoint_token_found = true;
        }

        llama_token token_id = node->value;
        const auto & token_data = vocab.id_to_token[token_id];

        // we set the user-defined token scores to 0 to make them more likely to be selected
        // (normal token scores are log probabilities, so they are negative)
        // score type is double here to make tokenization results exactly
        // the same as in the HF tokenizer using SentencePiece
        const double token_score = llama_is_user_defined_token(vocab, token_id) ? 0.0 : token_data.score;

        const double challenger_score = current_best.score_sum + token_score;

        struct best_tokenization & current_champ = tokenization_results[prefix_offset];
        if (challenger_score > current_champ.score_sum) {
            struct best_tokenization challenger = { token_id, input_offset, (float) challenger_score };
            current_champ = challenger;
        }
    }
```
One thing to note here is that `current_best` is set outside of the while loop
and is using the current value of `input_offset` which is 0:
```console
(gdb) p tokenization_results[0]
$247 = {token_id = 2, input_offset = 0, score_sum = 0}
```
This is the highest score for the character in position 0 (the first character which in this
case is 0xe2).
```
tokenization_results
index  char    value
  0    0xe2    {token_id = 2, input_offset = 0, score_sum = 0}
```

And we are taking that score and adding it to the token we found in the trie 
score which is the following:
```console
(gdb) p token_data
$243 = (const llama_vocab::token_data &) @0x7ffff76c6088:
{text = "▁", score = -2.01229286, attr = LLAMA_TOKEN_ATTR_NORMAL}
```
We are dealing with log probabilities to we are using addition and not multiplication.
And recall that log probabilities are 0-negative infinity. The closer to 0 the more
probable and the lower, more negative, the less probable.

Then we saving a refenece in `challenger_score` pointing to  entry 3 in `tokenization_results`,
the current value of `prefix_offset`:
```console
(gdb) p tokenization_results[3]
$250 = {token_id = 2, input_offset = 0, score_sum = -3.40282347e+38}
```
```
tokenization_results
index  char    value
  0    0xe2    {token_id = 2, input_offset = 0, score_sum = 0}
  3    0x81    {token_id = 3, input_offset = 0, score_sum = -2.01229286}
```
```c++
    if (challenger_score > current_champ.score_sum) {
        struct best_tokenization challenger = { token_id, input_offset, (float) challenger_score };
        current_champ = challenger;
    }
    if (-2.01229286 > -3.40282347e+38) {
        struct best_tokenization challenger = { 3, 0, (float) -2.01229286 };
        current_champ = challenger;
    }
```


0xe2	0x96	0x81
_wip_
