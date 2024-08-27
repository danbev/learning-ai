## SentencePiece
This is tokenizer/detokenizer used a by a number of NLP libraries.
by gglm for example.

SentencePiece can use different tokenization algorithms like BPE, Unigram.

### Training using Byte Pair Encoding (BPE)
This involves taking a text corpus. Lets take this simple one:
```
The quick brown fox jumps over the lazy dog.
A quick brown dog jumps over the lazy fox.
```
Our initial vocabulary would be all the individual characters in the text:
```
['\u0020' (space), 'A', 'T', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
'.']
```
Next we count the most frequent character pairs:
```
('t', 'h'): 4
('e', ' '): 4
('o', 'v'): 2
('v', 'e'): 2
...
```
And then we merge the most frequent pair, for example replacing `th` with `th_`:
```
Th_e quick brown fox jumps over th_e lazy dog.
A quick brown dog jumps over th_e lazy fox.
```
The above process continues until we have a vocabulary of the desired size.

The training process typically produces two main outputs:

Vocabulary File:
This is a list of all subword units (tokens) learned during training, usually
stored as a text file. Each line contains a token and its index (or just the
token, depending on the format).
```
<unk>	0
<s>	1
</s>	2
▁	3
e	4
t	5
a	6
o	7
i	8
n	9
s	10
r	11
...
▁the	58
ing	59
▁of	60
...
```

The merge table/operations:
```
('▁', 't'): '▁t'
('▁t', 'h'): '▁th'
('▁th', 'e'): '▁the'
('i', 'n'): 'in'
('in', 'g'): 'ing'
...
```

When we use the tokenizer to tokenize a sentence, we use both the vocabulary
file and the merge table to split the sentence into subword units.

The tokenization process applies these merges to the input text to produce the
final tokens.  To use the trained model:

* Load the model file
* Apply normalization to input text
* Start with characters/basic units
* Apply merge operations sequentially
* Look up resulting subwords in the vocabulary

### Byte fallback
If a token is not in the vocabulary, the tokenizer will fall back to a byte
level tokenizer. This is useful for handling out-of-vocabulary words.
For example:

Let's say we have a SentencePiece model trained on English text, and we
encounter this input:
```
"Hello, こんにちは! 😊"
```
Without byte fallback, the tokenizer might produce:
["Hello", ",", " ", <UNK>, <UNK>, <UNK>, "!", <UNK>]
```
With byte fallback:
```
["Hello", ",", " ", "<0xE3>", "<0x81>", "<0x93>", "<0xE3>", ...
(more bytes for Japanese characters) ..., "<0xF0>", "<0x9F>", "<0x98>", "<0x8A>"]
```
The byte fallback version preserves all information, allowing for perfect
reconstruction of the original input, even for characters not in the training
data.

### Detokenizing
This was something that I had not considered but not all tokenizers are
reversible. For example they might split up tokens so that is not possible
know how to put them back together again. For example:
```
"Hello world" -> "Hello" "world"
```

### SentencePiece in llama.cpp
This section will set through [tokenize.cpp](../fundamentals/llama.cpp/tokenize.cpp)
which be built and started using the following commands:
```console
$ cd ../fundamentals/llama.cpp/
$ make tokenize
$ gdb --args ./tokenize
(gdb) br tokenize.cpp:58
Breakpoint 1 at 0x168cf: file src/tokenize.cpp, line 58.
```

Now, in `llama_tokenize_internal` we have a switch statement on the
`vocab.type` which in this case is:
```console
(gdb) p vocab.type
$57 = LLAMA_VOCAB_TYPE_SPM
```
```c++
                bool is_prev_special = true;  // prefix with space if first token

                if (add_special && vocab.tokenizer_add_bos) {
                    GGML_ASSERT(vocab.special_bos_id != -1);
                    output.push_back(vocab.special_bos_id);
                    is_prev_special = true;
                }
```
I'm not sure why `is_prev_special` is set to true again here. But notice that
this is adding the special token for the beginning of the sentence.
```console
(gdb) p vocab.special_bos_id
$61 = 1
(gdb) p vocab.id_to_token[vocab.special_bos_id]
$62 = {text = "<s>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) p output
$63 = std::vector of length 1, capacity 1 = {1}
```
Next we iterate over all the fragments (see 'tokenizer_st_partition' above) and
```c++
                for (const auto & fragment : fragment_buffer) {
```
And in this case the `fragement_buffer` will contain the following:
```console
(gdb) p fragment_buffer
$65 = std::forward_list = {
[0] = {type = FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN, token = 1, _dummy = "",
      raw_text = "", offset = 0, length = 0},
[1] = {type = FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT, token = -1, _dummy = "",
      raw_text = "<s>What is LoRA?</s>", offset = 3, length = 13},
[2] = {type = FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN, token = 2, _dummy = "",
      raw_text = "", offset = 0, length = 0}}
```

Next we check the type of the fragment which in this case is of type token as
we can see above:
```c++
                    if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                       ...
                    } else { // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
                        output.push_back(fragment.token);
                        is_prev_special = true;
                    }
```
So this token gets added to the output:
```console
(gdb) p output
$66 = std::vector of length 2, capacity 2 = {1, 1}
```
Next iteration the fragment will be the raw string
```console
(gdb) p raw_text
$68 = "What is LoRA?"
```
And notice that a space is added before the raw text:
```c++
                        // prefix with space if previous is special
                        if (vocab.tokenizer_add_space_prefix && is_prev_special) {
                            raw_text = " " + raw_text;
                        }
```
Then a new `llm_tokenizer_spm` is created and the `tokenize` function is called:
```c++

                        llm_tokenizer_spm tokenizer(vocab);
                        llama_escape_whitespace(raw_text);
                        tokenizer.tokenize(raw_text, output);
                        is_prev_special = false;
```
The above will create a new tokenizer object, and then replace all whitespace
characters with a block underscore character `_`. I wonder what this function
is called `llama_escape_whitespace` instead of something like
`llama_replace_whitespace`.
```console
(gdb) p raw_text
$69 = " What is LoRA?"
(gdb) n
1296	                        tokenizer.tokenize(raw_text, output);
(gdb) p raw_text
$70 = "▁What▁is▁LoRA?
```
```c++
    void tokenize(const std::string & text, std::vector<llama_vocab::id> & output) {
        // split string into utf8 chars
        int index = 0;
        size_t offs = 0;
        while (offs < text.size()) {
```
Recall that `text.size()` return the number of bytes in the string, not the
number of characters.
```console
(gdb) p text
$72 = "▁What▁is▁LoRA?"
(gdb) p text.size()
$71 = 20
```
What will happen is that the string will be split into utf8 characters.
```c++
        int index = 0;
        size_t offs = 0;
        while (offs < text.size()) {
            llm_symbol sym;
            // Get length of utf8 character.
            size_t len = unicode_len_utf8(text[offs]);
            // Set the char* to the start of the utf8 character.
            sym.text = text.c_str() + offs;
            // Set the size of the utf8 character.
            sym.n = std::min(len, text.size() - offs);
            offs += sym.n;
            // Update the linked list indices prev and next.
            sym.prev = index - 1;
            sym.next = offs == text.size() ? -1 : index + 1;
            index++;
            symbols.emplace_back(sym);
        }
```
So each utf8 character is stored in a `llm_symbol` struct and then added to the
`symbols` vector:
```console
(gdb) until 224
llm_tokenizer_spm::tokenize (this=0x7fffffffd200, text="▁What▁is▁LoRA?", 
    output=std::vector of length 2, capacity 2 = {...}) at src/llama-vocab.cpp:224
224	        for (size_t i = 1; i < symbols.size(); ++i) {
(gdb) p symbols
$92 = std::vector of length 14, capacity 16 = {
{prev = -1, next = 1,  text = 0x555555a9d830 "▁What▁is▁LoRA?", n = 3},
{prev =  0, next = 2,  text = 0x555555a9d833 "What▁is▁LoRA?",  n = 1},
{prev =  1, next = 3,  text = 0x555555a9d834 "hat▁is▁LoRA?",   n = 1},
{prev =  2, next = 4,  text = 0x555555a9d835 "at▁is▁LoRA?",    n = 1},
{prev =  3, next = 5,  text = 0x555555a9d836 "t▁is▁LoRA?",     n = 1},
{prev =  4, next = 6,  text = 0x555555a9d837 "▁is▁LoRA?",      n = 3},
{prev =  5, next = 7,  text = 0x555555a9d83a "is▁LoRA?",       n = 1},
{prev =  6, next = 8,  text = 0x555555a9d83b "s▁LoRA?",        n = 1},
{prev =  7, next = 9,  text = 0x555555a9d83c "▁LoRA?",         n = 3},
{prev =  8, next = 10, text = 0x555555a9d83f "LoRA?",          n = 1},
{prev =  9, next = 11, text = 0x555555a9d840 "oRA?",           n = 1},
{prev = 10, next = 12, text = 0x555555a9d841 "RA?",            n = 1},
{prev = 11, next = 13, text = 0x555555a9d842 "A?",             n = 1},
{prev = 12, next = -1, text = 0x555555a9d843 "?",              n = 1}}
```
Notice that the `_` (block underscore) is 3 bytes long and the others are only
one as they are the same in ascii which are only one byte in UTF8.

```c
struct llm_symbol {
    using index = int;
    index prev;
    index next;
    const char * text;
    size_t n;
};
```
So a symbol entry has the index to the previous utf8 character, and the next
utf8 character in the string. It also has a char* to the current utf8
character and the size of the utf8 character.  The `prev` and `next`
allow this struct to act like a doubly linked list.

When tokenizing, especially with subword tokenization algorithms, you often need
to merge adjacent symbols. With prev and next indices, you can easily merge
symbols by updating these indices without moving data in memory.
It allows for processing symbols in a non-contiguous manner. You can "remove" a
symbol from the sequence by adjusting the prev and next pointers of its
neighbors, without physically removing it from the array.
If you need to remove a symbol during processing, you can do so by updating the
prev and next indices of adjacent symbols, rather than shifting all subsequent
elements in an array.

Take the string "hello" as an example:
```
Index: 0    1    2    3    4
Char:  H    e    l    l    o
text:  H    e    l    l    o
n:     1    1    1    1    1
prev:  -1   0    1    2    3
next:  1    2    3    4    -1
```
Now, lets say we are using Byte Pair Encoding (BPE) and the merging decides to
merge `l` and `o`, the last two characters.
```
Index: 0    1    2    3    4
Char:  H    e    l    lo   o
text:  H    e    l    lo   o
n:     1    1    1    2    1
prev:  -1   0    1    2    3
next:  1    2    3    -1   -1
```
Notice that we have just manipulated `prev` and `next` and the actual string
has not been updated at all.

Next we will iterate over all the symbols added above:
```c
        // seed the work queue with all possible 2-character tokens.
        for (size_t i = 1; i < symbols.size(); ++i) {
            try_add_bigram(i - 1, i);
        }
```
If symbols.size() is larger than `INT_MAX` (typically 2^31 - 1 or about 2.14
billion), this conversion could lead to overflow and undefined behavior.
But I don't think it is reasonable that the symbols, that is the number of
utf8 character to tokenize exceeds this value.

So at this point we have the utf8 characters from the input string to be
tokenized which are stored as `llm_symbol` in the symbols vector. Now, we
are going to iterate through them and call `try_add_bigram` which will try
to add adjacent utf8 characters to the work queue.
```c++
    void try_add_bigram(int left, int right) {
        if (left == -1 || right == -1) {
            return;
        }

        const std::string text = std::string(symbols[left].text, symbols[left].n + symbols[right].n);
        auto token = vocab.token_to_id.find(text);

        if (token == vocab.token_to_id.end()) {
            return;
        }

        if (static_cast<size_t>((*token).second) >= vocab.id_to_token.size()) {
            return;
        }

        const auto & tok_data = vocab.id_to_token[(*token).second];

        llm_bigram_spm bigram;
        bigram.left  = left;
        bigram.right = right;
        bigram.score = tok_data.score;
        bigram.size  = text.size();

        work_queue.push(bigram);

        // Do we need to support is_unused?
        rev_merge[text] = std::make_pair(left, right);
    }
```
Notice that the std::string constructor used here is the range constructor with
a pointer to the start of the string and the count of characters to copy:
```c++
const std::string text = std::string(symbols[left].text, symbols[left].n + symbols[right].n);
```
```console
(gdb) p symbols[left]
$4 = {prev = -1, next = 1, text = 0x555555a9d830 "▁What▁is▁LoRA?", n = 3}

(gdb) p symbols[right]
$5 = {prev = 0, next = 2, text = 0x555555a9d833 "What▁is▁LoRA?", n = 1}

(gdb) p symbols[right].n + symbols[left].n
$7 = 4
```
So we are specifying that the pointer to the new string should be that of the
left symbols char* (text member) and the `count.
So text in in this case should be '_W' (recall that the underscore is a block
underscore character and is 3 bytes long).
```console
(gdb) p unicode_len_utf8(text[0])
$10 = 3
(gdb) x/s &text[0]
```
This text will then be searched for in the models vocabulary. If it is not found
this function just returns.
In our case the iterator returned is:
```console
(gdb) p *token
$15 = {first = "▁W", second = 399}
```
Second in this case is the token id for this bigram and we can check if it is
in fact in the vocabulary:
```console
(gdb) p vocab.id_to_token[399]
$17 = {text = "▁W", score = -140, attr = LLAMA_TOKEN_ATTR_NORMAL}
```
Next we create a new `llm_bigram_spm` struct and add it to the work queue:
```c++
        llm_bigram_spm bigram;
        bigram.left  = left;
        bigram.right = right;
        bigram.score = tok_data.score;
        bigram.size  = text.size();

        work_queue.push(bigram);
```
```console
(gdb) p bigram
$22 = {left = 0, right = 1, score = -140, size = 4}
```
Notice that only the indices, the score, and the size is stored and no pointer
or token id.
```c++
    llm_bigram_spm::queue work_queue;
```
And queue is defines as follows in the `llm_bigram_spm` struct:
```
struct llm_bigram_spm {
    struct comparator {
        bool operator()(llm_bigram_spm & l, llm_bigram_spm & r) {
            return (l.score < r.score) || (l.score == r.score && l.left > r.left);
        }
    };
    using queue_storage = std::vector<llm_bigram_spm>;
    using queue = std::priority_queue<llm_bigram_spm, queue_storage, comparator>;
    llm_symbol::index left;
    llm_symbol::index right;
    float score;
    size_t size;
};
```
The last thing to happen in try_add_bigram is that the bigram is added to the
`rev_merge` map:
```c++
        rev_merge[text] = std::make_pair(left, right);
```
`rev_merge` is defined as:
```c++
    std::map<std::string, std::pair<int, int>> rev_merge;
```
In our case we know that text is "_W" and left is 0 and right is 1:
```console
(gdb) p rev_merge
$26 = std::map with 1 element = {["▁W"] = {first = 0, second = 1}}
```

The above might also be written as:
```c++
        rev_merge[text] = {left, right};
```
All the symbols will be added in the same way and afterwards the work_queue
and rev_merge will look like this:
```console
(gdb) p work_queue
$33 = std::priority_queue wrapping: std::vector of length 9, capacity 16 = {
{left = 3, right = 4, score = -12, size = 2},
{left = 0, right = 1, score = -140, size = 4},
{left = 6, right = 7, score = -16, size = 2},
{left = 9, right = 10, score = -3151, size = 2},
{left = 5, right = 6, score = -215, size = 4},
{left = 2, right = 3, score = -2091, size = 2},
{left = 8, right = 9, score = -106, size = 4},
{left = 1, right = 2, score = -8550, size = 2},
{left = 11, right = 12, score = -4458, size = 2}}

(gdb) p rev_merge
$34 = std::map with 9 elements = {
["Lo"] = {first = 9, second = 10},
["RA"] = {first = 11, second = 12},
["Wh"] = {first = 1, second = 2},
["at"] = {first = 3, second = 4},
["ha"] = {first = 2, second = 3},
["is"] = {first = 6, second = 7},
["▁L"] = {first = 8, second = 9},
["▁W"] = {first = 0, second = 1},
["▁i"] = {first = 5, second = 6}}
```
So after this the work_queue will be iterated over in the following loop:
```c++
        while (!work_queue.empty()) {
            auto bigram = work_queue.top();
            work_queue.pop();

            auto & left_sym = symbols[bigram.left];
            auto & right_sym = symbols[bigram.right];

            // if one of the symbols already got merged, skip it.
            if (left_sym.n == 0 || right_sym.n == 0 ||
                left_sym.n + right_sym.n != bigram.size) {
                continue;
            }

            // merge the right sym into the left one
            left_sym.n += right_sym.n;
            right_sym.n = 0;

            //LLAMA_LOG_INFO("left = '%*s' size = %zu\n", (int) left_sym.n, left_sym.text, bigram.size);

            // remove the right sym from the chain
            left_sym.next = right_sym.next;
            if (right_sym.next >= 0) {
                symbols[right_sym.next].prev = bigram.left;
            }

            // find more substitutions
            try_add_bigram(left_sym.prev, bigram.left);
            try_add_bigram(bigram.left, left_sym.next);
        }

        for (int i = 0; i != -1; i = symbols[i].next) {
            auto & symbol = symbols[i];
            resegment(symbol, output);
        }
```
First we get the top element from the work_queue using top and pop, and notice
that the first element is:
```console
(gdb) p bigram
$35 = {left = 3, right = 4, score = -12, size = 2}
```
And also notice that we are still using the symbols vector to get teh left and
right symbols.
```console
(gdb) p left_sym
$36 = (llm_symbol &) @0x555555a99cb8: {prev = 2, next = 4, text = 0x555555a9d835 "at▁is▁LoRA?", n = 1}
(gdb) p right_sym
$37 = (llm_symbol &) @0x555555a99cd0: {prev = 3, next = 5, text = 0x555555a9d836 "t▁is▁LoRA?", n = 1}
```
So we are going to merge the right symbol with the left symbol, which is done
by increating the lenght of the left symbol and setting the length of the right
symbol to 0:
```c++
            left_sym.n += right_sym.n;
            right_sym.n = 0;
```
And we also need to remove the right symbol from the linked list:
```c++
            left_sym.next = right_sym.next;
            // And the symbols after the old right need to point to our merged symbol.
            if (right_sym.next >= 0) {
                symbols[right_sym.next].prev = bigram.left;
            }
```
So that merged the two symbols into one and removed the right symbol from the
linked list. Then we have the following:
```c++
            // find more substitutions
            try_add_bigram(left_sym.prev, bigram.left);
            try_add_bigram(bigram.left, left_sym.next);
```
Now, after we have merged the left and right into the left symbol, that merge
might not be able to generate new merges that were not there before this merge.
So we look to the left of the newly merged symbol and try to find any bigram
pair and if so add it to the work queue. And the same for the right which my now
be possible to make a bigram. This is the above two lines of code are doing.

After all the the work_queue has been emptied we will iterate over all the
the symbols until we reach the end of the linked list:
```
        for (int i = 0; i != -1; i = symbols[i].next) {
            auto & symbol = symbols[i];
            resegment(symbol, output);
        }
```
```console
(gdb) p symbols
$53 = std::vector of length 14, capacity 16 = {
{prev = -1, next = 5, text = 0x555555a9d830 "▁What▁is▁LoRA?", n = 7},
{prev = 0, next = 2, text = 0x555555a9d833 "What▁is▁LoRA?", n = 0},
{prev = 0, next = 3, text = 0x555555a9d834 "hat▁is▁LoRA?", n = 0},
{prev = 0, next = 5, text = 0x555555a9d835 "at▁is▁LoRA?", n = 0},
{prev = 3, next = 5, text = 0x555555a9d836 "t▁is▁LoRA?", n = 0},
{prev = 0, next = 8, text = 0x555555a9d837 "▁is▁LoRA?", n = 5},
{prev = 5, next = 8, text = 0x555555a9d83a "is▁LoRA?", n = 0},
{prev = 6, next = 8, text = 0x555555a9d83b "s▁LoRA?", n = 0},
{prev = 5, next = 11, text = 0x555555a9d83c "▁LoRA?", n = 5},
{prev = 8, next = 10, text = 0x555555a9d83f "LoRA?", n = 0},
{prev = 8, next = 11, text = 0x555555a9d840 "oRA?", n = 0},
{prev = 8, next = 13, text = 0x555555a9d841 "RA?", n = 2},
{prev = 11, next = 13, text = 0x555555a9d842 "A?", n = 0},
{prev = 11, next = -1, text = 0x555555a9d843 "?", n = 1}}
```
So our first symbol will be:
```console
(gdb) p symbol
$55 = (llm_symbol &) @0x555555a99c70: {prev = -1, next = 5, text = 0x555555a9d830 "▁What▁is▁LoRA?", n = 7}
```
Which will be passed to resegment.
At this stage our output vector looks like this:
```console
(gdb) p output
$56 = std::vector of length 2, capacity 2 = {1, 1}
```
```c++
    void resegment(llm_symbol & symbol, std::vector<llama_vocab::id> & output) {
        auto text = std::string(symbol.text, symbol.n);
        auto token = vocab.token_to_id.find(text);

        // Do we need to support is_unused?
        if (token != vocab.token_to_id.end()) {
            output.push_back((*token).second);
            return;
        }

        const auto p = rev_merge.find(text);

        if (p == rev_merge.end()) {
            // output any symbols that did not form tokens as bytes.
            output.reserve(output.size() + symbol.n);
            for (int j = 0; j < (int)symbol.n; ++j) {
                llama_vocab::id token_id = llama_byte_to_token_impl(vocab, symbol.text[j]);
                output.push_back(token_id);
            }
            return;
        }

        resegment(symbols[p->second.first],  output);
        resegment(symbols[p->second.second], output);
    }
```
So this first creates a string from the passed in symbol and then searches the
vocab for this token. If it is found it is added to the output vector and we
```console
(gdb) p *token
$59 = {first = "▁What", second = 1724}
```
If the token was found it the tokens (second) is added to the output vector.
```console
(gdb) p output
$61 = std::vector of length 3, capacity 4 = {1, 1, 1724}
```
And if the token was not in the vocabulary it is looked up in the `rev_merge`
map. If the token was not found in the `rev_merge`, recall that `try_add_bigram`
searches the vocabulary for bigrams, and if it is not found there it is not
added to the `rev_merge` map. In this case we iterate over the utf8 characters 
in the symbol and add them to the output vector. This is done by calling
`llama_byte_to_token_impl`:
```c++
llama_token llama_byte_to_token_impl(const llama_vocab & vocab, uint8_t ch) {
    GGML_ASSERT(llama_vocab_get_type(vocab) != LLAMA_VOCAB_TYPE_NONE);
    static const char * hex = "0123456789ABCDEF";
    switch (llama_vocab_get_type(vocab)) {
        case LLAMA_VOCAB_TYPE_SPM:
        case LLAMA_VOCAB_TYPE_UGM: {
            const char buf[7] = { '<', '0', 'x', hex[ch >> 4], hex[ch & 15], '>', 0 };
            auto token = vocab.token_to_id.find(buf);
            if (token != vocab.token_to_id.end()) {
                return (*token).second;
            }
            // Try to fall back to just the byte as a string
            const char buf2[2] = { (char)ch, 0 };
            return vocab.token_to_id.at(buf2);
        }
        case LLAMA_VOCAB_TYPE_WPM:
        case LLAMA_VOCAB_TYPE_BPE: {
            return vocab.token_to_id.at(unicode_byte_to_utf8(ch));
        }
        default:
            GGML_ABORT("fatal error");
    }
}
```
Notice that this is will perform another lookup in the vocabulary, but this time 
it will look for the byte as a string (and not the symbol text). This is the
byte fall back part of SPM. When the tokenizer encounters a sequence of
characters that it can't match to any token in its vocabulary, it falls back to
encoding each byte individually. And the vocabulary would have tokens for these
bytes in the vocabulary.

_wip_
