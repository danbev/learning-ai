### llama-vocab.cpp warnings

Warning on master:
```console`
src\llama-vocab.cpp(138,26): warning C4244: 'return': conversion from 'long' to 'uint8_t', possible loss of data
src\llama-vocab.cpp(211,1): warning C4267: 'argument': conversion from 'size_t' to 'int', possible loss of data
src\llama-vocab.cpp(517,1): warning C4267: 'argument': conversion from 'size_t' to 'int', possible loss of data
src\llama-vocab.cpp(557,1): warning C4267: '=': conversion from 'size_t' to 'llm_symbol::index', possible loss of data
src\llama-vocab.cpp(560,1): warning C4267: '=': conversion from 'size_t' to 'int', possible loss of data
src\llama-vocab.cpp(654,1): warning C4267: 'initializing': conversion from 'size_t' to 'int', possible loss of data
src\llama-vocab.cpp(654,1): warning C4267: 'initializing': conversion from 'size_t' to 'const int', possible loss of data
src\llama-vocab.cpp(1517,22): warning C4267: 'return': conversion from 'size_t' to 'int32_t', possible loss of data
````


```
char src = 'A'; // 0x41
// binary 01000001 
uint8_t highbits = static_cast<uint8_t>(src) >> 4;
// binary 01000001 
              0100 = 4
          
lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
lookup[4] = 1

```
```c
    void tokenize(const std::string & text, std::vector<llama_vocab::id> & output) {
        // split string into utf8 chars
        int index = 0;
        size_t offs = 0;
        while (offs < text.size()) {
            llm_symbol sym;
            size_t len = unicode_len_utf8(text[offs]);
            sym.text = text.c_str() + offs;

            // In utf8 the first byte determins the lenght of the string 1-4 bytes
            // We only used the first char of the string to get the length above. But if
            // the string has been truncated or something there might not be any more
            // chars. This is why this extra check is here to catch this case.
            sym.n = std::min(len, text.size() - offs);
            // The make off be the index of the next char in the string to tokenize which should be the next utf8 character.
            offs += sym.n;

            sym.prev = index - 1;
            sym.next = offs == text.size() ? -1 : index + 1;
            index++;
            symbols.emplace_back(sym);
        }
```
And notice we are adding each llm_symbol created to the symbols vector:
```c
    std::vector<llm_symbol> symbols;
```

```c
struct llm_symbol {
    using index = int;
    index prev;
    index next;
    const char * text;
    size_t n;
};
```
So a symbol entry has the index to the previous utf8 charater, and the next
utf8 character in the string. It also has a char* to the current utf8
character and the size of the utf8 character.  The `prev` and `next`
allow this struct to act like a doubly linked list.

When tokenizing, especially with subword tokenization algorithms, you often need to merge adjacent symbols. With prev and next indices, you can easily merge symbols by updating these indices without moving data in memory.
It allows for processing symbols in a non-contiguous manner. You can "remove" a symbol from the sequence by adjusting the prev and next pointers of its neighbors, without physically removing it from the array.
If you need to remove a symbol during processing, you can do so by updating the prev and next indices of adjacent symbols, rather than shifting all subsequent elements in an array.

```
Index: 0    1    2    3    4
Char:  H    e    l    l    o
text:  H    e    l    l    o
n:     1    1    1    1    1
prev:  -1   0    1    2    3
next:  1    2    3    4    -1
```
Now, lets say we are using BPE and the merging decides to merge `l` and `o`,
the last two characters.
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

```c
        // seed the work queue with all possible 2-character tokens.
        for (size_t i = 1; i < symbols.size(); ++i) {
            try_add_bigram(i - 1, i);
        }
```
If symbols.size() is larger than INT_MAX (typically 2^31 - 1 or about 2.14
billion), this conversion could lead to overflow and undefined behavior.
But I don't think it is reasonable that the symbols, that is the number of
unicode character to tokenize exceeds this value.

So at the point we have the utf8 characters from the input string to be
tokenized which are stored as llm_symbol in the symbols vector. Now, we
are going to iterate through them and call `try_add_bigram`:
```c
    void add_new_bigram(int left, int right) {
        if (left == -1 || right == -1) {
            return;
        }

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
So, we called this only the integers -1 and 1 (the first time),
so left will be -1 and right 1. Next it will extract the strings
for each of these utf8 characters.
Next we are checking the if this pair of tokens (bigram) exists in our BPE
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
of tokens to be merged (is this merges.txt or something like that I've seen?).
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
This is populated in llama.cpp by by llm_load_vocab:
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
Recall that `left` and `right` are indexes not the strings.
```c++
    llm_bigram_bpe::queue work_queue;

    using queue_storage = std::vector<llm_bigram_bpe>;
    using queue = std::priority_queue<llm_bigram_bpe, queue_storage, comparator>;

    llm_bigram_bpe::queue work_queue;
```
So the priority queue will contain pairs (bigrams) with the highest rank (lowest value)
first.
Back in tokenize we then have:
```c++
            // build token(s)
            while (!work_queue.empty()) {
                auto bigram = work_queue.top();
                work_queue.pop();
                ...
            }
```
Recall that `top()` will return a const_reference to the top most element, but will not
remove it (the function is const):
```c++
const_reference top() const;
void pop();
```
And note that `pop()` is void. 

Notice that the following will create a new llm_bigram_bpe by using the copy constructor of
llm_bigram_bpe:
```c++
                auto bigram = work_queue.top();
```
So a new llm_bigram_bpe struct will be created on the stack and the all the members
copied. Now for `left` and `right` these are just integers so they are trivial to
copy. The same for `rank` and `size`. But for `text` a new string is allocated and
all characters copied.
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
Now, then we go through the work_queue until it is empty:
```c++
// build token(s)
            while (!work_queue.empty()) {
                auto bigram = std::move(const_cast<llm_bigram_bpe&>(work_queue.top()));
                work_queue.pop();

                auto & left_symbol = symbols[bigram.left];
                auto & right_symbol = symbols[bigram.right];

                // 
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
The last two lines are adding 
[A] -> [B] -> [C] -> [D] -> [E]

[A] -> [B] -> [C] -> [D] -> [E]
       ^     ^
       |     |
    left   right

[A] -> [BC] -> [D] -> [E]

add_new_bigram(left_symbol.prev, bigram.left);  // left side of current symbol
This adds a bigram for the symbol before BC and BC itself.
In our example, this would be A-BC.

add_new_bigram(bigram.left, left_symbol.next);  // right side of current symbol
This adds a bigram for BC and the symbol after it.
In our example, this would be BC-D.

[A] -> [BC] -> [D] -> [E]

But two new bigrams have been added to the work queue for future consideration:

A-BC
BC-D

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


After this we have:
```c++
for (int i = 0; i != -1; i = symbols[i].next) {
            auto & symbol = symbols[i];
            resegment(symbol, output);
        }
```
And resegment looks like this:
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
Now, I completely missing `rev_merge` when looking through try_add_bigram
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

So this map looks like this:
```c++
std::map<std::string, std::pair<int, int>> rev_merge;
```
And this is a map from a string, in this case a merged token to the pairs
that were merged to create it.
Let's say we merge symbols "A" (index 1) and "B" (index 2) into "AB". The rev_merge map would then contain:
{"AB" : (1, 2)}
This entry tells us that the symbol "AB" was formed by merging the symbols at indices 1 and 2.

