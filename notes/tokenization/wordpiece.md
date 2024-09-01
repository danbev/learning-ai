## WordPiece
WordPiece is a subword tokenization algorithm that is used in BERT (Google). 

Like BPE, WordPiece starts with a vocabulary of individual characters.
However, it also includes the full words in the training corpus as initial
tokens.

The key difference lies in how WordPiece scores potential merges. While BPE uses
frequency, WordPiece uses a likelihood-based approach.

It calculates the likelihood of the corpus given the current vocabulary.
For each possible merge, it computes how much this merge would increase the
likelihood of the corpus. The merge that results in the highest increase in
likelihood is chosen.

WordPiece typically uses '##' as a prefix to denote subwords that don't start a
word. For example, "playing" might be tokenized as ["play", "##ing"].

_wip_

### llama.cpp tokenizer
For this exploration I used 
[jina-embeddings-v2-small-en-q5_k_m.gguf](https://huggingface.co/djuna/jina-embeddings-v2-small-en-Q5_K_M-GGUF)

```console
$ gdb --args tokenize
(gdb) br llama-vocab.cpp:653
Breakpoint 1 at 0x1e85a6: file src/llama-vocab.cpp, line 653.
```
For this walkthrough I've updated the prompt to include the Swedish character
'Å' which will be useful when looking at the preprocessing how that works:
```c++
    std::string prompt = "ÅWhat is LoRA?";
```

```c++
struct llm_tokenizer_wpm {
    llm_tokenizer_wpm(const llama_vocab & vocab): vocab(vocab) {}

    void tokenize(const std::string & text, std::vector<llama_vocab::id> & output) const {
        const auto & token_map = vocab.token_to_id;

        // normalize and split by whitespace
        std::vector<std::string> words = preprocess(text);
        ...
    }
```
Lets start by taking a look at `preprocess`:
```c++
    // TODO: reduce string copies by using cpts_offs array
    std::vector<std::string> preprocess(const std::string & text) const {
        const std::vector<uint32_t> cpts_nfd = unicode_cpts_normalize_nfd(unicode_cpts_from_utf8(text));
        std::vector<std::string> words(1, "");

        for (const uint32_t cpt : cpts_nfd) {
            const auto flags = unicode_cpt_flags(cpt);

            if (flags.is_whitespace) {
                if (words.back().size()) {  // finish previous word if any
                    words.emplace_back();
                }
                continue;
            }

            assert (!flags.is_separator);
            if (cpt == 0 || cpt == 0xFFFD || flags.is_control) {
                continue;
            }

            const std::string s = unicode_cpt_to_utf8(unicode_tolower(cpt));
            if (flags.is_punctuation || ( cpt < 0x7F && flags.is_symbol ) || is_chinese_char(cpt)) {
                if (words.back().size()) {  // finish previous word if any
                    words.emplace_back();
                }
                words.back() = s;       // single char word
                words.emplace_back();   // start a new word
            } else {
                words.back() += s;  // append char to word
            }
        }

        if (!words.back().size()) {
            words.pop_back();
        }

        return words;
    }
```
So the first line will take the input text and convert it into unicode code
points using:
```c++ 
std::vector<uint32_t> unicode_cpts_from_utf8(const std::string & utf8) {
    std::vector<uint32_t> result;
    result.reserve(utf8.size());
    size_t offset = 0;
    while (offset < utf8.size()) {
        result.push_back(unicode_cpt_from_utf8(utf8, offset));
    }
    return result;
}
```
We are converting from UTF-8 which is a variable-length encoding of 1-4 bytes to
a fixed lenght encoding of 4 bytes. So the size of the output vector will most
likely be larger than the input string (unless the string contains only utf-8
characters that are 4-bytes long that is).
So the above will start at 0 until reaching the end of the utf8 string and
for each character call
```c++
    unicode_cpt_from_utf8(utf8, offset)).
```
For our first character, the Swedish character 'Å` it will be 2 bytes in utf-8,
and in unicode it is represented by a unique code point `U+00C5`
(LATIN CAPITAL LETTER A WITH RING ABOVE). So in UTF-8 we have:
```
(gdb) x/2xb &utf8[0]
0x7fffffffd260:	0xc3	0x85

0xC3 = 1100 0011 which indicates that the character is 2 bytes long

(gdb) x/tb &utf8[0]
0x7fffffffd260:	11000011
(gdb) x/tb &utf8[1]
0x7fffffffd261:	10000101
```

```c++
uint32_t unicode_cpt_from_utf8(const std::string & utf8, size_t & offset) {
    ...
    if (!(utf8[offset + 0] & 0x20)) {
        if (offset + 1 >= utf8.size() || ! ((utf8[offset + 1] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x1f) << 6) | (utf8[offset + 1] & 0x3f);
        offset += 2;
        return result;
    }
    ...
    throw std::invalid_argument("failed to convert utf8 to codepoint");
}
```
We can inspect the result of this and see that this is indeed `U+00C5`:
```console
(gdb) x/x &result
0x7fffffffceec:	0xc5
```
And we increment the offset by 2 because the character was to bytes long
2 bytes long utf-8 encoding.
The next character will then be processed but this time the offset will be 2.
After all characters have been processed we will have a vector of code points
which will be passed into `unicode_cpts_normalize_nfd`:
```c++
    const std::vector<uint32_t> cpts_nfd = unicode_cpts_normalize_nfd(unicode_cpts_from_utf8(text));
```

NFS stands for Normal Form Decomposed. The Swedish character 'Å` can be
represented as `U+00C5` like we way above or as `U+0041 U+030A` (LATIN CAPITAL
LETTER A + COMBINING RING ABOVE). The latter is the decomposed form of the
character. So this is what decomposing means, breaking down a character into
its base character and the combining characters.
With that in mind lets take a look at `unicode_cpts_normalize_nfd`:
```c++
std::vector<uint32_t> unicode_cpts_normalize_nfd(const std::vector<uint32_t> & cpts) {
    auto comp = [] (const uint32_t cpt, const range_nfd & range) {
        return cpt < range.first;
    };
    std::vector<uint32_t> result(cpts.size());
    for (size_t i = 0; i < cpts.size(); ++i) {
        const uint32_t cpt = cpts[i];
        auto it = std::upper_bound(unicode_ranges_nfd.cbegin(), unicode_ranges_nfd.cend(), cpt, comp) - 1;
        result[i] = (it->first <= cpt && cpt <= it->last) ? it->nfd : cpt;
    }
    return result;
}
```
This will iterate over the code points and for each code point it will find the
range that the code point belongs (if any) and replace the code point with the
NFD code point. So for the Swedish character `Å` 'U+00C5` we will look this
up in the `unicode_ranges_nfd` vector:
```c++
const std::vector<range_nfd> unicode_ranges_nfd = {  // start, last, nfd
{0x000000, 0x000000, 0x000000},
{0x0000C0, 0x0000C5, 0x000041},
...
```
And this will return `U+0041` which is the decomposed form of `U+00C5`.
```console
(gdb) p/x it->nfd
$21 = 0x41
```
Back in preprocess we will now have a vector of code points that have been
```console
(gdb) p cpts_nfd
$34 = std::vector of length 14, capacity 14 = {65, 87, 104, 97, 116, 32, 105, 115, 32, 76, 111, 82, 65, 63}
```
Next we are creating a vector of strings with a single empty string:
```c++
        std::vector<std::string> words(1, "");
```
We are then going to iterate over the code points and for each code point we
get the flags for each code point.
```c++
        for (const uint32_t cpt : cpts_nfd) {
            const auto flags = unicode_cpt_flags(cpt);
            ...
```
```c++
codepoint_flags unicode_cpt_flags(const uint32_t cp) {
    static const codepoint_flags undef(codepoint_flags::UNDEFINED);
    static const auto cpt_flags = unicode_cpt_flags_array();
    return cp < cpt_flags.size() ? cpt_flags[cp] : undef;
}
```

```
struct codepoint_flags {
    enum {
        UNDEFINED       = 0x0001,
        NUMBER          = 0x0002,  // regex: \p{N}
        LETTER          = 0x0004,  // regex: \p{L}
        SEPARATOR       = 0x0008,  // regex: \p{Z}
        ACCENT_MARK     = 0x0010,  // regex: \p{M}
        PUNCTUATION     = 0x0020,  // regex: \p{P}
        SYMBOL          = 0x0040,  // regex: \p{S}
        CONTROL         = 0x0080,  // regex: \p{C}
        MASK_CATEGORIES = 0x00FF,
    };

    // codepoint type
    uint16_t is_undefined   : 1;
    uint16_t is_number      : 1;  // regex: \p{N}
    uint16_t is_letter      : 1;  // regex: \p{L}
    uint16_t is_separator   : 1;  // regex: \p{Z}
    uint16_t is_accent_mark : 1;  // regex: \p{M}
    uint16_t is_punctuation : 1;  // regex: \p{P}
    uint16_t is_symbol      : 1;  // regex: \p{S}
    uint16_t is_control     : 1;  // regex: \p{C}
    // helper flags
    uint16_t is_whitespace  : 1;  // regex: \s
    uint16_t is_lowercase   : 1;
    uint16_t is_uppercase   : 1;
    uint16_t is_nfd         : 1;

    // decode from uint16
    inline codepoint_flags(const uint16_t flags=0) {
        *reinterpret_cast<uint16_t*>(this) = flags;
    }

    inline uint16_t as_uint() const {
        return *reinterpret_cast<const uint16_t*>(this);
    }

    inline uint16_t category_flag() const {
        return this->as_uint() & MASK_CATEGORIES;
    }
};
```
```console
(gdb) p cpt_flags[65]
$38 = {
    is_undefined = 0,
    is_number = 0,
    is_letter = 1,
    is_separator = 0,
    is_accent_mark = 0,
    is_punctuation = 0,
    is_symbol = 0,
    is_control = 0,
    is_whitespace = 0,
    is_lowercase = 0,
    is_uppercase = 1,
    is_nfd = 1}
```
In our case we will continue to the following if statement:
```c++
            const std::string s = unicode_cpt_to_utf8(unicode_tolower(cpt));
            if (flags.is_punctuation || ( cpt < 0x7F && flags.is_symbol ) || is_chinese_char(cpt)) {
                if (words.back().size()) {  // finish previous word if any
                    words.emplace_back();
                }
                words.back() = s;       // single char word
                words.emplace_back();   // start a new word
            } else {
                words.back() += s;  // append char to word
            }
```
So this is getting the codepoint for the lowercase value of the current code
point by looking it up in a table. This is then converted to utf8.
So our first character `U+0041` (A) will be converted to `U+0061` (a):
```console
(gdb) p s
$40 = "a"
```
And this will be appended by getting a reference to the last element in the
vector and appending the string to the current value (which is an empty string
in this case). So after this words will look like this:
```console
(gdb) p words
$41 = std::vector of length 1, capacity 1 = {"a"}
```
The final words vector will look like this:
```console
(gdb) p words
$44 = std::vector of length 5, capacity 8 = {"awhat", "is", "lora", "?"}
```
So we have no whitespace in the words vector and we have split the input string
into words. This is the first step in the tokenization process.
The words vector will then be returned.

Next, in tokenize we will iterate over the words:
```c++
        // find the longest tokens that form the words
        for (const std::string & word : words) {
            // skip empty words
            if (word.size() == 0) {
                continue;
            }

            // prepend phantom space
            const std::string word1 = "\xe2\x96\x81" + word;
            const int n = word1.size();
```
Notice that this phantom space is a UTF-8 character that is 3 bytes long
and is the "▁" character, U+2581 (LOWER ONE EIGHTH BLOCK) to signify the start
of a word. We also saw this in the Unigram tokenizer:
```c++
    // escaped space symbol - U+2581 (Lower One Eighth Block)
    const std::string escaped_space = "\xE2\x96\x81";
```
```console
(gdb) p word1
$23 = "▁awhat"
```
```c++
            const int n = word1.size();

            const size_t current_tokens = output.size();

            // we're at the start of a new word
            // move through character position in word
            for (int i = 0; i < n; ++i) {
                // loop through possible match length
                bool match = false;
                for (int j = std::min(n, i + vocab.max_token_len + 1); j > i; j--) {
                    auto it = token_map.find(word1.substr(i, j - i));
                    if (it != token_map.end()) {
                        output.push_back(it->second);
                        match = true;
                        i = j - 1;
                        break;
                    }
                }

                if (!match) { // discard all
                    output.resize(current_tokens);
                    break;  // and discard next tokens
                }
            }

            // we didn't find any matches for this word
            if (current_tokens == output.size()) {
                output.push_back(vocab.special_unk_id);
            }
        }
```
Notice that this is using `vocab.max_token_len` which is:
```console
(gdb) p vocab.max_token_len
$29 = 21
```
So this models has a maximum length of 21 for a single token. So we use the min
of this and the length of the word to search for a substring of the first words
with the phantom space prepended.
```console
(gdb) p word1.substr(i, j-i)
$28 = "▁awhat"
```
This will not be found in the token map to we continue the loop and shorten 
the string searched for:
```console
(gdb) p word1.substr(i, j-i)
$37 = "▁awha"
(gdb) p word1.substr(i, j-i)
$38 = "▁awh
(gdb) p word1.substr(i, j-i)
$39 = "▁aw"

(gdb) p *it
$41 = {first = "▁aw", second = 22091}
(gdb) p vocab.id_to_token[22091]
$42 = {text = "▁aw", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}

(gdb) p output
$43 = std::vector of length 1, capacity 1 = {101}
(gdb) p vocab.id_to_token[101]
$44 = {text = "[CLS]", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}

(gdb) p output
$45 = std::vector of length 2, capacity 2 = {101, 22091}
```
So notice it starts with the longest possible token to match and then shortens
the string until it finds a match (longest match first approach). 

The loop will continue with i incremented to become 4 ("▁aw"). And recall that
word1 is "▁awhat" so the next substring will be "hat" which will be found and
added to the output:
```console
(gdb) p *it
$51 = {first = "hat", second = 12707}
(gdb) p output
$52 = std::vector of length 3, capacity 4 = {101, 22091, 12707}
```
We will then do the same thing for the next word "is":
```
(gdb) p output
$61 = std::vector of length 4, capacity 4 = {101, 22091, 12707, 2003}
```

The final state out output will become:
```console
(gdb) p output
$65 = std::vector of length 7, capacity 8 = {101, 22091, 12707, 2003, 8840, 2527, 1029}
```
And that is what tokenize will return. 

Back in `llama_tokenize_internal` we then have:
```c++
                if (add_special) {
                    GGML_ASSERT(vocab.special_sep_id != -1);
                    output.push_back(vocab.special_sep_id);
                }
```
Which in this case will add the special token `[SEP]`:
```console
(gdb) p vocab.id_to_token[vocab.special_sep_id]
$64 = {text = "[SEP]", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
```

So the final tokenized output will be:
```console
(gdb) p output
$66 = std::vector of length 8, capacity 8 = {101, 22091, 12707, 2003, 8840, 2527, 1029, 102}
```

_wip_
