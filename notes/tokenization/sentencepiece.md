## SentencePiece
This is tokenizer/detokenizer used a by a number of NLP libraries.
by gglm for example.

### Training
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
