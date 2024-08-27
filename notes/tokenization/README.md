## Tokenization
The tokenization process involves splitting the input text into tokens, which
are then looked up in the vocabulary to get the token IDs.

Models include a vocabulary file, which is a list of all the tokens in the
model. There might be a configuration file in addition to this that specifies
the type of tokenizing that the model uses, like Byte-Pair Encoding (BPE),
WordPiece, SentencePiece, or Unigram, etc.

* [Byte Pair Encoding (BPE)](./bpe.md)
* [WordPiece](./wordpiece.md) TODO
* [SentencePiece](./sentencepiece.md)
* [Unigram](./unigram.md) TODO
