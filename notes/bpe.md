## Byte-Pair Encoding (BPE)
This is a subword tokenization algorithm.

### Training Process
During the training learning process we take a corpus of text and split it into
individual characters. So this will give us a text file of characters.

1) We then iterate over all of these characters and count the frequency adjacent
pairs.

2) The most frequent pair is merged into a single token. This merge operation is
recorded as a rule in a file, typically named merges.txt, where each line
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



### Token segmenter
Above we described the training process which produced a `merges.txt` file.
When we use this for tokenization we will have new text that we want to tokenize
and we can now used the learned merges.txt file to tokenize the text.
This time we can again split the text into characters and then we can look at
each pair of characters and see if it is in the merges.txt file. If it is we
replace it with the token that is on the right side of the rule in the merges.txt
