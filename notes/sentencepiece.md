## SentencePiece
This is tokenizer/detokenizer used a by a number of NLP libraries. It is used
by gglm for example.


### Detokenizing
This was something that I had not considered but not all tokenizers are
reversible. For example they might split up tokens so that is not possible
know how to put them back together again. For example:
```
"Hello world" -> "Hello" "world"
```

