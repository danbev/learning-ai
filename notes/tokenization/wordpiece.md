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

_wip_
