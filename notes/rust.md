## Rust crates for AI/ML
This document lists some of the more popular Rust crates for AI/ML. This will
probably not be a complete list as there are many crates for AI/ML in Rust and
more being developed.

There is [arewelearningyet](https://www.arewelearningyet.com/) which seems to
be a good resource for AI/ML in Rust.

The examples provides here are mostly just the same examples provides by the
upstream project but were still useful as not all examples worked out of the
box.

### Candle (Huggingface)
[Candle](https://github.com/huggingface/candle) is a project from Huggingface.
The goal is to make the deploying lightweight binaris possible and not have the
overhead of Python in production.

[Basic example](../hugging-face/rust/candle-example)

I think that Candle sounds interesting and having WASM support is a big plus
enabling running in any environment which has WASM support.

### llm
[llm](https://github.com/rustformers/llm) is a collection of Rust libraries for
Large Language Models (LLMs). The current recommendation is to use install from
source, that is using a github repository as a dependency.

[llm example](../fundamentals/rust/llm-example/README.md    )


### llm-chains
This seems to similar to what langhchain provides for Python.
[llm-chains](https://github.com/sobelio/llm-chain)


### Tokenizers (Huggingface)
[Tokenizers](https://github.com/huggingface/tokenizers) is a project also from
Huggingface that has implementations for the most common tokenizers.

### Rust-Bert
[rust-bert](https://github.com/guillaume-be/rust-bert) is a Rust implementation
of the popular [Huggingface](https://huggingface.co/)
[Transformers](https://github.com/huggingface/transformers) for Natural Language
Processing (NLP).

### rust-tokenizers
[rust-tokenizers](https://github.com/guillaume-be/rust-tokenizers) is also has
tokenizers implemented in Rust. This library is used by rust-bert.

[Basic Example](../embeddings/rust/rust-tokenizers-example).

### smartcore
[smartcore](https://smartcorelib.org/) is a machine learning library written in
Rust. 

[Basic Example](../fundamentals/rust/smartcore-example).


### safetensors
[safetensors](https://github.com/huggingface/safetensors/tree/main/safetensors)
is a project from Huggingface which provides a new format for storing tensors.
So after a model has been trained, the result will be weights and bias tensors
which the neural network will use to infer input data that it has not been seen
before (the whole point of training a model). These tensors can be quite large
and also need to be made available to the inference engine. There are currently
different [formats](https://github.com/huggingface/safetensors/tree/main/safetensors#yet-another-format-)
for storing these tensors, but they are not very efficient. 
The linked pages state that 
```
"This repository implements a new simple format for storing tensors safely
(as opposed to pickle) and that is still fast (zero-copy)."
```
Why is pickle not safe?  
(ChatGPT answer)  
The Python `pickle` module is used for serializing and deserializing Python
object structures. However, it's generally not considered safe for several
reasons, particularly when loading pickle data from an untrusted source.
Here's why:

#### Security Risks

1. **Arbitrary Code Execution**: The most critical issue is that pickled data 
can be crafted to execute arbitrary code during unpickling. An attacker can 
exploit this to run harmful code on the system that deserializes the object.

2. **Denial of Service**: Even if the pickled data doesn't execute arbitrary
code, it can be crafted in a way to consume excessive amounts of memory or CPU,
causing a Denial of Service (DoS) attack.

3. **Information Disclosure**: Maliciously crafted pickle data can also be
used to probe the internal data structures of the program that's doing the
unpickling, potentially revealing information that may be sensitive.

#### Version Incompatibility

1. **Non-Forward Compatible**: Pickle data may not be forward-compatible,
meaning data pickled with one version of Python may not be unpicklable with a
future version. While this is not a security issue, it can be a concern for
data integrity and longevity.

2. **Library Dependencies**: If you're pickling objects that are instances of
custom classes, unpickling them will require access to the same class
definitions. Any changes in the class definition between pickling and unpickling
can cause issues.
