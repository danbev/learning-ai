## Continuous Batching
When we perform inference we have a prompt, sometimes called a prefill, which
is first decoded:
```console
prompt = "What is the Capital of Sweden?"
```
This will be tokenized and the tokens will be added to a batch for processing,
and the result will be a probability distribution over all the tokens in the
models vocabulary.

Now, during inference the multihead attention is done using a large matrix
multiplication. We can think of this as each row being the token embeddings
for a single token in the prompt. The inference is usually done on a GPU which
means that the model weights have to be loaded into the GPU memory to be able
to perform this multiplication. This is a relatively slow operation and we want
to avoid doing this for every prompt or token that we want to infer.

What we can do is that we can try to fill up the GPU memory with as many tokens
as possible, which can belong to different sequences (different chat sessions
is one way I think we can think of this). 

Now, for the prompt we can process all the tokens in the prompt in one go, but
for output generation the answer is generated one token at a time.  This means
that the generated token is appended to the previous prompt and then we run
the inference again, and so on and so forth. This process is repeated until we
meet a stopping criteria, like a maximum number of tokens generated or a
special token like End Of Sentence (EOS).

We have discussed the KV-cache in other documents but just recall that for the
initial prompt all the values have to be computed, but after than values in the
KV-cache will be reused to avoid having to compute the dot product for tokens
that we have already processed/seen. This is why the first token generated takes
longer and this is why this is often reported as the time to first token as it
can be of interest to know how long it takes to generate the first token.

So lets say I have the following prompt: "What is the capital of Sweden", this
would be split and tokenized and token embeddings looked up in the model to get
the embeddings for each token. This is what is passed to the model. Now when I
say passed to the model what is really happening is that this data is added to a
buffer on the CPU and copied to the GPU using some library (like CUDA for
example).
Then there will be a program that triggers operations on the GPU to perform the
math operations on data that was copied to the GPU memory. For example going
through the attention layers, then the softmax (a bit simplified but bare with
me). The output is then copied back from the GPU to a CPU buffer. The values in
this matrix(?) are the logits (raw scores for each token in the vocabulary)
which can then be used with whatever sampling technique chosen. The token
sampled is then looked up in the models vocabulary and perhaps sent to the users
to be displayed, and then that token will be passed back into the decoder to
generate the next token. 

Now, the KV-cache is stored on the GPU and is populated as tokens are processed.
For each new token only the token embeddings for the next token in the sequence
needs to be copied from the CPU to the GPU.
And also note that the model weights stay loaded on the GPU and are not copied
again. So we could serve multiple multiple users if we can separate there tokens
from each other and keep the GPU busy. This can be done using a sequence id and
each user can have their own sequence id. This sequence id also need to be used
in the KV-cache as otherwise it would be possible for the attention to mix
tokens from different users.

And so what is the batch size?  
Like in my made up example above during token generation we would only be
passing in one token embedding, so we would have a batch size of 1 in that case.
Does this mean that we are only copying a single vector for the token embedding
to from the CPU to the GPU. But we also have the option to pass in more tokens, 
another row in a matrix and copy those together in one go to the GPU. I'm trying
to move away from saying passed to the model and break this down into what is
actually happening, we are placing the embeddings in a buffer and then
instructing the GPU layers software to trigger a copy from that CPU memory
buffer to the GPU's memory.

We mentioned earlier that the model weights are loaded from the CPU to the GPU
and they stay in place. A 7B parameter model using F16 precision will take up
about 14GB of GPU memory.

We also have the KV-Cache that is stored in GPU memory and this grows as the
number of tokens processed grows. 
```console
kv_cache_memory = (
    batch_size        # number of sequences
    × seq_length      # number of tokens per sequence
    × 2               # K and V matrices
    × hidden_dim      # size of each K/V vector
    × num_layers      # because each layer needs its own cache
    × bytes_per_value # for example 2 for FP16
)
```
So the longer the sequence the more memory. The number of layers also effects
the size of the cache as each layer needs its own cache. The number of tokens

To clarify something from above when I said that for a single token we would
only need to copy a single vector to the GPU, this is not true in reality. The
way this works is that the GPU is optimized to process data in chunks and we
copy the token embeddings into the CPU buffer that is to be copied to the GPU
in a specific way/format:
```
  0                             768
0 [seq_1_token1 embedding vector]
.
.
.
31 [seq_32_token1 embedding vector]
```
And this is what is referred to as a batch, and the batch size in this case
is 32.
Now, for the case where we only have a single sequence we might fill up the
complete batch 
```
  0                             768
0 [seq_1_token1 embedding vector]
1 [0, 0, ...                  ,0]
.
.
.
31[0, 0, ...                  ,0]
```
And so the reason why we want to fill up batches as much as possible is that we
are going to run through the same computation regardless if there is one token
of the entire batch is filled. So we want to fill it with as much useful
information as possible to not waste the computation. Its like a train on a
schedule, it will cost the same amount of enery and time to run regardless if
there is one passenger or if the train is full (disregarding the extra weight for
passengers perhaps).

This is why inference systems collect incoming requests into groups and then
fill batches completely.

So, when we create a llama_batch for decoding we might have two sequences with
different length prompts in each sequence. Now, before the inference engine can
return the result, both sequences have to have been completed. If one is much
longer that the other then the shorter one will have to wait for the longer one
to finish.

