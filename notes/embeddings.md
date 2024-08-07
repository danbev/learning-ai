## Embeddings
When working with natural language models the input is text but neural networks
operate on matrices/tensors of numbers, not text. Embedding vectors are a way
to turn text into vectors with the elements of the vector being numbers so that
text can be used with neural networks.

How this text to vectors of numbers is done can differ, and one example is doing
[one-hot-encoding](./one-hot-encoding.md). Another option is using a count
based approach. And we also have the option to use embeddings which is what this
document will address.

It's a way of representing data like strings, music, video as points in an
n-dimension space. Doing this can allow similar data points to cluster together.

Word to vector (Word2Vec) was invented by Google in 2013 which takes as input
a word and outputs an n-dimensional coordinate, a vector. So, simliar words
would be closer to each other. Think of this as the tip of the vectors are in
close proximity to each other if the words are simliar.

For songs similar sounding songs would be nearby each other. And for images,
simliar looking images would be closer to each other. This could be anything
really so we don't have to think just in terms of words.

How is this useful?  
Well, we can look at vectors/points close to get similar words, songs, movies
etc. This is called nearest neighbor search.

Embedding also allows us to compute similarity scores between these points
allowing us to ask how similar is this song to another song. We can use the
Euclidean distance, the dot product, cosine distance, etc to calculate this
distance.

The learning stage is what positions these vectors close to each other.

We can use a one dimensional vector to represent each word, for example:
```
Why He She They Gras Tree
1   3   4   5    10   11

        She                  Tree
Why   He   they           Gras
|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|-
1  2  3  4  5  6  7  8  0 10 11 12 13 14 
```
This way we can calculate how similar to word are using:
```
Why and Gras:
10 - 1 = 9

He and she:
4 - 3 = 1
```
We can also define the encoding should have two values per vector making it
2 dimensional. So instead of each word being a single value it each one will
be a vector. We can use more dimensions as well and they can be.


### Universal sentence encoder
In this case entire sentences can be transformed into vectors which enables us
to do the same with sentences that we could with single words above.

### Tokens vs Embeddings
(This section was generated by ChatGPT-4)

Tokens are the basic units of text that a language model processes. In natural
language processing (NLP), tokenization is the process of converting input text
into a list of tokens. These tokens can be words, parts of words (like subwords
or syllables), or even characters, depending on the granularity of the model's
vocabulary. For example, the sentence "Hello, world!" might be tokenized into
["Hello", ",", "world", "!"].

Embeddings are dense vector representations of tokens. They are used to capture
the semantic meaning of tokens in a continuous vector space. Each token is
mapped to a vector of fixed size (the embedding size). These vectors are learned
during the training process of the language model. Embeddings are designed to
represent linguistic features of tokens so that tokens with similar meanings
have similar embeddings.

Here's how they differ and relate to each other:

Granularity: Tokens are discrete and correspond directly to elements of the
text. Embeddings are continuous and represent the abstract features of tokens.

Purpose: Tokens are used to chop up the text into processable pieces. Embeddings
are used to understand and process the semantic relationships between these pieces.

Representation: Tokens are often represented by integers or strings in a lookup
table. Embeddings are multi-dimensional floating-point vectors.

Usage in Models: Tokens are the input to the embedding layer of a language
model. The embedding layer then translates each token into its corresponding
embedding.

In the workflow of a language model like GPT, the process starts with raw text,
which is tokenized into tokens. These tokens are then passed through an
embedding layer to get their embeddings. The embeddings are what the model
actually processes to understand the text and generate responses or predictions.

### llama.cpp embeddings example
This example can be used as follows:
```console
$ gdb --args ./llama-embedding -m models/llama-2-7b-chat.Q4_K_M.gguf --no-warmup --pooling mean  -p "What is LoRA?
```
Now, recall that first the prompt is split into tokens, which each have an id
from the model vocabulary.
This example will set `params.embeddings = true`: 
```c++
    params.embedding = true;
```
First we tokenize the prompt like we mentioned above.
```c++
        auto inp = ::llama_tokenize(ctx, prompt, true, false);
```
```console
(gdb) p inp
$2 = std::vector of length 6, capacity 15 = {1, 1724, 338, 4309, 4717, 29973}
(gdb) p model.vocab.id_to_token[1]
$6 = {text = "<s>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) p model.vocab.id_to_token[1724]
$7 = {text = "▁What", score = -1465, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p model.vocab.id_to_token[338]
$8 = {text = "▁is", score = -79, attr = LLAMA_TOKEN_ATTR_NORMAL}
```
So we have tokens for the prompt we passed in.
For model in this example it has an embedding size of 4096 and we will create
a vector large enough to hold an embedding:
```c++
    std::vector<float> embeddings(n_prompts * n_embd, 0);
```
A this point all values in the dimensions are zero.

```c++
    float * emb = embeddings.data();

    // final batch
    float * out = emb + p * n_embd;
    batch_decode(ctx, batch, out, s, n_embd, params.embd_normalize);

```
The batch looks like this:
```console
(gdb) p batch
$38 = {n_tokens = 6, token = 0x555555b2e570, embd = 0x0, pos = 0x555555b30580, n_seq_id = 0x555555b32590, 
  seq_id = 0x555555b345a0, logits = 0x555555ed1510 "\001\001\001\001\001\001", all_pos_0 = 0, all_pos_1 = 0, 
  all_seq_id = 0}
```
We  will call `llama_decode` just like we would for a formal decoding:
```c++
    if (llama_decode(ctx, batch) < 0) {
        fprintf(stderr, "%s : failed to decode\n", __func__);
    }
```
In `llama_decode` we can find the following:
```c++
    // this indicates we are doing pooled embedding, so we ignore batch.logits and output all tokens
    const bool embd_pooled = cparams.embeddings && cparams.pooling_type != LLAMA_POOLING_TYPE_NONE;

        } else if (cparams.embeddings) {
            res = nullptr; // do not extract logits for embedding case
            embd = gf->nodes[gf->n_nodes - 1];
            if (strcmp(embd->name, "result_embd_pooled") != 0) {
                embd = gf->nodes[gf->n_nodes - 2];
            }
            GGML_ASSERT(strcmp(embd->name, "result_embd_pooled") == 0 && "missing embeddings tensor");
```
And the `embd` tensor will look like this:
```console
$56 = {type = GGML_TYPE_F32,
backend = GGML_BACKEND_TYPE_CPU,
buffer = 0x0,
ne = {4096, 6, 1, 1},
nb = {4, 16384, 98304, 98304},
op = GGML_OP_MUL_MAT, op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0, 
src = {0x7ffe8402fac0, 0x7ffe8402f7e0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0, 
view_offs = 0, data = 0x0,
name = "result_embd_pooled", '\000' <repeats 45 times>, extra = 0x0}
```
So we can see that this has 6 rows and 4096 columns.
A little later in `llama_decode_internal` we have:
```c++
        llama_set_inputs(lctx, u_batch);
```
In our case we are using mean pooling. And just to be clear on what this, it is
called pooling because we are reducing/compressing a sequence of token
embeddings into a single embedding. This can be done in many different ways. In
mean pooling we take the average of each embedding dimension across all tokens
in a sequence.
```c++
    if (cparams.embeddings && cparams.pooling_type == LLAMA_POOLING_TYPE_MEAN) {
        const int64_t n_tokens = batch.n_tokens;

        GGML_ASSERT(lctx.inp_mean);
        GGML_ASSERT(ggml_backend_buffer_is_host(lctx.inp_mean->buffer));

        float * data = (float *) lctx.inp_mean->data;
        memset(lctx.inp_mean->data, 0, n_tokens * n_tokens * ggml_element_size(lctx.inp_mean));

        std::vector<uint64_t> sum(n_tokens, 0);
        for (int i = 0; i < n_tokens; ++i) {
            const llama_seq_id seq_id = batch.seq_id[i][0];

            GGML_ASSERT(seq_id < n_tokens && "seq_id cannot be larger than n_tokens with pooling_type == MEAN");

            sum[seq_id] += 1;
        }

        std::vector<float> div(n_tokens, 0.0f);
        for (int i = 0; i < n_tokens; ++i) {
            const uint64_t s = sum[i];
            if (s > 0) {
                div[i] = 1.0f/float(s);
            }
        }

        for (int i = 0; i < n_tokens; ++i) {
            const llama_seq_id seq_id = batch.seq_id[i][0];
            data[seq_id*n_tokens + i] = div[seq_id];
        }
    }
```
The above is setting up the a tensor, `inp_mean` to hold the matrix operation
that performs mean calculation.This tensor is created by the function
`llm.append_pooling` which is called by `llama_build_graph`.
```c++
    // add on pooling layer
    if (lctx.cparams.embeddings) {
        result = llm.append_pooling(result);
    }
```
And in `append_pooling` we find:
```c++

        switch (pooling_type) {
            case LLAMA_POOLING_TYPE_MEAN:
                {
                    struct ggml_tensor * inp_mean = build_inp_mean();
                    cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, inp)), inp_mean);
                } break;
```
And the `build_inp_mean` function looks like this:
```c++
    struct ggml_tensor * build_inp_mean() {
        lctx.inp_mean = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, n_tokens);
        cb(lctx.inp_mean, "inp_mean", -1);
        ggml_set_input(lctx.inp_mean);
        return lctx.inp_mean;
    }
```
So in this case we will have a 6x6 matrix to hold the mean values which indeed
is the case:
```console
(gdb) p *lctx.inp_mean 
$62 = {type = GGML_TYPE_F32,
backend = GGML_BACKEND_TYPE_CPU, buffer = 0x555555aab7c0,
ne = {6, 6, 1, 1}, 
nb = {4, 24, 144, 144},
op = GGML_OP_NONE, op_params = {0 <repeats 16 times>}, flags = 1, grad = 0x0, src = {
0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0, view_offs = 0, data = 0x7ffe3a230020, 
name = "inp_mean", '\000' <repeats 55 times>, extra = 0x0}
```

If we look back as how this tensor is populated we can see that is first figures
out how many tokens each sequence has. Potentially each token could belong to as
different sequence to at most 6 sums are stored in the `sum` vector.
```c++
        std::vector<uint64_t> sum(n_tokens, 0);
        for (int i = 0; i < n_tokens; ++i) {
            const llama_seq_id seq_id = batch.seq_id[i][0];

            GGML_ASSERT(seq_id < n_tokens && "seq_id cannot be larger than n_tokens with pooling_type == MEAN");

            sum[seq_id] += 1;
        }
```
```console
(gdb) p sum
$65 = std::vector of length 6, capacity 6 = {6, 0, 0, 0, 0, 0}
```
And then we will calculate the divisors for each sequence:
```c++
        std::vector<float> div(n_tokens, 0.0f);
        for (int i = 0; i < n_tokens; ++i) {
            const uint64_t s = sum[i];
            if (s > 0) {
                div[i] = 1.0f/float(s);
            }
        }
```
```console
(gdb) p div
$69 = std::vector of length 6, capacity 6 = {0.166666672, 0, 0, 0, 0, 0}
```
```c++
        float * data = (float *) lctx.inp_mean->data;

        for (int i = 0; i < n_tokens; ++i) {
            const llama_seq_id seq_id = batch.seq_id[i][0];
            data[seq_id*n_tokens + i] = div[seq_id];
        }
```
```console
(gdb) p data[0]
$78 = 0.166666672
(gdb) p data[1]
$79 = 0.166666672
(gdb) p data[2]
$80 = 0.166666672
(gdb) p data[3]
$81 = 0.166666672
(gdb) p data[4]
$82 = 0.166666672
(gdb) p data[5]
$83 = 0.166666672
(gdb) p data[6]
$84 = 0
```
Notice that only the first 6 elements are set to 0.166666672. The rest are zero.
and this is a 6x6 matrix so we have 36 elements in total.
And this matrix is then used in a matrix multiplation:

```c++
        switch (pooling_type) {
            case LLAMA_POOLING_TYPE_MEAN:
                {
                    struct ggml_tensor * inp_mean = build_inp_mean();
                    cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, inp)), inp_mean);
                } break;
```

In `llama_decode_internal` we also have:
```c++
                case LLAMA_POOLING_TYPE_MEAN:
                case LLAMA_POOLING_TYPE_CLS:
                case LLAMA_POOLING_TYPE_LAST:
                    {
                        // extract sequence embeddings
                        auto & embd_seq_out = lctx.embd_seq;
                        embd_seq_out.clear();

                        for (uint32_t i = 0; i < n_tokens; i++) {
                            const llama_seq_id seq_id = u_batch.seq_id[i][0];
                            if (embd_seq_out.find(seq_id) != embd_seq_out.end()) {
                                continue;
                            }
                            embd_seq_out[seq_id].resize(n_embd);
                            ggml_backend_tensor_get_async(backend_embd,
                                embd,
                                embd_seq_out[seq_id].data(),
                                (n_embd*seq_id)*sizeof(float), n_embd*sizeof(float));
                        }
                    } break;
```

And then in the for loop for embeddings (in embeddings.cpp):
```c++
    for (int i = 0; i < batch.n_tokens; i++) {
        if (!batch.logits[i]) {
            continue;
        }

        // try to get sequence embeddings - supported only when pooling_type is not NONE
        const float * embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
        GGML_ASSERT(embd != NULL && "failed to get sequence embeddings");

        float * out = output + batch.seq_id[i][0] * n_embd;
        llama_embd_normalize(embd, out, n_embd, embd_norm);
    }
```
We retrieve the embeddings for the sequence id.


In common.h we have the parameter for embedding normalization:
```c++
    int32_t embd_normalize = 2;     // normalisation for embendings (-1=none, 0=max absolute int16, 1=taxicab, 2=euclidean, >2=p-norm)
```
But the taxicab normalization is not implementes or perhaps it has been removed:
```c++
void llama_embd_normalize(const float * inp, float * out, int n, int embd_norm) {
    double sum = 0.0;

    switch (embd_norm) {
        case -1: // no normalisation
            sum = 1.0;
            break;
        case 0: // max absolute
            for (int i = 0; i < n; i++) {
                if (sum < std::abs(inp[i])) sum = std::abs(inp[i]);
            }
            sum /= 32760.0; // make an int16 range
            break;
        case 2: // euclidean
            for (int i = 0; i < n; i++) {
                sum += inp[i] * inp[i];
            }
            sum = std::sqrt(sum);
            break;
        default: // p-norm (euclidean is p-norm p=2)
            for (int i = 0; i < n; i++) {
                sum += std::pow(std::abs(inp[i]), embd_norm);
            }
            sum = std::pow(sum, 1.0 / embd_norm);
            break;
    }

    const float norm = sum > 0.0 ? 1.0 / sum : 0.0f;

    for (int i = 0; i < n; i++) {
        out[i] = inp[i] * norm;
    }
}
```
