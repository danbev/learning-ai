## Extended Batch API
This covers the following PR: https://github.com/ggml-org/llama.cpp/pull/24669,
which adds a new extended batch API to llama.cpp.

In the current/old llama_batch struct we had raw pointers and we could have
tokens ids or embeddings. So this was either, not both.

For certain things like M-RoPE where an image patch token has 4 position component
(hight, width, time, text), but the old pos array as one llama_pos per token.

Also speculative decoding with MTP needs both a token id and a hidden state at
the same slot. So when the MTP process() runs it need to pass both the token
id and the hidden state (as the embedding which is also not currently possible.

The above PR introduces a struct named token which will be what a slot in the
new extended batch will hold:
```c++
#define GGML_MROPE_SECTIONS   4

struct llama_batch_ext {
    ...
    // actual embd row width of this batch, set by the first set_token_embd()
    // must be either n_embd_inp or n_embd_inp_enc; encode/decode verify it against the graph input
    size_t n_embd = 0;

    struct token {
        llama_token  id = LLAMA_TOKEN_NULL;
        bool         has_embd = false; // whether embd_off is set
        size_t       embd_off = 0; // index offset in the embd array
        bool         output = false; // TODO: have dedicated output flags
        std::unordered_set<llama_seq_id> seq_ids;
        std::array<llama_pos, GGML_MROPE_SECTIONS> pos = {0, 0, 0, 0};
    };
    std::vector<token> tokens;
    std::vector<float> embd;
    ...
};
```
Notice that we have a token id, a bool that tell us if this token has an embedding
(for example the MTP case mentioned above). 
Also notice that pos is now and array of llama_pos which holds 4 elements so
we can accomodate M-RoPE.

The emd_off is used because the embd vector is for the whole batch which might
contain multiple tokens.
```c++
struct llama_embd {
    const float * data;
    size_t n_rows; // number of embedding rows in data
    size_t n_embd; // size of one row
};
```
And an example of an embedding could be something as simple as the token
embeddings looked up for a given number of token ids, or it could be the hidden
state of of a target models layers that are then needed for a draft model in
speculative drafting. So we could have n_rows (think of tokens in a sequence)
each having their embedding of a specific length. The n_embd (the size/dimension)
allows for (perhaps later) having differnet embedding sizes.

The llama_batch_ext struct also has the following fields:
```c++
    const size_t n_embd_inp;       // decoder embd row width
    const size_t n_embd_inp_enc;   // encoder embd row width (e.g. eagle3/dflash extracted features)
    ...
    size_t n_embd = 0;
```
n_embd_inp is a set using a hparam and is the dimension that the models graph
input expects. This is normally just n_embd, the models hidden size but for
Qwen-3-VL deepstack it is n_embd + (n_embd * n_deepstack_layers.

n_embd_inp_enc is also set using a hparam is what the encoder-side input expects
which is used for Eagle3/DFlash and is the size of the features extracted from
the selected layers of the target model and concatenated (so it has a different
width than normal embeddings.

When an embedding is added to an extended batch using:
```c++
bool llama_batch_ext::set_token_embd(int32_t idx, llama_embd embd_in) {
    if (idx < 0 || idx >= (int32_t) tokens.size()) {
        return false;
    }
    if (!embd_in.data) {
        return false;
    }

    const size_t n_total = embd_in.n_rows * embd_in.n_embd;
    if (n_embd == 0) {
        // if the total size of the embedding is not the same as the expected
        // models input size, or the encoders input size then return false.
        if (n_total != n_embd_inp && n_total != n_embd_inp_enc) {
            LLAMA_LOG_ERROR("%s: embedding size mismatch, got %zu rows x %zu = %zu, expected %zu or %zu\n",
                    __func__, embd_in.n_rows, embd_in.n_embd, n_total, n_embd_inp, n_embd_inp_enc);
            return false;
        }
        n_embd = n_total;
    } else if (n_total != n_embd) {
        LLAMA_LOG_ERROR("%s: embedding size mismatch, got %zu rows x %zu = %zu, expected %zu\n",
                __func__, embd_in.n_rows, embd_in.n_embd, n_total, n_embd);
        return false;
    }

    token & t = tokens[idx];

    if (t.has_embd) {
        LLAMA_LOG_ERROR("%s: embedding for token %d is already set\n", __func__, idx);
        return false;
    }

    t.has_embd = true;
    t.embd_off = embd.size();
    embd.insert(embd.end(), embd_in.data, embd_in.data + n_total);

    return true;
}
```

```c++
llama_batch_ext b(/*n_tokens_max*/ 64,
                  /*n_embd_inp*/ 2,
                  /*n_embd_inp_enc*/ 2,
                  /*n_seq_max*/ 4,
                  /*mem*/ nullptr,
                  /*n_vocab*/ 0,
                  /*n_pos_per_embd*/ 1);

// add a token (which could be a token id or a embedding, for both.
int32_t idx = b.add_token(/*seq_id*/ 0);   // idx = 0

// add embeding float data  (n_rows=1, n_embd=2)
float data[] = { 0.0f, 1.0f };
b.set_token_embd(idx, { data, /*n_rows*/ 1, /*n_embd*/ 2 });
//  -> batch.n_embd is locked to 1*2 = 2
//  -> token[0].has_embd = true, token[0].embd_off = 0
//  -> batch.embd = [0.0, 1.0]

llama_pos pos = 0;
b.set_token_pos(idx, &pos);
b.set_output(idx, true);
```
token[0].id stays LLAMA_TOKEN_NULL — this slot has an embedding but no token ID.

A new function has been added to the public API named llama_process:
```c++
    LLAMA_API int32_t llama_process(
                                struct llama_context * ctx,
                             enum llama_process_type   type,
                              struct llama_batch_ext * batch);

    enum llama_process_type {
        LLAMA_PROCESS_TYPE_ENCODE,
        LLAMA_PROCESS_TYPE_DECODE,
    };

```
```c++
int32_t llama_decode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->decode(batch);
    if (ret != 0 && ret != 1) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }

    return ret;
}

int llama_context::decode(const llama_batch & batch_inp) {
    llama_batch_compat compat(this, batch_inp);
    return decode(*compat.batch_ext);
}
```
And llama_context::decode has been updated to take the extended batch:
```c++
int llama_context::decode(const llama_batch_ext & batch_inp) {
    ...

    if (!balloc->init(batch_inp, vocab, output_all)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }
```
```c++
bool llama_batch_allocr::init(
        const llama_batch_ext & batch_inp,
        const llama_vocab & vocab,
        bool output_all) {
```
