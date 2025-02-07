### Token probs mismatch

Issue: https://github.com/ggerganov/llama.cpp/issues/11728

To reproduce this I've used a different model than reported but I don't think
that matters in this case. I've used a different prompt to force a leading white
space to be produced in the output.

```console
$ ./build/bin/llama-server -m models/llama-2-7b.Q4_K_M.gguf -n 99
```

```console
$ curl -fsS \
    --url http://127.0.0.1:8080/completion \
    --header "Content-Type: application/json" \
    --data '{"prompt": "What to the capital of Sweden? Answer with a whitespace.","n_predict": 1, "n_probs": 10, "temperature":0}'

{
  "index": 0,
  "content": " Hinweis",
  "tokens": [],
  "id_slot": 0,
  "stop": true,
  "model": "gpt-3.5-turbo",
  "tokens_predicted": 1,
  "tokens_evaluated": 13,
  "generation_settings": {
    "n_predict": 1,
    "seed": 4294967295,
    "temperature": 0.0,
    "dynatemp_range": 0.0,
    "dynatemp_exponent": 1.0,
    "top_k": 40,
    "top_p": 0.949999988079071,
    "min_p": 0.05000000074505806,
    "xtc_probability": 0.0,
    "xtc_threshold": 0.10000000149011612,
    "typical_p": 1.0,
    "repeat_last_n": 64,
    "repeat_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "dry_multiplier": 0.0,
    "dry_base": 1.75,
    "dry_allowed_length": 2,
    "dry_penalty_last_n": 4096,
    "dry_sequence_breakers": [
      "\n",
      ":",
      "\"",
      "*"
    ],
    "mirostat": 0,
    "mirostat_tau": 5.0,
    "mirostat_eta": 0.10000000149011612,
    "stop": [],
    "max_tokens": 1,
    "n_keep": 0,
    "n_discard": 0,
    "ignore_eos": false,
    "stream": false,
    "logit_bias": [],
    "n_probs": 10,
    "min_keep": 0,
    "grammar": "",
    "grammar_trigger_words": [],
    "grammar_trigger_tokens": [],
    "preserved_tokens": [],
    "samplers": [
      "penalties",
      "dry",
      "top_k",
      "typ_p",
      "top_p",
      "min_p",
      "xtc",
      "temperature"
    ],
    "speculative.n_max": 16,
    "speculative.n_min": 5,
    "speculative.p_min": 0.8999999761581421,
    "timings_per_token": false,
    "post_sampling_probs": false,
    "lora": []
  },
  "prompt": "<s> What to the capital of Sweden? Answer with a whitespace.",
  "has_new_line": false,
  "truncated": false,
  "stop_type": "limit",
  "stopping_word": "",
  "tokens_cached": 13,
  "timings": {
    "prompt_n": 13,
    "prompt_ms": 300.789,
    "prompt_per_token_ms": 23.137615384615383,
    "prompt_per_second": 43.21966561277175,
    "predicted_n": 1,
    "predicted_ms": 3.726,
    "predicted_per_token_ms": 3.726,
    "predicted_per_second": 268.3843263553409
  },
  "completion_probabilities": [
    {
      "id": 18627,
      "token": " Hinweis",
      "bytes": [
        32,
        72,
        105,
        110,
        119,
        101,
        105,
        115
      ],
      "logprob": -4.0167436599731445,
      "top_logprobs": [
        {
          "id": 18627,
          "token": "Hinweis",
          "bytes": [
            72,
            105,
            110,
            119,
            101,
            105,
            115
          ],
          "logprob": -4.0167436599731445
        },
        ...
```
Notice that tthat the `completion_probabilities` has a leading white space in the `token` field,
but also note that the token id is `18627`.
Then in the `top_logprobs` we have the same token but without the leading white space, and the
same token id `18627`.

If we inspect this token id in the models vocabulary we can see:
```console
(lldb) p ctx->model.vocab.pimpl->id_to_token[18627]
(std::vector<llama_vocab::token_data>::value_type) 
(text = "▁Hinweis", score = -18368, attr = LLAMA_TOKEN_ATTR_NORMAL)
```
Notice that the `__` is the Lower One Eighth Block Unicode character which is used by thi
model to mark word boundries. So actually if we detokenize this token we should have a
leading white space.

And we can inspect the probabilities:
```console
(lldb) p probs
(const std::vector<completion_token_output> &) size=1: {
  [0] = {
    tok = 18627
    prob = 0.0180115215
    text_to_send = " Hinweis"
    probs = size=10 {
      [0] = (tok = 18627, txt = "Hinweis", prob = 0.0180115215)
      [1] = (tok = 19838, txt = "Unterscheidung", prob = 0.0169409495)
      [2] = (tok = 23196, txt = "nobody", prob = 0.0140749179)
      [3] = (tok = 26077, txt = "everybody", prob = 0.0128059033)
      [4] = (tok = 24366, txt = "sierp", prob = 0.0121691627)
      [5] = (tok = 6610, txt = "Einzeln", prob = 0.0102084409)
      [6] = (tok = 25145, txt = "kwiet", prob = 0.00851798616)
      [7] = (tok = 23996, txt = "живело", prob = 0.00680034328)
      [8] = (tok = 27581, txt = "hopefully", prob = 0.00639356672)
      [9] = (tok = 23795, txt = "paździer", prob = 0.00621746061)
    }
  }
}
```
The first value that we are seeing in the server client output above is the
value of `text_to_send`. And we can also see that the value in the props array
does not match the `text_to_send` value.


```c++
    void update_slots() {
        ...

                completion_token_output result;
                result.tok          = id;
                result.text_to_send = common_token_to_piece(ctx, result.tok, accept_special_token(slot, result.tok));
                result.prob         = 1.0f; // TODO: set it here instead of doing inside populate_token_probs

                if (slot.params.sampling.n_probs > 0) {
                    populate_token_probs(slot, result, slot.params.post_sampling_probs, params_base.special, tok_idx);
                }
```

```console
(lldb) p result.text_to_send
(std::string) " Hinweis"

(lldb) p params_base.special
(bool) false
```

In `populate_token_probs` we have the following:
```
    void populate_token_probs(const server_slot & slot, completion_token_output & result, bool post_sampling, bool special, int idx) {
        ...

            // TODO: optimize this with min-p optimization
            std::vector<llama_token_data> cur = get_token_probabilities(ctx, idx);

            // set probability for sampled token
            for (size_t i = 0; i < n_vocab; i++) {
                // set probability for sampled token
                if (cur[i].id == result.tok) {
                    result.prob = cur[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < std::min(n_vocab, n_probs); i++) {
                result.probs.push_back({
                    cur[i].id,
                    common_detokenize(ctx, {cur[i].id}, special),
                    cur[i].p
                });
            }
        }
```
So `cur` are the propabilities for the token ids in the vocabulary. And we can print out
the first 7 values:
```console
(lldb) p cur
(std::vector<llama_token_data>) size=32000 {
  [0] = (id = 18627, logit = 8.03423404, p = 0.0180115215)
  [1] = (id = 19838, logit = 7.97295618, p = 0.0169409495)
  [2] = (id = 23196, logit = 7.78761673, p = 0.0140749179)
  [3] = (id = 26077, logit = 7.69312858, p = 0.0128059033)
  [4] = (id = 24366, logit = 7.64212751, p = 0.0121691627)
  [5] = (id = 6610, logit = 7.46643733, p = 0.0102084409)
  [6] = (id = 25145, logit = 7.2854023, p = 0.00851798616)
```
The first value will be passed to `common_detokenize` to get the text representation of the token.
The detokenized token will be the following:
```console
(lldb) p text
(std::string) "Hinweis"
```
This is where the mismatch is happening I think. I'm looking at the response the `text_to_send` is
`" Hinweis"` but the token id is `18627` and the detokenized token is `"Hinweis"`.
I believe that if the detokenization will add the leading white space depends on the `add_space_prefix`
field on the `llama_vocab` object is set to true:
```console
(lldb) p this->add_space_prefix
(bool) true
```
```c++
int32_t llama_vocab::impl::detokenize(
               const llama_token * tokens,
                         int32_t   n_tokens,
                            char * text,
                         int32_t   text_len_max,
                            bool   remove_special,
                            bool   unparse_special) const {
                                ...
    // remove the leading space
    bool remove_space = add_space_prefix;

    if (remove_special && add_bos) {
        if (n_tokens > 0 && tokens[0] == special_bos_id) {
            remove_space = false;
            n_tokens--;
            tokens++;
        }
    }

    if (remove_special && add_eos) {
        if (n_tokens > 0 && tokens[n_tokens - 1] == special_eos_id) {
            n_tokens--;
        }
    }

    for (int32_t i = 0; i < n_tokens; ++i) {
        GGML_ASSERT(avail >= 0);
        int32_t n_chars = token_to_piece(tokens[i], text, avail, remove_space, unparse_special);
        remove_space = false;
        ...
    }
```
So I'm not sure how this should be handled in a "proper" way. It seems like the detokenization is
working as expected but if we want/need a match of the `text_to_send` and the highest probability
then perhaps adding the check above might be a solution.
