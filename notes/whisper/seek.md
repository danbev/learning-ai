## whisper.cpp seek
This document is about describing how seek is used in the function 
`whisper_full_with_state`.

The first thing that happens in this function is that the log mel spectrogram
is computed from the audio samples:
```c++
int whisper_full_with_state(
        struct whisper_context * ctx,
          struct whisper_state * state,
    struct whisper_full_params   params,
                   const float * samples,
                           int   n_samples) {
    // clear old results
    auto & result_all = state->result_all;

    result_all.clear();

    if (n_samples > 0) {
        // compute log mel spectrogram
        if (whisper_pcm_to_mel_with_state(ctx, state, samples, n_samples, params.n_threads) != 0) {
            WHISPER_LOG_ERROR("%s: failed to compute log mel spectrogram\n", __func__);
            return -2;
        }
    }
```

Then we have the following:
```c++
    ...
    if (params.token_timestamps) {
        state->t_beg    = 0;
        state->t_last   = 0;
        state->tid_last = 0;
        if (n_samples > 0) {
            state->energy = get_signal_energy(samples, n_samples, 32);
        }
    }
```
TODO: link to document on `get_signal_energy`.

Following that then have:
```c++
    const int seek_start = params.offset_ms/10;
```
```console
(gdb) p params.offset_ms
$4 = 20000
(gdb) p seek_start
$7 = 2000
```
So the above is converting the `offset_ms` parameter to seconds.

Next `seek_end` is set using `params.duration_ms` if it is set.
```c++
    const int seek_end = params.duration_ms == 0 ? whisper_n_len_from_state(state) : seek_start + params.duration_ms/10;
```
```console
(gdb) p whisper_n_len_from_state(state)
$9 = 3003
(gdb) p state->mel.n_len_org
$10 = 3003
```

And a bit further down we have:
```c++
    int seek = seek_start;
```

And following that we will call then encoder and pass in `seek`:
```c++
        // encode audio features starting at offset seek
        if (!whisper_encode_internal(*ctx, *state, seek, params.n_threads, params.abort_callback, params.abort_callback_user_data)) {
            WHISPER_LOG_ERROR("%s: failed to encode\n", __func__);
            return -6;
        }
```

And a little further down still have have the decode call:
```c++
                if (!whisper_decode_internal(*ctx, *state, state->batch, params.n_threads, false, params.abort_callback, params.abort_callback_user_data)) {
                    WHISPER_LOG_ERROR("%s: failed to decode\n", __func__);
                    return -8;
                }
```

The first token will be:
```c++
                auto t0 = seek + 2*(tokens_cur.front().tid - whisper_token_beg(ctx));
```
```console
(gdb) p ctx->vocab->id_to_token[tokens_cur[0].id]
$52 = "[_BEG_]"
(gdb) p ctx->vocab->id_to_token[tokens_cur[1].id]
$53 = " There"
(gdb) p ctx->vocab->id_to_token[tokens_cur[2].id]
$54 = " once"
(gdb) p ctx->vocab->id_to_token[tokens_cur[3].id]
$55 = " lived"
(gdb) p ctx->vocab->id_to_token[tokens_cur[4].id]
$56 = " a"
(gdb) p ctx->vocab->id_to_token[tokens_cur[5].id]
$57 = " poor"

(gdb) p ctx->vocab->id_to_token[tokens_cur[0].tid]
$60 = "[_BEG_]"
(gdb) p ctx->vocab->id_to_token[tokens_cur[1].tid]
$61 = "[_BEG_]"
(gdb) p ctx->vocab->id_to_token[tokens_cur[2].tid]
$62 = "[_TT_1056]"
(gdb) p ctx->vocab->id_to_token[tokens_cur[3].tid]
$63 = "[_TT_1056]"
(gdb) p ctx->vocab->id_to_token[tokens_cur[4].tid]
$64 = "[_TT_1076]"
(gdb) p ctx->vocab->id_to_token[tokens_cur[5].tid]
$65 = "[_TT_1080]"
```
So if I'm reading this correctly the timestamps ids are specifying that " once"
is at time 1056 (ms?)

```console
 40       "text": " There once lived a poor tailor, who had a son called Aladdin, a careless idle-boy, who",
 41       "tokens": [
 42         {
 43           "text": "[_BEG_]",
 44           "timestamps": {
 45             "from": "00:00:00,000",
 46             "to": "00:00:00,000"
 47           },
 48           "offsets": {
 49             "from": 0,
 50             "to": 0
 51           },
 52           "id": 50363,
 53           "p": 0.755106,
 54           "t_dtw": -1
 55         },
 56         {
 57           "text": " There",
 58           "timestamps": {
 59             "from": "00:00:01,140",
 60             "to": "00:00:21,120"
 61           },
 62           "offsets": {
 63             "from": 1140,
 64             "to": 21120
 65           },
 66           "id": 1318,
 67           "p": 0.338112,
 68           "t_dtw": -1
 69         },
 70         {
 71           "text": " once",
 72           "timestamps": {
 73             "from": "00:00:21,120",
 74             "to": "00:00:21,290"
 75           },
 76           "offsets": {
 77             "from": 21120,
 78             "to": 21290
 ```
 Notice that the segment timestamps are incorrect, they should start at something
 like 20 seconds. And also the first token tiemstamps is incorrect, at least
 its start/from time is incorrect, it should be 20 seconds as well. But the to/end
 are correct.

Hmm, this does not look correct:
```console
(gdb) p tokens_cur.front().tid
$2 = 50363
(gdb) p tokens_cur[0].tid
$3 = 50363
(gdb) p tokens_cur[1].tid
$4 = 50363
(gdb) p tokens_cur[1].id
$5 = 1318
(gdb) p ctx->vocab->id_to_token[tokens_cur[1].id]
$6 = " There"
(gdb) p ctx->vocab->id_to_token[tokens_cur[0].id]
$7 = "[_BEG_]"
```
Notice here that the first token is `[_BEG_]` and this has the same token id.
The second token is " There" but this also has the same token id as the first
token.


In our case this is set to 0 so there won't be any offset applied in this case.
But we can set the timestamps offset using the `-ot/--offset-t` parameter:
```console
$ ./build/bin/whisper-cli -f samples/aladdin-first30.mp3 \
    -m ./models/ggml-medium.en.bin \
    --vad-model ./models/for-tests-silero-v5.1.2-ggml.bin \
    -ojf \
    -of jfk \
    -ot 20000
```

`seek` is then used to calculate the start timestamp:
```c++
                auto t0 = seek + 2*(tokens_cur.front().tid - whisper_token_beg(ctx));
```
And then later to set the timestamp on the segment:
```c++
                            result_all.push_back({ tt0, tt1, text, state->no_speech_prob, {}, speaker_turn_next });
```
And like we looked at before this is what
`whisper_exp_compute_token_level_timestamps` will use to compute the token
level timestamps.


### decoding
In `whisper_full_with_state` the flow is something like this:
* Input audio samples are converted into log mel spectrogram.
* The spectrogram is encoded and made availble to the decoders cross attentio
* whisper_decode is called.
* process the logits (this is where -INF filtering is applied) also calls `whisper_compute_probs`.
* Each decoder gets a copy of the logits, the probabilities, and log probabilities.
  There are multiple decoders to support beam search. And each one will be
  passed to `whisper_sample_token_topk`. So we want like 5 separate "lines" of
  decoding to happen and then the best one is later selected.
* A batch will then be used with all the sequences for the decoder bean "lines"
  So this will pass a batch of sequences to the decoder (`whisper_decode_internal`).


```console
whisper_exp_compute_token_level_timestamps: thold_pt = 0.010, thold_ptsum = 0.010

segment.t0 is 0, setting it to the first token timestamp

whisper_exp_compute_token_level_timestamps: segment 0: t0 = 2112, t1 = 2600, n_tokens = 24
Token 0: id=50363, tid=50363, tt=2112, pt=0.755, ptsum=0.775, tid_last=50363
Token 1: id=1318, tid=50363, tt=2112, pt=0.000, ptsum=0.000, tid_last=50363
Token 2: id=1752, tid=51419, tt=4224, pt=0.263, ptsum=0.029, tid_last=50363
Token 3: id=5615, tid=51419, tt=4224, pt=0.026, ptsum=0.006, tid_last=50363
Token 4: id=257, tid=51439, tt=4264, pt=0.033, ptsum=0.004, tid_last=50363
Token 5: id=3595, tid=51443, tt=4272, pt=0.210, ptsum=0.021, tid_last=50363
Token 6: id=35280, tid=51859, tt=5104, pt=0.013, ptsum=0.004, tid_last=50363
Token 7: id=11, tid=51487, tt=4360, pt=0.045, ptsum=0.002, tid_last=50363
Token 8: id=508, tid=51491, tt=4368, pt=0.045, ptsum=0.006, tid_last=50363
Token 9: id=550, tid=51499, tt=4384, pt=0.039, ptsum=0.000, tid_last=50363
Token 10: id=257, tid=51503, tt=4392, pt=0.076, ptsum=0.001, tid_last=50363
Token 11: id=3367, tid=51505, tt=4396, pt=0.104, ptsum=0.013, tid_last=50363
Token 12: id=1444, tid=51520, tt=4426, pt=0.191, ptsum=0.182, tid_last=50363
Token 13: id=978, tid=51559, tt=4504, pt=0.041, ptsum=0.007, tid_last=50363
Token 14: id=46782, tid=51151, tt=3688, pt=0.011, ptsum=0.000, tid_last=50363
Token 15: id=11, tid=51576, tt=4538, pt=0.040, ptsum=0.002, tid_last=50363
Token 16: id=257, tid=51577, tt=4540, pt=0.114, ptsum=0.013, tid_last=50363
Token 17: id=36138, tid=51589, tt=4564, pt=0.090, ptsum=0.001, tid_last=50363
Token 18: id=21696, tid=51614, tt=4614, pt=0.032, ptsum=0.004, tid_last=50363
Token 19: id=12, tid=51624, tt=4634, pt=0.069, ptsum=0.004, tid_last=50363
Token 20: id=7081, tid=51645, tt=4676, pt=0.006, ptsum=0.000, tid_last=50363
Token 21: id=11, tid=51655, tt=4696, pt=0.024, ptsum=0.001, tid_last=50363
Token 22: id=508, tid=51660, tt=4706, pt=0.110, ptsum=0.009, tid_last=50363
Token 23: id=51663, tid=51663, tt=4712, pt=0.079, ptsum=0.793, tid_last=50363

[00:00:21.120 --> 00:00:26.000]   There once lived a poor tailor, who had a son called Aladdin, a careless idle-boy, who

whisper_exp_compute_token_level_timestamps: thold_pt = 0.010, thold_ptsum = 0.010

whisper_exp_compute_token_level_timestamps: segment 1: t0 = 2600, t1 = 5600, n_tokens = 18

Token 0: id=50363, tid=50363, tt=2600, pt=0.993, ptsum=0.996, tid_last=50363
Token 1: id=561, tid=50363, tt=2600, pt=0.000, ptsum=0.000, tid_last=50363
Token 2: id=466, tid=50413, tt=2700, pt=0.008, ptsum=0.000, tid_last=50363
Token 3: id=2147, tid=50713, tt=3300, pt=0.006, ptsum=0.000, tid_last=50363
Token 4: id=475, tid=50413, tt=2700, pt=0.058, ptsum=0.000, tid_last=50363
Token 5: id=711, tid=50713, tt=3300, pt=0.010, ptsum=0.000, tid_last=50363
Token 6: id=477, tid=50713, tt=3300, pt=0.009, ptsum=0.000, tid_last=50363
Token 7: id=1110, tid=50713, tt=3300, pt=0.003, ptsum=0.000, tid_last=50363
Token 8: id=890, tid=51491, tt=4856, pt=0.005, ptsum=0.000, tid_last=50363
Token 9: id=287, tid=50413, tt=2700, pt=0.010, ptsum=0.000, tid_last=50363
Token 10: id=262, tid=50463, tt=2800, pt=0.010, ptsum=0.000, tid_last=50363
Token 11: id=6483, tid=50463, tt=2800, pt=0.098, ptsum=0.000, tid_last=50363
Token 12: id=11, tid=50500, tt=2874, pt=0.036, ptsum=0.001, tid_last=50363
Token 13: id=351, tid=50510, tt=2894, pt=0.055, ptsum=0.001, tid_last=50363
Token 14: id=1310, tid=50513, tt=2900, pt=0.104, ptsum=0.002, tid_last=50363
Token 15: id=21696, tid=50532, tt=2938, pt=0.011, ptsum=0.001, tid_last=50363
Token 16: id=6510, tid=50548, tt=2970, pt=0.234, ptsum=0.006, tid_last=50363
Token 17: id=50256, tid=50563, tt=3000, pt=0.021, ptsum=0.018, tid_last=50363
whisper_exp_compute_token_level_timestamps: ---------> token id: 50256
[00:00:26.000 --> 00:00:56.000]   would do nothing but play all day long in the streets, with little idle boys
```

```c++
        if (token.pt > thold_pt && token.ptsum > thold_ptsum && token.tid > tid_last && tt <= t1) {
            if (j > 0) {
                tokens[j - 1].t1 = tt;
            }
            tokens[j].t0 = tt;
            tid_last = token.tid;
        }
```

* token.pt > thold_pt       - Token probability > 0.010
* token.ptsum > thold_ptsum - Token probability sum > 0.010
* token.tid > tid_last      - Current timestamp ID > last assigned timestamp ID
* tt <= t1                  - Calculated time <= segment end time
