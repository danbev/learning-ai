## Token level timestamps
The motivation for this document is that I was going to update whisper-cli's
json output function to include VAD mapped timestamps and naively though that it
would simply be a matter of doing the following:
```c++
    auto token = whisper_full_get_token_data(ctx, i, j);
    start_obj(nullptr);
        value_s("text", whisper_token_to_str(ctx, token.id), false);
        if(token.t0 > -1 && token.t1 > -1) {
            // Always resolve t0 and t1 in case VAD is enbled so we map
            // the timestamps to the original audio.
            token.t0 = whisper_full_get_segment_t0(ctx, i);
            token.t1 = whisper_full_get_segment_t1(ctx, i);
            // If we have per-token timestamps, write them out
            times_o(token.t0, token.t1, false);
        }
        ...
```
But this would report the same t0 and t1 segments for all tokens which is not
what the token.t0 and token.t1 field represent. So I need to look into how these
timestamps are generated.

When `whisper_full_with_state` is called and `params.token_timestamps` is true
the the following code will be executed:
```c++
    if (params.token_timestamps) {
        state->t_beg    = 0;
        state->t_last   = 0;
        state->tid_last = 0;
        if (n_samples > 0) {
            state->energy = get_signal_energy(samples, n_samples, 32);
        }
    }
```
And `state-energy` is defined as follows:
```c++
struct whisper_state {
    ...

    std::vector<float> energy; // PCM signal energy
```
And if we look at the `get_signal_energy` function we find:
```c++
static std::vector<float> get_signal_energy(const float * signal, int n_samples, int n_samples_per_half_window) {
    const int hw = n_samples_per_half_window;

    std::vector<float> result(n_samples);

    for (int i = 0; i < n_samples; i++) {
        float sum = 0;
        for (int j = -hw; j <= hw; j++) {
            if (i + j >= 0 && i + j < n_samples) {
                sum += fabs(signal[i + j]);
            }
        }
        result[i] = sum/(2*hw + 1);
    }

    return result;
}
```
```console
Thread 1 "whisper-cli" hit Breakpoint 3, get_signal_energy (signal=0x555557c3fbd0, n_samples=176000, n_samples_per_half_window=32)
    at /home/danbev/work/ai/whisper-work/src/whisper.cpp:8328
8328	    const int hw = n_samples_per_half_window;
(gdb) p n_samples_per_half_window 
$13 = 32
(gdb) p n_samples
$14 = 176000
```
So the samples are floating point numbers which represent the voltage of the
signal after it has been sampled and quantized and possibly normalized.
The for loop is like a 1d convolution where for each sample we take the sum of
the 32 samples before and after the current sample so we can calculate a moving
average (so a window of 64).

So for each value in the samples this is taking a window of 64 and calculating
the sum of those amplitudes, and then setting the energy level for this sample
to the mean. At 16kHz 65 samples is about 4ms.
The raw amplitudes, like the raw signal voltage, fluctuate rapidly, so it is
more useful to see an average of a number of samples to see what it actually looks
like.
```
Raw audio:      /\/\/\|\|\|\/\/\/\_____/\/\/\|\|\|\/\/\/\______
                      ^Word 1^         ^Word 2^
```
```
Energy profile:  ____/‾‾‾‾\____/‾‾‾‾\____
                      ^Word 1^  ^Word 2^
```

This moving average approach effectively acts as a low-pass filter, removing
high-frequency oscillations in the signal while preserving the overall amplitude
which is useful for detecting the presence or absence of speech.

Alright so back to the `whisper_full_with_state` and how the token timestamps
are generated:
```c++
    if (params.token_timestamps) {
        whisper_exp_compute_token_level_timestamps(
                *ctx, *state, result_all.size() - 1, params.thold_pt, params.thold_ptsum);

        if (params.max_len > 0) {
            n_new = whisper_wrap_segment(*ctx, *state, params.max_len, params.split_on_word);
        }
    }
```
```c++
static void whisper_exp_compute_token_level_timestamps(
        struct whisper_context & ctx,
          struct whisper_state & state,
                           int   i_segment,
                         float   thold_pt,
                         float   thold_ptsum) {
    auto & segment = state.result_all[i_segment];
    auto & tokens  = segment.tokens;

    const int n_samples = state.energy.size();
    ...

    const int64_t t0 = segment.t0;
    const int64_t t1 = segment.t1;

    const int n = tokens.size();
```
So t0 and t1 are the segments timestamps:
```console
(gdb) p t0
$31 = 0
(gdb) p t1
$32 = 808

(gdb) p segment.text
$34 = " And so, my fellow Americans ask not what your country can do for you, ask what you can do for your country."

(gdb) p n
$35 = 27
```
```c++
    auto & t_beg    = state.t_beg;
    auto & t_last   = state.t_last;
    auto & tid_last = state.tid_last;

    for (int j = 0; j < n; ++j) {
        auto & token = tokens[j];
```
`t_beg` and `t_last` are for DTW handling.
So this will iterate over all the tokens in the segment, which is 27 for this
debugging session.

```console
(gdb) p token
$42 = (whisper_token_data &) @0x55555cd44c20: {
id = 50364, tid = 50364, p = 0.989245474, plog = -0.0108127594, pt = 0.989245474, 
ptsum = 0.999060988, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0}
```

The first token will get some special treatment:
```c++
        if (j == 0) {
            if (token.id == whisper_token_beg(&ctx)) {
                tokens[j    ].t0 = t0;
                tokens[j    ].t1 = t0;
                tokens[j + 1].t0 = t0;

                t_beg    = t0;
                t_last   = t0;
                tid_last = whisper_token_beg(&ctx);
            } else {
                tokens[j    ].t0 = t_last;
            }
        }
```
```console
(gdb) p token.id
$11 = 50364
(gdb) p whisper_token_beg(&ctx)
$12 = 50364
```
So in our case the first token is the beginning of the segment and we set the
duration to zero (t0 == t1) as this is not a real spoken content but a marker
token. And it will also set the next tokens start time to the same start, as
this is where the actual spoken content starts. And it also sets the last token
id to be the beginning token id.

Next, we have calculate the token time (I think):
```c++
        const int64_t tt = t_beg + 2*(token.tid - whisper_token_beg(&ctx));
```
So what this is doing is that is it calculating/estimating a timestamp for the
token with token id `token.tid`. This will be an offset from `t_beg` which I
think is the beginning timestamp for this segment.
```console
(gdb) p ctx->vocab.n_vocab
$17 = 51865
```
Notice that we take the current token id and subtract that from the beginning
of segment id which is 50364. So we are finding out how far way this token is
from the beginning of segment token. We then multiply this by 2. Why 2? I think
2 might represent a time unit, like each token corresponds to 2 time units in
the real world (takes that amount of time to speak). And we then offset this
product from the beginning of the segment which gives us a rough estimate of
the timestamp for this token.

Following this we have:
```c++
        tokens[j].id    = token.id;
        tokens[j].tid   = token.tid;
        tokens[j].p     = token.p;
        tokens[j].pt    = token.pt;
        tokens[j].ptsum = token.ptsum;
```
These are actually setting the values on the same object, that is tokens[j] and
token refer to the same object. I've opened https://github.com/ggml-org/whisper.cpp/pull/3178
with a suggestion to remove these to avoid confusion.

Next we have:
```c++
        tokens[j].vlen = voice_length(whisper_token_to_str(&ctx, token.id));
```
So this is first getting the string representation of the token id:
```console
(gdb) p whisper_token_to_str(&ctx, token.id)
$8 = 0x55555635e488 "[_BEG_]"
```
So this will be passed to `voice_length`:
```c++
// a cost-function / heuristic that is high for text that takes longer to pronounce
// obviously, can be improved
static float voice_length(const std::string & text) {
    float res = 0.0f;

    for (char c : text) {
        if (c == ' ') {
            res += 0.01f;
        } else if (c == ',') {
            res += 2.00f;
        } else if (c == '.') {
            res += 3.00f;
        } else if (c == '!') {
            res += 3.00f;
        } else if (c == '?') {
            res += 3.00f;
        } else if (c >= '0' && c <= '9') {
            res += 3.00f;
        } else {
            res += 1.00f;
        }
    }

    return res;
}
```
This is a cost function that is used to estimate the length of time it would
take to pronounce the text. Notice that it will go throuch each character in the
string.

So a space only takes 0.01 seconds, while a comma takes 2 seconds etc. And this
score will be stored for the token. It feels like for the beginning token this
would not have to be calculated but perhaps there is a reason for this that
might become clear later.

The last thing to happen in the token "initialization/setting" loop the setting
of the tokens timestamp if the following conditions are met:
```c++
        if (token.pt > thold_pt && token.ptsum > thold_ptsum && token.tid > tid_last && tt <= t1) {
            if (j > 0) {
                tokens[j - 1].t1 = tt;
            }
            tokens[j].t0 = tt;
            tid_last = token.tid;
        }

```
The `thold_pt` and `thols_ptsum` are parameters:
```console
(gdb) p params.thold_pt
$13 = 0.00999999978
(gdb) p params.thold_ptsum
$14 = 0.00999999978
(gdb) p 0.01f
$26 = 0.00999999978
```
And the default values for these are:
```c++
        /*.thold_pt          =*/ 0.01f,
        /*.thold_ptsum       =*/ 0.01f,
```
```c++
        float p;           // probability of the token
        float plog;        // log probability of the token
        float pt;          // probability of the timestamp token
        float ptsum;       // sum of probabilities of all timestamp tokens
```
So the if statement determines if we should set the tokens timestamp, and is
checking the probability of the timestamp for the token and token probability
sum meet the thresholds. It also makes sure that the token id is greater than
the last token id. This last one is confusing to me. So currently `tid_last` is
the beginning token id which is 50364 so this timestamp for the token will not
be set this time. Now, we need to understand that the model can generate
timestamp tokens and that these are all the tokens greater than 50363:
```console
(gdb) p whisper_token_to_str(&ctx, 50364)
$14 = 0x55555635e488 "[_BEG_]"
(gdb) p whisper_token_to_str(&ctx, 50365)
$15 = 0x55555635e528 "[_TT_1]"
(gdb) p whisper_token_to_str(&ctx, 50366)
$16 = 0x55555635e5c8 "[_TT_2]"
```
So only timestamp tokens will be set, provided they also meet the other
conditions.

For example, token 7 will have the following values:
```console
(gdb) p token
$7 = (whisper_token_data &) @0x55555cd44538: {id = 1029, tid = 50466, p = 0.471208423, plog = -0.752454758, pt = 0.0830370337,
  ptsum = 0.0403849259, t0 = -1, t1 = -1, t_dtw = -1, vlen = 3.00999999}

(gdb) p whisper_token_to_str(&ctx, token.tid)
$8 = 0x555556362448 "[_TT_102]"
```
This is a timestamp token (TT) and the value indicates that this timestamp token
is 102. This means 102 time frames where each frame is 20ms.
```
102 × 0.02 seconds = 2.04 seconds
```
So just to clarify this we have token id's and token timestamp ids:
```console
(gdb) p whisper_token_to_str(&ctx, tokens[0].id)
$68 = 0x55555635e488 "[_BEG_]"
(gdb) p whisper_token_to_str(&ctx, tokens[1].id)
$69 = 0x555555bbd1a8 " And"
(gdb) p whisper_token_to_str(&ctx, tokens[2].id)
$70 = 0x555555bbbee8 " so"
(gdb) p whisper_token_to_str(&ctx, tokens[3].id)
$71 = 0x555555b595b8 ","
(gdb) p whisper_token_to_str(&ctx, tokens[4].id)
$72 = 0x555555bbf228 " my"
(gdb) p whisper_token_to_str(&ctx, tokens[5].id)
$73 = 0x555555cc5d48 " fellow"
(gdb) p whisper_token_to_str(&ctx, tokens[6].id)
$74 = 0x555555ca2ca8 " Americans"

(gdb) p whisper_token_to_str(&ctx, tokens[0].tid)
$75 = 0x55555635e488 "[_BEG_]"
(gdb) p whisper_token_to_str(&ctx, tokens[1].tid)
$76 = 0x55555635e488 "[_BEG_]"
(gdb) p whisper_token_to_str(&ctx, tokens[2].tid)
$77 = 0x555556361688 "[_TT_80]"
(gdb) p whisper_token_to_str(&ctx, tokens[3].tid)
$78 = 0x555556377528 "[_TT_641]"
(gdb) p whisper_token_to_str(&ctx, tokens[4].tid)
$79 = 0x55555635fec8 "[_TT_42]"
(gdb) p whisper_token_to_str(&ctx, tokens[5].tid)
$80 = 0x55555638c248 "[_TT_1174]"
(gdb) p whisper_token_to_str(&ctx, tokens[6].tid)
$81 = 0x555556362308 "[_TT_100]"
```
So at this stage only tokens with high probability of the timestamp will have
their timestamps set. This means that the model is not sure about when this
token occurs in the audio.

After all the tokens have been processes above we will exit the for loops and
continue with:
```c++
    tokens[n - 2].t1 = t1;
    tokens[n - 1].t0 = t1;
    tokens[n - 1].t1 = t1;

    t_last = t1;
```
The above is setting the end time of the second to last token to the t1 which
is the end boundry for the segment. Then it makes the last token a zero duration
token. This is done to make sure that the last token is not a timestamp token
```console
(gdb) p whisper_token_to_str(&ctx, tokens[n-1].tid)
$115 = 0x55555636e108 "[_TT_404]"
(gdb) p whisper_token_to_str(&ctx, tokens[n-1].id)
$116 = 0x55555636e108 "[_TT_404]"
(gdb) p whisper_token_to_str(&ctx, tokens[n-1].tid)
$117 = 0x55555636e108 "[_TT_404]"
(gdb) p whisper_token_to_str(&ctx, tokens[n-2].id)
$118 = 0x555555b596f8 "."
(gdb) p whisper_token_to_str(&ctx, tokens[n-3].id)
$119 = 0x555555bf94c8 " country"
```

After that we have:
```c++
    // find intervals of tokens with unknown timestamps
    // fill the timestamps by proportionally splitting the interval based on the token voice lengths
    {
        int p0 = 0;
        int p1 = 0;

        while (true) {
            while (p1 < n && tokens[p1].t1 < 0) {
                p1++;
            }

            if (p1 >= n) {
                p1--;
            }

            //printf("p0=%d p1=%d t0=%lld t1=%lld\n", p0, p1, tokens[p0].t0, tokens[p1].t1);

            if (p1 > p0) {
                double psum = 0.0;
                for (int j = p0; j <= p1; j++) {
                    psum += tokens[j].vlen;
                }

                //printf("analyzing %d - %d, psum = %f\n", p0, p1, psum);

                const double dt = tokens[p1].t1 - tokens[p0].t0;

                // split the time proportionally to the voice length
                for (int j = p0 + 1; j <= p1; j++) {
                    const double ct = tokens[j - 1].t0 + dt*tokens[j - 1].vlen/psum;

                    tokens[j - 1].t1 = ct;
                    tokens[j    ].t0 = ct;
                }
            }

            p1++;
            p0 = p1;
            if (p1 >= n) {
                break;
            }
        }
    }
```
So this will go through the tokens and first find the first token which has an
end time stamp set. Then it will iterate over the token that come before that
token and perform the computations:
```c++
                double psum = 0.0;
                for (int j = p0; j <= p1; j++) {
                    psum += tokens[j].vlen;
                }

                //printf("analyzing %d - %d, psum = %f\n", p0, p1, psum);

                const double dt = tokens[p1].t1 - tokens[p0].t0;

                // split the time proportionally to the voice length
                for (int j = p0 + 1; j <= p1; j++) {
                    const double ct = tokens[j - 1].t0 + dt*tokens[j - 1].vlen/psum;

                    tokens[j - 1].t1 = ct;
                    tokens[j    ].t0 = ct;
                }
```
This starts by summing all the voice lengths for the tokens in the range.
`dt` is the time duration of the tokens in the range (end token - start token).
Following that the cutoff time, `ct`, is calculated:
```
proportional_duration = (dt * tokens[j - 1].vlen) / psum
```
This gives us the proportion of the total duration that this token should take
for a token considering its voice length.

```c++
    // VAD
    // expand or contract tokens based on voice activity
    {
        const int hw = WHISPER_SAMPLE_RATE/8;

        for (int j = 0; j < n; j++) {
            if (tokens[j].id >= whisper_token_eot(&ctx)) {
                continue;
            }

            int s0 = timestamp_to_sample(tokens[j].t0, n_samples);
            int s1 = timestamp_to_sample(tokens[j].t1, n_samples);
```
This will create a half window of 2000 samples which is 125ms. And note that
s0 and s1 are the number of `samples` for the start and end time of the token and
not starting and end tokens which I mistakenly thought at first.
So how this works is that if we pass a timestamp into the following function
it will convert this timestamp into a sample index in the audio array:
```c++
static int timestamp_to_sample(int64_t t, int n_samples) {
    return std::max(0, std::min((int) n_samples - 1, (int) ((t*WHISPER_SAMPLE_RATE)/100)));
}
```
Here `t` is the timestamp in centiseconds (I think) and `t*WHISPER_SAMPLE_RATE/100`
will convert this into a sample index.

The window is added before the current tokens start and after it's end time.
Then the energy levels for the window is calculated:
```c++
            const int ss0 = std::max(s0 - hw, 0);
            const int ss1 = std::min(s1 + hw, n_samples);
            const int ns = ss1 - ss0;

            float sum = 0.0f;

            for (int k = ss0; k < ss1; k++) {
                sum += state.energy[k];
            }

            const float thold = 0.5*sum/ns;
```
Notice that there is a thold of 0.5.

Next we have, which is going to handle the start time s0, which remember is
an index into the samples, so s0 would be a sample index:
```c++
            {
                int k = s0;
                if (state.energy[k] > thold && j > 0) {
                    while (k > 0 && state.energy[k] > thold) {
                        k--;
                    }
                    tokens[j].t0 = sample_to_timestamp(k);
                    if (tokens[j].t0 < tokens[j - 1].t1) {
                        tokens[j].t0 = tokens[j - 1].t1;
                    } else {
                        s0 = k;
                    }
                } else {
         --->       while (state.energy[k] < thold && k < s1) {
                        k++;
                    }
                    s0 = k;
                    tokens[j].t0 = sample_to_timestamp(k);
                }
            }
```
In our case k=0, and there is no voice activity at energy[k], so this will
iterate over the energy levels starting from 0 and checking each one against the
threshold until it either find energy state above the threshold or reaches the
end of the tokens time range (in samples). It will continue doing this, moving
forward until it finds a value that is greater than the threshold, that is where
there is speech.
```console
(gdb) p k
$175 = 1668
```
At this point it will set the token start timestamp to the timestamp of the
sample where start of speech is detected.
```
(gdb) p sample_to_timestamp(k)
$180 = 10
```
If there had been speech activity at the start the code will check backwards
from s0 (index) and find where the speech activity started and then set the
tokens start time to that sample timestamp. But notice that there is also a
check to avoid setting the start time to a value that is less than the end time
of the previous token. So if there is an overlap then the start time will be set
to the end time of the previous token.

And a similar things is the performed for s1.

Now, the tokens[j].t0 and tokens[j].t1 are timestamps in centiseconds and are
stored as uint64_t types.

### VAD token timestamp integration
When VAD is enabled, this is the VAD that uses a model to detect voice activity
and then filters the input audio samples to only include the parts where there
is speach. This means that the timestamps that whisper sees are the ones that
are relative to the filtered audio and not the original audio. But what we have
done is aligned these to the original audios timestamps by using a mapping.
This is what is reported by the following timestamps:
```console
	"transcription": [
		{
			"timestamps": {
				"from": "00:00:00,290",
				"to": "00:00:10,450"
			},
			"offsets": {
				"from": 290,
				"to": 10450
			},
			"text": " And so, my fellow Americans ask not what your country can do for you, ask what you can do for your country.",
```
These timestamps are retreived by calling:
```c++
static void output_json(
             struct whisper_context * ctx,
                      std::ofstream & fout,
               const whisper_params & params,
    std::vector<std::vector<float>>   pcmf32s) {
        ...
            const int n_segments = whisper_full_n_segments(ctx);
            for (int i = 0; i < n_segments; ++i) {
                const char * text = whisper_full_get_segment_text(ctx, i);

                const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
                const int64_t t1 = whisper_full_get_segment_t1(ctx, i);

                start_obj(nullptr);
                    times_o(t0, t1, false);
                    value_s("text", text, !params.diarize && !params.tinydiarize && !full);

                    ...
```
`whisper_full_get_segment_t0` and `whisper_full_get_segment_t1` will will return
timestamps that are aligned to the original audio and these are the timestamps
for the segment as a whole. There is also the option to enable token level
timestamps using
`--output-json-full/-ojf`:
```console
			"tokens": [
				{
					"text": "[_BEG_]",
					"timestamps": {
						"from": "00:00:00,000",
						"to": "00:00:00,000"
					},
					"offsets": {
						"from": 0,
						"to": 0
					},
					"id": 50364,
					"p": 0.989351,
					"t_dtw": -1
				},
				{
					"text": " And",
					"timestamps": {
						"from": "00:00:00,040",
						"to": "00:00:00,240"
					},
					"offsets": {
						"from": 40,
						"to": 240
					},
					"id": 400,
					"p": 0.584645,
					"t_dtw": -1
				},
```
Now, these timestamps per token and are generated by the following code:
```c++
                    if (full) {
                        start_arr("tokens");
                        const int n = whisper_full_n_tokens(ctx, i);
                        for (int j = 0; j < n; ++j) {
                            auto token = whisper_full_get_token_data(ctx, i, j);
                            start_obj(nullptr);
                                value_s("text", whisper_token_to_str(ctx, token.id), false);
                                if(token.t0 > -1 && token.t1 > -1) {
                                    // If we have per-token timestamps, write them out
                                    times_o(token.t0, token.t1, false);
                                }
                                value_i("id", token.id, false);
                                value_f("p", token.p, false);
                                value_f("t_dtw", token.t_dtw, true);
                            end_obj(j == (n - 1));
                        }
                        end_arr(!params.diarize && !params.tinydiarize);
                    }
```
Above we are getting the token data for each token and using the token
timestamps to output them for each token. But there is no alignment to the
original audio in this case.

I've tried to address this in https://github.com/ggml-org/whisper.cpp/pull/3218.

While working on this I also used an example audio file that was referenced in
https://github.com/ggml-org/whisper.cpp/issues/3207 where an audio book was used
, specifically Aladdin. If we listen to the audio we can hear an introduction
that is about 20 seconds by a speaker with a broad Australian accent. It sounds
like it is the same speaker that is reading the book, but the accent is not as
strong as in the introduction. Now, the interesting thing is that the
introduction does not get transcribed, but the rest of the audio book does. Why
is this? 
I wanted to see how OpenAI's Whisper would process this same audio file:
```console
(whisper-venv) $ whisper samples/aladdin.mp3 --model medium.en --word_timestamps True
100%|█████████████████████████████████████| 1.42G/1.42G [07:07<00:00, 3.57MiB/s]t
[00:00.000 --> 00:25.960]  There once lived a poor tailor, who had a son called Aladdin, a careless idle boy, who
[00:25.960 --> 00:31.760]  would do nothing but play all day long in the streets with little idle boys like himself.
[00:31.760 --> 00:34.960]  This so grieved the father that he died.
[00:34.960 --> 00:40.200]  Yet in spite of his mother's tears and prayers, Aladdin did not mend his ways.
[00:40.200 --> 00:45.720]  One day, when he was playing in the streets as usual, a stranger asked him his age, and
[00:45.720 --> 00:48.320]  if he was not the son of Mustapha the tailor.
[00:48.320 --> 00:53.840]  "'I am, sir,' replied Aladdin, but he died a long while ago.
[00:53.840 --> 01:00.200]  On this the stranger, who was a famous African magician, fell on his neck and kissed him,
[01:00.200 --> 01:01.200]  saying,
[01:01.200 --> 01:05.360]  "'I am your uncle, and knew you from your likeness to my brother.
[01:05.360 --> 01:09.840]  Go to your mother and tell her I am coming.'
[01:09.840 --> 01:13.360]  Aladdin ran home and told his mother of his newly found uncle.
[01:13.360 --> 01:15.680]  "'Indeed, child,' she said.
...
```
And the first token timestamps:
```console
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 20.059999999999995,
      "end": 28.98,
      "text": " There once lived a poor tailor, who had a son called Aladdin, a careless idle boy, who",
      "tokens": [
        50363,
        1318,
        1752,
        5615,
        257,
        3595,
        35280,
        11,
        508,
        550,
        257,
        3367,
        1444,
        978,
        46782,
        11,
        257,
        36138,
        21696,
        2933,
        11,
        508,
        51661
      ],
      "temperature": 0.0,
      "avg_logprob": -0.22713401794433594,
      "compression_ratio": 1.075,
      "no_speech_prob": 0.0505792535841465,
      "words": [
        {
          "word": " There",
          "start": 20.059999999999995,
          "end": 20.639999999999997,
          "probability": 0.1740105301141739
        },
```

And using `whisper-cli` with the same audio file:
```console
[00:00:00.000 --> 00:00:25.960]   There once lived a poor tailor, who had a son called Aladdin, a careless idle boy, who
[00:00:25.960 --> 00:00:31.760]   would do nothing but play all day long in the streets with little idle boys like himself.
[00:00:31.760 --> 00:00:34.960]   This so grieved the father that he died.
[00:00:34.960 --> 00:00:40.200]   Yet in spite of his mother's tears and prayers, Aladdin did not mend his ways.
[00:00:40.200 --> 00:00:45.720]   One day, when he was playing in the streets as usual, a stranger asked him his age, and
[00:00:45.720 --> 00:00:48.400]   if he was not the son of Mustapha the tailor.
[00:00:48.400 --> 00:00:53.840]   "I am, sir," replied Aladdin, "but he died a long while ago."
[00:00:53.840 --> 00:01:00.160]   On this, the stranger, who was a famous African magician, fell on his neck and kissed him,
[00:01:00.160 --> 00:01:07.000]   saying, "I am your uncle, and knew you from your likeness to my brother; go to your mother,
[00:01:07.000 --> 00:01:09.860]   and tell her I am coming."
[00:01:09.860 --> 00:01:13.360]   Aladdin ran home and told his mother of his newly found uncle.
[00:01:13.360 --> 00:01:20.800]   "Indeed, child," she said, "your father had a brother, but I always thought he was dead."
```
And the full json output:
```console
 "transcription": [
    {
      "timestamps": {
        "from": "00:00:00,000",
        "to": "00:00:25,960"
      },
      "offsets": {
        "from": 0,
        "to": 25960
      },
      "text": " There once lived a poor tailor, who had a son called Aladdin, a careless idle boy, who",
      "tokens": [
        {
          "text": "[_BEG_]",
          "timestamps": {
            "from": "00:00:00,000",
            "to": "00:00:00,000"
          },
          "offsets": {
            "from": 0,
            "to": 0
          },
          "id": 50363,
          "p": 0.781686,
          "t_dtw": -1
        },
        {
          "text": " There",
          "timestamps": {
            "from": "00:00:01,120",
            "to": "00:00:21,120"
          },
          "offsets": {
            "from": 1120,
            "to": 21120
          },
          "id": 1318,
          "p": 0.374393,
          "t_dtw": -1
        },
```
Notice that this differs from OpenAI's whisper in tha the `from` is 00:00:01,120
in our case and 00:00:20,059 in OpenAI's Whisper.

And using whisper-cli with VAD:
```console
[00:00:01.120 --> 00:00:25.960]   There once lived a poor Taylor, who had a son called Aladdin, a careless idle-boy, who
[00:00:25.960 --> 00:00:31.750]   would do nothing but play all day long in the streets, with little idle-boys like himself.
[00:00:31.750 --> 00:00:37.790]   Aladdin so grieved the father that he died; yet in spite of his mother's tears and prayers,
[00:00:37.790 --> 00:00:40.180]   Aladdin did not mend his ways.
[00:00:40.180 --> 00:00:45.670]   One day, when he was playing in the streets as usual, a stranger asked him his age, and
[00:00:45.670 --> 00:00:48.140]   if he was not the son of Mustapha the Taylor.
[00:00:48.140 --> 00:00:53.830]   "I am, sir," replied Aladdin; but he died a long while ago.
[00:00:53.830 --> 00:01:00.070]   On this the stranger, who was a famous African magician, fell on his neck and kissed him,
[00:01:00.070 --> 00:01:05.160]   saying, "I am your uncle, and knew you from your likeness to my brother.
[00:01:05.160 --> 00:01:09.820]   Go to your mother, and tell her I am coming."
[00:01:09.820 --> 00:01:13.480]   Aladdin ran home, and told his mother of his newly found uncle.
[00:01:13.480 --> 00:01:19.430]   "Indeed, child," she said, "your father had a brother, but I always thought he was dead."
```

And the full json output:
```console
  "transcription": [
    {
      "timestamps": {
        "from": "00:00:01,120",
        "to": "00:00:25,960"
      },
      "offsets": {
        "from": 1120,
        "to": 25960
      },
      "text": " There once lived a poor Taylor, who had a son called Aladdin, a careless idle-boy, who",
      "tokens": [
        {
          "text": "[_BEG_]",
          "timestamps": {
            "from": "00:00:01,120",
            "to": "00:00:01,120"
          },
          "offsets": {
            "from": 1120,
            "to": 1120
          },
          "id": 50363,
          "p": 0.925896,
          "t_dtw": -1
        },
        {
          "text": " There",
          "timestamps": {
            "from": "00:00:01,120",
            "to": "00:00:21,080"
          },
          "offsets": {
            "from": 1120,
            "to": 21080
          },
          "id": 1318,
          "p": 0.179201,
          "t_dtw": -1
        },
```
Notice here that we have the same from and to timestamp issue as with non VAD.
