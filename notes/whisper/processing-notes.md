## Processing notes
_wip_ will more into a better place later.

So the input audio will first be converted to log mel spectrograms.
Then the encoder will processes the log mel spectrograms and the output
of this will be used in the cross attention of the decoder.

There is a parameter named `params.offset_ms` which is the offset in milliseconds
which can be specified which will cause that amount of audio to be skipped. This can
be useful if there is an introduction/credits at the start of the audio which can
be the case with audio books for example.
```c++
    const int seek_start = params.offset_ms/10;
    const int seek_end = params.duration_ms == 0 ? whisper_n_len_from_state(state) : seek_start + params.duration_ms/10;
```
I'm just noting this as I'll return to this later.

### Decoders
There will be a number of decoders created.
```c++
            for (int j = 0; j < n_decoders_cur; ++j) {
                auto & decoder = state->decoders[j];

                decoder.sequence.tokens.clear();
                decoder.sequence.result_len       = 0;
                decoder.sequence.sum_logprobs_all = 0.0;
                decoder.sequence.sum_logprobs     = -INFINITY;
                decoder.sequence.avg_logprobs     = -INFINITY;
                decoder.sequence.entropy          = 0.0;
                decoder.sequence.score            = -INFINITY;

                decoder.seek_delta = 100*WHISPER_CHUNK_SIZE;

                decoder.failed    = false;
                decoder.completed = false;
                decoder.has_ts    = false;

                if (params.grammar_rules != nullptr) {
                    decoder.grammar = whisper_grammar_init(params.grammar_rules, params.n_grammar_rules, params.i_start_rule);
                } else {
                    decoder.grammar = {};
                }
            }
```

```c++
static std::vector<whisper_token_data> whisper_sample_token_topk(
            whisper_context & ctx,
            whisper_decoder & decoder,
                        int   k) {
    const auto & vocab = ctx.vocab;

    const auto & probs    = decoder.probs;
    const auto & logits   = decoder.logits;
    const auto & logprobs = decoder.logprobs;

    const int n_logits = vocab.n_vocab;

    auto & logits_id = decoder.logits_id;

    logits_id.resize(n_logits);
    for (int i = 0; i < n_logits; ++i) {
        logits_id[i].first = logits[i];
        logits_id[i].second = i;
    }
```
```console
(lldb) p n_logits
(const int) 51864

(lldb) p logits_id
(std::vector<whisper_pair<double, int> > &) size=0 {}
std::vector<whisper_pair<double, whisper_vocab::id>> logits_id;
```
Notice that `logits_id` is a vector of pairs the first element is the logit value
and the second element is the token id. So first this is resized to
the number of logits which in this case is 51864. Then the logit for that position
is stored in the first element and the vocab id is stored in the second element.
```console
(lldb) p logits_id
(std::vector<whisper_pair<double, int> > &) size=51864: {
  [0] = (first = -Inf, second = 0)
  [1] = (first = -Inf, second = 1)
  [2] = (first = -Inf, second = 2)
  [3] = (first = -Inf, second = 3)
  [4] = (first = -Inf, second = 4)
  [5] = (first = -Inf, second = 5)
  [6] = (first = -Inf, second = 6)
  [7] = (first = -Inf, second = 7)
  [8] = (first = -Inf, second = 8)
  [9] = (first = -Inf, second = 9)
  ...
```
```c++

    {
        using pair_type = std::remove_reference<decltype(logits_id)>::type::value_type;
        std::partial_sort(
                logits_id.begin(),
                logits_id.begin() + k, logits_id.end(),
                [](const pair_type & a, const pair_type & b) {
            return a.first > b.first;
        });
    }
```
And after the partial sort:
```console
(lldb) p logits_id
(std::vector<whisper_pair<double, int> > &) size=51864: {
  [0] = (first = 12.230437278747559, second = 50363)
  [1] = (first = 4.4614405632019043, second = 50367)
  [2] = (first = 4.2268733978271484, second = 50365)
  [3] = (first = 3.646557092666626,  second = 50366)
  [4] = (first = 3.5466647148132324, second = 50364)
  [5] = (first = -Inf, second = 5)
```
So these are the k highest logits. The token id's for these are:
```console
(lldb) p ctx.vocab.id_to_token.at(50363)
(std::map<int, std::string>::mapped_type) "[_BEG_]"
(lldb) p ctx.vocab.id_to_token.at(50367)
(std::map<int, std::string>::mapped_type) "[_TT_4]"
(lldb) p ctx.vocab.id_to_token.at(50365)
(std::map<int, std::string>::mapped_type) "[_TT_2]"
(lldb) p ctx.vocab.id_to_token.at(50366)
(std::map<int, std::string>::mapped_type) "[_TT_3]"
(lldb) p ctx.vocab.id_to_token.at(50364)
(std::map<int, std::string>::mapped_type) "[_TT_1]"
```
Next the return value is created and since this is top k and we
have 5 that will be returned:
```c++

    std::vector<whisper_token_data> result;
    result.reserve(k);
```

Next, the token id
```c++
    whisper_token tid = vocab.token_beg;
```
```console
(lldb) p ctx.vocab.token_beg
(whisper_vocab::id) 50363
```

Then we set the token probability to 0.0 and the sum of probabilities to 0.0.
```c++
    float pt    = 0.0;
    float ptsum = 0.0;
```

Next we have a loop that will iterate from 50363 to 51864 (n_logits). This will
skip any probabilities that have a value of `-INFINITY` which is something that
is done in the filtering step (a previous step).

So this is dealing with tokens that are higher that the begin token (50363)
and these are the token timestamps:
```console
(lldb) p ctx.vocab.id_to_token.at(50363)
(std::map<int, std::string>::mapped_type) "[_BEG_]"
(lldb) p ctx.vocab.id_to_token.at(50364)
(std::map<int, std::string>::mapped_type) "[_TT_1]"
(lldb) p ctx.vocab.id_to_token.at(50365)
(std::map<int, std::string>::mapped_type) "[_TT_2]"
(lldb) p ctx.vocab.id_to_token.at(50366)
(std::map<int, std::string>::mapped_type) "[_TT_3]"
(lldb) p ctx.vocab.id_to_token.at(50367)
(std::map<int, std::string>::mapped_type) "[_TT_4]"
(lldb) p ctx.vocab.id_to_token.at(51863)
(std::map<int, std::string>::mapped_type) "[_TT_1500]"
```

So we have this block of code, which will got through all the logits for the
token timestamps, that is the tokens ids that are greater than 50363:
```c++
    float pt    = 0.0;
    float ptsum = 0.0;

    {
        double sum_ts = 0.0;
        double max_ts = 0.0;

        for (int i = vocab.token_beg; i < n_logits; i++) {
            if (probs[i] == -INFINITY) {
                continue;
            }
```
Next in the loop we will check if the current max timestamp is less than
the current probability and if so we update the max timestamp and we set
the token id to current index, which represents a token timestamp in the
vocabulary.

The `sum_ts` is also calculated but not used inside of this loop.
```c++

            sum_ts += probs[i];
            if (max_ts < probs[i]) {
                max_ts = probs[i];
                tid = i;
            }
        }

        pt    = max_ts/(sum_ts + 1e-10);
        ptsum = sum_ts;
    }
```
So we first go through all the probabilities that for token 50363 and higher
which are the bos token and the timestamp token ids (should the bos token really
be included here?) to find the one with the highest probability. But that
probability if for the total tokens and not just the timestamp tokens. So we
need to convert this into a percentage of the timestamp tokens using the sum_ts:
```c++
        pt    = max_ts/(sum_ts + 1e-10);
```
`pt` is the probability of the top timestamp is calculated by dividing the
maximum timestamp by the sum of all timestamps plus a small value to avoid
division by zero (stored in `pt`). If this is a high value then the model
is very confident about the token timing.

`ptsum` is the sum of all valid timestamps probabilities, and if the model
if confidant that there is a timestamp token here it should be close to
1.0. A lower values might indicate that the model thinks that this is
a regular text token and not a timestamp.

I'm thinking about the bos token being included in the timestamp tokens and if
this might be an issue. For example, if we examine the first iteration:
```console
(gdb) p i
$22 = 50363

(gdb) p probs[i]
$23 = 0.755105972
```
This will end up being the `max_ts` value:
```console
(gdb) until 6545
whisper_sample_token_topk (ctx=..., decoder=..., k=5) at /home/danbev/work/ai/whisper-work/src/whisper.cpp:6545
6545	        pt    = max_ts/(sum_ts + 1e-10);

(gdb) p max_ts
$24 = 0.75510597229003906
```


Now, logits is used for finding the most promising tokens from the raw
logits. This was done so that the timestamp tokens could be determined (when 
this was said). And now we want to sample from the top k logits for the language
token (what was said):
```c++

    std::discrete_distribution<> dist(probs.begin(), probs.end());

    for (int i = 0; i < k; ++i) {
        const auto id = dist(decoder.rng);

        result.push_back({ id, tid, probs[id], logprobs[id], pt, ptsum, -1, -1, -1, 0.0f, });

        if (result[i].id >= vocab.token_beg) {
            result[i].tid = result[i].id;
            result[i].pt  = result[i].p;
        }
    }

    return result;
```
And this will be done for all decoders.

Now, lets look at the second token:
```console
(lldb) p logits_id
(std::vector<whisper_pair<double, int> > &) size=51864: {
  [0] = (first = 6.0752081871032715, second = 8355)
  [1] = (first = 5.5768733024597168, second = 978)
  [2] = (first = 5.2564191818237305, second = 1318)
  [3] = (first = 4.6284208297729492, second = 317)
  [4] = (first = 3.4560511112213135, second = 366)
  [5] = (first = -8.5640945434570313, second = 2)
```
```console
(lldb) p ctx.vocab.id_to_token.at(8355)
(std::map<int, std::string>::mapped_type) " AL"
(lldb) p ctx.vocab.id_to_token.at(978)
(std::map<int, std::string>::mapped_type) " Al"
(lldb) p ctx.vocab.id_to_token.at(1318)
(std::map<int, std::string>::mapped_type) " There"
(lldb) p ctx.vocab.id_to_token.at(317)
(std::map<int, std::string>::mapped_type) " A"
(lldb) p ctx.vocab.id_to_token.at(366)
(std::map<int, std::string>::mapped_type) " \""
(lldb) p ctx.vocab.id_to_token.at(2)
(std::map<int, std::string>::mapped_type) "#"
```
And tid will be set to 50363 initially for this as it is for all:
```console
(lldb) p tid
(whisper_token) 50363
```
And we will then compute the token timestamp probabilites, but this time
through the `tid` will never be updated:
```console
(lldb) p tid
(whisper_token) 50363
(lldb) p max_ts
(double) 0
(lldb) p sum_ts
(double) 0
```

The following is from the console output from using a audio sample from an audio
book which contains an introduction. The model used is `meduim.en` and this model
was trained to skip the introduction, which causes the token level timestamps
to not start an 0/beginning but at a later time. This is causing our whisper
token level timestamps to initially start at 0 but then have a end/to timestamp
of the time after the introduction.

But there is also another issue in that the token level timestamps get
repeated in the second segment here:
```console
whisper_exp_compute_token_level_timestamps: thold_pt = 0.010, thold_ptsum = 0.010

segment.t0 is 0, setting it to the first token timestamp

whisper_exp_compute_token_level_timestamps: segment 0: t0 = 2112, t1 = 2600, n_tokens = 24
Token 0: id=50363, tid=50363, tt=2112, pt=0.755, ptsum=0.775, tid_last=50363
Token 1: id=1318,  tid=50363, tt=2112, pt=0.000, ptsum=0.000, tid_last=50363
Token 2: id=1752,  tid=51419, tt=4224, pt=0.263, ptsum=0.029, tid_last=50363
Token 3: id=5615,  tid=51419, tt=4224, pt=0.026, ptsum=0.006, tid_last=50363
Token 4: id=257,   tid=51439, tt=4264, pt=0.033, ptsum=0.004, tid_last=50363
Token 5: id=3595,  tid=51443, tt=4272, pt=0.210, ptsum=0.021, tid_last=50363
Token 6: id=35280, tid=51859, tt=5104, pt=0.013, ptsum=0.004, tid_last=50363
Token 7: id=11,    tid=51487, tt=4360, pt=0.045, ptsum=0.002, tid_last=50363
Token 8: id=508,   tid=51491, tt=4368, pt=0.045, ptsum=0.006, tid_last=50363
Token 9: id=550,   tid=51499, tt=4384, pt=0.039, ptsum=0.000, tid_last=50363
Token 10: id=257,  tid=51503, tt=4392, pt=0.076, ptsum=0.001, tid_last=50363
Token 11: id=3367, tid=51505, tt=4396, pt=0.104, ptsum=0.013, tid_last=50363
Token 12: id=1444, tid=51520, tt=4426, pt=0.191, ptsum=0.182, tid_last=50363
Token 13: id=978,  tid=51559, tt=4504, pt=0.041, ptsum=0.007, tid_last=50363
Token 14: id=46782,tid=51151, tt=3688, pt=0.011, ptsum=0.000, tid_last=50363
Token 15: id=11,   tid=51576, tt=4538, pt=0.040, ptsum=0.002, tid_last=50363
Token 16: id=257,  tid=51577, tt=4540, pt=0.114, ptsum=0.013, tid_last=50363
Token 17: id=36138,tid=51589, tt=4564, pt=0.090, ptsum=0.001, tid_last=50363
Token 18: id=21696,tid=51614, tt=4614, pt=0.032, ptsum=0.004, tid_last=50363
Token 19: id=12,   tid=51624, tt=4634, pt=0.069, ptsum=0.004, tid_last=50363
Token 20: id=7081, tid=51645, tt=4676, pt=0.006, ptsum=0.000, tid_last=50363
Token 21: id=11,   tid=51655, tt=4696, pt=0.024, ptsum=0.001, tid_last=50363
Token 22: id=508,  tid=51660, tt=4706, pt=0.110, ptsum=0.009, tid_last=50363
Token 23: id=51663,tid=51663, tt=4712, pt=0.079, ptsum=0.793, tid_last=50363

[00:00:21.120 --> 00:00:26.000]   There once lived a poor tailor, who had a son called Aladdin, a careless idle-boy, who

whisper_exp_compute_token_level_timestamps: thold_pt = 0.010, thold_ptsum = 0.010

whisper_exp_compute_token_level_timestamps: segment 1: t0 = 2600, t1 = 5600, n_tokens = 18

Token 0: id=50363, tid=50363, tt=2600, pt=0.993, ptsum=0.996, tid_last=50363
Token 1: id=561,   tid=50363, tt=2600, pt=0.000, ptsum=0.000, tid_last=50363
Token 2: id=466,   tid=50413, tt=2700, pt=0.008, ptsum=0.000, tid_last=50363
Token 3: id=2147,  tid=50713, tt=3300, pt=0.006, ptsum=0.000, tid_last=50363
Token 4: id=475,   tid=50413, tt=2700, pt=0.058, ptsum=0.000, tid_last=50363
Token 5: id=711,   tid=50713, tt=3300, pt=0.010, ptsum=0.000, tid_last=50363
Token 6: id=477,   tid=50713, tt=3300, pt=0.009, ptsum=0.000, tid_last=50363
Token 7: id=1110,  tid=50713, tt=3300, pt=0.003, ptsum=0.000, tid_last=50363
Token 8: id=890,   tid=51491, tt=4856, pt=0.005, ptsum=0.000, tid_last=50363
Token 9: id=287,   tid=50413, tt=2700, pt=0.010, ptsum=0.000, tid_last=50363
Token 10: id=262,  tid=50463, tt=2800, pt=0.010, ptsum=0.000, tid_last=50363
Token 11: id=6483, tid=50463, tt=2800, pt=0.098, ptsum=0.000, tid_last=50363
Token 12: id=11,   tid=50500, tt=2874, pt=0.036, ptsum=0.001, tid_last=50363
Token 13: id=351,  tid=50510, tt=2894, pt=0.055, ptsum=0.001, tid_last=50363
Token 14: id=1310, tid=50513, tt=2900, pt=0.104, ptsum=0.002, tid_last=50363
Token 15: id=21696,tid=50532, tt=2938, pt=0.011, ptsum=0.001, tid_last=50363
Token 16: id=6510, tid=50548, tt=2970, pt=0.234, ptsum=0.006, tid_last=50363
Token 17: id=50256,tid=50563, tt=3000, pt=0.021, ptsum=0.018, tid_last=50363
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

### whisper_exp_compute_token_level_timestamps

```c++
static void whisper_exp_compute_token_level_timestamps(
        struct whisper_context & ctx,
          struct whisper_state & state,
                           int   i_segment,
                         float   thold_pt,
                         float   thold_ptsum) {
```

```console
(gdb) p segment
$8 = (whisper_segment &) @0x555555b489b0: {t0 = 0, t1 = 2600,
  text = " There once lived a poor tailor, who had a son called Aladdin, a careless idle-boy, who"

(gdb) p segment.tokens.size()
$15 = 24
```

```c++
    auto & t_beg    = state.t_beg;
    auto & t_last   = state.t_last;   // last assigned timestamp
    auto & tid_last = state.tid_last; // last assigned timestamp id
```

The first for loop will iterate over all the the tokens in the segment (24 in
this case):
```c++
    const int n = tokens.size();
    ...
    const int64_t t0 = segment.t0;
    const int64_t t1 = segment.t1;

    for (int j = 0; j < n; ++j) {
        auto & token = tokens[j];
```
There is some special handling for the first token:
```console
(gdb) p token
$20 = (whisper_token_data &) @0x55555d8727f0:
{id = 50363, tid = 50363, p = 0.755105972, plog = -0.280897141, pt = 0.755105972, 
ptsum = 0.775414705, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0}
```
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
This is checking if the first token is the begin token and if so it will set
this tokens start time stamp (t0) to the segment start time, and likewise the
end timestamps (this is not a spoken word but a token of the model so this makes
sense). It will also set the next tokens start time to the segment start time
as this is the first "real" token.
```console
(gdb) p tokens[j+1]
$23 = {id = 1318, tid = 50363, p = 0.338111937, plog = -1.08437824, pt = 0, ptsum = 0, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0}
(gdb) p ctx.vocab.id_to_token.at(tokens[j+1].id)
$26 = " There"
```
The block then sets `t_beg`, `t_last` to the segment start time.

Next the token timestamp, `tt`, is caclulated:
```c++
        const int64_t tt = t_beg + 2*(token.tid - whisper_token_beg(&ctx));
```
In this case it will simply become `t_beg` which is 2112 in this case.

Next the spoken voice length of the of this token is calculated/predicted:
```c++
        tokens[j].vlen = voice_length(whisper_token_to_str(&ctx, token.id));
```

Following that we have a check if the token probability (`pt`) is greater than
the threshold probability (`thold_pt`) and if the token probability sum
(`ptsum`) is greater than the threshold probability sum (`thold_ptsum`). And it
also states that the token timestamp it (tid) must be greater than the last
token timestamp it (so they are increasing), and also that the current token
timestamp (tt) is less than or equal to the segment end time (t1).
```c++
        if (token.pt > thold_pt && token.ptsum > thold_ptsum && token.tid > tid_last && tt <= t1) {
            if (j > 0) {
                tokens[j - 1].t1 = tt;
            }
            tokens[j].t0 = tt;
            tid_last = token.tid;
        }
```
```console
(gdb) p token.pt > thold_pt
$41 = true
(gdb) p token.ptsum > thold_ptsum
$42 = true
(gdb) p token.tid > tid_last
$43 = false
```
So this is not the case for this first token.
The next token is:
```console
(gdb) p token
$44 = (whisper_token_data &) @0x55555d872828:
{id = 1318, tid = 50363, p = 0.338111937, plog = -1.08437824, pt = 0, ptsum = 0, 
t0 = 2112, t1 = -1, t_dtw = -1, vlen = 0}
```
This token will get the same tt value which also makes sense as this is the first
spoken token:
```console
(gdb) p ctx.vocab.id_to_token.at(tokens[j].id)
$47 = " There"

(gdb) p token.pt > thold_pt
$48 = false
```
So this token will already have gotten a start timestamp (t0) as this was done
by the special handling of of the first token.
```console
(gdb) p token.pt
$52 = 0.262935758
(gdb) p token.pt > thold_pt
$53 = true
(gdb) p token.ptsum > thold_ptsum
$54 = true
(gdb) p token.tid > tid_last
$55 = true
(gdb) p tt <= t1
$56 = false
(gdb) p t1
$57 = 2600
(gdb) p tt
$58 = 4224

(gdb) p ctx.vocab.id_to_token.at(token.tid)
$60 = "[_TT_1056]"
```
So the caclulated token timestamp is 4224 which is greater than the segment end
time (2600). And this will be the case for all the following tokens as well in
this segment. This was a bit suprising to me at first but we have to understand
that unassigned timestamps (-1) will get handled later on in this function.

After the loop as completed we have the following code:
```c++
    tokens[n - 2].t1 = t1;
    tokens[n - 1].t0 = t1;
    tokens[n - 1].t1 = t1;

    t_last = t1;
```
This is updating the second to last token to have an end timestamp (t1) which is
and also set the end token timestamps to same value (this is not a spoken word):
```console
(gdb) p ctx.vocab.id_to_token.at(tokens[22].id)
$104 = " who"
(gdb) p ctx.vocab.id_to_token.at(tokens[23].id)
$105 = "[_TT_1300]"
```
And here `t_last` is also updated.

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
```
This will find the tokens ranges that have unassiged end timestamps (t1).
```c++
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
```
So this will first caclulate the sum of the voice lengths for this range of
tokens:
```console
(gdb) p p0
$120 = 1
(gdb) p p1
$121 = 22
```
Then `dt` is calculated as the difference between the end timestamp of the
```console
(gdb) p tokens[p1].t1
$122 = 2600
(gdb) p tokens[p0].t0
$123 = 2112
(gdb) p tokens[p1].t1 - tokens[p0].t0
$124 = 488
```
So we have a time difference of 488 centiseconds(?) between the first token and
the last token in this range.
```
    0 1                                              24
   [  [                                              ]]
                  unassigned timestamps
                  488 centiseconds (or 4.88 seconds)
```
This is giving each token a slice of the 4.88 seconds proportionally to the
the speaking duration (vlen)
```
const double ct = tokens[j - 1].t0 + dt*tokens[j - 1].vlen/psum;

tokens[j - 1].t0         = start of pervious token
dt                       = total time which is 488 centiseconds
tokens[j - 1].vlen       = how long this token takes to speak (typically)
psum                     = sum of all voice lengths in this range
tokens[j - 1].vlen/psum  = how long this token takes to speak (typically)

(gdb) p ct
$137 = 2144.9676382665052
(gdb) p tokens[j-1].t0
$138 = 2112
(gdb) p tokens[j-1].vlen
$139 = 5.01000023
(gdb) p psum
$140 = 74.160001754760742
(gdb) p dt
$141 = 488
(gdb) p dt*tokens[j-1].vlen/psum
$143 = 32.967638266505105
So token 1 lasts for 32.967638266505105 centiseconds (or 0.32967638266505105 seconds)
and we add this to the start timestamp of the previous token (2112) to get
the end timestamp of this token (2144.9676382665052).

(gdb) p ct
$142 = 2144.9676382665052
```
This is then set as the end timestamp of the previous token and the start of
the current token.

Following this we have a fix up for loop:
```c++
    // fix up (just in case)
    for (int j = 0; j < n - 1; j++) {
        if (tokens[j].t1 < 0) {
            tokens[j + 1].t0 = tokens[j].t1;
        }

        if (j > 0) {
            if (tokens[j - 1].t1 > tokens[j].t0) {
                tokens[j].t0 = tokens[j - 1].t1;
                tokens[j].t1 = std::max(tokens[j].t0, tokens[j].t1);
            }
        }
    }
```
Notice that this is checking if the end timestamp is less than 0 (-1/unsigned)
The last part of this function deals with VAD and those are only tokens that
are related to speach and all others are ignored and my main focus at the moment
is on the timestamps.

So for the first segment everything looks good.

Lets now look at the second segment:
```console
(gdb) p segment
$167 = (whisper_segment &) @0x55555d87f998: {t0 = 2600, t1 = 5600,
  text = " would do nothing but play all day long in the streets, with little idle boys",
  (gdb) p n
$169 = 18

(gdb) p t_beg
$170 = (long &) @0x555556de6e58: 2112
(gdb) p t_last 
$171 = (long &) @0x555556de6e60: 2600
(gdb) p tid_last 
$172 = (int &) @0x555556de6e68: 50363

(gdb) p t0
$174 = 2600
(gdb) p t1
$175 = 5600
```
So must like with the first segment the first token looks like this:
```console
(gdb) p token
$176 = (whisper_token_data &) @0x55555d889a70:
{id = 50363, tid = 50363, p = 0.992690206, plog = -0.00733661652, pt = 0.992690206,
ptsum = 0.996166945, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0}
```
```console
(gdb) p tokens[0].tid
$192 = 50363
(gdb) p tokens[1].tid
$193 = 50363
(gdb) p tokens[2].tid
$194 = 50413
(gdb) p tokens[3].tid
$195 = 50713
(gdb) p tokens[4].tid
$196 = 50413
```
Simliar to the first segment only the first and last token get an timestamp
assigned:
```console
gdb) p tokens[0]
$210 = {id = 50363, tid = 50363, p = 0.992690206, plog = -0.00733661652, pt = 0.992690206, ptsum = 0.996166945, t0 = 2600,
  t1 = 2600, t_dtw = -1, vlen = 7}
(gdb) p tokens[1]
$211 = {id = 561, tid = 50363, p = 0.731397986, plog = -0.312797546, pt = 0, ptsum = 0, t0 = 2600, t1 = -1, t_dtw = -1,
  vlen = 5.01000023}
(gdb) p tokens[2]
$212 = {id = 466, tid = 50413, p = 0.976555586, plog = -0.0237236023, pt = 0.00771873724, ptsum = 0.000139371186, t0 = -1, t1 = -1,
  t_dtw = -1, vlen = 2.00999999}
(gdb) p tokens[3]
$213 = {id = 2147, tid = 50713, p = 0.99731046, plog = -0.00269317627, pt = 0.00601998903, ptsum = 2.43153554e-05, t0 = -1, t1 = -1,
  t_dtw = -1, vlen = 7.01000023}
(gdb) p tokens[4]
$214 = {id = 475, tid = 50413, p = 0.923540831, plog = -0.0795402527, pt = 0.0576140955, ptsum = 6.76912459e-05, t0 = -1, t1 = -1,
  t_dtw = -1, vlen = 3.00999999}
(gdb) p tokens[5]
$215 = {id = 711, tid = 50713, p = 0.978035629, plog = -0.0222091675, pt = 0.0098046232, ptsum = 0.000181172945, t0 = -1, t1 = -1,
  t_dtw = -1, vlen = 4.01000023}
(gdb) p tokens[6]
$216 = {id = 477, tid = 50713, p = 0.993187368, plog = -0.0068359375, pt = 0.00864463951, ptsum = 5.97840408e-05, t0 = -1, t1 = -1,
  t_dtw = -1, vlen = 3.00999999}
(gdb) p tokens[7]
$217 = {id = 1110, tid = 50713, p = 0.991071761, plog = -0.00896835327, pt = 0.00321834092, ptsum = 7.68845712e-05, t0 = -1,
  t1 = -1, t_dtw = -1, vlen = 3.00999999}
(gdb) p tokens[8]
$218 = {id = 890, tid = 51491, p = 0.987279475, plog = -0.012802124, pt = 0.00459564524, ptsum = 0.000134842005, t0 = -1, t1 = -1,
  t_dtw = -1, vlen = 4.01000023}
(gdb) p tokens[9]
$219 = {id = 287, tid = 50413, p = 0.981878519, plog = -0.0182876587, pt = 0.0104531404, ptsum = 0.000100290883, t0 = -1, t1 = -1,
  t_dtw = -1, vlen = 2.00999999}
(gdb) p tokens[10]
$220 = {id = 262, tid = 50463, p = 0.989247382, plog = -0.0108108521, pt = 0.00953417644, ptsum = 7.15266942e-05, t0 = -1, t1 = -1,
  t_dtw = -1, vlen = 3.00999999}
(gdb) p tokens[11]
$221 = {id = 6483, tid = 50463, p = 0.994812191, plog = -0.00520133972, pt = 0.0977459028, ptsum = 0.000195648099, t0 = -1, t1 = -1,
  t_dtw = -1, vlen = 7.01000023}
(gdb) p tokens[12]
$222 = {id = 11, tid = 50500, p = 0.619108319, plog = -0.479475021, pt = 0.0358210579, ptsum = 0.000778557034, t0 = -1, t1 = -1,
  t_dtw = -1, vlen = 2}
(gdb) p tokens[13]
$223 = {id = 351, tid = 50510, p = 0.988524973, plog = -0.0115413666, pt = 0.054746557, ptsum = 0.00117425004, t0 = -1, t1 = -1,
  t_dtw = -1, vlen = 4.01000023}
(gdb) p tokens[14]
$224 = {id = 1310, tid = 50513, p = 0.964262009, plog = -0.0363922119, pt = 0.104207866, ptsum = 0.00171425869, t0 = -1, t1 = -1,
  t_dtw = -1, vlen = 6.01000023}
(gdb) p tokens[15]
$225 = {id = 21696, tid = 50532, p = 0.964747667, plog = -0.0358886719, pt = 0.0109941205, ptsum = 0.000934321957, t0 = -1, t1 = -1,
  t_dtw = -1, vlen = 4.01000023}
(gdb) p tokens[16]
$226 = {id = 6510, tid = 50548, p = 0.961094618, plog = -0.0396823883, pt = 0.234275848, ptsum = 0.00559387729, t0 = -1, t1 = 5600,
  t_dtw = -1, vlen = 4.01000023}
(gdb) p tokens[17]
$227 = {id = 50256, tid = 50563, p = 0.167677432, plog = -1.7857132, pt = 0.0212340672, ptsum = 0.0178918764, t0 = 5600, t1 = 5600,
  t_dtw = -1, vlen = 13}
```
So like before we will not go through the unassinged tokens to assign timestamps
to them, and the range is much like the first segment the entire segment apart
from the first token and last segment, so from 1 16:
```console
(gdb) p dt
$236 = 3000
(gdb) p psum
$237 = 63.150002002716064
```
So this range has 3000 centicseconds (or 30 seconds) between the first token
and the last token in this range.
```console
(gdb) p dt*tokens[j - 1].vlen/psum
$240 = 238.00475391907463

(gdb) p ct
$242 = 2838.0047539190746
```
So 2838 will be the end timestamp of the first token in this range and the 
start timestamp of the current token. (that was for j=2)
```console
(gdb) p tokens[1]
$284 = {id = 561, tid = 50363, p = 0.731397986, plog = -0.312797546, pt = 0, ptsum = 0,
t0 = 2600, t1 = 2838, t_dtw = -1, vlen = 5.01000023}
(gdb) p tokens[2]
$285 = {id = 466, tid = 50413, p = 0.976555586, plog = -0.0237236023, pt = 0.00771873724, ptsum = 0.000139371186,
t0 = 2838, t1 = 2933, t_dtw = -1, vlen = 2.00999999}
(gdb) p tokens[3]
$286 = {id = 2147, tid = 50713, p = 0.99731046, plog = -0.00269317627, pt = 0.00601998903, ptsum = 2.43153554e-05,
t0 = 2933, t1 = 3266, t_dtw = -1, vlen = 7.01000023}
(gdb) p tokens[4]
$287 = {id = 475, tid = 50413, p = 0.923540831, plog = -0.0795402527, pt = 0.0576140955, ptsum = 6.76912459e-05,
t0 = 3266, t1 = 3408, t_dtw = -1, vlen = 3.00999999}
(gdb) p tokens[5]
$288 = {id = 711, tid = 50713, p = 0.978035629, plog = -0.0222091675, pt = 0.0098046232, ptsum = 0.000181172945,
t0 = 3408, t1 = 3598, t_dtw = -1, vlen = 4.01000023}
(gdb) p tokens[6]
$289 = {id = 477, tid = 50713, p = 0.993187368, plog = -0.0068359375, pt = 0.00864463951, ptsum = 5.97840408e-05,
t0 = 3598, t1 = 3740, t_dtw = -1, vlen = 3.00999999}
(gdb) p tokens[7]
$290 = {id = 1110, tid = 50713, p = 0.991071761, plog = -0.00896835327, pt = 0.00321834092, ptsum = 7.68845712e-05,
t0 = 3740, t1 = 3882, t_dtw = -1, vlen = 3.00999999}
(gdb) p tokens[8]
$291 = {id = 890, tid = 51491, p = 0.987279475, plog = -0.012802124, pt = 0.00459564524, ptsum = 0.000134842005,
t0 = 3882, t1 = 4072, t_dtw = -1, vlen = 4.01000023}
(gdb) p tokens[9]
$292 = {id = 287, tid = 50413, p = 0.981878519, plog = -0.0182876587, pt = 0.0104531404, ptsum = 0.000100290883,
t0 = 4072, t1 = 4167, t_dtw = -1, vlen = 2.00999999}
(gdb) p tokens[10]
$293 = {id = 262, tid = 50463, p = 0.989247382, plog = -0.0108108521, pt = 0.00953417644, ptsum = 7.15266942e-05,
t0 = 4167, t1 = 4309, t_dtw = -1, vlen = 3.00999999}
(gdb) p tokens[11]
$294 = {id = 6483, tid = 50463, p = 0.994812191, plog = -0.00520133972, pt = 0.0977459028, ptsum = 0.000195648099,
t0 = 4309, t1 = 4642, t_dtw = -1, vlen = 7.01000023}
(gdb) p tokens[12]
$295 = {id = 11, tid = 50500, p = 0.619108319, plog = -0.479475021, pt = 0.0358210579, ptsum = 0.000778557034,
t0 = 4642, t1 = 4737, t_dtw = -1, vlen = 2}
(gdb) p tokens[13]
$296 = {id = 351, tid = 50510, p = 0.988524973, plog = -0.0115413666, pt = 0.054746557, ptsum = 0.00117425004,
t0 = 4737, t1 = 4927, t_dtw = -1, vlen = 4.01000023}
(gdb) p tokens[14]
$297 = {id = 1310, tid = 50513, p = 0.964262009, plog = -0.0363922119, pt = 0.104207866, ptsum = 0.00171425869,
t0 = 4927, t1 = 5212, t_dtw = -1, vlen = 6.01000023}
(gdb) p tokens[15]
$298 = {id = 21696, tid = 50532, p = 0.964747667, plog = -0.0358886719, pt = 0.0109941205, ptsum = 0.000934321957,
t0 = 5212, t1 = 5402, t_dtw = -1, vlen = 4.01000023}
(gdb) p tokens[16]
$299 = {id = 6510, tid = 50548, p = 0.961094618, plog = -0.0396823883, pt = 0.234275848, ptsum = 0.00559387729,
t0 = 5402, t1 = 5600, t_dtw = -1, vlen = 4.01000023}
```
So start and end times looks good and they are in increasing order.

I'll probably need to go through this multiple times so lets set a conditional
```console
(gdb) br whisper.cpp:8387 if i_segment == 1
(gdb) r
(gdb) c
```
Now, the "fix up" section did nothing at all and next we have the VAD section:
```c++
        const int hw = WHISPER_SAMPLE_RATE/8;

        for (int j = 0; j < n; j++) {
            if (tokens[j].id >= whisper_token_eot(&ctx)) {
                continue;
            }

            int s0 = timestamp_to_sample(tokens[j].t0, n_samples);
            int s1 = timestamp_to_sample(tokens[j].t1, n_samples);
```
Where `s0` and `s1` are the start and end in samples (not centiseconds):
These are then "windowed" so there is 0.125 seconds before and after the start
and end tokens:
```c++
            const int ss0 = std::max(s0 - hw, 0);
            const int ss1 = std::min(s1 + hw, n_samples);
```
```console
(gdb) br whisper.cpp:8536 if j == 4 && i_segment == 1
```

Ah, this is interesting:
```console
(gdb) p tokens[j].t0
$17 = 3266
(gdb) p tokens[j].t1
$18 = 3408
(gdb) p s0
$19 = 480652
(gdb) p s1
$20 = 480652
```
Notice that both the start end timestamps samples are the same which should not
be the case. These are calculated using:
```c++
            int s0 = timestamp_to_sample(tokens[j].t0, n_samples);
            int s1 = timestamp_to_sample(tokens[j].t1, n_samples);
```

```c++
static int timestamp_to_sample(int64_t t, int n_samples) {
    return std::max(0, std::min((int) n_samples - 1, (int) ((t*WHISPER_SAMPLE_RATE)/100)));
}
```
Now, `n_samples` is:
```console
(gdb) p n_samples
$1 = 480653

(gdb) p tokens[j].t0
$1 = 3266

(gdb) p tokens[j].t0 * 16000 / 100
$2 = 522560

(gdb) p n_samples - 1
$3 = 480652
```
And we will have a similar situation for t1 as well, and `n_samples-1` will be
returned. This is because of how timestamps are now shifted because of the skipping
of the introduction, which is causing this error.

The issue occurs when token timestamps exceed the energy array bounds due to
segment timing misalignment:
```
                  (skipped introduction)
                    ↓
Audio segment:     [2600ms → 5600ms]  (3 seconds of actual audio)
Energy array:      [0 → 480652]       (samples for 3 seconds)
Token timestamps:  [3266ms → 3408ms]  (absolute timestamps)
```
So both s0 and t1 get clamped to the maximum sample index (480652).

The solution is to convert timestamps to segment-relative.
```c++
static int timestamp_to_sample(int64_t t, int64_t segment_t0, int n_samples) {
    // Convert absolute timestamp to segment-relative timestamp
    int64_t relative_t = t - segment_t0;
    int sample = (int)((relative_t * WHISPER_SAMPLE_RATE) / 100);
    return std::max(0, std::min(n_samples - 1, sample));
}

static int64_t sample_to_timestamp(int i_sample, int64_t segment_t0) {
    int64_t relative_timestamp = (100ll * i_sample) / WHISPER_SAMPLE_RATE;
    return relative_timestamp + segment_t0;
}
```
```console
whisper : fix timestamp coordinate transformation for segment processing

This commit addresses an issue with timestamp coordinate transformation
when processing audio segments with non-zero start times (skipped intro).
When processing segments with non-zero start times token timestamps are
absolute while energy arrays are segment-relative.

The motivation for this is that the token level timestamps are incorrect
an audio sample is processed and where the model has been trained to
skip introduction/credits which is the case with the medium model. This
caused `timestamp_to_sample` to return clamped values at array bounds,
breaking VAD code.
```
