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
``c++
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

The we se the token probability to 0.0 and the sum of probabilities to 0.0.
```c++
    float pt    = 0.0;
    float ptsum = 0.0;
```

Next we have a loop that will iterate from 50363 to 51864. This will skip
any probabilities that have a value of `-INFINITY` which is something that
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

```c++
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
The `sum_ts` is calculated but not used inside of this loop.
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
The the probability of the top timestamp is calculated by dividing the
maximum timestamp by the sum of all timestamps plus a small value to avoid
division by zero (stored in `pt`). If this is a high value then the model
is very confident about the token timing.

`ptsum` is the sum of all valid timestamps properties, and if the model
if confidant that there is a timestamp token here it should be close to
1.0. A lower values might indicate that the model thinks that this is
a regular text token and not a timestamp.
```console
(lldb) p pt
(float) 0.997391045

(lldb) p ptsum
(float) 0.97783631
```
Now, logits is used for finding the k most promising tokens from the raw
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
