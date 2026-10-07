## llama-speculative

### Walkthrough
Breaking in main the first significant thing that happens is that the prompt
is processed by both the target model and the draft model:
```console
    // Tokenize the prompt
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx_tgt, params.prompt, true, true);
```
```console
(gdb) p params.prompt
$6 = "bcdbc"

(gdb) p inp
$5 = std::vector of length 5, capacity 7 = {1, 2, 3, 1, 2}
```
Next, the prompt is processed by both the target model and the draft model:
```c++
    // eval the prompt with both models
    {
        common_batch batch = common_batch_get_one(ctx_tgt, inp.data(), n_input - 1);
        llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.get());

        batch = common_batch_get_one(ctx_tgt, &inp.back(), 1);
        llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.get());

        batch = common_batch_get_one(ctx_dft, inp.data(), n_input);
        llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());
    }
```
Notice that we are doing 3 procees calles (decodes). The first one is the initial
prompt using the target model. Notice that the first batch is using n_input - 1
so it is using the first 4 tokens and not all 5. It then performs a second decode
using only the last token. I think this is done so that we can later use batch
token index 0 to retrieve the logits (need to verify this).
And then we process the entire prompt using the draft model. So at this point
both models have a populated kv cache.

A bit further down we can see that we will initializa a sampler for the target model:
```c++
    // target model sampling context (reuse the llama_context's sampling instance)
    struct common_sampler * smpl = common_sampler_init(model_tgt, params.sampling);
```
This will be using the seed we specified on the command line:
```console
(gdb) p params.sampling.seed
$11 = 1000
```
Then we create a vector to store the draft sequences:
```c++
    // draft sequence data
    std::vector<seq_draft> drafts(n_seq_dft);
```
```console
(gdb) ptype seq_draft
type = struct seq_draft {
    bool active;
    bool drafting;
    bool skip;
    int i_batch_dft;
    std::vector<int> i_batch_tgt;
    std::vector<int> tokens;
    std::vector<std::vector<llama_token_data>> dists;
    common_sampler *smpl;
}
(gdb) p n_seq_dft
$12 = 1
```
And we allocate a sampler for each draft (only one in this case):
```c++
    for (int s = 0; s < n_seq_dft; ++s) {
        // allocate llama_sampler for each draft sequence
        drafts[s].smpl = common_sampler_init(model_dft, params.sampling);
    }
```
Next, we create a common_batch for both models:
```console
    common_batch batch_dft(ctx_dft);
    common_batch batch_tgt(ctx_tgt);
```
Next we resize the batch index of the target and set the index to 0, the first
entry which I think was the reason for the second decode of the prompt on the
target model:
```c++
    // sample from the last token of the prompt
    drafts[0].i_batch_tgt.resize(1);
    drafts[0].i_batch_tgt[0] = 0;
```
Then we have the outer while loop:
```c++
    while (true) {
        std::set<int> active_seqs = {};
        ...
        llama_token token_id;
        std::string token_str;

        // loop until we fail to accept a drafted token or we run out of drafted tokens
        while (true) {
            ...
                bool accept = false;
                if (params.sampling.temp > 0) {
                    // stochastic verification
                    common_sampler_sample(smpl, ctx_tgt, drafts[s_keep].i_batch_tgt[i_dft], true);
                                                               ↑                             ↑
                                                             index                          grammar_first
                    auto & dist_tgt = *common_sampler_get_candidates(smpl, true);
```
```console
(gdb) p drafts[s_keep].i_batch_tgt[i_dft]
$18 = 0
(gdb) p s_keep
```
So this is going to sample from the last token of the prompt using the target model
and notice that it is using 0 as there was only one token in the latest batch
for the target model. This is just the "normal" target model processing which
would happen even if we did not use speculative decoding. The target model
processes the prompt and generates a token. So the above will run through the
target models sampler chain.
And then we get the candidates:
```console
(gdb) p dist_tgt.size
$21 = 3
(gdb) p dist_tgt.data[0]
$22 = {id = 5, logit = 3.35087276, p = 0.82117784}
(gdb) p dist_tgt.data[1]
$23 = {id = 3, logit = 1.56315184, p = 0.13741681}
(gdb) p dist_tgt.data[2]
$24 = {id = 2, logit = 0.363543451, p = 0.0414053649}
```
Next we have:
```c++
                    float p_tgt = 0.0f;
                    float p_dft = 0.0f;

                    while (active_seqs.size() > 0) {
                        ...
                    }
```
But at this point there are no active sequences.
```c++
                    if (!accept) {
                        // all drafted tokens were rejected
                        // sample from the target model
                        LOG_DBG("all drafted tokens were rejected, sampling from residual distribution\n");
                        std::vector<float> probs(dist_tgt.size);   // size = 3
                        for (size_t i = 0; i < dist_tgt.size; ++i) {
                            probs[i] = dist_tgt.data[i].p;
                        }

                        std::discrete_distribution<> dist(probs.begin(), probs.end());

                        const int idx = dist(rng);

                        token_id = dist_tgt.data[idx].id;
                        common_sampler_accept(smpl, token_id, true);
                        token_str = common_token_to_piece(ctx_tgt, token_id);
                    }
```
So this will sample from the target model and accept the token and set the
token_id and token_str:
```console
(gdb) p probs
$29 = std::vector of length 3, capacity 3 = {0.82117784, 0.13741681, 0.0414053649}

(gdb) p idx
$31 = 0
(gdb) p token_id
$32 = 5
(gdb) p token_str
$33 = "f"
```
Next, we prepare the kv cache (memory) for th next round.
```console
                // Removes all tokens that don't belong to sequence 0 (s_keep),
                // which is the only sequence at this moment.
                llama_memory_seq_keep(mem_dft, s_keep);
                // copy all the tokens from sequence 0 to sequence 0 (all positions -1, -1)
                llama_memory_seq_cp  (mem_dft, s_keep, 0, -1, -1);
                // keep only sequence 0 (if we had a different sequence initially then
                // we copied in the previous step and now we only keep sequence 0)
                llama_memory_seq_keep(mem_dft, 0);

                // remove entries from position 5 and above (-1)
                llama_memory_seq_rm  (mem_tgt, s_keep, n_past_tgt, -1);

                // Same as what we did above for the draft model
                llama_memory_seq_keep(mem_tgt, s_keep);
                llama_memory_seq_cp  (mem_tgt, s_keep, 0, -1, -1);
                llama_memory_seq_keep(mem_tgt, 0);
```
```console
(gdb) p s_keep
$34 = 0
(gdb) p n_past_tgt
$35 = 5
(gdb) p n_past_dft
$36 = 5

```
Next, we only have one sequence so the following will reset its drafts entry:
```c++
            for (int s = 0; s < n_seq_dft; ++s) {
                drafts[s].active = false;
                drafts[s].tokens.clear();
                drafts[s].i_batch_tgt.clear();
                drafts[s].dists.clear();
            }
```
Next, we will set the token_id (5)
```c++
            drafts[0].tokens.push_back(token_id);
            drafts[0].dists.push_back(std::vector<llama_token_data>());
            drafts[0].i_batch_tgt.push_back(0);
```
Then we clear the draft batch and then add the token_id to the draft:
```c++
            batch_dft.clear();
            batch_dft.add(token_id, n_past_dft, 0, true);

            llama_memory_seq_rm(mem_dft, 0, n_past_dft, -1);
            llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch_dft.get());
```
And this will decode the next token using the draft model.

Next we will free the draft models sampler and then clone the target models
sampler. We need to do this as samplers can be stateful (rules, and repetition
penalities, grammar constraints etc). So the draft will start off at the same
state but as it is a clone they will evolve independently.
```c++
        if (drafts[0].smpl) {
            common_sampler_free(drafts[0].smpl);
        }
        drafts[0].smpl = common_sampler_clone(smpl);
```
A bit later we have:
```c++
        batch_tgt.clear();
        batch_tgt.add(drafts[0].tokens[0], n_past_tgt, 0, true);
```
So this is clearing the target batch and adding the sampled token to the target
batch. This is just like normal target model processing, decode the prompt,
sample the next token and then add it to the batch and decode, and then repeat.
But here we can append more tokens to the batch using what the draft model
produces whish is a key to the speculative decoding speedup.

We currently have n_draft=3 and here we will sample three tokens from the draft
model:
```c++
        // sample n_draft tokens from the draft model using tree-based sampling
        for (int i = 0; i < n_draft; ++i) {
            batch_dft.clear();
            ...

            for (int s = 0; s < n_seq_dft; ++s) {
                common_sampler_sample(drafts[s].smpl, ctx_dft, drafts[s].i_batch_dft, true);
                const auto * cur_p = common_sampler_get_candidates(drafts[s].smpl, true);
```
So we now sample a token from the draft model (using a clone of the target model
sampler).
```console
(gdb) p cur_p->size
$51 = 3
(gdb) p cur_p->data[0]
$52 = {id = 4, logit = 3.38620996, p = 0.668726742}
(gdb) p cur_p->data[1]
$53 = {id = 6, logit = 2.3550868, p = 0.238472119}
(gdb) p cur_p->data[2]
$54 = {id = 0, logit = 1.41129327, p = 0.0928011313}
```

```c++
                std::vector<int> sa(1, s);
```
Next we have branch splitting which does not happen for our case as we have
-np 1.
```c++
                for (int is = 0; is < (int) sa.size(); ++is) {
                    const llama_token id = cur_p->data[is].id;

                    const int s = sa[is];

                    common_sampler_accept(drafts[s].smpl, id, true);

                    drafts[s].tokens.push_back(id);
                    drafts[s].dists.push_back({cur_p->data, cur_p->data + cur_p->size});
                    drafts[s].i_batch_tgt.push_back(batch_tgt.size());
                    batch_tgt.add(id, n_past_tgt + i + 1, s, true);

                    drafts[s].i_batch_dft = batch_dft.size();
                    batch_dft.add(id, n_past_cur, s, true);
```
```console
(gdb) p id
$57 = 4

(gdb) p drafts[s].tokens
$60 = std::vector of length 2, capacity 2 = {5, 4}
```
And that is then end of the seq_dft loop.

Next we have we will decode using the draft model.
```c++
            // evaluate the drafted tokens on the draft model
            llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch_dft.get());
            ++n_past_cur;
            ++n_drafted;
```
And this will take us back to the begining of the draft loop where i will
be incremented. 
```c++
        for (int i = 0; i < n_draft; ++i) {
```

The next sampled token will be:
```console
(gdb) p cur_p->data[0]
$66 = {id = 6, logit = 3.46636915, p = 0.59843868}
(gdb) p vocab_dft->pimpl->id_to_token[6]
$69 = {text = "g", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}

(gdb) p cur_p->data[1]
$67 = {id = 2, logit = 2.49519038, p = 0.226590693}
(gdb) p vocab_dft->pimpl->id_to_token[2]
$70 = {text = "c", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}

(gdb) p cur_p->data[2]
$68 = {id = 4, logit = 2.2366631, p = 0.174970612}
(gdb) p vocab_dft->pimpl->id_to_token[4]
$71 = {text = "e", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
```

```console
(gdb) p drafts[0].tokens
$73 = std::vector of length 3, capacity 4 = {5, 4, 6}
```

And the last prediction was:
```console
gdb) p cur_p.data[0]
$76 = {id = 3, logit = 4.05669165, p = 0.944831908}
(gdb) p cur_p.data[1]
$77 = {id = 6, logit = 0.608694673, p = 0.0300544277}
(gdb) p cur_p.data[2]
$78 = {id = 1, logit = 0.429097474, p = 0.0251136813}

```
We will then break out of the draft loop.
```c++
        // evaluate the target model on the drafted tokens
        {
            llama_memory_seq_keep(mem_tgt, 0);
            for (int s = 1; s < n_seq_dft; ++s) {
                llama_memory_seq_cp(mem_tgt, 0, s, -1, -1);
            }

            // LOG_DBG("target batch: %s\n", LOG_BATCH_TOSTR_PRETTY(ctx_tgt, batch_tgt).c_str());
            llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch_tgt.get());
            ++n_past_tgt;
        }
```
And this will take us back to outer while loop:
```c++
    while (true) {
        std::set<int> active_seqs = {};
```
```console
127.42.609.004 D draft 0: [ 'e':4, 'g':6, 'd':3 ]
(gdb) p tokens
$79 = std::vector of length 3, capacity 4 = {4, 6, 3}
```
The target model predicted:
```console
(gdb) p dist_tgt.data[0]
$81 = {id = 6, logit = 1.29377508, p = 0.451691478}
(gdb) p dist_tgt.data[1]
$82 = {id = 4, logit = 1.02573812, p = 0.345489562}
(gdb) p dist_tgt.data[2]
$83 = {id = 7, logit = 0.493089437, p = 0.20281896}
```

After the target generates f, the draft decoces it and produces logits for the
token after bcbdcf:
```c++
common_sampler_sample(drafts[s].smpl, ctx_dft, drafts[s].i_batch_dft, true);
```
This samples a token:
(gdb) p cur_p->data[0]
$52 = {id = 4, logit = 3.38620996, p = 0.668726742}
(gdb) p vocab_dft->pimpl->id_to_token[4]
$95 = {text = "e", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}

(gdb) p cur_p->data[1]
$53 = {id = 6, logit = 2.3550868, p = 0.238472119}
(gdb) p vocab_dft->pimpl->id_to_token[6]
$96 = {text = "g", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}

(gdb) p cur_p->data[2]
$54 = {id = 0, logit = 1.41129327, p = 0.0928011313}
(gdb) p vocab_dft->pimpl->id_to_token[2]
$97 = {text = "c", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
```

We sample using the following code:
```c++
                common_sampler_sample(drafts[s].smpl, ctx_dft, drafts[s].i_batch_dft, true);

                const auto * cur_p = common_sampler_get_candidates(drafts[s].smpl, true);
```
But then later we do:
```c++
                // add drafted token for each sequence
                for (int is = 0; is < (int) sa.size(); ++is) {
                    const llama_token id = cur_p->data[is].id;
```
Notice we are always selecting the first candidate.
```console
(gdb) p cur_p.data[0]
$3 = {id = 4, logit = 3.38620996, p = 0.668726742}
(gdb) p cur_p.data[1]
$4 = {id = 6, logit = 2.3550868, p = 0.238472119}
(gdb) p cur_p.data[2]
$5 = {id = 0, logit = 1.41129327, p = 0.0928011313}
```
So in this case it was the highest probability token. But using top-k means that
the sampler should select the top k tokens, and then we should randomlly sample
from those 3. But this is not the case here, we always select the first candidate.
So, top-k keeps three candidates, sampler randomlly selects one according to
their probabilites, code ignores that selection.

With the current seed I'm using this is not really easy to see, but if I change
the seed, I used 822 then I get:
```console
521	                const llama_token selected = common_sampler_sample(drafts[s].smpl, ctx_dft, drafts[s].i_batch_dft, true);
(gdb) n
523	                const auto * cur_p = common_sampler_get_candidates(drafts[s].smpl, true);
(gdb) p selected
$1 = 6
(gdb) n
525	                for (int k = 0; k < std::min(n_seq_dft + 3, (int) cur_p->size); ++k) {
(gdb) p cur_p.data[0]
$2 = {id = 4, logit = 3.38620996, p = 0.668726742}
(gdb) p cur_p.data[1]
$3 = {id = 6, logit = 2.3550868, p = 0.238472119}
(gdb) p cur_p.data[2]
$4 = {id = 0, logit = 1.41129327, p = 0.0928011313}
```
And notice that the selected token was 6, but we still selected token 4 here:
```console
572	                    const llama_token id = cur_p->data[is].id;

(gdb) p id
$5 = 4
```
So regardless of what the sampler selects we will always just use the first
candidate. So we could use the sampled_token above, for example:
```c++
                for (int is = 0; is < (int) sa.size(); ++is) {
                    const llama_token id = sampled_token;
```
And that would work for the single branch path case, but notice if we have multiple
branches then they will all get the same token which is not correct, we want them
to have different tokens to be able to explore different paths.

Using the sampled token fixes single-path proposal selection. But applying that
same token to every split branch duplicates the paths. We need a way to generate
alternative branches whose proposal probabilities agree with the verification
algorithm. So lets say we set `-np 2 --spec-draft-p-split 0.1 -kvu` which will
give us 2 drafting paths (branches)
```c++
            for (int s = 0; s < n_seq_dft; ++s) {
                const llama_token sampled_token = common_sampler_sample(drafts[s].smpl, ctx_dft, drafts[s].i_batch_dft, true);
            }
```
```console
(gdb) p i
$5 = 0

(gdb) p sampled_token
$4 = 6

(gdb) p cur_p.data[0]
$7 = {id = 4, logit = 3.38620996, p = 0.668726742}
(gdb) p cur_p.data[1]
$8 = {id = 6, logit = 2.3550868, p = 0.238472119}
(gdb) p cur_p.data[2]
$9 = {id = 0, logit = 1.41129327, p = 0.0928011313}

0.58.128.145 D  - draft candidate   0 for seq   0, pos   0:      4 (   0.669) 'e'
0.58.128.154 D  - draft candidate   1 for seq   0, pos   0:      6 (   0.238) 'g'
0.58.128.157 D  - draft candidate   2 for seq   0, pos   0:      0 (   0.093) 'a'
```
Then we have will iterate over 8 possible branches but also limited by the
number of allowed drafts (np) and then current number of sequences we have:
```c++
                for (int f = 1; f < 8; ++f) {
                    if (n_seq_cur < n_seq_dft && cur_p->data[f].p > p_draft_split) {
```
```console
(gdb) p n_seq_cur
$6 = 1
(gdb) p n_seq_dft
$7 = 2

(gdb) p cur_p->data[f].p
$9 = 0.238472119
(gdb) p cur_p->data[f].p > p_draft_split
$10 = true
(gdb) p p_draft_split
$11 = 0.100000001
```
```c++
                        LOG_DBG("splitting seq %3d into %3d\n", s, n_seq_cur);

                         
                        //TODO: what exactly is this doing?
                        llama_memory_seq_rm(mem_dft,    n_seq_cur, -1, -1);
                        llama_memory_seq_cp(mem_dft, s, n_seq_cur, -1, -1);

                        for (int t : drafts[s].i_batch_tgt) {
                            batch_tgt.add_seq(t, n_seq_cur);
                        }

                        for (int t : drafts[s].i_batch_tgt) {
                            batch_tgt.add_seq(t, n_seq_cur);
                        }

                        // copy the draft state
                        drafts[n_seq_cur].active   = true;
                        drafts[n_seq_cur].drafting = true;
                        drafts[n_seq_cur].skip     = true;

                        drafts[n_seq_cur].tokens      = drafts[s].tokens;
                        drafts[n_seq_cur].dists       = drafts[s].dists;
                        drafts[n_seq_cur].i_batch_dft = drafts[s].i_batch_dft;
                        drafts[n_seq_cur].i_batch_tgt = drafts[s].i_batch_tgt;

                        // free the drafts sampler so that we can clone the
                        // "original" sampler so that the start with the same
                        // state.
                        if (drafts[n_seq_cur].smpl) {
                            common_sampler_free(drafts[n_seq_cur].smpl);
                        }
                        drafts[n_seq_cur].smpl = common_sampler_clone(drafts[s].smpl);

                        sa.push_back(n_seq_cur);

                        n_seq_cur++;
```
```console
4.23.780.065 D splitting seq   0 into   1
```
In our case np is 2 so that will only happen once. n_seq_cur will be incremented
to 2. And will also have two sequences in it now {0, 1}.

So we will iterator over the following 2 times, and notice that this is where
we will use the sampled_token from above.
```c++
                for (int is = 0; is < (int) sa.size(); ++is) {
                    const llama_token id = sampled_token;

                    // sequence id
                    const int s = sa[is];

                    // accept the token into this draft's sampler
                    common_sampler_accept(drafts[s].smpl, id, true);

                    // add the token to the correct sequence (s)
                    drafts[s].tokens.push_back(id);

                    // save cur_p.data into drafts[s].dists
                    drafts[s].dists.push_back({cur_p->data, cur_p->data + cur_p->size});

                    drafts[s].i_batch_tgt.push_back(batch_tgt.size());

                    // add the sampled token (id) to the batch as sequence s
                    batch_tgt.add(id, n_past_tgt + i + 1, s, true);

                    drafts[s].i_batch_dft = batch_dft.size();
```
The second iteration will the do the same thing:
```console
573	                for (int is = 0; is < (int) sa.size(); ++is) {
(gdb)
574	                    const llama_token id = sampled_token;
(gdb) n
576	                    const int s = sa[is];
(gdb) p id
$32 = 6
(gdb) p is
$33 = 1
```
This will then continue the outer loop:
```console
            for (int s = 0; s < n_seq_dft; ++s) {
                if (!drafts[s].drafting || drafts[s].skip) {
                    continue;
                }
```
And we will continue and also break out of the loop.

So then we will process the batch with contain two sequences:
```console
(gdb) p batch_dft.tokens[0]
$41 = {id = 6, pos = {_M_elems = {6, 0, 0, 0}}, seq_id = 0, output = true, embd = {data = 0x0, n_rows = 0,
    n_embd = 0}, seq_ids_extra = std::vector of length 0, capacity 0, decision_order = 0}
(gdb) p batch_dft.tokens[1]
$42 = {id = 6, pos = {_M_elems = {6, 0, 0, 0}}, seq_id = 1, output = true, embd = {data = 0x0, n_rows = 0,
    n_embd = 0}, seq_ids_extra = std::vector of length 0, capacity 0, decision_order = 0}
```
```c++
            llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch_dft.get());
```
After that we will be continue the for loop:
```c++
        for (int i = 0; i < n_draft; ++i) {
            batch_dft.clear();
```
And again will go through the same process. So clearly the current solution
is incorrect here, we are using the same token id for both sequences.

That is the first issue we have.

Then in the rejection branch (which is the else block) we have the following:
```c++
                        if (r <= p_tgt / p_dft) {
                            s_keep = s;
                            accept = true;
                            token_id = drafts[s].tokens[i_dft];
                            token_str = common_token_to_piece(ctx_tgt, token_id);
                            common_sampler_accept(smpl, token_id, true);

                            LOG_DBG("draft token %d of sequence %d (%d, '%s') accepted\n", i_dft, s, token_id, token_str.c_str());
                            break;
                        } else {
                            LOG_DBG("draft token %d of sequence %d (%d, '%s') rejected\n", i_dft, s, drafts[s].tokens[i_dft], common_token_to_piece(ctx_tgt, drafts[s].tokens[i_dft]).c_str());
                            drafts[s].active = false;

                            // sort dist by id
                            std::sort(dist_tgt.data,
                                dist_tgt.data + dist_tgt.size,
                                [](const llama_token_data &a, const llama_token_data &b) {
                                    return a.id < b.id;
                            });
                            std::sort(dist_dft.data,
                                dist_dft.data + dist_dft.size,
                                [](const llama_token_data &a, const llama_token_data &b) {
                                    return a.id < b.id;
                            });
```
```console
(gdb) p dist_dft.data[0]
$11 = {id = 0, logit = 1.41129327,  p = 0.0928011313}   "a"
(gdb) p dist_dft.data[1]
$12 = {id = 4, logit = 3.38620996,  p = 0.668726742}    "e"
(gdb) p dist_dft.data[2]
$13 = {id = 6, logit = 2.3550868,   p = 0.238472119}    "g"

(gdb) p dist_tgt.data[0]
$14 = {id = 4, logit = 1.02573812,  p = 0.345489562}    "e"
(gdb) p dist_tgt.data[1]
$15 = {id = 6, logit = 1.29377508,  p = 0.451691478}    "g"
(gdb) p dist_tgt.data[2]
$16 = {id = 7, logit = 0.493089437, p = 0.20281896}     "h"
```
```c++

                            float sum_probs = 0.0f;
                            for (size_t i = 0; i < dist_tgt.size; i++) {
                                if (i < dist_dft.size) {
                                    dist_tgt.data[i].p = std::max(0.0f, dist_tgt.data[i].p - dist_dft.data[i].p);
                                } else {
                                    dist_tgt.data[i].p = std::max(0.0f, dist_tgt.data[i].p);
                                }
```
This will set dist_tgt.data[i].p = max(0.0f, dist_tgt.data[i].p - dist_dft.data[i].p);
```console

(gdb) p dist_tgt.data[i].p
$20 = 0.345489562
(gdb) p dist_dft.data[i].p
$21 = 0.0928011313

(gdb) p dist_tgt.data[i].p - dist_dft.data[i].p
$22 = 0.252688438
(gdb) p dist_tgt.data[0].p
$27 = 0.252688438

(gdb) p sum_probs
$28 = 0.252688438

Next iteration:
(gdb) p dist_tgt.data[i].p - dist_dft.data[i].p
$30 = -0.217035264

(gdb) p dist_tgt.data[1].p
$31 = 0

(gdb) p dist_tgt.data[2].p
$34 = 0
```
And then we normalize
```c++
                            for (size_t i = 0; i < dist_tgt.size; i++) {
                                dist_tgt.data[i].p /= sum_probs;
                        }
```
```console
(gdb) p dist_tgt.data[0].p
$36 = 1
(gdb) p dist_tgt.data[1].p
$37 = 0
(gdb) p dist_tgt.data[2].p
$38 = 0
```
So even if e is rejected this code will select e again.


### multi-branch speculation
Above we discussed how a speculative decoding works and we had a single draft
"path". That is we got the anchor, accepted prefix token, and drafted additional
tokens. If a token is rejected then we thrown away the invalid drafts and we use
the residual probabilities to sample from the target model (we have to produce
something). And then we start over. But what if the drafts second choice was
better and might have been accepted if we had take it instead. What this technique
does is allow to have multiple draft paths so that we can explore multiple drafts
options:
```
        [accepted prefix]
        /   split       \          prob > --spec-draft-p-split
   token A             token B 
      |                  |
   token C             token D
      \                  /
      token E        token E  
```
For the split we need to select different tokens, ofgt



Suppose we changed the assignment of the token id back to:
```c++
const llama_token id = cur_p->data[is].id;
```
At the prefix of `bcdbcf` (the prompt was bcdbc and f the token predicted by the
target model, the first tokens would be:
```console
branch 0: e
branch 1: g
```
We would then run:
```c++
std::uniform_int_distribution<unsigned int> u_int_dist(0, active_seqs.size() - 1);
int s = *std::next(active_seqs.begin(), u_int_dist(rng));
```
Which chooses either branch 0 or 1 with equal probability. If it chooses branch
1 it verifies:
```console
p_tgt(g) = 0.4517
p_dft(g) = 0.2485

acceptance ratio = 0.4517 / 0.2485 ≈ 1.814

```



