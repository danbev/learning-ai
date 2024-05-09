## llama.cpp timing issue
This is for an issue I'm working on which is an issue related to handling
sessions of tokens in llama.cpp. The issue is with the number of prompt tokens
reported (`n_p_eval`) by `llama_get_timings`.

### Steps to reproduce
Start the main application and specify the following parameters:
```console
./main -m models/tinyllama-1.1b-1t-openorca.Q2_K.gguf --prompt 'The answer to 1 + 1 is' -n 5 --verbose-prompt --prompt-cache main-session.txt
```
Notice that we are specifying the `--prompt-cache` parameter to store the
session in a file called `main-session.txt`.

```console
<s> The answer to 1 + 1 is
[llama_decode_internal] n_tokens: 10, token[0]: 1
[llama_synchronize] Enter n_eval: 0, n_p_eval: 0
[llama_synchronize] Exit n_eval: 0, n_p_eval: 10
[llama_synchronize] Enter n_eval: 0, n_p_eval: 10
[llama_synchronize] Exit n_eval: 0, n_p_eval: 10
[llama_synchronize] Enter n_eval: 0, n_p_eval: 10
[llama_synchronize] Exit n_eval: 0, n_p_eval: 10
 [llama_decode_internal] n_tokens: 1, token[0]: 29871
[llama_synchronize] Enter n_eval: 0, n_p_eval: 10
[llama_synchronize] Exit n_eval: 1, n_p_eval: 10
[llama_synchronize] Enter n_eval: 1, n_p_eval: 10
[llama_synchronize] Exit n_eval: 1, n_p_eval: 10
2
[llama_decode_internal] n_tokens: 1, token[0]: 29906
[llama_synchronize] Enter n_eval: 1, n_p_eval: 10
[llama_synchronize] Exit n_eval: 2, n_p_eval: 10
[llama_synchronize] Enter n_eval: 2, n_p_eval: 10
[llama_synchronize] Exit n_eval: 2, n_p_eval: 10
.[llama_decode_internal] n_tokens: 1, token[0]: 29889
[llama_synchronize] Enter n_eval: 2, n_p_eval: 10
[llama_synchronize] Exit n_eval: 3, n_p_eval: 10
[llama_synchronize] Enter n_eval: 3, n_p_eval: 10
[llama_synchronize] Exit n_eval: 3, n_p_eval: 10
 The
[llama_decode_internal] n_tokens: 1, token[0]: 450
[llama_synchronize] Enter n_eval: 3, n_p_eval: 10
[llama_synchronize] Exit n_eval: 4, n_p_eval: 10
[llama_synchronize] Enter n_eval: 4, n_p_eval: 10
[llama_synchronize] Exit n_eval: 4, n_p_eval: 10
 answer
[llama_print_timings] n_eval: 4, n_p_eval: 10

llama_print_timings:        load time =     106,78 ms
llama_print_timings:      sample time =       0,42 ms /     5 runs   (    0,08 ms per token, 11990,41 tokens per second)
llama_print_timings: prompt eval time =     405,00 ms /    10 tokens (   40,50 ms per token,    24,69 tokens per second)
llama_print_timings:        eval time =     191,03 ms /     4 runs   (   47,76 ms per token,    20,94 tokens per second)
llama_print_timings:       total time =     598,19 ms /    14 tokens
```
Notice that `llama_decode` was called with a batch size of 10 tokens which is
the prompt we specified. And notice that this is reflected by number of tokens
in the prompt (`n_p_eval`). I've added a print statement to
`llama_print_timings` to print the number of tokens in the prompt.

Now, let run this again and this time the prompt tokens should be taken from the
session file:
```
<s> The answer to 1 + 1 is
[llama_synchronize] Enter n_eval: 0, n_p_eval: 0
[llama_synchronize] Exit n_eval: 0, n_p_eval: 0
[llama_synchronize] Enter n_eval: 0, n_p_eval: 0
[llama_synchronize] Exit n_eval: 0, n_p_eval: 0
 
[llama_decode_internal] n_tokens: 1, token[0]: 29871
[llama_synchronize] Enter n_eval: 0, n_p_eval: 0
[llama_synchronize] Exit n_eval: 1, n_p_eval: 0
[llama_synchronize] Enter n_eval: 1, n_p_eval: 0
[llama_synchronize] Exit n_eval: 1, n_p_eval: 0
2
[llama_decode_internal] n_tokens: 1, token[0]: 29906
[llama_synchronize] Enter n_eval: 1, n_p_eval: 0
[llama_synchronize] Exit n_eval: 2, n_p_eval: 0
[llama_synchronize] Enter n_eval: 2, n_p_eval: 0
[llama_synchronize] Exit n_eval: 2, n_p_eval: 0
.
[llama_decode_internal] n_tokens: 1, token[0]: 29889
[llama_synchronize] Enter n_eval: 2, n_p_eval: 0
[llama_synchronize] Exit n_eval: 3, n_p_eval: 0
[llama_synchronize] Enter n_eval: 3, n_p_eval: 0
[llama_synchronize] Exit n_eval: 3, n_p_eval: 0

[llama_decode_internal] n_tokens: 1, token[0]: 13
[llama_synchronize] Enter n_eval: 3, n_p_eval: 0
[llama_synchronize] Exit n_eval: 4, n_p_eval: 0
[llama_synchronize] Enter n_eval: 4, n_p_eval: 0
[llama_synchronize] Exit n_eval: 4, n_p_eval: 0
As
[llama_print_timings] n_eval: 4, n_p_eval: 0

llama_print_timings:        load time =     102,90 ms
llama_print_timings:      sample time =       0,46 ms /     5 runs   (    0,09 ms per token, 10989,01 tokens per second)
llama_print_timings: prompt eval time =       0,00 ms /     1 tokens (    0,00 ms per token,      inf tokens per second)
llama_print_timings:        eval time =     182,13 ms /     4 runs   (   45,53 ms per token,    21,96 tokens per second)
llama_print_timings:       total time =     206,21 ms /     5 tokens
```
So this looks good to me and the prompt is not sent to `llama_decode`, but
notice in the print statement from `llama_print_timings`:
```console
[llama_print_timings] n_eval: 4, n_p_eval: 0
```
This is printed before the call to `llama_get_timings` and aims to show that
`n_p_eval` is 0, but that this function call will return 1 for `n_p_eval`.
```c++
    LLAMA_LOG_INFO("[llama_print_timings] n_eval: %d, n_p_eval: %d\n", ctx->n_eval, ctx->n_p_eval);
    const llama_timings timings = llama_get_timings(ctx);
```
Now, `llama_get_timings` is called and it returns the following:
```console
struct llama_timings llama_get_timings(struct llama_context * ctx) {
    struct llama_timings result = {
        /*.t_start_ms  =*/ 1e-3 * ctx->t_start_us,
        /*.t_end_ms    =*/ 1.00 * ggml_time_ms(),
        /*.t_load_ms   =*/ 1e-3 * ctx->t_load_us,
        /*.t_sample_ms =*/ 1e-3 * ctx->t_sample_us,
        /*.t_p_eval_ms =*/ 1e-3 * ctx->t_p_eval_us,
        /*.t_eval_ms   =*/ 1e-3 * ctx->t_eval_us,

        /*.n_sample =*/ std::max(1, ctx->n_sample),
        /*.n_p_eval =*/ std::max(1, ctx->n_p_eval),
        /*.n_eval   =*/ std::max(1, ctx->n_eval),
    };

    return result;
}
```
If `ctx->n_p_eval` is zero as in our case, then this will return 1 for
`n_p_eval` which I think might be wrong as `n_p_eval` is never updated.

Instead this could be updated to return 0 if `ctx->n_p_eval` is zero:
```c++
        /*.n_p_eval =*/ std::max(0, ctx->n_p_eval),
```

With this change and re-running the application (remember to remove the
main-session.txt file), we get the following output:
```console
<s> The answer to 1 + 1 is
[llama_decode_internal] n_tokens: 10, token[0]: 1
[llama_synchronize] Enter n_eval: 0, n_p_eval: 0
[llama_synchronize] Exit n_eval: 0, n_p_eval: 10
[llama_synchronize] Enter n_eval: 0, n_p_eval: 10
[llama_synchronize] Exit n_eval: 0, n_p_eval: 10
[llama_synchronize] Enter n_eval: 0, n_p_eval: 10
[llama_synchronize] Exit n_eval: 0, n_p_eval: 10
 
[llama_decode_internal] n_tokens: 1, token[0]: 29871
[llama_synchronize] Enter n_eval: 0, n_p_eval: 10
[llama_synchronize] Exit n_eval: 1, n_p_eval: 10
[llama_synchronize] Enter n_eval: 1, n_p_eval: 10
[llama_synchronize] Exit n_eval: 1, n_p_eval: 10
4
[llama_decode_internal] n_tokens: 1, token[0]: 29946
[llama_synchronize] Enter n_eval: 1, n_p_eval: 10
[llama_synchronize] Exit n_eval: 2, n_p_eval: 10
[llama_synchronize] Enter n_eval: 2, n_p_eval: 10
[llama_synchronize] Exit n_eval: 2, n_p_eval: 10
.
[llama_decode_internal] n_tokens: 1, token[0]: 29889
[llama_synchronize] Enter n_eval: 2, n_p_eval: 10
[llama_synchronize] Exit n_eval: 3, n_p_eval: 10
[llama_synchronize] Enter n_eval: 3, n_p_eval: 10
[llama_synchronize] Exit n_eval: 3, n_p_eval: 10

[llama_decode_internal] n_tokens: 1, token[0]: 13
[llama_synchronize] Enter n_eval: 3, n_p_eval: 10
[llama_synchronize] Exit n_eval: 4, n_p_eval: 10
[llama_synchronize] Enter n_eval: 4, n_p_eval: 10
[llama_synchronize] Exit n_eval: 4, n_p_eval: 10
The[llama_print_timings] n_eval: 4, n_p_eval: 10

llama_print_timings:        load time =     110,17 ms
llama_print_timings:      sample time =       0,52 ms /     5 runs   (    0,10 ms per token,  9596,93 tokens per second)
llama_print_timings: prompt eval time =     406,30 ms /    10 tokens (   40,63 ms per token,    24,61 tokens per second)
llama_print_timings:        eval time =     186,48 ms /     4 runs   (   46,62 ms per token,    21,45 tokens per second)
llama_print_timings:       total time =     595,12 ms /    14 tokens
```
And running a second time:
```console
s> The answer to 1 + 1 is
[llama_synchronize] Enter n_eval: 0, n_p_eval: 0
[llama_synchronize] Exit n_eval: 0, n_p_eval: 0
[llama_synchronize] Enter n_eval: 0, n_p_eval: 0
[llama_synchronize] Exit n_eval: 0, n_p_eval: 0
 [llama_decode_internal] n_tokens: 1, token[0]: 29871
[llama_synchronize] Enter n_eval: 0, n_p_eval: 0
[llama_synchronize] Exit n_eval: 1, n_p_eval: 0
[llama_synchronize] Enter n_eval: 1, n_p_eval: 0
[llama_synchronize] Exit n_eval: 1, n_p_eval: 0
3[llama_decode_internal] n_tokens: 1, token[0]: 29941
[llama_synchronize] Enter n_eval: 1, n_p_eval: 0
[llama_synchronize] Exit n_eval: 2, n_p_eval: 0
[llama_synchronize] Enter n_eval: 2, n_p_eval: 0
[llama_synchronize] Exit n_eval: 2, n_p_eval: 0
.[llama_decode_internal] n_tokens: 1, token[0]: 29889
[llama_synchronize] Enter n_eval: 2, n_p_eval: 0
[llama_synchronize] Exit n_eval: 3, n_p_eval: 0
[llama_synchronize] Enter n_eval: 3, n_p_eval: 0
[llama_synchronize] Exit n_eval: 3, n_p_eval: 0
 The[llama_decode_internal] n_tokens: 1, token[0]: 450
[llama_synchronize] Enter n_eval: 3, n_p_eval: 0
[llama_synchronize] Exit n_eval: 4, n_p_eval: 0
[llama_synchronize] Enter n_eval: 4, n_p_eval: 0
[llama_synchronize] Exit n_eval: 4, n_p_eval: 0
 second
[llama_print_timings] n_eval: 4, n_p_eval: 0

llama_print_timings:        load time =     105,21 ms
llama_print_timings:      sample time =       0,38 ms /     5 runs   (    0,08 ms per token, 13123,36 tokens per second)
llama_print_timings: prompt eval time =       0,00 ms /     0 tokens (    -nan ms per token,     -nan tokens per second)
llama_print_timings:        eval time =     181,53 ms /     4 runs   (   45,38 ms per token,    22,04 tokens per second)
llama_print_timings:       total time =     204,61 ms /     4 tokens
```
The motivation for this change is that it reflects the actual number of prompt
tokens were zero in this case and not 1 as reported previously.

