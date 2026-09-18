## llama.cpp server notes
This document contains notes about the llama.cpp server.

### Starting the server
```console
$ ./build/bin/llama-server -m models/llama-2-7b.Q4_K_M.gguf -n 20
```
This server has a GUI which can be accessed at `http://localhost:8080/`.

### Calling the server using curl:
```console
$ ./call-server.sh | jq
{
  "content": " The LoRaWAN Specification\n броја 1.0.2\nThe Low Power",
  "id_slot": 0,
  "stop": true,
  "model": "models/llama-2-7b.Q4_K_M.gguf",
  "tokens_predicted": 20,
  "tokens_evaluated": 6,
  "generation_settings": {
    "n_ctx": 4096,
    "n_predict": 20,
    "model": "models/llama-2-7b.Q4_K_M.gguf",
    "seed": 4294967295,
    "seed_cur": 4203817392,
    "temperature": 0.800000011920929,
    "dynatemp_range": 0.0,
    "dynatemp_exponent": 1.0,
    "top_k": 40,
    "top_p": 0.949999988079071,
    "min_p": 0.05000000074505806,
    "tfs_z": 1.0,
    "typical_p": 1.0,
    "repeat_last_n": 64,
    "repeat_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "mirostat": 0,
    "mirostat_tau": 5.0,
    "mirostat_eta": 0.10000000149011612,
    "penalize_nl": false,
    "stop": [],
    "max_tokens": 20,
    "n_keep": 0,
    "n_discard": 0,
    "ignore_eos": false,
    "stream": false,
    "n_probs": 0,
    "min_keep": 0,
    "grammar": "",
    "samplers": [
      "top_k",
      "tfs_z",
      "typ_p",
      "top_p",
      "min_p",
      "temperature"
    ]
  },
  "prompt": "What is LoRA:",
  "has_new_line": true,
  "truncated": false,
  "stopped_eos": false,
  "stopped_word": false,
  "stopped_limit": true,
  "stopping_word": "",
  "tokens_cached": 25,
  "timings": {
    "prompt_n": 6,
    "prompt_ms": 315.259,
    "prompt_per_token_ms": 52.54316666666667,
    "prompt_per_second": 19.031970538509604,
    "predicted_n": 20,
    "predicted_ms": 876.246,
    "predicted_per_token_ms": 43.8123,
    "predicted_per_second": 22.824640568972644
  },
  "index": 0
}
```

### server_context
There is a single shared server_context which is main:
```c++
int main(int argc, char ** argv) {
    ...
    // struct that contains llama context and inference
    server_context ctx_server;
    ...

    if (!ctx_server.load_model(params)) {
        clean_up();
        t.join();
        LOG_ERR("%s: exiting due to model loading error\n", __func__);
        return 1;
    }
```
```c++
struct server_context {
    common_params params_base;

    // note: keep these alive - they determine the lifetime of the model, context, etc.
    common_init_result llama_init;
    common_init_result llama_init_dft;

    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    ...
};
```
And in load_model we can see that the the llama_context is set:
```c++
    bool load_model(const common_params & params) {
        SRV_INF("loading model '%s'\n", params.model.path.c_str());

        params_base = params;

        llama_init = common_init_from_params(params_base);

        model = llama_init.model.get();
        ctx   = llama_init.context.get();
```
So all slots in the server will share this single llama_context, model, vocab.

### samplers
When a new request comes this will be handled by on_new_task:
```c++
    ctx_server.queue_tasks.on_new_task([&ctx_server](server_task && task) {
        ctx_server.process_single_task(std::move(task));
    });
```
```c++
    void process_single_task(server_task && task) {
        switch (task.type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
                {
                    const int id_slot = task.id_slot;

                    server_slot * slot = id_slot != -1 ? get_slot_by_id(id_slot) : get_available_slot(task);

                    if (slot == nullptr) {
                        // if no slot is available, we defer this task for processing later
                        SRV_DBG("no slot is available, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (!launch_slot_with_task(*slot, std::move(task))) {
                        SRV_ERR("failed to launch slot with task, id_task = %d\n", task.id);
                        break;
                    }
                } break;
```
And if we look in launch_slot_with_task we can see the following:
```c++
    bool launch_slot_with_task(server_slot & slot, server_task && task) {
        slot.reset();
        ...
        // initialize samplers
        {
            if (slot.smpl != nullptr) {
                common_sampler_free(slot.smpl);
            }

            slot.smpl = common_sampler_init(model, task.params.sampling);
            if (slot.smpl == nullptr) {
                // for now, the only error that may happen here is invalid grammar
                send_error(task, "Failed to parse grammar", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
        }
```
So for each new task/request, if a previous sampler exists is it freed and
a new one is created using common_sampler_init.
Each request can specify different sampling parameters in its request, which
is done by params_from_json_cmpl. These will override the servers global defaults
which are set when the server starts.


### slots
The concept of a slot is something that can be good to know up front.
A slot is the server’s long-lived execution context for a single client request.
The number of slots created is determined by the --parallel command line argument 
and this is done in ctx_server.init:
```c++
    void init() {
        const int32_t n_ctx_slot = n_ctx / params_base.n_parallel;

        SRV_INF("initializing slots, n_slots = %d\n", params_base.n_parallel);

        for (int i = 0; i < params_base.n_parallel; i++) {
            server_slot slot;

            slot.id = i;
            slot.ctx = ctx;
            slot.n_ctx = n_ctx_slot;
```
The number of slots is determined by the --parallel command line argument and
this becomes n_parallel. So if we only have 1 slot this means that only one
request will be processed at a time.
The ctx is llama_context from the server_context (this).

So even with n_parallel 1, we can still serve multiple clients/request, but they
will run one after the other. But we can also set it to 2 and then 2 requests
can run and the tokens will be added to the same batch but will have separate
sequence ids for each request.

The total KV/context budget n_ctx is split across slots, so:
```c++
n_ctx_slot = n_ctx / n_parallel
```
Pushing n_parallel higher reduces the context length available per request
unless you also bump --ctx-size.

Each slot needs its own sampler state, prompt cache, etc., so memory footprints
and decode latency go up with larger n_parallel.

```c++
struct server_context {

    common_params params_base;
    // slots / clients
    std::vector<server_slot> slots;
    ...
}
```

A server_context has a llama_batch member:
```c++
    // batching
    llama_batch batch;
```
This is used for decoding and is reset/cleared before each decode:
```c++
        common_batch_clear(batch);
```
```c++
                    while (slot.n_past < slot.n_prompt_tokens() && batch.n_tokens < n_batch) {
                        // get next token to process
                        llama_token cur_tok = input_tokens[slot.n_past];
                        if (cur_tok == LLAMA_TOKEN_NULL) {
                            break; // end of text chunk
                        }

                        // if this is an alora request with pre-invocation
                        // tokens that are not cached, we need to stop filling
                        // this batch at those pre-invocation tokens.
                        if (alora_scale > 0 && slot.n_past == slot.alora_invocation_start - 1) {
                            SLT_DBG(slot, "stop prompt batch filling at (n_past = %d, alora_invocation_start = %d)\n", slot.n_past, slot.alora_invocation_start);
                            break;
                        }

                        // embedding requires all tokens in the batch to be output
                        common_batch_add(batch,
                            cur_tok,
                            slot.prompt.tokens.pos_next(),
                            { slot.id },  // <--- This is where we set the sequence id to the slot id
                            slot.need_embd());
                        slot.prompt.tokens.push_back(cur_tok);

                        slot.n_prompt_tokens_processed++;
                        slot.n_past++;

                        // process the last few tokens of the prompt separately in order to allow for a checkpoint to be created.
                        if (do_checkpoint && slot.n_prompt_tokens() - slot.n_past == 64) {
                            break;
                        }
                    }
```
The following was a little confusing to me at first:
```c++
                    // entire prompt has been processed
                    if (slot.n_past == slot.n_prompt_tokens()) {
                        slot.state = SLOT_STATE_DONE_PROMPT;

                        GGML_ASSERT(batch.n_tokens > 0);

                        common_sampler_reset(slot.smpl);

                        // Process all prompt tokens through sampler system
                        for (int i = 0; i < slot.n_prompt_tokens(); ++i) {
                            llama_token id = input_tokens[i];
                            if (id != LLAMA_TOKEN_NULL) {
                                common_sampler_accept(slot.smpl, id, false);
                            }
                        }
```
The sampler is reset (the prompt has not been processed yet) but we are calling
common_sampler_accept for all the prompt tokens. But this is just to replay
the prompt tokens into the sampler. Notice that it passes in false
That simply tells the sampler "these tokens are already in the context," so its
penalty history matches what the model has seen. The false flag means "don’t
advance the grammar state," because prompt tokens may not have been constrained
by the runtime grammar filter.
Once that history is reconstructed, the subsequent call to common_sampler_sample
can look at the logits of the last prompt token and choose the first generated
token with the correct penalties/grammar state in place.

Next we have:
  ```c++
                        // extract the logits only for the last token
                        batch.logits[batch.n_tokens - 1] = true;

                        slot.n_decoded = 0;
                        slot.i_batch   = batch.n_tokens - 1;
```
This is setting the last tokens logits (which we can think of as output logits
for this token) to true. And then notice that slot.i_batch is set to the last
token, which is the index into the batch for this sequence. So if we want to
get the logits for this sequence this is the value we would use.

The actual llama_decode then happens shortly after this:
```c++
        // process the created batch of tokens
        for (int32_t i = 0; i < batch.n_tokens; i = i_next) {
            const int32_t n_tokens = std::min(n_batch, batch.n_tokens - i);

            llama_batch batch_view = {
                n_tokens,
                batch.token    + i,
                nullptr,
                batch.pos      + i,
                batch.n_seq_id + i,
                batch.seq_id   + i,
                batch.logits   + i,
            };

            const int ret = llama_decode(ctx, batch_view);
            ...

            i_next = i + n_tokens;
```
So in update_slot, we iterate over each slot and each slots sampler is able
to accept the tokens that are to be processes. Then the tokens are added
to the batch using the slot's id as the sequence id.

After decoding we also iterate over all the slots so that their samplers can
sample the token generated for that slot, notice that this uses the token index
in the batch (tok_idx):
```c++
            for (auto & slot : slots) {
                ...

                const int tok_idx = slot.i_batch - i;

                llama_token id = common_sampler_sample(slot.smpl, ctx, tok_idx);

                slot.i_batch = -1;

                common_sampler_accept(slot.smpl, id, true);

                slot.n_decoded += 1;
```
And this is where the samplers get a chance to sampler the tokens for the
specific index.
```console
(gdb) p tok_idx
$42 = 25
(gdb) p id
$40 = 1318
(gdb) p this->vocab->pimpl->id_to_token[id]
$41 = {text = "The", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
```

For GPU sampling perhaps there should be a way of calling a function like
llama_has_sampled_token as the GPU samplers might already have sampled a token.
And perhaps we could add llama_has_sampled_probs what can be checked in
common_sampler::set_logits to populate llama_token_data_array.
```c++
                completion_token_output result;
                result.tok          = id;
                result.text_to_send = common_token_to_piece(ctx, result.tok, accept_special_token(slot, result.tok));
                result.prob         = 1.0f; // TODO: set it here instead of doing inside populate_token_probs
```
```console
(gdb) p result
$44 = {tok = 1318, prob = 1, text_to_send = "The", probs = std::vector of length 0, capacity 0}
```
Next we have process_token:
```c++
                if (!process_token(result, slot)) {
                    // release slot because of stop condition
                    slot.print_timings();
                    send_final_response(slot);
                    metrics.on_prediction(slot);
                    slot.release();

                    continue;
                }
```


### Walkthrough
This section will step through the server code to understand how it works.

```console
$ lldb ./build/bin/llama-server -- -m models/llama-2-7b.Q4_K_M.gguf -n 20
(lldb) br set -f server.cpp -l 2436
Breakpoint 1: where = llama-server`main + 120 at server.cpp:2436:19, address = 0x000000010000221c

(lldb) r
Process 94087 launched: '/Users/danbev/work/llama.cpp/build/bin/llama-server' (arm64)
Process 94087 stopped
* thread #1, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
    frame #0: 0x000000010000221c llama-server`main(argc=5, argv=0x000000016fdff2e0) at server.cpp:2436:19
   2433
   2434	int main(int argc, char ** argv) {
   2435	    // own arguments required by this example
-> 2436	    common_params params;
   2437
   2438	    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
   2439	        return 1;
Target 0: (llama-server) stopped.
```

To be able to step through the actual server processing we can set a breakpoint
in the repsonse handler:
```c++
    auto middleware_server_state = [&res_error, &state](const httplib::Request & req, httplib::Response & res) {
        server_state current_state = state.load();
        if (current_state == SERVER_STATE_LOADING_MODEL) {
            auto tmp = string_split<std::string>(req.path, '.');
            if (req.path == "/" || tmp.back() == "html") {
                res.set_content(reinterpret_cast<const char*>(loading_html), loading_html_len, "text/html; charset=utf-8");
                res.status = 503;
            } else {
                res_error(res, format_error_response("Loading model", ERROR_TYPE_UNAVAILABLE));
            }
            return false;
        }
        return true;
    };
```
I've not used httplib.h before and I'm not 100% sure about how request are
processed and how these handler, like the one above are called. So lets set
a breakpoint in httplib.h Server::process_request:
```console
(gdb) br httplib.h:7133
Breakpoint 2 at 0x555555608119: file /home/danbev/work/ai/llama.cpp-debug/examples/server/httplib.h, line 7133.
```
With that done we need to call the server and we can do that using curl like
we showed earlier.

So, that should hit our breakpoint. One thing to keep in mind is that there
will be multiple threads running and we can disable the other threads by:
```console
(gdb) set scheduler-locking on
```
```c++
  // Routing
  auto routed = false;
#ifdef CPPHTTPLIB_NO_EXCEPTIONS
  routed = routing(req, res, strm);
#else
```
```cpp
inline bool Server::routing(Request &req, Response &res, Stream &strm) {
  if (pre_routing_handler_ &&
      pre_routing_handler_(req, res) == HandlerResponse::Handled) {
    return true;
  }
  ...
```
This handler is registered in server.cpp:
```cpp
    // register server middlewares
    svr->set_pre_routing_handler([&middleware_validate_api_key, &middleware_server_state](const httplib::Request & req, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
        // If this is OPTIONS request, skip validation because browsers don't include Authorization header
        if (req.method == "OPTIONS") {
            res.set_header("Access-Control-Allow-Credentials", "true");
            res.set_header("Access-Control-Allow-Methods",     "GET, POST");
            res.set_header("Access-Control-Allow-Headers",     "*");
            res.set_content("", "text/html"); // blank response, no data
            return httplib::Server::HandlerResponse::Handled; // skip further processing
        }
        if (!middleware_server_state(req, res)) {
            return httplib::Server::HandlerResponse::Handled;
        }
        if (!middleware_validate_api_key(req, res)) {
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    auto middleware_server_state = [&res_error, &state](const httplib::Request & req, httplib::Response & res) {
        server_state current_state = state.load();
        if (current_state == SERVER_STATE_LOADING_MODEL) {
            auto tmp = string_split<std::string>(req.path, '.');
            if (req.path == "/" || tmp.back() == "html") {
                res.set_content(reinterpret_cast<const char*>(loading_html), loading_html_len, "text/html; charset=utf-8");
                res.status = 503;
            } else {
                res_error(res, format_error_response("Loading model", ERROR_TYPE_UNAVAILABLE));
            }
            return false;
        }
        return true;
    };
```
So if the request is a CORS (Cross-Origin Resource Sharing) request then it
will be handled and a response sent back to the browser agent. So this will
then enter the `middleware_server_state` handler which will check the model
is still loading and in that case return with an unavailable error.
So that will then return us to `Server::process_request`:
```cpp
  // Regular handler
  if (req.method == "GET" || req.method == "HEAD") {
    return dispatch_request(req, res, get_handlers_);
  } else if (req.method == "POST") {
    return dispatch_request(req, res, post_handlers_);
  } else if (req.method == "PUT") {
    return dispatch_request(req, res, put_handlers_);
  } else if (req.method == "DELETE") {
    return dispatch_request(req, res, delete_handlers_);
  } else if (req.method == "OPTIONS") {
    return dispatch_request(req, res, options_handlers_);
  } else if (req.method == "PATCH") {
    return dispatch_request(req, res, patch_handlers_);
  }

  res.status = StatusCode::BadRequest_400;
  return false;
```
```console
(gdb) p post_handlers_.size()
$10 = 18
```
These are handlers that are registered in server.cpp:
```cpp
    // register API routes
    svr->Get ("/health",              handle_health); // public endpoint (no API key check)
    svr->Get ("/metrics",             handle_metrics);
    svr->Get ("/props",               handle_props);
    svr->Post("/props",               handle_props_change);
    svr->Get ("/models",              handle_models); // public endpoint (no API key check)
    svr->Get ("/v1/models",           handle_models); // public endpoint (no API key check)
    svr->Post("/completion",          handle_completions); // legacy
    svr->Post("/completions",         handle_completions);
    ...
```
```cpp
    const auto handle_completions = [&handle_completions_impl](const httplib::Request & req, httplib::Response & res) {
        json data = json::parse(req.body);
        return handle_completions_impl(
            SERVER_TASK_TYPE_COMPLETION,
            data,
            req.is_connection_closed,
            res,
            OAICOMPAT_TYPE_NONE);
    };
```
And this will call the `handle_completions_impl` function:
```cpp
    const auto handle_completions_impl = [&ctx_server, &res_error, &res_ok](
            server_task_type type,
            json & data,
            std::function<bool()> is_connection_closed,
            httplib::Response & res,
            oaicompat_type oaicompat) {
        GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

        if (ctx_server.params_base.embedding) {
            res_error(res, format_error_response("This server does not support completions. Start it without `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
            return;
        }

        auto completion_id = gen_chatcmplid();
        std::vector<server_task> tasks;

        try {
            std::vector<llama_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, data.at("prompt"), true, true);
            tasks.reserve(tokenized_prompts.size());
            for (size_t i = 0; i < tokenized_prompts.size(); i++) {
                server_task task = server_task(type);

                task.id    = ctx_server.queue_tasks.get_new_id();
                task.index = i;

                task.prompt_tokens    = std::move(tokenized_prompts[i]);
                task.params           = server_task::params_from_json_cmpl(
                                            ctx_server.ctx,
                                            ctx_server.params_base,
                                            data);
                task.id_selected_slot = json_value(data, "id_slot", -1);

                // OAI-compat
                task.params.oaicompat         = oaicompat;
                task.params.oaicompat_cmpl_id = completion_id;
                // oaicompat_model is already populated by params_from_json_cmpl

                tasks.push_back(task);
            }
        } catch (const std::exception & e) {
            res_error(res, format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
            return;
        }
        ...
```
A completion id is generated for this request  (chat completion id), and then 
the prompt is tokenized:
```console
(gdb) p tokenized_prompts
$19 = std::vector of length 1, capacity 1 = {std::vector of length 6, capacity 15 = {1, 1724, 338, 4309, 4717, 29973}}

(gdb) p type
$20 = SERVER_TASK_TYPE_COMPLETION

(gdb) ptype server_task
type = struct server_task {
    int id;
    int index;
    server_task_type type;
    int id_target;
    slot_params params;
    llama_tokens prompt_tokens;
    int id_selected_slot;
    server_task::slot_action slot_action;
    bool metrics_reset_bucket;
    std::vector<common_adapter_lora_info> set_lora;

    server_task(server_task_type);
    static slot_params params_from_json_cmpl(const llama_context *, const common_params &, const json &);
    static std::unordered_set<int> get_list_id(const std::vector<server_task> &);
}
```
After that we have the following line:
```cpp
                task.id    = ctx_server.queue_tasks.get_new_id();
```
So the `server_context` has a `queue_tasks` member which is of type
`server_queue`:
```console
(gdb) ptype ctx_server.queue_tasks
type = struct server_queue {
    int id;
    bool running;
    std::deque<server_task> queue_tasks;
    std::deque<server_task> queue_tasks_deferred;
    std::mutex mutex_tasks;
    std::condition_variable condition_tasks;
    std::function<void(server_task)> callback_new_task;
    std::function<void()> callback_update_slots;

    int post(server_task, bool);
    int post(std::vector<server_task> &, bool);
    void defer(server_task);
    int get_new_id(void);
    void on_new_task(std::function<void(server_task)>);
    void on_update_slots(std::function<void()>);
    void pop_deferred_task(void);
    void terminate(void);
    void start_loop(void);
  private:
    void cleanup_pending_task(int);
}
```
We can see that this has a double ended queue (deque), and also the 
get_new_id function.

_wip_

### index_html_gz
In server.cpp we have the following code:
```console
            // using embedded static index.html
            svr->Get("/", [](const httplib::Request & req, httplib::Response & res) {
                if (req.get_header_value("Accept-Encoding").find("gzip") == std::string::npos) {
                    res.set_content("Error: gzip is not supported by this browser", "text/plain");
                } else {
                    res.set_header("Content-Encoding", "gzip");
                    res.set_content(reinterpret_cast<const char*>(index_html_gz), index_html_gz_len, "text/html; charset=utf-8");
                }
                return false;
            });
```

Now, `index_html_gz` is a gzipped file in `tools/server/public` which is built
by `tools/server/webui/package.json`:
```console
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "analyze": "ANALYZE=1 npx vite-bundle-visualizer"
  },
```
We can inspect the vite configuration which is in `vite.config.js`:
```js
...
     llamaCppBuildPlugin() {
        ...
				try {
					const indexPath = resolve('../public/index.html');
					const gzipPath = resolve('../public/index.html.gz');

					if (!existsSync(indexPath)) {
						return;
					}

					let content = readFileSync(indexPath, 'utf-8');

					const faviconPath = resolve('static/favicon.svg');
					if (existsSync(faviconPath)) {
						const faviconContent = readFileSync(faviconPath, 'utf-8');
						const faviconBase64 = Buffer.from(faviconContent).toString('base64');
						const faviconDataUrl = `data:image/svg+xml;base64,${faviconBase64}`;

						content = content.replace(/href="[^"]*favicon\.svg"/g, `href="${faviconDataUrl}"`);

						console.log('✓ Inlined favicon.svg as base64 data URL');
					}

					content = content.replace(/\r/g, '');
					content = GUIDE_FOR_FRONTEND + '\n' + content;

					const compressed = fflate.gzipSync(Buffer.from(content, 'utf-8'), { level: 9 });

                    // because gzip header contains machine-specific info, we must remove these data from the header
                    // timestamp
					compressed[0x4] = 0;
					compressed[0x5] = 0;
					compressed[0x6] = 0;
					compressed[0x7] = 0;
					compressed[0x9] = 0;
```
This is reading the `public/index.html` file which is then gzipped and the
timestamp and OS fields are zeroed out.

So when we run `npm run build` in the `webui` directory, the `index.html` file
is built and gzipped and the resulting `index.html.gz` file is placed in the
public directory.

And then when we build `llama-server` using cmake we can see the following
in `examples/server/CMakeLists.txt`:
```cmake
set(PUBLIC_ASSETS
    index.html.gz
    loading.html
)

foreach(asset ${PUBLIC_ASSETS})
    set(input "${CMAKE_CURRENT_SOURCE_DIR}/public/${asset}")
    set(output "${CMAKE_CURRENT_BINARY_DIR}/${asset}.hpp")
    list(APPEND TARGET_SRCS ${output})
    add_custom_command(
        DEPENDS "${input}"
        OUTPUT "${output}"
        COMMAND "${CMAKE_COMMAND}" "-DINPUT=${input}" "-DOUTPUT=${output}" -P "${PROJECT_SOURCE_DIR}/scripts/xxd.cmake"
    )
    set_source_files_properties(${output} PROPERTIES GENERATED TRUE)
endforeach()
```
Notice that this is actually generating a `.hpp` file from the `.gz` file:
```console
/home/danbev/work/ai/llama.cpp-debug/build/examples/server/index.html.gz.hpp
```

This is passed to the script `xxd.cmake`:
```
# CMake equivalent of `xxd -i ${INPUT} ${OUTPUT}`
```
xxd is a hexdump/converter util and the `-i` flag is to output C-style arrays.


If we look in includes in server.cpp we find:
```cpp
#include "index.html.gz.hpp"
```

And in build/tools/server/index.html.gz.hpp we find:
```cpp
unsigned char index_html_gz[] = {0x1f,0x8b,...

unsigned int index_html_gz_len = 1207150;
```
And this is how the `index.html.gz` file is included in the server:
```cpp
    res.set_content(reinterpret_cast<const char*>(index_html_gz), index_html_gz_len, "text/html; charset=utf-8");
```

### GPU Sampling with llama-server

Currently the GPU sampling works in a similar manner to how pooling works, it
is an option function that is called in build_graph:
```c++
    // add GPU sampling layers (if any)
    llm->build_sampling(*this, params);
```
GPU samplers can be configured by creating sampler chains, where each sampler
chain is associated with a specific sequence id:
```c++
    struct llama_sampler_chain_params params = llama_sampler_chain_default_params();
    struct llama_sampler * chain = llama_sampler_chain_init(params);
    llama_sampler_chain_add(chain, llama_sampler_gpu_init_greedy());
    std::vector<llama_sampler_seq_config> sampler_configs = {
        { 0, gpu_sampler_chain }
    };
```
The struct is defined as:
```c++
    struct llama_sampler_seq_config {
        llama_seq_id           seq_id;
        struct llama_sampler * sampler;
    };
```
And these sampler configs are then passed into as context params:
```c++
        llama_context_params cparams = llama_context_default_params();
        cparams.samplers = sampler_configs.data();
        cparams.n_samplers = sampler_configs.size();
```
When the graph is built then the configured samplers will be added the
computation graph and be part of the computed graph. This is done in the
samplers _apply function which allows it to add operations/nodes to the computation 
graph.

This enables the sampling to happen fully, or partially on the GPU. The samplers
could sample a single token in which case that is what will be transferred from
the device memory to host memory after llama_decode has been called.
The sampled token can then be retrieved using:
```c++
    llama_token id = llama_get_sampled_token_ith(test_ctx.ctx, index);
```

Is it also possible to run a GPU sampler that only filters the logits and then
only the filtered logits are transferred back to the host and the sampling can
proceed on the CPU with the normal(CPU) sampler chain. In this case one configures
the CPU samplers as usual but they will now operate on already filtered logits.

Similar to the above with logits, it is possible for a GPU sampler to compute
the full probability distribution and transfer that to the host. And similar
to the logits filtering, the CPU samplers can then operate on the full
probability.


### speculative decoding
This section looks into speculative decoding in llama-server.

I used the following prompt: "What is the capital of Sweden?", and the target
model produced "The" which llama-server displays in the UI.

Now, to be clear on this and what will happen in the server is that it will
process this request like any other request initially. So it will decode the
prompt and produce a token by normal llama_decode and sampling and this sampled
token will be displayed in the UI.
What does differ for speculative decoding is that `pre_decode` does some work
to prepare for speculative decoding. But the actual speculative decoding happens
for token N+1 which is good to keep in mind when stepping through the code.

The overall processing of a request in llama-server looks something like this
(simplified):
* update_slots() 
  * pre_decode()
  * decode (calls llama_decode)
  * post_decode()


If we look in `pre_decode` we find the following related to speculative decoding:
```c++
        iterate(slots, [&](server_slot & slot) {
            ...

            generating.push_back(&slot);

            if (spec) {
                common_speculative_get_draft_params(spec.get(), slot.id).drafting = false;
                ...
```
So this is resetting drafting to false.
```console
(gdb) p common_speculative_get_draft_params(spec.get(), slot.id)
$3 = (common_speculative_draft_params &) @0xaaaabe613980: {
    drafting = false,
    n_max = -1,
    n_past = 0,
    id_last = 0,
    prompt = 0x0,
    result = 0x0,
    result_q = 0x0,
    sampling = 0x0}
```
Next we have:
```c++
    const bool use_ckpt_tgt = ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
    const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
```
```console
(gdb) p ctx_tgt_seq_rm_type
$9 = COMMON_CONTEXT_SEQ_RM_TYPE_RS
(gdb) p use_ckpt_tgt
$11 = false


(gdb) p ctx_dft_seq_rm_type
$10 = COMMON_CONTEXT_SEQ_RM_TYPE_PART
(gdb) p use_ckpt_dft
$12 = false
```
Back in `load_model` we had the following:
```c++
        ctx_tgt_seq_rm_type = common_context_can_seq_rm(ctx_tgt);
        ...

        if (ctx_dft) {
            ctx_dft_seq_rm_type = common_context_can_seq_rm(ctx_dft);
        }
```
The function `common_context_can_seq_rm` will use the passed in llama_context's
and decode two dummy tokens and then try to remove from memory to figure out what 
type of memory the target and draft model use (a bit simplified).
```c++
                                seq_id
                                  ↓
    if (!llama_memory_seq_rm(mem, 0, 1, -1)) {
                                     ↑   ↑
                                     p0  p1
        COM_TRC("%s", "the context does not support partial sequence removal\n");
        res = COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
        goto done;
    }
```

These are checkpoints for the target model and for the draft model.
```c++
enum common_context_seq_rm_type {
    COMMON_CONTEXT_SEQ_RM_TYPE_NO           = 0, // seq_rm not supported (e.g. no memory module)
    COMMON_CONTEXT_SEQ_RM_TYPE_PART         = 1, // can seq_rm partial sequences
    COMMON_CONTEXT_SEQ_RM_TYPE_FULL         = 2, // can seq_rm full sequences only
    COMMON_CONTEXT_SEQ_RM_TYPE_RS           = 3, // can seq_rm partial sequences, bounded by n_rs_seq
};
```

These are needed depending on the type of model that is being used. If we have
a recurrent model then we cannot simply remove processed tokens from its memory
as the memory is a latent hidden memory which moved forward. With standard
transformer models we can just remove processed tokens with out any such issues.
So we need to store recurrent model memory states, called checkpoints so that if
we need to reject tokens then we can restore the memory to a specific point.

Next we have:
```c++
            const int n_draft_max = slot.get_n_draft_max();
```
```console
(gdb) p n_draft_max
$13 = 130693
```
This is how much room is left in the context window.
Then if we have room we will:
```c++
                if (n_draft_max > 0) {
                    GGML_ASSERT(slot.can_speculate());

                    slot.spec_draft_q.clear();
```
This `spec_draft_q` hold draft candidates per token and this is just clearing
the old entries.

Next we have:
```c++
                    if (!slot.spec_draft.empty()) {
                        // we have a previous (partial) draft to reuse
                        if (use_ckpt_tgt) {
                            GGML_ASSERT(!slot.spec_ckpt.empty());
                        }
                    } else {
                        GGML_ASSERT(slot.spec_i_batch.empty());

```
`spec_draft` is a vector of llama_token's (token ids). In the else branch which
is the path this session takes, the `spec_i_batch` vector asserted to be empty.
This vector holds indices like row N of the targets decode output corresponds
to the draft position N-1. More on this later.

Next we will update the speculative decoding checkpoint with position information
```
                        slot.spec_ckpt.update_pos(
                                slot.prompt.n_tokens(),
                                llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id),
                                llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id));
```
```console
(gdb) p slot.spec_ckpt
$19 = {n_tokens = 377, id_task = -1, pos_min = 376, pos_max = 376, data_tgt = std::vector of length 0, capacity 0,
  data_dft = std::vector of length 0, capacity 0, data_spec = std::vector of length 0, capacity 0}
```

If we need to save check points for the draft model the following will be called:
```c++
                        if (use_ckpt_dft) {
                            slot.spec_ckpt.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                        }
```

Next the current slots (sequence) prompt tokens are stored in spec_prompt:
```c++
                        slot.spec_prompt = slot.prompt.tokens.get_text_tokens();
```
One thing to note here is that tokens_get_text_tokens only returns text tokens, 
it has a check for `LLAMA_TOKEN_NULL`:
```c++
llama_tokens server_tokens::get_text_tokens() const {
    llama_tokens res;
    res.reserve(tokens.size());
    for (llama_token t : tokens) {
        if (t != LLAMA_TOKEN_NULL) {
            res.push_back(t);
        }
    }
    return res;
}
```
And named return value optimization (NVRO) is in place here to the returned
vector will be move-assigned into `slot.spec_prompt`.

Next we have:
```c++
                        const bool spec_reject = slot.use_spec_rejection();
```
```c++
    // at temp 0 both p and q are point masses, so rejection is the same as sample-and-match
    bool use_spec_rejection() const {
        return task && task->params.sampling.temp > 0.0f;
    }
```
If temperature is 0 only one token gets probability 1.0 and the rest 0.0.  We
say that that distribution has become a the point mass, a distribution
that put 100% of its probability on a single value. Recall that rejection sampling
exist to handle the case where the draft doesn't just take its argmax but samples
probailistcially, so at temp 0 there is nothing for the rejection sampler to do
that argmax matching doesn't already do.
```console
(gdb) p slot.task->params.sampling.temp
$23 = 0.800000012
```
Next we have:
```c++
                        common_speculative_get_draft_params(spec.get(), slot.id) = {
                            /* .drafting = */ true,
                            /* .n_max    = */ n_draft_max,
                            /* .n_past   = */ slot.prompt.n_tokens(),
                            /* .id_last  = */ slot.sampled,
                            /* .prompt   = */ &slot.spec_prompt,
                            /* .result   = */ &slot.spec_draft,
                            /* .result_q = */ spec_reject ? &slot.spec_draft_q : nullptr,
                            /* .sampling = */ spec_reject ? &slot.task->params.sampling : nullptr,
                        };

                        drafting.push_back(&slot);
```
Notice that this is setting .sampling to the current tasks params.sampling when
`spec_reject` is enabled. And `drafting` was created previously in this function:
```c++
        std::vector<server_slot *> drafting;
```

Then we have:
```c++
        // generate the actual drafts (if any)
        if (!drafting.empty()) {
            queue_tasks.yield_to_queue([&]() {
                common_speculative_draft(spec.get());
            });
        }
```
```c++
void common_speculative_draft(common_speculative * spec) {
    ...
    for (auto & impl : spec->impls) {
        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft(dparams);
            impl->n_call_draft++;
        }
```
We can inspect impl here which is:
```console
(gdb) p impl->type
$34 = COMMON_SPECULATIVE_TYPE_DRAFT_MTP

(gdb) p *(void**)impl
$37 = (void *) 0xfffff3549b98 <vtable for common_speculative_impl_draft_mtp+16>

(gdb) info vtbl *impl
vtable for 'common_speculative_impl' @ 0xfffff3549b98 (subobject @ 0xaaaac12f7390):
[0]: 0xfffff2f2f4dc <common_speculative_impl_draft_mtp::~common_speculative_impl_draft_mtp()>
[1]: 0xfffff2f2f6a8 <common_speculative_impl_draft_mtp::~common_speculative_impl_draft_mtp()>
[2]: 0xfffff2f2f6d0 <common_speculative_impl_draft_mtp::begin(int, std::vector<int, std::allocator<int> > const&)>
[3]: 0xfffff2f2f818 <common_speculative_impl_draft_mtp::process(llama_batch const&)>
[4]: 0xfffff2f310b0 <common_speculative_impl_draft_mtp::draft(std::vector<common_speculative_draft_params, std::allocator<common_speculative_draft_params> >&)>
[5]: 0xfffff2f31ed8 <common_speculative_impl_draft_mtp::accept(int, unsigned short, bool)>
[6]: 0xfffff2f28ce4 <common_speculative_impl::get_state(int, std::vector<unsigned char, std::allocator<unsigned char> >&) const>
[7]: 0xfffff2f28d00 <common_speculative_impl::set_state(int, std::vector<unsigned char, std::allocator<unsigned char> > const&)>
```
So the call of the draft function will be the draft function in
common_speculative_impl_draft_mtp:
```c++
struct common_speculative_impl_draft_mtp : public common_speculative_impl {
    ...

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

```
```console
(gdb) p params.mparams.path
$43 = "/home/danbev/work/models/qwen/Qwen3.8-27B-GGUF/mtp-Qwen3.8-27B-Q4_0.gguf"
```
A little later we have:
```c++
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;

            spec_retune(smpls, smpls_cfg, llama_get_model(ctx_dft), seq_id, dp, params.probabilistic);
```
So this will iterate over all the sequences and call `spec_retune`, notice that
we can passing in the `common_params_sampling`, the draft model, the sequence id, 
the 
```console
(gdb) p smpls
$56 = std::vector of length 1, capacity 1 = {std::unique_ptr<common_sampler> = {get() = 0xaaaac1d00a90}}

(gdb) p smpls_cfg
$55 = std::vector of length 0, capacity 0

(gdb) p llama_get_model(ctx_dft)->name
$60 = "Qwen3.8-27B"

(gdb) p seq_id
$61 = 0

(gdb) p dp
$53 = (common_speculative_draft_params &) @0xaaaabe613980: {drafting = true, n_max = 130693, n_past = 377,
  id_last = 760, prompt = 0xaaaac12f5730, result = 0xaaaac12f5700, result_q = 0xaaaac12f5718,
  sampling = 0xaaaaabbbc448}

(gdb) p params.probabilistic
$62 = true
```
We can find `spec_retune` in speculative.cpp
```c++
static void spec_retune(
        std::vector<common_sampler_ptr> & smpls,
        std::vector<common_params_sampling> & cfg,
        const llama_model * model,
        llama_seq_id seq_id,
        common_speculative_draft_params & dp,
        bool probabilistic) {
    // greedy drafting leaves no candidates behind, so the verifier falls back to sample-and-match
    if (!probabilistic) {
        dp.result_q = nullptr;
    }


    if (dp.result_q == nullptr || dp.sampling == nullptr) {
        return;
    }
```
Notice that `cfg` was an empty vector above, if it is not the same size as
the samplers then it will be resized (to 1 in our case):
```c++
    if (cfg.size() != smpls.size()) {
        cfg.resize(smpls.size());
    }
```
Next we retrieve the sampler for this sequence:
```c++
    auto & cur = cfg[seq_id];
```
At this point since we just resized cfg the actual element will just be the 
a default initialized `common_params_sampling`.

Just to make this clear as there are samplers all over the places. The samplers
that are passed into this function as smpls are the draft models samplers/configs:
```c++
struct common_speculative_impl_draft_mtp : public common_speculative_impl {
    common_params_speculative_draft params;

    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;
```
So this is the draft models samplers.

Now the `common_speculative_draft_params` also has sampler _configs_, not samplers:
```c++
struct common_speculative_draft_params {
    ...

    // the target's config; only temp and seed are read, to retune the draft sampler
    const common_params_sampling * sampling = nullptr;
};
```
So like the comment says this is the target models sampler configuration.

Next we check the current sampling configuration against the target models
sampling temperature, and the same thing for the seed.
```c++
    if (cur.temp == dp.sampling->temp && cur.seed == dp.sampling->seed) {
        return;
    }
```
Now recall that cfg was initially empty and that we resized it, and added a 
default initialized `common_params_sampling`. So this is like caching the setting
and the check here is to see if those caches settings are the same as the 
target models sampling parameters. If they were the same then there is not need
to "retune".

Next, we will updated the "cache" sampling configuration and set it to the
target models temperature and seed values:
```c++
    cur.temp = dp.sampling->temp;
    cur.seed = dp.sampling->seed;
```
And then we will create a new sampling params instance to use to create/replace
the current sequences sampler:
```c++
    common_params_sampling sparams;
    sparams.no_perf  = false;
    sparams.top_k    = 10;
    sparams.temp     = cur.temp;
    sparams.seed     = cur.seed; // must be explicit, the default reseeds at random
    sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TEMPERATURE };

    smpls[seq_id].reset(common_sampler_init(model, sparams));
```
So after this the sequences sampler will have been updated (or not if the configuration
was the same), and we have the current target models configuration stored in
cfg.

So back in the draft function we then how:
```c++
            common_sampler_reset(smpls[seq_id].get());
```
This will end up in common/samping.cpp, which will clear the ring buffer (prev)
and the call the sampler chains reset functions:
```c++
    void reset() {
        prev.clear();

        llama_sampler_reset(chain);
    }
```
After that we have:
```c++
            common_batch_add(batch, dp.id_last, dp.n_past, { seq_id }, true);
```
This will updated the batch with the token `dp.id_last` with a pos of `dp.n_past`
the sequence id array and that should output logits.
```console
(gdb) p this->params.ctx_tgt->model->vocab->pimpl->id_to_token[dp.id_last]
$100 = {text = "The", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
```
Next we have the copying of the hidden state from the target models forward pass,
so this batch will carry both tokens and embeddings.

The following will write to dst which is batch.emb + (batch.n_tokens -1) * n_embd.
And note that batch.n_tokens was incremented by common_batch_add above so it
will be the same index. 
```c++
    std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, pending_h[seq_id].data(), row_bytes);
```
Next we have:
```c++
            if (chain_heads) {
                chain_h[seq_id].assign(pending_h[seq_id].begin(), pending_h[seq_id].end());
            }
```
Now, just to remind myself of where we are as I found I lost track here.
```c++
    void draft(common_speculative_draft_params_vec & dparams) override {
        ...
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            ...
            if (chain_heads) {
                chain_h[seq_id].assign(pending_h[seq_id].begin(), pending_h[seq_id].end());
            }
        }
```
So we are iterating over all the sequences, and we are just dealing with one token,
per sequence which is the target model token that was predicted by the target
model's decode process, before this functions is called. This is setting up the
seed/anchor token from the target model for the drafting process.

There are models like Step3.5Flash that have multiple heads for drafting, for
which chain_heads will be true. In which case this will put one hidden state to
go with id_last. But is that not what we did above too when we added the 
pending_h state to the batch embeddings?  Yes, but these models each drafted token
get predicted by a different head/layer and a head has no memory of the tokens
drafted before it, they need to be able to replay the whole sequence thus far
through it. So they need not just the token id but also its paired hidden state.
So at this point we actaully have the pending_h (hidden state or the target token)
in two places, as the embedding in the batch and for chain_heads models also
in chain_h. But the embedding in the batch is transient, it only exists for the
next llama_decode. So only the first head (0) will see these embeddings. But
chain_h is persistent and survives the entire drafting round, and adds to be
each iteration. Alright lets look at the rest of this function and hopefully
this will make more sense. 

So after that "draft setup" loop we have another loop, n_drafting is a local
variable and it is incremented for each sequence that we processed above, in our
case we have 1:
```c++
        int i = 0;

        while (n_drafting > 0) {
```

The first thing that happens is a chech for chain_heads and this:
```c++
            if (chain_heads) {
                auto * mem_dft = llama_get_memory(ctx_dft);
                for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                    if (drafting[seq_id]) {
                        llama_memory_seq_rm(mem_dft, seq_id, dparams[seq_id].n_past, -1);
                    }
                }
                llama_set_nextn_layer_offset(ctx_dft, i);
            }
```
Now, we are above to run `llama_decode` which is doing to run the current draft
models computation graph.  For our current model this will be the graph built
by: (in src/models/qwen35.cpp)

```c++
std::unique_ptr<llm_graph_context> llama_model_qwen35::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}
```


```c++
llama_model_qwen4exp::graph::graph(const llama_model & model, const llm_graph_params & params) :

```


```c++
    void pre_decode() {

        iterate(slots, [&](server_slot & slot) {
            if (slot.state == SLOT_STATE_GENERATING && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
                if (!params_base.ctx_shift) {
                    // this check is redundant (for good)
                    // we should never get here, because generation should already stopped in process_token()
                    send_error(slot, "context shift is disabled", ERROR_TYPE_SERVER);
                    slot.release();
                    return;
                }
            ...
        });
```
So this will call iterate, and notice that a lambda is passed as the second
argument:
```c++
    void iterate(std::vector<server_slot> & slots, std::function<void(server_slot &)> callback) {
        for (auto & slot : slots) {
            try {
                callback(slot);
            } catch (const std::exception & e) {
                SLT_ERR(slot, "got exception: %s\n", e.what());
                send_error(slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                slot.release();
            }
        }
    }
```

### probabalistic speculative decoding
PR: https://github.com/ggml-org/llama.cpp/pull/27694

In the previous section we walked through the normal speculative decoding process
and here we are going to look at a new probablistic speculative decodeing
implementation which will not just do a greedy/argmax sampling, but instead use
probabablistic sampling to improve acceptance lenghts/rates and throughput.

If we look in server-context.cpp and its `post_decode` function we have the following
if statement where diffenent types of speculative drafting is chosen:
```c++
                std::vector<llama_token> accepted;
                if (!synth_probs.empty()) {
                    accepted = server_sample_and_accept_synth(
                            slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft,
                            synth_probs, slot.spec_synth_rng, slot.spec_is_replay);
                } else if (use_rejection) {
                    accepted = common_sampler_sample_and_accept_n_rejection(slot.smpl.get(),
                        slot.ctx_tgt,
                        slot.spec_i_batch,
                        slot.spec_draft,
                        slot.spec_draft_q,
                        slot.spec_is_replay);
```
I used the following prompt: "What is the capital of Sweden?", and the target
model produced "The" which llama-server displays in the UI.
```console
(gdb) p this->params.ctx_tgt->model->vocab->pimpl->id_to_token[dp.id_last]
$100 = {text = "The", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}

(gdb) p slot.spec.dparams[0].id_last
$21 = 760
(gdb) p slot.ctx_tgt->model->vocab->pimpl->id_to_token[760]
$22 = {text = "The", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
```

```c++
std::vector<llama_token> common_sampler_sample_and_accept_n_rejection(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, const std::vector<std::vector<llama_token_data>> & draft_q, bool is_replay, bool grammar_first) {
    GGML_ASSERT(idxs.size()    == draft.size() + 1 && "idxs.size() must be draft.size() + 1");
    GGML_ASSERT((is_replay || draft_q.size() == draft.size()) && "draft_q must have one entry per draft token");

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    ...
    size_t i = 0;
    for (; i < draft.size(); i++) {
```
So the above will iterate over draft tokens, which are what the draft model
predicted in the speculative draft function earlier. In this case it predicted
the following draft tokens:
```console
(gdb) p draft
$5 = std::vector of length 3, capacity 4 = {1156, 369, 9859}

(gdb) p ctx->model->vocab->pimpl->id_to_token[draft[0]]
$6 = {text = "Ġuser", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}

(gdb) p ctx->model->vocab->pimpl->id_to_token[draft[1]]
$7 = {text = "Ġis", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}

(gdb) p ctx->model->vocab->pimpl->id_to_token[draft[2]]
$8 = {text = "Ġasking", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}

(gdb) p idxs.size()
$13 = 4
```
Inside the loop which handles one of the above drafts we have:
```c++
        const llama_token id_tgt = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);
```
And keep in mind that this is the acceptance/rejection part of the process so
we already have the draft models output, we now need to compare this to the
target models to decide which tokens are to be accepted/rejected.

So this is going to start with idxs[i] which is 0, so we are asking the target
model to sample a token for the first output logits that it has (the latest
llama_decode which contained the draft "prefix/prompt".
```console
(gdb) p id_tgt
$26 = 1156
(gdb) p ctx->model->vocab->pimpl->id_to_token[1156]
$27 = {text = "Ġuser", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
```
Next we have:
```console
        if (is_replay) {
            common_sampler_accept(gsmpl, draft[i], true);
            result.push_back(draft[i]);
            continue;
        }
```
If a previous draft verification round only partially accepted its draft we
can't go throught the upcoming process as these tokens were already decided
during the original attempt. If we don't the later code would consumer from
gsmpl->rng which would move its stat forward, plus we risk that the outcome is
different which is also incorrect. So if we are just replaying then we accept
the token.

Next we have:
```c++
        const auto * cur_p = common_sampler_get_candidates(gsmpl, true);
        const auto & q     = draft_q[i];

        const bool masked = !grammar_first && grammar_should_apply(gsmpl);
```
Recall that grammar is a way to define what tokens are syntactically allowed to
come next. For example, if we have a json grammer we might know that certain
tokens are not valid next tokens. Masking allows us to set this tokens .logit
to -INFINITY as after any softmax it contributes 0 probability, we are masking
out the token in question.
```c++
    std::vector<llama_token_data> cand; // candidate array masked by the grammar, if there is one
    ...
        if (masked) {
            cand.assign(cur_p->data, cur_p->data + cur_p->size);

            llama_token_data_array arr = { cand.data(), cand.size(), -1, false };
            llama_sampler_apply(gsmpl->grmr, &arr);
        }
```
So in our case what we got out of common_sampler_get_candidates), which is basically
get data from the common_sampler_sample operation is:
```console
(gdb) p *cur_p
$40 = {data = 0xaaaac406c000, size = 1, selected = 0, sorted = true}
(gdb) p *cur_p->data
$42 = {id = 1156, logit = 32.1972351, p = 1}
```
So whis sampled only one candidate with a large logit of 32.1972. But imagine
that we had multiple candidated tokens, in that case they would be compied into
`cand` and then a `llama_token_data_array` would be created to store them so
that `llama_samper_apply can be applied, enabling it to check if the specific
token keeps the grammar valid. If a token breaks the grammar it will gets its
logits set to -INFINITY. Note that this updated only happends to the cand vector
entries and not the cur_p.

Next we have:
```c++
        // a candidate the grammar rejects carries no probability, whatever the target thinks
        auto p_raw = [&](size_t k) {
            return masked && cand[k].logit == -INFINITY ? 0.0f : cur_p->data[k].p;
        };

        // masking drops probability mass, so rescale what is left or the residual is over-weighted
        float p_sum = 0.0f;
        if (masked) {
            for (size_t k = 0; k < cur_p->size; ++k) {
                p_sum += p_raw(k);
            }
        }

        const float p_norm = masked && p_sum > 0.0f ? 1.0f/p_sum : 1.0f;
```
If the grammar rules out any candidates the probs left no longer sum to 1 so we
need to rescale them.
We check each cand logit (in the masked copy cand that is) for -INIFITY and if
so p_raw returns 0.0 for that candidate prob.

```console
{token_a: 0.6, token_b: 0.3, token_c: 0.1}           sum: 1.0
                  ↑
            grammar excluded
                   
{token_a: 0.6,               token_c: 0.1}           sum: 0.7
```
If we used 0.6 this would be incorrect now that token_b is not considered. Instead
we need the following values:
```console
token_a = 0.6/0.7 ≈ 0.857
token_c = 0.1/0.7 ≈ 0.143
               ↑
              p_sum
```
And this division by p_sum, is done using the reciprocl of p_sum:
```c++
        const float p_norm = masked && p_sum > 0.0f ? 1.0f/p_sum : 1.0f;
```
```console
1/p_sum ≈ p_norm
1/0.7   ≈ 1.4286

0.6 * 1.4286 ≈ 0.857
0.1 * 1.4286 ≈ 0.143
```
Next we have the draft models probability for the draft token:
```c++
        const float q_x = prob_of(q.data(), q.size(), draft[i]);
```
```console
(gdb) p q
$46 = std::vector of length 10, capacity 10 = {
{id = 1156,  logit = 31.0534611, p = 0.999769628},
{id = 3296,  logit = 22.3407536, p = 0.000164444413},
{id = 4087,  logit = 21.0080185, p = 4.33730202e-05},
{id = 4145,  logit = 19.3865623, p = 8.57097439e-06},
{id = 3134,  logit = 18.725771,  p = 4.42641522e-06},
{id = 1428,  logit = 18.5575294, p = 3.7409834e-06},
{id = 43070, logit = 18.0778046, p = 2.31549529e-06},
{id = 6511,  logit = 18.0051365, p = 2.15320097e-06},
{id = 2570,  logit = 16.9828358, p = 7.74649095e-07},
{id = 846,   logit = 16.6903572, p = 5.78206766e-07}}
(gdb) p q.size()
$47 = 10
(gdb) p draft[i]
$48 = 1156
```
And `prob_of` is static function so we will be passing in n=10, and id=1156:
```c++
static float prob_of(const llama_token_data * data, size_t n, llama_token id) {
    for (size_t k = 0; k < n; ++k) {
        if (data[k].id == id) {
            return data[k].p;
        }
    }
    return 0.0f;
}
```
So the above will iterate over 0 to 10, and check each element in q to find
the passed in token id. And if it cannot be found it returns 0;
```console
(gdb) p q_x
$53 = 0.999769628
```
Then we have p_x which is the target models probability for this same token:
```c++
        float p_x = 0.0f;
        for (size_t k = 0; k < cur_p->size; ++k) {
            if (cur_p->data[k].id == draft[i]) {
                p_x = p_of(k);
                break;
            }
        }
```
This is very similar to what we did for the draft token but notice that in this
case we also call p_of(k) to take into consideration grammar rejection and masking
and also normalization like we disussed above.
```console
(gdb) p p_x
$54 = 1
```
Next we have the following check. This is checking to see if the draft probability
for the current draft token is greater than 0, and if so it will check if the
targets probability is greater than or equal to the draft models probability.

If the target probability is greater that the draft then this will be accepted
unconditionally. The target is at least as confidant as the draft so its a keeper.

But if the target models is less confidant we will sample from the uniform
distribution and check if that sampled value is less than the target prob / draft prob:
```c++
        if (q_x > 0.0f && (p_x >= q_x || uni(gsmpl->rng) < p_x / q_x)) {
            common_sampler_accept(gsmpl, draft[i], true);
            result.push_back(draft[i]);
            continue;
        }
```
Lets say that p_x = 0.3 (target) and q_x = 0.6 (draft), this means that the draft
is proposing this token twice as often as it should be, so we would only accept
it half of the time, p/q, 0.3/0.6 = 0.5 to compensate/correct for that.
So for our case q_x > 0.0f is true so we don't execute the right hand side of the
&&, we accept this token, add the token to the result and continue with the
next token.
Next token is:
```console
(gdb) p id_tgt
$58 = 369

(gdb) p p_x
$60 = 1
(gdb) p q_x
$61 = 0.981178284
```
All the draft tokens (3 of them) enter the above if statement and just continue
so the following will look at the other case.

First we clear the residual (whats left over) vector which is of type:
```console
(gdb) ptype residual
type = std::vector<llama_token_data>
(gdb) ptype llama_token_data
type = struct llama_token_data {
    llama_token id;
    float logit;
    float p;
}
```
```c++
        residual.clear();
```
To understand what this residual is we need to consider that when we reject a
token we don't just throw it away and roll back to the previous token. We have
to emit a replacement token right there where the rejected token's position.

Lets say the initial prompt produced token T₀, and our draft model speculates
3 tokens [d₁, d₂, d₃]. So the target model will process [d₁, d₂, d₃] as a prompt.

Now, suppose d₀ is rejected, so we roll back to T₀.
```console
[t0, d₁, d₂, d₃]
  ↑  ↑
  | rejected
rollback
```
So in this case we would have run the expensive large target model's forward
pass and it would have produced 0 tokens, it is still at t0.

This would never happen with a non-speculative decoding and would be worse than
if we had just decoded a single token. We need to guarantee that speculative
decoding is always at least as fast as standard decoding so every verification
pass must emit at least one token.
So if d₁ failes the target models discards d₂ and d₃, but it corrects and replaces
d₁.
```console
[t0, d₁, d₂, d₃]
     ↑
    rejected
    corrected
    replaced

And then generation keeps going but this time anchored by d₁ (the corrected
token that is). How this is corrected/replaced is what the residual is all about.

If we were not doing probabalistic sampling and instread greedy/argmax we would
simply take argmax(p) = t₁, that is if the probability of d₁ != t₁ we would
reject d₁ and output t₁ as the replacement and reject d₂ and d₃.

But for a probabalistic model the model cannot just pick argmax, we need to
sample a token so that the final stream of text has the exact same distribution
as if the large target model had generated it alone without any speculative
draft. We have logits from the target models forward pass. So what about just
sampling a token from the targets distribution p(x) then?  
Lets take tokens A, B, and C:
```
Target wants:   A (50%), B(30%), C(20%)
Draft proposed: B
```
Target rolled the dice and rejected B. If we were to sample p(x), B still has a
30% chance of being drawn! This is now what we want as the target model just
rejected the draft models proposal (B). This would lead to the output distribution
being corrupted, the model would generate B 42% of the time instead of 30%.

The residual is the corrected distribution. The replacement token must be sampled
from the left overs from the draft (A an C in our example):
```console
                   max(0, p(x) - q(x)) 
P_residual(x) =  --------------------
                 Σ max(0, p(z) - q(z)) 
                 z
```
Lets start with the denominator:
```console
A: max(0, 0.5 - 0.2) = max(0, 0.3)  = 0.3
B: max(0, 0.3 - 0.7) = max(0, -0.4) = 0.0
C: max(0, 0.2 - 0.1) = max(0, 0.1)  = 0.1

Sum: 0.3 + 0.0 + 0.1 = 0.4
```
This is what the following code is doing:
```c++
        float sum = 0.0f;
        for (size_t k = 0; k < cur_p->size; ++k) {
            const float r = p_of(k) - prob_of(q.data(), q.size(), cur_p->data[k].id);
            if (r > 0.0f) {
                residual.push_back({ cur_p->data[k].id, 0.0f, r });
                sum += r;
            }
        }
```
`sum` is our denominator. And notice that if the probability is less than or 0.0
then it is not inlcuded in the residuals (which would happen for B in our case)
nor the sum.

```console

                   max(0, p(x) - q(x)) 
P_residual(x) =  ---------------------
                       0.4

                  max(0, p(A) - q(A))    max(0, 0.5 - 0.2)   0.3
P_residual(A) =  --------------------- = ----------------- = --- = 0.75 (75%)
                       0.4                     0.4           0.4

                  max(0, p(B) - q(B))    max(0, 0.3 - 0.7)   0.0
P_residual(B) =  --------------------- = ----------------- = --- = 0.00 (0%)
                       0.4                     0.4           0.4

                  max(0, p(C) - q(C))    max(0, 0.2 - 0.1)   0.1
P_residual(C) =  --------------------- = ----------------- = --- = 0.25 (25%)
                       0.4                     0.4           0.4

P_residual { A: 0.75, B: 0.00, C: 0.25 }
```
In our case the residual vector will only contain:
```console
residual { A: 0.75, C: 0.25 }
```

Lets take a look by forcing the residual path:
```c++
        //if (q_x > 0.0f && (p_x >= q_x || uni(gsmpl->rng) < p_x / q_x)) {
        if (false) {
            common_sampler_accept(gsmpl, draft[i], true);
            result.push_back(draft[i]);
            continue;
        }
```
```c++
        residual.clear();
        float sum = 0.0f;
        for (size_t k = 0; k < cur_p->size; ++k) {
            const float r = p_of(k) - prob_of(q.data(), q.size(), cur_p->data[k].id);
            if (r > 0.0f) {
                residual.push_back({ cur_p->data[k].id, 0.0f, r });
                sum += r;
            }
        }
```
```console
(gdb) p residual
$5 = std::vector of length 1, capacity 1 = {{id = 1156, logit = 0, p = 0.000230371952}}
(gdb) p sum
$8 = 0.000230371952
```
```c++
        llama_token id = id_tgt;
        if (sum > 0.0f) {
            // sample from [0, 1) and scale to [0, sum)
            float u = uni(gsmpl->rng) * sum;
```
uni picks a random fraction between 0.0 and 1.0. Multiplying by sum moves this
sampled fraction to be in the sum range:
```console
0.0 -----------------------------------------------------------> sum
    | token 0 (p0 | token 1 (p1) | token 2 (p2) | token 3 (p3) |
                                   ↑
                                   u
```
And lets say that u lands on the above point.
```c++
            // get the last residuals token id.
            id = residual.back().id;

            // iterate over all the residuals
            for (const auto & e : residual) {
                u -= e.p;
                if (u <= 0.0f) {
                    id = e.id;
                    break;
                }
            }
        }
```
The `u -= e.p` is subtracting from the point u above moving it in the range
[0, sum).
```console
Iteration 0 (token 0):
u -= p0
Since u was past token 0, subtracting p0 leaves u > 0.0 so we move on to the next token.

Iteration 1 (token 1)
u -= p1
Since u was past token 1, subtracting p1 leaves u > 0.0 so we move on to the next token.

Iteration 2 (token 2)
u -= p2
Now subtraction overshoots and u drops below 0.0. And this confirms that the
initial point landed inside tokens 2's segment. 
So id = token 2 and we break.
```

```console
residual = [
    { id: A, p: 0.3 },
    { id: C, p: 0.1 }
]
sum = 0.4

0.0 ------------------------ 0.3 ------------- 0.4
|       Token A (0.3)         | Token C (0.1)  |
|<------- 75% of ruler ------>|< 25% of ruler >|

float u = uni(gsmpl->rng) * sum;

id = residual.back().id; // id is initialized to C

Case 1: u langs between 0.0 and 0.3% (75%)
uni = 0.5 (randomlly selected)
u   = 0.5 * 0.4 = 0.20

Iteration 1: (e = A, e.p = 0.3)
u -= 0.3 -> u = 0.20 - 0.30 = -0.10
u <= 0.0f is true
id = A
break

Case 2: u lands between 0.3 and 0.4 (25%)
uni = 0.9
u   = 0.9 * 0.4 = 0.36

Iteration 1: (e = A, e.p = 0.3)
u -= 0.3 -> u = 0.36 - 0.30 = 0.06
u <= 0.0f is false
Token A is skipped

Iteration 2: (e = B, e.p = 0.1)
u -= 0.1 -> u = 0.06 - 0.10 = -0.04
u <= 0.0f  is true
id = C
break

Token C is selected
```


```console
(gdb) p id
$9 = 1156
```

Example draft tokens:
```console
token  |   p (target)    |     q (draft)
----------------------------------------
  A    |     0.5         |       0.2
  B    |     0.3         |       0.7
  C    |     0.2         |       0.1
```
Accepted:
```console
token  |     p/q         |     accept probability
------------------------------------------------
  A    | 0.5/0.2 = 2.5   |       1.0 (capped)
  B    | 0.3/0.7 ≈ 0.43  |       0.43
  C    | 0.2/0.1 = 2.0   |       1.0 (capped)
```
Residual:
```console
token  |     p - q       |     residual ?  (we only keep positive values)
------------------------------------------------
  A    | 0.5 - 0.2 = 0.3 |       yes
  B    | 0.3 - 0.7 = -0.4|       no (excluded)
  C    | 0.2 - 0.1 = 0.1 |       yes

Sum of residuals: 0.3 + 0.1 = 0.4
Normalized:
A gets 75%
B gets  0%
C gets 25%
```

Now, lets say we make 100 runs:
```console
Target model wants:
A: 50 runs (50%)
B: 30 runs (30%)
C: 20 runs (20%)

The draft model is biased and proposes:
A: 20 runs (20%)
B: 70 runs (70%)
C: 10 runs (10%)
```

The acceptance pass:
```console
Draft model proposes tokens according to its distribution:

Token A (drafted 20 times)
Target model want 50% but draft only gave 20%. p/q = 0.5/0.2 = 2.5 (capped at 100%)
All 20 tokens are accepted.
Sum of accepted tokens: 20

Token B (drafted 70 times)
Target model wants 30%, but draft gave 70%. p/q = 0.3/0.7 ≈ 42.86%
Out of 70 proposed by the draft model 3/7 * 70 = 30 are accepted.
The remaining 40 (of 70) are rejected.
Sum of accepted tokens: 20 + 30 = 50

Token C (drafted 10 times)
Target model wants 20%, draft have 10%. p/q = 0.2/0.1 = 2.0 (capped at 100%)
All 10 proposals of C are accepted.
Sum of accepted tokens: 20 + 30 + 10 = 60
```
So out of 100 runs we can see that 60 tokens were accepted and 40 were rejected.

The "quota" for the target for token B is already met so the 40 rejections can
only be made up from token A and C:
```console
A gets 30/40 = 75%
B gets 0/40  =  0%
C gets 10/40 = 25%
```
So whenever we get a rejection we sample from this normalized distribution:
```console
A: 75% * 40 = 30 additional tokens. Total 20 + 30 = 50 (50%)
B:  0% * 40 = 0  additional tokens. Total 30 + 0  = 30 (30%)
C: 25% * 40 = 10 additional tokens. Total 10 + 10 = 20 (20%)
```
And notice that this matches the target distribution p(x) exactly.
Now, look at the following code:
```c++
        float sum = 0.0f;
        for (size_t k = 0; k < cur_p->size; ++k) {
            // (target prob of token k) - (draft prob of token k)
            const float r = p_of(k) - prob_of(q.data(), q.size(), cur_p->data[k].id);
            if (r > 0.0f) {
                residual.push_back({ cur_p->data[k].id, 0.0f, r });
                sum += r;
            }
        }
```
```console
Iteration 1 (k = A):

p_of(A)       = 0.5
prob_of(q, A) = 0.2

r = 0.5 - 0.2 = 0.3
0.3 > 0.0f == true
residual.pushback({ id: A, p: 0.3})
sum += 0.3     running total: 0.3

Iteration 2 (k = B):

p_of(B)       = 0.3
prob_of(q, B) = 0.7

r = 0.3 - 0.7 = -0.4
-0.4 > 0.0f == false
B is skipped.

Iteration 3 (k = C):

p_of(C)       = 0.2
prob_of(q, C) = 0.1

r = 0.2 - 0.1 = 0.1
0.1 > 0.0f == true
residual.pushback({ id: C, p: 0.1})
sum += 0.3     running total: 0.4

After the look finishes:
residual = [ {id: A, p: 0.3}, {id: C, p: 0.1} ]
sum      = 0.4
```

Next we initialize id to be the current target models id_tgt
```c++
        llama_token id = id_tgt;

        if (sum > 0.0f) {
            // uni(gsmpl->rng) random draw from [0.0, 1.0)
            // * sum so that that value lands in the range [0.0, 0.4)
            float u = uni(gsmpl->rng) * sum;
            id = residual.back().id;

            for (const auto & e : residual) {
                u -= e.p;
                if (u <= 0.0f) {
                    id = e.id;
                    break;
                }
            }
        }
```



Lets say the draft proposes token B, it gets acccepted around 43% of the time
and rejected the other 57% of the time. When it gets rejected we draw from the
residual where we have a 75% chance of A, and a 25% chance of C, but never B.

```console
P(final = A) = (A proposed & accepted) +
               (B proposed and rejected & residual picks A +
               (c proposed & rejected & residual picks A (impossible as C is always accepted (1.0))
             = (0.2 * 1.0) + (0.7 * 0.57 ) + 0
             = 0.2 + 0.3
             = 0.5                           p(A) from the table above

P(final = B) = (B proposed & accepted)
             = 0.7 * 0.43
             = 0.3                           p(B) from the table above

P(final = C) = (C proposed & accepted) + (B proposed & rejected & residual picks C)
             = (0.1 * 1.0) + (0.7 * 0.57 * 0.25)
             = 0.1 + 0.1
             = 0.2                           p(C) from the table above
```


```c++

        float sum = 0.0f;
        // same loop that we have seen above
        for (size_t k = 0; k < cur_p->size; ++k) {
            const float r = p_of(k) - prob_of(q.data(), q.size(), cur_p->data[k].id);
            if (r > 0.0f) {
                residual.push_back({ cur_p->data[k].id, 0.0f, r });
                sum += r;
            }
        }
```



_wip_


So we have:
```console
row       input token      output logits
0         "The"            what comes after "The"
1         "Ġuser"          what comes after "Ġuser"
3.        "Ġasking"        what comes after "Ġasking"


Next we have the copying of the hidden state from the target models forward pass,
so this batch will carry both tokens and embeddings.

The following will write to dst which is batch.emb + (batch.n_tokens -1) * n_embd.
And note that batch.n_tokens was incremented by common_batch_add above so it
will be the same index. 
```c++
    std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, pending_h[seq_id].data(), row_bytes);
```
Next we have:
```c++
            if (chain_heads) {
                chain_h[seq_id].assign(pending_h[seq_id].begin(), pending_h[seq_id].end());
            }
```
Now, just to remind myself of where we are as I found I lost track here.
```c++
    void draft(common_speculative_draft_params_vec & dparams) override {
        ...
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            ...
            if (chain_heads) {
                chain_h[seq_id].assign(pending_h[seq_id].begin(), pending_h[seq_id].end());
            }
        }
```
So we are iterating over all the sequences, and we are just dealing with one token,
per sequence which is the target model token that was predicted by the target
model's decode process, before this functions is called. This is setting up the
seed/anchor token from the target model for the drafting process.

There are models like Step3.5Flash that have multiple heads for drafting, for
which chain_heads will be true. In which case this will put one hidden state to
go with id_last. But is that not what we did above too when we added the 
pending_h state to the batch embeddings?  Yes, but these models each drafted token
get predicted by a different head/layer and a head has no memory of the tokens
drafted before it, they need to be able to replay the whole sequence thus far
through it. So they need not just the token id but also its paired hidden state.
So at this point we actaully have the pending_h (hidden state or the target token)
in two places, as the embedding in the batch and for chain_heads models also
in chain_h. But the embedding in the batch is transient, it only exists for the
next llama_decode. So only the first head (0) will see these embeddings. But
chain_h is persistent and survives the entire drafting round, and adds to be
each iteration. Alright lets look at the rest of this function and hopefully
this will make more sense. 

So after that "draft setup" loop we have another loop, n_drafting is a local
variable and it is incremented for each sequence that we processed above, in our
case we have 1:
```c++
        int i = 0;

        while (n_drafting > 0) {
```

The first thing that happens is a chech for chain_heads and this:
```c++
            if (chain_heads) {
                auto * mem_dft = llama_get_memory(ctx_dft);
                for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                    if (drafting[seq_id]) {
                        llama_memory_seq_rm(mem_dft, seq_id, dparams[seq_id].n_past, -1);
                    }
                }
                llama_set_nextn_layer_offset(ctx_dft, i);
            }
```
Now, we are above to run `llama_decode` which is doing to run the current draft
models computation graph.  For our current model this will be the graph built
by: (in src/models/qwen35.cpp)

```c++
std::unique_ptr<llm_graph_context> llama_model_qwen35::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}
```


```c++
llama_model_qwen4exp::graph::graph(const llama_model & model, const llm_graph_params & params) :

```


```c++
    void pre_decode() {

        iterate(slots, [&](server_slot & slot) {
            if (slot.state == SLOT_STATE_GENERATING && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
                if (!params_base.ctx_shift) {
                    // this check is redundant (for good)
                    // we should never get here, because generation should already stopped in process_token()
                    send_error(slot, "context shift is disabled", ERROR_TYPE_SERVER);
                    slot.release();
                    return;
                }
            ...
        });
```
So this will call iterate, and notice that a lambda is passed as the second
argument:
```c++
    void iterate(std::vector<server_slot> & slots, std::function<void(server_slot &)> callback) {
        for (auto & slot : slots) {
            try {
                callback(slot);
            } catch (const std::exception & e) {
                SLT_ERR(slot, "got exception: %s\n", e.what());
                send_error(slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                slot.release();
            }
        }
    }
```
