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
So all slot in the server will share this single llama_context, model, vocab.

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
The number slots created is determined by the --parallel command line argument 
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
this becomes n_parallel. So if we only 1 slot this means that only one request
will be processed at a time.
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

