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

Now, `index_html_gz` gzipped file in `examples/server/public` which is built
by `examples/server/webui/package.json`:
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
      writeBundle() {
        const outputIndexHtml = path.join(config.build.outDir, 'index.html');
        const content = GUIDE_FOR_FRONTEND + '\n' + fs.readFileSync(outputIndexHtml, 'utf-8');
        const compressed = zlib.gzipSync(Buffer.from(content, 'utf-8'), { level: 9 });

        // because gzip header contains machine-specific info, we must remove these data from the header
        // timestamp
        compressed[0x4] = 0;
        compressed[0x5] = 0;
        compressed[0x6] = 0;
        compressed[0x7] = 0;
        // OS
        compressed[0x9] = 0;
```
This is reading the `webui/index.html` file and prepending `GUIDE_FOR_FRONTEND`
warning to it. This is then gzipped and the timestamp and OS fields are zeroed
out.
So when we run `npm run build` in the `webui` directory, the `index.html` file
is built and gzipped and the resulting `index.html.gz` file is copied to the

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
Notice that this is actually generateing a `.hpp` file from the `.gz` file:
```console
/home/danbev/work/ai/llama.cpp-debug/build/examples/server/index.html.gz.hpp
```

Now, this is passed to the script `xxd.cmake`:
```
# CMake equivalent of `xxd -i ${INPUT} ${OUTPUT}`
```
xxd is a hexdump/converter util and the `-i` flag is to output C-style arrays.


If we look in includes in server.cpp we find:
```cpp
#include "index.html.gz.hpp"
```

```cpp
unsigned char index_html_gz[] = {0x1f,0x8b,...

unsigned int index_html_gz_len = 1207150;
```
And this is how the `index.html.gz` file is included in the server:
```cpp
    res.set_content(reinterpret_cast<const char*>(index_html_gz), index_html_gz_len, "text/html; charset=utf-8");
```

### Slots
This section aims to explain what slots are in the context of llama-server.
