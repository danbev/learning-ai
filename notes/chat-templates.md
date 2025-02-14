## Chat templates
For models that support chat most often (perhaps always) provide a chat template
with the model.

For example, if we look at the [meta-llama/Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct/blob/main/tokenizer_config.json) model, we can
see that it contains the following chat template in its tokenizer_config.json:
```
{{- bos_token }}

{%- if custom_tools is defined %}
    {%- set tools = custom_tools %}
{%- endif %}

{%- if not tools_in_user_message is defined %}
    {%- set tools_in_user_message = true %}
{%- endif %}

{%- if not date_string is defined %}
    {%- set date_string = "26 Jul 2024" %}
{%- endif %}

{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}

{# This block extracts the system message, so we can slot it into the right place. #}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content']|trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "" %}
{%- endif %}

{# System message + builtin tools #}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}

{%- if builtin_tools is defined or tools is not none %}
    {{- "Environment: ipython\n" }}
{%- endif %}

{%- if builtin_tools is defined %}
    {{- "Tools: " + builtin_tools | reject('equalto', 'code_interpreter') | join(", ") + "\n\n"}}
{%- endif %}

{{- "Cutting Knowledge Date: December 2023\n" }}
{{- "Today Date: " + date_string + "\n\n" }}

{%- if tools is not none and not tools_in_user_message %}
    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
{%- endif %}

{{- system_message }}
{{- "<|eot_id|>" }}

{# Custom tools are passed in a user message with some extra guidance #}
{%- if tools_in_user_message and not tools is none %}
    {# Extract the first user message so we can plug it in here #}
    {%- if messages | length != 0 %}
        {%- set first_user_message = messages[0]['content']|trim %}
        {%- set messages = messages[1:] %}
    {%- else %}
        {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}
    {%- endif %}

    {{- '<|start_header_id|>user<|end_header_id|>\n\n' -}}
    {{- "Given the following functions, please respond with a JSON for a function call " }}
    {{- "with its proper arguments that best answers the given prompt.\n\n" }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}

    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}

    {{- first_user_message + "<|eot_id|>"}}
{%- endif %}

{%- for message in messages %}
    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}

    {%- elif 'tool_calls' in message %}
        {%- if not message.tool_calls|length == 1 %}
            {{- raise_exception("This model only supports single tool-calls at once!") }}
        {%- endif %}

        {%- set tool_call = message.tool_calls[0].function %}

        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}
            {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
            {{- "<|python_tag|>" + tool_call.name + ".call(" }}
            {%- for arg_name, arg_val in tool_call.arguments | items %}
                {{- arg_name + '="' + arg_val + '"' }}
                {%- if not loop.last %}
                    {{- ", " }}
                {%- endif %}
            {%- endfor %}
            {{- ")" }}
        {%- else  %}
            {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
            {{- '{"name": "' + tool_call.name + '", ' }}
            {{- '"parameters": ' }}
            {{- tool_call.arguments | tojson }}
            {{- "}" }}
        {%- endif %}

        {%- if builtin_tools is defined %}
            {# This means we're in ipython mode #}
            {{- "<|eom_id|>" }}
        {%- else %}
            {{- "<|eot_id|>" }}
        {%- endif %}

    {%- elif message.role == "tool" or message.role == "ipython" %}
        {{- "<|start_header_id|>ipython<|end_header_id|>\n\n" }}
        {%- if message.content is mapping or message.content is iterable %}
            {{- message.content | tojson }}
        {%- else %}
            {{- message.content }}
        {%- endif %}
        {{- "<|eot_id|>" }}
    {%- endif %}
{%- endfor %}

{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}
```
Lets take a look at how this works with `llama-cli`:
```c++
    const llama_vocab * vocab = llama_model_get_vocab(model);
    auto chat_templates = common_chat_templates_from_model(model, params.chat_template);
```
```c++
common_chat_templates common_chat_templates_from_model(const struct llama_model * model, const std::string & chat_template_override)
{
    std::string default_template_src;
    std::string template_tool_use_src;

    bool has_explicit_template = !chat_template_override.empty();
    if (chat_template_override.empty()) {
        auto str = llama_model_chat_template(model, /* name */ nullptr);
        if (str) {
            default_template_src = str;
            has_explicit_template = true;
        }
        str = llama_model_chat_template(model, /* name */ "tool_use");
        if (str) {
            template_tool_use_src = str;
            has_explicit_template = true;
        }
    } else {
```
This will call `llama_model_chat_template` which looks like this:
```c++
const char * llama_model_chat_template(const struct llama_model * model, const char * name) {
    const auto key = name ? LLM_KV(model->arch, name)(LLM_KV_TOKENIZER_CHAT_TEMPLATE_N)
        : LLM_KV(model->arch)(LLM_KV_TOKENIZER_CHAT_TEMPLATE);
    const auto & it = model->gguf_kv.find(key);
    if (it == model->gguf_kv.end()) {
        return nullptr;
    }

    return it->second.c_str();
}
```
Now, it looks like a model can have a chat template without a name and also
a named chat template:
```c++
    { LLM_KV_TOKENIZER_CHAT_TEMPLATE,        "tokenizer.chat_template"                 },
    { LLM_KV_TOKENIZER_CHAT_TEMPLATE_N,      "tokenizer.chat_template.%s"              },
````
In our case the name is nullptr so we will look for `tokenizer.chat_template`.
And notice that this string is copied into `default_template_src`.

Notice that after this there is a second call looking for `tool_use`:
```c++
        str = llama_model_chat_template(model, /* name */ "tool_use");
        if (str) {
            template_tool_use_src = str;
            has_explicit_template = true;
        }
```
So if there is a `tokenizer.chat.template.tool_use` in this model then it will
be used instead of the default one. This is not the case in this example to
str will be nullptr so `default_tool_use_src` will be empty.
After that we have a check if the default template src is empty of if it is
just the string "chatml". So the template can be an actual template of just
the string chatml:
```c++
    if (default_template_src.empty() || default_template_src == "chatml") {
        if (!template_tool_use_src.empty()) {
            default_template_src = template_tool_use_src;
        } else {
            default_template_src = CHATML_TEMPLATE_SRC;
        }
    }

#define CHATML_TEMPLATE_SRC \
    "{%- for message in messages -%}\n" \
    "  {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' -}}\n" \
    "{%- endfor -%}\n" \
    "{%- if add_generation_prompt -%}\n" \
    "  {{- '<|im_start|>assistant\n' -}}\n" \
    "{%- endif -%}"
```
After that we have:
```c++
    auto vocab = llama_model_get_vocab(model);
    const auto get_token = [&](llama_token token, const char * name, const char * jinja_variable_name) {
        if (token == LLAMA_TOKEN_NULL) {
            if (default_template_src.find(jinja_variable_name) != std::string::npos
                || template_tool_use_src.find(jinja_variable_name) != std::string::npos) {
                LOG_WRN("%s: warning: vocab does not have a %s token, jinja template won't work as intended.\n", __func__, name);
            }
            return std::string();
        } else {
            return common_token_to_piece(vocab, token, true);
        }
    };
```
So the lambda will perform a check that the token is not null, that is the token
is not in the models vocabulary. If this token is used in the template then it
will not work which is why it is checking the templates to see if it is used.
And notice that an empty string is returned.

Next the string representation of biginning of sequence token is fetched from
the vocabulary using the `get_token` lambda:
```c++
    auto token_bos = get_token(llama_vocab_bos(vocab), "BOS", "bos_token");
    auto token_eos = get_token(llama_vocab_eos(vocab), "EOS", "eos_token");
```
```console
(gdb) p token_bos
$32 = "<|begin_of_text|>"

(gdb) p token_eos
$33 = "<|eot_id|>"
```
Now, recall that the return type of this function is `common_chat_templates`,
so this is what will be returned below:
```c++
namespace minja {
    class chat_template;
}

typedef minja::chat_template common_chat_template;

struct common_chat_templates {
    bool has_explicit_template; // Model had builtin template or template overridde was specified.
    std::unique_ptr<common_chat_template> template_default; // always set (defaults to chatml)
    std::unique_ptr<common_chat_template> template_tool_use;
};
```
The actual implementation of the chat template can be found in 

```c++
    try {
        return {
            has_explicit_template,
            std::make_unique<minja::chat_template>(default_template_src, token_bos, token_eos),
            template_tool_use_src.empty()
                ? nullptr
                : std::make_unique<minja::chat_template>(template_tool_use_src, token_bos, token_eos),
        };
    } catch (const std::exception & e) {
        LOG_ERR("%s: failed to parse chat template: %s\n", __func__, e.what());
        return {
            has_explicit_template,
            std::make_unique<minja::chat_template>(CHATML_TEMPLATE_SRC, token_bos, token_eos),
            nullptr,
        };
    }
```
So the first value of the struct is the bool, following that is a unique pointer
which will call the constructor in `common/chat-template.hpp`:
```c++
    chat_template(const std::string & source, const std::string & bos_token, const std::string & eos_token)
        : source_(source), bos_token_(bos_token), eos_token_(eos_token)
    {
       ...
```
And that will return us back to main.cpp:
```c++
    const bool has_chat_template = chat_templates.has_explicit_template && chat_templates.template_default;
    if (params.conversation_mode == COMMON_CONVERSATION_MODE_AUTO) {
        if (has_chat_template) {
            LOG_INF("%s: chat template is available, enabling conversation mode (disable it with -no-cnv)\n", __func__);
            params.conversation_mode = COMMON_CONVERSATION_MODE_ENABLED;
        } else {
            params.conversation_mode = COMMON_CONVERSATION_MODE_DISABLED;
        }
    }

    if (params.conversation_mode) {
        if (params.enable_chat_template) {
            LOG_INF("%s: chat template example:\n%s\n", __func__, common_chat_format_example(*chat_templates.template_default, params.use_jinja).c_str());
        } else {
            LOG_INF("%s: in-suffix/prefix is specified, chat template will be disabled\n", __func__);
        }
    }
```
```console
(gdb)
main: chat template example:
<|start_header_id|>system<|end_header_id|>

You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there<|eot_id|><|start_header_id|>user<|end_header_id|>

How are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```
```c++
struct common_chat_msg {
    std::string role;
    std::string content;
    std::vector<common_tool_call> tool_calls;
    std::string reasoning_content = "";
};

    std::vector<common_chat_msg> chat_msgs;
    ...
    auto chat_add_and_format = [&chat_msgs, &chat_templates](const std::string & role, const std::string & content) {
        common_chat_msg new_msg{role, content, {}};
        auto formatted = common_chat_format_single(*chat_templates.template_default, chat_msgs, new_msg, role == "user", g_params->use_jinja);
        chat_msgs.push_back({role, content, {}});
        LOG_DBG("formatted: '%s'\n", formatted.c_str());
        return formatted;
    };

    {
        auto prompt = (params.conversation_mode && params.enable_chat_template)
            // format the system prompt in conversation mode (fallback to default if empty)
            ? chat_add_and_format("system", params.prompt.empty() ? DEFAULT_SYSTEM_MESSAGE : params.prompt)
            // otherwise use the prompt as is
            : params.prompt;
        if (params.interactive_first || !params.prompt.empty() || session_tokens.empty()) {
            LOG_DBG("tokenize the prompt\n");
            embd_inp = common_tokenize(ctx, prompt, true, true);
        } else {
            LOG_DBG("use session tokens\n");
            embd_inp = session_tokens;
        }

        LOG_DBG("prompt: \"%s\"\n", prompt.c_str());
        LOG_DBG("tokens: %s\n", string_from(ctx, embd_inp).c_str());
    }
```
```console
(gdb) p content
$38 = "\"What is LoRA?\""
(gdb) p role
$39 = "system"
(gdb) p new_msg
$40 = {
  role = "system",
  content = "\"What is LoRA?\"",
  tool_calls = std::vector of length 0, capacity 0,
  reasoning_content = ""
}
```

```c++
std::string common_chat_format_single(
        const common_chat_template & tmpl,
        const std::vector<common_chat_msg> & past_msg,
        const common_chat_msg & new_msg,
        bool add_ass,
        bool use_jinja) {
    std::ostringstream ss;
    auto fmt_past_msg = past_msg.empty() ? "" : common_chat_apply_template(tmpl, past_msg, false, use_jinja);
    std::vector<common_chat_msg> chat_new(past_msg);
    // if the past_msg ends with a newline, we must preserve it in the formatted version
    if (add_ass && !fmt_past_msg.empty() && fmt_past_msg.back() == '\n') {
        ss << "\n";
    };
    // format chat with new_msg
    chat_new.push_back(new_msg);
    auto fmt_new_msg = common_chat_apply_template(tmpl, chat_new, add_ass, use_jinja);
    // get the diff part
    ss << fmt_new_msg.substr(fmt_past_msg.size(), fmt_new_msg.size() - fmt_past_msg.size());
    return ss.str();
}
```
```console
(gdb) p formatted
$43 = "<|start_header_id|>system<|end_header_id|>\n\n\"What is LoRA?\"<|eot_id|>"
```
And this will then be set as the prompt and tokenized just like if there was
no template and just the prompt from this point onwards.

So lets try a chat template which can be found in `models/templates.
```console
$ gdb --args ./build/bin/llama-cli -m ../llama.cpp/models/Meta-Llama-3.1-8B-Instruct-Q3_K_S.gguf --no-warmup --prompt '"What is LoRA?"' -ngl 40 --chat-template-file models/templates/meta-llama-Llama-3.1-8B-Instruct.jinja
(gdb) br main.cpp:161
Breakpoint 1 at 0xe6d7c: file /home/danbev/work/ai/llama.cpp-debug/examples/main/main.cpp, line 161.
```
This time there will a a `chat_template` which will be the contents for
`meta-llama-Llamam-3.1-8B-Instruct.jinja` and it will be passed into:
```c++
    auto chat_templates = common_chat_templates_from_model(model, params.chat_template);
```

I think that `chat-template.hpp` and `minja.hpp` come from
https://github.com/google/minja.


### "response_format" on the OpenAI compatible "v1/chat/completions" issue
https://github.com/ggerganov/llama.cpp/issues/11847

```console
{
    "error": {
        "code": 400,
        "message": "Either \"json_schema\" or \"grammar\" can be specified, but not both",
        "type": "invalid_request_error"
    }
}
```
The server log looks like this:
```console
srv  log_server_r: response: {"error":{"code":400,"message":"Either \"json_schema\" or \"grammar\" can be specified, but not both","type":"invalid_request_error"}}
```

If we take a look at this request processing on the server we can look at
this handler:
```c++
    const auto handle_chat_completions = [&ctx_server, &params, &res_error, &handle_completions_impl](const httplib::Request & req, httplib::Response & res) {
        LOG_DBG("request: %s\n", req.body.c_str());
        if (ctx_server.params_base.embedding) {
            res_error(res, format_error_response("This server does not support completions. Start it without `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
            return;
        }

        auto body = json::parse(req.body);
        json data = oaicompat_completion_params_parse(body, params.use_jinja, params.reasoning_format, ctx_server.chat_templates);

        return handle_completions_impl(
            SERVER_TASK_TYPE_COMPLETION,
            data,
            req.is_connection_closed,
            res,
            OAICOMPAT_TYPE_CHAT);
    };
```
We can inspect the body from the request:
```console
(gdb) pjson body
{
    "model": "llama-2-7b-chat",
    "messages": [
        {
            "role": "user",
            "content": "hello"
        }
    ],
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "chat_response",
            "strict": true,
            "schema": {
                "type": "object",
                "properties": {
                    "response": {
                        "type": "string"
                    }
                },
                "required": [
                    "response"
                ],
                "additionalProperties": false
            }
        }
    }
}
```
And this looks good and there is no `grammar` attribute in the body.

Next we have the call to:
```c++
        json data = oaicompat_completion_params_parse(body, params.use_jinja, params.reasoning_format, ctx_server.chat_templates);
```
And if we inspect the data after this call we do see the `grammar` attribute:
```console
(gdb) pjson data | shell jq
{
    "stop": [],
    "json_schema": {
        "type": "object",
        "properties": {
            "response": {
                "type": "string"
            }
        },
        "required": [
            "response"
        ],
        "additionalProperties": false
    },
    "chat_format": 1,
    "prompt": "<|im_start|>system\nRespond in JSON format, either with `tool_call` (a request to call tools) or with `response` reply to the user's request<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n",
    "grammar": "alternative-0 ::= \"{\" space alternative-0-tool-call-kv \"}\" space\nalternative-0-tool-call ::= \nalternative-0-tool-call-kv ::= \"\\\"tool_call\\\"\" space \":\" space alternative-0-tool-call\nalternative-1 ::= \"{\" space alternative-1-response-kv \"}\" space\nalternative-1-response ::= \"{\" space alternative-1-response-response-kv \"}\" space\nalternative-1-response-kv ::= \"\\\"response\\\"\" space \":\" space alternative-1-response\nalternative-1-response-response-kv ::= \"\\\"response\\\"\" space \":\" space string\nchar ::= [^\"\\\\\\x7F\\x00-\\x1F] | [\\\\] ([\"\\\\bfnrt] | \"u\" [0-9a-fA-F]{4})\nroot ::= alternative-0 | alternative-1\nspace ::= | \" \" | \"\\n\" [ \\t]{0,20}\nstring ::= \"\\\"\" char* \"\\\"\" space\n",
    "grammar_lazy": false,
    "grammar_triggers": [],
    "preserved_tokens": [],
    "model": "llama-2-7b-chat",
    "messages": [
        {
            "role": "user",
            "content": "hello"
        }
    ],
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "chat_response",
            "strict": true,
            "schema": {
                "type": "object",
                "properties": {
                    "response": {
                        "type": "string"
                    }
                },
                "required": [
                    "response"
                ],
                "additionalProperties": false
            }
        }
    }
}
```
If we look in oaicompat_completion_params_parse we can see the following:
```c++
    // Apply chat template to the list of messages
    if (use_jinja) {
        auto tool_choice = json_value(body, "tool_choice", std::string("auto"));
        if (tool_choice != "none" && tool_choice != "auto" && tool_choice != "required") {
            throw std::runtime_error("Invalid tool_choice: " + tool_choice);
        }
        if (tool_choice != "none" && llama_params.contains("grammar")) {
            throw std::runtime_error("Cannot use custom grammar constraints with tools.");
        }
        common_chat_inputs inputs;
        inputs.extract_reasoning   = reasoning_format != COMMON_REASONING_FORMAT_NONE;
        inputs.messages            = body.at("messages");
        inputs.tools               = tools;
        inputs.tool_choice         = tool_choice;
        inputs.parallel_tool_calls = json_value(body, "parallel_tool_calls", false);
        if (inputs.parallel_tool_calls && !tmpl.original_caps().supports_parallel_tool_calls) {
            LOG_DBG("Disabling parallel_tool_calls because the template does not support it\n");
            inputs.parallel_tool_calls = false;
        }
        inputs.stream = stream;
        // TODO: support mixing schema w/ tools beyond generic format.
        inputs.json_schema = json_value(llama_params, "json_schema", json());
        auto chat_params = common_chat_params_init(tmpl, inputs);

        llama_params["chat_format"] = static_cast<int>(chat_params.format);
        llama_params["prompt"] = chat_params.prompt;
        llama_params["grammar"] = chat_params.grammar;
        llama_params["grammar_lazy"] = chat_params.grammar_lazy;
        auto grammar_triggers = json::array();
        for (const auto & trigger : chat_params.grammar_triggers) {
            grammar_triggers.push_back({
                {"word", trigger.word},
                {"at_start", trigger.at_start},
            });
        }
        llama_params["grammar_triggers"] = grammar_triggers;
```
And if we inspect the `chat_params` we can see that the `grammar` attribute is
there:
```console
(gdb) p chat_params.grammar
$2 = "alternative-0 ::= \"{\" space alternative-0-tool-call-kv \"}\" space\nalternative-0-tool-call ::= \nalternative-0-tool-call-kv ::= \"\\\"tool_call\\\"\" space \":\" space alternative-0-tool-call\nalternative-1 ::= \""...
```
Perhaps the grammer should be conditioned on the json_schama:
```c++
        if (inputs.json_schema == nullptr) {
            llama_params["grammar"] = chat_params.grammar;
            llama_params["grammar_lazy"] = chat_params.grammar_lazy;
            auto grammar_triggers = json::array();
            for (const auto & trigger : chat_params.grammar_triggers) {
                grammar_triggers.push_back({
                    {"word", trigger.word},
                    {"at_start", trigger.at_start},
                });
            }
            llama_params["grammar_triggers"] = grammar_triggers;
        }
```
