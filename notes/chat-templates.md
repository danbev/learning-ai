## Chat templates
For models that support chat most often (perhaps always) provide a chat template
with the model. Different models are trained with different types of chat
interaction and using templates allows client program to interact with the model
in a way that is compatible with the model. So if a different model is later
used then the client program should not have to be changed as long as there is
a chat template for the new model.

So chat templates are only about input to the LLM, taking the input from the
client program transforming the input into a format that matches a format that
the model in question was trained on. If the client program later wants to use a
different model only the chat template needs to be updated to match the new
model but the rest of the client code can stay the same?

Now, just avoid any potential confusion here with grammars and schemas in
llama.cpp. These make sure that the tokens that the inference engine outputs
adhere to the grammar. I'm actually just talking about grammars here as for
json-schemas they are actually converted into a grammar using
`json_schema_to_grammar`.

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
```console
(venv) $ gdb --args build/bin/llama-cli -m ~/Downloads/mistralai_Mistral-Small-3.2-24B-Instruct-2506-IQ2_XXS.gguf -c 0 -fa --jinja -p "Test" --verbose-prompt -t 1 --no-warmup --system-prompt "You are a helpful assistent that answers in a sarcastic way"
(gdb) br main.cpp:153
Breakpoint 1 at 0x83391: file /home/danbev/work/ai/llama.cpp/tools/main/main.cpp, line 153.
```

```c++
int main(int argc, char ** argv) {
    ...

    std::vector<common_chat_msg> chat_msgs;
```
And a `common_chat_msg` looks like this:
```console
(gdb) ptype common_chat_msg
type = struct common_chat_msg {
    std::string role;
    std::string content;
    std::vector<common_chat_msg_content_part> content_parts;
    std::vector<common_chat_tool_call> tool_calls;
    std::string reasoning_content;
    std::string tool_name;
    std::string tool_call_id;

    bool empty(void) const;
    void ensure_tool_call_ids_set(std::vector<std::string> &,
        const std::function<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >()> &);
    bool operator==(const common_chat_msg &) const;
    bool operator!=(const common_chat_msg &) const;
}
```
So we can have multiple chat messages and I'm thinking that there are for different
things like perhaps one for the system prompt, one for the user and one for the
assistant (something like that).

The next chat template related code is the following:
```c++
    const llama_vocab * vocab = llama_model_get_vocab(model);
    auto chat_templates = common_chat_templates_init(model, params.chat_template);
```
```c++
common_chat_templates_ptr common_chat_templates_init(
    const struct llama_model * model,
    const std::string & chat_template_override,
    const std::string & bos_token_override,
    const std::string & eos_token_override)
{
    std::string default_template_src;
    std::string template_tool_use_src;

    bool has_explicit_template = !chat_template_override.empty();
```
So we can see that we have strings for a default template and one for too usage.
And `has_explicit_template` is set to true if we pass in a template from 
`params.chat_template`. TODO: I wonder if this is important with regards to the
bos token and eos tokens and how they are handled.
In this case the arguments to this function are (including the default parameters
that we did not pass):
```console
(gdb) s
common_chat_templates_init (model=0x555555db0b20,
                            chat_template_override="",
                            bos_token_override="",
                            eos_token_override="")
    at /home/danbev/work/ai/llama.cpp/common/chat.cpp:530
```

Next, since we did not pass in a `chat_template_override` we will will try to
load one from the model:
```c++
    if (chat_template_override.empty()) {
        GGML_ASSERT(model != nullptr);
        const auto * str = llama_model_chat_template(model, /* name */ nullptr);

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
        default_template_src = chat_template_override;
    }
```
We are passing in `nullptr` as the name so this will look up the value for the
key (from `llama-arch.cpp`):
```c++
    `{ LLM_KV_TOKENIZER_CHAT_TEMPLATE,        "tokenizer.chat_template"                 },
```
```c++
const char * llama_model_chat_template(const llama_model * model, const char * name) {
    const auto key = name ? LLM_KV(model->arch, name)(LLM_KV_TOKENIZER_CHAT_TEMPLATE)
        : LLM_KV(model->arch)(LLM_KV_TOKENIZER_CHAT_TEMPLATE);
    const auto & it = model->gguf_kv.find(key);

    if (it == model->gguf_kv.end()) {
        // one-off fix for very popular models (so we are not flooded with issues)
        // do not extend this list unless absolutely necessary
        // Mistral-Small-2503 does not have built-in chat template
        llama_vocab_pre_type pre_type = model->vocab.get_pre_type();
        if (!name && pre_type == LLAMA_VOCAB_PRE_TYPE_TEKKEN && model->layers.size() == 40) {
            return "mistral-v7-tekken";
        }

        return nullptr;
    }

    return it->second.c_str();
}
```
The string returned from this will be:
```console
(gdb) p str
$3 = 0x555555db3680 "{%- set today = strftime_now(\"%Y-%m-%d\") %}\n{%- set default_system_message = \"You are Mistral Small 3, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.\\nYou"...

To print out the complete string do:
(gdb) set print elements 0
(gdb) p str
```
So since we have a template we will set this as the `default_template_src`, and
this will also set `has_explicit_template` to true.
Following that we have and notice that we are doing the same thing but this
time passing in a name of the template; `tool_use`:
```c++
        str = llama_model_chat_template(model, /* name */ "tool_use");
        if (str) {
            template_tool_use_src = str;
            has_explicit_template = true;
        }
```
So there might be a default template and or a named template for tool use, in
both cases the is counted as an explicit template. In this session there is
not tool use template.

Following that there is a check if the `default_template_src` is empty or if it
the source is simply the string 'chatml':
```c++
    if (default_template_src.empty() || default_template_src == "chatml") {
        if (!template_tool_use_src.empty()) {
            default_template_src = template_tool_use_src;
        } else {
            default_template_src = CHATML_TEMPLATE_SRC;
        }
    }
```
Next we have the following check:
```c++
    // TODO @ngxson : this is a temporary hack to prevent chat template from throwing an error
    // Ref: https://github.com/ggml-org/llama.cpp/pull/15230#issuecomment-3173959633
    if (default_template_src.find("<|channel|>") != std::string::npos
            // search for the error message and patch it
            && default_template_src.find("in message.content or") != std::string::npos) {
        string_replace_all(default_template_src,
            "{%- if \"<|channel|>analysis<|message|>\" in message.content or \"<|channel|>final<|message|>\" in message.content %}",
            "{%- if false %}");
    }
```
So if the the current `default_template_src` contains the string
`<|channel|>` and also the string `in message.content or` then it will replace
with `{%- if false %}`.
TOOD: readup on on gpt-oss channel tags and chat template in general.

Folloing that we have:
```c++
    std::string token_bos = bos_token_override;
    std::string token_eos = eos_token_override;
    bool add_bos = false;
    bool add_eos = false;
```
And recall that the bos and eos overrides come from default parameters of which
are "" for this session. And then we can also see that `add_bos` and `add_eos`
are set to false. 
```c++
    if (model) {
        const auto * vocab = llama_model_get_vocab(model);
        const auto get_token = [&](llama_token token, const char * name, const char * jinja_variable_name) {
            if (token == LLAMA_TOKEN_NULL) {
                if (default_template_src.find(jinja_variable_name) != std::string::npos
                    || template_tool_use_src.find(jinja_variable_name) != std::string::npos) {
                    LOG_WRN("common_chat_templates_init: warning: vocab does not have a %s token, jinja template won't work as intended.\n", name);
                }
                return std::string();
            }
            return common_token_to_piece(vocab, token, true);
        };
```
So we have a lambda `get_token` which will take a token id, and name of the token
and a jinja variables name. Notice that this will first check that the variable
is used in the template and log a warning if that is not the case and return
an empty string.
And it will then call `common_token_to_piece` which will convert the token to
a `piece` of text representing the token. 
```c++
std::string common_token_to_piece(
          const struct llama_vocab * vocab,
                       llama_token   token,
                       bool          special = true);
```
So the `get_token` lambda will get the string representation of the token it
passes into it. Perhaps a name like `token_to_string` or `token_to_text` would
have been more descriptive.

And notice that this will call `llama_token_to_piece` and pass in 0 as the left
strip argument.

Next, we use that lambda to get with the string representation of the of the
bos and eos tokens:
```c++
        token_bos = get_token(llama_vocab_bos(vocab), "BOS", "bos_token");
        token_eos = get_token(llama_vocab_eos(vocab), "EOS", "eos_token");
```
```console
(gdb) p token_bos
$9 = "<s>"
(gdb) p token_eos
$10 = "</s>"
```
The we get the `add_bos` and `add_eos` fiels from the model vocabulary:
```c++
        add_bos = llama_vocab_get_add_bos(vocab);
        add_eos = llama_vocab_get_add_eos(vocab);
    }
```
```console
(gdb) p add_bos
$13 = true
(gdb) p add_eos
$14 = false
```
After that we create a unique pointer to a `common_chat_templates` and calling
the default constructor:
```c++
    common_chat_templates_ptr tmpls(new common_chat_templates());
```
This is what the struct looks like:
```c++
typedef minja::chat_template common_chat_template;

struct common_chat_templates {
    bool add_bos;
    bool add_eos;
    bool has_explicit_template; // Model had builtin template or template overridde was specified.
    std::unique_ptr<common_chat_template> template_default; // always set (defaults to chatml)
    std::unique_ptr<common_chat_template> template_tool_use;
};
```
Notice that `common_chat_template` is a typedef for `minja::chat_template`. Minja
is a header only minimal implementation of Jinja2. Notice that this contains more
than one template, hence the plural `common_chat_templates`.

Then we set the fields on the the allocated struct:
```c++
    tmpls->has_explicit_template = has_explicit_template;
    tmpls->add_bos = add_bos;
    tmpls->add_eos = add_eos;
```
And then we try to create a `chat_template` from the `default_template_src`, and
passing in the string representations for bos and eos:
```c++
    try {
        tmpls->template_default = std::make_unique<minja::chat_template>(default_template_src, token_bos, token_eos);
    } catch (const std::exception & e) {
        LOG_ERR("%s: failed to parse chat template (defaulting to chatml): %s \n", __func__, e.what());
        tmpls->template_default = std::make_unique<minja::chat_template>(CHATML_TEMPLATE_SRC, token_bos, token_eos);
    }
```
In `vendor/minja/chat-template.hpp` we have the following constructor:
```c++
    chat_template(const std::string & source, const std::string & bos_token, const std::string & eos_token)
        : source_(source), bos_token_(bos_token), eos_token_(eos_token)
    {
```
And the same is then done for the tool use template. After that the function
returns the unique pointer tmpls. And this will return us back into main.cpp.

The next template related code is the following:
```c++
    // auto enable conversation mode if chat template is available
    const bool has_chat_template = common_chat_templates_was_explicit(chat_templates.get());
    if (params.conversation_mode == COMMON_CONVERSATION_MODE_AUTO) {
        if (has_chat_template) {
            LOG_INF("%s: chat template is available, enabling conversation mode (disable it with -no-cnv)\n", __func__);
            params.conversation_mode = COMMON_CONVERSATION_MODE_ENABLED;
        } else {
            params.conversation_mode = COMMON_CONVERSATION_MODE_DISABLED;
        }
    }
```
So if the model has a chat template and the conversation mode is set to
`COMMON_CONVERSATION_MODE_AUTO` then the conversation mode will be set to
`COMMON_CONVERSATION_MODE_ENABLED`. The second check is needed as the model
might have only had a tool usage template.

Next, we have another check:
```c++
    // in case user force-activate conversation mode (via -cnv) without proper chat template, we show a warning
    if (params.conversation_mode && !has_chat_template) {
        LOG_WRN("%s: chat template is not available or is not supported. This may cause the model to output suboptimal responses\n", __func__);
    }
```

After that we have another lambda, this one takes a role and content and binds
the chat messages vector and the chat templates pointer:
```c++
    auto chat_add_and_format = [&chat_msgs, &chat_templates](
        const std::string & role, const std::string & content) {

        common_chat_msg new_msg;
        new_msg.role = role;
        new_msg.content = content;

        auto formatted = common_chat_format_single(chat_templates.get(), chat_msgs, new_msg, role == "user", g_params->use_jinja);
        chat_msgs.push_back(new_msg);
        LOG_DBG("formatted: '%s'\n", formatted.c_str());
        return formatted;
    };
```
I'll go through this lambda in more detail later when we see how it is used.
```c++
    std::string prompt;
    {
        if (params.conversation_mode && params.enable_chat_template) {
            if (!params.system_prompt.empty()) {
                // format the system prompt (will use template default if empty)
                chat_add_and_format("system", params.system_prompt);
            }
```
If a system prompt had been passed in using `--system-prompt/sys` then that
string would have been passed to the lambda `chat_add_and_format` including
the role of `system`.

But we do have a user prompt (specified using `--prompt/p`):
```c++
            if (!params.prompt.empty()) {
                // format and append the user prompt
                chat_add_and_format("user", params.prompt);
            } else {
                waiting_for_first_input = true;
            }
```
So we will be calling the lambda `chat_add_and_format` with the role of "user",
and the prompt which is simply `Test` in this case:
```c++
    auto chat_add_and_format = [&chat_msgs, &chat_templates](const std::string & role, const std::string & content) {
        common_chat_msg new_msg;
        new_msg.role = role;
        new_msg.content = content;

        auto formatted = common_chat_format_single(chat_templates.get(),
                                                   chat_msgs, new_msg,
                                                   role == "user",
                                                   g_params->use_jinja);
        chat_msgs.push_back(new_msg);
        LOG_DBG("formatted: '%s'\n", formatted.c_str());
        return formatted;
    };
```
First a new `common_chat_msg` is created:
```console
(gdb) p new_msg
$4 = {role = "", content = "", content_parts = std::vector of length 0, capacity 0,
  tool_calls = std::vector of length 0, capacity 0, reasoning_content = "", tool_name = "", tool_call_id = ""}
```
```c++
std::string common_chat_format_single(
        const struct common_chat_templates * tmpls,
        const std::vector<common_chat_msg> & past_msg,
        const common_chat_msg & new_msg,
        bool add_ass,   // add assistent as opposed to user role I think.
        bool use_jinja) {

    common_chat_templates_inputs inputs;
    inputs.use_jinja = use_jinja;
    inputs.add_bos = tmpls->add_bos;
    inputs.add_eos = tmpls->add_eos;
```
First the inputs are set. Hmm, I think I'd like to add a system message to see
how this is handled.
```c++
    std::string fmt_past_msg;
    if (!past_msg.empty()) {
        inputs.messages = past_msg;
        inputs.add_generation_prompt = false;
        fmt_past_msg = common_chat_templates_apply(tmpls, inputs).prompt;
    }
    std::ostringstream ss;
    // if the past_msg ends with a newline, we must preserve it in the formatted version
    if (add_ass && !fmt_past_msg.empty() && fmt_past_msg.back() == '\n') {
        ss << "\n";
    };
    // format chat with new_msg
    inputs.messages.push_back(new_msg);
    inputs.add_generation_prompt = add_ass;
    auto fmt_new_msg = common_chat_templates_apply(tmpls, inputs).prompt;
    // get the diff part
    ss << fmt_new_msg.substr(fmt_past_msg.size(), fmt_new_msg.size() - fmt_past_msg.size());
    return ss.str();
}
```
And the more interesting case will be after the system message has been processed
and the user message is processed.
```console
(gdb) p fmt_past_msg
$20 = "[SYSTEM_PROMPT]You are a helpful assistent that answers in a sarcastic way[/SYSTEM_PROMPT]"

(gdb) p fmt_new_msg
$21 = "[SYSTEM_PROMPT]You are a helpful assistent that answers in a sarcastic way[/SYSTEM_PROMPT][INST]Test[/INST]"
```
And notice that at the end we get the substring:
```console
(gdb) p fmt_new_msg.substr(fmt_past_msg.size(), fmt_new_msg.size() - fmt_past_msg.size())
$22 = "[INST]Test[/INST]"
```

So the entire prompt will be:
```console
(gdb) p prompt
$29 = "[SYSTEM_PROMPT]You are a helpful assistent that answers in a sarcastic way[/SYSTEM_PROMPT][INST]Test[/INST]"
```
Which amounts to 19 tokens and this is what will be sent for inference/decoding
the first time.

During an interactive session when we type a new message, for example:
```console
<bos><start_of_turn>user
Test<end_of_turn>
<start_of_turn>model
[New Thread 0x7fffb23b0000 (LWP 169710)]
[New Thread 0x7fffb0bff000 (LWP 169711)]
[New Thread 0x7fff9fe09000 (LWP 169712)]
Okay, I'm ready! I'm eager to test. Please tell me what you'd like me to test. I'm here to help!
<end_of_turn>

> What is the Capital of Sweden?
```
This will "land" us in main.cpp:
```c++
                bool another_line = true;
                do {
                    another_line = console::readline(line, params.multiline_input);
                    buffer += line;
                } while (another_line);
                ...

                    bool format_chat = params.conversation_mode && params.enable_chat_template;
                    std::string user_inp = format_chat
                        ? chat_add_and_format("user", std::move(buffer))
                        : std::move(buffer);
```
And in our case `format_chat` is true so a user message will be formatted
using the template:
```console
(gdb) p user_inp
$3 = "\n<start_of_turn>user\nWhat is the Capital of Sweden?<end_of_turn>\n<start_of_turn>model\n"
```

And this will end up in `common_chat_format_single`. Now, all past messages
in a conversation are stored in `past_msg`, for example we might have the
following:
```console
(gdb) p past_msg
$4 = std::vector of length 2, capacity 2 = {{role = "user", content = "Test", content_parts = std::vector of length 0, capacity 0,
    tool_calls = std::vector of length 0, capacity 0, reasoning_content = "", tool_name = "", tool_call_id = ""}, {
    role = "assistant",
    content = "assistant<|channel|>analysis<|message|>The user just typed \"Test\". They might be testing the system. Likely we respond with something indicating we received the test. Maybe a friendly acknowledgement.<|end|><|start|>assistant<|channel|>final<|message|>Got it! How can I assist you today?", content_parts = std::vector of length 0, capacity 0,
    tool_calls = std::vector of length 0, capacity 0, reasoning_content = "", tool_name = "", tool_call_id = ""}}
```
And our new message is:
```console
(gdb) p new_msg
$5 = (const common_chat_msg &) @0x7fffffffb3a0: {role = "user", content = "What is the capitlal of Sweden?", 
  content_parts = std::vector of length 0, capacity 0, tool_calls = std::vector of length 0, capacity 0, reasoning_content = "", 
  tool_name = "", tool_call_id = ""}
```

```c++
std::string common_chat_format_single(
        const struct common_chat_templates * tmpls,
        const std::vector<common_chat_msg> & past_msg,
        const common_chat_msg & new_msg,
        bool add_ass,
        bool use_jinja) {

    common_chat_templates_inputs inputs;
    inputs.use_jinja = use_jinja;
    inputs.add_bos = tmpls->add_bos;
    inputs.add_eos = tmpls->add_eos;

    std::string fmt_past_msg;
    if (!past_msg.empty()) {
        inputs.messages = past_msg;
        inputs.add_generation_prompt = false;
        fmt_past_msg = common_chat_templates_apply(tmpls, inputs).prompt;
    }
```
So this is passing in the past messages into template engine and it will return
the processed templates as a string. Notice that we are setting `add_generation_prompt`
to false.
We can inspect the template using:
```console
(gdb) call (void)printf("%s", tmpls->template_default->source_.c_str())
...
       {%- elif loop.last and not add_generation_prompt %}
            {#- Only render the CoT if the final turn is an assistant turn and add_generation_prompt is false #}
            {#- This is a situation that should only occur in training, never in inference. #}
            {%- if "thinking" in message %}
                {{- "<|start|>assistant<|channel|>analysis<|message|>" + message.thinking + "<|end|>" }}
            {%- endif %}

            {#- <|return|> indicates the end of generation, but <|end|> does not #}
            {#- <|return|> should never be an input to the model, but we include it as the final token #}
            {#- when training, so the model learns to emit it. #}
            {{- "<|start|>assistant<|channel|>final<|message|>" + message.content + "<|return|>" }}
```
And if we were to set `add_generation_prompt` to true then the template would
add the following:
```console
{#- Generation prompt #}                                                        
{%- if add_generation_prompt -%}                                                
<|start|>assistant
```
So we can't just set `add_generation_prompt` to true here as that will also 
cause the substring operation to fail.

We could force this different for gpt-oss doing something like this:
```console
(gdb) p fmt_new_msg.substr(fmt_past_msg.size() -3, fmt_new_msg.size() - fmt_past_msg.size())
$12 = "<|start|>user<|message|>What is the capitlal of Sweden?<|end|><|start|>assist"
```
This is only to show the issue and I don't mean to suggest that this is fix in
any way.

Lets just make sure how the formatted string that is returned is actually used
and if this really matters at all:
```c++
                    std::string user_inp = format_chat
                        ? chat_add_and_format("user", std::move(buffer))
                        : std::move(buffer);
                    const auto line_pfx = common_tokenize(ctx, params.input_prefix, false, true);
                    const auto line_inp = common_tokenize(ctx, user_inp,            false, format_chat);
                    const auto line_sfx = common_tokenize(ctx, params.input_suffix, false, true);
```
```console
(gdb) p user_inp
$13 = "tart|>user<|message|>What is the capitlal of Sweden?<|end|><|start|>assistant"
(gdb) p line_inp
$14 = std::vector of length 17, capacity 77 = {83, 497, 91, 29, 1428, 200008, 4827, 382, 290, 41415, 46006, 328, 42009, 30, 200007,
  200006, 173781}
```
So lets set user_inp to the correct value and then tokenize it:
```console
(gdb) call (char*)strcpy((char*)user_inp.data(), "<|start|>user<|message|>What is the capital of Sweden?<|end|><|start|>assistant")
$17 = 0x555570d68e40 "<|start|>user<|message|>What is the capital of Sweden?<|end|><|start|>assistant"
(gdb) p line_inp
$19 = std::vector of length 13, capacity 76 = {200006, 1428, 200008, 4827, 382, 290, 9029, 328, 42009, 30, 200007, 200006, 105782}
```
So we have the following difference in tokens:
```console
vector of length 17 {83    ,  497,     91,   29, 1428, 200008, 4827, 382,   290, 41415,  46006,    328,  42009, 30, 200007, 200006, 173781}
vector of length 13 {200006, 1428, 200008, 4827,  382,    290, 9029, 328, 42009,    30, 200007, 200006, 105782}
```
So we will be sending more tokens to the model than we need to and 20006 is a
special token and it might not be optimal for the model.


```c++
    std::ostringstream ss;
    // if the past_msg ends with a newline, we must preserve it in the formatted version
    if (add_ass && !fmt_past_msg.empty() && fmt_past_msg.back() == '\n') {
        ss << "\n";
    };
    // format chat with new_msg
    inputs.messages.push_back(new_msg);
    inputs.add_generation_prompt = add_ass;
    auto fmt_new_msg = common_chat_templates_apply(tmpls, inputs).prompt;
    // get the diff part
    ss << fmt_new_msg.substr(fmt_past_msg.size(), fmt_new_msg.size() - fmt_past_msg.size());
    return ss.str();
}

```

_wip_

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
This time there will a `chat_template` which will be the contents of
`meta-llama-Llamam-3.1-8B-Instruct.jinja` and it will be passed into:
```c++
    auto chat_templates = common_chat_templates_from_model(model, params.chat_template);
```

I think that `chat-template.hpp` and `minja.hpp` come from
https://github.com/google/minja.

### json_schema and grammar conflict in server.cpp
This was an issue that was reported in:
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

### llama-server
```console
$ gdb --args build/bin/llama-server -m ~/work/ai/models/converted/gpt-oss-20b.gguf -c 0 -fa --verbose-prompt -ngl 15 --no-warmup -sp --reasoning-format none --verbose -t 1 --threads-http 1 --jinja
```
The chat templates are initialized in  `load_model`:
```c++
    chat_templates = common_chat_templates_init(model, params_base.chat_template);
```
And this is the same function as llama-cli called which we went through above.
```c++
        try {
            common_chat_format_example(chat_templates.get(), params.use_jinja, params.default_template_kwargs);
        } catch (const std::exception & e) {
            SRV_WRN("%s: Chat template parsing error: %s\n", __func__, e.what());
            SRV_WRN("%s: The chat template that comes with this model is not yet supported, falling back to chatml. This may cause the model to output suboptimal responses\n", __func__);
            chat_templates = common_chat_templates_init(model, "chatml");
        }
```

```c++
std::string common_chat_format_example(const struct common_chat_templates * tmpls, bool use_jinja, const std::map<std::string, std::string> & chat_template_kwargs) {
    common_chat_templates_inputs inputs;
    inputs.use_jinja = use_jinja;
    inputs.add_bos = tmpls->add_bos;
    inputs.add_eos = tmpls->add_eos;
    inputs.chat_template_kwargs = chat_template_kwargs;
    auto add_simple_msg = [&](auto role, auto content) {
        common_chat_msg msg;
        msg.role = role;
        msg.content = content;
        inputs.messages.push_back(msg);
    };
    add_simple_msg("system",    "You are a helpful assistant");
    add_simple_msg("user",      "Hello");
    add_simple_msg("assistant", "Hi there");
    add_simple_msg("user",      "How are you?");
    return common_chat_templates_apply(tmpls, inputs).prompt;
}
```
So this is building up the inputs to pass to a template. We can see that it is
setting up the system message for a chat assistant. So all of the messages are
added to the messages vector in inputs and the will be applied by the template
engine.
And `common_chat_templates_apply` is something we've also seen before:
```c++
common_chat_params common_chat_templates_apply(
    const struct common_chat_templates * tmpls,
    const struct common_chat_templates_inputs & inputs)
{
    GGML_ASSERT(tmpls != nullptr);
    return inputs.use_jinja
        ? common_chat_templates_apply_jinja(tmpls, inputs)
        : common_chat_templates_apply_legacy(tmpls, inputs);
}
```

Later in `server_context::init` we have:
```c++
        oai_parser_opt = {
            /* use_jinja             */ params_base.use_jinja,
            /* prefill_assistant     */ params_base.prefill_assistant,
            /* reasoning_format      */ params_base.reasoning_format,
            /* chat_template_kwargs  */ params_base.default_template_kwargs,
            /* common_chat_templates */ chat_templates.get(),
            /* allow_image           */ mctx ? mtmd_support_vision(mctx) : false,
            /* allow_audio           */ mctx ? mtmd_support_audio (mctx) : false,
            /* enable_thinking       */ params_base.reasoning_budget != 0,
        };
```
Notice that the chat template is being passed set here.

This is later used in `handle_chat_completions`:
```c++
    const auto handle_chat_completions = [&ctx_server, &handle_completions_impl](const httplib::Request & req, httplib::Response & res) {
        LOG_DBG("request: %s\n", req.body.c_str());

        auto body = json::parse(req.body);
        std::vector<raw_buffer> files;
        json data = oaicompat_chat_params_parse(
            body,
            ctx_server.oai_parser_opt,
            files);

        handle_completions_impl(
            SERVER_TASK_TYPE_COMPLETION,
            data,
            files,
            req.is_connection_closed,
            res,
            OAICOMPAT_TYPE_CHAT);
    };
```
In `oaicompat_chat_params_parse` we can see that the chat template inputs are
set:
```c++
static json oaicompat_chat_params_parse(
    json & body, /* openai api json semantics */
    const oaicompat_parser_options & opt,
    std::vector<raw_buffer> & out_files)
{
    ...
    common_chat_templates_inputs inputs;
    inputs.messages              = common_chat_msgs_parse_oaicompat(messages);
    inputs.tools                 = common_chat_tools_parse_oaicompat(tools);
    inputs.tool_choice           = common_chat_tool_choice_parse_oaicompat(tool_choice);
    inputs.json_schema           = json_schema.is_null() ? "" : json_schema.dump();
    inputs.grammar               = grammar;
    inputs.use_jinja             = opt.use_jinja;
    inputs.parallel_tool_calls   = json_value(body, "parallel_tool_calls", false);
    inputs.add_generation_prompt = json_value(body, "add_generation_prompt", true);
    inputs.reasoning_format      = opt.reasoning_format;
    inputs.enable_thinking       = opt.enable_thinking;
    ...
    // Apply chat template to the list of messages
    auto chat_params = common_chat_templates_apply(opt.tmpls, inputs);

    /* Append assistant prefilled message */
    if (prefill_assistant_message) {
        if (!last_message.content_parts.empty()) {
            for (auto & p : last_message.content_parts) {
                chat_params.prompt += p.text;
            }
        } else {
            chat_params.prompt += last_message.content;
        }
    }

    llama_params["chat_format"]      = static_cast<int>(chat_params.format);
    llama_params["prompt"]           = chat_params.prompt;
```
Notice that this function returns a json object. And that the output of the
template engine is set as the `prompt` attribute in the json object.

Now, lets first inspect the `body`:
```console
(gdb) pjson body
{
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "What is the capital of Sweden?"
        }
    ],
    "stream": true,
    "cache_prompt": true,
    "reasoning_format": "none",
    "samplers": "edkypmxt",
    "temperature": 0.8,
    "dynatemp_range": 0,
    "dynatemp_exponent": 1,
    "top_k": 42,
    "top_p": 0.8,
    "min_p": 0.05,
    "typical_p": 1,
    "xtc_probability": 0,
    "xtc_threshold": 0.1,
    "repeat_last_n": 64,
    "repeat_penalty": 1,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "dry_multiplier": 0,
    "dry_base": 1.75,
    "dry_allowed_length": 2,
    "dry_penalty_last_n": -1,
    "max_tokens": -1,
    "timings_per_token": true
}
```
Notice that the system prompt is taken from the web UI and the settings page so
this can be changed by the user.

And this is calling `common_chat_templates_apply` which we have seen before.

Inspecting the prompt after the template has been applied we can see:
```console
(gdb) p chat_params.prompt
$11 = "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2025-08-20\n\nReasoning: medium\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>developer<|message|># Instructions\n\nYou are a helpful assistant.\n\n<|end|><|start|>user<|message|>What is the capital of Sweden?<|end|><|start|>assistant"
```
Notice that this does not have the issue that `llama-cli` has/had with regards
to the `<|start|>` token.

The returned json from this function will look like this:
```console
(gdb) pjson llama_params
{
    "stop": [],
    "chat_format": 12,
    "prompt": "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2025-08-20\n\nReasoning: medium\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>developer<|message|># Instructions\n\nYou are a helpful assistant.\n\n<|end|><|start|>user<|message|>What is the capital of Sweden?<|end|><|start|>assistant",
    "grammar_lazy": false,
    "grammar_triggers": [],
    "preserved_tokens": [
        "<|channel|>",
        "<|constrain|>",
        "<|message|>",
        "<|start|>",
        "<|end|>"
    ],
    "thinking_forced_open": false,
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "What is the capital of Sweden?"
        }
    ],
    "stream": true,
    "cache_prompt": true,
    "reasoning_format": "none",
    "samplers": "edkypmxt",
    "temperature": 0.8,
    "dynatemp_range": 0,
    "dynatemp_exponent": 1,
    "top_k": 42,
    "top_p": 0.8,
    "min_p": 0.05,
    "typical_p": 1,
    "xtc_probability": 0,
    "xtc_threshold": 0.1,
    "repeat_last_n": 64,
    "repeat_penalty": 1,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "dry_multiplier": 0,
    "dry_base": 1.75,
    "dry_allowed_length": 2,
    "dry_penalty_last_n": -1,
    "max_tokens": -1,
    "timings_per_token": true
}
```
This will be returned to server.cpp, `handle_chat_completions `:
```c++
        json data = oaicompat_chat_params_parse(
            body,
            ctx_server.oai_parser_opt,
            files);

        handle_completions_impl(
            SERVER_TASK_TYPE_COMPLETION,
            data,
            files,
            req.is_connection_closed,
            res,
            OAICOMPAT_TYPE_CHAT);
```
```c++
    const auto handle_completions_impl = [&ctx_server, &res_error, &res_ok](
            server_task_type type,
            json & data,
            const std::vector<raw_buffer> & files,
            const std::function<bool()> & is_connection_closed,
            httplib::Response & res,
            oaicompat_type oaicompat) -> void {

    ...
    try {
            std::vector<server_task> tasks;

            const auto & prompt = data.at("prompt");

            if (oaicompat && has_mtmd) {
                ...
            } else {
                auto tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, prompt, true, true);
                for (auto & p : tokenized_prompts) {
                    auto tmp = server_tokens(p, ctx_server.mctx != nullptr);
                    inputs.push_back(std::move(tmp));
                }
            }
```
```console
(gdb) p tokenized_prompts
$16 = std::vector of length 1, capacity 1 = {std::vector of length 87, capacity 416 = {200006, 17360, 200008, 3575, 553, 17554,
    162016, 11, 261, 4410, 6439, 2359, 22203, 656, 7788, 17527, 558, 87447, 100594, 25, 220, 1323, 19, 12, 3218, 198, 6576, 3521,
    25, 220, 1323, 20, 12, 3062, 12, 455, 279, 30377, 289, 25, 14093, 279, 2, 13888, 18403, 25, 8450, 11, 49159, 11, 1721, 13,
    21030, 2804, 413, 7360, 395, 1753, 3176, 13, 200007, 200006, 77944, 200008, 2, 68406, 279, 3575, 553, 261, 10297, 29186, 364,
    200007, 200006, 1428, 200008, 4827, 382, 290, 9029, 328, 42009, 30, 200007, 200006, 173781}}
```

