## Jev in llama.cpp
This document will go through the various Jev (systemone) type of models in
llama.cpp.

The first OpenJev will contain a walk through and some general background/overview
of the process (including llama-server) and then I'll try to cover the specifics
of other models in separate sections.

The current models can found in [Decision models](https://huggingface.co/collections/ggml-org/decision-models)

### Processing a request
When a request is posted to llama-server the handler that will handle the request
is `post_systemone`:
```c++
    this->post_systemone = [this](const server_http_req & req) {
        auto res = create_response();
        const auto & decision = ctx_server.decision;
        if (decision.type == COMMON_DECISION_TYPE_NONE) {
            res->error(format_error_response("This model is not a decision model", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        const json body = json::parse(req.body);
        const auto questions = decision.parse_questions(body);
```
So the complete body of the request will be parsed into a json object:
```console
(gdb) pjson body
{
    "state": {
        "message": "Hi, I was charged twice for my order #4471 and I want a refund.",
        "plan": "pro",
        "order": {
            "id": 4471,
            "items": [
                "phone case",
                "charger"
            ]
        }
    },
    "questions": {
        "intent": {
            "type": "choice",
            "instructions": "What does the customer want?",
            "criteria": {
                "refund": "wants money back",
                "cancel": "wants to cancel an order",
                "track": "wants to know where an order is",
                "other": "anything else"
            }
        },
        "urgent": {
            "type": "noul",
            "instructions": "Does this need a human within the hour?"
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated is the customer?",
            "criteria": [
                "calm",
                "mildly annoyed",
                "annoyed",
                "angry"
            ]
        },
        "refund": {
            "type": "noul",
            "instructions": "Is a refund requested?",
            "criteria": {
                "true": "money back is asked",
                "false": "no money back is asked"
            }
        },
        "team": {
            "type": "choice",
            "instructions": "Which team?",
            "criteria": {
                "billing": null,
                "shipping": null,
                "technical": null,
                "sales": null,
                "legal": null,
                "returns": null,
                "fraud": null,
                "accounts": null,
                "retention": null,
                "other": null
            }
        }
    }
}
(gdb) ptype questions
type = const std::vector<server_decision_question>

(gdb) ptype server_decision_question
type = struct server_decision_question {
    std::string id;
    server_decision_question_type type;
    json instructions;
    std::vector<server_decision_option> options;
}

(gdb) pfield questions id
[0]: "intent"
[1]: "urgent"
[2]: "frustration"
[3]: "refund"
[4]: "team"

``
Then we will parse the state:
```c++
        std::vector<raw_buffer> files;
        const json state = decision.parse_state(body, files);
```
```console
(gdb) pjson state
{
    "message": "Hi, I was charged twice for my order #4471 and I want a refund.",
    "plan": "pro",
    "order": {
        "id": 4471,
        "items": [
            "phone case",
            "charger"
        ]
    }
}
```
And files is an output parameter which could contains image files if any where
used.
```c++
        auto & rd = res->rd;
        {
            std::vector<server_task> tasks;
            if (decision.is_joint()) {
                server_task task = server_task(SERVER_TASK_TYPE_DECISION);
                task.id = rd.get_new_id();
                decision.fill_task_joint(state, questions, task);
                tasks.push_back(std::move(task));
            } else {
               ...
```
Is joint means that all questions in the request are evaluated together in one
prompt and one task. Currently only Clef uses this:
```console
    bool is_joint() const {
        return type == COMMON_DECISION_TYPE_CLEF;
    }
```
```c++
            } else {
                for (const auto & question : questions) {
                    for (size_t variant = 0; variant < decision.n_variants(question); variant++) {
                        server_task task = server_task(SERVER_TASK_TYPE_DECISION);
                        task.id = rd.get_new_id();
                        decision.fill_task(state, questions, question, variant, files, ctx_server.mctx, ctx_server.init_opt, task);
                        tasks.push_back(std::move(task));
                    }
                }
            }
```
So in our case we will iterate over the 5 questions we have. We will create
on server task per variant. Only Lev currently creates a second variant and I'll explain
what this is later. And notice that decision.fill_task is called to fill the
task.

```c++
void server_decision_context::fill_task(
        const json & state,
        const std::vector<server_decision_question> & questions,
        const server_decision_question & question,
        size_t variant,
        const std::vector<raw_buffer> & files,
        mtmd_context * mctx,
        const mtmd_helper_init_opt & init_opt,
        server_task & task) const {
    const std::string prompt = render(state, questions, question, variant, files.size());
```
Render is what will "render" the prompt that will sent to the model for process
and it is per task.
```c++
std::string server_decision_context::render(
        const json & state,
        const std::vector<server_decision_question> & questions,
        const server_decision_question & question,
        size_t variant,
        size_t n_images) const {
    // the template is given raw JSON values, it serializes the ones that are not strings
    json inp = json{
        {"id",           question.id},
        {"type",         decision_question_type_name(question.type)},
        {"instructions", question.instructions},
        {"state",        state},
        {"options",      render_options(question, variant)},
    };
```
So we first have our ninja input json:
```console
(gdb) pjson inp
{
    "id": "intent",
    "type": "choice",
    "instructions": "What does the customer want?",
    "state": {
        "message": "Hi, I was charged twice for my order #4471 and I want a refund.",
        "plan": "pro",
        "order": {
            "id": 4471,
            "items": [
                "phone case",
                "charger"
            ]
        }
    },
    "options": [
        {
            "key": "refund",
            "description": "wants money back",
            "label": "A"
        },
        {
            "key": "cancel",
            "description": "wants to cancel an order",
            "label": "B"
        },
        {
            "key": "track",
            "description": "wants to know where an order is",
            "label": "C"
        },
        {
            "key": "other",
            "description": "anything else",
            "label": "D"
        }
    ]
}
```
This will be possibly be modified depending on the model type.
```c++
    jinja::context ctx(tmpl->source());
```

```console
(gdb) p type
$38 = COMMON_DECISION_TYPE_LEV

(gdb) printf "%s\n", tmpl->source().c_str()
<|im_start|>system
You are a System One decision model. You read the Evidence and answer each Criterion by choosing exactly one of the listed options. You never explain. You answer with the single option label only.<|im_end|>
<|im_start|>user
# Evidence
{{ state if state is string else state | tojson }}

# Criterion
{% if instructions %}{{ instructions if instructions is string else instructions | tojson }}{% else %}{{ id }}{% endif %}{{ '\n\n' }}{% if type == 'noul' %}{{ '# Scale\n0 = certainly no ... 8 = certainly yes\n' }}{% for o in options %}{% if o.key == 'true' and o.description %}yes: {{ o.description if o.description is string else o.description | tojson }}{{ '\n' }}{% endif %}{% endfor %}{% for o in options %}{% if o.key == 'false' and o.description %}no: {{ o.description if o.description is string else o.description | tojson }}{{ '\n' }}{% endif %}{% endfor %}{{ '\nRespond with only a digit from 0 to 8.' }}{% else %}{{ '# Options\n' }}{% for o in options %}{{ o.label }}. {% if type == 'score' %}(level {{ o.key }} of {{ options | length - 1 }}) {{ o.description if o.description is string else o.description | tojson }}{% else %}{{ o.key }}{% if o.description %}: {{ o.description if o.description is string else o.description | tojson }}{% endif %}{% endif %}{{ '\n' }}{% endfor %}{{ '\nRespond with only the letter of ' }}{% if type == 'score' %}the level that best matches.{% else %}the best option.{% endif %}{% endif %}{{ '\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n' }}
```

Next we use the ninja context and pass it the inp json from above which will
create variables in the ninja context for the template parameters.
```c++
    jinja::global_from_json(ctx, inp, false);
```
Next we create a ninja runtime for the context and execute the template:
```c++
    jinja::runtime runtime(ctx);
    const jinja::value results = runtime.execute(tmpl->prog);
    return jinja::runtime::gather_string_parts(results)->as_string().str();
}
```
So for the `intent` we will get the following prompt:
```console
(gdb) p jinja::runtime::gather_string_parts(results)->as_string().str()
$48 = "<|im_start|>system\nYou are a System One decision model. You read the Evidence and answer each Criterion by choosing exactly one of the listed options. You never explain. You answer with the single option label only.<|im_end|>\n<|im_start|>user\n# Evidence\n{\"message\": \"Hi, I was charged twice for my order #4471 and I want a refund.\", \"order\": {\"id\": 4471, \"items\": [\"phone case\", \"charger\"]}, \"plan\": \"pro\"}\n\n# Criterion\nWhat does the customer want?\n\n# Options\nA. refund: wants money back\nB. cancel: wants to cancel an order\nC. track: wants to know where an order is\nD. other: anything else\n\nRespond with only the letter of the best option.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
```
This will return us back into fill_task:
```c++
    const std::string prompt = render(state, questions, question, variant, files.size());

    if (type == COMMON_DECISION_TYPE_OPENJEV || type == COMMON_DECISION_TYPE_LEV || type == COMMON_DECISION_TYPE_NIMBLE) {
        // lev reads the ratings of a noul question at its first labels, not at the digits
        task.decision.labels.assign(labels.begin(), labels.begin() + n_outputs(question));
        if (!files.empty()) {
            task.tokens = process_mtmd_prompt(mctx, prompt, files, init_opt);
            return;
        }
    }
```
The above will copy the first labels from the labels vector to the
task.decision.labels for the number of output the specific question has. In our
case this is 4:
```console
(gdb) p n_outputs(question)
$56 = 4

(gdb) p task.decision.labels
$57 = std::vector of length 4, capacity 4 = {32, 33, 34, 35}
```
And we can use the vocab to see that these token ids map to:
```console
(gdb) p this->vocab->pimpl->id_to_token[task.decision.labels[0]]
$64 = {text = "A", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}

(gdb) p this->vocab->pimpl->id_to_token[task.decision.labels[1]]
$65 = {text = "B", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}

(gdb) p this->vocab->pimpl->id_to_token[task.decision.labels[2]]
$66 = {text = "C", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}

(gdb) p this->vocab->pimpl->id_to_token[task.decision.labels[3]]
$67 = {text = "D", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
```
After that the prompt will be tokenized:
```c++
    llama_tokens tokens = common_tokenize(vocab, prompt, false, true);
```
```console
(gdb) p tokens.size()
$68 = 173
```
And the last thing that happens in this function is where the boolean is for
`has_mtmd`:
```c++
    task.tokens = server_tokens(tokens, false);
```
server_tokens is a struct that wraps llama_tokens and provides a lot of useful
helper methods. It also provides support for images but that this not use in
this particular case.
After this function returns (fill_task) we will be back in post_systemone function
where we are iterating over the questions:
```c++
                        decision.fill_task(state, questions, question, variant, files, ctx_server.mctx, ctx_server.init_opt, task);
                        tasks.push_back(std::move(task));
```
And this will move the newly filled task the tasks vector. And this will happen
for all the questions and variants.

We then have:
```c++
            if (decision.can_share_prompt()) {
                tasks = server_decision_group_tasks(std::move(tasks), params.n_parallel);
            }
            rd.post_tasks(std::move(tasks));
```
```c++
    // true if the questions of a request start with the same tokens, and the model can continue from them
    bool can_share_prompt() const {
        switch (type) {
            case COMMON_DECISION_TYPE_OPENJEV:
            case COMMON_DECISION_TYPE_LEV:
            case COMMON_DECISION_TYPE_KEV:
            case COMMON_DECISION_TYPE_NIMBLE:
                return true;
            default:
                return false;
        }
    }
```

_wip_

## OpenJev

### Download the OpenJev model
```console
(venv) $ hf download ggml-org/OpenJev-GGUF --local-dir models/
```

### Start server
```console
model=models/OpenJev-Q8_0.gguf
gdb --args $build_dir/bin/$cmd -m $model -ngl 10 --port 8080 -c 4096
```

### Curl request
```conole
curl -s -X POST http://127.0.0.1:8080/v1/systemone \
  -H "Content-Type: application/json" \
  -d @- <<'EOF' | jq .
{
  "state": {
    "message": "Hi, I was charged twice for my order #4471 and I want a refund.",
    "plan": "pro",
    "order": {
      "id": 4471,
      "items": ["phone case", "charger"]
    }
  },
  "questions": {
    "intent": {
      "type": "choice",
      "instructions": "What does the customer want?",
      "criteria": {
        "refund": "wants money back",
        "cancel": "wants to cancel an order",
        "track": "wants to know where an order is",
        "other": "anything else"
      }
    },
    "urgent": {
      "type": "noul",
      "instructions": "Does this need a human within the hour?"
    },
    "frustration": {
      "type": "score",
      "instructions": "How frustrated is the customer?",
      "criteria": ["calm", "mildly annoyed", "annoyed", "angry"]
    },
    "refund": {
      "type": "noul",
      "instructions": "Is a refund requested?",
      "criteria": {
        "true": "money back is asked",
        "false": "no money back is asked"
      }
    },
    "team": {
      "type": "choice",
      "instructions": "Which team?",
      "criteria": {
        "billing": null,
        "shipping": null,
        "technical": null,
        "sales": null,
        "legal": null,
        "returns": null,
        "fraud": null,
        "accounts": null,
        "retention": null,
        "other": null
      }
    }
  }
}
EOF
```

```c++
void server_decision_context::fill_task(
        const json & state,
        const server_decision_question & question,
        size_t variant,
        const std::vector<raw_buffer> & files,
        mtmd_context * mctx,
        const mtmd_helper_init_opt & init_opt,
        server_task & task) const {
    const std::string prompt = render(state, question, variant, files.size());
```
```console
(gdb) p prompt
$1 = "<|im_start|>user\nState:\n
{\"message\": \"Hi, I was charged twice for my order #4471 and I want a refund.\", \"plan\": \"pro\", \"order\": {\"id\": 4471, \"items\": [\"phone case\", \"charger\"]}
}
\n\n
Question: What does the customer want?\n
Options:\n
[A] refund: wants money back\n
[B] cancel: wants to cancel an order\n
[C] track: wants to know where an order is\n
[D] other: anything else
\n\n
Answer with the letter of the best option only.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
```
So we are passing the above prompt to llama_process/llama_decode and we will just
take the last predicted logits and then select the 4 tokens we are interested in,
the probabilites for token id A, B, C, and D. And we take the one with the higest
probabilitiy. There could be other token ids with hight probabilites then these
tokens but we simply ignore them.

So this is using the normal autoregressive computation graph, same as Qwen35 but
this model has been specialized for the prompt format above and interpreting
the output to make decisions. So nothing in changes in the ggml graph iteself
for OpenJev. But for Kev which is also based on Qwen35 it does add two tensors.

### Kev
```console
(venv) $ hf download ggml-org/Kev-4B-GGUF Kev-4B-Q4_K_M.gguf --local-dir models
```

Like OpenJev this models is also based on Qwen35 and it adds two tensors
```c++
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    if (model.cls_out) {
        ggml_tensor * embd = build_lora_mm(model.cls_out, cur);
        if (model.cls_out_b) {
            embd = ggml_add(ctx0, embd, model.cls_out_b);
        }
        cb(embd, "result_embd_proj", -1);
        res->t_embd = embd;
        ggml_build_forward_expand(gf, embd);
    }

    // LM head
    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
```

These two learned projections
```console
(gdb) p model.cls_out->ne
$1 = {2560, 512, 1, 1}

(gdb) p model.cls_out_b->ne
$2 = {512, 1, 1, 1}

(gdb) p cur->ne
$3 = {2560, 4, 1, 1}

```
So this is a proj = W * hidden + bias. We are going from the 2560 to 512 and
then adding the bias.

Now Kev does not use labels like OpenJev but instead uses markers:
```console
(gdb) printf "%s\n", prompt.c_str()
<|fim_prefix|>message: Hi, I was charged twice for my order #4471 and I want a refund.
plan: pro
order:
  id: 4471
  items:
    - phone case
    - charger<|fim_middle|>What does the customer want?
<|box_start|>refund: wants money back<|box_end|>
<|box_start|>cancel: wants to cancel an order<|box_end|>
<|box_start|>track: wants to know where an order is<|box_end|>
<|box_start|>other: anything else<|box_end|>
<|fim_suffix|>
```
In server-decision.cpp we have:
```c++
void server_decision_context::fill_task(
        const json & state,
        const server_decision_question & question,
        size_t variant,
        const std::vector<raw_buffer> & files,
        mtmd_context * mctx,
        const mtmd_helper_init_opt & init_opt,
        server_task & task) const {
        ...

    if (type == COMMON_DECISION_TYPE_KEV) {
        // an option is read at its end token, the question at the last token
        for (size_t i = 0; i < tokens.size(); i++) {
            // when the token matches the <|box_end|> marker add that token index
            // to the decision markers.
            if (tokens[i] == token_marker) {
                task.decision.markers.push_back(i);
            }
        }

        if (task.decision.markers.size() != question.options.size()) {
            throw std::runtime_error("unexpected layout of the decision prompt");
        }

        task.decision.pointer = tokens.size() - 1;
    }
    task.tokens = server_tokens(tokens, false);
```
In server-context.cpp we later have:
```c++
    void send_decision(const server_slot & slot, const common_batch & batch, int32_t i_batch) {
        auto res = std::make_unique<server_task_result_decision>();
        res->id       = slot.task->id;
        res->index    = slot.task->index;
        res->n_tokens = slot.task->n_tokens();

        const auto & decision = slot.task->decision;

        if (!decision.labels.empty()) {
            ...

        } else {
            // the outputs of this slot in this batch are the last tokens of the prompt
            std::vector<int32_t> idx;
            for (int i = 0; i < batch.size(); ++i) {
                if (batch.tokens[i].output && batch.tokens[i].seq_id == slot.id) {
                    idx.push_back(i);
                }
            }
            const int32_t pos_first = slot.prompt.n_tokens() - (int32_t) idx.size();
            auto get_embd = [&](int32_t pos) -> const float * {
                const int32_t i = pos - pos_first;
                return i >= 0 && i < (int32_t) idx.size() ? llama_get_embeddings_ith(slot.ctx_tgt, idx[i]) : nullptr;
            };


            const int32_t n_embd_out = llama_model_n_embd_out(model_tgt);
            const int32_t n_pointer  = n_embd_out / 2;
            const float * embd_q = decision.pointer >= 0 ? get_embd(decision.pointer) : nullptr;
```
The embd_q
```console
(gdb) p n_embd_out
$18 = 512

(gdb) p n_pointer
$19 = 256

(gdb) p res->n_tokens
$33 = 93

(gdb) p decision.pointer
$34 = 92
```
The decision.pointer is pointing to the index in the prompt, and this is used
to resovle and lookup, using get_embd, that vector in the projected embedding
matrix. And this is the last entry in that matrix so we can think of it as the
what the model has predicted, but these are not logits remember, they are
projected embeddings of the last token. This is named as the embd_q for query
or question. This is what the model has predicted.

Next we are doing to iterate over all the markers that we stored earlier, for
example:
```console
<|box_start|>refund: wants money back<|box_end|>
```
And we will use the marker indices to look up the projected embeddings for them
from the same matrix as the query. And we want to compare them with the query
using the dot product.

`n_pointer` is 256, but each projected embedding contains 512 features. The
loop reads the first 256 elements of `embd_q` and the last 256 elements of
`embd`. This is because `cls_out` combines two projection matrices: one maps the
2560 hidden features to 256 query features, and the other maps them to 256 key
features. Their outputs are stored together as `[query | key]`. Here we compare
the query part at the final prompt position with the key part at each option's
end marker.

In conversion/lev.py we have:
```python
@ModelBase.register("KevModel")
@ModelBase.example("jaredpalmer/kev-4b")
class KevModel(_DecisionLoraMixin, Qwen3_5TextModel):
    model_arch = gguf.MODEL_ARCH.QWEN35
    ...

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        yield from super().generate_extra_tensors()
        # pointer head: the output of a token is [q | k]
        head = self.head["head"]
        yield "classifier.out_proj.weight", torch.cat([head["q.weight"], head["k.weight"]], dim=0)
        yield "classifier.out_proj.bias",   torch.cat([head["q.bias"],   head["k.bias"]],   dim=0)
```

```c++
            for (const int32_t marker : decision.markers) {
                // Read the 512 projected features at this option's end marker.
                const float * embd = get_embd(marker);

                // This applies to models like Laya/Julia.
                if (decision.pointer < 0) {
                    res->scores.push_back(embd[decision.column]);
                    continue;
                }

                float dot = 0.0f;
                for (int32_t i = 0; i < n_pointer; i++) {
                    dot += embd_q[i] * embd[n_pointer + i];
                }
                res->scores.push_back(dot / sqrtf((float) n_pointer));
            }
```

```c++
void server_decision_context::init(const llama_model * model) {
    ...
    } else if (model_type == COMMON_DECISION_TYPE_KEV) {
        // the hidden state of an option is read at the token that ends it
        const auto toks = common_tokenize(vocab, "<|box_end|>", false, true);
        if (toks.size() != 1) {
            throw std::runtime_error("decision model has no <|box_end|> token");
        }
        token_marker  = toks[0];
        n_options_max = 255;
}
```
```console
(gdb) p res->scores
$37 = std::vector of length 4, capacity 4 = {2.16662145, -3.83496332, -11.2319431, -0.0366165936}
(gdb) p *res
$39 = {<server_task_result> = {
    _vptr.server_task_result = 0x7ffff7e30828 <vtable for server_task_result_decision+16>, id = 0, id_slot = -1, 
    index = 0}, scores = std::vector of length 4, capacity 4 = {2.16662145, -3.83496332, -11.2319431, 
    -0.0366165936}, n_tokens = 93}
```
This is now end up in:
```c++
        auto all_results = rd.wait_for_all(req.should_stop);
```

### lev
Download the lev model:
```console
(venv) $ hf download ggml-org/lev-GGUF lev-Q4_K_M.gguf --local-dir models
```
We can inspect the model:
```console
(venv) $ gguf-dump models/lev-Q4_K_M.gguf
...
      4: STRING     |        1 | general.architecture = 'qwen35'
      5: STRING     |        1 | general.type = 'model'
      6: STRING     |        1 | general.name = 'lev'
      ...
     34: STRING     |        1 | qwen35.decision.type = 'lev'
     35: FLOAT32    |        1 | qwen35.decision.temperature.noul = 2.3332931995391846
     36: FLOAT32    |        1 | qwen35.decision.temperature.score = 2.803323745727539
     37: FLOAT32    |        1 | qwen35.decision.temperature.choice = 1.7666659355163574
     38: FLOAT32    |        1 | qwen35.decision.temperature.choice.small = 1.7897834777832031
     39: FLOAT32    |        1 | qwen35.decision.temperature.choice.mid = 1.608568549156189
     40: FLOAT32    |        1 | qwen35.decision.temperature.choice.large = 1.6641112565994263
     ...
```
So lets set a break point in server_decision_context::fill_task and look at the
prompt and what it looks like for this model:
```console
(gdb) printf "%s\n", prompt.c_str()
<|im_start|>system
You are a System One decision model. You read the Evidence and answer each Criterion by choosing exactly one of the listed options. You never explain. You answer with the single option label only.<|im_end|>
<|im_start|>user
# Evidence
{"message": "Hi, I was charged twice for my order #4471 and I want a refund.", "order": {"id": 4471, "items": ["phone case", "charger"]}, "plan": "pro"}

# Criterion
What does the customer want?

# Options
A. refund: wants money back
B. cancel: wants to cancel an order
C. track: wants to know where an order is
D. other: anything else

Respond with only the letter of the best option.
<|im_end|>
<|im_start|>assistant
<think>

</think>
```
Lev, like OpenJev used labels like A, B, C and so on. And like OpenJev where
we showed how it selects those labels and then performs a softmax on them, ignoring
the other tokens in the vocab.


```c++
size_t server_decision_context::n_variants(const server_decision_question & question) const {
    // lev shows the options of a choice in 2 orders, to cancel the preference for the first label
    if (type == COMMON_DECISION_TYPE_LEV && question.type == SERVER_DECISION_QUESTION_CHOICE && question.options.size() > 1) {
        return 2;
    }
    return 1;
}
```


In server-context.cpp send_desision:
```c++
    void send_decision(const server_slot & slot, const common_batch & batch, int32_t i_batch) {
        auto res = std::make_unique<server_task_result_decision>();
        res->id       = slot.task->id;
        res->index    = slot.task->index;
        res->n_tokens = slot.task->n_tokens();

        const auto & decision = slot.task->decision;
```
```console
(gdb) p decision
$4 = (const server_task::decision &) @0x5555558d2ea0: {
labels = std::vector of length 4, capacity 4 = {32, 33, 34, 35},
markers = std::vector of length 0, capacity 0, column = 0,
pointer = -1, order = std::vector of length 0, capacity 0, n_scores = 0}
```
This will take the same path as OpenJev (which also uses labels):
```c++
        if (!decision.labels.empty()) {
            const float * logits = llama_get_logits_ith(slot.ctx_tgt, i_batch);
            if (logits == nullptr) {
                send_error(slot, "failed to get logits", ERROR_TYPE_SERVER);
                return;
            }

            const int32_t n_vocab = llama_vocab_n_tokens(vocab);
            for (const llama_token label : decision.labels) {
                GGML_ASSERT(label >= 0 && label < n_vocab);
                res->scores.push_back(logits[label]);
            }
```
So the above will add the logits for the lables to res->scores.

Later in format_answer:
```c++
json server_decision_context::format_answer(const server_decision_question & question, const std::vector<std::vector<float>> & scores) const {
    const size_t n = n_outputs(question);
    ...

    // probabilities are calculated just like before

    json answer = json{{"type", decision_question_type_name(question.type)}};

    if (question.type == SERVER_DECISION_QUESTION_NOUL) {
        if (type == COMMON_DECISION_TYPE_LEV) {
            double expected = 0.0;
            for (size_t i = 0; i < n; i++) {
                expected += probs[i] * i / (n - 1);
            }
            answer["noul"] = expected;
            return answer;
        }
```


### laya
TODO:


### nimble
TODO:


### clef
TODO:



