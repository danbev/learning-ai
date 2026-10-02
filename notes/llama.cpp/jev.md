### Jev in llama.cpp

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

### OpenJev
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


### laya



