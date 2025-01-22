### Llama 3.2 Vision 
This document contains notes about llama 3.2 vision and about integrating this
into llama.cpp.

The model architecture is similar to Llama 3.1 but with addition of a vision
model in addition to the text model. The architecture is named `mllama` for
multi-modal llama.

Paper: https://arxiv.org/pdf/2407.21783

* [Vision API PR](https://github.com/ggerganov/llama.cpp/pull/11292)
* [Discussing about multi-model .gguf models](https://github.com/ggerganov/llama.cpp/discussions/11139?sort=old)

### vocab
One interesting thing with this model is that is has a vocab size specified as:
```console
"vocab_size": 128256 
```
But the special token `<|image|>` is at index 128256, so the actual vocab size
is 128257. We can see this by inspecting the actual vocabulary array in
convert_hf_to_gguf.py:
```python
    tokenizer = AutoTokenizer.from_pretrained(self.dir_model, trust_remote_code=is_cli_non_interactive)
    print(f'tokenizer len: {len(tokenizer.vocab)}')
    vocab_size = self.hparams.get("vocab_size", len(tokenizer.vocab))
    assert max(tokenizer.vocab.values()) <= vocab_size
```
```console
tokenizer len: 128257
```
This causes problems as there is a tensor that depend on the vocab size being
128256:
```console
      1:  525336576 |  4096, 128256,    1,     1 | Q6_K    | output.weight
```

The image token needs to be in our models vocab, in `vocab.id_to_token` that is,
so that it is resolved correctly and the correct token id passed to the model.

For example, in `llama_decode_impl`:
```c++
            if (n_outputs_new) {
                GGML_ASSERT( n_outputs_prev + n_outputs_new <= n_outputs);
                GGML_ASSERT((n_outputs_prev + n_outputs_new)*n_vocab <= (int64_t) lctx.logits_size);
                ggml_backend_tensor_get_async(backend_res, res, logits_out, 0, n_outputs_new*(n_vocab)*sizeof(float));
            }
```
So as far as I can tell we need to have the additional image token in the
actual vocab list, `id_to_token` in llama.cpp. The vocabulary size is determined
by calling:
```c++
int32_t llama_vocab_n_tokens(const struct llama_vocab * vocab) {
    return vocab->n_tokens();
}

uint32_t llama_vocab::n_tokens() const {
    return (uint32_t) pimpl->id_to_token.size();
}
```
And notice that this is using the size of the `id_to_token` vector
to determine the vocab size. Now, this vector is resized in llama-vocab.cpp:
```c++
    uint32_t n_tokens = gguf_get_arr_n(ctx, token_idx);
    id_to_token.resize(n_tokens);
```
```console
(gdb) p  n_tokens
$1 = 128256
```

I think a way to handle this is to leave the vocab size as 128256 when
converting the model, so that id_to_token will have the correct size. And then
add a special token for the image token.
So adding the following to the converted .gguf model:
```console
     60: UINT32     |        1 | tokenizer.ggml.image_token_id = 128256
```
And then adding this to the vocab special tokens in llama-arch.cpp:
```c++
enum llm_kv {
    ...
    LLM_KV_TOKENIZER_IMAGE_ID,
    ...
```
And the in llama-vocab.cpp:
```c++
struct llama_vocab::impl {
    ...
    llama_token special_image_id = LLAMA_TOKEN_NULL;
    ...
}
```
And the update the handling of special tokens in llama-vocab.cpp:
```c++
void llama_vocab::impl::load(llama_model_loader & ml, const LLM_KV & kv) {
   ...
   // special tokens                                                               
    {                                                                               
        const std::vector<std::pair<enum llm_kv, int32_t &>> special_token_types = {
            { LLM_KV_TOKENIZER_BOS_ID,     special_bos_id     },                    
            { LLM_KV_TOKENIZER_EOS_ID,     special_eos_id     },                    
            { LLM_KV_TOKENIZER_EOT_ID,     special_eot_id     },                    
            { LLM_KV_TOKENIZER_EOM_ID,     special_eom_id     },                    
            { LLM_KV_TOKENIZER_UNK_ID,     special_unk_id     },                    
            { LLM_KV_TOKENIZER_SEP_ID,     special_sep_id     },                    
            { LLM_KV_TOKENIZER_PAD_ID,     special_pad_id     },                    
            { LLM_KV_TOKENIZER_MASK_ID,    special_mask_id    },                    
            { LLM_KV_TOKENIZER_IMAGE_ID,   special_image_id   },                    
            { LLM_KV_TOKENIZER_FIM_PRE_ID, special_fim_pre_id },                    
            { LLM_KV_TOKENIZER_FIM_SUF_ID, special_fim_suf_id },                    
            { LLM_KV_TOKENIZER_FIM_MID_ID, special_fim_mid_id },                    
            { LLM_KV_TOKENIZER_FIM_PAD_ID, special_fim_pad_id },                    
            { LLM_KV_TOKENIZER_FIM_REP_ID, special_fim_rep_id },                    
            { LLM_KV_TOKENIZER_FIM_SEP_ID, special_fim_sep_id },                    
                                                                                    
            // deprecated                                                           
            { LLM_KV_TOKENIZER_PREFIX_ID, special_fim_pre_id },                     
            { LLM_KV_TOKENIZER_SUFFIX_ID, special_fim_suf_id },                     
            { LLM_KV_TOKENIZER_MIDDLE_ID, special_fim_mid_id },                 
        };
```
Hmm, this will still not work as if we print out the tokens for the following
prompt we will see that it will not use the correct image token id:
```console
prompt: <|image|>What is in this image?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

token = 27
token = 91
token = 1843
token = 91
token = 29
token = 3923
token = 374
token = 304
token = 420
token = 2217
token = 30
token = 128009
token = 128006
token = 78191
token = 128007
token = 271
```
So perhaps we should let the vocabulary size be 128257 so that the image token
is included in `id_to_token` and then modify the shape of `output.weight` that
depends on the size being 128256. 

### Model conversion
So we first need to convert the model to GGUF format which is done by the
`convert_hf_to_gguf.py` script. This model consists of not just one model but
it has two which is also reflected in the config.json file of the model. The
language model is in a `text_config` attribute, and the vision model is a
`vision_config` attribute:
```console
{
  "architectures": [
    "MllamaForConditionalGeneration"
  ],
  "image_token_index": 128256,
  "model_type": "mllama",
  "text_config": {
      ...
  }
  "vision_config": {
      ...
  }
```
And we can see the architecture is `MllamaForConditionalGeneration`  and the
model type is `mllama`. 

If we inspect `model.safetensors.index.json` we can find a few tensor that
were not in previous version of Llama:
```console
    "language_model.model.layers.13.cross_attn.k_norm.weight": "model-00002-of-00005.safetensors",
    "language_model.model.layers.13.cross_attn.k_proj.weight": "model-00002-of-00005.safetensors",
    "language_model.model.layers.13.cross_attn.o_proj.weight": "model-00002-of-00005.safetensors",
    "language_model.model.layers.13.cross_attn.q_norm.weight": "model-00002-of-00005.safetensors",
    "language_model.model.layers.13.cross_attn.q_proj.weight": "model-00002-of-00005.safetensors",
    "language_model.model.layers.13.cross_attn.v_proj.weight": "model-00002-of-00005.safetensors",
    "language_model.model.layers.13.cross_attn_attn_gate": "model-00002-of-00005.safetensors",
    "language_model.model.layers.13.cross_attn_mlp_gate": "model-00002-of-00005.safetensors",
```
These tensors exist for blocks 3, 8, 13, 18, 23, 28, 33, 38. Which also matches
the following attribute in the config.json file:
```console
    "cross_attention_layers": [
      3,
      8,
      13,
      18,
      23,
      28,
      33,
      38
    ],
```
As far as I know there are currently no tensor names like this so we need to
add them to the `gguf-py/gguf/constants.py` file:
```console
    MODEL_TENSOR.CROSS_ATTN_K_NORM          "blk.{bid}.cross_attn_k_norm",
    MODEL_TENSOR.CROSS_ATTN_K_PROJ          "blk.{bid}.cross_attn_k_proj",
    MODEL_TENSOR.CROSS_ATTN_Q_NORM          "blk.{bid}.cross_attn_q_norm",
    MODEL_TENSOR.CROSS_ATTN_Q_PROJ          "blk.{bid}.cross_attn_q_proj",
    MODEL_TENSOR.CROSS_ATTN_O_PROJ          "blk.{bid}.cross_attn_o_proj",
    MODEL_TENSOR.CROSS_ATTN_V_PROJ          "blk.{bid}.cross_attn_v_proj",
    MODEL_TENSOR.CROSS_ATTN_ATTN_GATE       "blk.{bid}.cross_attn_attn_gate",
    MODEL_TENSOR.CROSS_ATTN_MPL_GATE        "blk.{bid}.cross_attn_mpl_gate",
```

### projection/adapter layer(tensors)
```console
"multi_modal_projector.bias"
"multi_modal_projector.weight"
```

### Vision model overview


### Vision model layer (tensors)
The vision model has 8 global layers and 32 hidden layers:
```console
    "num_global_layers": 8,
    "num_hidden_layers": 32,
```

There are also tensors that are not part of any layers (global)

First we have convert the image into patches using the following tensor:
```console
"vision_model.patch_embedding.weight"
```

Then a class embedding token is added:
```console
"vision_model.class_embedding"    [1280] (hidden_size)
```

Then we have the pre-tile positional embedding:
```console
"vision_model.pre_tile_positional_embedding.embedding.weight"
"vision_model.pre_tile_positional_embedding.gate"
```
Then we apply pre-normalization:
```console
"vision_model.layernorm_pre.weight"
"vision_model.layernorm_pre.bias"
```
Then we have the gated positional embedding:
```
"vision_model.gated_positional_embedding.embedding"
"vision_model.gated_positional_embedding.gate"
"vision_model.gated_positional_embedding.tile_embedding.weight"
```
Then we have the post-tile positional embedding:
```
"vision_model.post_tile_positional_embedding.embedding.weight"
"vision_model.post_tile_positional_embedding.gate"
```
And lastly we have the post-normalization:
```console
"vision_model.layernorm_post.bias"
"vision_model.layernorm_post.weight"
```

Tiling is done to support higher resolution images. In a standard vision
tranformer an image would be split into patches, perhaps 16x16 pixels and then
processed as a sequence. When tiling is used the image is first split into
smaller tiles, perhaps 224x224 each, and then each tile is processed in the same
way as a standared vision transformer (each tile is split into patches and then
processed as a sequence).

The model has 8 (`num_global_layers`) global layers:
```console
"vision_model.global_transformer.layers.{bid}.gate_attn"
"vision_model.global_transformer.layers.{bid}.gate_ffn"
"vision_model.global_transformer.layers.{bid}.input_layernorm.bias"
"vision_model.global_transformer.layers.{bid}.input_layernorm.weight"
"vision_model.global_transformer.layers.{bid}.mlp.fc1.bias"
"vision_model.global_transformer.layers.{bid}.mlp.fc1.weight"
"vision_model.global_transformer.layers.{bid}.mlp.fc2.bias"
"vision_model.global_transformer.layers.{bid}.mlp.fc2.weight"
"vision_model.global_transformer.layers.{bid}.post_attention_layernorm.bias"
"vision_model.global_transformer.layers.{bid}.post_attention_layernorm.weight"
"vision_model.global_transformer.layers.{bid}.self_attn.k_proj.weight"
"vision_model.global_transformer.layers.{bid}.self_attn.o_proj.weight"
"vision_model.global_transformer.layers.{bid}.self_attn.q_proj.weight"
"vision_model.global_transformer.layers.{bid}.self_attn.v_proj.weight"

fc = fully connected.
```

And 32 (`num_hidden_layers`) hidden layers:
```console
"vision_model.transformer.layers.{bid}.input_layernorm.bias"
"vision_model.transformer.layers.{bid}.input_layernorm.weight"
"vision_model.transformer.layers.{bid}.mlp.fc1.bias"
"vision_model.transformer.layers.{bid}.mlp.fc1.weight"
"vision_model.transformer.layers.{bid}.mlp.fc2.bias"
"vision_model.transformer.layers.{bid}.mlp.fc2.weight"
"vision_model.transformer.layers.{bid}.post_attention_layernorm.bias"
"vision_model.transformer.layers.{bid}.post_attention_layernorm.weight"
"vision_model.transformer.layers.{bid}.self_attn.k_proj.weight"
"vision_model.transformer.layers.{bid}.self_attn.o_proj.weight"
"vision_model.transformer.layers.{bid}.self_attn.q_proj.weight"
"vision_model.transformer.layers.{bid}.self_attn.v_proj.weight"
}


```

I initially thougth that having a single model for both the lanuage and vision
model was a good idea, simpler to manage for users. But I had not considered
that it might not optimal from a performance perspective. If we have separate
models.  
The following is from a [discussion](https://github.com/ggerganov/llama.cpp/discussions/11139#discussioncomment-11783418)
on this topic:
```
Having separate models allows to create separate contexts for the encoder and
decoder which gives more fine-grained control over the computation - how many
layers to offload, which devices to use, how much memory to reserve, etc.
Also, computations of the encoder and the decoder could be interleaved which is
important for high-performance scenarios - for example, while we are decoding
the response for an image we could be already encoding the next images.

Having a single GGUF for the entire vision model is definitely more convenient
for users and distribution. But maybe this can be achieved by extending GGUF to
allow packing multiple GGUFs (like an archive).
```
So I'm going to create two models for Llama 3.2 Vision Instruct and then take
a look at how packaging multiple GGUFs could be done.


The current `convert_hf_to_gguf.py` script really only support a single model
as output as it is now. But long term I think there will be multiple models
that have more than one model in them. I'm thinking of text-to-speech models
which can contain a voice decoder model in addition to the language model.

So a language model like Llama 3.2 Vision Instruct be registereded using the
`@Model.register` decorator:
```python
@Model.register("MllamaForConditionalGeneration")
class MLlamaModel(Model):
    model_arch = gguf.MODEL_ARCH.MLLAMA
```
Now, if a model has a vision model in addition to the language model, we could
we might expect there to be a command line option to specify that the vision
model should be extracted to a separate model. So perhaps there should be
a vision_model attibute for the MLlamaModel class for the vision encoder which
would have a different `model_arch` attribute, like
`gguf.MODEL_ARCH.MLLAMA_VISION`.


### Language model layers (tensors)
```console
"language_model.lm_head.weight"
"language_model.model.embed_tokens.weight"
"language_model.model.norm.weight"
```

The language model has 40 hidden layers (`num_hidden_layers`):
```console
"language_model.model.layers.{bid}.input_layernorm.weight"
"language_model.model.layers.{bid}.mlp.down_proj.weight"
"language_model.model.layers.{bid}.mlp.gate_proj.weight"
"language_model.model.layers.{bid}.mlp.up_proj.weight"
"language_model.model.layers.{bid}.post_attention_layernorm.weight"
"language_model.model.layers.{bid}.self_attn.k_proj.weight"
"language_model.model.layers.{bid}.self_attn.o_proj.weight"
"language_model.model.layers.{bid}.self_attn.q_proj.weight"
"language_model.model.layers.{bid}.self_attn.v_proj.weight"
```



### Tasks
- [] What tokenizer should be used?
- [] Support both models in `convert_hf_to_gguf.py`

### converted model
```console
(venv) $ ./inspect-model.sh models/llama-3-2-11b.gguf
INFO:gguf-dump:* Loading: models/llama-3-2-11b.gguf
* File is LITTLE endian, script is running on a LITTLE endian host.
* Dumping 28 key/value pair(s)
      1: UINT32     |        1 | GGUF.version = 3
      2: UINT64     |        1 | GGUF.tensor_count = 906
      3: UINT64     |        1 | GGUF.kv_count = 25
      4: STRING     |        1 | general.architecture = 'mllama'
      5: STRING     |        1 | general.type = 'model'
      6: STRING     |        1 | general.name = 'Llama 3.2 11B Vision Instruct'
      7: STRING     |        1 | general.finetune = 'Vision-Instruct'
      8: STRING     |        1 | general.basename = 'Llama-3.2'
      9: STRING     |        1 | general.size_label = '11B'
     10: STRING     |        1 | general.license = 'llama3.2'
     11: [STRING]   |        6 | general.tags
     12: [STRING]   |        8 | general.languages
     13: UINT32     |        1 | mllama.block_count = 40
     14: UINT32     |        1 | mllama.context_length = 131072
     15: UINT32     |        1 | mllama.embedding_length = 4096
     16: UINT32     |        1 | mllama.feed_forward_length = 14336
     17: UINT32     |        1 | mllama.attention.head_count = 32
     18: UINT32     |        1 | general.file_type = 1
     19: STRING     |        1 | tokenizer.ggml.model = 'gpt2'
     20: STRING     |        1 | tokenizer.ggml.pre = 'llama-bpe'
     21: [STRING]   |   128257 | tokenizer.ggml.tokens
     22: [INT32]    |   128257 | tokenizer.ggml.token_type
     23: [STRING]   |   280147 | tokenizer.ggml.merges
     24: UINT32     |        1 | tokenizer.ggml.bos_token_id = 128000
     25: UINT32     |        1 | tokenizer.ggml.eos_token_id = 128009
     26: UINT32     |        1 | tokenizer.ggml.padding_token_id = 128004
     27: STRING     |        1 | tokenizer.chat_template = '{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- s'
     28: UINT32     |        1 | general.quantization_version = 2
* Dumping 906 tensor(s)
      1:  525369344 |  4096, 128264,     1,     1 | F16     | token_embd.weight
      2:       4096 |  4096,     1,     1,     1 | F32     | blk.0.attn_norm.weight
      3:   58720256 | 14336,  4096,     1,     1 | F16     | blk.0.ffn_down.weight
      4:   58720256 |  4096, 14336,     1,     1 | F16     | blk.0.ffn_gate.weight
      5:   58720256 |  4096, 14336,     1,     1 | F16     | blk.0.ffn_up.weight
      6:       4096 |  4096,     1,     1,     1 | F32     | blk.0.post_attention_norm.weight
      7:    4194304 |  4096,  1024,     1,     1 | F16     | blk.0.attn_k.weight
      8:   16777216 |  4096,  4096,     1,     1 | F16     | blk.0.attn_output.weight
      9:   16777216 |  4096,  4096,     1,     1 | F16     | blk.0.attn_q.weight
     10:    4194304 |  4096,  1024,     1,     1 | F16     | blk.0.attn_v.weight
     11:       4096 |  4096,     1,     1,     1 | F32     | blk.1.attn_norm.weight
     12:   58720256 | 14336,  4096,     1,     1 | F16     | blk.1.ffn_down.weight
     13:   58720256 |  4096, 14336,     1,     1 | F16     | blk.1.ffn_gate.weight
     14:   58720256 |  4096, 14336,     1,     1 | F16     | blk.1.ffn_up.weight
     15:       4096 |  4096,     1,     1,     1 | F32     | blk.1.post_attention_norm.weight
     16:    4194304 |  4096,  1024,     1,     1 | F16     | blk.1.attn_k.weight
     17:   16777216 |  4096,  4096,     1,     1 | F16     | blk.1.attn_output.weight
     18:   16777216 |  4096,  4096,     1,     1 | F16     | blk.1.attn_q.weight
     19:    4194304 |  4096,  1024,     1,     1 | F16     | blk.1.attn_v.weight
     20:       4096 |  4096,     1,     1,     1 | F32     | blk.2.attn_norm.weight
     21:   58720256 | 14336,  4096,     1,     1 | F16     | blk.2.ffn_down.weight
     22:   58720256 |  4096, 14336,     1,     1 | F16     | blk.2.ffn_gate.weight
     23:   58720256 |  4096, 14336,     1,     1 | F16     | blk.2.ffn_up.weight
     24:       4096 |  4096,     1,     1,     1 | F32     | blk.2.post_attention_norm.weight
     25:    4194304 |  4096,  1024,     1,     1 | F16     | blk.2.attn_k.weight
     26:   16777216 |  4096,  4096,     1,     1 | F16     | blk.2.attn_output.weight
     27:   16777216 |  4096,  4096,     1,     1 | F16     | blk.2.attn_q.weight
     28:    4194304 |  4096,  1024,     1,     1 | F16     | blk.2.attn_v.weight
     29:        128 |   128,     1,     1,     1 | F32     | blk.3.cross_attn_k_norm.weight
     30:    4194304 |  4096,  1024,     1,     1 | F16     | blk.3.cross_attn_k_proj.weight
     31:   16777216 |  4096,  4096,     1,     1 | F16     | blk.3.cross_attn_output_proj.weight
     32:        128 |   128,     1,     1,     1 | F32     | blk.3.cross_attn_q_norm.weight
     33:   16777216 |  4096,  4096,     1,     1 | F16     | blk.3.cross_attn_q_proj.weight
     34:    4194304 |  4096,  1024,     1,     1 | F16     | blk.3.cross_attn_v_proj.weight
     35:          1 |     1,     1,     1,     1 | F32     | blk.3.cross_attn_attn_gate
     36:          1 |     1,     1,     1,     1 | F32     | blk.3.cross_attn_mpl_gate
     37:       4096 |  4096,     1,     1,     1 | F32     | blk.3.attn_norm.weight
     38:   58720256 | 14336,  4096,     1,     1 | F16     | blk.3.ffn_down.weight
     39:   58720256 |  4096, 14336,     1,     1 | F16     | blk.3.ffn_gate.weight
     40:   58720256 |  4096, 14336,     1,     1 | F16     | blk.3.ffn_up.weight
     41:       4096 |  4096,     1,     1,     1 | F32     | blk.3.post_attention_norm.weight
     42:       4096 |  4096,     1,     1,     1 | F32     | blk.4.attn_norm.weight
     43:   58720256 | 14336,  4096,     1,     1 | F16     | blk.4.ffn_down.weight
     44:   58720256 |  4096, 14336,     1,     1 | F16     | blk.4.ffn_gate.weight
     45:   58720256 |  4096, 14336,     1,     1 | F16     | blk.4.ffn_up.weight
     46:       4096 |  4096,     1,     1,     1 | F32     | blk.4.post_attention_norm.weight
     47:    4194304 |  4096,  1024,     1,     1 | F16     | blk.4.attn_k.weight
     48:   16777216 |  4096,  4096,     1,     1 | F16     | blk.4.attn_output.weight
     49:   16777216 |  4096,  4096,     1,     1 | F16     | blk.4.attn_q.weight
     50:    4194304 |  4096,  1024,     1,     1 | F16     | blk.4.attn_v.weight
     51:   16777216 |  4096,  4096,     1,     1 | F16     | blk.5.attn_q.weight
     52:       1280 |  1280,     1,     1,     1 | F32     | vis.class_embd
     53:    2049280 |  1280,  1601,     1,     1 | F32     | vis.gated_pos_embd_embd
     54:          1 |     1,     1,     1,     1 | F32     | vis.gated_pos_embd_gate
     55:   73774080 | 8197120,     9,     1,     1 | F16     | vis.gated_pos_embd_tile_embedding.weight
     56:          1 |     1,     1,     1,     1 | F32     | vis.glob.blk.0.gate_attn
     57:          1 |     1,     1,     1,     1 | F32     | vis.glob.blk.0.gate_ffn
     58:       1280 |  1280,     1,     1,     1 | F32     | vis.glob.blk.0.input_norm.bias
     59:       1280 |  1280,     1,     1,     1 | F32     | vis.glob.blk.0.input_norm.weight
     60:       5120 |  5120,     1,     1,     1 | F32     | vis.glob.blk.0.mlp_fc1.bias
     61:    6553600 |  1280,  5120,     1,     1 | F16     | vis.glob.blk.0.mlp_fc1.weight
     62:       1280 |  1280,     1,     1,     1 | F32     | vis.glob.blk.0.mlp_fc2.bias
     63:    6553600 |  5120,  1280,     1,     1 | F16     | vis.glob.blk.0.mlp_fc2.weight
     64:       1280 |  1280,     1,     1,     1 | F32     | vis.glob.blk.0.post_attn_norm.bias
     65:       1280 |  1280,     1,     1,     1 | F32     | vis.glob.blk.0.post_attn_norm.weight
     66:    1638400 |  1280,  1280,     1,     1 | F16     | vis.glob.blk.0.attn_k_proj.weight
     67:    1638400 |  1280,  1280,     1,     1 | F16     | vis.glob.blk.0.attn_out_proj.weight
     68:    1638400 |  1280,  1280,     1,     1 | F16     | vis.glob.blk.0.attn_q_proj.weight
     69:    1638400 |  1280,  1280,     1,     1 | F16     | vis.glob.blk.0.attn_v_proj.weight
     70:          1 |     1,     1,     1,     1 | F32     | vis.glob.blk.1.gate_attn
     71:          1 |     1,     1,     1,     1 | F32     | vis.glob.blk.1.gate_ffn
     72:       1280 |  1280,     1,     1,     1 | F32     | vis.glob.blk.1.input_norm.bias
     73:       1280 |  1280,     1,     1,     1 | F32     | vis.glob.blk.1.input_norm.weight
     74:       5120 |  5120,     1,     1,     1 | F32     | vis.glob.blk.1.mlp_fc1.bias
     75:    6553600 |  1280,  5120,     1,     1 | F16     | vis.glob.blk.1.mlp_fc1.weight
     76:       1280 |  1280,     1,     1,     1 | F32     | vis.glob.blk.1.mlp_fc2.bias
     77:    6553600 |  5120,  1280,     1,     1 | F16     | vis.glob.blk.1.mlp_fc2.weight
     78:       1280 |  1280,     1,     1,     1 | F32     | vis.glob.blk.1.post_attn_norm.bias
     79:       1280 |  1280,     1,     1,     1 | F32     | vis.glob.blk.1.post_attn_norm.weight
     80:    1638400 |  1280,  1280,     1,     1 | F16     | vis.glob.blk.1.attn_k_proj.weight
     81:    1638400 |  1280,  1280,     1,     1 | F16     | vis.glob.blk.1.attn_out_proj.weight
     82:    1638400 |  1280,  1280,     1,     1 | F16     | vis.glob.blk.1.attn_q_proj.weight
     83:    1638400 |  1280,  1280,     1,     1 | F16     | vis.glob.blk.1.attn_v_proj.weight
     84:          1 |     1,     1,     1,     1 | F32     | vis.glob.blk.2.gate_attn
     85:          1 |     1,     1,     1,     1 | F32     | vis.glob.blk.2.gate_ffn
     86:       1280 |  1280,     1,     1,     1 | F32     | vis.glob.blk.2.input_norm.bias
     87:       1280 |  1280,     1,     1,     1 | F32     | vis.glob.blk.2.input_norm.weight
     88:       5120 |  5120,     1,     1,     1 | F32     | vis.glob.blk.2.mlp_fc1.bias
     89:    6553600 |  1280,  5120,     1,     1 | F16     | vis.glob.blk.2.mlp_fc1.weight
     90:       1280 |  1280,     1,     1,     1 | F32     | vis.glob.blk.2.mlp_fc2.bias
     91:    6553600 |  5120,  1280,     1,     1 | F16     | vis.glob.blk.2.mlp_fc2.weight
     92:       1280 |  1280,     1,     1,     1 | F32     | vis.glob.blk.2.post_attn_norm.bias
     93:       1280 |  1280,     1,     1,     1 | F32     | vis.glob.blk.2.post_attn_norm.weight
     94:    1638400 |  1280,  1280,     1,     1 | F16     | vis.glob.blk.2.attn_k_proj.weight
     95:    1638400 |  1280,  1280,     1,     1 | F16     | vis.glob.blk.2.attn_out_proj.weight
     96:    1638400 |  1280,  1280,     1,     1 | F16     | vis.glob.blk.2.attn_q_proj.weight
     97:    1638400 |  1280,  1280,     1,     1 | F16     | vis.glob.blk.2.attn_v_proj.weight
     98:          1 |     1,     1,     1,     1 | F32     | vis.glob.blk.3.gate_attn
     99:          1 |     1,     1,     1,     1 | F32     | vis.glob.blk.3.gate_ffn
    100:       1280 |  1280,     1,     1,     1 | F32     | vis.glob.blk.3.input_norm.bias
    101:       1280 |  1280,     1,     1,     1 | F32     | vis.glob.blk.3.input_norm.weight
    102:       5120 |  5120,     1,     1,     1 | F32     | vis.glob.blk.3.mlp_fc1.bias
    103:    6553600 |  1280,  5120,     1,     1 | F16     | vis.glob.blk.3.mlp_fc1.weight
    104:       1280 |  1280,     1,     1,     1 | F32     | vis.glob.blk.3.mlp_fc2.bias
    105:    6553600 |  5120,  1280,     1,     1 | F16     | vis.glob.blk.3.mlp_fc2.weight
    106:       1280 |  1280,     1,     1,     1 | F32     | vis.glob.blk.3.post_attn_norm.bias
    107:       1280 |  1280,     1,     1,     1 | F32     | vis.glob.blk.3.post_attn_norm.weight
    108:    1638400 |  1280,  1280,     1,     1 | F16     | vis.glob.blk.3.attn_k_proj.weight
    109:    1638400 |  1280,  1280,     1,     1 | F16     | vis.glob.blk.3.attn_out_proj.weight
    110:    1638400 |  1280,  1280,     1,     1 | F16     | vis.glob.blk.3.attn_q_proj.weight
    111:    1638400 |  1280,  1280,     1,     1 | F16     | vis.glob.blk.3.attn_v_proj.weight
    112:          1 |     1,     1,     1,     1 | F32     | vis.glob.blk.4.gate_attn
    113:          1 |     1,     1,     1,     1 | F32     | vis.glob.blk.4.gate_ffn
    114:       1280 |  1280,     1,     1,     1 | F32     | vis.glob.blk.4.input_norm.bias
    115:       1280 |  1280,     1,     1,     1 | F32     | vis.glob.blk.4.input_norm.weight
    116:       5120 |  5120,     1,     1,     1 | F32     | vis.glob.blk.4.mlp_fc1.bias
    117:    6553600 |  1280,  5120,     1,     1 | F16     | vis.glob.blk.4.mlp_fc1.weight
    118:       1280 |  1280,     1,     1,     1 | F32     | vis.glob.blk.4.mlp_fc2.bias
    119:    6553600 |  5120,  1280,     1,     1 | F16     | vis.glob.blk.4.mlp_fc2.weight
    120:       1280 |  1280,     1,     1,     1 | F32     | vis.glob.blk.4.post_attn_norm.bias
    121:       1280 |  1280,     1,     1,     1 | F32     | vis.glob.blk.4.post_attn_norm.weight
    122:    1638400 |  1280,  1280,     1,     1 | F16     | vis.glob.blk.4.attn_k_proj.weight
    123:    1638400 |  1280,  1280,     1,     1 | F16     | vis.glob.blk.4.attn_out_proj.weight
    124:    1638400 |  1280,  1280,     1,     1 | F16     | vis.glob.blk.4.attn_q_proj.weight
    125:    1638400 |  1280,  1280,     1,     1 | F16     | vis.glob.blk.4.attn_v_proj.weight
    126:          1 |     1,     1,     1,     1 | F32     | vis.glob.blk.5.gate_attn
    127:          1 |     1,     1,     1,     1 | F32     | vis.glob.blk.5.gate_ffn
    128:       1280 |  1280,     1,     1,     1 | F32     | vis.glob.blk.5.input_norm.bias
    129:       1280 |  1280,     1,     1,     1 | F32     | vis.glob.blk.5.input_norm.weight
    130:       5120 |  5120,     1,     1,     1 | F32     | vis.glob.blk.5.mlp_fc1.bias
    131:    6553600 |  1280,  5120,     1,     1 | F16     | vis.glob.blk.5.mlp_fc1.weight
    132:       1280 |  1280,     1,     1,     1 | F32     | vis.glob.blk.5.mlp_fc2.bias
    133:    6553600 |  5120,  1280,     1,     1 | F16     | vis.glob.blk.5.mlp_fc2.weight
    134:       1280 |  1280,     1,     1,     1 | F32     | vis.glob.blk.5.post_attn_norm.bias
    135:       1280 |  1280,     1,     1,     1 | F32     | vis.glob.blk.5.post_attn_norm.weight
    136:    1638400 |  1280,  1280,     1,     1 | F16     | vis.glob.blk.5.attn_k_proj.weight
    137:    1638400 |  1280,  1280,     1,     1 | F16     | vis.glob.blk.5.attn_out_proj.weight
    138:    1638400 |  1280,  1280,     1,     1 | F16     | vis.glob.blk.5.attn_q_proj.weight
    139:    1638400 |  1280,  1280,     1,     1 | F16     | vis.glob.blk.5.attn_v_proj.weight
    140:          1 |     1,     1,     1,     1 | F32     | vis.glob.blk.6.gate_attn
    141:          1 |     1,     1,     1,     1 | F32     | vis.glob.blk.6.gate_ffn
    142:       1280 |  1280,     1,     1,     1 | F32     | vis.glob.blk.6.input_norm.bias
    143:       1280 |  1280,     1,     1,     1 | F32     | vis.glob.blk.6.input_norm.weight
    144:       5120 |  5120,     1,     1,     1 | F32     | vis.glob.blk.6.mlp_fc1.bias
    145:    6553600 |  1280,  5120,     1,     1 | F16     | vis.glob.blk.6.mlp_fc1.weight
    146:       1280 |  1280,     1,     1,     1 | F32     | vis.glob.blk.6.mlp_fc2.bias
    147:    6553600 |  5120,  1280,     1,     1 | F16     | vis.glob.blk.6.mlp_fc2.weight
    148:       1280 |  1280,     1,     1,     1 | F32     | vis.glob.blk.6.post_attn_norm.bias
    149:       1280 |  1280,     1,     1,     1 | F32     | vis.glob.blk.6.post_attn_norm.weight
    150:    1638400 |  1280,  1280,     1,     1 | F16     | vis.glob.blk.6.attn_k_proj.weight
    151:    1638400 |  1280,  1280,     1,     1 | F16     | vis.glob.blk.6.attn_out_proj.weight
    152:    1638400 |  1280,  1280,     1,     1 | F16     | vis.glob.blk.6.attn_q_proj.weight
    153:    1638400 |  1280,  1280,     1,     1 | F16     | vis.glob.blk.6.attn_v_proj.weight
    154:          1 |     1,     1,     1,     1 | F32     | vis.glob.blk.7.gate_attn
    155:          1 |     1,     1,     1,     1 | F32     | vis.glob.blk.7.gate_ffn
    156:       1280 |  1280,     1,     1,     1 | F32     | vis.glob.blk.7.input_norm.bias
    157:       1280 |  1280,     1,     1,     1 | F32     | vis.glob.blk.7.input_norm.weight
    158:       5120 |  5120,     1,     1,     1 | F32     | vis.glob.blk.7.mlp_fc1.bias
    159:    6553600 |  1280,  5120,     1,     1 | F16     | vis.glob.blk.7.mlp_fc1.weight
    160:       1280 |  1280,     1,     1,     1 | F32     | vis.glob.blk.7.mlp_fc2.bias
    161:    6553600 |  5120,  1280,     1,     1 | F16     | vis.glob.blk.7.mlp_fc2.weight
    162:       1280 |  1280,     1,     1,     1 | F32     | vis.glob.blk.7.post_attn_norm.bias
    163:       1280 |  1280,     1,     1,     1 | F32     | vis.glob.blk.7.post_attn_norm.weight
    164:    1638400 |  1280,  1280,     1,     1 | F16     | vis.glob.blk.7.attn_k_proj.weight
    165:    1638400 |  1280,  1280,     1,     1 | F16     | vis.glob.blk.7.attn_out_proj.weight
    166:    1638400 |  1280,  1280,     1,     1 | F16     | vis.glob.blk.7.attn_q_proj.weight
    167:    1638400 |  1280,  1280,     1,     1 | F16     | vis.glob.blk.7.attn_v_proj.weight
    168:       1280 |  1280,     1,     1,     1 | F32     | vis.layernorm_post.bias
    169:       1280 |  1280,     1,     1,     1 | F32     | vis.layernorm_post.weight
    170:       1280 |  1280,     1,     1,     1 | F32     | vis.layernorm_pre.bias
    171:       1280 |  1280,     1,     1,     1 | F32     | vis.layernorm_pre.weight
    172:     752640 |    14,    14,     3,  1280 | F16     | vis.patch_embd.weight
    173:      46080 |  5120,     9,     1,     1 | F16     | vis.post_tile_pos_embd_embd.weight
    174:          1 |     1,     1,     1,     1 | F32     | vis.post_tile_pos_embd_gate
    175:      46080 |  5120,     9,     1,     1 | F16     | vis.pre_tile_pos_embd_embd.weight
    176:          1 |     1,     1,     1,     1 | F32     | vis.pre_tile_pos_embd_gate
    177:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.0.input_norm.bias
    178:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.0.input_norm.weight
    179:       5120 |  5120,     1,     1,     1 | F32     | vis.blk.0.mpl_fc1.bias
    180:    6553600 |  1280,  5120,     1,     1 | F16     | vis.blk.0.mpl_fc1.weight
    181:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.0.mpl_fc2.bias
    182:    6553600 |  5120,  1280,     1,     1 | F16     | vis.blk.0.mpl_fc2.weight
    183:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.0.post_attn_norm.bias
    184:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.0.post_attn_norm.weight
    185:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.0.attn_k_proj.weight
    186:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.0.attn_out_proj.weight
    187:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.0.attn_q_proj.weight
    188:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.0.attn_v_proj.weight
    189:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.1.input_norm.bias
    190:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.1.input_norm.weight
    191:       5120 |  5120,     1,     1,     1 | F32     | vis.blk.1.mpl_fc1.bias
    192:    6553600 |  1280,  5120,     1,     1 | F16     | vis.blk.1.mpl_fc1.weight
    193:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.1.mpl_fc2.bias
    194:    6553600 |  5120,  1280,     1,     1 | F16     | vis.blk.1.mpl_fc2.weight
    195:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.1.post_attn_norm.bias
    196:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.1.post_attn_norm.weight
    197:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.1.attn_k_proj.weight
    198:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.1.attn_out_proj.weight
    199:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.1.attn_q_proj.weight
    200:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.1.attn_v_proj.weight
    201:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.10.input_norm.bias
    202:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.10.input_norm.weight
    203:       5120 |  5120,     1,     1,     1 | F32     | vis.blk.10.mpl_fc1.bias
    204:    6553600 |  1280,  5120,     1,     1 | F16     | vis.blk.10.mpl_fc1.weight
    205:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.10.mpl_fc2.bias
    206:    6553600 |  5120,  1280,     1,     1 | F16     | vis.blk.10.mpl_fc2.weight
    207:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.10.post_attn_norm.bias
    208:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.10.post_attn_norm.weight
    209:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.10.attn_k_proj.weight
    210:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.10.attn_out_proj.weight
    211:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.10.attn_q_proj.weight
    212:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.10.attn_v_proj.weight
    213:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.11.input_norm.bias
    214:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.11.input_norm.weight
    215:       5120 |  5120,     1,     1,     1 | F32     | vis.blk.11.mpl_fc1.bias
    216:    6553600 |  1280,  5120,     1,     1 | F16     | vis.blk.11.mpl_fc1.weight
    217:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.11.mpl_fc2.bias
    218:    6553600 |  5120,  1280,     1,     1 | F16     | vis.blk.11.mpl_fc2.weight
    219:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.11.post_attn_norm.bias
    220:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.11.post_attn_norm.weight
    221:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.11.attn_k_proj.weight
    222:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.11.attn_out_proj.weight
    223:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.11.attn_q_proj.weight
    224:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.11.attn_v_proj.weight
    225:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.12.input_norm.bias
    226:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.12.input_norm.weight
    227:       5120 |  5120,     1,     1,     1 | F32     | vis.blk.12.mpl_fc1.bias
    228:    6553600 |  1280,  5120,     1,     1 | F16     | vis.blk.12.mpl_fc1.weight
    229:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.12.mpl_fc2.bias
    230:    6553600 |  5120,  1280,     1,     1 | F16     | vis.blk.12.mpl_fc2.weight
    231:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.12.post_attn_norm.bias
    232:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.12.post_attn_norm.weight
    233:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.12.attn_k_proj.weight
    234:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.12.attn_out_proj.weight
    235:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.12.attn_q_proj.weight
    236:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.12.attn_v_proj.weight
    237:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.13.input_norm.bias
    238:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.13.input_norm.weight
    239:       5120 |  5120,     1,     1,     1 | F32     | vis.blk.13.mpl_fc1.bias
    240:    6553600 |  1280,  5120,     1,     1 | F16     | vis.blk.13.mpl_fc1.weight
    241:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.13.mpl_fc2.bias
    242:    6553600 |  5120,  1280,     1,     1 | F16     | vis.blk.13.mpl_fc2.weight
    243:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.13.post_attn_norm.bias
    244:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.13.post_attn_norm.weight
    245:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.13.attn_k_proj.weight
    246:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.13.attn_out_proj.weight
    247:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.13.attn_q_proj.weight
    248:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.13.attn_v_proj.weight
    249:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.14.input_norm.bias
    250:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.14.input_norm.weight
    251:       5120 |  5120,     1,     1,     1 | F32     | vis.blk.14.mpl_fc1.bias
    252:    6553600 |  1280,  5120,     1,     1 | F16     | vis.blk.14.mpl_fc1.weight
    253:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.14.mpl_fc2.bias
    254:    6553600 |  5120,  1280,     1,     1 | F16     | vis.blk.14.mpl_fc2.weight
    255:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.14.post_attn_norm.bias
    256:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.14.post_attn_norm.weight
    257:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.14.attn_k_proj.weight
    258:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.14.attn_out_proj.weight
    259:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.14.attn_q_proj.weight
    260:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.14.attn_v_proj.weight
    261:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.15.input_norm.bias
    262:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.15.input_norm.weight
    263:       5120 |  5120,     1,     1,     1 | F32     | vis.blk.15.mpl_fc1.bias
    264:    6553600 |  1280,  5120,     1,     1 | F16     | vis.blk.15.mpl_fc1.weight
    265:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.15.mpl_fc2.bias
    266:    6553600 |  5120,  1280,     1,     1 | F16     | vis.blk.15.mpl_fc2.weight
    267:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.15.post_attn_norm.bias
    268:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.15.post_attn_norm.weight
    269:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.15.attn_k_proj.weight
    270:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.15.attn_out_proj.weight
    271:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.15.attn_q_proj.weight
    272:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.15.attn_v_proj.weight
    273:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.16.input_norm.bias
    274:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.16.input_norm.weight
    275:       5120 |  5120,     1,     1,     1 | F32     | vis.blk.16.mpl_fc1.bias
    276:    6553600 |  1280,  5120,     1,     1 | F16     | vis.blk.16.mpl_fc1.weight
    277:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.16.mpl_fc2.bias
    278:    6553600 |  5120,  1280,     1,     1 | F16     | vis.blk.16.mpl_fc2.weight
    279:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.16.post_attn_norm.bias
    280:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.16.post_attn_norm.weight
    281:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.16.attn_k_proj.weight
    282:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.16.attn_out_proj.weight
    283:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.16.attn_q_proj.weight
    284:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.16.attn_v_proj.weight
    285:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.17.input_norm.bias
    286:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.17.input_norm.weight
    287:       5120 |  5120,     1,     1,     1 | F32     | vis.blk.17.mpl_fc1.bias
    288:    6553600 |  1280,  5120,     1,     1 | F16     | vis.blk.17.mpl_fc1.weight
    289:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.17.mpl_fc2.bias
    290:    6553600 |  5120,  1280,     1,     1 | F16     | vis.blk.17.mpl_fc2.weight
    291:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.17.post_attn_norm.bias
    292:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.17.post_attn_norm.weight
    293:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.17.attn_k_proj.weight
    294:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.17.attn_out_proj.weight
    295:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.17.attn_q_proj.weight
    296:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.17.attn_v_proj.weight
    297:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.18.input_norm.bias
    298:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.18.input_norm.weight
    299:       5120 |  5120,     1,     1,     1 | F32     | vis.blk.18.mpl_fc1.bias
    300:    6553600 |  1280,  5120,     1,     1 | F16     | vis.blk.18.mpl_fc1.weight
    301:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.18.mpl_fc2.bias
    302:    6553600 |  5120,  1280,     1,     1 | F16     | vis.blk.18.mpl_fc2.weight
    303:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.18.post_attn_norm.bias
    304:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.18.post_attn_norm.weight
    305:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.18.attn_k_proj.weight
    306:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.18.attn_out_proj.weight
    307:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.18.attn_q_proj.weight
    308:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.18.attn_v_proj.weight
    309:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.19.input_norm.bias
    310:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.19.input_norm.weight
    311:       5120 |  5120,     1,     1,     1 | F32     | vis.blk.19.mpl_fc1.bias
    312:    6553600 |  1280,  5120,     1,     1 | F16     | vis.blk.19.mpl_fc1.weight
    313:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.19.mpl_fc2.bias
    314:    6553600 |  5120,  1280,     1,     1 | F16     | vis.blk.19.mpl_fc2.weight
    315:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.19.post_attn_norm.bias
    316:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.19.post_attn_norm.weight
    317:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.19.attn_k_proj.weight
    318:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.19.attn_out_proj.weight
    319:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.19.attn_q_proj.weight
    320:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.19.attn_v_proj.weight
    321:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.2.input_norm.bias
    322:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.2.input_norm.weight
    323:       5120 |  5120,     1,     1,     1 | F32     | vis.blk.2.mpl_fc1.bias
    324:    6553600 |  1280,  5120,     1,     1 | F16     | vis.blk.2.mpl_fc1.weight
    325:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.2.mpl_fc2.bias
    326:    6553600 |  5120,  1280,     1,     1 | F16     | vis.blk.2.mpl_fc2.weight
    327:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.2.post_attn_norm.bias
    328:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.2.post_attn_norm.weight
    329:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.2.attn_k_proj.weight
    330:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.2.attn_out_proj.weight
    331:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.2.attn_q_proj.weight
    332:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.2.attn_v_proj.weight
    333:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.20.input_norm.bias
    334:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.20.input_norm.weight
    335:       5120 |  5120,     1,     1,     1 | F32     | vis.blk.20.mpl_fc1.bias
    336:    6553600 |  1280,  5120,     1,     1 | F16     | vis.blk.20.mpl_fc1.weight
    337:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.20.mpl_fc2.bias
    338:    6553600 |  5120,  1280,     1,     1 | F16     | vis.blk.20.mpl_fc2.weight
    339:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.20.post_attn_norm.bias
    340:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.20.post_attn_norm.weight
    341:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.20.attn_k_proj.weight
    342:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.20.attn_out_proj.weight
    343:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.20.attn_q_proj.weight
    344:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.20.attn_v_proj.weight
    345:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.21.input_norm.bias
    346:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.21.input_norm.weight
    347:       5120 |  5120,     1,     1,     1 | F32     | vis.blk.21.mpl_fc1.bias
    348:    6553600 |  1280,  5120,     1,     1 | F16     | vis.blk.21.mpl_fc1.weight
    349:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.21.mpl_fc2.bias
    350:    6553600 |  5120,  1280,     1,     1 | F16     | vis.blk.21.mpl_fc2.weight
    351:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.21.post_attn_norm.bias
    352:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.21.post_attn_norm.weight
    353:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.21.attn_k_proj.weight
    354:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.21.attn_out_proj.weight
    355:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.21.attn_q_proj.weight
    356:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.21.attn_v_proj.weight
    357:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.22.input_norm.bias
    358:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.22.input_norm.weight
    359:       5120 |  5120,     1,     1,     1 | F32     | vis.blk.22.mpl_fc1.bias
    360:    6553600 |  1280,  5120,     1,     1 | F16     | vis.blk.22.mpl_fc1.weight
    361:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.22.mpl_fc2.bias
    362:    6553600 |  5120,  1280,     1,     1 | F16     | vis.blk.22.mpl_fc2.weight
    363:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.22.post_attn_norm.bias
    364:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.22.post_attn_norm.weight
    365:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.22.attn_k_proj.weight
    366:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.22.attn_out_proj.weight
    367:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.22.attn_q_proj.weight
    368:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.22.attn_v_proj.weight
    369:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.23.input_norm.bias
    370:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.23.input_norm.weight
    371:       5120 |  5120,     1,     1,     1 | F32     | vis.blk.23.mpl_fc1.bias
    372:    6553600 |  1280,  5120,     1,     1 | F16     | vis.blk.23.mpl_fc1.weight
    373:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.23.mpl_fc2.bias
    374:    6553600 |  5120,  1280,     1,     1 | F16     | vis.blk.23.mpl_fc2.weight
    375:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.23.post_attn_norm.bias
    376:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.23.post_attn_norm.weight
    377:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.23.attn_k_proj.weight
    378:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.23.attn_out_proj.weight
    379:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.23.attn_q_proj.weight
    380:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.23.attn_v_proj.weight
    381:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.24.input_norm.bias
    382:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.24.input_norm.weight
    383:       5120 |  5120,     1,     1,     1 | F32     | vis.blk.24.mpl_fc1.bias
    384:    6553600 |  1280,  5120,     1,     1 | F16     | vis.blk.24.mpl_fc1.weight
    385:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.24.mpl_fc2.bias
    386:    6553600 |  5120,  1280,     1,     1 | F16     | vis.blk.24.mpl_fc2.weight
    387:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.24.post_attn_norm.bias
    388:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.24.post_attn_norm.weight
    389:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.24.attn_k_proj.weight
    390:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.24.attn_out_proj.weight
    391:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.24.attn_q_proj.weight
    392:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.24.attn_v_proj.weight
    393:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.25.input_norm.bias
    394:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.25.input_norm.weight
    395:       5120 |  5120,     1,     1,     1 | F32     | vis.blk.25.mpl_fc1.bias
    396:    6553600 |  1280,  5120,     1,     1 | F16     | vis.blk.25.mpl_fc1.weight
    397:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.25.mpl_fc2.bias
    398:    6553600 |  5120,  1280,     1,     1 | F16     | vis.blk.25.mpl_fc2.weight
    399:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.25.post_attn_norm.bias
    400:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.25.post_attn_norm.weight
    401:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.25.attn_k_proj.weight
    402:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.25.attn_out_proj.weight
    403:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.25.attn_q_proj.weight
    404:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.25.attn_v_proj.weight
    405:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.26.input_norm.bias
    406:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.26.input_norm.weight
    407:       5120 |  5120,     1,     1,     1 | F32     | vis.blk.26.mpl_fc1.bias
    408:    6553600 |  1280,  5120,     1,     1 | F16     | vis.blk.26.mpl_fc1.weight
    409:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.26.mpl_fc2.bias
    410:    6553600 |  5120,  1280,     1,     1 | F16     | vis.blk.26.mpl_fc2.weight
    411:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.26.post_attn_norm.bias
    412:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.26.post_attn_norm.weight
    413:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.26.attn_k_proj.weight
    414:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.26.attn_out_proj.weight
    415:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.26.attn_q_proj.weight
    416:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.26.attn_v_proj.weight
    417:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.27.input_norm.bias
    418:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.27.input_norm.weight
    419:       5120 |  5120,     1,     1,     1 | F32     | vis.blk.27.mpl_fc1.bias
    420:    6553600 |  1280,  5120,     1,     1 | F16     | vis.blk.27.mpl_fc1.weight
    421:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.27.mpl_fc2.bias
    422:    6553600 |  5120,  1280,     1,     1 | F16     | vis.blk.27.mpl_fc2.weight
    423:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.27.post_attn_norm.bias
    424:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.27.post_attn_norm.weight
    425:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.27.attn_k_proj.weight
    426:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.27.attn_out_proj.weight
    427:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.27.attn_q_proj.weight
    428:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.27.attn_v_proj.weight
    429:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.28.input_norm.bias
    430:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.28.input_norm.weight
    431:       5120 |  5120,     1,     1,     1 | F32     | vis.blk.28.mpl_fc1.bias
    432:    6553600 |  1280,  5120,     1,     1 | F16     | vis.blk.28.mpl_fc1.weight
    433:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.28.mpl_fc2.bias
    434:    6553600 |  5120,  1280,     1,     1 | F16     | vis.blk.28.mpl_fc2.weight
    435:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.28.post_attn_norm.bias
    436:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.28.post_attn_norm.weight
    437:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.28.attn_k_proj.weight
    438:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.28.attn_out_proj.weight
    439:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.28.attn_q_proj.weight
    440:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.28.attn_v_proj.weight
    441:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.29.input_norm.bias
    442:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.29.input_norm.weight
    443:       5120 |  5120,     1,     1,     1 | F32     | vis.blk.29.mpl_fc1.bias
    444:    6553600 |  1280,  5120,     1,     1 | F16     | vis.blk.29.mpl_fc1.weight
    445:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.29.mpl_fc2.bias
    446:    6553600 |  5120,  1280,     1,     1 | F16     | vis.blk.29.mpl_fc2.weight
    447:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.29.post_attn_norm.bias
    448:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.29.post_attn_norm.weight
    449:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.29.attn_k_proj.weight
    450:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.29.attn_out_proj.weight
    451:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.29.attn_q_proj.weight
    452:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.29.attn_v_proj.weight
    453:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.3.input_norm.bias
    454:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.3.input_norm.weight
    455:       5120 |  5120,     1,     1,     1 | F32     | vis.blk.3.mpl_fc1.bias
    456:    6553600 |  1280,  5120,     1,     1 | F16     | vis.blk.3.mpl_fc1.weight
    457:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.3.mpl_fc2.bias
    458:    6553600 |  5120,  1280,     1,     1 | F16     | vis.blk.3.mpl_fc2.weight
    459:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.3.post_attn_norm.bias
    460:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.3.post_attn_norm.weight
    461:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.3.attn_k_proj.weight
    462:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.3.attn_out_proj.weight
    463:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.3.attn_q_proj.weight
    464:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.3.attn_v_proj.weight
    465:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.30.input_norm.bias
    466:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.30.input_norm.weight
    467:       5120 |  5120,     1,     1,     1 | F32     | vis.blk.30.mpl_fc1.bias
    468:    6553600 |  1280,  5120,     1,     1 | F16     | vis.blk.30.mpl_fc1.weight
    469:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.30.mpl_fc2.bias
    470:    6553600 |  5120,  1280,     1,     1 | F16     | vis.blk.30.mpl_fc2.weight
    471:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.30.post_attn_norm.bias
    472:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.30.post_attn_norm.weight
    473:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.30.attn_k_proj.weight
    474:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.30.attn_out_proj.weight
    475:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.30.attn_q_proj.weight
    476:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.30.attn_v_proj.weight
    477:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.31.input_norm.bias
    478:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.31.input_norm.weight
    479:       5120 |  5120,     1,     1,     1 | F32     | vis.blk.31.mpl_fc1.bias
    480:    6553600 |  1280,  5120,     1,     1 | F16     | vis.blk.31.mpl_fc1.weight
    481:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.31.mpl_fc2.bias
    482:    6553600 |  5120,  1280,     1,     1 | F16     | vis.blk.31.mpl_fc2.weight
    483:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.31.post_attn_norm.bias
    484:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.31.post_attn_norm.weight
    485:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.31.attn_k_proj.weight
    486:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.31.attn_out_proj.weight
    487:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.31.attn_q_proj.weight
    488:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.31.attn_v_proj.weight
    489:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.4.input_norm.bias
    490:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.4.input_norm.weight
    491:       5120 |  5120,     1,     1,     1 | F32     | vis.blk.4.mpl_fc1.bias
    492:    6553600 |  1280,  5120,     1,     1 | F16     | vis.blk.4.mpl_fc1.weight
    493:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.4.mpl_fc2.bias
    494:    6553600 |  5120,  1280,     1,     1 | F16     | vis.blk.4.mpl_fc2.weight
    495:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.4.post_attn_norm.bias
    496:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.4.post_attn_norm.weight
    497:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.4.attn_k_proj.weight
    498:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.4.attn_out_proj.weight
    499:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.4.attn_q_proj.weight
    500:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.4.attn_v_proj.weight
    501:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.5.input_norm.bias
    502:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.5.input_norm.weight
    503:       5120 |  5120,     1,     1,     1 | F32     | vis.blk.5.mpl_fc1.bias
    504:    6553600 |  1280,  5120,     1,     1 | F16     | vis.blk.5.mpl_fc1.weight
    505:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.5.mpl_fc2.bias
    506:    6553600 |  5120,  1280,     1,     1 | F16     | vis.blk.5.mpl_fc2.weight
    507:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.5.post_attn_norm.bias
    508:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.5.post_attn_norm.weight
    509:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.5.attn_k_proj.weight
    510:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.5.attn_out_proj.weight
    511:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.5.attn_q_proj.weight
    512:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.5.attn_v_proj.weight
    513:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.6.input_norm.bias
    514:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.6.input_norm.weight
    515:       5120 |  5120,     1,     1,     1 | F32     | vis.blk.6.mpl_fc1.bias
    516:    6553600 |  1280,  5120,     1,     1 | F16     | vis.blk.6.mpl_fc1.weight
    517:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.6.mpl_fc2.bias
    518:    6553600 |  5120,  1280,     1,     1 | F16     | vis.blk.6.mpl_fc2.weight
    519:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.6.post_attn_norm.bias
    520:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.6.post_attn_norm.weight
    521:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.6.attn_k_proj.weight
    522:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.6.attn_out_proj.weight
    523:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.6.attn_q_proj.weight
    524:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.6.attn_v_proj.weight
    525:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.7.input_norm.bias
    526:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.7.input_norm.weight
    527:       5120 |  5120,     1,     1,     1 | F32     | vis.blk.7.mpl_fc1.bias
    528:    6553600 |  1280,  5120,     1,     1 | F16     | vis.blk.7.mpl_fc1.weight
    529:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.7.mpl_fc2.bias
    530:    6553600 |  5120,  1280,     1,     1 | F16     | vis.blk.7.mpl_fc2.weight
    531:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.7.post_attn_norm.bias
    532:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.7.post_attn_norm.weight
    533:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.7.attn_k_proj.weight
    534:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.7.attn_out_proj.weight
    535:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.7.attn_q_proj.weight
    536:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.7.attn_v_proj.weight
    537:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.8.input_norm.bias
    538:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.8.input_norm.weight
    539:       5120 |  5120,     1,     1,     1 | F32     | vis.blk.8.mpl_fc1.bias
    540:    6553600 |  1280,  5120,     1,     1 | F16     | vis.blk.8.mpl_fc1.weight
    541:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.8.mpl_fc2.bias
    542:    6553600 |  5120,  1280,     1,     1 | F16     | vis.blk.8.mpl_fc2.weight
    543:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.8.post_attn_norm.bias
    544:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.8.post_attn_norm.weight
    545:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.8.attn_k_proj.weight
    546:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.8.attn_out_proj.weight
    547:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.8.attn_q_proj.weight
    548:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.8.attn_v_proj.weight
    549:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.9.input_norm.bias
    550:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.9.input_norm.weight
    551:       5120 |  5120,     1,     1,     1 | F32     | vis.blk.9.mpl_fc1.bias
    552:    6553600 |  1280,  5120,     1,     1 | F16     | vis.blk.9.mpl_fc1.weight
    553:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.9.mpl_fc2.bias
    554:    6553600 |  5120,  1280,     1,     1 | F16     | vis.blk.9.mpl_fc2.weight
    555:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.9.post_attn_norm.bias
    556:       1280 |  1280,     1,     1,     1 | F32     | vis.blk.9.post_attn_norm.weight
    557:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.9.attn_k_proj.weight
    558:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.9.attn_out_proj.weight
    559:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.9.attn_q_proj.weight
    560:    1638400 |  1280,  1280,     1,     1 | F16     | vis.blk.9.attn_v_proj.weight
    561:       4096 |  4096,     1,     1,     1 | F32     | blk.10.attn_norm.weight
    562:   58720256 | 14336,  4096,     1,     1 | F16     | blk.10.ffn_down.weight
    563:   58720256 |  4096, 14336,     1,     1 | F16     | blk.10.ffn_gate.weight
    564:   58720256 |  4096, 14336,     1,     1 | F16     | blk.10.ffn_up.weight
    565:       4096 |  4096,     1,     1,     1 | F32     | blk.10.post_attention_norm.weight
    566:    4194304 |  4096,  1024,     1,     1 | F16     | blk.10.attn_k.weight
    567:   16777216 |  4096,  4096,     1,     1 | F16     | blk.10.attn_output.weight
    568:   16777216 |  4096,  4096,     1,     1 | F16     | blk.10.attn_q.weight
    569:    4194304 |  4096,  1024,     1,     1 | F16     | blk.10.attn_v.weight
    570:       4096 |  4096,     1,     1,     1 | F32     | blk.11.attn_norm.weight
    571:   58720256 | 14336,  4096,     1,     1 | F16     | blk.11.ffn_down.weight
    572:   58720256 |  4096, 14336,     1,     1 | F16     | blk.11.ffn_gate.weight
    573:   58720256 |  4096, 14336,     1,     1 | F16     | blk.11.ffn_up.weight
    574:       4096 |  4096,     1,     1,     1 | F32     | blk.11.post_attention_norm.weight
    575:    4194304 |  4096,  1024,     1,     1 | F16     | blk.11.attn_k.weight
    576:   16777216 |  4096,  4096,     1,     1 | F16     | blk.11.attn_output.weight
    577:   16777216 |  4096,  4096,     1,     1 | F16     | blk.11.attn_q.weight
    578:    4194304 |  4096,  1024,     1,     1 | F16     | blk.11.attn_v.weight
    579:       4096 |  4096,     1,     1,     1 | F32     | blk.12.attn_norm.weight
    580:   58720256 | 14336,  4096,     1,     1 | F16     | blk.12.ffn_down.weight
    581:   58720256 |  4096, 14336,     1,     1 | F16     | blk.12.ffn_gate.weight
    582:   58720256 |  4096, 14336,     1,     1 | F16     | blk.12.ffn_up.weight
    583:       4096 |  4096,     1,     1,     1 | F32     | blk.12.post_attention_norm.weight
    584:    4194304 |  4096,  1024,     1,     1 | F16     | blk.12.attn_k.weight
    585:   16777216 |  4096,  4096,     1,     1 | F16     | blk.12.attn_output.weight
    586:   16777216 |  4096,  4096,     1,     1 | F16     | blk.12.attn_q.weight
    587:    4194304 |  4096,  1024,     1,     1 | F16     | blk.12.attn_v.weight
    588:        128 |   128,     1,     1,     1 | F32     | blk.13.cross_attn_k_norm.weight
    589:    4194304 |  4096,  1024,     1,     1 | F16     | blk.13.cross_attn_k_proj.weight
    590:   16777216 |  4096,  4096,     1,     1 | F16     | blk.13.cross_attn_output_proj.weight
    591:        128 |   128,     1,     1,     1 | F32     | blk.13.cross_attn_q_norm.weight
    592:   16777216 |  4096,  4096,     1,     1 | F16     | blk.13.cross_attn_q_proj.weight
    593:    4194304 |  4096,  1024,     1,     1 | F16     | blk.13.cross_attn_v_proj.weight
    594:          1 |     1,     1,     1,     1 | F32     | blk.13.cross_attn_attn_gate
    595:          1 |     1,     1,     1,     1 | F32     | blk.13.cross_attn_mpl_gate
    596:       4096 |  4096,     1,     1,     1 | F32     | blk.13.attn_norm.weight
    597:   58720256 | 14336,  4096,     1,     1 | F16     | blk.13.ffn_down.weight
    598:   58720256 |  4096, 14336,     1,     1 | F16     | blk.13.ffn_gate.weight
    599:   58720256 |  4096, 14336,     1,     1 | F16     | blk.13.ffn_up.weight
    600:       4096 |  4096,     1,     1,     1 | F32     | blk.13.post_attention_norm.weight
    601:       4096 |  4096,     1,     1,     1 | F32     | blk.14.attn_norm.weight
    602:   58720256 | 14336,  4096,     1,     1 | F16     | blk.14.ffn_down.weight
    603:   58720256 |  4096, 14336,     1,     1 | F16     | blk.14.ffn_gate.weight
    604:   58720256 |  4096, 14336,     1,     1 | F16     | blk.14.ffn_up.weight
    605:       4096 |  4096,     1,     1,     1 | F32     | blk.14.post_attention_norm.weight
    606:    4194304 |  4096,  1024,     1,     1 | F16     | blk.14.attn_k.weight
    607:   16777216 |  4096,  4096,     1,     1 | F16     | blk.14.attn_output.weight
    608:   16777216 |  4096,  4096,     1,     1 | F16     | blk.14.attn_q.weight
    609:    4194304 |  4096,  1024,     1,     1 | F16     | blk.14.attn_v.weight
    610:       4096 |  4096,     1,     1,     1 | F32     | blk.15.attn_norm.weight
    611:   58720256 | 14336,  4096,     1,     1 | F16     | blk.15.ffn_down.weight
    612:   58720256 |  4096, 14336,     1,     1 | F16     | blk.15.ffn_gate.weight
    613:   58720256 |  4096, 14336,     1,     1 | F16     | blk.15.ffn_up.weight
    614:       4096 |  4096,     1,     1,     1 | F32     | blk.15.post_attention_norm.weight
    615:    4194304 |  4096,  1024,     1,     1 | F16     | blk.15.attn_k.weight
    616:   16777216 |  4096,  4096,     1,     1 | F16     | blk.15.attn_output.weight
    617:   16777216 |  4096,  4096,     1,     1 | F16     | blk.15.attn_q.weight
    618:    4194304 |  4096,  1024,     1,     1 | F16     | blk.15.attn_v.weight
    619:   58720256 |  4096, 14336,     1,     1 | F16     | blk.16.ffn_gate.weight
    620:    4194304 |  4096,  1024,     1,     1 | F16     | blk.16.attn_k.weight
    621:   16777216 |  4096,  4096,     1,     1 | F16     | blk.16.attn_output.weight
    622:   16777216 |  4096,  4096,     1,     1 | F16     | blk.16.attn_q.weight
    623:    4194304 |  4096,  1024,     1,     1 | F16     | blk.16.attn_v.weight
    624:       4096 |  4096,     1,     1,     1 | F32     | blk.5.attn_norm.weight
    625:   58720256 | 14336,  4096,     1,     1 | F16     | blk.5.ffn_down.weight
    626:   58720256 |  4096, 14336,     1,     1 | F16     | blk.5.ffn_gate.weight
    627:   58720256 |  4096, 14336,     1,     1 | F16     | blk.5.ffn_up.weight
    628:       4096 |  4096,     1,     1,     1 | F32     | blk.5.post_attention_norm.weight
    629:    4194304 |  4096,  1024,     1,     1 | F16     | blk.5.attn_k.weight
    630:   16777216 |  4096,  4096,     1,     1 | F16     | blk.5.attn_output.weight
    631:    4194304 |  4096,  1024,     1,     1 | F16     | blk.5.attn_v.weight
    632:       4096 |  4096,     1,     1,     1 | F32     | blk.6.attn_norm.weight
    633:   58720256 | 14336,  4096,     1,     1 | F16     | blk.6.ffn_down.weight
    634:   58720256 |  4096, 14336,     1,     1 | F16     | blk.6.ffn_gate.weight
    635:   58720256 |  4096, 14336,     1,     1 | F16     | blk.6.ffn_up.weight
    636:       4096 |  4096,     1,     1,     1 | F32     | blk.6.post_attention_norm.weight
    637:    4194304 |  4096,  1024,     1,     1 | F16     | blk.6.attn_k.weight
    638:   16777216 |  4096,  4096,     1,     1 | F16     | blk.6.attn_output.weight
    639:   16777216 |  4096,  4096,     1,     1 | F16     | blk.6.attn_q.weight
    640:    4194304 |  4096,  1024,     1,     1 | F16     | blk.6.attn_v.weight
    641:       4096 |  4096,     1,     1,     1 | F32     | blk.7.attn_norm.weight
    642:   58720256 | 14336,  4096,     1,     1 | F16     | blk.7.ffn_down.weight
    643:   58720256 |  4096, 14336,     1,     1 | F16     | blk.7.ffn_gate.weight
    644:   58720256 |  4096, 14336,     1,     1 | F16     | blk.7.ffn_up.weight
    645:       4096 |  4096,     1,     1,     1 | F32     | blk.7.post_attention_norm.weight
    646:    4194304 |  4096,  1024,     1,     1 | F16     | blk.7.attn_k.weight
    647:   16777216 |  4096,  4096,     1,     1 | F16     | blk.7.attn_output.weight
    648:   16777216 |  4096,  4096,     1,     1 | F16     | blk.7.attn_q.weight
    649:    4194304 |  4096,  1024,     1,     1 | F16     | blk.7.attn_v.weight
    650:        128 |   128,     1,     1,     1 | F32     | blk.8.cross_attn_k_norm.weight
    651:    4194304 |  4096,  1024,     1,     1 | F16     | blk.8.cross_attn_k_proj.weight
    652:   16777216 |  4096,  4096,     1,     1 | F16     | blk.8.cross_attn_output_proj.weight
    653:        128 |   128,     1,     1,     1 | F32     | blk.8.cross_attn_q_norm.weight
    654:   16777216 |  4096,  4096,     1,     1 | F16     | blk.8.cross_attn_q_proj.weight
    655:    4194304 |  4096,  1024,     1,     1 | F16     | blk.8.cross_attn_v_proj.weight
    656:          1 |     1,     1,     1,     1 | F32     | blk.8.cross_attn_attn_gate
    657:          1 |     1,     1,     1,     1 | F32     | blk.8.cross_attn_mpl_gate
    658:       4096 |  4096,     1,     1,     1 | F32     | blk.8.attn_norm.weight
    659:   58720256 | 14336,  4096,     1,     1 | F16     | blk.8.ffn_down.weight
    660:   58720256 |  4096, 14336,     1,     1 | F16     | blk.8.ffn_gate.weight
    661:   58720256 |  4096, 14336,     1,     1 | F16     | blk.8.ffn_up.weight
    662:       4096 |  4096,     1,     1,     1 | F32     | blk.8.post_attention_norm.weight
    663:       4096 |  4096,     1,     1,     1 | F32     | blk.9.attn_norm.weight
    664:   58720256 | 14336,  4096,     1,     1 | F16     | blk.9.ffn_down.weight
    665:   58720256 |  4096, 14336,     1,     1 | F16     | blk.9.ffn_gate.weight
    666:   58720256 |  4096, 14336,     1,     1 | F16     | blk.9.ffn_up.weight
    667:       4096 |  4096,     1,     1,     1 | F32     | blk.9.post_attention_norm.weight
    668:    4194304 |  4096,  1024,     1,     1 | F16     | blk.9.attn_k.weight
    669:   16777216 |  4096,  4096,     1,     1 | F16     | blk.9.attn_output.weight
    670:   16777216 |  4096,  4096,     1,     1 | F16     | blk.9.attn_q.weight
    671:    4194304 |  4096,  1024,     1,     1 | F16     | blk.9.attn_v.weight
    672:       4096 |  4096,     1,     1,     1 | F32     | blk.16.attn_norm.weight
    673:   58720256 | 14336,  4096,     1,     1 | F16     | blk.16.ffn_down.weight
    674:   58720256 |  4096, 14336,     1,     1 | F16     | blk.16.ffn_up.weight
    675:       4096 |  4096,     1,     1,     1 | F32     | blk.16.post_attention_norm.weight
    676:       4096 |  4096,     1,     1,     1 | F32     | blk.17.attn_norm.weight
    677:   58720256 | 14336,  4096,     1,     1 | F16     | blk.17.ffn_down.weight
    678:   58720256 |  4096, 14336,     1,     1 | F16     | blk.17.ffn_gate.weight
    679:   58720256 |  4096, 14336,     1,     1 | F16     | blk.17.ffn_up.weight
    680:       4096 |  4096,     1,     1,     1 | F32     | blk.17.post_attention_norm.weight
    681:    4194304 |  4096,  1024,     1,     1 | F16     | blk.17.attn_k.weight
    682:   16777216 |  4096,  4096,     1,     1 | F16     | blk.17.attn_output.weight
    683:   16777216 |  4096,  4096,     1,     1 | F16     | blk.17.attn_q.weight
    684:    4194304 |  4096,  1024,     1,     1 | F16     | blk.17.attn_v.weight
    685:        128 |   128,     1,     1,     1 | F32     | blk.18.cross_attn_k_norm.weight
    686:    4194304 |  4096,  1024,     1,     1 | F16     | blk.18.cross_attn_k_proj.weight
    687:   16777216 |  4096,  4096,     1,     1 | F16     | blk.18.cross_attn_output_proj.weight
    688:        128 |   128,     1,     1,     1 | F32     | blk.18.cross_attn_q_norm.weight
    689:   16777216 |  4096,  4096,     1,     1 | F16     | blk.18.cross_attn_q_proj.weight
    690:    4194304 |  4096,  1024,     1,     1 | F16     | blk.18.cross_attn_v_proj.weight
    691:          1 |     1,     1,     1,     1 | F32     | blk.18.cross_attn_attn_gate
    692:          1 |     1,     1,     1,     1 | F32     | blk.18.cross_attn_mpl_gate
    693:       4096 |  4096,     1,     1,     1 | F32     | blk.18.attn_norm.weight
    694:   58720256 | 14336,  4096,     1,     1 | F16     | blk.18.ffn_down.weight
    695:   58720256 |  4096, 14336,     1,     1 | F16     | blk.18.ffn_gate.weight
    696:   58720256 |  4096, 14336,     1,     1 | F16     | blk.18.ffn_up.weight
    697:       4096 |  4096,     1,     1,     1 | F32     | blk.18.post_attention_norm.weight
    698:       4096 |  4096,     1,     1,     1 | F32     | blk.19.attn_norm.weight
    699:   58720256 | 14336,  4096,     1,     1 | F16     | blk.19.ffn_down.weight
    700:   58720256 |  4096, 14336,     1,     1 | F16     | blk.19.ffn_gate.weight
    701:   58720256 |  4096, 14336,     1,     1 | F16     | blk.19.ffn_up.weight
    702:       4096 |  4096,     1,     1,     1 | F32     | blk.19.post_attention_norm.weight
    703:    4194304 |  4096,  1024,     1,     1 | F16     | blk.19.attn_k.weight
    704:   16777216 |  4096,  4096,     1,     1 | F16     | blk.19.attn_output.weight
    705:   16777216 |  4096,  4096,     1,     1 | F16     | blk.19.attn_q.weight
    706:    4194304 |  4096,  1024,     1,     1 | F16     | blk.19.attn_v.weight
    707:       4096 |  4096,     1,     1,     1 | F32     | blk.20.attn_norm.weight
    708:   58720256 | 14336,  4096,     1,     1 | F16     | blk.20.ffn_down.weight
    709:   58720256 |  4096, 14336,     1,     1 | F16     | blk.20.ffn_gate.weight
    710:   58720256 |  4096, 14336,     1,     1 | F16     | blk.20.ffn_up.weight
    711:       4096 |  4096,     1,     1,     1 | F32     | blk.20.post_attention_norm.weight
    712:    4194304 |  4096,  1024,     1,     1 | F16     | blk.20.attn_k.weight
    713:   16777216 |  4096,  4096,     1,     1 | F16     | blk.20.attn_output.weight
    714:   16777216 |  4096,  4096,     1,     1 | F16     | blk.20.attn_q.weight
    715:    4194304 |  4096,  1024,     1,     1 | F16     | blk.20.attn_v.weight
    716:       4096 |  4096,     1,     1,     1 | F32     | blk.21.attn_norm.weight
    717:   58720256 | 14336,  4096,     1,     1 | F16     | blk.21.ffn_down.weight
    718:   58720256 |  4096, 14336,     1,     1 | F16     | blk.21.ffn_gate.weight
    719:   58720256 |  4096, 14336,     1,     1 | F16     | blk.21.ffn_up.weight
    720:       4096 |  4096,     1,     1,     1 | F32     | blk.21.post_attention_norm.weight
    721:    4194304 |  4096,  1024,     1,     1 | F16     | blk.21.attn_k.weight
    722:   16777216 |  4096,  4096,     1,     1 | F16     | blk.21.attn_output.weight
    723:   16777216 |  4096,  4096,     1,     1 | F16     | blk.21.attn_q.weight
    724:    4194304 |  4096,  1024,     1,     1 | F16     | blk.21.attn_v.weight
    725:       4096 |  4096,     1,     1,     1 | F32     | blk.22.attn_norm.weight
    726:   58720256 | 14336,  4096,     1,     1 | F16     | blk.22.ffn_down.weight
    727:   58720256 |  4096, 14336,     1,     1 | F16     | blk.22.ffn_gate.weight
    728:   58720256 |  4096, 14336,     1,     1 | F16     | blk.22.ffn_up.weight
    729:       4096 |  4096,     1,     1,     1 | F32     | blk.22.post_attention_norm.weight
    730:    4194304 |  4096,  1024,     1,     1 | F16     | blk.22.attn_k.weight
    731:   16777216 |  4096,  4096,     1,     1 | F16     | blk.22.attn_output.weight
    732:   16777216 |  4096,  4096,     1,     1 | F16     | blk.22.attn_q.weight
    733:    4194304 |  4096,  1024,     1,     1 | F16     | blk.22.attn_v.weight
    734:        128 |   128,     1,     1,     1 | F32     | blk.23.cross_attn_k_norm.weight
    735:    4194304 |  4096,  1024,     1,     1 | F16     | blk.23.cross_attn_k_proj.weight
    736:   16777216 |  4096,  4096,     1,     1 | F16     | blk.23.cross_attn_output_proj.weight
    737:        128 |   128,     1,     1,     1 | F32     | blk.23.cross_attn_q_norm.weight
    738:   16777216 |  4096,  4096,     1,     1 | F16     | blk.23.cross_attn_q_proj.weight
    739:    4194304 |  4096,  1024,     1,     1 | F16     | blk.23.cross_attn_v_proj.weight
    740:          1 |     1,     1,     1,     1 | F32     | blk.23.cross_attn_attn_gate
    741:          1 |     1,     1,     1,     1 | F32     | blk.23.cross_attn_mpl_gate
    742:       4096 |  4096,     1,     1,     1 | F32     | blk.23.attn_norm.weight
    743:   58720256 | 14336,  4096,     1,     1 | F16     | blk.23.ffn_down.weight
    744:   58720256 |  4096, 14336,     1,     1 | F16     | blk.23.ffn_gate.weight
    745:   58720256 |  4096, 14336,     1,     1 | F16     | blk.23.ffn_up.weight
    746:       4096 |  4096,     1,     1,     1 | F32     | blk.23.post_attention_norm.weight
    747:       4096 |  4096,     1,     1,     1 | F32     | blk.24.attn_norm.weight
    748:   58720256 | 14336,  4096,     1,     1 | F16     | blk.24.ffn_down.weight
    749:   58720256 |  4096, 14336,     1,     1 | F16     | blk.24.ffn_gate.weight
    750:   58720256 |  4096, 14336,     1,     1 | F16     | blk.24.ffn_up.weight
    751:       4096 |  4096,     1,     1,     1 | F32     | blk.24.post_attention_norm.weight
    752:    4194304 |  4096,  1024,     1,     1 | F16     | blk.24.attn_k.weight
    753:   16777216 |  4096,  4096,     1,     1 | F16     | blk.24.attn_output.weight
    754:   16777216 |  4096,  4096,     1,     1 | F16     | blk.24.attn_q.weight
    755:    4194304 |  4096,  1024,     1,     1 | F16     | blk.24.attn_v.weight
    756:       4096 |  4096,     1,     1,     1 | F32     | blk.25.attn_norm.weight
    757:   58720256 | 14336,  4096,     1,     1 | F16     | blk.25.ffn_down.weight
    758:   58720256 |  4096, 14336,     1,     1 | F16     | blk.25.ffn_gate.weight
    759:   58720256 |  4096, 14336,     1,     1 | F16     | blk.25.ffn_up.weight
    760:       4096 |  4096,     1,     1,     1 | F32     | blk.25.post_attention_norm.weight
    761:    4194304 |  4096,  1024,     1,     1 | F16     | blk.25.attn_k.weight
    762:   16777216 |  4096,  4096,     1,     1 | F16     | blk.25.attn_output.weight
    763:   16777216 |  4096,  4096,     1,     1 | F16     | blk.25.attn_q.weight
    764:    4194304 |  4096,  1024,     1,     1 | F16     | blk.25.attn_v.weight
    765:       4096 |  4096,     1,     1,     1 | F32     | blk.26.attn_norm.weight
    766:   58720256 | 14336,  4096,     1,     1 | F16     | blk.26.ffn_down.weight
    767:   58720256 |  4096, 14336,     1,     1 | F16     | blk.26.ffn_gate.weight
    768:   58720256 |  4096, 14336,     1,     1 | F16     | blk.26.ffn_up.weight
    769:       4096 |  4096,     1,     1,     1 | F32     | blk.26.post_attention_norm.weight
    770:    4194304 |  4096,  1024,     1,     1 | F16     | blk.26.attn_k.weight
    771:   16777216 |  4096,  4096,     1,     1 | F16     | blk.26.attn_output.weight
    772:   16777216 |  4096,  4096,     1,     1 | F16     | blk.26.attn_q.weight
    773:    4194304 |  4096,  1024,     1,     1 | F16     | blk.26.attn_v.weight
    774:   58720256 |  4096, 14336,     1,     1 | F16     | blk.27.ffn_gate.weight
    775:   58720256 |  4096, 14336,     1,     1 | F16     | blk.27.ffn_up.weight
    776:    4194304 |  4096,  1024,     1,     1 | F16     | blk.27.attn_k.weight
    777:   16777216 |  4096,  4096,     1,     1 | F16     | blk.27.attn_output.weight
    778:   16777216 |  4096,  4096,     1,     1 | F16     | blk.27.attn_q.weight
    779:    4194304 |  4096,  1024,     1,     1 | F16     | blk.27.attn_v.weight
    780:       4096 |  4096,     1,     1,     1 | F32     | blk.27.attn_norm.weight
    781:   58720256 | 14336,  4096,     1,     1 | F16     | blk.27.ffn_down.weight
    782:       4096 |  4096,     1,     1,     1 | F32     | blk.27.post_attention_norm.weight
    783:        128 |   128,     1,     1,     1 | F32     | blk.28.cross_attn_k_norm.weight
    784:    4194304 |  4096,  1024,     1,     1 | F16     | blk.28.cross_attn_k_proj.weight
    785:   16777216 |  4096,  4096,     1,     1 | F16     | blk.28.cross_attn_output_proj.weight
    786:        128 |   128,     1,     1,     1 | F32     | blk.28.cross_attn_q_norm.weight
    787:   16777216 |  4096,  4096,     1,     1 | F16     | blk.28.cross_attn_q_proj.weight
    788:    4194304 |  4096,  1024,     1,     1 | F16     | blk.28.cross_attn_v_proj.weight
    789:          1 |     1,     1,     1,     1 | F32     | blk.28.cross_attn_attn_gate
    790:          1 |     1,     1,     1,     1 | F32     | blk.28.cross_attn_mpl_gate
    791:       4096 |  4096,     1,     1,     1 | F32     | blk.28.attn_norm.weight
    792:   58720256 | 14336,  4096,     1,     1 | F16     | blk.28.ffn_down.weight
    793:   58720256 |  4096, 14336,     1,     1 | F16     | blk.28.ffn_gate.weight
    794:   58720256 |  4096, 14336,     1,     1 | F16     | blk.28.ffn_up.weight
    795:       4096 |  4096,     1,     1,     1 | F32     | blk.28.post_attention_norm.weight
    796:       4096 |  4096,     1,     1,     1 | F32     | blk.29.attn_norm.weight
    797:   58720256 | 14336,  4096,     1,     1 | F16     | blk.29.ffn_down.weight
    798:   58720256 |  4096, 14336,     1,     1 | F16     | blk.29.ffn_gate.weight
    799:   58720256 |  4096, 14336,     1,     1 | F16     | blk.29.ffn_up.weight
    800:       4096 |  4096,     1,     1,     1 | F32     | blk.29.post_attention_norm.weight
    801:    4194304 |  4096,  1024,     1,     1 | F16     | blk.29.attn_k.weight
    802:   16777216 |  4096,  4096,     1,     1 | F16     | blk.29.attn_output.weight
    803:   16777216 |  4096,  4096,     1,     1 | F16     | blk.29.attn_q.weight
    804:    4194304 |  4096,  1024,     1,     1 | F16     | blk.29.attn_v.weight
    805:       4096 |  4096,     1,     1,     1 | F32     | blk.30.attn_norm.weight
    806:   58720256 | 14336,  4096,     1,     1 | F16     | blk.30.ffn_down.weight
    807:   58720256 |  4096, 14336,     1,     1 | F16     | blk.30.ffn_gate.weight
    808:   58720256 |  4096, 14336,     1,     1 | F16     | blk.30.ffn_up.weight
    809:       4096 |  4096,     1,     1,     1 | F32     | blk.30.post_attention_norm.weight
    810:    4194304 |  4096,  1024,     1,     1 | F16     | blk.30.attn_k.weight
    811:   16777216 |  4096,  4096,     1,     1 | F16     | blk.30.attn_output.weight
    812:   16777216 |  4096,  4096,     1,     1 | F16     | blk.30.attn_q.weight
    813:    4194304 |  4096,  1024,     1,     1 | F16     | blk.30.attn_v.weight
    814:       4096 |  4096,     1,     1,     1 | F32     | blk.31.attn_norm.weight
    815:   58720256 | 14336,  4096,     1,     1 | F16     | blk.31.ffn_down.weight
    816:   58720256 |  4096, 14336,     1,     1 | F16     | blk.31.ffn_gate.weight
    817:   58720256 |  4096, 14336,     1,     1 | F16     | blk.31.ffn_up.weight
    818:       4096 |  4096,     1,     1,     1 | F32     | blk.31.post_attention_norm.weight
    819:    4194304 |  4096,  1024,     1,     1 | F16     | blk.31.attn_k.weight
    820:   16777216 |  4096,  4096,     1,     1 | F16     | blk.31.attn_output.weight
    821:   16777216 |  4096,  4096,     1,     1 | F16     | blk.31.attn_q.weight
    822:    4194304 |  4096,  1024,     1,     1 | F16     | blk.31.attn_v.weight
    823:       4096 |  4096,     1,     1,     1 | F32     | blk.32.attn_norm.weight
    824:   58720256 | 14336,  4096,     1,     1 | F16     | blk.32.ffn_down.weight
    825:   58720256 |  4096, 14336,     1,     1 | F16     | blk.32.ffn_gate.weight
    826:   58720256 |  4096, 14336,     1,     1 | F16     | blk.32.ffn_up.weight
    827:       4096 |  4096,     1,     1,     1 | F32     | blk.32.post_attention_norm.weight
    828:    4194304 |  4096,  1024,     1,     1 | F16     | blk.32.attn_k.weight
    829:   16777216 |  4096,  4096,     1,     1 | F16     | blk.32.attn_output.weight
    830:   16777216 |  4096,  4096,     1,     1 | F16     | blk.32.attn_q.weight
    831:    4194304 |  4096,  1024,     1,     1 | F16     | blk.32.attn_v.weight
    832:        128 |   128,     1,     1,     1 | F32     | blk.33.cross_attn_k_norm.weight
    833:    4194304 |  4096,  1024,     1,     1 | F16     | blk.33.cross_attn_k_proj.weight
    834:   16777216 |  4096,  4096,     1,     1 | F16     | blk.33.cross_attn_output_proj.weight
    835:        128 |   128,     1,     1,     1 | F32     | blk.33.cross_attn_q_norm.weight
    836:   16777216 |  4096,  4096,     1,     1 | F16     | blk.33.cross_attn_q_proj.weight
    837:    4194304 |  4096,  1024,     1,     1 | F16     | blk.33.cross_attn_v_proj.weight
    838:          1 |     1,     1,     1,     1 | F32     | blk.33.cross_attn_attn_gate
    839:          1 |     1,     1,     1,     1 | F32     | blk.33.cross_attn_mpl_gate
    840:       4096 |  4096,     1,     1,     1 | F32     | blk.33.attn_norm.weight
    841:   58720256 | 14336,  4096,     1,     1 | F16     | blk.33.ffn_down.weight
    842:   58720256 |  4096, 14336,     1,     1 | F16     | blk.33.ffn_gate.weight
    843:   58720256 |  4096, 14336,     1,     1 | F16     | blk.33.ffn_up.weight
    844:       4096 |  4096,     1,     1,     1 | F32     | blk.33.post_attention_norm.weight
    845:       4096 |  4096,     1,     1,     1 | F32     | blk.34.attn_norm.weight
    846:   58720256 | 14336,  4096,     1,     1 | F16     | blk.34.ffn_down.weight
    847:   58720256 |  4096, 14336,     1,     1 | F16     | blk.34.ffn_gate.weight
    848:   58720256 |  4096, 14336,     1,     1 | F16     | blk.34.ffn_up.weight
    849:       4096 |  4096,     1,     1,     1 | F32     | blk.34.post_attention_norm.weight
    850:    4194304 |  4096,  1024,     1,     1 | F16     | blk.34.attn_k.weight
    851:   16777216 |  4096,  4096,     1,     1 | F16     | blk.34.attn_output.weight
    852:   16777216 |  4096,  4096,     1,     1 | F16     | blk.34.attn_q.weight
    853:    4194304 |  4096,  1024,     1,     1 | F16     | blk.34.attn_v.weight
    854:       4096 |  4096,     1,     1,     1 | F32     | blk.35.attn_norm.weight
    855:   58720256 | 14336,  4096,     1,     1 | F16     | blk.35.ffn_down.weight
    856:   58720256 |  4096, 14336,     1,     1 | F16     | blk.35.ffn_gate.weight
    857:   58720256 |  4096, 14336,     1,     1 | F16     | blk.35.ffn_up.weight
    858:       4096 |  4096,     1,     1,     1 | F32     | blk.35.post_attention_norm.weight
    859:    4194304 |  4096,  1024,     1,     1 | F16     | blk.35.attn_k.weight
    860:   16777216 |  4096,  4096,     1,     1 | F16     | blk.35.attn_output.weight
    861:   16777216 |  4096,  4096,     1,     1 | F16     | blk.35.attn_q.weight
    862:    4194304 |  4096,  1024,     1,     1 | F16     | blk.35.attn_v.weight
    863:       4096 |  4096,     1,     1,     1 | F32     | blk.36.attn_norm.weight
    864:   58720256 | 14336,  4096,     1,     1 | F16     | blk.36.ffn_down.weight
    865:   58720256 |  4096, 14336,     1,     1 | F16     | blk.36.ffn_gate.weight
    866:   58720256 |  4096, 14336,     1,     1 | F16     | blk.36.ffn_up.weight
    867:       4096 |  4096,     1,     1,     1 | F32     | blk.36.post_attention_norm.weight
    868:    4194304 |  4096,  1024,     1,     1 | F16     | blk.36.attn_k.weight
    869:   16777216 |  4096,  4096,     1,     1 | F16     | blk.36.attn_output.weight
    870:   16777216 |  4096,  4096,     1,     1 | F16     | blk.36.attn_q.weight
    871:    4194304 |  4096,  1024,     1,     1 | F16     | blk.36.attn_v.weight
    872:       4096 |  4096,     1,     1,     1 | F32     | blk.37.attn_norm.weight
    873:   58720256 | 14336,  4096,     1,     1 | F16     | blk.37.ffn_down.weight
    874:   58720256 |  4096, 14336,     1,     1 | F16     | blk.37.ffn_gate.weight
    875:   58720256 |  4096, 14336,     1,     1 | F16     | blk.37.ffn_up.weight
    876:       4096 |  4096,     1,     1,     1 | F32     | blk.37.post_attention_norm.weight
    877:    4194304 |  4096,  1024,     1,     1 | F16     | blk.37.attn_k.weight
    878:   16777216 |  4096,  4096,     1,     1 | F16     | blk.37.attn_output.weight
    879:   16777216 |  4096,  4096,     1,     1 | F16     | blk.37.attn_q.weight
    880:    4194304 |  4096,  1024,     1,     1 | F16     | blk.37.attn_v.weight
    881:        128 |   128,     1,     1,     1 | F32     | blk.38.cross_attn_k_norm.weight
    882:    4194304 |  4096,  1024,     1,     1 | F16     | blk.38.cross_attn_k_proj.weight
    883:   16777216 |  4096,  4096,     1,     1 | F16     | blk.38.cross_attn_output_proj.weight
    884:        128 |   128,     1,     1,     1 | F32     | blk.38.cross_attn_q_norm.weight
    885:   16777216 |  4096,  4096,     1,     1 | F16     | blk.38.cross_attn_q_proj.weight
    886:    4194304 |  4096,  1024,     1,     1 | F16     | blk.38.cross_attn_v_proj.weight
    887:          1 |     1,     1,     1,     1 | F32     | blk.38.cross_attn_attn_gate
    888:          1 |     1,     1,     1,     1 | F32     | blk.38.cross_attn_mpl_gate
    889:       4096 |  4096,     1,     1,     1 | F32     | blk.38.attn_norm.weight
    890:   58720256 | 14336,  4096,     1,     1 | F16     | blk.38.ffn_down.weight
    891:   58720256 |  4096, 14336,     1,     1 | F16     | blk.38.ffn_gate.weight
    892:   58720256 |  4096, 14336,     1,     1 | F16     | blk.38.ffn_up.weight
    893:       4096 |  4096,     1,     1,     1 | F32     | blk.38.post_attention_norm.weight
    894:    4194304 |  4096,  1024,     1,     1 | F16     | blk.39.attn_k.weight
    895:   16777216 |  4096,  4096,     1,     1 | F16     | blk.39.attn_output.weight
    896:   16777216 |  4096,  4096,     1,     1 | F16     | blk.39.attn_q.weight
    897:    4194304 |  4096,  1024,     1,     1 | F16     | blk.39.attn_v.weight
    898:  525336576 |  4096, 128256,     1,     1 | F16     | lm_head.weight
    899:       4096 |  4096,     1,     1,     1 | F32     | blk.39.attn_norm.weight
    900:   58720256 | 14336,  4096,     1,     1 | F16     | blk.39.ffn_down.weight
    901:   58720256 |  4096, 14336,     1,     1 | F16     | blk.39.ffn_gate.weight
    902:   58720256 |  4096, 14336,     1,     1 | F16     | blk.39.ffn_up.weight
    903:       4096 |  4096,     1,     1,     1 | F32     | blk.39.post_attention_norm.weight
    904:       4096 |  4096,     1,     1,     1 | F32     | lm.norm.weight
    905:       4096 |  4096,     1,     1,     1 | F32     | mm.projector.bias
    906:   31457280 |  7680,  4096,     1,     1 | F16     | mm.projector.weight
```
