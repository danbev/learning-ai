### Llama 3.2 Vision Instruct
This document contains notes about Llama 3.2 Vision Instruct and about
supporting this type of multi-modal model in llama.cpp.

The model architecture is similar to Llama 3.1 but with addition of a vision
encoder model, in addition to the text model. The architecture is named
`mllama` for multi-modal llama.

Paper: https://arxiv.org/pdf/2407.21783

* 11B Modified ViT 16x16 patches
* 90B Enhanced ViT 16x16 patches

So this model has a variant of a vision transformer (Vit). Now, one thing that
I need to keep in mind is that there is support being added to llama.cpp for a
[vision api](https://github.com/ggerganov/llama.cpp/pull/9687). This is
currently based on the Llava example which uses a "prompt based" variant, more
on this later in this document.

#### Model notes
So we first need to convert the model to GGUF format which is done by the
`convert_hf_to_gguf.py` script.

This model consists of not just one model but it has two which is also reflected
in the `config.json` file of the model. The language model is in a `text_config`
attribute, and the vision model is a `vision_config` attribute.

### Vision API in llama.cpp
The current Vision API in llama.cpp includes an example where a llava model is
used. This model contains both the language model and the vision model (like
Llava 1.6 does). So the arch type of this model is `llama`:
```console
(venv) $ ./inspect-model.sh ~/work/ai/llama.cpp/models/llava-1.5.7b-hf.gguf
INFO:gguf-dump:* Loading: /home/danbev/work/ai/llama.cpp/models/llava-1.5.7b-hf.gguf
* File is LITTLE endian, script is running on a LITTLE endian host.
* Dumping 49 key/value pair(s)
      1: UINT32     |        1 | GGUF.version = 3
      2: UINT64     |        1 | GGUF.tensor_count = 684
      3: UINT64     |        1 | GGUF.kv_count = 46
      4: STRING     |        1 | general.architecture = 'llama'
      ...
     23: STRING     |        1 | vision.type = 'clip-vit'
    ...
```
Now, this `vision_type` will be checked in `llm_load_hparams` and this will
set `model.has_vision` to true. This flag will later be used in
`llm_load_tensors` to load the vision tensors. And since the arch of the
model is `llama` this means that the switch-case for `LLM_ARCH_LLAMA` in 
`llm_load_hparams` will also be executed, and similarly for `llm_load_tensors`.
the tensors for the `llama` architecture will also be loaded.

Some of the functions in the current vision API are prefixed with `clip_` which
is due to the VIT implementation is based on OpenAI' CLIP/VIT implementation.
I overlooked this initially and considered it as something that would be very
different from the implementation for mllama, but now I think I should go
through this again and see if there are consolidation that can be done, perhaps
there are only minor differences between the two and that the difference is
the image preprocessing (which I think is might be different, at least it might
be different or different models so this would have to be taking into account).

### Prompt-based models
This type of model converts an image into an embedding that is similar to a
textual embeddings. These embeddings are then prepended or appended to text
token embeddings and input (prompt therefor the name prompt-based) to the LLM.
This does not require any change to the LLM model. 
So in the case of LLava we had in the previous section it can use the standard
`llama` model architecture, there are no changes to the LLM.

### Cross-attention based models
This type also has an encoder that converts an image into an embedding but
instead of passing these embeddings with the text token embeddings the model
in modified to include them in cross-attention layers.

### Vision model (current)
So I think this is similar to the llava example in llama.cpp where we have the
LLM model and the image encoder model. For example, with llava1.6 that model
contains both the vision encoder and the LLM model, and we have to extract
the vision encoder from the model to use it with the llava example.
For example, when we run `llama-llava-cli` we specify both the model and the
projection models:
```console
./build/bin/llama-llava-cli -m models/vicuna-7b-q5_k.gguf \
    --mmproj models/mmproj-vicuna7b-f16.gguf \
    --image ~/work/ai/learning-ai/notes/apollo11.jpg -c 4096 -ngl 15
```
But I believe that for the new Vision API in llama.cpp is will be possible to
just pass a single model to llama.cpp and not have to have two separate models.

If we inspect the tensors that are in `model.safetensors.index.json` we can see
it has both the text language model tensors, and the vision model tensors.

So I think that the Llama 3.2-vision model will work in a similar way. First one
or more images would be read and split into patches which would then be
passed to the vision encoder model. This model will pass the patches through the
vision model, going through the self-attention for the patches (all the layers
in the model). The output of this will be patch embeddings that can then be
passed into the language model.

### Vision model layer (tensors)
The vision model has 8 global layers and 32 hidden layers:
```console
    "num_global_layers": 8,
    "num_hidden_layers": 32,
```

```console
"vision_model.patch_embedding.weight"
"vision_model.class_embedding"

"vision_model.pre_tile_positional_embedding.embedding.weight" 
"vision_model.pre_tile_positional_embedding.gate"

"vision_model.layernorm_pre.weight"
"vision_model.layernorm_pre.bias"

"vision_model.gated_positional_embedding.embedding"
"vision_model.gated_positional_embedding.gate"

"vision_model.gated_positional_embedding.tile_embedding.weight"
"vision_model.post_tile_positional_embedding.embedding.weight"
"vision_model.post_tile_positional_embedding.gate"

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
"vision_model.global_transformer.layers.{bid}.input_layernorm.weight"
"vision_model.global_transformer.layers.{bid}.input_layernorm.bias"

"vision_model.global_transformer.layers.{bid}.self_attn.q_proj.weight"
"vision_model.global_transformer.layers.{bid}.self_attn.k_proj.weight"
"vision_model.global_transformer.layers.{bid}.self_attn.v_proj.weight"

"vision_model.global_transformer.layers.{bid}.self_attn.o_proj.weight"

"vision_model.global_transformer.layers.{bid}.post_attention_layernorm.weight"
"vision_model.global_transformer.layers.{bid}.post_attention_layernorm.bias"

"vision_model.global_transformer.layers.{bid}.mlp.fc1.weight"
"vision_model.global_transformer.layers.{bid}.mlp.fc1.bias"

"vision_model.global_transformer.layers.{bid}.gate_attn"
"vision_model.global_transformer.layers.{bid}.gate_ffn"

"vision_model.global_transformer.layers.{bid}.mlp.fc2.bias"
"vision_model.global_transformer.layers.{bid}.mlp.fc2.weight"

fc = fully connected.
```

And 32 (`num_hidden_layers`) hidden layers:
```console
"vision_model.transformer.layers.{bid}.input_layernorm.bias"
"vision_model.transformer.layers.{bid}.input_layernorm.weight"

"vision_model.transformer.layers.{bid}.self_attn.k_proj.weight"
"vision_model.transformer.layers.{bid}.self_attn.q_proj.weight"
"vision_model.transformer.layers.{bid}.self_attn.v_proj.weight"
"vision_model.transformer.layers.{bid}.self_attn.o_proj.weight"

"vision_model.transformer.layers.{bid}.post_attention_layernorm.bias"
"vision_model.transformer.layers.{bid}.post_attention_layernorm.weight"

"vision_model.transformer.layers.{bid}.mlp.fc1.bias"
"vision_model.transformer.layers.{bid}.mlp.fc1.weight"

"vision_model.transformer.layers.{bid}.mlp.fc2.bias"
"vision_model.transformer.layers.{bid}.mlp.fc2.weight"
}
```

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
All blocks have the above tensors, but there are also blocks that have
additional tensors. For example block 13 has:
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
These tensors exist for blocks 3, 8, 13, 18, 23, 28, 33, 38 as well. Which also
matches the following attribute in the config.json file:
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

#### supported_aspect_ratios
This is defined for the vision model and looks like this:
```console
  "vision_config": {
    ...
    "supported_aspect_ratios": [
      [ 1, 1 ],
      [ 1, 2 ],
      [ 1, 3 ],
      [ 1, 4 ],
      [ 2, 1 ],
      [ 2, 2 ],
      [ 3, 1 ],
      [ 4, 1 ]
    ],
```
These are ratios that the model is designed to handle efficiently. Each sublist
specified a width to height ration:
* 1, 1  - square
* 1, 2  - portrait
* 2, 1  - landscape

Though the name is very different in CLIP the hyperparameter called
`image_grid_pinpoints` seems to serve the same purpose if we look at the code
in llama.cpp's clip implementation ([notes](https://github.com/danbev/learning-ai/blob/main/notes/vision/llava.md)).
This is stored like this for CLIP models:
```console
     20: [INT32]    |       10 | clip.vision.image_grid_pinpoints
```
So this is just a list of integers and they are in pairs which is how llama.cpp
/clip.cpp handles them. So I think we can store `supported_aspect_ratios` the
same way.

#### image_token_index
This is a property in config.json and looks like this:
```console
{
  "architectures": [
    "MllamaForConditionalGeneration"
  ],
  "image_token_index": 128256,
  "model_type": "mllama",
```
This is special token used to identify image tokens when we combine text and
image. For example:
```
text = "This image <|image|> shows a cat"
```
And this would get tokenized into:
```
["This", "image", 128256, "shows", "a", "cat"]
```
Where the other words would also be integer tokens which are indices into
the vocabulary.

This is what this token looks like in the `tokenizer_config.json`:
```
    "128256": {
      "content": "<|image|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    }
```

Now, the vocabulary is stored in the `text_config` attribute:
```
"vocab_size": 128256
```
Notice that this is the same value as the `image_token_index` value.
This value should be part of the model but I missed this originally. 

The issue is that when we pass in the vocabulary size to create the tensor
for `language_model.model.embed_tokens.weight` this is not match the value
of the actual tensor in the model:
```c++
model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
```
In our case we know that the `vocab_size` is 128256, so this is what will be
passed.

If we inspect the model we can see that the actual value of the tensor is infact
128264:
```console
  2:  525369344 |  4096, 128264,     1,     1 | Q6_K    | token_embd.weight
```
So this is something that happens and I think the cause is that the actual
tensor has this shape:
```console
(venv) $ python ~/work/ai/learning-ai/fundamentals/python/src/list-safetensors.py model-00001-of-00005.safetensors

Tensor shapes in the file:
--------------------------------------------------
language_model.model.embed_tokens.weight: torch.Size([128264, 4096])
...
```
Notice that the shape of `language_model.model.embed_tokens.weight` is
`torch.Size([128264, 4096])` which is 8 more than the `vocab_size` value. So
we need to make sure that this value is correct when loading the tensor in
llama.cpp `llm_load_tensor`. We don't have any control over the values of the
tensor in the model, but we can acount for this when loading this tensor.

### max position embedding
The maxium position embedding is calculated using the image size and the patch
size as follows:
```
max_pos_embd = (image_size // patch_size)^2 + 1

The +1 is for the CLS token.
And we have both width and height so we have to square.
```
In the python conversion script this is done using:
```console
        max_pos_embd = (self.vision_config["image_size"] // self.vision_config["patch_size"])**2 + 1
        self.gguf_writer.add_vision_clip_max_position_embeddings(max_pos_embd)
```
The actual values are the following:
```console
image_size 560
patch_size 14
max_pos_embd 1601
```

### Pre-processor config
There is information in `preprocessor_config.json` which is needed for the
pre-processing of images
```console
{
  "do_convert_rgb": true,
  "do_normalize": true,
  "do_pad": true,
  "do_rescale": true,
  "do_resize": true,
  "image_mean": [
    0.48145466,
    0.4578275,
    0.40821073
  ],
  "image_processor_type": "MllamaImageProcessor",
  "image_std": [
    0.26862954,
    0.26130258,
    0.27577711
  ],
  "max_image_tiles": 4,
  "resample": 2,
  "rescale_factor": 0.00392156862745098,
  "size": {
    "height": 560,
    "width": 560
  }
}
```
This information will be needed to preprocess image which I think will be
part of the new Vision API in llama.cpp.


### Tensors
So as we discussed earlier the model has two parts, the language model and the
vision model. 

#### Language model tensors
The tensors that are for the LLM are named with a prefix `language_model` prefix,
for example 
```console
language_model.lm_head.weight
language_model.model.embed_tokens.weight
language_model.model.norm.weight
```

And we have 32 normal layers which each have 9 tensors:
```console
language_model.model.layers.0.input_layernorm.weight
language_model.model.layers.0.mlp.down_proj.weight
language_model.model.layers.0.mlp.gate_proj.weight
language_model.model.layers.0.mlp.up_proj.weight
language_model.model.layers.0.post_attention_layernorm.weight
language_model.model.layers.0.self_attn.k_proj.weight
language_model.model.layers.0.self_attn.o_proj.weight
language_model.model.layers.0.self_attn.q_proj.weight
language_model.model.layers.0.self_attn.v_proj.weight
```

And 8 layers that are involved in the cross-attention and each have 13 tensors:
```console
language_model.model.layers.3.cross_attn.k_norm.weight
language_model.model.layers.3.cross_attn.k_proj.weight
language_model.model.layers.3.cross_attn.o_proj.weight
language_model.model.layers.3.cross_attn.q_norm.weight    
language_model.model.layers.3.cross_attn.q_proj.weight
language_model.model.layers.3.cross_attn.v_proj.weight
language_model.model.layers.3.cross_attn_attn_gate
language_model.model.layers.3.cross_attn_mlp_gate
language_model.model.layers.3.input_layernorm.weight
language_model.model.layers.3.mlp.down_proj.weight
language_model.model.layers.3.mlp.gate_proj.weight
language_model.model.layers.3.mlp.up_proj.weight
language_model.model.layers.3.post_attention_layernorm.weight
```

```
multi_modal_projector.bias
multi_modal_projector.weight
````

#### Vision model tensors
The tensors that are for the LLM are named with a prefix `vision_model` prefix,

The following are the tensors that are not part of the layers:
```console
vision_model.class_embedding
vision_model.patch_embedding.weight
vision_model.gated_positional_embedding.embedding
vision_model.gated_positional_embedding.gate
vision_model.gated_positional_embedding.tile_embedding.weight
vision_model.layernorm_post.bias
vision_model.layernorm_post.weight
vision_model.layernorm_pre.bias
vision_model.layernorm_pre.weight
vision_model.post_tile_positional_embedding.embedding.weight
vision_model.post_tile_positional_embedding.gate
vision_model.pre_tile_positional_embedding.embedding.weight
vision_model.pre_tile_positional_embedding.gate
vision_model.transformer.layers.0.input_layernorm.bias
```

##### Graph
```c++
inp_raw =
struct ggml_tensor *inp_raw = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32,
                                                 image_size_width,
                                                 image_size_height,
                                                 num_channels,
                                                 num_tiles);
```
So we have the raw image (w*h) with 3 channels each and 4 tiles. A tile is a
special segment in the image.
```console
channels  = 3
num_tiles = 4
   
t_0
    c_0
         [0        ...      width]
         ...
         ...
         ...
         [height   ...      width]

    ...

    c_2
         [0        ...      width]
         ...
         ...
         ...
         [height   ...      width]

...

t_3
    c_0
         [0        ...      width]
         ...
         ...
         ...
         [height   ...      width]

    ...

    c_2
         [0        ...      width]
         ...
         ...
         ...
         [height   ...      width]
```
So this is bascially representing the image 4 times.

Next a 2d convolution is applied over the image which is using the
`patch_embedding` tensor as the kernel:
```c++
struct ggml_tensor *inp = ggml_conv_2d(ctx0, model.patch_embeddings, inp_raw, patch_size, patch_size, 0, 0, 1, 1);
```

```c++
 struct ggml_tensor *aspect_ratios = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, imgs->size);
    ggml_set_name(aspect_ratios, "aspect_ratios");
    ggml_set_input(aspect_ratios);

    if (model.pre_tile_position_embeddings != nullptr) {
        struct ggml_tensor *pre_tile_position_embeddings = ggml_get_rows(ctx0, model.pre_tile_position_embeddings, aspect_ratios);
        ggml_set_name(pre_tile_position_embeddings, "pre_tile_position_embeddings");

        pre_tile_position_embeddings = ggml_reshape_3d(ctx0, pre_tile_position_embeddings, hidden_size, 1, num_tiles);
        if (model.pre_tile_position_embeddings_gate != nullptr) {
            pre_tile_position_embeddings = ggml_mul_inplace(ctx0, pre_tile_position_embeddings, model.pre_tile_position_embeddings_gate);
        }

        inp = ggml_add(ctx0, inp, pre_tile_position_embeddings);
    }
```

### Ollama 3.2 Vision model
I've not been able to get the model to work locally and the only output it
results in when passing an image is a bunch of question marks. I've downloaded
the model that ollama uses (which is publicly available which was a little
surprising) and I've been able to inspect it. In ollama they have split the
model into a the llm part and the projector (vision) part much like what is
done for the llava 1.5 models in llama.cpp.
The models are availble in the blobs directory so we can inspect it.

The projector contains 512 tensors:
```console
 6       2: UINT64     |        1 | GGUF.tensor_count = 512
```
And the llm model contains 396 tensors:
```console
  5       2: UINT64     |        1 | GGUF.tensor_count = 396
```
That is a total of 908 tensors.

In the model I converted I only have 907:
```console
       2: UINT64     |        1 | GGUF.tensor_count = 907
```
So this was good to find out and I need to investigate which tensor is missing.

One thing I can also do is use the language model from ollama and run it without
the updates made for vision and see if that works (so just a pure chat and
not image data).

I found that the missing tensor is:
```console
      5:          1 |     1,     1,     1,     1 | F32     | v.tile_position_embd.gate
```

### Image related hyperparameters
These are the parameters related to images:
```console
  "vision_config": {
    "attention_heads": 16,
    "hidden_act": "gelu",
    "hidden_size": 1280,
    "image_size": 560,
    "intermediate_size": 5120,
    "max_length": 20,
    "max_num_tiles": 4,
    "min_length": 0,
    "model_type": "mllama_vision_model",
    "num_channels": 3,
    "num_global_layers": 8,
    "num_hidden_layers": 32,
    "patch_size": 14,
    "supported_aspect_ratios": [
      [ 1, 1 ], [ 1, 2 ], [ 1, 3 ], [ 1, 4 ], [ 2, 1 ], [ 2, 2 ], [ 3, 1 ], [ 4, 1 ]
    ],
    "vision_output_dim": 7680
  }
}
```
And we also have the following pre-processor config:
```console
{
  "do_convert_rgb": true,
  "do_normalize": true,
  "do_pad": true,
  "do_rescale": true,
  "do_resize": true,
  "image_mean": [
    0.48145466,
    0.4578275,
    0.40821073
  ],
  "image_processor_type": "MllamaImageProcessor",
  "image_std": [
    0.26862954,
    0.26130258,
    0.27577711
  ],
  "max_image_tiles": 4,
  "resample": 2,
  "rescale_factor": 0.00392156862745098,
  "size": {
    "height": 560,
    "width": 560
  }
}
```
So we can see that the width and height of the image is 560 x 560.
```
image_size / patch_size = 560 / 14 = 40
And we have both width and height so we have to square.
40 x 40 = 1600

And optionally we have a CLS token:
1600 + 1 = 1601

If we have 4 tiles then we get:
1601 x 4 = 6404

We can write this as:
(560 / 14)^2 + 1 x 4 = 6404
```


### Prompting
So the prompt for the vision model has a special token for images which is
`<|image|>` and this has token id 128256. But when I run the example I have
and tokenize the following prompt:
```console
(gdb) p params.prompt
$8 = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n\n<|image|>Describe this image in two sentences<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```
I get the following tokens:
```console
(gdb) p tokens_list
$1 = std::vector of length 20, capacity 160 = {128000, 128006, 882, 128007, 1432, 27, 91, 1843, 91, 29, 75885, 420, 
  2217, 304, 1403, 23719, 128009, 128006, 78191, 128007}
(gdb) p model.vocab.id_to_token[304]
$2 = {text = "Ġin", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p model.vocab.id_to_token[128007]
$3 = {text = "<|end_header_id|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) p model.vocab.id_to_token[1432]
$4 = {text = "ĊĊĊ", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p model.vocab.id_to_token[27]
$5 = {text = "<", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p model.vocab.id_to_token[91]
$6 = {text = "|", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p model.vocab.id_to_token[1843]
$7 = {text = "image", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
```
Notice that the image token is not recognized and it should be a single token
and on multiple tokens.
```console
(gdb) p model.vocab.id_to_token[12856]
$9 = {text = "_window", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
```
So there must be something wrong with how I've configured the tokenizer
when converting the model I think.

```console
  25: STRING     |        1 | tokenizer.ggml.model = 'gpt2'
     26: STRING     |        1 | tokenizer.ggml.pre = 'llama-bpe'
     27: [STRING]   |   128257 | tokenizer.ggml.tokens
     28: [INT32]    |   128257 | tokenizer.ggml.token_type
     29: [STRING]   |   280147 | tokenizer.ggml.merges
     30: UINT32     |        1 | tokenizer.ggml.bos_token_id = 128000
     31: UINT32     |        1 | tokenizer.ggml.eos_token_id = 128009
     32: UINT32     |        1 | tokenizer.ggml.padding_token_id = 128004
     33: STRING     |        1 | tokenizer.chat_template = '{{- bos_token }}\n{%- if custom_tools is defined %}\n
     ```
```
This issue seems to have been related to the size of the vocabulary in the
model which is specified as 128256 but the actual size of the vocabulary is:
```console
(venv) $ python src/inspect-token-config.py
Loading tokenizer from: /home/danbev/work/ai/llama-models/Llama-3.2-11B-Vision-Instruct
Vocabulary size: 128257
Max token ID: 128256
Last token (by max ID): <|image|>, ID: 128256

Tokenized text: {'input_ids': [128000, 9906, 1917, 0, 1115, 374, 264, 1296, 13], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}

Decoded text: <|begin_of_text|>Hello world! This is a test.

Sample vocabulary items:
ID 128247: <|reserved_special_token_238|>
ID 128248: <|reserved_special_token_239|>
ID 128249: <|reserved_special_token_240|>
ID 128250: <|reserved_special_token_241|>
ID 128251: <|reserved_special_token_242|>
ID 128252: <|reserved_special_token_243|>
ID 128253: <|reserved_special_token_244|>
ID 128254: <|reserved_special_token_245|>
ID 128255: <|reserved_special_token_246|>
ID 128256: <|image|>
```
So the actual size of the vocabulary is 128257 and not 128256. I've corrected
this (in a poor way but I will fix this later) and now the tokens look like
this:
```console
(gdb) p tokens_list
$1 = std::vector of length 16, capacity 159 = {128000, 128006, 882, 128007, 271, 128256, 75885, 420, 2217, 304,
  1403, 23719, 128009, 128006, 78191, 128007}
```
Compared to before:
```console
$1 = std::vector of length 20, capacity 160 = {128000, 128006, 882, 128007, 1432, 27, 91, 1843, 91, 29, 75885, 420, 
  2217, 304, 1403, 23719, 128009, 128006, 78191, 128007}
```
And we can check the image token:
```console
(gdb) p ctx.model.vocab.id_to_token[128256]
$2 = {text = "<|image|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
```
This seems to be a known issue and is mentioned [here](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/mllama.md#usage-tips).


### Instruction prompting
Running locally I've been able to verify what the prompt should look like:
```console
(venv) $ python src/llama-3.2-instruct.py
The model weights are not tied. Please use the `tie_weights` method before using the `infer_auto_device` function.
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████| 5/5 [00:11<00:00,  2.25s/it]
Some parameters are on the meta device because they were offloaded to the cpu.
prompt: ['<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n<|image|>What does the image show?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n']
<|begin_of_text|><|begin_of_text|><|start_header_id|>user<|end_header_id|>

<|image|>What does the image show?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

The image shows the Eiffel Tower in Paris, France.<|eot_id|>
```

```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n<|image|>What does the image show?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
```

With the following prompt I can get a somewhat reasonable response from the
model I converted (the unquantized model)::
```console
prompt = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat is LoRA?<|eot_id|><|start_header_id|>assistant<|end_header_id|>";
```

### Image types
The new vision api has the following types:
```c++
    // represent an RGB image
    // size of data must be equal to 3*nx*ny
    typedef struct llama_img {
        uint32_t nx;
        uint32_t ny;
        unsigned char * data;
    } llama_img;

    // Input data for llama_vision_decode
    typedef struct llama_batch_img {
        int32_t      n_imgs;
        llama_img ** imgs;
        llama_pos *  pos;
    } llama_batch_img;
```
So if we have one image this would be something like the following:
```console
llama_img img = {
    .nx = 560,
    .ny = 560,
    .data = some_rgb_data_pointer
};

llama_img* img_array[1] = { &img };
llama_pos positions[1] = { 0 };

// Create a batch with one image
llama_batch_img img_batch = {
    .n_imgs = 1,
    .imgs = img_array,
    .pos = positions
};

To access the first image in the batch:
img_batch.imgs[0]->nx
```

There are significant differences between the llava based one and mllama. The
llava one is a "prompt" based model where the image is first encoded into patch
embedding and then projected inte same space as the text embeddings, and these
are then passed to the llava model. In mllama cross-attention is used. So we
still have to encode the image into patch embeddings but instead of being
passed as normal language tokens these embeddings are passed as embd tokens in
the batch to be the model to be decoded where the special image token
('<|image|>') is used in the language tokens. This is then passed to the to the
model and the cross-attention layers are used to combine the image embeddings
with the text embeddings. In the future it might be possible to pass a single
batch containing both language tokens and image patch embedding to be processed.

So the embeddings for the image encoding has the following shape (the final
tenosor:
```console
(gdb) p embeddings->ne
$12 = {1280, 1601, 4, 1}
(gdb) p 1280 * 1601 * 4
$13 = 8197120

z_0
   0    [0                        1279]
        ...
        ...
        ...
        ...
   1600 [0                        1279]

z_1
   0    [0                        1279]
        ...
        ...
        ...
        ...
   1600 [0                        1279]

z_2
   0    [0                        1279]
        ...
        ...
        ...
        ...
   1600 [0                        1279]

z_3
   0    [0                        1279]
        ...
        ...
        ...
        ...
   1600 [0                        1279]
```
And `inp_raw` has the following shape:
```console
(gdb) p inp_raw->ne
$11 = {560, 560, 3, 4}
(gdb) p 560*560*3*4
$6 = 3763200
```

```console
(gdb) p ggml_nbytes(inp_raw)
$10 = 15052800
```

The size of the actual image is:
```console
(gdb) p nx
$7 = 1280
(gdb) p ny
$8 = 853
(gdb) p n
$9 = 1091840
```

When we build the graph we have the following code:
```c++
    const int num_padding_patches = 8 - (embeddings->ne[1] % 8) % 8;
                                                                                
    embeddings = ggml_pad(ctx0, embeddings, 0, num_padding_patches, 0, 0);          
    embeddings = ggml_view_3d(ctx0, embeddings, embeddings->ne[0], embeddings->ne[1] * embeddings->ne[2], batch_size, embeddings->nb[1], embeddings->nb[2] * embeddings->ne[3], 0)
```
```console
(gdb) p num_padding_patches
$21 = 7
(gdb) p embeddings->ne
$22 = {1280, 1601, 4, 1}
```
Lets try to visualize this:
```console
z_0   
        0 [0                        1279]
        ...
        ...
        ...
      1600[0                        1279]

z_1   
        0 [0                        1279]
        ...
        ...
        ...
      1600[0                        1279]

z_2   
        0 [0                        1279]
        ...
        ...
        ...
      1600[0                        1279]

z_3   
        0 [0                        1279]
        ...
        ...
        ...
      1600[0                        1279]
```
So this is shape of the embedding tensor before the padding. And we are only
going to pad the second dimension by 7.
```
z_0   
        0 [0                        1279]
        ...
        ...
        ...
      1607[0                        1279]

z_1   
        0 [0                        1279]
        ...
        ...
        ...
      1607[0                        1279]

z_2   
        0 [0                        1279]
        ...
        ...
        ...
      1607[0                        1279]

z_3   
        0 [0                        1279]
        ...
        ...
        ...
      1607[0                        1279]
```
And this is the shape after the padding operation:
```console
(gdb) p embeddings->ne
$23 = {1280, 1608, 4, 1}
```
Now lets look at the reshaping operation after the padding:
```console
embeddings = ggml_view_3d(ctx0,
    embeddings,
    embeddings->ne[0],
    embeddings->ne[1] * embeddings->ne[2],
    batch_size,
    embeddings->nb[1],
    embeddings->nb[2] * embeddings->ne[3],
    0);
```
So the first dimension is kept the same. The second dimension (1608) is
multiplied by the third dimension (4) which is 6432. The third dimension
is set to the batch_size which is currently 1. The strides are set and
the offset is zero resulting in:
```console
(gdb) p embeddings->ne
$28 = {1280, 6432, 1, 1}
```

Output from ollama:
```console
danbev] num_positions: 1601
[danbev] image data length: 15052800
[danbev] image width: 560
[danbev] image height: 560
[danbev] inp_raw[0]: 560
[danbev] inp_raw[1]: 560
[danbev] inp_raw[2]: 3
[danbev] inp_raw[3]: 4
[danbev] data size:  15052800
[danbev] ggml_nbytes: 15052800
[danbev] copy embeddings size: 104923136


[danbev] mllama_image_load_data n: 15052800
[danbev] mllama_image_load_data aspect_ratio_id: 6
[danbev] mllama_n_positions: 1601
[danbev] mllama_n_tiles: 4
[danbev] numTokens: 6404
[danbev] numEmbed: 4096
[danbev] Total size (numTokens * numEmbed): 26230784
```

```console
time=2024-11-23T10:07:22.952+01:00 level=INFO source=runner.go:357 msg="[danbev] sequence loop...." seqIdx=0
time=2024-11-23T10:07:22.952+01:00 level=INFO source=runner.go:414 msg="[danbev] adding embeddings (input.embed) to batch as embed"
time=2024-11-23T10:07:22.952+01:00 level=INFO source=llama.go:375 msg="[danbev] Add token" isEmbedding=false batchSize=512 maxSeq=1 embedSize=0 token=128006 pos=0 seqIds=[0]
time=2024-11-23T10:07:22.952+01:00 level=INFO source=runner.go:414 msg="[danbev] adding embeddings (input.embed) to batch as embed"
time=2024-11-23T10:07:22.952+01:00 level=INFO source=llama.go:375 msg="[danbev] Add token" isEmbedding=false batchSize=512 maxSeq=1 embedSize=0 token=882 pos=1 seqIds=[0]
time=2024-11-23T10:07:22.952+01:00 level=INFO source=runner.go:414 msg="[danbev] adding embeddings (input.embed) to batch as embed"
time=2024-11-23T10:07:22.952+01:00 level=INFO source=llama.go:375 msg="[danbev] Add token" isEmbedding=false batchSize=512 maxSeq=1 embedSize=0 token=128007 pos=2 seqIds=[0]
time=2024-11-23T10:07:22.952+01:00 level=INFO source=runner.go:414 msg="[danbev] adding embeddings (input.embed) to batch as embed"
time=2024-11-23T10:07:22.952+01:00 level=INFO source=llama.go:375 msg="[danbev] Add token" isEmbedding=false batchSize=512 maxSeq=1 embedSize=0 token=271 pos=3 seqIds=[0]
time=2024-11-23T10:07:22.952+01:00 level=INFO source=runner.go:432 msg="[danbev] before decode" batch="&{c:{n_tokens:4 token:0x78ceec1a1f00 embd:<nil> n_embd:0 pos:0x78ceec1a2710 n_seq_id:0x78ceec1a2f20 seq_id:0x78ceec28bab0 logits:0x78cecc707bc0 all_pos_0:0 all_pos_1:0 all_seq_id:0 _:[0 0 0 0]} batchSize:512 maxSeq:1 embedSize:0}"
[danbev] llama.cpp set_inputs NO batch.embd

time=2024-11-23T10:07:22.968+01:00 level=INFO source=runner.go:438 msg="[danbev] after decode"

time=2024-11-23T10:07:22.968+01:00 level=INFO source=runner.go:357 msg="[danbev] sequence loop...." seqIdx=0
time=2024-11-23T10:07:22.968+01:00 level=INFO source=runner.go:414 msg="[danbev] adding embeddings (input.embed) to batch as embed"
time=2024-11-23T10:07:22.968+01:00 level=INFO source=llama.go:375 msg="[danbev] Add token" isEmbedding=true batchSize=1 maxSeq=1 embedSize=26230784 token=0 pos=4 seqIds=[0]
time=2024-11-23T10:07:23.013+01:00 level=INFO source=runner.go:432 msg="[danbev] before decode" batch="&{c:{n_tokens:1 token:<nil> embd:0x78ca97a00010 n_embd:26230784 pos:0x78ceec018d20 n_seq_id:0x78ceec018d40 seq_id:0x78ceec018d60 logits:0x78ceec018da0 all_pos_0:0 all_pos_1:0 all_seq_id:0 _:[0 0 0 0]} batchSize:1 maxSeq:1 embedSize:26230784}"
[danbev] llama.cpp set_inputs batch.embd
[danbev] llama.cpp --------- cross_attn_state from batch.embd -------------

time=2024-11-23T10:07:23.167+01:00 level=INFO source=runner.go:438 msg="[danbev] after decode"

time=2024-11-23T10:07:23.167+01:00 level=INFO source=runner.go:357 msg="[danbev] sequence loop...." seqIdx=0
time=2024-11-23T10:07:23.167+01:00 level=INFO source=runner.go:414 msg="[danbev] adding embeddings (input.embed) to batch as embed"
time=2024-11-23T10:07:23.167+01:00 level=INFO source=llama.go:375 msg="[danbev] Add token" isEmbedding=false batchSize=512 maxSeq=1 embedSize=0 token=128256 pos=5 seqIds=[0]
time=2024-11-23T10:07:23.167+01:00 level=INFO source=runner.go:414 msg="[danbev] adding embeddings (input.embed) to batch as embed"
time=2024-11-23T10:07:23.167+01:00 level=INFO source=llama.go:375 msg="[danbev] Add token" isEmbedding=false batchSize=512 maxSeq=1 embedSize=0 token=3923 pos=6 seqIds=[0]
time=2024-11-23T10:07:23.167+01:00 level=INFO source=runner.go:414 msg="[danbev] adding embeddings (input.embed) to batch as embed"
time=2024-11-23T10:07:23.167+01:00 level=INFO source=llama.go:375 msg="[danbev] Add token" isEmbedding=false batchSize=512 maxSeq=1 embedSize=0 token=374 pos=7 seqIds=[0]
time=2024-11-23T10:07:23.167+01:00 level=INFO source=runner.go:414 msg="[danbev] adding embeddings (input.embed) to batch as embed"
time=2024-11-23T10:07:23.167+01:00 level=INFO source=llama.go:375 msg="[danbev] Add token" isEmbedding=false batchSize=512 maxSeq=1 embedSize=0 token=304 pos=8 seqIds=[0]
time=2024-11-23T10:07:23.167+01:00 level=INFO source=runner.go:414 msg="[danbev] adding embeddings (input.embed) to batch as embed"
time=2024-11-23T10:07:23.167+01:00 level=INFO source=llama.go:375 msg="[danbev] Add token" isEmbedding=false batchSize=512 maxSeq=1 embedSize=0 token=420 pos=9 seqIds=[0]
time=2024-11-23T10:07:23.167+01:00 level=INFO source=runner.go:414 msg="[danbev] adding embeddings (input.embed) to batch as embed"
time=2024-11-23T10:07:23.167+01:00 level=INFO source=llama.go:375 msg="[danbev] Add token" isEmbedding=false batchSize=512 maxSeq=1 embedSize=0 token=2217 pos=10 seqIds=[0]
time=2024-11-23T10:07:23.167+01:00 level=INFO source=runner.go:414 msg="[danbev] adding embeddings (input.embed) to batch as embed"
time=2024-11-23T10:07:23.167+01:00 level=INFO source=llama.go:375 msg="[danbev] Add token" isEmbedding=false batchSize=512 maxSeq=1 embedSize=0 token=30 pos=11 seqIds=[0]
time=2024-11-23T10:07:23.167+01:00 level=INFO source=runner.go:414 msg="[danbev] adding embeddings (input.embed) to batch as embed"
time=2024-11-23T10:07:23.167+01:00 level=INFO source=llama.go:375 msg="[danbev] Add token" isEmbedding=false batchSize=512 maxSeq=1 embedSize=0 token=128009 pos=12 seqIds=[0]
time=2024-11-23T10:07:23.167+01:00 level=INFO source=runner.go:414 msg="[danbev] adding embeddings (input.embed) to batch as embed"
time=2024-11-23T10:07:23.167+01:00 level=INFO source=llama.go:375 msg="[danbev] Add token" isEmbedding=false batchSize=512 maxSeq=1 embedSize=0 token=128006 pos=13 seqIds=[0]
time=2024-11-23T10:07:23.167+01:00 level=INFO source=runner.go:414 msg="[danbev] adding embeddings (input.embed) to batch as embed"
time=2024-11-23T10:07:23.167+01:00 level=INFO source=llama.go:375 msg="[danbev] Add token" isEmbedding=false batchSize=512 maxSeq=1 embedSize=0 token=78191 pos=14 seqIds=[0]
time=2024-11-23T10:07:23.167+01:00 level=INFO source=runner.go:414 msg="[danbev] adding embeddings (input.embed) to batch as embed"
time=2024-11-23T10:07:23.167+01:00 level=INFO source=llama.go:375 msg="[danbev] Add token" isEmbedding=false batchSize=512 maxSeq=1 embedSize=0 token=128007 pos=15 seqIds=[0]
time=2024-11-23T10:07:23.167+01:00 level=INFO source=runner.go:414 msg="[danbev] adding embeddings (input.embed) to batch as embed"
time=2024-11-23T10:07:23.167+01:00 level=INFO source=llama.go:375 msg="[danbev] Add token" isEmbedding=false batchSize=512 maxSeq=1 embedSize=0 token=271 pos=16 seqIds=[0]
time=2024-11-23T10:07:23.167+01:00 level=INFO source=runner.go:432 msg="[danbev] before decode" batch="&{c:{n_tokens:12 token:0x78ceec1a1f00 embd:<nil> n_embd:0 pos:0x78ceec1a2710 n_seq_id:0x78ceec1a2f20 seq_id:0x78ceec28bab0 logits:0x78cecc707bc0 all_pos_0:0 all_pos_1:0 all_seq_id:0 _:[0 0 0 0]} batchSize:512 maxSeq:1 embedSize:0}"
```

```
(gdb) p ctx.model.vocab.id_to_token[128006]
$1 = {text = "<|start_header_id|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) p ctx.model.vocab.id_to_token[882]
$2 = {text = "user", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p ctx.model.vocab.id_to_token[128007]
$3 = {text = "<|end_header_id|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) p ctx.model.vocab.id_to_token[271]
$5 = {text = "ĊĊ", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}  
```
prompt="<|start_header_id|>user<|end_header_id|>\n\n[img-0]<|image|>What is in this image?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


### Inspect ollama Llama-3.2-11B-Vision-Instruct model
Inspecting the model:
```console
(venv) $ cat ~/.ollama/models/manifests/registry.ollama.ai/x/llama3.2-vision/latest | jq '.layers[0].digest'
"sha256:652e85aa1e14c9087a4ccc3ab516fb794cbcf152f8b4b8d3c0b828da4ada62d9"

(venv) $ ./inspect-model.sh /home/danbev/.ollama/models/blobs/sha256-652e85aa1e14c9087a4ccc3ab516fb794cbcf152f8b4b8d3c0b828da4ada62d9
INFO:gguf-dump:* Loading: /home/danbev/.ollama/models/blobs/sha256-652e85aa1e14c9087a4ccc3ab516fb794cbcf152f8b4b8d3c0b828da4ada62d9
```
Inspecting the projector:
```console
(venv) $ ./inspect-model.sh /home/danbev/.ollama/models/blobs/sha256-622429e8d31810962dd984bc98559e706db2fb1d40e99cb073beb7148d909d73
```

### Model issues

#### pre_tile_position_embeddings weights
This tensor is defined as
```
46080 |  5120,     9,     1,     1 | F16     | v.enc.pre_tile_position_embd.weight
```
And in ollama's model is is:
```
 46080 |  5120,     9,     1,     1 | F32     | v.pre_tile_position_embd.weight
```
Notice that the type if F32 and not F16. And this is also quantized if we
quantize our model.

ours:
```console
 503:   31457280 |    7680,   4096,     1,     1 | Q4_1    | v.enc.mmproj.weight           <--- wrong data type   
    504:     752640 |    14,    14,     3,  1280 | F16     | v.enc.patch_embd.weight
    505:    2049280 |  1280,  1601,     1,     1 | F32     | v.enc.position_embd
    506:          1 |     1,     1,     1,     1 | F32     | v.enc.position_gate
    507:       1280 |  1280,     1,     1,     1 | F32     | v.enc.post_ln.bias
    508:       1280 |  1280,     1,     1,     1 | F32     | v.enc.post_ln.weight
    509:      46080 |  5120,     9,     1,     1 | F32     | v.enc.post_tile_position_embd.weight
    510:          1 |     1,     1,     1,     1 | F32     | v.enc.post_tile_position_gate
    511:       1280 |  1280,     1,     1,     1 | F32     | v.enc.pre_ln.bias
    512:       1280 |  1280,     1,     1,     1 | F32     | v.enc.pre_ln.weight
    513:      46080 |  5120,     9,     1,     1 | F32     | v.enc.pre_tile_position_embd.weight
    514:          1 |     1,     1,     1,     1 | F32     | v.enc.pre_tile_position_gate
    515:   73774080 | 8197120,   9,     1,     1 | F16     | v.enc.tile_position_embd.weight <--- wrong data type

```
ollama:
```console
      1:   31457280 |  7680,  4096,     1,     1 | F16     | mm.0.weight
      2:       4096 |  4096,     1,     1,     1 | F32     | mm.0.bias
      3:       1280 |  1280,     1,     1,     1 | F32     | v.class_embd
      4:     752640 |    14,    14,     3,  1280 | F16     | v.patch_embd.weight
      5:          1 |     1,     1,     1,     1 | F32     | v.tile_position_embd.gate
      6:          1 |     1,     1,     1,     1 | F32     | v.position_embd.gate
      7:    2049280 |  1280,  1601,     1,     1 | F16     | v.position_embd.weight
      8:   73774080 | 8197120,   9,     1,     1 | F32     | v.tile_position_embd.weight
      9:          1 |     1,     1,     1,     1 | F32     | v.pre_tile_position_embd.gate
     10:      46080 |  5120,     9,     1,     1 | F32     | v.pre_tile_position_embd.weight
     11:          1 |     1,     1,     1,     1 | F32     | v.post_tile_position_embd.gate
     12:      46080 |  5120,     9,     1,     1 | F32     | v.post_tile_position_embd.weight
     13:       1280 |  1280,     1,     1,     1 | F32     | v.pre_ln.weight
     14:       1280 |  1280,     1,     1,     1 | F32     | v.pre_ln.bias
     15:       1280 |  1280,     1,     1,     1 | F32     | v.post_ln.weight
     16:       1280 |  1280,     1,     1,     1 | F32     | v.post_ln.bias
```

llama.cpp:
```console
 4:  525369344 |  4096, 128264,     1,     1 | Q4_1    | token_embd.weight
```

#### `v.patch_embd`
ollama:
```console
752640 |    14,    14,     3,  1280 | F16     | v.patch_embd.weight
```

```console
752640 |    14,    14,     3,  1280 | F16     | v.enc.patch_embd.weight
```

### issue
When I run the Llava example the tensor for the conv2d operation will be
placed on the GPU:
```console
Backend type: CUDA0
inp_conv2d[0] = 0.000000 (isnan=0)
inp_conv2d[1] = 0.000000 (isnan=0)
inp_conv2d[2] = 0.000000 (isnan=0)
inp_conv2d[3] = 0.000000 (isnan=0)
inp_conv2d[4] = 0.000000 (isnan=0)
inp_conv2d[5] = 0.000000 (isnan=0)
inp_conv2d[6] = 0.000000 (isnan=0)
inp_conv2d[7] = 0.000000 (isnan=0)
inp_conv2d[8] = 0.000000 (isnan=0)
inp_conv2d[9] = 0.000000 (isnan=0)
```
The above was printed before the computation graph was executed by the
scheduler.

```console
(gdb) p *model.patch_embeddings
$4 = {type = GGML_TYPE_F16, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x55555605e5a0, ne = {14, 14, 3, 1024}, nb = {
    2, 28, 392, 1176}, op = GGML_OP_NONE, op_params = {0 <repeats 16 times>}, flags = 0, src = {0x0, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0, view_offs = 0, data = 0x7ffd553947a0,
  name = "v.enc.embd.patch.weight", '\000' <repeats 40 times>, extra = 0x0,
  padding = "\000\000\000\000\000\000\000"}
```

```console
(venv) $ ./inspect-model.sh ../llama.cpp/models/llava-1.5.7b-hf.gguf | grep patch
INFO:gguf-dump:* Loading: ../llama.cpp/models/llava-1.5.7b-hf.gguf
     25: UINT32     |        1 | vision.patch_size = 14
     34: STRING     |        1 | vision.clip.patch_merge_type = 'flat'
     98:     602112 |    14,    14,     3,  1024 | F16     | v.enc.embd.patch.weight
```

```console
(gdb) p *inp_raw
$1 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x555555ddeda0, ne = {336, 336, 3, 1}, nb = {
    4, 1344, 451584, 1354752}, op = GGML_OP_NONE, op_params = {0 <repeats 16 times>}, flags = 1, src = {0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0, view_offs = 0, data = 0x7ffab2441000,
  name = "inp_raw", '\000' <repeats 56 times>, extra = 0x0, padding = "\000\000\000\000\000\000\000"}

Backend type: CPU
inp_raw[0] = 1.171218 (isnan=0)
inp_raw[1] = 1.171218 (isnan=0)
inp_raw[2] = 1.171218 (isnan=0)
inp_raw[3] = 1.171218 (isnan=0)
inp_raw[4] = 1.171218 (isnan=0)
inp_raw[5] = 1.171218 (isnan=0)
inp_raw[6] = 1.127423 (isnan=0)
inp_raw[7] = 1.112824 (isnan=0)
inp_raw[8] = 1.069029 (isnan=0)
inp_raw[9] = 1.025234 (isnan=0)

Backend type: CUDA0
inp_conv2d[0] = 0.000000 (isnan=0)
inp_conv2d[1] = 0.000000 (isnan=0)
inp_conv2d[2] = 0.000000 (isnan=0)
inp_conv2d[3] = 0.000000 (isnan=0)
inp_conv2d[4] = 0.000000 (isnan=0)
inp_conv2d[5] = 0.000000 (isnan=0)
inp_conv2d[6] = 0.000000 (isnan=0)
inp_conv2d[7] = 0.000000 (isnan=0)
inp_conv2d[8] = 0.000000 (isnan=0)
inp_conv2d[9] = 0.000000 (isnan=0)

Backend type: CPU
patch_embeddings[0] = 0.000000 (isnan=0)
patch_embeddings[1] = 0.000000 (isnan=0)
patch_embeddings[2] = 0.000000 (isnan=0)
patch_embeddings[3] = 0.000000 (isnan=0)
patch_embeddings[4] = 0.000000 (isnan=0)
patch_embeddings[5] = 0.000000 (isnan=0)
patch_embeddings[6] = 0.000000 (isnan=0)
patch_embeddings[7] = -0.000000 (isnan=0)
patch_embeddings[8] = -0.000000 (isnan=0)
patch_embeddings[9] = 0.000000 (isnan=0)
```


MLlama example:
```console
inp_raw Backend type: CPU
inp_raw[0] = 1.156620 (isnan=0)
inp_raw[1] = 1.156620 (isnan=0)
inp_raw[2] = 1.171218 (isnan=0)
inp_raw[3] = 1.171218 (isnan=0)
inp_raw[4] = 1.171218 (isnan=0)
inp_raw[5] = 1.156620 (isnan=0)
inp_raw[6] = 1.171218 (isnan=0)
inp_raw[7] = 1.156620 (isnan=0)
inp_raw[8] = 1.112824 (isnan=0)
inp_raw[9] = 1.098226 (isnan=0)

inp_after_conv2d Backend type: CUDA0
inp_after_conv2d[0] = 0.000000 (isnan=0)
inp_after_conv2d[1] = 0.000000 (isnan=0)
inp_after_conv2d[2] = 0.000000 (isnan=0)
inp_after_conv2d[3] = 0.000000 (isnan=0)
inp_after_conv2d[4] = 0.000000 (isnan=0)
inp_after_conv2d[5] = 0.000000 (isnan=0)
inp_after_conv2d[6] = 0.000000 (isnan=0)
inp_after_conv2d[7] = 0.000000 (isnan=0)
inp_after_conv2d[8] = 0.000000 (isnan=0)
inp_after_conv2d[9] = 0.000000 (isnan=0)

Backend type: CPU
patch_embeddings[0] = 0.000000 (isnan=0)
patch_embeddings[1] = -0.000000 (isnan=0)
patch_embeddings[2] = 0.000000 (isnan=0)
patch_embeddings[3] = 0.000000 (isnan=0)
patch_embeddings[4] = 0.000000 (isnan=0)
patch_embeddings[5] = 0.000000 (isnan=0)
patch_embeddings[6] = 0.000000 (isnan=0)
patch_embeddings[7] = 0.000000 (isnan=0)
patch_embeddings[8] = -0.000000 (isnan=0)
patch_embeddings[9] = 0.000000 (isnan=0)


```
```console
(gdb) p *model.patch_embeddings
$1 = {type = GGML_TYPE_F16, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x55555a4830e0, ne = {14, 14, 3, 1280}, nb = {
    2, 28, 392, 1176}, op = GGML_OP_NONE, op_params = {0 <repeats 16 times>}, flags = 0, src = {0x0, 0x0, 0x0, 0x0,




Backend type: CPU
after_pre_tile_position_embeddings[0] = 0.242798 (isnan=0)
after_pre_tile_position_embeddings[1] = -0.385010 (isnan=0)
after_pre_tile_position_embeddings[2] = 0.044067 (isnan=0)
after_pre_tile_position_embeddings[3] = -0.288818 (isnan=0)
after_pre_tile_position_embeddings[4] = -0.059143 (isnan=0)
after_pre_tile_position_embeddings[5] = -0.113281 (isnan=0)
after_pre_tile_position_embeddings[6] = -0.445801 (isnan=0)
after_pre_tile_position_embeddings[7] = 0.079895 (isnan=0)
after_pre_tile_position_embeddings[8] = -0.218384 (isnan=0)
after_pre_tile_position_embeddings[9] = -0.167725 (isnan=0)
Backend type: CPU
embeddings[0] = 0.242798 (isnan=0)
embeddings[1] = -0.385010 (isnan=0)
embeddings[2] = 0.044067 (isnan=0)
embeddings[3] = -0.288818 (isnan=0)
embeddings[4] = -0.059143 (isnan=0)
embeddings[5] = -0.113281 (isnan=0)
embeddings[6] = -0.445801 (isnan=0)
embeddings[7] = 0.079895 (isnan=0)
embeddings[8] = -0.218384 (isnan=0)
embeddings[9] = -0.167725 (isnan=0)
Backend type: CPU
after_class_embeddings[0] = -0.167969 (isnan=0)
after_class_embeddings[1] = 0.072754 (isnan=0)
after_class_embeddings[2] = -0.002396 (isnan=0)
after_class_embeddings[3] = 0.021851 (isnan=0)
after_class_embeddings[4] = 0.035400 (isnan=0)
after_class_embeddings[5] = -0.068359 (isnan=0)
after_class_embeddings[6] = 0.074219 (isnan=0)
after_class_embeddings[7] = -0.016602 (isnan=0)
after_class_embeddings[8] = -0.004120 (isnan=0)
after_class_embeddings[9] = 0.045898 (isnan=0)
Backend type: CPU
positions[0] = 0
positions[1] = 1
positions[2] = 2
positions[3] = 3
positions[4] = 4
positions[5] = 5
positions[6] = 6
positions[7] = 7
positions[8] = 8
positions[9] = 9
Backend type: CUDA0
after_position_embd[0] = -0.216930 (isnan=0)
after_position_embd[1] = 0.131767 (isnan=0)
after_position_embd[2] = -0.003434 (isnan=0)
after_position_embd[3] = 0.061733 (isnan=0)
after_position_embd[4] = 0.079498 (isnan=0)
after_position_embd[5] = -0.074155 (isnan=0)
after_position_embd[6] = 0.101131 (isnan=0)
after_position_embd[7] = 0.004637 (isnan=0)
after_position_embd[8] = 0.001554 (isnan=0)
after_position_embd[9] = 0.087240 (isnan=0)


    0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0, view_offs = 0, data = 0x7ffb804a4dc0,
  name = "v.enc.patch_embd.weight", '\000' <repeats 40 times>, extra = 0x0,
  padding = "\000\000\000\000\000\000\000"}
```

```console
(venv) $ ./inspect-model.sh models/llama-3-2-11b.gguf | grep patch
INFO:gguf-dump:* Loading: models/llama-3-2-11b.gguf
     37: UINT32     |        1 | vision.patch_size = 14
    173:     752640 |    14,    14,     3,  1280 | F16     | v.enc.patch_embd.weight
```
I should fix the inconsistency with the naming here, this shouuld be `v.enc.embd.patch`.

```console
(gdb) p *inp_raw
$6 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x5555567393b0, ne = {560, 560, 3, 4}, nb = {
    4, 2240, 1254400, 3763200}, op = GGML_OP_NONE, op_params = {0 <repeats 16 times>}, flags = 1, src = {0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0, view_offs = 0, data = 0x7ff8a1f45000,
  name = "inp_raw", '\000' <repeats 56 times>, extra = 0x0, padding = "\000\000\000\000\000\000\000"}
```


llama.cpp:
```console
(venv) $ python read-tensor.py models/llama-3-2-11b.gguf v.enc.embd.pos_gate

Tensor Information:
Name: v.enc.embd.pos_gate
Shape: 1
Type: F32
Total elements: 1

First 10 values:
[0] = -1.328125
```
Ollama:

(venv) $ python read-tensor.py /home/danbev/.ollama/models/blobs/sha256-622429e8d31810962dd984bc98559e706db2fb1d40e99cb073beb7148d909d73 v.position_embd.gate

Tensor Information:
Name: v.position_embd.gate
Shape: 1
Type: F32
Total elements: 1

First 10 values:
[0] = 1.8687903881072998
``
And the orignal model has this value:
```console
Tensor Information:
Name: vision_model.gated_positional_embedding.gate
Shape: torch.Size([1])
Type: torch.bfloat16
First 10 values:
[0] = -1.328125
```


```console
Tensor Information:
Name: vision_model.gated_positional_embedding.embedding
Shape: torch.Size([1601, 1280])
Type: torch.bfloat16
First 10 values:
[0] = 0.036865234375
[1] = -0.04443359375
[2] = 0.000782012939453125
[3] = -0.030029296875
[4] = -0.033203125
[5] = 0.004364013671875
[6] = -0.020263671875
[7] = -0.0159912109375
[8] = -0.0042724609375
[9] = -0.0311279296875
```


(gdb) p *KQ_mul
$1 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0, ne = {6432, 6432, 16, 1}, nb = {4,
    25728, 165482496, 2647719936}, op = GGML_OP_MUL_MAT, op_params = {0 <repeats 16 times>}, flags = 0, src = {
    0x55555a4ac420, 0x55555a4abe60, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0, view_offs = 0,
  data = 0x0, name = "KQ_mul-0", '\000' <repeats 55 times>, extra = 0x0, padding = "\000\000\000\000\000\000\000"}

(gdb) p *KQV_mul
$5 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0, ne = {80, 6432, 16, 1}, nb = {4, 320,
    2058240, 32931840}, op = GGML_OP_MUL_MAT, op_params = {0 <repeats 16 times>}, flags = 0, src = {0x55555a4ac9e0,
    0x55555a4ace30, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0, view_offs = 0, data = 0x0,
  name = '\000' <repeats 63 times>, extra = 0x0, padding = "\000\000\000\000\000\000\000"}


### Image processing issue
Using the same image this is the output from ollama:
```console
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:70 msg="[danbev] Image.NewEmbed" i=0 data=28
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:70 msg="[danbev] Image.NewEmbed" i=1 data=12
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:70 msg="[danbev] Image.NewEmbed" i=2 data=148
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:70 msg="[danbev] Image.NewEmbed" i=3 data=63
Bytes [0]: 28, 12, 148, 63

time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:70 msg="[danbev] Image.NewEmbed" i=4 data=28
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:70 msg="[danbev] Image.NewEmbed" i=5 data=12
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:70 msg="[danbev] Image.NewEmbed" i=6 data=148
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:70 msg="[danbev] Image.NewEmbed" i=7 data=63
Bytes [1]: 28, 12, 148, 63

time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:70 msg="[danbev] Image.NewEmbed" i=8 data=28
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:70 msg="[danbev] Image.NewEmbed" i=9 data=12
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:70 msg="[danbev] Image.NewEmbed" i=10 data=148
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:70 msg="[danbev] Image.NewEmbed" i=11 data=63
Bytes [2]: 28, 12, 148, 63

time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:70 msg="[danbev] Image.NewEmbed" i=12 data=28
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:70 msg="[danbev] Image.NewEmbed" i=13 data=12
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:70 msg="[danbev] Image.NewEmbed" i=14 data=148
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:70 msg="[danbev] Image.NewEmbed" i=15 data=63
Bytes [3]: 28, 12, 148, 63

time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:70 msg="[danbev] Image.NewEmbed" i=16 data=28
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:70 msg="[danbev] Image.NewEmbed" i=17 data=12
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:70 msg="[danbev] Image.NewEmbed" i=18 data=148
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:70 msg="[danbev] Image.NewEmbed" i=19 data=63
Bytes [4]: 28, 12, 148, 63

time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:80 msg="As float32 " i=0 data=1.1566195487976074
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:80 msg="As float32 " i=1 data=1.1566195487976074
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:80 msg="As float32 " i=2 data=1.1566195487976074
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:80 msg="As float32 " i=3 data=1.1566195487976074
Bytes [5]: 121, 234, 149, 63
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:80 msg="As float32 " i=4 data=1.1566195487976074
Bytes [6]: 121, 234, 149, 63
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:80 msg="As float32 " i=5 data=1.1712180376052856
Bytes [7]: 28, 12, 148, 63
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:80 msg="As float32 " i=6 data=1.1712180376052856
Bytes [8]: 121, 234, 149, 63
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:80 msg="As float32 " i=7 data=1.1566195487976074
Bytes [9]: 121, 234, 149, 63
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:80 msg="As float32 " i=8 data=1.1712180376052856
Bytes [10]: 121, 234, 149, 63
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:80 msg="As float32 " i=9 data=1.1712180376052856
Bytes [11]: 121, 234, 149, 63
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:80 msg="As float32 " i=10 data=1.1712180376052856
Bytes [12]: 121, 234, 149, 63
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:80 msg="As float32 " i=11 data=1.1712180376052856
Bytes [13]: 28, 12, 148, 63
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:80 msg="As float32 " i=12 data=1.1712180376052856
Bytes [14]: 28, 12, 148, 63
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:80 msg="As float32 " i=13 data=1.1566195487976074
Bytes [15]: 121, 234, 149, 63
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:80 msg="As float32 " i=14 data=1.1566195487976074
Bytes [16]: 121, 234, 149, 63
Bytes [17]: 121, 234, 149, 63
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:80 msg="As float32 " i=15 data=1.1712180376052856
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:80 msg="As float32 " i=16 data=1.1712180376052856
Bytes [18]: 28, 12, 148, 63
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:80 msg="As float32 " i=17 data=1.1712180376052856
Bytes [19]: 28, 12, 148, 63
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:80 msg="As float32 " i=18 data=1.1566195487976074
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:80 msg="As float32 " i=19 data=1.1566195487976074
And this is the output I get after processing:
```console
Normalized[0] = 1.156620
Normalized[1] = 1.156620
Normalized[2] = 1.156620
Normalized[3] = 1.156620
Normalized[4] = 1.156620
Normalized[5] = 1.156620
Normalized[6] = 1.171218
Normalized[7] = 1.171218
Normalized[8] = 1.156620
Normalized[9] = 1.171218
Normalized[10] = 1.171218
Normalized[11] = 1.171218
Normalized[12] = 1.171218
Normalized[13] = 1.171218
Normalized[14] = 1.171218
Normalized[15] = 1.156620
Normalized[16] = 1.156620
Normalized[17] = 1.156620
Normalized[18] = 1.171218
Normalized[19] = 1.171218
[simple-vision-mllama] loaded image data[0] = 28
[simple-vision-mllama] loaded image data[1] = 12
[simple-vision-mllama] loaded image data[2] = 148
[simple-vision-mllama] loaded image data[3] = 63
[simple-vision-mllama] loaded image data[4] = 28
[simple-vision-mllama] loaded image data[5] = 12
[simple-vision-mllama] loaded image data[6] = 148
[simple-vision-mllama] loaded image data[7] = 63
[simple-vision-mllama] loaded image data[8] = 28
[simple-vision-mllama] loaded image data[9] = 12
[simple-vision-mllama] loaded image data[10] = 148
[simple-vision-mllama] loaded image data[11] = 63
[simple-vision-mllama] loaded image data[12] = 28
[simple-vision-mllama] loaded image data[13] = 12
[simple-vision-mllama] loaded image data[14] = 148
[simple-vision-mllama] loaded image data[15] = 63
[simple-vision-mllama] loaded image data[16] = 28
[simple-vision-mllama] loaded image data[17] = 12
[simple-vision-mllama] loaded image data[18] = 148
[simple-vision-mllama] loaded image data[19] = 63
```
So these look pretty similar. Now, let print out the values before
we set them as inputs before the graph computation.

This is from llama.cpp (vision.cpp):
```console
Before encode_image_with_ca_vision:
Input image size: 560x560
Input data[0] = 1.156620
Input data[1] = 1.156620
Input data[2] = 1.156620
Input data[3] = 1.156620
Input data[4] = 1.156620
Input data[5] = 1.156620
Input data[6] = 1.171218
Input data[7] = 1.171218
Input data[8] = 1.156620
Input data[9] = 1.171218
Input data[10] = 1.171218
Input data[11] = 1.171218
Input data[12] = 1.171218
Input data[13] = 1.171218
Input data[14] = 1.171218
Input data[15] = 1.156620
Input data[16] = 1.156620
Input data[17] = 1.156620
Input data[18] = 1.171218
Input data[19] = 1.171218


inp_raw backend type: CPU
First values of inp_raw:
inp_raw[0] = 1.156620
inp_raw[1] = 1.156620
inp_raw[2] = 1.171218
inp_raw[3] = 1.171218
inp_raw[4] = 1.171218
inp_raw[5] = 1.156620
inp_raw[6] = 1.171218
inp_raw[7] = 1.156620
inp_raw[8] = 1.112824
inp_raw[9] = 1.098226
inp_raw[10] = 1.083627
inp_raw[11] = 0.996037
inp_raw[12] = 0.981439
inp_raw[13] = 0.908446
inp_raw[14] = 0.850053
inp_raw[15] = 0.820856
inp_raw[16] = 0.820856
inp_raw[17] = 0.820856
inp_raw[18] = 0.791659
inp_raw[19] = 0.791659
inp_raw[20] = 0.791659
inp_raw[21] = 0.806257
inp_raw[22] = 0.937643
inp_raw[23] = 1.142021
inp_raw[24] = 1.127423
inp_raw[25] = 0.966840
inp_raw[26] = 1.419391
inp_raw[27] = 1.638368
inp_raw[28] = 1.725958
inp_raw[29] = 1.448588
```

```console
First values after inp_raw:
inp_raw[0] = 1.156620 (isnan=0)
inp_raw[1] = 1.156620 (isnan=0)
inp_raw[2] = 1.156620 (isnan=0)
inp_raw[3] = 1.156620 (isnan=0)
inp_raw[4] = 1.156620 (isnan=0)
inp_raw[5] = 1.171218 (isnan=0)
inp_raw[6] = 1.171218 (isnan=0)
inp_raw[7] = 1.156620 (isnan=0)
inp_raw[8] = 1.171218 (isnan=0)
inp_raw[9] = 1.171218 (isnan=0)
inp_raw[10] = 1.171218 (isnan=0)
inp_raw[11] = 1.171218 (isnan=0)
inp_raw[12] = 1.171218 (isnan=0)
inp_raw[13] = 1.156620 (isnan=0)
inp_raw[14] = 1.156620 (isnan=0)
inp_raw[15] = 1.171218 (isnan=0)
inp_raw[16] = 1.171218 (isnan=0)
inp_raw[17] = 1.171218 (isnan=0)
inp_raw[18] = 1.156620 (isnan=0)
inp_raw[19] = 1.156620 (isnan=0)
inp_raw[20] = 1.142021 (isnan=0)
inp_raw[21] = 1.127423 (isnan=0)
inp_raw[22] = 1.112824 (isnan=0)
inp_raw[23] = 1.112824 (isnan=0)
inp_raw[24] = 1.112824 (isnan=0)
inp_raw[25] = 1.112824 (isnan=0)
inp_raw[26] = 1.098226 (isnan=0)
inp_raw[27] = 1.098226 (isnan=0)
inp_raw[28] = 1.025234 (isnan=0)
inp_raw[29] = 0.996037 (isnan=0)
```

```console
[danbev] num_positions: 1601
[danbev] image data length: 15052800
[danbev] image width: 560
[danbev] image height: 560
```
So inp_raw is image witdh * image height * channels * tiles:
```console
(gdb) p ggml_nelements(inp_raw)
$4 = 3763200
(gdb) p ggml_nelements(inp_raw) * 4
$5 = 15052800
(gdb) p ggml_nbytes(inp_raw)
$6 = 15052800

(gdb) p 560 * 560 * 3 * 4
$7 = 3763200
(gdb) p 560 * 560 * 3 * 4 * 4
$8 = 15052800
```
And we have 4 bytes per entry which is the last 4. Now, in ollama they can
just to the following:
```c++
    {
        struct ggml_tensor *inp_raw = ggml_graph_get_tensor(gf, "inp_raw");
        ggml_backend_tensor_set(inp_raw, imgs->data[0].data.data(), 0, ggml_nbytes(inp_raw));
    }
```
What is the size of the image data that is passed in here?
In my example the size is Copying 13102080 bytes of data


### Prediction issue
```console
inp_after_conv2d Backend type: CUDA0
inp_after_conv2d[0] = -0.002579 (isnan=0)
inp_after_conv2d[1] = 0.000140 (isnan=0)
inp_after_conv2d[2] = 0.001678 (isnan=0)
inp_after_conv2d[3] = -0.001488 (isnan=0)
inp_after_conv2d[4] = 0.000246 (isnan=0)
inp_after_conv2d[5] = 0.000538 (isnan=0)
inp_after_conv2d[6] = -0.003235 (isnan=0)
inp_after_conv2d[7] = 0.000212 (isnan=0)
inp_after_conv2d[8] = -0.000094 (isnan=0)
inp_after_conv2d[9] = 0.001617 (isnan=0)
```
```console
inp_after_conv2d[0] = -4.334547 (isnan=0)
inp_after_conv2d[1] = -0.271873 (isnan=0)
inp_after_conv2d[2] = -0.259376 (isnan=0)
inp_after_conv2d[3] = 0.546166 (isnan=0)
inp_after_conv2d[4] = -0.929878 (isnan=0)
inp_after_conv2d[5] = -3.958198 (isnan=0)
inp_after_conv2d[6] = 3.839755 (isnan=0)
inp_after_conv2d[7] = 9.393863 (isnan=0)
inp_after_conv2d[8] = 6.277499 (isnan=0)
inp_after_conv2d[9] = -2.536955 (isnan=0)
```


pre_tile_position_embeddings_gate[0] = 0.635149 (isnan=0)





inp_after_conv2d Backend type: CUDA0
inp_after_conv2d[0] = -0.002579 (isnan=0)
inp_after_conv2d[1] = 0.000140 (isnan=0)
inp_after_conv2d[2] = 0.001678 (isnan=0)
inp_after_conv2d[3] = -0.001488 (isnan=0)
inp_after_conv2d[4] = 0.000246 (isnan=0)
inp_after_conv2d[5] = 0.000538 (isnan=0)
inp_after_conv2d[6] = -0.003235 (isnan=0)
inp_after_conv2d[7] = 0.000212 (isnan=0)
inp_after_conv2d[8] = -0.000094 (isnan=0)
inp_after_conv2d[9] = 0.001617 (isnan=0)


Tensor type: f32
Tensor backend buffer type: CUDA0
inp_after_conv2d[0] = -4.334547 (isnan=0)
inp_after_conv2d[1] = -0.271873 (isnan=0)
inp_after_conv2d[2] = -0.259376 (isnan=0)
inp_after_conv2d[3] = 0.546166 (isnan=0)
inp_after_conv2d[4] = -0.929878 (isnan=0)
inp_after_conv2d[5] = -3.958198 (isnan=0)
inp_after_conv2d[6] = 3.839755 (isnan=0)
inp_after_conv2d[7] = 9.393863 (isnan=0)
inp_after_conv2d[8] = 6.277499 (isnan=0)
inp_after_conv2d[9] = -2.536955 (isnan=0)



Tensor type: f32
inp_after_conv2d Backend type: CUDA0
inp_after_conv2d[0] = 4.817005 (isnan=0)
inp_after_conv2d[1] = -1.839836 (isnan=0)
inp_after_conv2d[2] = 0.149719 (isnan=0)
inp_after_conv2d[3] = -1.052750 (isnan=0)
inp_after_conv2d[4] = 0.045052 (isnan=0)
inp_after_conv2d[5] = 5.482178 (isnan=0)
inp_after_conv2d[6] = -20.486389 (isnan=0)
inp_after_conv2d[7] = -20.188850 (isnan=0)
inp_after_conv2d[8] = -5.085636 (isnan=0)
inp_after_conv2d[9] = -12.865143 (isnan=0)

index 6:
model.pre_tile_positional_embeddings[6][0] = 0.008728 (isnan=0)
model.pre_tile_positional_embeddings[6][1] = -0.194336 (isnan=0)
model.pre_tile_positional_embeddings[6][2] = -0.074707 (isnan=0)
model.pre_tile_positional_embeddings[6][3] = -0.125000 (isnan=0)
model.pre_tile_positional_embeddings[6][4] = -0.125977 (isnan=0)
model.pre_tile_positional_embeddings[6][5] = -0.157227 (isnan=0)
model.pre_tile_positional_embeddings[6][6] = 0.049316 (isnan=0)
model.pre_tile_positional_embeddings[6][7] = -0.098145 (isnan=0)
model.pre_tile_positional_embeddings[6][8] = 0.080078 (isnan=0)
model.pre_tile_positional_embeddings[6][9] = -0.134766 (isnan=0)
model.pre_tile_positional_embeddings[6][10] = 0.010620 (isnan=0)
model.pre_tile_positional_embeddings[6][11] = -0.159180 (isnan=0)
model.pre_tile_positional_embeddings[6][12] = -0.187500 (isnan=0)
model.pre_tile_positional_embeddings[6][13] = -0.119141 (isnan=0)
model.pre_tile_positional_embeddings[6][14] = 0.025269 (isnan=0)
model.pre_tile_positional_embeddings[6][15] = -0.283203 (isnan=0)
model.pre_tile_positional_embeddings[6][16] = -0.166016 (isnan=0)
model.pre_tile_positional_embeddings[6][17] = -0.114258 (isnan=0)
model.pre_tile_positional_embeddings[6][18] = -0.213867 (isnan=0)
model.pre_tile_positional_embeddings[6][19] = -0.097168 (isnan=0)

The ones from get_rows:
pre_tile_position_embeddings[0] = 0.151923 (isnan=0)
pre_tile_position_embeddings[1] = 1.211459 (isnan=0)
pre_tile_position_embeddings[2] = 4.014451 (isnan=0)
pre_tile_position_embeddings[3] = 0.946178 (isnan=0)
pre_tile_position_embeddings[4] = -5.457583 (isnan=0)
pre_tile_position_embeddings[5] = -0.536786 (isnan=0)
pre_tile_position_embeddings[6] = -12.600449 (isnan=0)
pre_tile_position_embeddings[7] = -4.407394 (isnan=0)
pre_tile_position_embeddings[8] = -6.397855 (isnan=0)
pre_tile_position_embeddings[9] = -1.396894 (isnan=0)
pre_tile_position_embeddings[10] = -2.710128 (isnan=0)
pre_tile_position_embeddings[11] = 5.050854 (isnan=0)
pre_tile_position_embeddings[12] = 4.036346 (isnan=0)
pre_tile_position_embeddings[13] = -1.571802 (isnan=0)
pre_tile_position_embeddings[14] = -7.525723 (isnan=0)
pre_tile_position_embeddings[15] = -0.567268 (isnan=0)
pre_tile_position_embeddings[16] = -3.055859 (isnan=0)
pre_tile_position_embeddings[17] = 5.793323 (isnan=0)
pre_tile_position_embeddings[18] = 3.643304 (isnan=0)
pre_tile_position_embeddings[19] = -5.682245 (isnan=0)
pre_tile_position_embeddings[20] = -4.672400 (isnan=0)
pre_tile_position_embeddings[21] = -0.146351 (isnan=0)
pre_tile_position_embeddings[22] = 6.713018 (isnan=0)
pre_tile_position_embeddings[23] = -2.168883 (isnan=0)
pre_tile_position_embeddings[24] = -0.882346 (isnan=0)
pre_tile_position_embeddings[25] = -1.765321 (isnan=0)
pre_tile_position_embeddings[26] = 0.616241 (isnan=0)
pre_tile_position_embeddings[27] = -2.195093 (isnan=0)
pre_tile_position_embeddings[28] = -0.563926 (isnan=0)
pre_tile_position_embeddings[29] = -1.999931 (isnan=0)

```console
(venv) $ python read-tensor.py models/llama-3-2-11b.gguf v.enc.embd.patch.weight

Tensor Information:
Name: v.enc.embd.patch.weight
Shape: 14 x 14 x 3 x 1280
Type: F32
Total elements: 752640

First 10 values:
[0] = 0.0064697265625
[1] = 0.00543212890625
[2] = 4.4345855712890625e-05
[3] = -0.00933837890625
[4] = -0.00579833984375
[5] = 0.00836181640625
[6] = 0.0037994384765625
[7] = 0.0054931640625
[8] = 0.0157470703125
[9] = 0.00299072265625
```
```console
(venv) $ python read-tensor.py /home/danbev/.ollama/models/blobs/sha256-622429e8d31810962dd984bc98559e706db2fb1d40e99cb073beb7148d909d73 v.patch_embd.weight

Tensor Information:
Name: v.patch_embd.weight
Shape: 14 x 14 x 3 x 1280
Type: F16
Total elements: 752640

First 10 values:
[0] = 0.0064697265625
[1] = 0.00543212890625
[2] = 4.4345855712890625e-05
[3] = -0.00933837890625
[4] = -0.00579833984375
[5] = 0.00836181640625
[6] = 0.0037994384765625
[7] = 0.0054931640625
[8] = 0.0157470703125
[9] = 0.00299072265625
```
But when I read this tensor in llama.cpp I get this:
```console
Tensor type: f32
model.patch_embeddings[0] = 0.000000 (isnan=0)
model.patch_embeddings[1] = 0.978516 (isnan=0)
model.patch_embeddings[2] = 0.000000 (isnan=0)
model.patch_embeddings[3] = 0.961914 (isnan=0)
model.patch_embeddings[4] = 0.000000 (isnan=0)
model.patch_embeddings[5] = 0.528320 (isnan=0)
model.patch_embeddings[6] = 0.000000 (isnan=0)
model.patch_embeddings[7] = -1.024414 (isnan=0)
model.patch_embeddings[8] = 0.000000 (isnan=0)
model.patch_embeddings[9] = -0.967773 (isnan=0)
```
Ah this was because I was reading/converting to float16 because I've gone back
and forth on the type of tensors when testing.
```console
Backend type: CPU
Tensor type: f32
model.patch_embeddings[0] = 0.006470
model.patch_embeddings[1] = 0.005432
model.patch_embeddings[2] = 0.000044
model.patch_embeddings[3] = -0.009338
model.patch_embeddings[4] = -0.005798
model.patch_embeddings[5] = 0.008362
model.patch_embeddings[6] = 0.003799
model.patch_embeddings[7] = 0.005493
model.patch_embeddings[8] = 0.015747
model.patch_embeddings[9] = 0.002991
```
And in ollama they are:
```console
Tensor backend buffer type: CUDA0
Tensor type: f16
model.patch_embeddings[0] = 0.006470
model.patch_embeddings[1] = 0.005432
model.patch_embeddings[2] = 0.000044
model.patch_embeddings[3] = -0.009338
model.patch_embeddings[4] = -0.005798
model.patch_embeddings[5] = 0.008362
model.patch_embeddings[6] = 0.003799
model.patch_embeddings[7] = 0.005493
model.patch_embeddings[8] = 0.015747
model.patch_embeddings[9] = 0.002991
```

In ollama:
```console
pre_tile_position_embeddings[0] = 2.360522 (isnan=0)
pre_tile_position_embeddings[1] = 0.542406 (isnan=0)
pre_tile_position_embeddings[2] = 1.326151 (isnan=0)
pre_tile_position_embeddings[3] = -0.383138 (isnan=0)
pre_tile_position_embeddings[4] = -0.896916 (isnan=0)
pre_tile_position_embeddings[5] = -0.538128 (isnan=0)
pre_tile_position_embeddings[6] = -0.362522 (isnan=0)
pre_tile_position_embeddings[7] = 0.237987 (isnan=0)
pre_tile_position_embeddings[8] = 0.090309 (isnan=0)
pre_tile_position_embeddings[9] = -1.181206 (isnan=0)
```

### pre_tile_position_embeddings
```console
Backend type: CPU
Tensor type: f32
pre_tile_position_embeddings[0] = 0.172945 (isnan=0)
pre_tile_position_embeddings[1] = 1.207378 (isnan=0)
pre_tile_position_embeddings[2] = 3.902387 (isnan=0)
pre_tile_position_embeddings[3] = 1.060965 (isnan=0)
pre_tile_position_embeddings[4] = -5.529974 (isnan=0)
pre_tile_position_embeddings[5] = -0.492022 (isnan=0)
pre_tile_position_embeddings[6] = -12.454433 (isnan=0)
pre_tile_position_embeddings[7] = -4.333441 (isnan=0)
pre_tile_position_embeddings[8] = -6.424330 (isnan=0)
pre_tile_position_embeddings[9] = -1.306060 (isnan=0)
```
Now these values are gotten by using a get rows operation, so there is not
math operation or anything that should mess with the values. So we should see
the same values I believe.
```c++
        struct ggml_tensor * tile_position_embeddings = ggml_get_rows(ctx0, model.tile_position_embeddings, aspect_ratios);
        ggml_set_name(tile_position_embeddings, "tile_position_embd");
```
The aspect ratio tensor is 
```console
Backend type: CPU
Tensor type: f32
Values from row 6:
```
This is from llama.cpp:
```console
model.pre_tile_positional_embeddings[6][0] = 0.008728
model.pre_tile_positional_embeddings[6][1] = -0.194336
model.pre_tile_positional_embeddings[6][2] = -0.074707
model.pre_tile_positional_embeddings[6][3] = -0.125000
model.pre_tile_positional_embeddings[6][4] = -0.125977
model.pre_tile_positional_embeddings[6][5] = -0.157227
model.pre_tile_positional_embeddings[6][6] = 0.049316
model.pre_tile_positional_embeddings[6][7] = -0.098145
model.pre_tile_positional_embeddings[6][8] = 0.080078
model.pre_tile_positional_embeddings[6][9] = -0.134766
```
And this is from ollama:
```console
```

```console
(venv) $ ./read-tensor2.py models/llama-3-2-11b.gguf v.enc.pre_tile_pos_embd.weight 6

Tensor Information:
Name: v.enc.pre_tile_pos_embd.weight
Shape: 5120 x 9
Type: F32
Total elements: 46080

Values for row 6 (up to 10):
[6, 0] = 0.00872802734375
[6, 1] = -0.1943359375
[6, 2] = -0.07470703125
[6, 3] = -0.125
[6, 4] = -0.1259765625
[6, 5] = -0.1572265625
[6, 6] = 0.04931640625
[6, 7] = -0.09814453125
[6, 8] = 0.080078125
[6, 9] = -0.134765625
```

```console
(venv) $ ./read-tensor2.py /home/danbev/.ollama/models/blobs/sha256-622429e8d31810962dd984bc98559e706db2fb1d40e99cb073beb7148d909d73 v.pre_tile_position_embd.weight 6

Tensor Information:
Name: v.pre_tile_position_embd.weight
Shape: 5120 x 9
Type: F32
Total elements: 46080

Values for row 6 (up to 10):
[6, 0] = 0.00872802734375
[6, 1] = -0.1943359375
[6, 2] = -0.07470703125
[6, 3] = -0.125
[6, 4] = -0.1259765625
[6, 5] = -0.1572265625
[6, 6] = 0.04931640625
[6, 7] = -0.09814453125
[6, 8] = 0.080078125
[6, 9] = -0.134765625
```

But this is what actually get selected:
```console
Backend type: CPU
Tensor type: f32
pre_tile_position_embeddings[0] = -0.633367 (isnan=0)
pre_tile_position_embeddings[1] = 6.685422 (isnan=0)
pre_tile_position_embeddings[2] = -1.039765 (isnan=0)
pre_tile_position_embeddings[3] = 0.576086 (isnan=0)
pre_tile_position_embeddings[4] = 4.381856 (isnan=0)
pre_tile_position_embeddings[5] = -0.598374 (isnan=0)
pre_tile_position_embeddings[6] = -7.988229 (isnan=0)
pre_tile_position_embeddings[7] = -0.992554 (isnan=0)
pre_tile_position_embeddings[8] = -2.655503 (isnan=0)
pre_tile_position_embeddings[9] = -5.103714 (isnan=0)
```

### Difference in models
In llama.cpp the value of `position_embeddings_gate` is:
```console
Backend type: CPU
Tensor type: f32
    model.position_embeddings_gate = -1.328125            MISMATCH

Backend type: CPU
Tensor type: f32
    model.pre_tile_position_embeddings_gate = 0.750000    MISMATCH

Backend type: CPU
Tensor type: f32
    model.post_tile_position_embeddings_gate = -0.197266  same value
```

But in ollama it is:
```console
Tensor type: f32
Tensor backend buffer type: CUDA0
tile_position_embeddings_gate?[0] = -0.868790              MISSMATCH

Tensor type: f32
Tensor backend buffer type: CUDA0
pre_tile_position_embeddings_gate = 0.635149               MISSMATCH

Tensor type: f32
Tensor backend buffer type: CUDA0
post_tile_position_embeddings_gate = -0.194746             same value
```

If I read this value from our model I get:
```console
(venv) $ ./read-tensor.py models/llama-3-2-11b.gguf v.enc.embd.pos_gate

Tensor Information:
Name: v.enc.embd.pos_gate
Shape: 1
Type: F32
Total elements: 1

First 10 values:
[0] = -1.328125
```
And if we read the same value from the safetensor we get:
```console
(venv) $ python read-safetensor.py

Tensor Information:
Name: vision_model.gated_positional_embedding.gate
Shape: torch.Size([1])
Type: torch.bfloat16
First 10 values:
[0] = -1.328125

Tensor Information:
Name: vision_model.post_tile_positional_embedding.gate
Shape: torch.Size([1])
Type: torch.bfloat16
First 10 values:
[0] = -0.197265625

Tensor Information:
Name: vision_model.pre_tile_positional_embedding.gate
Shape: torch.Size([1])
Type: torch.bfloat16
First 10 values:
[0] = 0.75
```

v.enc.embd.pos_gate      : -0.8687899708747864
v.enc.pre_tile_pos_gate  : 0.6351490020751953
v.enc.post_tile_pos_gate : -0.197265625



```console
(venv) $ ./read-tensor.py /home/danbev/.ollama/models/blobs/sha256-622429e8d31810962dd984bc98559e706db2fb1d40e99cb073beb7148d909d73 v.position_embd.weight

Tensor Information:
Name: v.position_embd.weight
Shape: 1280 x 1601
Type: F16
Total elements: 2049280

First 10 values:
[0] = 0.036865234375
[1] = -0.04443359375
[2] = 0.000782012939453125
[3] = -0.030029296875
[4] = -0.033203125
[5] = 0.004364013671875
[6] = -0.020263671875
[7] = -0.0159912109375
[8] = -0.0042724609375
[9] = -0.0311279296875

(venv) $ ./read-tensor.py models/llama-3-2-11b.gguf v.enc.embd.pos

Tensor Information:
Name: v.enc.embd.pos
Shape: 1280 x 1601
Type: F32
Total elements: 2049280

First 10 values:
[0] = 0.036865234375
[1] = -0.04443359375
[2] = 0.000782012939453125
[3] = -0.030029296875
[4] = -0.033203125
[5] = 0.004364013671875
[6] = -0.020263671875
[7] = -0.0159912109375
[8] = -0.0042724609375
[9] = -0.0311279296875

```



Tensor type: f32
model.post_norm_w[0] = 1.289062
model.post_norm_w[1] = 1.304688
model.post_norm_w[2] = 1.296875
model.post_norm_w[3] = 1.250000
model.post_norm_w[4] = 1.218750
model.post_norm_w[5] = 1.250000
model.post_norm_w[6] = 1.296875
model.post_norm_w[7] = 1.250000
model.post_norm_w[8] = 1.289062
model.post_norm_w[9] = 1.335938


(venv) $ ./read-tensor.py /home/danbev/.ollama/models/blobs/sha256-622429e8d31810962dd984bc98559e706db2fb1d40e99cb073beb7148d909d73 v.post_ln.weight

Tensor Information:
Name: v.post_ln.weight
Shape: 1280
Type: F32
Total elements: 1280

First 10 values:
[0] = 1.2890625
[1] = 1.3046875
[2] = 1.296875
[3] = 1.25
[4] = 1.21875
[5] = 1.25
[6] = 1.296875
[7] = 1.25
[8] = 1.2890625
[9] = 1.3359375

(venv) $ ./read-tensor.py /home/danbev/.ollama/models/blobs/sha256-622429e8d31810962dd984bc98559e706db2fb1d40e99cb073beb7148d909d73 v.pre_ln.weight

Tensor Information:
Name: v.pre_ln.weight
Shape: 1280
Type: F32
Total elements: 1280

First 10 values:
[0] = 0.005157470703125
[1] = -0.0086669921875
[2] = 0.984375
[3] = 0.00031280517578125
[4] = -0.00139617919921875
[5] = 0.006195068359375
[6] = 0.427734375
[7] = 0.00153350830078125
[8] = 1.265625
[9] = -0.004974365234375


31457280 |  7680,  4096,     1,     1 | F32     | v.enc.mmproj.weight
    4096 |  4096,     1,     1,     1 | F32     | v.enc.mmproj.bias

31457280 |  7680,  4096,     1,     1 | F16     | mm.0.weight
    4096 |  4096,     1,     1,     1 | F32     | mm.0.bias



### Tokenizer
```console
(venv) $ ./inspect-model.sh /home/danbev/.ollama/models/blobs/sha256-652e85aa1e14c9087a4ccc3ab516fb794cbcf152f8b4b8d3c0b828da4ada62d9
INFO:gguf-dump:* Loading: /home/danbev/.ollama/models/blobs/sha256-652e85aa1e14c9087a4ccc3ab516fb794cbcf152f8b4b8d3c0b828da4ada62d9
* File is LITTLE endian, script is running on a LITTLE endian host.
* Dumping 29 key/value pair(s)
      1: UINT32     |        1 | GGUF.version = 3
      2: UINT64     |        1 | GGUF.tensor_count = 396
      3: UINT64     |        1 | GGUF.kv_count = 26
      4: STRING     |        1 | general.architecture = 'mllama'
      5: STRING     |        1 | general.type = 'model'
      6: STRING     |        1 | general.name = 'Model'
      7: STRING     |        1 | general.size_label = '10B'
      8: UINT32     |        1 | mllama.block_count = 40
      9: UINT32     |        1 | mllama.context_length = 131072
     10: UINT32     |        1 | mllama.embedding_length = 4096
     11: UINT32     |        1 | mllama.feed_forward_length = 14336
     12: UINT32     |        1 | mllama.attention.head_count = 32
     13: UINT32     |        1 | mllama.attention.head_count_kv = 8
     14: FLOAT32    |        1 | mllama.rope.freq_base = 500000.0
     15: FLOAT32    |        1 | mllama.attention.layer_norm_rms_epsilon = 9.999999747378752e-06
     16: UINT32     |        1 | general.file_type = 15
     17: UINT32     |        1 | mllama.vocab_size = 128256
     18: UINT32     |        1 | mllama.rope.dimension_count = 128
     19: [INT32]    |        8 | mllama.attention.cross_attention_layers
     20: STRING     |        1 | tokenizer.ggml.model = 'gpt2'
     21: STRING     |        1 | tokenizer.ggml.pre = 'smaug-bpe'
     22: [STRING]   |   128257 | tokenizer.ggml.tokens
     23: [INT32]    |   128257 | tokenizer.ggml.token_type
     24: [STRING]   |   280147 | tokenizer.ggml.merges
     25: UINT32     |        1 | tokenizer.ggml.bos_token_id = 128000
     26: UINT32     |        1 | tokenizer.ggml.eos_token_id = 128009
     27: UINT32     |        1 | tokenizer.ggml.padding_token_id = 128004
     28: STRING     |        1 | tokenizer.chat_template = '{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- s'
     29: UINT32     |        1 | general.quantization_version = 2

```
```console




(venv) $ ./inspect-model.sh models/llama-3-2-11b.gguf
INFO:gguf-dump:* Loading: models/llama-3-2-11b.gguf
* File is LITTLE endian, script is running on a LITTLE endian host.
* Dumping 56 key/value pair(s)
      1: UINT32     |        1 | GGUF.version = 3
      2: UINT64     |        1 | GGUF.tensor_count = 907
      3: UINT64     |        1 | GGUF.kv_count = 53
      4: STRING     |        1 | general.architecture = 'mllama'
      5: STRING     |        1 | general.type = 'model'
      6: STRING     |        1 | general.name = 'Llama 3.2 11B Vision Instruct'
      7: STRING     |        1 | general.finetune = 'Vision-Instruct'
      8: STRING     |        1 | general.basename = 'Llama-3.2'
      9: STRING     |        1 | general.size_label = '11B'
     10: STRING     |        1 | general.license = 'llama3.2'
     11: [STRING]   |        6 | general.tags
     12: [STRING]   |        8 | general.languages
     13: UINT32     |        1 | mllama.image_token_index = 128256
     14: UINT32     |        1 | mllama.context_length = 131072
     15: UINT32     |        1 | mllama.block_count = 40
     16: UINT32     |        1 | mllama.embedding_length = 4096
     17: UINT32     |        1 | mllama.feed_forward_length = 14336
     18: UINT32     |        1 | mllama.attention.head_count = 32
     19: UINT32     |        1 | mllama.attention.head_count_kv = 8
     20: FLOAT32    |        1 | mllama.rope.freq_base = 500000.0
     21: FLOAT32    |        1 | mllama.attention.layer_norm_rms_epsilon = 9.999999747378752e-06
     22: UINT32     |        1 | general.file_type = 1
     23: [INT32]    |        8 | mllama.cross_attention_layers
     24: UINT32     |        1 | mllama.vocab_size = 128257
     25: UINT32     |        1 | mllama.rope.dimension_count = 128
     26: STRING     |        1 | vision.type = 'cross-vit'
     27: STRING     |        1 | vision.architecture = 'mllama_vision_model'
     28: UINT32     |        1 | vision.image_size = 560
     29: UINT32     |        1 | vision.block_count = 32
     30: FLOAT32    |        1 | vision.attention.layer_norm_rms_epsilon = 9.999999747378752e-06
     31: UINT32     |        1 | vision.embedding_length = 1280
     32: STRING     |        1 | vision.cross.mllama.activation_function = 'gelu'
     33: UINT32     |        1 | vision.feed_forward_length = 5120
     34: UINT32     |        1 | vision.cross.mllama.global_block_count = 8
     35: UINT32     |        1 | vision.cross.mllama.max_num_tiles = 4
     36: UINT32     |        1 | vision.cross.mllama.channels_count = 3
     37: UINT32     |        1 | vision.patch_size = 14
     38: [INT32]    |        5 | vision.cross.mllama.intermediate_layers_indices
     39: UINT32     |        1 | vision.attention.head_count = 16
     40: UINT32     |        1 | vision.cross.mllama.output_dim = 7680
     41: STRING     |        1 | vision.cross.mllama.model_type = 'mllama_vision_model'
     42: UINT32     |        1 | vision.clip.max_position_embeddings = 1601
     43: [INT32]    |       16 | vision.cross.mllama.supported_aspect_ratios
     44: [FLOAT32]  |        3 | vision.image_mean
     45: [FLOAT32]  |        3 | vision.image_std
     46: UINT32     |        1 | vision.clip.projection_dim = 7680
     47: STRING     |        1 | tokenizer.ggml.model = 'gpt2'
     48: STRING     |        1 | tokenizer.ggml.pre = 'smaug-bpe'
     49: [STRING]   |   128257 | tokenizer.ggml.tokens
     50: [INT32]    |   128257 | tokenizer.ggml.token_type
     51: [STRING]   |   280147 | tokenizer.ggml.merges
     52: UINT32     |        1 | tokenizer.ggml.bos_token_id = 128000
     53: UINT32     |        1 | tokenizer.ggml.eos_token_id = 128009
     54: UINT32     |        1 | tokenizer.ggml.padding_token_id = 128004
     55: STRING     |        1 | tokenizer.chat_template = '{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- s'
     56: UINT32     |        1 | general.quantization_version = 2
 ```

### Vocab
llama.cpp:
```console
  47: STRING     |        1 | tokenizer.ggml.model = 'gpt2'
     48: STRING     |        1 | tokenizer.ggml.pre = 'smaug-bpe'
     49: [STRING]   |   128257 | tokenizer.ggml.tokens
     50: [INT32]    |   128257 | tokenizer.ggml.token_type
     51: [STRING]   |   280147 | tokenizer.ggml.merges
     52: UINT32     |        1 | tokenizer.ggml.bos_token_id = 128000
     53: UINT32     |        1 | tokenizer.ggml.eos_token_id = 128009
     54: UINT32     |        1 | tokenizer.ggml.padding_token_id = 128004
     55: UINT32     |        1 | tokenizer.ggml.eot_token_id = 128000
     56: UINT32     |        1 | tokenizer.ggml.start_header_token_id = 128006
     57: UINT32     |        1 | tokenizer.ggml.end_header_token_id = 128007
     58: UINT32     |        1 | tokenizer.ggml.eom_token_id = 128008
     59: UINT32     |        1 | tokenizer.ggml.python_tag_token_id = 128010
     60: UINT32     |        1 | tokenizer.ggml.image_token_id = 128256
     61: STRING     |        1 | tokenizer.chat_template = '{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- s'
     62: UINT32     |        1 | general.quantization_version = 2
 ```
 ollama:
 ```console
     20: STRING     |        1 | tokenizer.ggml.model = 'gpt2'
     21: STRING     |        1 | tokenizer.ggml.pre = 'smaug-bpe'
     22: [STRING]   |   128257 | tokenizer.ggml.tokens
     23: [INT32]    |   128257 | tokenizer.ggml.token_type
     24: [STRING]   |   280147 | tokenizer.ggml.merges
     25: UINT32     |        1 | tokenizer.ggml.bos_token_id = 128000
     26: UINT32     |        1 | tokenizer.ggml.eos_token_id = 128009
     27: UINT32     |        1 | tokenizer.ggml.padding_token_id = 128004
     28: STRING     |        1 | tokenizer.chat_template = '{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- s'
     29: UINT32     |        1 | general.quantization_version = 2


```
Now, if I run the ollama language model with the same program (no image) I get:
```console
prompt = <|begin_of_text|><|start_header_id|>user<|end_header_id|>

What is the Eiffel Tower?<|eot_id|><|start_header_id|>assistant<|end_header_id|>


token = 128000
token = 128006
token = 882
token = 128007
token = 271
token = 3923
token = 374
token = 279
token = 469
token = 3168
token = 301
token = 22703
token = 30
token = 128009
token = 128006
token = 78191
token = 128007
token = 271
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 1)
[New Thread 0x7fffabe00000 (LWP 1840205)]
[New Thread 0x7fffab400000 (LWP 1840206)]
[New Thread 0x7fffaaa00000 (LWP 1840207)]
ggml_backend_cuda_graph_compute: disabling CUDA graphs due to batch size > 1 [l_out-24] [4096 18 1 1]
The Eiffel Tower is an iconic iron lattice tower located in Paris,ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 0)
 France. It was built for the 1889 World's Fair, held to celebrate
main: decoded 32 tokens in 13.38 s, speed: 2.39 t/s


llama_perf_context_print:        load time =   14388.28 ms
llama_perf_context_print: prompt eval time =    6141.74 ms /    18 tokens (  341.21 ms per token,     2.93 tokens per second)
llama_perf_context_print:        eval time =   12827.71 ms /    31 runs   (  413.80 ms per token,     2.42 tokens per second)
llama_perf_context_print:       total time =   27771.62 ms /    49 tokens
[Thread 0x7fffc5e00000 (LWP 1840196) exited]
[Thread 0x7fffaaa00000 (LWP 1840207) exited]
[Thread 0x7fffab400000 (LWP 1840206) exited]
[Thread 0x7fffabe00000 (LWP 1840205) exited]
[Thread 0x7fffc5400000 (LWP 1840197) exited]
[Thread 0x7fffc7a00000 (LWP 1840189) exited]
[Thread 0x7ffff339b000 (LWP 1840184) exited]
[Thread 0x7fffc4a00000 (LWP 1840198) exited]
[New process 1840184]
[Inferior 1 (process 1840184) exited normally]
```
But when I run this with my converted model I don't get as good of an answer
as this.
```console
prompt = <|begin_of_text|><|start_header_id|>user<|end_header_id|>

What is the Eiffel Tower?<|eot_id|><|start_header_id|>assistant<|end_header_id|>


token = 128000
token = 128006
token = 882
token = 128007
token = 271
token = 3923
token = 374
token = 279
token = 469
token = 3168
token = 301
token = 22703
token = 30
token = 128009
token = 128006
token = 78191
token = 128007
token = 271
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 1)
[New Thread 0x7fffabe00000 (LWP 1843202)]
[New Thread 0x7fffab400000 (LWP 1843203)]
[New Thread 0x7fffaaa00000 (LWP 1843204)]
ggml_backend_cuda_graph_compute: disabling CUDA graphs due to batch size > 1 [l_out-24] [4096 18 1 1]
I think you meant to type "Eiffel Tower" doesn't seemggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 0)
 to be a well-known landmark.

main: decoded 22 tokens in 19.26 s, speed: 1.14 t/s


llama_perf_context_print:        load time =   46051.65 ms
llama_perf_context_print: prompt eval time =   18818.83 ms /    18 tokens ( 1045.49 ms per token,     0.96 tokens per second)
llama_perf_context_print:        eval time =   19149.48 ms /    22 runs   (  870.43 ms per token,     1.15 tokens per second)
llama_perf_context_print:       total time =   65314.36 ms /    40 tokens
[Thread 0x7fffc5e00000 (LWP 1843158) exited]
[Thread 0x7fffaaa00000 (LWP 1843204) exited]
[Thread 0x7fffab400000 (LWP 1843203) exited]
[Thread 0x7fffabe00000 (LWP 1843202) exited]
[Thread 0x7fffc4a00000 (LWP 1843160) exited]
[Thread 0x7fffc5400000 (LWP 1843159) exited]
[Thread 0x7ffff339b000 (LWP 1843146) exited]
[Thread 0x7fffc7a00000 (LWP 1843149) exited]
[New process 1843146]
```
SO the tokens are the exact same. And the model graphs seems to work fine as
it can respond with a good response for the ollama model.
Could there be difference in the actual model.

```console
prompt = What is the Eiffel Tower?
token = 3923
token = 374
token = 279
token = 469
token = 3168
token = 301
token = 22703
token = 30
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 1)
[New Thread 0x7fffabe00000 (LWP 1846433)]
[New Thread 0x7fffab400000 (LWP 1846434)]
[New Thread 0x7fffaaa00000 (LWP 1846435)]
ggml_backend_cuda_graph_compute: disabling CUDA graphs due to batch size > 1 [l_out-24] [4096 8 1 1]
**
 : 1035
The : 791
 E : 469
iff : 3168
el : 301
 Tower : 22703
 is : 374
 a : 264
  : 220
1 : 16
, : 11
000 : 931
-ton : 75735
ne : 818
 steel : 9699
 structure : 6070
 that : 430
 stands : 13656
  : 220
1 : 16
, : 11
000 : 931
 feet : 7693
 above : 3485
 the : 279
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 0)
 ground : 5015
 and : 323
 is : 374
 a : 264
  : 220
1 : 16
, : 11
```

```console
token = 128000
token = 128006
token = 882
token = 128007
token = 271
token = 3923
token = 374
token = 279
token = 469
token = 3168
token = 301
token = 22703
token = 30
token = 128009
token = 128006
token = 78191
token = 128007
token = 271
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 1)
[New Thread 0x7fffabe00000 (LWP 1847036)]
[New Thread 0x7fffab400000 (LWP 1847037)]
[New Thread 0x7fffaaa00000 (LWP 1847038)]
ggml_backend_cuda_graph_compute: disabling CUDA graphs due to batch size > 1 [l_out-24] [4096 18 1 1]
I : 40
 think : 1781
 you : 499
 meant : 8967
 to : 311
 type : 955
 " : 330
E : 36
iff : 3168
el : 301
 Tower : 22703
" : 1
 doesn : 3250
't : 956
 seem : 2873
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 0)
 to : 311
 be : 387
 a : 264
 well : 1664
-known : 22015
 landmark : 38350
. : 13
```

### Troubleshooting
I've tried using my converted model with only the language model and this did
not initially work when using template formatting. But it worked well without
and could give a good anwser to the question: "What is the Eiffel Tower?".

So I managed to find an issue with one of the weights which has the output 
norm which I had mistakenly set to be the same as the attention norm.


### Computation graph language model
We first have the token embeddings matrix which is what is used to get the
token embeddings from the token indices:
```c++
model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab + 7}, 0);
```
This from the tensor named `token_embd`
```console
(venv) $ ./read-tensor.py models/llama-3-2-11b.gguf token_embd.weight

Tensor Information:
Name: token_embd.weight
Shape: 4096 x 128264
Type: F32
Total elements: 525369344

First 10 values:
[0] = 0.001007080078125
[1] = 0.005584716796875
[2] = -0.0034027099609375
[3] = -0.0012359619140625
[4] = -0.003570556640625
[5] = 0.0006256103515625
[6] = -0.001495361328125
[7] = -0.002166748046875
[8] = -0.0036163330078125
[9] = -0.00433349609375
```
Ollama's:
```console
(venv) $ ./read-tensor.py /home/danbev/.ollama/models/blobs/sha256-652e85aa1e14c9087a4ccc3ab516fb794cbcf152f8b4b8d3c0b828da4ada62d9 token_embd.weight

Tensor Information:
Name: token_embd.weight
Shape: 4096 x 128264
Type: Q4_K
Total elements: 525369344

First 10 values:
[0] = 111
[1] = 3
[2] = 37
[3] = 13
[4] = 161
[5] = 191
[6] = 171
[7] = 161
[8] = 229
[9] = 250
```


Notice that this is larger then the language vocabulary by 8 and I think this
is because there are 8 special tokens that are added to the vocabulary.

How this done is by `llm_build_inp_embd` which uses ggml_get_rows:
```c++
        lctx.inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.n_tokens);
        cb(lctx.inp_tokens, "inp_tokens", -1);
        ggml_set_input(lctx.inp_tokens);

        inpL = ggml_get_rows(ctx, tok_embd, lctx.inp_tokens);
```

The first thing in a layer is the attention normalization:
```c++
            // norm
            cur = llm_build_norm(ctx0, inpL, hparams,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);
```
```c++
layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
```
This is from the tensor named `attn_norm.weight`
```console
(venv) $ ./read-tensor.py models/llama-3-2-11b.gguf blk.0.attn_norm.weight

Tensor Information:
Name: blk.0.attn_norm.weight
Shape: 4096
Type: F32
Total elements: 4096

First 10 values:
[0] = 0.047119140625
[1] = 0.1875
[2] = 0.41796875
[3] = 0.01708984375
[4] = 0.43359375
[5] = 0.021484375
[6] = -0.00020599365234375
[7] = 0.004547119140625
[8] = 0.0341796875
[9] = 0.024658203125
```
Ollam's:
```console
(venv) $ ./read-tensor.py /home/danbev/.ollama/models/blobs/sha256-652e85aa1e14c9087a4ccc3ab516fb794cbcf152f8b4b8d3c0b828da4ada62d9 blk.0.attn_norm.weight

Tensor Information:
Name: blk.0.attn_norm.weight
Shape: 4096
Type: F32
Total elements: 4096

First 10 values:
[0] = 0.047119140625
[1] = 0.1875
[2] = 0.41796875
[3] = 0.01708984375
[4] = 0.43359375
[5] = 0.021484375
[6] = -0.00020599365234375
[7] = 0.004547119140625
[8] = 0.0341796875
[9] = 0.024658203125
```

After this we have a condition for if the layer we are iterating over is one
of the cross attention layers:
```console
  23: [INT32]    |        8 | mllama.cross_attention_layers
```
These are layers 3, 8, 13, 18, 23, 28, 33, 38.

After the attention computation we then have:
```c++
      // feed-forward network
                cur = llm_build_norm(ctx0, ffn_inp, hparams,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, cb, il);
                cb(cur, "ffn_norm", il);
```
And this is using the tensor named `blk.%d.post_attention_norm`:
```console
(venv) $ ./read-tensor.py models/llama-3-2-11b.gguf blk.0.post_attention_norm.weight

Tensor Information:
Name: blk.0.post_attention_norm.weight
Shape: 4096
Type: F32
Total elements: 4096

First 10 values:
[0] = 0.134765625
[1] = 0.125
[2] = 0.1376953125
[3] = 0.1357421875
[4] = 0.1259765625
[5] = 0.134765625
[6] = 0.134765625
[7] = 0.134765625
[8] = 0.134765625
[9] = 0.134765625
```
Ollama's:
```console
(venv) $ ./read-tensor.py /home/danbev/.ollama/models/blobs/sha256-652e85aa1e14c9087a4ccc3ab516fb794cbcf152f8b4b8d3c0b828da4ada62d9 blk.0.ffn_norm.weight

Tensor Information:
Name: blk.0.ffn_norm.weight
Shape: 4096
Type: F32
Total elements: 4096

First 10 values:
[0] = 0.134765625
[1] = 0.125
[2] = 0.1376953125
[3] = 0.1357421875
[4] = 0.1259765625
[5] = 0.134765625
[6] = 0.134765625
[7] = 0.134765625
[8] = 0.134765625
[9] = 0.134765625
```

After the layers we then have:
```c++
        cur = llm_build_norm(ctx0, cur, hparams,
                model.output_norm, NULL,
                LLM_NORM_RMS, cb, -1);
        cb(cur, "result_norm", -1);
```
This is using the tensor named `output_norm.weight`:
```console
(venv) $ ./read-tensor.py models/llama-3-2-11b.gguf output_norm.weight

Tensor Information:
Name: output_norm.weight
Shape: 4096
Type: F32
Total elements: 4096

First 10 values:
[0] = 2.46875
[1] = 2.390625
[2] = 2.53125
[3] = 2.421875
[4] = 2.390625
[5] = 2.46875
[6] = 2.265625
[7] = 2.4375
[8] = 2.296875
[9] = 2.328125
```
Ollama's:
```console
(venv) $ ./read-tensor.py /home/danbev/.ollama/models/blobs/sha256-652e85aa1e14c9087a4ccc3ab516fb794cbcf152f8b4b8d3c0b828da4ada62d9 output_norm.weight

Tensor Information:
Name: output_norm.weight
Shape: 4096
Type: F32
Total elements: 4096

First 10 values:
[0] = 2.46875
[1] = 2.390625
[2] = 2.53125
[3] = 2.421875
[4] = 2.390625
[5] = 2.46875
[6] = 2.265625
[7] = 2.4375
[8] = 2.296875
[9] = 2.328125

```

And this normalized tensor is then multiplied by the tensor named `output.weight`:
```console
(venv) $ ./read-tensor.py models/llama-3-2-11b.gguf output.weight

Tensor Information:
Name: output.weight
Shape: 4096 x 128256
Type: F32
Total elements: 525336576

First 10 values:
[0] = 0.0081787109375
[1] = 0.007171630859375
[2] = 0.012451171875
[3] = 0.023681640625
[4] = -0.017578125
[5] = 0.01275634765625
[6] = -0.02001953125
[7] = -0.005279541015625
[8] = -0.0015411376953125
[9] = 0.01556396484375
```
Ollama's:
```console
(venv) $ ./read-tensor.py /home/danbev/.ollama/models/blobs/sha256-652e85aa1e14c9087a4ccc3ab516fb794cbcf152f8b4b8d3c0b828da4ada62d9 output.weight

Tensor Information:
Name: output.weight
Shape: 4096 x 128256
Type: Q6_K
Total elements: 525336576

First 10 values:
[0] = 5
[1] = 182
[2] = 191
[3] = 160
[4] = 184
[5] = 159
[6] = 27
[7] = 247
[8] = 98
[9] = 11
```

The espilon value are in ollama:
```console
9.999999747378752e-06
9.999999747378752e-06
```
And in llama.cpp:
```console
9.999999747378752e-06
```

```console
(venv) $ ./read-tensor.py models/llama-3-2-11b.gguf blk.0.ffn_gate.weight

Tensor Information:
Name: blk.0.ffn_gate.weight
Shape: 4096 x 14336
Type: F16
Total elements: 58720256

First 10 values:
[0] = -0.0166015625
[1] = -0.0062255859375
[2] = -0.0013885498046875
[3] = -0.000461578369140625
[4] = 0.007293701171875
[5] = 0.0038604736328125
[6] = -0.0037994384765625
[7] = -0.0274658203125
[8] = -0.021728515625
[9] = 0.00131988525390625
```

```console
(venv) $ ./read-tensor.py /home/danbev/.ollama/models/blobs/sha256-652e85aa1e14c9087a4ccc3ab516fb794cbcf152f8b4b8d3c0b828da4ada62d9 blk.0.ffn_gate.weight

Tensor Information:
Name: blk.0.ffn_gate.weight
Shape: 4096 x 14336
Type: Q4_K
Total elements: 58720256

First 10 values:
[0] = 232
[1] = 3
[2] = 39
[3] = 15
[4] = 246
[5] = 247
[6] = 251
[7] = 248
[8] = 255
[9] = 248
```

```console
(venv) $ ./read-tensor.py models/llama-3-2-11b.gguf blk.0.attn_output.weight

Tensor Information:
Name: blk.0.attn_output.weight
Shape: 4096 x 4096
Type: F16
Total elements: 16777216

First 10 values:
[0] = 0.00592041015625
[1] = -0.001983642578125
[2] = -0.0101318359375
[3] = -0.00110626220703125
[4] = 0.003387451171875
[5] = -0.00994873046875
[6] = -0.00848388671875
[7] = -0.000186920166015625
[8] = -0.00015735626220703125
[9] = -0.00130462646484375
```
```console
(venv) $ ./read-tensor.py /home/danbev/.ollama/models/blobs/sha256-652e85aa1e14c9087a4ccc3ab516fb794cbcf152f8b4b8d3c0b828da4ada62d9 blk.0.attn_output.weight

Tensor Information:
Name: blk.0.attn_output.weight
Shape: 4096 x 4096
Type: Q4_K
Total elements: 16777216

First 10 values:
[0] = 16
[1] = 3
[2] = 48
[3] = 13
[4] = 230
[5] = 227
[6] = 229
[7] = 229
[8] = 225
[9] = 229
```

```console
(venv) $ ./read-tensor.py /home/danbev/.ollama/models/blobs/sha256-652e85aa1e14c9087a4ccc3ab516fb794cbcf152f8b4b8d3c0b828da4ada62d9 blk.3.cross_attn_mlp_gate

Tensor Information:
Name: blk.3.cross_attn_mlp_gate
Shape: 1
Type: F32
Total elements: 1

First 10 values:
[0] = 0.006256103515625

```console
(venv) $ ./read-tensor.py models/llama-3-2-11b.gguf blk.3.cross_attn_ffn_gate

Tensor Information:
Name: blk.3.cross_attn_ffn_gate
Shape: 1
Type: F32
Total elements: 1

First 10 values:
[0] = 0.006256103515625
```


```console
(venv) $ ./read-tensor.py models/llama-3-2-11b.gguf blk.3.cross_attn_gate

Tensor Information:
Name: blk.3.cross_attn_gate
Shape: 1
Type: F32
Total elements: 1

First 10 values:
[0] = 0.000545501708984375
```
```console
(venv) $ ./read-tensor.py /home/danbev/.ollama/models/blobs/sha256-652e85aa1e14c9087a4ccc3ab516fb794cbcf152f8b4b8d3c0b828da4ada62d9 blk.3.cross_attn_attn_gate

Tensor Information:
Name: blk.3.cross_attn_attn_gate
Shape: 1
Type: F32
Total elements: 1

First 10 values:
[0] = 0.000545501708984375
```

```
(venv) $ ./read-tensor.py models/llama-3-2-11b.gguf blk.3.cross_attn_q.weight

Tensor Information:
Name: blk.3.cross_attn_q.weight
Shape: 4096 x 4096
Type: F16
Total elements: 16777216

First 10 values:
[0] = -0.01263427734375
[1] = 0.0615234375
[2] = 0.02392578125
[3] = -0.0023193359375
[4] = -0.004852294921875
[5] = 0.017333984375
[6] = 0.0576171875
[7] = 0.000858306884765625
[8] = 0.0198974609375
[9] = -0.033203125
```
```console
(venv) $ ./read-tensor.py /home/danbev/.ollama/models/blobs/sha256-652e85aa1e14c9087a4ccc3ab516fb794cbcf152f8b4b8d3c0b828da4ada62d9 blk.3.cross_attn_q_proj.weight

Tensor Information:
Name: blk.3.cross_attn_q_proj.weight
Shape: 4096 x 4096
Type: Q4_K
Total elements: 16777216

First 10 values:
[0] = 69
[1] = 10
[2] = 1
[3] = 22
[4] = 163
[5] = 95
[6] = 175
[7] = 191
[8] = 215
[9] = 93
```
```console
(venv) $ ./read-tensor.py models/llama-3-2-11b.gguf blk.3.cross_attn_k_norm.weight

Tensor Information:
Name: blk.3.cross_attn_k_norm.weight
Shape: 128
Type: F32
Total elements: 128

First 10 values:
[0] = 1.3359375
[1] = 1.359375
[2] = 1.28125
[3] = 1.2890625
[4] = 1.2578125
[5] = 1.2890625
[6] = 1.3125
[7] = 1.296875
[8] = 1.3359375
[9] = 1.1484375
```
```console
(venv) $ ./read-tensor.py /home/danbev/.ollama/models/blobs/sha256-652e85aa1e14c9087a4ccc3ab516fb794cbcf152f8b4b8d3c0b828da4ada62d9 blk.3.cross_attn_k_norm.weight

Tensor Information:
Name: blk.3.cross_attn_k_norm.weight
Shape: 128
Type: F32
Total elements: 128

First 10 values:
[0] = 1.3359375
[1] = 1.359375
[2] = 1.28125
[3] = 1.2890625
[4] = 1.2578125
[5] = 1.2890625
[6] = 1.3125
[7] = 1.296875
[8] = 1.3359375
[9] = 1.1484375
```

```console
(venv) $ ./read-tensor.py models/llama-3-2-11b.gguf blk.3.cross_attn_q_norm.weight

Tensor Information:
Name: blk.3.cross_attn_q_norm.weight
Shape: 128
Type: F32
Total elements: 128

First 10 values:
[0] = 1.3359375
[1] = 1.359375
[2] = 1.28125
[3] = 1.2890625
[4] = 1.2578125
[5] = 1.2890625
[6] = 1.3125
[7] = 1.296875
[8] = 1.3359375
[9] = 1.1484375
```
```console
(venv) $ ./read-tensor.py /home/danbev/.ollama/models/blobs/sha256-652e85aa1e14c9087a4ccc3ab516fb794cbcf152f8b4b8d3c0b828da4ada62d9 blk.3.cross_attn_q_norm.weight

Tensor Information:
Name: blk.3.cross_attn_q_norm.weight
Shape: 128
Type: F32
Total elements: 128

First 10 values:
[0] = 1.3359375
[1] = 1.359375
[2] = 1.28125
[3] = 1.2890625
[4] = 1.2578125
[5] = 1.2890625
[6] = 1.3125
[7] = 1.296875
[8] = 1.3359375
[9] = 1.1484375
```

```console
(venv) $ ./read-tensor.py models/llama-3-2-11b.gguf rope_freqs.weight

Tensor Information:
Name: rope_freqs.weight
Shape: 64
Type: F32
Total elements: 64

First 10 values:
[0] = 1.0
[1] = 1.0
[2] = 1.0
[3] = 1.0
[4] = 1.0
[5] = 1.0
[6] = 1.0
[7] = 1.0
[8] = 1.0
[9] = 1.0

```

```console
(venv) $ ./read-tensor.py /home/danbev/.ollama/models/blobs/sha256-652e85aa1e14c9087a4ccc3ab516fb794cbcf152f8b4b8d3c0b828da4ada62d9 rope_freqs.weight

Tensor Information:
Name: rope_freqs.weight
Shape: 64
Type: F32
Total elements: 64

First 10 values:
[0] = 1.0
[1] = 1.0
[2] = 1.0
[3] = 1.0
[4] = 1.0
[5] = 1.0
[6] = 1.0
[7] = 1.0
[8] = 1.0
[9] = 1.0
```

Now if I run the llama.cpp code but use the ollama model I get the following
for a non-image prompt:
```console
llama_new_context_with_model: graph splits = 225 (with bs=512), 3 (with bs=1)
prompt = <|start_header_id|>user<|end_header_id|>

What is the Eiffel Tower?<|eot_id|><|start_header_id|>assistant<|end_header_id|>


token = 128006
token = 882
token = 128007
token = 271
token = 3923
token = 374
token = 279
token = 469
token = 3168
token = 301
token = 22703
token = 30
token = 128009
token = 128006
token = 78191
token = 128007
token = 271
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 1)
ggml_gallocr_reserve_n: reallocating CUDA_Host buffer from size 16.01 MiB to 426.13 MiB
[New Thread 0x7fffabe00000 (LWP 110381)]
[New Thread 0x7fffab400000 (LWP 110382)]
[New Thread 0x7fffaaa00000 (LWP 110383)]
ggml_backend_cuda_graph_compute: disabling CUDA graphs due to batch size > 1 [l_out-24] [4096 17 1 1]
ggml_backend_cuda_graph_compute: CUDA graph update failed
The : 791
ggml_backend_cuda_graph_compute: CUDA graph update failed
ggml_backend_cuda_graph_compute: CUDA graph update failed
ggml_backend_cuda_graph_compute: CUDA graph update failed
 E : 469
ggml_backend_cuda_graph_compute: CUDA graph update failed
ggml_backend_cuda_graph_compute: CUDA graph update failed
ggml_backend_cuda_graph_compute: CUDA graph update failed
iff : 3168
ggml_backend_cuda_graph_compute: CUDA graph update failed
ggml_backend_cuda_graph_compute: CUDA graph update failed
ggml_backend_cuda_graph_compute: CUDA graph update failed
el : 301
ggml_backend_cuda_graph_compute: CUDA graph update failed
ggml_backend_cuda_graph_compute: CUDA graph update failed
ggml_backend_cuda_graph_compute: CUDA graph update failed
 Tower : 22703
ggml_backend_cuda_graph_compute: CUDA graph update failed
ggml_backend_cuda_graph_compute: CUDA graph update failed
ggml_backend_cuda_graph_compute: CUDA graph update failed
 is : 374
ggml_backend_cuda_graph_compute: CUDA graph update failed
ggml_backend_cuda_graph_compute: CUDA graph update failed
ggml_backend_cuda_graph_compute: CUDA graph update failed
 a : 264
ggml_backend_cuda_graph_compute: CUDA graph update failed
ggml_backend_cuda_graph_compute: CUDA graph update failed
ggml_backend_cuda_graph_compute: CUDA graph update failed
 famous : 11495
```

With my version of llama.cpp:
```console
token = 128006
token = 882
token = 128007
token = 271
token = 3923
token = 374
token = 279
token = 469
token = 3168
token = 301
token = 22703
token = 30
token = 128009
token = 128006
token = 78191
token = 128007
token = 271
```
So the tokens generated for the input are identical. This will then be used
to lookup the embeddings from the token embeddings matrix.

And it seems to try to predict something from an image even if there is not
image tag nore is there a question about an image.

So the this is using the exact same graph, but the model is not the same. I've
inspected the weights above and they seem to match up, at least the ones that
are not quentized.

Could it be that one of the weights have been mixed up in the case of llama.cpp
and they are not used in the correct place. Perhaps I can change

llama.cpp startup:
```console
llm_load_print_meta: general.name     = Llama 3.2 11B Vision Instruct New
llm_load_print_meta: BOS token        = 128000 '<|begin_of_text|>'
llm_load_print_meta: EOS token        = 128009 '<|eot_id|>'
llm_load_print_meta: EOT token        = 128009 '<|eot_id|>'
llm_load_print_meta: EOM token        = 128008 '<|eom_id|>'
llm_load_print_meta: PAD token        = 128004 '<|finetune_right_pad_id|>'
llm_load_print_meta: LF token         = 128 'Ä'
llm_load_print_meta: EOG token        = 128008 '<|eom_id|>'
llm_load_print_meta: EOG token        = 128009 '<|eot_id|>'
```


ollama startup:
```console
llm_load_vocab: control token: 128256 '<|image|>' is not marked as EOG
llm_load_vocab: control token: 128255 '<|reserved_special_token_246|>' is not marked as EOG
llm_load_vocab: control token: 128250 '<|reserved_special_token_241|>' is not marked as EOG
llm_load_vocab: control token: 128247 '<|reserved_special_token_238|>' is not marked as EOG
llm_load_vocab: control token: 128244 '<|reserved_special_token_235|>' is not marked as EOG
llm_load_vocab: control token: 128243 '<|reserved_special_token_234|>' is not marked as EOG
llm_load_vocab: control token: 128242 '<|reserved_special_token_233|>' is not marked as EOG
llm_load_vocab: control token: 128241 '<|reserved_special_token_232|>' is not marked as EOG
llm_load_vocab: control token: 128236 '<|reserved_special_token_227|>' is not marked as EOG
llm_load_vocab: control token: 128232 '<|reserved_special_token_223|>' is not marked as EOG
llm_load_vocab: control token: 128231 '<|reserved_special_token_222|>' is not marked as EOG
llm_load_vocab: control token: 128229 '<|reserved_special_token_220|>' is not marked as EOG
llm_load_vocab: control token: 128226 '<|reserved_special_token_217|>' is not marked as EOG
llm_load_vocab: control token: 128219 '<|reserved_special_token_210|>' is not marked as EOG
llm_load_vocab: control token: 128215 '<|reserved_special_token_206|>' is not marked as EOG
llm_load_vocab: control token: 128214 '<|reserved_special_token_205|>' is not marked as EOG
llm_load_vocab: control token: 128208 '<|reserved_special_token_199|>' is not marked as EOG
llm_load_vocab: control token: 128207 '<|reserved_special_token_198|>' is not marked as EOG
llm_load_vocab: control token: 128205 '<|reserved_special_token_196|>' is not marked as EOG
llm_load_vocab: control token: 128201 '<|reserved_special_token_192|>' is not marked as EOG
llm_load_vocab: control token: 128200 '<|reserved_special_token_191|>' is not marked as EOG
llm_load_vocab: control token: 128199 '<|reserved_special_token_190|>' is not marked as EOG
llm_load_vocab: control token: 128197 '<|reserved_special_token_188|>' is not marked as EOG
llm_load_vocab: control token: 128195 '<|reserved_special_token_186|>' is not marked as EOG
llm_load_vocab: control token: 128194 '<|reserved_special_token_185|>' is not marked as EOG
llm_load_vocab: control token: 128189 '<|reserved_special_token_180|>' is not marked as EOG
llm_load_vocab: control token: 128188 '<|reserved_special_token_179|>' is not marked as EOG
llm_load_vocab: control token: 128186 '<|reserved_special_token_177|>' is not marked as EOG
llm_load_vocab: control token: 128185 '<|reserved_special_token_176|>' is not marked as EOG
llm_load_vocab: control token: 128181 '<|reserved_special_token_172|>' is not marked as EOG
llm_load_vocab: control token: 128180 '<|reserved_special_token_171|>' is not marked as EOG
llm_load_vocab: control token: 128179 '<|reserved_special_token_170|>' is not marked as EOG
llm_load_vocab: control token: 128178 '<|reserved_special_token_169|>' is not marked as EOG
llm_load_vocab: control token: 128177 '<|reserved_special_token_168|>' is not marked as EOG
llm_load_vocab: control token: 128176 '<|reserved_special_token_167|>' is not marked as EOG
llm_load_vocab: control token: 128172 '<|reserved_special_token_163|>' is not marked as EOG
llm_load_vocab: control token: 128171 '<|reserved_special_token_162|>' is not marked as EOG
llm_load_vocab: control token: 128170 '<|reserved_special_token_161|>' is not marked as EOG
llm_load_vocab: control token: 128169 '<|reserved_special_token_160|>' is not marked as EOG
llm_load_vocab: control token: 128166 '<|reserved_special_token_157|>' is not marked as EOG
llm_load_vocab: control token: 128163 '<|reserved_special_token_154|>' is not marked as EOG
llm_load_vocab: control token: 128159 '<|reserved_special_token_150|>' is not marked as EOG
llm_load_vocab: control token: 128157 '<|reserved_special_token_148|>' is not marked as EOG
llm_load_vocab: control token: 128156 '<|reserved_special_token_147|>' is not marked as EOG
llm_load_vocab: control token: 128155 '<|reserved_special_token_146|>' is not marked as EOG
llm_load_vocab: control token: 128152 '<|reserved_special_token_143|>' is not marked as EOG
llm_load_vocab: control token: 128150 '<|reserved_special_token_141|>' is not marked as EOG
llm_load_vocab: control token: 128148 '<|reserved_special_token_139|>' is not marked as EOG
llm_load_vocab: control token: 128147 '<|reserved_special_token_138|>' is not marked as EOG
llm_load_vocab: control token: 128145 '<|reserved_special_token_136|>' is not marked as EOG
llm_load_vocab: control token: 128143 '<|reserved_special_token_134|>' is not marked as EOG
llm_load_vocab: control token: 128142 '<|reserved_special_token_133|>' is not marked as EOG
llm_load_vocab: control token: 128139 '<|reserved_special_token_130|>' is not marked as EOG
llm_load_vocab: control token: 128137 '<|reserved_special_token_128|>' is not marked as EOG
llm_load_vocab: control token: 128136 '<|reserved_special_token_127|>' is not marked as EOG
llm_load_vocab: control token: 128135 '<|reserved_special_token_126|>' is not marked as EOG
llm_load_vocab: control token: 128134 '<|reserved_special_token_125|>' is not marked as EOG
llm_load_vocab: control token: 128132 '<|reserved_special_token_123|>' is not marked as EOG
llm_load_vocab: control token: 128129 '<|reserved_special_token_120|>' is not marked as EOG
llm_load_vocab: control token: 128125 '<|reserved_special_token_116|>' is not marked as EOG
llm_load_vocab: control token: 128124 '<|reserved_special_token_115|>' is not marked as EOG
llm_load_vocab: control token: 128123 '<|reserved_special_token_114|>' is not marked as EOG
llm_load_vocab: control token: 128120 '<|reserved_special_token_111|>' is not marked as EOG
llm_load_vocab: control token: 128116 '<|reserved_special_token_107|>' is not marked as EOG
llm_load_vocab: control token: 128113 '<|reserved_special_token_104|>' is not marked as EOG
llm_load_vocab: control token: 128111 '<|reserved_special_token_102|>' is not marked as EOG
llm_load_vocab: control token: 128110 '<|reserved_special_token_101|>' is not marked as EOG
llm_load_vocab: control token: 128109 '<|reserved_special_token_100|>' is not marked as EOG
llm_load_vocab: control token: 128107 '<|reserved_special_token_98|>' is not marked as EOG
llm_load_vocab: control token: 128104 '<|reserved_special_token_95|>' is not marked as EOG
llm_load_vocab: control token: 128103 '<|reserved_special_token_94|>' is not marked as EOG
llm_load_vocab: control token: 128102 '<|reserved_special_token_93|>' is not marked as EOG
llm_load_vocab: control token: 128098 '<|reserved_special_token_89|>' is not marked as EOG
llm_load_vocab: control token: 128092 '<|reserved_special_token_83|>' is not marked as EOG
llm_load_vocab: control token: 128091 '<|reserved_special_token_82|>' is not marked as EOG
llm_load_vocab: control token: 128090 '<|reserved_special_token_81|>' is not marked as EOG
llm_load_vocab: control token: 128088 '<|reserved_special_token_79|>' is not marked as EOG
llm_load_vocab: control token: 128086 '<|reserved_special_token_77|>' is not marked as EOG
llm_load_vocab: control token: 128082 '<|reserved_special_token_73|>' is not marked as EOG
llm_load_vocab: control token: 128079 '<|reserved_special_token_70|>' is not marked as EOG
llm_load_vocab: control token: 128077 '<|reserved_special_token_68|>' is not marked as EOG
llm_load_vocab: control token: 128076 '<|reserved_special_token_67|>' is not marked as EOG
llm_load_vocab: control token: 128074 '<|reserved_special_token_65|>' is not marked as EOG
llm_load_vocab: control token: 128069 '<|reserved_special_token_60|>' is not marked as EOG
llm_load_vocab: control token: 128068 '<|reserved_special_token_59|>' is not marked as EOG
llm_load_vocab: control token: 128066 '<|reserved_special_token_57|>' is not marked as EOG
llm_load_vocab: control token: 128064 '<|reserved_special_token_55|>' is not marked as EOG
llm_load_vocab: control token: 128063 '<|reserved_special_token_54|>' is not marked as EOG
llm_load_vocab: control token: 128061 '<|reserved_special_token_52|>' is not marked as EOG
llm_load_vocab: control token: 128060 '<|reserved_special_token_51|>' is not marked as EOG
llm_load_vocab: control token: 128058 '<|reserved_special_token_49|>' is not marked as EOG
llm_load_vocab: control token: 128055 '<|reserved_special_token_46|>' is not marked as EOG
llm_load_vocab: control token: 128047 '<|reserved_special_token_38|>' is not marked as EOG
llm_load_vocab: control token: 128046 '<|reserved_special_token_37|>' is not marked as EOG
llm_load_vocab: control token: 128045 '<|reserved_special_token_36|>' is not marked as EOG
llm_load_vocab: control token: 128044 '<|reserved_special_token_35|>' is not marked as EOG
llm_load_vocab: control token: 128039 '<|reserved_special_token_30|>' is not marked as EOG
llm_load_vocab: control token: 128037 '<|reserved_special_token_28|>' is not marked as EOG
llm_load_vocab: control token: 128036 '<|reserved_special_token_27|>' is not marked as EOG
llm_load_vocab: control token: 128033 '<|reserved_special_token_24|>' is not marked as EOG
llm_load_vocab: control token: 128029 '<|reserved_special_token_20|>' is not marked as EOG
llm_load_vocab: control token: 128028 '<|reserved_special_token_19|>' is not marked as EOG
llm_load_vocab: control token: 128025 '<|reserved_special_token_16|>' is not marked as EOG
llm_load_vocab: control token: 128024 '<|reserved_special_token_15|>' is not marked as EOG
llm_load_vocab: control token: 128023 '<|reserved_special_token_14|>' is not marked as EOG
llm_load_vocab: control token: 128022 '<|reserved_special_token_13|>' is not marked as EOG
llm_load_vocab: control token: 128019 '<|reserved_special_token_10|>' is not marked as EOG
llm_load_vocab: control token: 128017 '<|reserved_special_token_8|>' is not marked as EOG
llm_load_vocab: control token: 128016 '<|reserved_special_token_7|>' is not marked as EOG
llm_load_vocab: control token: 128014 '<|reserved_special_token_5|>' is not marked as EOG
llm_load_vocab: control token: 128012 '<|reserved_special_token_3|>' is not marked as EOG
llm_load_vocab: control token: 128011 '<|reserved_special_token_2|>' is not marked as EOG
llm_load_vocab: control token: 128004 '<|finetune_right_pad_id|>' is not marked as EOG
llm_load_vocab: control token: 128002 '<|reserved_special_token_0|>' is not marked as EOG
llm_load_vocab: control token: 128253 '<|reserved_special_token_244|>' is not marked as EOG
llm_load_vocab: control token: 128191 '<|reserved_special_token_182|>' is not marked as EOG
llm_load_vocab: control token: 128184 '<|reserved_special_token_175|>' is not marked as EOG
llm_load_vocab: control token: 128138 '<|reserved_special_token_129|>' is not marked as EOG
llm_load_vocab: control token: 128183 '<|reserved_special_token_174|>' is not marked as EOG
llm_load_vocab: control token: 128041 '<|reserved_special_token_32|>' is not marked as EOG
llm_load_vocab: control token: 128049 '<|reserved_special_token_40|>' is not marked as EOG
llm_load_vocab: control token: 128093 '<|reserved_special_token_84|>' is not marked as EOG
llm_load_vocab: control token: 128216 '<|reserved_special_token_207|>' is not marked as EOG
llm_load_vocab: control token: 128108 '<|reserved_special_token_99|>' is not marked as EOG
llm_load_vocab: control token: 128209 '<|reserved_special_token_200|>' is not marked as EOG
llm_load_vocab: control token: 128146 '<|reserved_special_token_137|>' is not marked as EOG
llm_load_vocab: control token: 128032 '<|reserved_special_token_23|>' is not marked as EOG
llm_load_vocab: control token: 128130 '<|reserved_special_token_121|>' is not marked as EOG
llm_load_vocab: control token: 128202 '<|reserved_special_token_193|>' is not marked as EOG
llm_load_vocab: control token: 128075 '<|reserved_special_token_66|>' is not marked as EOG
llm_load_vocab: control token: 128096 '<|reserved_special_token_87|>' is not marked as EOG
llm_load_vocab: control token: 128187 '<|reserved_special_token_178|>' is not marked as EOG
llm_load_vocab: control token: 128144 '<|reserved_special_token_135|>' is not marked as EOG
llm_load_vocab: control token: 128230 '<|reserved_special_token_221|>' is not marked as EOG
llm_load_vocab: control token: 128007 '<|end_header_id|>' is not marked as EOG
llm_load_vocab: control token: 128056 '<|reserved_special_token_47|>' is not marked as EOG
llm_load_vocab: control token: 128057 '<|reserved_special_token_48|>' is not marked as EOG
llm_load_vocab: control token: 128062 '<|reserved_special_token_53|>' is not marked as EOG
llm_load_vocab: control token: 128154 '<|reserved_special_token_145|>' is not marked as EOG
llm_load_vocab: control token: 128153 '<|reserved_special_token_144|>' is not marked as EOG
llm_load_vocab: control token: 128213 '<|reserved_special_token_204|>' is not marked as EOG
llm_load_vocab: control token: 128173 '<|reserved_special_token_164|>' is not marked as EOG
llm_load_vocab: control token: 128161 '<|reserved_special_token_152|>' is not marked as EOG
llm_load_vocab: control token: 128042 '<|reserved_special_token_33|>' is not marked as EOG
llm_load_vocab: control token: 128182 '<|reserved_special_token_173|>' is not marked as EOG
llm_load_vocab: control token: 128095 '<|reserved_special_token_86|>' is not marked as EOG
llm_load_vocab: control token: 128119 '<|reserved_special_token_110|>' is not marked as EOG
llm_load_vocab: control token: 128237 '<|reserved_special_token_228|>' is not marked as EOG
llm_load_vocab: control token: 128149 '<|reserved_special_token_140|>' is not marked as EOG
llm_load_vocab: control token: 128043 '<|reserved_special_token_34|>' is not marked as EOG
llm_load_vocab: control token: 128140 '<|reserved_special_token_131|>' is not marked as EOG
llm_load_vocab: control token: 128174 '<|reserved_special_token_165|>' is not marked as EOG
llm_load_vocab: control token: 128240 '<|reserved_special_token_231|>' is not marked as EOG
llm_load_vocab: control token: 128158 '<|reserved_special_token_149|>' is not marked as EOG
llm_load_vocab: control token: 128053 '<|reserved_special_token_44|>' is not marked as EOG
llm_load_vocab: control token: 128027 '<|reserved_special_token_18|>' is not marked as EOG
llm_load_vocab: control token: 128003 '<|reserved_special_token_1|>' is not marked as EOG
llm_load_vocab: control token: 128020 '<|reserved_special_token_11|>' is not marked as EOG
llm_load_vocab: control token: 128117 '<|reserved_special_token_108|>' is not marked as EOG
llm_load_vocab: control token: 128162 '<|reserved_special_token_153|>' is not marked as EOG
llm_load_vocab: control token: 128227 '<|reserved_special_token_218|>' is not marked as EOG
llm_load_vocab: control token: 128160 '<|reserved_special_token_151|>' is not marked as EOG
llm_load_vocab: control token: 128013 '<|reserved_special_token_4|>' is not marked as EOG
llm_load_vocab: control token: 128089 '<|reserved_special_token_80|>' is not marked as EOG
llm_load_vocab: control token: 128164 '<|reserved_special_token_155|>' is not marked as EOG
llm_load_vocab: control token: 128001 '<|end_of_text|>' is not marked as EOG
llm_load_vocab: control token: 128114 '<|reserved_special_token_105|>' is not marked as EOG
llm_load_vocab: control token: 128251 '<|reserved_special_token_242|>' is not marked as EOG
llm_load_vocab: control token: 128126 '<|reserved_special_token_117|>' is not marked as EOG
llm_load_vocab: control token: 128054 '<|reserved_special_token_45|>' is not marked as EOG
llm_load_vocab: control token: 128225 '<|reserved_special_token_216|>' is not marked as EOG
llm_load_vocab: control token: 128248 '<|reserved_special_token_239|>' is not marked as EOG
llm_load_vocab: control token: 128252 '<|reserved_special_token_243|>' is not marked as EOG
llm_load_vocab: control token: 128217 '<|reserved_special_token_208|>' is not marked as EOG
llm_load_vocab: control token: 128006 '<|start_header_id|>' is not marked as EOG
llm_load_vocab: control token: 128212 '<|reserved_special_token_203|>' is not marked as EOG
llm_load_vocab: control token: 128078 '<|reserved_special_token_69|>' is not marked as EOG
llm_load_vocab: control token: 128238 '<|reserved_special_token_229|>' is not marked as EOG
llm_load_vocab: control token: 128087 '<|reserved_special_token_78|>' is not marked as EOG
llm_load_vocab: control token: 128228 '<|reserved_special_token_219|>' is not marked as EOG
llm_load_vocab: control token: 128059 '<|reserved_special_token_50|>' is not marked as EOG
llm_load_vocab: control token: 128101 '<|reserved_special_token_92|>' is not marked as EOG
llm_load_vocab: control token: 128210 '<|reserved_special_token_201|>' is not marked as EOG
llm_load_vocab: control token: 128085 '<|reserved_special_token_76|>' is not marked as EOG
llm_load_vocab: control token: 128072 '<|reserved_special_token_63|>' is not marked as EOG
llm_load_vocab: control token: 128071 '<|reserved_special_token_62|>' is not marked as EOG
llm_load_vocab: control token: 128050 '<|reserved_special_token_41|>' is not marked as EOG
llm_load_vocab: control token: 128198 '<|reserved_special_token_189|>' is not marked as EOG
llm_load_vocab: control token: 128073 '<|reserved_special_token_64|>' is not marked as EOG
llm_load_vocab: control token: 128000 '<|begin_of_text|>' is not marked as EOG
llm_load_vocab: control token: 128224 '<|reserved_special_token_215|>' is not marked as EOG
llm_load_vocab: control token: 128218 '<|reserved_special_token_209|>' is not marked as EOG
llm_load_vocab: control token: 128112 '<|reserved_special_token_103|>' is not marked as EOG
llm_load_vocab: control token: 128204 '<|reserved_special_token_195|>' is not marked as EOG
llm_load_vocab: control token: 128052 '<|reserved_special_token_43|>' is not marked as EOG
llm_load_vocab: control token: 128031 '<|reserved_special_token_22|>' is not marked as EOG
llm_load_vocab: control token: 128118 '<|reserved_special_token_109|>' is not marked as EOG
llm_load_vocab: control token: 128010 '<|python_tag|>' is not marked as EOG
llm_load_vocab: control token: 128239 '<|reserved_special_token_230|>' is not marked as EOG
llm_load_vocab: control token: 128203 '<|reserved_special_token_194|>' is not marked as EOG
llm_load_vocab: control token: 128133 '<|reserved_special_token_124|>' is not marked as EOG
llm_load_vocab: control token: 128249 '<|reserved_special_token_240|>' is not marked as EOG
llm_load_vocab: control token: 128168 '<|reserved_special_token_159|>' is not marked as EOG
llm_load_vocab: control token: 128128 '<|reserved_special_token_119|>' is not marked as EOG
llm_load_vocab: control token: 128106 '<|reserved_special_token_97|>' is not marked as EOG
llm_load_vocab: control token: 128040 '<|reserved_special_token_31|>' is not marked as EOG
llm_load_vocab: control token: 128233 '<|reserved_special_token_224|>' is not marked as EOG
llm_load_vocab: control token: 128167 '<|reserved_special_token_158|>' is not marked as EOG
llm_load_vocab: control token: 128131 '<|reserved_special_token_122|>' is not marked as EOG
llm_load_vocab: control token: 128115 '<|reserved_special_token_106|>' is not marked as EOG
llm_load_vocab: control token: 128235 '<|reserved_special_token_226|>' is not marked as EOG
llm_load_vocab: control token: 128192 '<|reserved_special_token_183|>' is not marked as EOG
llm_load_vocab: control token: 128065 '<|reserved_special_token_56|>' is not marked as EOG
llm_load_vocab: control token: 128141 '<|reserved_special_token_132|>' is not marked as EOG
llm_load_vocab: control token: 128097 '<|reserved_special_token_88|>' is not marked as EOG
llm_load_vocab: control token: 128099 '<|reserved_special_token_90|>' is not marked as EOG
llm_load_vocab: control token: 128193 '<|reserved_special_token_184|>' is not marked as EOG
llm_load_vocab: control token: 128094 '<|reserved_special_token_85|>' is not marked as EOG
llm_load_vocab: control token: 128151 '<|reserved_special_token_142|>' is not marked as EOG
llm_load_vocab: control token: 128223 '<|reserved_special_token_214|>' is not marked as EOG
llm_load_vocab: control token: 128234 '<|reserved_special_token_225|>' is not marked as EOG
llm_load_vocab: control token: 128221 '<|reserved_special_token_212|>' is not marked as EOG
llm_load_vocab: control token: 128035 '<|reserved_special_token_26|>' is not marked as EOG
llm_load_vocab: control token: 128034 '<|reserved_special_token_25|>' is not marked as EOG
llm_load_vocab: control token: 128254 '<|reserved_special_token_245|>' is not marked as EOG
llm_load_vocab: control token: 128196 '<|reserved_special_token_187|>' is not marked as EOG
llm_load_vocab: control token: 128100 '<|reserved_special_token_91|>' is not marked as EOG
llm_load_vocab: control token: 128190 '<|reserved_special_token_181|>' is not marked as EOG
llm_load_vocab: control token: 128211 '<|reserved_special_token_202|>' is not marked as EOG
llm_load_vocab: control token: 128175 '<|reserved_special_token_166|>' is not marked as EOG
llm_load_vocab: control token: 128084 '<|reserved_special_token_75|>' is not marked as EOG
llm_load_vocab: control token: 128081 '<|reserved_special_token_72|>' is not marked as EOG
llm_load_vocab: control token: 128105 '<|reserved_special_token_96|>' is not marked as EOG
llm_load_vocab: control token: 128083 '<|reserved_special_token_74|>' is not marked as EOG
llm_load_vocab: control token: 128220 '<|reserved_special_token_211|>' is not marked as EOG
llm_load_vocab: control token: 128018 '<|reserved_special_token_9|>' is not marked as EOG
llm_load_vocab: control token: 128005 '<|step_id|>' is not marked as EOG
llm_load_vocab: control token: 128051 '<|reserved_special_token_42|>' is not marked as EOG
llm_load_vocab: control token: 128206 '<|reserved_special_token_197|>' is not marked as EOG
llm_load_vocab: control token: 128048 '<|reserved_special_token_39|>' is not marked as EOG
llm_load_vocab: control token: 128165 '<|reserved_special_token_156|>' is not marked as EOG
llm_load_vocab: control token: 128021 '<|reserved_special_token_12|>' is not marked as EOG
llm_load_vocab: control token: 128070 '<|reserved_special_token_61|>' is not marked as EOG
llm_load_vocab: control token: 128246 '<|reserved_special_token_237|>' is not marked as EOG
llm_load_vocab: control token: 128122 '<|reserved_special_token_113|>' is not marked as EOG
llm_load_vocab: control token: 128080 '<|reserved_special_token_71|>' is not marked as EOG
llm_load_vocab: control token: 128038 '<|reserved_special_token_29|>' is not marked as EOG
llm_load_vocab: control token: 128245 '<|reserved_special_token_236|>' is not marked as EOG
llm_load_vocab: control token: 128030 '<|reserved_special_token_21|>' is not marked as EOG
llm_load_vocab: control token: 128222 '<|reserved_special_token_213|>' is not marked as EOG
llm_load_vocab: control token: 128067 '<|reserved_special_token_58|>' is not marked as EOG
llm_load_vocab: control token: 128121 '<|reserved_special_token_112|>' is not marked as EOG
llm_load_vocab: control token: 128015 '<|reserved_special_token_6|>' is not marked as EOG
llm_load_vocab: control token: 128026 '<|reserved_special_token_17|>' is not marked as EOG
llm_load_vocab: control token: 128127 '<|reserved_special_token_118|>' is not marked as EOG
llm_load_vocab: special tokens cache size = 257
llm_load_vocab: token to piece cache size = 0.7999 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = mllama
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 128256
llm_load_print_meta: n_merges         = 280147
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 131072
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_layer          = 40
llm_load_print_meta: n_head           = 32
llm_load_print_meta: n_head_kv        = 8
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 4
llm_load_print_meta: n_embd_k_gqa     = 1024
llm_load_print_meta: n_embd_v_gqa     = 1024
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-05
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 14336
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 500000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 131072
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: ssm_dt_b_c_rms   = 0
llm_load_print_meta: model type       = 11B
llm_load_print_meta: model ftype      = Q4_K - Medium
llm_load_print_meta: model params     = 9.78 B
llm_load_print_meta: model size       = 5.55 GiB (4.87 BPW)
llm_load_print_meta: general.name     = Model
llm_load_print_meta: BOS token        = 128000 '<|begin_of_text|>'
llm_load_print_meta: EOS token        = 128009 '<|eot_id|>'
llm_load_print_meta: EOT token        = 128009 '<|eot_id|>'
llm_load_print_meta: EOM token        = 128008 '<|eom_id|>'
llm_load_print_meta: PAD token        = 128004 '<|finetune_right_pad_id|>'
llm_load_print_meta: LF token         = 128 'Ä'
llm_load_print_meta: EOG token        = 128008 '<|eom_id|>'
llm_load_print_meta: EOG token        = 128009 '<|eot_id|>'
llm_load_print_meta: max token length = 256
```




arch: 50, layer: 34, cross_attention_layers: 0, n_embed_head_k: 128, n_head_kv: 8

arch: 50, layer: 38, cross_attention_layers: 1, n_embed_head_k: 128, n_head_kv: 8


     20: STRING     |        1 | tokenizer.ggml.model = 'gpt2'
     21: STRING     |        1 | tokenizer.ggml.pre = 'smaug-bpe'
     22: [STRING]   |   128257 | tokenizer.ggml.tokens
     23: [INT32]    |   128257 | tokenizer.ggml.token_type
     24: [STRING]   |   280147 | tokenizer.ggml.merges
     25: UINT32     |        1 | tokenizer.ggml.bos_token_id = 128000
     26: UINT32     |        1 | tokenizer.ggml.eos_token_id = 128009
     27: UINT32     |        1 | tokenizer.ggml.padding_token_id = 128004
     28: STRING     |        1 | tokenizer.chat_template = '{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- s'
     29: UINT32     |        1 | general.quantization_version = 2

word: ĠÐ·Ð°ÐºÐ¾Ð½Ð¾Ð´Ð°Ð² ÑģÑĤÐ²Ð°
word: ãĢĢ ãĤ¤
word: Ġëħ¸ íķĺìļ°
word: ĠD Ã¼ÅŁ
word: ĠÐ³ ÑĥÑģÑĤ
word: ĠÐĴ Ð°ÑĪ
word: ĠØ§Ùħ ØªÛĮ
word: Ġpar amet
word: Ġparam et
word: Ġpara met
word: ĠÎłÎ±Î½ ÎµÏĢ
word: à¹Į à¸ģà¸£
word: à¹Įà¸ģ à¸£
word: Î¶ Î±
word: ĠëįĶ ìļ±
word: ÙĪ ÙĦØ§Øª
word: ÙĪÙĦ Ø§Øª
word: ÙĪÙĦØ§ Øª
word: Ð² Ð°ÑĤÐ¸ÑģÑı
word: Ð²Ð° ÑĤÐ¸ÑģÑı
word: Ð²Ð°ÑĤÐ¸ ÑģÑı
word: Ġk Ã¶k
word: ĠkÃ¶ k
word: ÙĨ Ø¨
word: ĠÐ²ÑĭÑģÐ¾Ðº Ð¾Ð¹
word: ãĥ¼ ãĥ¼
word: ãĥ¼ãĥ ¼
word: éĶ ¦


word: ĠÐ·Ð°ÐºÐ¾Ð½Ð¾Ð´Ð°Ð² ÑģÑĤÐ²Ð°
word: ãĢĢ ãĤ¤
word: Ġëħ¸ íķĺìļ°
word: ĠD Ã¼ÅŁ
word: ĠÐ³ ÑĥÑģÑĤ
word: ĠÐĴ Ð°ÑĪ
word: ĠØ§Ùħ ØªÛĮ
word: Ġpar amet
word: Ġparam et
word: Ġpara met
word: ĠÎłÎ±Î½ ÎµÏĢ
word: à¹Į à¸ģà¸£
word: à¹Įà¸ģ à¸£
word: Î¶ Î±
word: ĠëįĶ ìļ±
word: ÙĪ ÙĦØ§Øª
word: ÙĪÙĦ Ø§Øª
word: ÙĪÙĦØ§ Øª
word: Ð² Ð°ÑĤÐ¸ÑģÑı
word: Ð²Ð° ÑĤÐ¸ÑģÑı
word: Ð²Ð°ÑĤÐ¸ ÑģÑı
word: Ġk Ã¶k
word: ĠkÃ¶ k
word: ÙĨ Ø¨
word: ĠÐ²ÑĭÑģÐ¾Ðº Ð¾Ð¹
word: ãĥ¼ ãĥ¼
word: ãĥ¼ãĥ ¼
word: éĶ ¦


Rope settings ollama:
rope parameters: n_rot: 128, rope_type: 0, n_ctx_orig: 131072, freq_base: 100, freq_scale: 500000.000000, ext_factor: 1.000000, attn_factor: 0.000000, beta_fast: 1.000000, beta_slow: 32.000000

Rope settings llama.cpp:
rope parameters: n_rot: 128, rope_type: 0, n_ctx_orig: 131072, freq_base: 100, freq_scale: 500000.000000, ext_factor: 1.000000, attn_factor: 0.000000, beta_fast: 1.000000, beta_slow: 32.000000

Ollama:
kq_scale: 0.088388

llama.cpp:
kq_scale: 0.088388



llama.cpp input embeddings for 128006:
```console
input_embeddings tensor type: f32
input_embeddings backend type: CPU
input_embeddings[0] = 0.250884
input_embeddings[1] = -1.903877
input_embeddings[2] = 1.126612
input_embeddings[3] = 0.874009
input_embeddings[4] = -0.151682
input_embeddings[5] = 1.005559
input_embeddings[6] = 2.459111
input_embeddings[7] = -0.477424
input_embeddings[8] = 0.324140
input_embeddings[9] = -1.908321
input_embeddings[10] = 1.061303
input_embeddings[11] = -0.503033
input_embeddings[12] = 1.330023
input_embeddings[13] = -0.804025
input_embeddings[14] = -0.962649
input_embeddings[15] = -0.704584
input_embeddings[16] = 1.547000



(venv) $ ./read-embd-token.py models/llama-3-2-11b.gguf

Tensor Information:
Name: token_embd.weight
Shape: 4096 x 128264
Type: F32
Total elements: 525369344

Embedding values for token 128006:
inp_tokens[0] = 0.0108642578125
inp_tokens[1] = -0.0137939453125
inp_tokens[2] = 0.000736236572265625
inp_tokens[3] = -4.880365614062863e-23
inp_tokens[4] = -0.0147705078125
inp_tokens[5] = 3.5982356646056704e-23
inp_tokens[6] = -0.0032501220703125
inp_tokens[7] = 0.006988525390625
inp_tokens[8] = -0.0135498046875
inp_tokens[9] = -0.00482177734375
inp_tokens[10] = 0.0032806396484375
inp_tokens[11] = -0.003814697265625
inp_tokens[12] = 0.00087738037109375
inp_tokens[13] = -0.00830078125
inp_tokens[14] = 0.0034027099609375
inp_tokens[15] = 0.00701904296875
inp_tokens[16] = 0.02099609375
```

```console
(venv) $ python print-safe-embeddings.py

Tensor Information:
Name: language_model.model.embed_tokens.weight
Shape: torch.Size([128264, 4096])
Type: torch.bfloat16

Embedding values for token 128006:
inp_tokens[0] = -0.00018405914306640625
inp_tokens[1] = -0.000240325927734375
inp_tokens[2] = 0.000164031982421875
inp_tokens[3] = -0.000537872314453125
inp_tokens[4] = 0.0002651214599609375
inp_tokens[5] = -1.2814998626708984e-05
inp_tokens[6] = -0.0002899169921875
inp_tokens[7] = 0.00106048583984375
inp_tokens[8] = 3.6716461181640625e-05
inp_tokens[9] = 0.000530242919921875
inp_tokens[10] = -0.00020599365234375
inp_tokens[11] = -0.0003948211669921875
inp_tokens[12] = 0.000965118408203125
inp_tokens[13] = -0.000164031982421875
inp_tokens[14] = -0.0005645751953125
inp_tokens[15] = 0.000518798828125
inp_tokens[16] = -6.341934204101562e-05
```

ollama:
(venv) $ ./read-embd-token-q.py /home/danbev/.ollama/models/blobs/sha256-652e85aa1e14c9087a4ccc3ab516fb794cbcf152f8b4b8d3c0b828da4ada62d9

Tensor Information:
Name: token_embd.weight
Shape: 4096 x 128264
Type: Q4_K
Total elements: 525369344

Raw quantized values:
raw[0] = [111   3  37 ... 159  73 109]
raw[1] = [125   3 179 ... 127  86  89]
raw[2] = [ 52   3 254 ... 168 232 191]
raw[3] = [173   2 223 ... 166 107 138]
raw[4] = [ 41   3 152 ... 239 175 159]
raw[5] = [249   2  83 ... 168 174 122]
raw[6] = [116   2 227 ... 122 199 137]
raw[7] = [191   2  45 ... 167 156   4]
raw[8] = [114   3 146 ... 254 141 216]
raw[9] = [ 46   3 158 ... 171 236 169]
raw[10] = [240   2  92 ... 185 156 149]
raw[11] = [168   5 125 ... 191 117 116]
raw[12] = [152   2  51 ... 173 203 116]
raw[13] = [ 59   6 110 ... 175  90 103]
raw[14] = [ 65   3 126 ... 146 120 104]
raw[15] = [202   4 165 ... 216 117  97]
raw[16] = [ 13   6  51 ... 250 199 138]

Quantized data size: 128264
Bytes per value: 1


(gdb) p model.vocab.token_to_id
$14 = std::unordered_map with 128257 elements = {["<|image|>"] = 128256,
  ["<|reserved_special_token_246|>"] = 128255, ["<|reserved_special_token_241|>"] = 128250,
  ["<|reserved_special_token_238|>"] = 128247, ["<|reserved_special_token_235|>"] = 128244,
  ["<|reserved_special_token_234|>"] = 128243, ["<|reserved_special_token_233|>"] = 128242,
  ["<|reserved_special_token_232|>"] = 128241, ["<|reserved_special_token_227|>"] = 128236,
  ["<|reserved_special_token_223|>"] = 128232, ["<|reserved_special_token_222|>"] = 128231,
  ["<|reserved_special_token_220|>"] = 128229, ["<|reserved_special_token_217|>"] = 128226,
  ["<|reserved_special_token_210|>"] = 128219, ["<|reserved_special_token_206|>"] = 128215,
  ["<|reserved_special_token_205|>"] = 128214, ["<|reserved_special_token_199|>"] = 128208,
  ["<|reserved_special_token_198|>"] = 128207, ["<|reserved_special_token_196|>"] = 128205,
  ["<|reserved_special_token_192|>"] = 128201, ["<|reserved_special_token_191|>"] = 128200,
  ["<|reserved_special_token_190|>"] = 128199, ["<|reserved_special_token_188|>"] = 128197,
  ["<|reserved_special_token_186|>"] = 128195, ["<|reserved_special_token_185|>"] = 128194,
  ["<|reserved_special_token_180|>"] = 128189, ["<|reserved_special_token_179|>"] = 128188,
  ["<|reserved_special_token_177|>"] = 128186, ["<|reserved_special_token_176|>"] = 128185,
  ["<|reserved_special_token_172|>"] = 128181, ["<|reserved_special_token_171|>"] = 128180,
  ["<|reserved_special_token_170|>"] = 128179, ["<|reserved_special_token_169|>"] = 128178,
  ["<|reserved_special_token_168|>"] = 128177, ["<|reserved_special_token_167|>"] = 128176,
  ["<|reserved_special_token_163|>"] = 128172, ["<|reserved_special_token_162|>"] = 128171,
  ["<|reserved_special_token_161|>"] = 128170, ["<|reserved_special_token_160|>"] = 128169,
  ["<|reserved_special_token_157|>"] = 128166, ["<|reserved_special_token_154|>"] = 128163,
  ["<|reserved_special_token_150|>"] = 128159, ["<|reserved_special_token_148|>"] = 128157,
  ["<|reserved_special_token_147|>"] = 128156, ["<|reserved_special_token_146|>"] = 128155,
  ["<|reserved_special_token_143|>"] = 128152, ["<|reserved_special_token_141|>"] = 128150,
  ["<|reserved_special_token_139|>"] = 128148, ["<|reserved_special_token_138|>"] = 128147,
  ["<|reserved_special_token_136|>"] = 128145, ["<|reserved_special_token_134|>"] = 128143,
  ["<|reserved_special_token_133|>"] = 128142, ["<|reserved_special_token_130|>"] = 128139,
  ["<|reserved_special_token_128|>"] = 128137, ["<|reserved_special_token_127|>"] = 128136,
  ["<|reserved_special_token_126|>"] = 128135, ["<|reserved_special_token_125|>"] = 128134,
  ["<|reserved_special_token_123|>"] = 128132, ["<|reserved_special_token_120|>"] = 128129,
  ["<|reserved_special_token_116|>"] = 128125, ["<|reserved_special_token_115|>"] = 128124,
  ["<|reserved_special_token_114|>"] = 128123, ["<|reserved_special_token_111|>"] = 128120,
  ["<|reserved_special_token_107|>"] = 128116, ["<|reserved_special_token_104|>"] = 128113,
  ["<|reserved_special_token_102|>"] = 128111, ["<|reserved_special_token_101|>"] = 128110,
  ["<|reserved_special_token_100|>"] = 128109, ["<|reserved_special_token_98|>"] = 128107,
  ["<|reserved_special_token_95|>"] = 128104, ["<|reserved_special_token_94|>"] = 128103,
  ["<|reserved_special_token_93|>"] = 128102, ["<|reserved_special_token_89|>"] = 128098,
  ["<|reserved_special_token_83|>"] = 128092, ["<|reserved_special_token_82|>"] = 128091,
  ["<|reserved_special_token_81|>"] = 128090, ["<|reserved_special_token_79|>"] = 128088,
  ["<|reserved_special_token_77|>"] = 128086, ["<|reserved_special_token_73|>"] = 128082,
  ["<|reserved_special_token_70|>"] = 128079, ["<|reserved_special_token_68|>"] = 128077,
  ["<|reserved_special_token_67|>"] = 128076, ["<|reserved_special_token_65|>"] = 128074,
  ["<|reserved_special_token_60|>"] = 128069, ["<|reserved_special_token_59|>"] = 128068,
  ["<|reserved_special_token_57|>"] = 128066, ["<|reserved_special_token_55|>"] = 128064,
  ["<|reserved_special_token_54|>"] = 128063, ["<|reserved_special_token_52|>"] = 128061,
  ["<|reserved_special_token_51|>"] = 128060, ["<|reserved_special_token_49|>"] = 128058,
  ["<|reserved_special_token_46|>"] = 128055, ["<|reserved_special_token_38|>"] = 128047,
  ["<|reserved_special_token_37|>"] = 128046, ["<|reserved_special_token_36|>"] = 128045,
  ["<|reserved_special_token_35|>"] = 128044, ["<|reserved_special_token_30|>"] = 128039,
  ["<|reserved_special_token_28|>"] = 128037, ["<|reserved_special_token_27|>"] = 128036,
  ["<|reserved_special_token_24|>"] = 128033, ["<|reserved_special_token_20|>"] = 128029,
  ["<|reserved_special_token_19|>"] = 128028, ["<|reserved_special_token_16|>"] = 128025,
  ["<|reserved_special_token_15|>"] = 128024, ["<|reserved_special_token_14|>"] = 128023,
  ["<|reserved_special_token_13|>"] = 128022, ["<|reserved_special_token_10|>"] = 128019,
  ["<|reserved_special_token_8|>"] = 128017, ["<|reserved_special_token_7|>"] = 128016,
  ["<|reserved_special_token_5|>"] = 128014, ["<|reserved_special_token_3|>"] = 128012,
  ["<|reserved_special_token_2|>"] = 128011, ["<|eom_id|>"] = 128008, ["<|finetune_right_pad_id|>"] = 128004,
  ["<|reserved_special_token_0|>"] = 128002, ["ÙĨØ¨"] = 127996, ["Ð²Ð°ÑĤÐ¸ÑģÑı"] = 127994,
  ["ĠëįĶìļ±"] = 127992, ["Î¶Î±"] = 127991, ["ĠØ§ÙħØªÛĮ"] = 127987, ["ĠDÃ¼ÅŁ"] = 127984,
  ["ĠÐ·Ð°ÐºÐ¾Ð½Ð¾Ð´Ð°Ð²ÑģÑĤÐ²Ð°"] = 127981,
  ["à¸Ľà¸ģà¸Ħà¸£à¸Ńà¸ĩ"] = 127980, ["à¸´à¸§à¹Ģà¸ķà¸Ńà¸£"] = 127979,
  ["Ð»ÐµÐ½Ð½Ñĸ"] = 127978...}



(venv) $ ./read-embd-token.py models/llama-3-2-11b-f32.gguf

Tensor Information:
Name: token_embd.weight
Shape: 4096 x 128264
Type: F32
Total elements: 525369344

Embedding values for token 128006:
inp_tokens[0] = 0.0108642578125
inp_tokens[1] = -0.0137939453125
inp_tokens[2] = 0.000736236572265625
inp_tokens[3] = -4.880365614062863e-23
inp_tokens[4] = -0.0147705078125
inp_tokens[5] = 3.5982356646056704e-23
inp_tokens[6] = -0.0032501220703125
inp_tokens[7] = 0.006988525390625
inp_tokens[8] = -0.0135498046875
inp_tokens[9] = -0.00482177734375
inp_tokens[10] = 0.0032806396484375
inp_tokens[11] = -0.003814697265625
inp_tokens[12] = 0.00087738037109375
inp_tokens[13] = -0.00830078125
inp_tokens[14] = 0.0034027099609375
inp_tokens[15] = 0.00701904296875
inp_tokens[16] = 0.02099609375

(venv) $ python print-safe-embeddings.py

Tensor Information:
Name: language_model.model.embed_tokens.weight
Shape: torch.Size([128264, 4096])
Type: torch.bfloat16

Embedding values for token 128006:
inp_tokens[0] = -0.00018405914306640625
inp_tokens[1] = -0.000240325927734375
inp_tokens[2] = 0.000164031982421875
inp_tokens[3] = -0.000537872314453125
inp_tokens[4] = 0.0002651214599609375
inp_tokens[5] = -1.2814998626708984e-05
inp_tokens[6] = -0.0002899169921875
inp_tokens[7] = 0.00106048583984375
inp_tokens[8] = 3.6716461181640625e-05
inp_tokens[9] = 0.000530242919921875
inp_tokens[10] = -0.00020599365234375
inp_tokens[11] = -0.0003948211669921875
inp_tokens[12] = 0.000965118408203125
inp_tokens[13] = -0.000164031982421875
inp_tokens[14] = -0.0005645751953125
inp_tokens[15] = 0.000518798828125
inp_tokens[16] = -6.341934204101562e-05

token = 128006
token = 882
token = 128007
token = 271
token = 3923
token = 374
token = 279
token = 469
token = 3168
token = 301
token = 22703
token = 30
token = 128009
token = 128006
token = 78191
token = 128007
token = 271
kq_scale: 0.088388
[New Thread 0x7fffabe00000 (LWP 84068)]
[New Thread 0x7fffab400000 (LWP 84069)]
[New Thread 0x7fffaaa00000 (LWP 84070)]
input_embeddings tensor type: f32
input_embeddings backend type: CPU
input_embeddings[0] = -0.309861
input_embeddings[1] = -0.889324
input_embeddings[2] = 1.652823
input_embeddings[3] = 2.878123
input_embeddings[4] = 2.721913
input_embeddings[5] = 1.118232
input_embeddings[6] = -4.750814
input_embeddings[7] = 1.661370
input_embeddings[8] = 1.025107
input_embeddings[9] = -0.051233
input_embeddings[10] = 7.026322
input_embeddings[11] = 2.739259
input_embeddings[12] = 1.209301
input_embeddings[13] = -1.540354
input_embeddings[14] = -2.568749
input_embeddings[15] = -2.265143
input_embeddings[16] = 2.384460


(venv) $ ./read-embd-token-q.py /home/danbev/.ollama/models/blobs/sha256-652e85aa1e14c9087a4ccc3ab516fb794cbcf152f8b4b8d3c0b828da4ada62d9

Tensor Information:
Name: token_embd.weight
Shape: 4096 x 128264
Type: Q4_K
Total elements: 525369344

Raw quantized values:
raw[0] = [111   3  37 ... 159  73 109]
raw[1] = [125   3 179 ... 127  86  89]
raw[2] = [ 52   3 254 ... 168 232 191]
raw[3] = [173   2 223 ... 166 107 138]
raw[4] = [ 41   3 152 ... 239 175 159]
raw[5] = [249   2  83 ... 168 174 122]
raw[6] = [116   2 227 ... 122 199 137]
raw[7] = [191   2  45 ... 167 156   4]
raw[8] = [114   3 146 ... 254 141 216]
raw[9] = [ 46   3 158 ... 171 236 169]
raw[10] = [240   2  92 ... 185 156 149]
raw[11] = [168   5 125 ... 191 117 116]
raw[12] = [152   2  51 ... 173 203 116]
raw[13] = [ 59   6 110 ... 175  90 103]
raw[14] = [ 65   3 126 ... 146 120 104]
raw[15] = [202   4 165 ... 216 117  97]
raw[16] = [ 13   6  51 ... 250 199 138]

Quantized data size: 128264
Bytes per value: 1


ollama:
(gdb) p lctx.model.vocab.cache_special_tokens
$3 = std::vector of length 257, capacity 512 = {128173, 128158, 128159, 128160, 128161, 128162, 128163, 128164,
  128165, 128166, 128167, 128168, 128169, 128170, 128171, 128172, 128157, 128174, 128175, 128176, 128177, 128178,
  128179, 128180, 128181, 128182, 128183, 128184, 128185, 128186, 128187, 128188, 128142, 128127, 128128, 128129,
  128130, 128131, 128132, 128133, 128134, 128135, 128136, 128137, 128138, 128139, 128140, 128141, 128189, 128143,
  128144, 128145, 128146, 128147, 128148, 128149, 128150, 128151, 128152, 128153, 128154, 128155, 128156, 128236,
  128221, 128222, 128223, 128224, 128225, 128226, 128227, 128228, 128229, 128230, 128231, 128232, 128233, 128234,
  128235, 128220, 128237, 128238, 128239, 128240, 128241, 128242, 128243, 128244, 128245, 128246, 128247, 128248,
  128249, 128250, 128251, 128205, 128190, 128191, 128192, 128193, 128194, 128195, 128196, 128197, 128198, 128199,
  128200, 128201, 128202, 128203, 128204, 128126, 128206, 128207, 128208, 128209, 128210, 128211, 128212, 128213,
  128214, 128215, 128216, 128217, 128218, 128219, 128125, 128252, 128253, 128254, 128255, 128109, 128110, 128111,
  128112, 128113, 128114, 128116, 128115, 128124, 128123, 128122, 128121, 128120, 128119, 128118, 128117, 128041,
  128042, 128043, 128044, 128045, 128046, 128047, 128048, 128049, 128050, 128051, 128052, 128053, 128054, 128055,
  128056, 128057, 128058, 128059, 128060, 128063, 128061, 128029, 128019, 128020, 128021, 128022, 128023, 128024,
  128025, 128026, 128027, 128028, 128040, 128030, 128031, 128032, 128033, 128034, 128035, 128036, 128037, 128038,
  128039, 128097, 128086, 128087, 128088, 128089, 128090, 128091, 128092, 128093, 128094, 128095, 128096, 128062,
  128098, 128099, 128100, 128101, 128102, 128103, 128104, 128105, 128106, 128107, 128108, 128084, 128064, 128065,
  128066, 128067, 128068, 128069, 128070, 128071, 128072, 128073, 128075, 128076, 128077, 128078, 128079, 128080,
  128081, 128074, 128085, 128083, 128082, 128003, 128002, 128011, 128012, 128013, 128014, 128015, 128016, 128017,
  128018, 128004, 128006, 128000, 128007, 128001, 128010, 128005, 128009, 128008, 128256}

  [{first = "!(", second = ":"}] = 127765...}, special_bos_id = 128000, special_eos_id = 128009,
  special_eot_id = 128009, special_eom_id = 128008, special_unk_id = -1, special_sep_id = -1,
  special_pad_id = 128004, special_cls_id = -1, special_mask_id = -1, linefeed_id = 128, special_fim_pre_id = -1,
  special_fim_suf_id = -1, special_fim_mid_id = -1, special_fim_pad_id = -1, special_fim_rep_id = -1,
  special_fim_sep_id = -1, special_eog_ids = std::set with 2 elements = {[0] = 128008, [1] = 128009},
  special_image_id = -1, tokenizer_add_space_prefix = false, tokenizer_add_bos = false, tokenizer_add_eos = false,
  tokenizer_ignore_merges = false, tokenizer_clean_spaces = true, tokenizer_remove_extra_whitespaces = false,
  tokenizer_escape_whitespaces = true, tokenizer_treat_whitespace_as_suffix = false,
  precompiled_charsmap = std::vector of length 0, capacity 0, tokenizer = 0x555559cc1980}


(gdb) p model.vocab.cache_special_tokens
$3 = std::vector of length 257, capacity 512 = {128173, 128158, 128159, 128160, 128161, 128162, 128163, 128164,
  128165, 128166, 128167, 128168, 128169, 128170, 128171, 128172, 128157, 128174, 128175, 128176, 128177, 128178,
  128179, 128180, 128181, 128182, 128183, 128184, 128185, 128186, 128187, 128188, 128142, 128127, 128128, 128129,
  128130, 128131, 128132, 128133, 128134, 128135, 128136, 128137, 128138, 128139, 128140, 128141, 128189, 128143,
  128144, 128145, 128146, 128147, 128148, 128149, 128150, 128151, 128152, 128153, 128154, 128155, 128156, 128236,
  128221, 128222, 128223, 128224, 128225, 128226, 128227, 128228, 128229, 128230, 128231, 128232, 128233, 128234,
  128235, 128220, 128237, 128238, 128239, 128240, 128241, 128242, 128243, 128244, 128245, 128246, 128247, 128248,
  128249, 128250, 128251, 128205, 128190, 128191, 128192, 128193, 128194, 128195, 128196, 128197, 128198, 128199,
  128200, 128201, 128202, 128203, 128204, 128126, 128206, 128207, 128208, 128209, 128210, 128211, 128212, 128213,
  128214, 128215, 128216, 128217, 128218, 128219, 128125, 128252, 128253, 128254, 128255, 128109, 128110, 128111,
  128112, 128113, 128114, 128116, 128115, 128124, 128123, 128122, 128121, 128120, 128119, 128118, 128117, 128041,
  128042, 128043, 128044, 128045, 128046, 128047, 128048, 128049, 128050, 128051, 128052, 128053, 128054, 128055,
  128056, 128057, 128058, 128059, 128060, 128063, 128061, 128029, 128019, 128020, 128021, 128022, 128023, 128024,
  128025, 128026, 128027, 128028, 128040, 128030, 128031, 128032, 128033, 128034, 128035, 128036, 128037, 128038,
  128039, 128097, 128086, 128087, 128088, 128089, 128090, 128091, 128092, 128093, 128094, 128095, 128096, 128062,
  128098, 128099, 128100, 128101, 128102, 128103, 128104, 128105, 128106, 128107, 128108, 128084, 128064, 128065,
  128066, 128067, 128068, 128069, 128070, 128071, 128072, 128073, 128075, 128076, 128077, 128078, 128079, 128080,
  128081, 128074, 128085, 128083, 128082, 128003, 128002, 128011, 128012, 128013, 128014, 128015, 128016, 128017,
  128018, 128004, 128006, 128000, 128007, 128001, 128010, 128005, 128009, 128008, 128256}

      [{first = "!(", second = ":"}] = 127765...}, special_bos_id = 128000, special_eos_id = 128009,
  special_eot_id = 128009, special_eom_id = 128008, special_unk_id = -1, special_sep_id = -1,
  special_pad_id = 128004, special_cls_id = -1, special_mask_id = -1, linefeed_id = 128, special_fim_pre_id = -1,
  special_fim_suf_id = -1, special_fim_mid_id = -1, special_fim_pad_id = -1, special_fim_rep_id = -1,
  special_fim_sep_id = -1, special_eog_ids = std::set with 2 elements = {[0] = 128008, [1] = 128009},
  special_image_id = -1, tokenizer_add_space_prefix = false, tokenizer_add_bos = false, tokenizer_add_eos = false,
  tokenizer_ignore_merges = false, tokenizer_clean_spaces = true, tokenizer_remove_extra_whitespaces = false,
  tokenizer_escape_whitespaces = true, tokenizer_treat_whitespace_as_suffix = false,
  precompiled_charsmap = std::vector of length 0, capacity 0, tokenizer = 0x555559d14c20}
```


### Ollama Sampling params
```
source=runner.go:128 msg="[danbev] --------------> sampling params: "
sparams="&{
TopK:40
TopP:0.9
MinP:0
TfsZ:1
TypicalP:1
Temp:0 RepeatLastN:64
PenaltyRepeat:1.1
PenaltyFreq:0
PenaltyPresent:0
Mirostat:0
MirostatTau:5
MirostatEta:0.1
PenalizeNl:true
Seed:4294967295 Grammar:}"
```

