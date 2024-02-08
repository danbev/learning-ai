## Large Language and Vision Assistent
LLaVA is an open-source chatbot trained by fine-tuning LLaMA/Vicuna on
GPT-generated multimodal instruction-following data.

It applies instruction tuning to visual data. And is an example of
Large Multimodal Language Model (LLM).

In llama.cpp there is now an example of LLaVA.

Llava has a LLM (which is LLama), a Visual Transformer (ViT),  and a Projection W.


First we clone git clone https://huggingface.co/liuhaotian/llava-v1.5-7b whic
is the intruction tuned model.

I did the following in the directory above my checked out llama.cpp directory.

We need to checkout the LLaVA model:
```console
$ git clone https://huggingface.co/liuhaotian/llava-v1.5-7b
```

And we need the Vision Transformer (ViT) model:
```console
$ git clone https://huggingface.co/openai/clip-vit-large-patch14-336
```

Create a Python virtual environment and install the required packages:
```console
$ python3.11 -m venv llava-venv
$ source llava-venv/bin/activate
(llava-venv) $ pip install torch numpy gguf transformers pillow sentencepiece
```
Then we can run the script llava-surgery.py script:
```console
(llava-venv) $ python examples/llava/llava-surgery.py -m ../llava-v1.5-7b/
Done!
Now you can convert ../llava-v1.5-7b/ to a a regular LLaMA GGUF file.
Also, use ../llava-v1.5-7b//llava.projector to prepare a llava-encoder.gguf file.
```

We then need to convert the image encoderl to the GGUF file format:
```console
(llava-venv) $ python ./examples/llava/convert-image-encoder-to-gguf.py -m ../clip-vit-large-patch14-336 --llava-projector ../llava-v1.5-7b/llava.projector --output-dir ../llava-v1.5-7b
gguf: This GGUF file is for Little Endian only
Projector tensors added

skipping parameter: logit_scale
skipping parameter: text_model.embeddings.token_embedding.weight
skipping parameter: text_model.embeddings.position_embedding.weight
skipping parameter: text_model.encoder.layers.0.self_attn.k_proj.weight
skipping parameter: text_model.encoder.layers.0.self_attn.k_proj.bias
skipping parameter: text_model.encoder.layers.0.self_attn.v_proj.weight
skipping parameter: text_model.encoder.layers.0.self_attn.v_proj.bias
skipping parameter: text_model.encoder.layers.0.self_attn.q_proj.weight
skipping parameter: text_model.encoder.layers.0.self_attn.q_proj.bias
skipping parameter: text_model.encoder.layers.0.self_attn.out_proj.weight
skipping parameter: text_model.encoder.layers.0.self_attn.out_proj.bias
skipping parameter: text_model.encoder.layers.0.layer_norm1.weight
skipping parameter: text_model.encoder.layers.0.layer_norm1.bias
skipping parameter: text_model.encoder.layers.0.mlp.fc1.weight
skipping parameter: text_model.encoder.layers.0.mlp.fc1.bias
skipping parameter: text_model.encoder.layers.0.mlp.fc2.weight
skipping parameter: text_model.encoder.layers.0.mlp.fc2.bias
skipping parameter: text_model.encoder.layers.0.layer_norm2.weight
skipping parameter: text_model.encoder.layers.0.layer_norm2.bias
skipping parameter: text_model.encoder.layers.1.self_attn.k_proj.weight
skipping parameter: text_model.encoder.layers.1.self_attn.k_proj.bias
skipping parameter: text_model.encoder.layers.1.self_attn.v_proj.weight
skipping parameter: text_model.encoder.layers.1.self_attn.v_proj.bias
skipping parameter: text_model.encoder.layers.1.self_attn.q_proj.weight
skipping parameter: text_model.encoder.layers.1.self_attn.q_proj.bias
skipping parameter: text_model.encoder.layers.1.self_attn.out_proj.weight
skipping parameter: text_model.encoder.layers.1.self_attn.out_proj.bias
skipping parameter: text_model.encoder.layers.1.layer_norm1.weight
skipping parameter: text_model.encoder.layers.1.layer_norm1.bias
skipping parameter: text_model.encoder.layers.1.mlp.fc1.weight
skipping parameter: text_model.encoder.layers.1.mlp.fc1.bias
skipping parameter: text_model.encoder.layers.1.mlp.fc2.weight
skipping parameter: text_model.encoder.layers.1.mlp.fc2.bias
skipping parameter: text_model.encoder.layers.1.layer_norm2.weight
skipping parameter: text_model.encoder.layers.1.layer_norm2.bias
skipping parameter: text_model.encoder.layers.2.self_attn.k_proj.weight
skipping parameter: text_model.encoder.layers.2.self_attn.k_proj.bias
skipping parameter: text_model.encoder.layers.2.self_attn.v_proj.weight
skipping parameter: text_model.encoder.layers.2.self_attn.v_proj.bias
skipping parameter: text_model.encoder.layers.2.self_attn.q_proj.weight
skipping parameter: text_model.encoder.layers.2.self_attn.q_proj.bias
skipping parameter: text_model.encoder.layers.2.self_attn.out_proj.weight
skipping parameter: text_model.encoder.layers.2.self_attn.out_proj.bias
skipping parameter: text_model.encoder.layers.2.layer_norm1.weight
skipping parameter: text_model.encoder.layers.2.layer_norm1.bias
skipping parameter: text_model.encoder.layers.2.mlp.fc1.weight
skipping parameter: text_model.encoder.layers.2.mlp.fc1.bias
skipping parameter: text_model.encoder.layers.2.mlp.fc2.weight
skipping parameter: text_model.encoder.layers.2.mlp.fc2.bias
skipping parameter: text_model.encoder.layers.2.layer_norm2.weight
skipping parameter: text_model.encoder.layers.2.layer_norm2.bias
skipping parameter: text_model.encoder.layers.3.self_attn.k_proj.weight
skipping parameter: text_model.encoder.layers.3.self_attn.k_proj.bias
skipping parameter: text_model.encoder.layers.3.self_attn.v_proj.weight
skipping parameter: text_model.encoder.layers.3.self_attn.v_proj.bias
skipping parameter: text_model.encoder.layers.3.self_attn.q_proj.weight
skipping parameter: text_model.encoder.layers.3.self_attn.q_proj.bias
skipping parameter: text_model.encoder.layers.3.self_attn.out_proj.weight
skipping parameter: text_model.encoder.layers.3.self_attn.out_proj.bias
skipping parameter: text_model.encoder.layers.3.layer_norm1.weight
skipping parameter: text_model.encoder.layers.3.layer_norm1.bias
skipping parameter: text_model.encoder.layers.3.mlp.fc1.weight
skipping parameter: text_model.encoder.layers.3.mlp.fc1.bias
skipping parameter: text_model.encoder.layers.3.mlp.fc2.weight
skipping parameter: text_model.encoder.layers.3.mlp.fc2.bias
skipping parameter: text_model.encoder.layers.3.layer_norm2.weight
skipping parameter: text_model.encoder.layers.3.layer_norm2.bias
skipping parameter: text_model.encoder.layers.4.self_attn.k_proj.weight
skipping parameter: text_model.encoder.layers.4.self_attn.k_proj.bias
skipping parameter: text_model.encoder.layers.4.self_attn.v_proj.weight
skipping parameter: text_model.encoder.layers.4.self_attn.v_proj.bias
skipping parameter: text_model.encoder.layers.4.self_attn.q_proj.weight
skipping parameter: text_model.encoder.layers.4.self_attn.q_proj.bias
skipping parameter: text_model.encoder.layers.4.self_attn.out_proj.weight
skipping parameter: text_model.encoder.layers.4.self_attn.out_proj.bias
skipping parameter: text_model.encoder.layers.4.layer_norm1.weight
skipping parameter: text_model.encoder.layers.4.layer_norm1.bias
skipping parameter: text_model.encoder.layers.4.mlp.fc1.weight
skipping parameter: text_model.encoder.layers.4.mlp.fc1.bias
skipping parameter: text_model.encoder.layers.4.mlp.fc2.weight
skipping parameter: text_model.encoder.layers.4.mlp.fc2.bias
skipping parameter: text_model.encoder.layers.4.layer_norm2.weight
skipping parameter: text_model.encoder.layers.4.layer_norm2.bias
skipping parameter: text_model.encoder.layers.5.self_attn.k_proj.weight
skipping parameter: text_model.encoder.layers.5.self_attn.k_proj.bias
skipping parameter: text_model.encoder.layers.5.self_attn.v_proj.weight
skipping parameter: text_model.encoder.layers.5.self_attn.v_proj.bias
skipping parameter: text_model.encoder.layers.5.self_attn.q_proj.weight
skipping parameter: text_model.encoder.layers.5.self_attn.q_proj.bias
skipping parameter: text_model.encoder.layers.5.self_attn.out_proj.weight
skipping parameter: text_model.encoder.layers.5.self_attn.out_proj.bias
skipping parameter: text_model.encoder.layers.5.layer_norm1.weight
skipping parameter: text_model.encoder.layers.5.layer_norm1.bias
skipping parameter: text_model.encoder.layers.5.mlp.fc1.weight
skipping parameter: text_model.encoder.layers.5.mlp.fc1.bias
skipping parameter: text_model.encoder.layers.5.mlp.fc2.weight
skipping parameter: text_model.encoder.layers.5.mlp.fc2.bias
skipping parameter: text_model.encoder.layers.5.layer_norm2.weight
skipping parameter: text_model.encoder.layers.5.layer_norm2.bias
skipping parameter: text_model.encoder.layers.6.self_attn.k_proj.weight
skipping parameter: text_model.encoder.layers.6.self_attn.k_proj.bias
skipping parameter: text_model.encoder.layers.6.self_attn.v_proj.weight
skipping parameter: text_model.encoder.layers.6.self_attn.v_proj.bias
skipping parameter: text_model.encoder.layers.6.self_attn.q_proj.weight
skipping parameter: text_model.encoder.layers.6.self_attn.q_proj.bias
skipping parameter: text_model.encoder.layers.6.self_attn.out_proj.weight
skipping parameter: text_model.encoder.layers.6.self_attn.out_proj.bias
skipping parameter: text_model.encoder.layers.6.layer_norm1.weight
skipping parameter: text_model.encoder.layers.6.layer_norm1.bias
skipping parameter: text_model.encoder.layers.6.mlp.fc1.weight
skipping parameter: text_model.encoder.layers.6.mlp.fc1.bias
skipping parameter: text_model.encoder.layers.6.mlp.fc2.weight
skipping parameter: text_model.encoder.layers.6.mlp.fc2.bias
skipping parameter: text_model.encoder.layers.6.layer_norm2.weight
skipping parameter: text_model.encoder.layers.6.layer_norm2.bias
skipping parameter: text_model.encoder.layers.7.self_attn.k_proj.weight
skipping parameter: text_model.encoder.layers.7.self_attn.k_proj.bias
skipping parameter: text_model.encoder.layers.7.self_attn.v_proj.weight
skipping parameter: text_model.encoder.layers.7.self_attn.v_proj.bias
skipping parameter: text_model.encoder.layers.7.self_attn.q_proj.weight
skipping parameter: text_model.encoder.layers.7.self_attn.q_proj.bias
skipping parameter: text_model.encoder.layers.7.self_attn.out_proj.weight
skipping parameter: text_model.encoder.layers.7.self_attn.out_proj.bias
skipping parameter: text_model.encoder.layers.7.layer_norm1.weight
skipping parameter: text_model.encoder.layers.7.layer_norm1.bias
skipping parameter: text_model.encoder.layers.7.mlp.fc1.weight
skipping parameter: text_model.encoder.layers.7.mlp.fc1.bias
skipping parameter: text_model.encoder.layers.7.mlp.fc2.weight
skipping parameter: text_model.encoder.layers.7.mlp.fc2.bias
skipping parameter: text_model.encoder.layers.7.layer_norm2.weight
skipping parameter: text_model.encoder.layers.7.layer_norm2.bias
skipping parameter: text_model.encoder.layers.8.self_attn.k_proj.weight
skipping parameter: text_model.encoder.layers.8.self_attn.k_proj.bias
skipping parameter: text_model.encoder.layers.8.self_attn.v_proj.weight
skipping parameter: text_model.encoder.layers.8.self_attn.v_proj.bias
skipping parameter: text_model.encoder.layers.8.self_attn.q_proj.weight
skipping parameter: text_model.encoder.layers.8.self_attn.q_proj.bias
skipping parameter: text_model.encoder.layers.8.self_attn.out_proj.weight
skipping parameter: text_model.encoder.layers.8.self_attn.out_proj.bias
skipping parameter: text_model.encoder.layers.8.layer_norm1.weight
skipping parameter: text_model.encoder.layers.8.layer_norm1.bias
skipping parameter: text_model.encoder.layers.8.mlp.fc1.weight
skipping parameter: text_model.encoder.layers.8.mlp.fc1.bias
skipping parameter: text_model.encoder.layers.8.mlp.fc2.weight
skipping parameter: text_model.encoder.layers.8.mlp.fc2.bias
skipping parameter: text_model.encoder.layers.8.layer_norm2.weight
skipping parameter: text_model.encoder.layers.8.layer_norm2.bias
skipping parameter: text_model.encoder.layers.9.self_attn.k_proj.weight
skipping parameter: text_model.encoder.layers.9.self_attn.k_proj.bias
skipping parameter: text_model.encoder.layers.9.self_attn.v_proj.weight
skipping parameter: text_model.encoder.layers.9.self_attn.v_proj.bias
skipping parameter: text_model.encoder.layers.9.self_attn.q_proj.weight
skipping parameter: text_model.encoder.layers.9.self_attn.q_proj.bias
skipping parameter: text_model.encoder.layers.9.self_attn.out_proj.weight
skipping parameter: text_model.encoder.layers.9.self_attn.out_proj.bias
skipping parameter: text_model.encoder.layers.9.layer_norm1.weight
skipping parameter: text_model.encoder.layers.9.layer_norm1.bias
skipping parameter: text_model.encoder.layers.9.mlp.fc1.weight
skipping parameter: text_model.encoder.layers.9.mlp.fc1.bias
skipping parameter: text_model.encoder.layers.9.mlp.fc2.weight
skipping parameter: text_model.encoder.layers.9.mlp.fc2.bias
skipping parameter: text_model.encoder.layers.9.layer_norm2.weight
skipping parameter: text_model.encoder.layers.9.layer_norm2.bias
skipping parameter: text_model.encoder.layers.10.self_attn.k_proj.weight
skipping parameter: text_model.encoder.layers.10.self_attn.k_proj.bias
skipping parameter: text_model.encoder.layers.10.self_attn.v_proj.weight
skipping parameter: text_model.encoder.layers.10.self_attn.v_proj.bias
skipping parameter: text_model.encoder.layers.10.self_attn.q_proj.weight
skipping parameter: text_model.encoder.layers.10.self_attn.q_proj.bias
skipping parameter: text_model.encoder.layers.10.self_attn.out_proj.weight
skipping parameter: text_model.encoder.layers.10.self_attn.out_proj.bias
skipping parameter: text_model.encoder.layers.10.layer_norm1.weight
skipping parameter: text_model.encoder.layers.10.layer_norm1.bias
skipping parameter: text_model.encoder.layers.10.mlp.fc1.weight
skipping parameter: text_model.encoder.layers.10.mlp.fc1.bias
skipping parameter: text_model.encoder.layers.10.mlp.fc2.weight
skipping parameter: text_model.encoder.layers.10.mlp.fc2.bias
skipping parameter: text_model.encoder.layers.10.layer_norm2.weight
skipping parameter: text_model.encoder.layers.10.layer_norm2.bias
skipping parameter: text_model.encoder.layers.11.self_attn.k_proj.weight
skipping parameter: text_model.encoder.layers.11.self_attn.k_proj.bias
skipping parameter: text_model.encoder.layers.11.self_attn.v_proj.weight
skipping parameter: text_model.encoder.layers.11.self_attn.v_proj.bias
skipping parameter: text_model.encoder.layers.11.self_attn.q_proj.weight
skipping parameter: text_model.encoder.layers.11.self_attn.q_proj.bias
skipping parameter: text_model.encoder.layers.11.self_attn.out_proj.weight
skipping parameter: text_model.encoder.layers.11.self_attn.out_proj.bias
skipping parameter: text_model.encoder.layers.11.layer_norm1.weight
skipping parameter: text_model.encoder.layers.11.layer_norm1.bias
skipping parameter: text_model.encoder.layers.11.mlp.fc1.weight
skipping parameter: text_model.encoder.layers.11.mlp.fc1.bias
skipping parameter: text_model.encoder.layers.11.mlp.fc2.weight
skipping parameter: text_model.encoder.layers.11.mlp.fc2.bias
skipping parameter: text_model.encoder.layers.11.layer_norm2.weight
skipping parameter: text_model.encoder.layers.11.layer_norm2.bias
skipping parameter: text_model.final_layer_norm.weight
skipping parameter: text_model.final_layer_norm.bias
  Converting to float32
v.class_embd - f32 - shape = (1024,)
tensor v.patch_embd.weight is always saved in f16
v.patch_embd.weight - f16 - shape = (1024, 3, 14, 14)
  Converting to float16
v.position_embd.weight - f16 - shape = (577, 1024)
  Converting to float32
v.pre_ln.weight - f32 - shape = (1024,)
  Converting to float32
v.pre_ln.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.0.attn_k.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.0.attn_k.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.0.attn_v.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.0.attn_v.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.0.attn_q.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.0.attn_q.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.0.attn_out.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.0.attn_out.bias - f32 - shape = (1024,)
  Converting to float32
v.blk.0.ln1.weight - f32 - shape = (1024,)
  Converting to float32
v.blk.0.ln1.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.0.ffn_down.weight - f16 - shape = (4096, 1024)
  Converting to float32
v.blk.0.ffn_down.bias - f32 - shape = (4096,)
  Converting to float16
v.blk.0.ffn_up.weight - f16 - shape = (1024, 4096)
  Converting to float32
v.blk.0.ffn_up.bias - f32 - shape = (1024,)
  Converting to float32
v.blk.0.ln2.weight - f32 - shape = (1024,)
  Converting to float32
v.blk.0.ln2.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.1.attn_k.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.1.attn_k.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.1.attn_v.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.1.attn_v.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.1.attn_q.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.1.attn_q.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.1.attn_out.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.1.attn_out.bias - f32 - shape = (1024,)
  Converting to float32
v.blk.1.ln1.weight - f32 - shape = (1024,)
  Converting to float32
v.blk.1.ln1.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.1.ffn_down.weight - f16 - shape = (4096, 1024)
  Converting to float32
v.blk.1.ffn_down.bias - f32 - shape = (4096,)
  Converting to float16
v.blk.1.ffn_up.weight - f16 - shape = (1024, 4096)
  Converting to float32
v.blk.1.ffn_up.bias - f32 - shape = (1024,)
  Converting to float32
v.blk.1.ln2.weight - f32 - shape = (1024,)
  Converting to float32
v.blk.1.ln2.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.2.attn_k.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.2.attn_k.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.2.attn_v.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.2.attn_v.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.2.attn_q.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.2.attn_q.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.2.attn_out.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.2.attn_out.bias - f32 - shape = (1024,)
  Converting to float32
v.blk.2.ln1.weight - f32 - shape = (1024,)
  Converting to float32
v.blk.2.ln1.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.2.ffn_down.weight - f16 - shape = (4096, 1024)
  Converting to float32
v.blk.2.ffn_down.bias - f32 - shape = (4096,)
  Converting to float16
v.blk.2.ffn_up.weight - f16 - shape = (1024, 4096)
  Converting to float32
v.blk.2.ffn_up.bias - f32 - shape = (1024,)
  Converting to float32
v.blk.2.ln2.weight - f32 - shape = (1024,)
  Converting to float32
v.blk.2.ln2.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.3.attn_k.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.3.attn_k.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.3.attn_v.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.3.attn_v.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.3.attn_q.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.3.attn_q.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.3.attn_out.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.3.attn_out.bias - f32 - shape = (1024,)
  Converting to float32
v.blk.3.ln1.weight - f32 - shape = (1024,)
  Converting to float32
v.blk.3.ln1.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.3.ffn_down.weight - f16 - shape = (4096, 1024)
  Converting to float32
v.blk.3.ffn_down.bias - f32 - shape = (4096,)
  Converting to float16
v.blk.3.ffn_up.weight - f16 - shape = (1024, 4096)
  Converting to float32
v.blk.3.ffn_up.bias - f32 - shape = (1024,)
  Converting to float32
v.blk.3.ln2.weight - f32 - shape = (1024,)
  Converting to float32
v.blk.3.ln2.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.4.attn_k.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.4.attn_k.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.4.attn_v.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.4.attn_v.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.4.attn_q.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.4.attn_q.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.4.attn_out.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.4.attn_out.bias - f32 - shape = (1024,)
  Converting to float32
v.blk.4.ln1.weight - f32 - shape = (1024,)
  Converting to float32
v.blk.4.ln1.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.4.ffn_down.weight - f16 - shape = (4096, 1024)
  Converting to float32
v.blk.4.ffn_down.bias - f32 - shape = (4096,)
  Converting to float16
v.blk.4.ffn_up.weight - f16 - shape = (1024, 4096)
  Converting to float32
v.blk.4.ffn_up.bias - f32 - shape = (1024,)
  Converting to float32
v.blk.4.ln2.weight - f32 - shape = (1024,)
  Converting to float32
v.blk.4.ln2.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.5.attn_k.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.5.attn_k.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.5.attn_v.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.5.attn_v.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.5.attn_q.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.5.attn_q.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.5.attn_out.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.5.attn_out.bias - f32 - shape = (1024,)
  Converting to float32
v.blk.5.ln1.weight - f32 - shape = (1024,)
  Converting to float32
v.blk.5.ln1.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.5.ffn_down.weight - f16 - shape = (4096, 1024)
  Converting to float32
v.blk.5.ffn_down.bias - f32 - shape = (4096,)
  Converting to float16
v.blk.5.ffn_up.weight - f16 - shape = (1024, 4096)
  Converting to float32
v.blk.5.ffn_up.bias - f32 - shape = (1024,)
  Converting to float32
v.blk.5.ln2.weight - f32 - shape = (1024,)
  Converting to float32
v.blk.5.ln2.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.6.attn_k.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.6.attn_k.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.6.attn_v.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.6.attn_v.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.6.attn_q.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.6.attn_q.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.6.attn_out.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.6.attn_out.bias - f32 - shape = (1024,)
  Converting to float32
v.blk.6.ln1.weight - f32 - shape = (1024,)
  Converting to float32
v.blk.6.ln1.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.6.ffn_down.weight - f16 - shape = (4096, 1024)
  Converting to float32
v.blk.6.ffn_down.bias - f32 - shape = (4096,)
  Converting to float16
v.blk.6.ffn_up.weight - f16 - shape = (1024, 4096)
  Converting to float32
v.blk.6.ffn_up.bias - f32 - shape = (1024,)
  Converting to float32
v.blk.6.ln2.weight - f32 - shape = (1024,)
  Converting to float32
v.blk.6.ln2.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.7.attn_k.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.7.attn_k.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.7.attn_v.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.7.attn_v.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.7.attn_q.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.7.attn_q.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.7.attn_out.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.7.attn_out.bias - f32 - shape = (1024,)
  Converting to float32
v.blk.7.ln1.weight - f32 - shape = (1024,)
  Converting to float32
v.blk.7.ln1.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.7.ffn_down.weight - f16 - shape = (4096, 1024)
  Converting to float32
v.blk.7.ffn_down.bias - f32 - shape = (4096,)
  Converting to float16
v.blk.7.ffn_up.weight - f16 - shape = (1024, 4096)
  Converting to float32
v.blk.7.ffn_up.bias - f32 - shape = (1024,)
  Converting to float32
v.blk.7.ln2.weight - f32 - shape = (1024,)
  Converting to float32
v.blk.7.ln2.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.8.attn_k.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.8.attn_k.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.8.attn_v.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.8.attn_v.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.8.attn_q.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.8.attn_q.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.8.attn_out.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.8.attn_out.bias - f32 - shape = (1024,)
  Converting to float32
v.blk.8.ln1.weight - f32 - shape = (1024,)
  Converting to float32
v.blk.8.ln1.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.8.ffn_down.weight - f16 - shape = (4096, 1024)
  Converting to float32
v.blk.8.ffn_down.bias - f32 - shape = (4096,)
  Converting to float16
v.blk.8.ffn_up.weight - f16 - shape = (1024, 4096)
  Converting to float32
v.blk.8.ffn_up.bias - f32 - shape = (1024,)
  Converting to float32
v.blk.8.ln2.weight - f32 - shape = (1024,)
  Converting to float32
v.blk.8.ln2.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.9.attn_k.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.9.attn_k.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.9.attn_v.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.9.attn_v.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.9.attn_q.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.9.attn_q.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.9.attn_out.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.9.attn_out.bias - f32 - shape = (1024,)
  Converting to float32
v.blk.9.ln1.weight - f32 - shape = (1024,)
  Converting to float32
v.blk.9.ln1.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.9.ffn_down.weight - f16 - shape = (4096, 1024)
  Converting to float32
v.blk.9.ffn_down.bias - f32 - shape = (4096,)
  Converting to float16
v.blk.9.ffn_up.weight - f16 - shape = (1024, 4096)
  Converting to float32
v.blk.9.ffn_up.bias - f32 - shape = (1024,)
  Converting to float32
v.blk.9.ln2.weight - f32 - shape = (1024,)
  Converting to float32
v.blk.9.ln2.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.10.attn_k.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.10.attn_k.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.10.attn_v.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.10.attn_v.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.10.attn_q.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.10.attn_q.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.10.attn_out.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.10.attn_out.bias - f32 - shape = (1024,)
  Converting to float32
v.blk.10.ln1.weight - f32 - shape = (1024,)
  Converting to float32
v.blk.10.ln1.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.10.ffn_down.weight - f16 - shape = (4096, 1024)
  Converting to float32
v.blk.10.ffn_down.bias - f32 - shape = (4096,)
  Converting to float16
v.blk.10.ffn_up.weight - f16 - shape = (1024, 4096)
  Converting to float32
v.blk.10.ffn_up.bias - f32 - shape = (1024,)
  Converting to float32
v.blk.10.ln2.weight - f32 - shape = (1024,)
  Converting to float32
v.blk.10.ln2.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.11.attn_k.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.11.attn_k.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.11.attn_v.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.11.attn_v.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.11.attn_q.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.11.attn_q.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.11.attn_out.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.11.attn_out.bias - f32 - shape = (1024,)
  Converting to float32
v.blk.11.ln1.weight - f32 - shape = (1024,)
  Converting to float32
v.blk.11.ln1.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.11.ffn_down.weight - f16 - shape = (4096, 1024)
  Converting to float32
v.blk.11.ffn_down.bias - f32 - shape = (4096,)
  Converting to float16
v.blk.11.ffn_up.weight - f16 - shape = (1024, 4096)
  Converting to float32
v.blk.11.ffn_up.bias - f32 - shape = (1024,)
  Converting to float32
v.blk.11.ln2.weight - f32 - shape = (1024,)
  Converting to float32
v.blk.11.ln2.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.12.attn_k.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.12.attn_k.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.12.attn_v.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.12.attn_v.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.12.attn_q.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.12.attn_q.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.12.attn_out.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.12.attn_out.bias - f32 - shape = (1024,)
  Converting to float32
v.blk.12.ln1.weight - f32 - shape = (1024,)
  Converting to float32
v.blk.12.ln1.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.12.ffn_down.weight - f16 - shape = (4096, 1024)
  Converting to float32
v.blk.12.ffn_down.bias - f32 - shape = (4096,)
  Converting to float16
v.blk.12.ffn_up.weight - f16 - shape = (1024, 4096)
  Converting to float32
v.blk.12.ffn_up.bias - f32 - shape = (1024,)
  Converting to float32
v.blk.12.ln2.weight - f32 - shape = (1024,)
  Converting to float32
v.blk.12.ln2.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.13.attn_k.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.13.attn_k.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.13.attn_v.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.13.attn_v.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.13.attn_q.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.13.attn_q.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.13.attn_out.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.13.attn_out.bias - f32 - shape = (1024,)
  Converting to float32
v.blk.13.ln1.weight - f32 - shape = (1024,)
  Converting to float32
v.blk.13.ln1.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.13.ffn_down.weight - f16 - shape = (4096, 1024)
  Converting to float32
v.blk.13.ffn_down.bias - f32 - shape = (4096,)
  Converting to float16
v.blk.13.ffn_up.weight - f16 - shape = (1024, 4096)
  Converting to float32
v.blk.13.ffn_up.bias - f32 - shape = (1024,)
  Converting to float32
v.blk.13.ln2.weight - f32 - shape = (1024,)
  Converting to float32
v.blk.13.ln2.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.14.attn_k.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.14.attn_k.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.14.attn_v.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.14.attn_v.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.14.attn_q.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.14.attn_q.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.14.attn_out.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.14.attn_out.bias - f32 - shape = (1024,)
  Converting to float32
v.blk.14.ln1.weight - f32 - shape = (1024,)
  Converting to float32
v.blk.14.ln1.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.14.ffn_down.weight - f16 - shape = (4096, 1024)
  Converting to float32
v.blk.14.ffn_down.bias - f32 - shape = (4096,)
  Converting to float16
v.blk.14.ffn_up.weight - f16 - shape = (1024, 4096)
  Converting to float32
v.blk.14.ffn_up.bias - f32 - shape = (1024,)
  Converting to float32
v.blk.14.ln2.weight - f32 - shape = (1024,)
  Converting to float32
v.blk.14.ln2.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.15.attn_k.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.15.attn_k.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.15.attn_v.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.15.attn_v.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.15.attn_q.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.15.attn_q.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.15.attn_out.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.15.attn_out.bias - f32 - shape = (1024,)
  Converting to float32
v.blk.15.ln1.weight - f32 - shape = (1024,)
  Converting to float32
v.blk.15.ln1.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.15.ffn_down.weight - f16 - shape = (4096, 1024)
  Converting to float32
v.blk.15.ffn_down.bias - f32 - shape = (4096,)
  Converting to float16
v.blk.15.ffn_up.weight - f16 - shape = (1024, 4096)
  Converting to float32
v.blk.15.ffn_up.bias - f32 - shape = (1024,)
  Converting to float32
v.blk.15.ln2.weight - f32 - shape = (1024,)
  Converting to float32
v.blk.15.ln2.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.16.attn_k.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.16.attn_k.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.16.attn_v.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.16.attn_v.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.16.attn_q.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.16.attn_q.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.16.attn_out.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.16.attn_out.bias - f32 - shape = (1024,)
  Converting to float32
v.blk.16.ln1.weight - f32 - shape = (1024,)
  Converting to float32
v.blk.16.ln1.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.16.ffn_down.weight - f16 - shape = (4096, 1024)
  Converting to float32
v.blk.16.ffn_down.bias - f32 - shape = (4096,)
  Converting to float16
v.blk.16.ffn_up.weight - f16 - shape = (1024, 4096)
  Converting to float32
v.blk.16.ffn_up.bias - f32 - shape = (1024,)
  Converting to float32
v.blk.16.ln2.weight - f32 - shape = (1024,)
  Converting to float32
v.blk.16.ln2.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.17.attn_k.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.17.attn_k.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.17.attn_v.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.17.attn_v.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.17.attn_q.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.17.attn_q.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.17.attn_out.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.17.attn_out.bias - f32 - shape = (1024,)
  Converting to float32
v.blk.17.ln1.weight - f32 - shape = (1024,)
  Converting to float32
v.blk.17.ln1.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.17.ffn_down.weight - f16 - shape = (4096, 1024)
  Converting to float32
v.blk.17.ffn_down.bias - f32 - shape = (4096,)
  Converting to float16
v.blk.17.ffn_up.weight - f16 - shape = (1024, 4096)
  Converting to float32
v.blk.17.ffn_up.bias - f32 - shape = (1024,)
  Converting to float32
v.blk.17.ln2.weight - f32 - shape = (1024,)
  Converting to float32
v.blk.17.ln2.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.18.attn_k.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.18.attn_k.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.18.attn_v.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.18.attn_v.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.18.attn_q.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.18.attn_q.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.18.attn_out.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.18.attn_out.bias - f32 - shape = (1024,)
  Converting to float32
v.blk.18.ln1.weight - f32 - shape = (1024,)
  Converting to float32
v.blk.18.ln1.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.18.ffn_down.weight - f16 - shape = (4096, 1024)
  Converting to float32
v.blk.18.ffn_down.bias - f32 - shape = (4096,)
  Converting to float16
v.blk.18.ffn_up.weight - f16 - shape = (1024, 4096)
  Converting to float32
v.blk.18.ffn_up.bias - f32 - shape = (1024,)
  Converting to float32
v.blk.18.ln2.weight - f32 - shape = (1024,)
  Converting to float32
v.blk.18.ln2.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.19.attn_k.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.19.attn_k.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.19.attn_v.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.19.attn_v.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.19.attn_q.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.19.attn_q.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.19.attn_out.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.19.attn_out.bias - f32 - shape = (1024,)
  Converting to float32
v.blk.19.ln1.weight - f32 - shape = (1024,)
  Converting to float32
v.blk.19.ln1.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.19.ffn_down.weight - f16 - shape = (4096, 1024)
  Converting to float32
v.blk.19.ffn_down.bias - f32 - shape = (4096,)
  Converting to float16
v.blk.19.ffn_up.weight - f16 - shape = (1024, 4096)
  Converting to float32
v.blk.19.ffn_up.bias - f32 - shape = (1024,)
  Converting to float32
v.blk.19.ln2.weight - f32 - shape = (1024,)
  Converting to float32
v.blk.19.ln2.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.20.attn_k.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.20.attn_k.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.20.attn_v.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.20.attn_v.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.20.attn_q.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.20.attn_q.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.20.attn_out.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.20.attn_out.bias - f32 - shape = (1024,)
  Converting to float32
v.blk.20.ln1.weight - f32 - shape = (1024,)
  Converting to float32
v.blk.20.ln1.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.20.ffn_down.weight - f16 - shape = (4096, 1024)
  Converting to float32
v.blk.20.ffn_down.bias - f32 - shape = (4096,)
  Converting to float16
v.blk.20.ffn_up.weight - f16 - shape = (1024, 4096)
  Converting to float32
v.blk.20.ffn_up.bias - f32 - shape = (1024,)
  Converting to float32
v.blk.20.ln2.weight - f32 - shape = (1024,)
  Converting to float32
v.blk.20.ln2.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.21.attn_k.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.21.attn_k.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.21.attn_v.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.21.attn_v.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.21.attn_q.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.21.attn_q.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.21.attn_out.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.21.attn_out.bias - f32 - shape = (1024,)
  Converting to float32
v.blk.21.ln1.weight - f32 - shape = (1024,)
  Converting to float32
v.blk.21.ln1.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.21.ffn_down.weight - f16 - shape = (4096, 1024)
  Converting to float32
v.blk.21.ffn_down.bias - f32 - shape = (4096,)
  Converting to float16
v.blk.21.ffn_up.weight - f16 - shape = (1024, 4096)
  Converting to float32
v.blk.21.ffn_up.bias - f32 - shape = (1024,)
  Converting to float32
v.blk.21.ln2.weight - f32 - shape = (1024,)
  Converting to float32
v.blk.21.ln2.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.22.attn_k.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.22.attn_k.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.22.attn_v.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.22.attn_v.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.22.attn_q.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.22.attn_q.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.22.attn_out.weight - f16 - shape = (1024, 1024)
  Converting to float32
v.blk.22.attn_out.bias - f32 - shape = (1024,)
  Converting to float32
v.blk.22.ln1.weight - f32 - shape = (1024,)
  Converting to float32
v.blk.22.ln1.bias - f32 - shape = (1024,)
  Converting to float16
v.blk.22.ffn_down.weight - f16 - shape = (4096, 1024)
  Converting to float32
v.blk.22.ffn_down.bias - f32 - shape = (4096,)
  Converting to float16
v.blk.22.ffn_up.weight - f16 - shape = (1024, 4096)
  Converting to float32
v.blk.22.ffn_up.bias - f32 - shape = (1024,)
  Converting to float32
v.blk.22.ln2.weight - f32 - shape = (1024,)
  Converting to float32
v.blk.22.ln2.bias - f32 - shape = (1024,)
skipping parameter: vision_model.post_layernorm.weight
skipping parameter: vision_model.post_layernorm.bias
skipping parameter: visual_projection.weight
skipping parameter: text_projection.weight
Done. Output file: ../llava-v1.5-7b/mmproj-model-f16.gguf
```

Then we need to convert the llama part of llava to GGUF format:
```console
(llava-venv) $ python ./convert.py ../llava-v1.5-7b
Loading model file ../llava-v1.5-7b/pytorch_model-00001-of-00002.bin
Loading model file ../llava-v1.5-7b/pytorch_model-00001-of-00002.bin
Loading model file ../llava-v1.5-7b/pytorch_model-00002-of-00002.bin
params = Params(n_vocab=32000, n_embd=4096, n_layer=32, n_ctx=4096, n_ff=11008, n_head=32, n_head_kv=32, n_experts=None, n_experts_used=None, f_norm_eps=1e-05, rope_scaling_type=None, f_rope_freq_base=None, f_rope_scale=None, n_orig_ctx=None, rope_finetuned=None, ftype=None, path_model=PosixPath('../llava-v1.5-7b'))
Found vocab files: {'tokenizer.model': PosixPath('../llava-v1.5-7b/tokenizer.model'), 'vocab.json': None, 'tokenizer.json': None}
Loading vocab file '../llava-v1.5-7b/tokenizer.model', type 'spm'
Vocab info: <SentencePieceVocab with 32000 base tokens and 0 added tokens>
Special vocab info: <SpecialVocab with 0 merges, special tokens {'bos': 1, 'eos': 2, 'pad': 0}, add special tokens {'bos': True, 'eos': False}>
Permuting layer 0
Permuting layer 1
Permuting layer 2
Permuting layer 3
Permuting layer 4
Permuting layer 5
Permuting layer 6
Permuting layer 7
Permuting layer 8
Permuting layer 9
Permuting layer 10
Permuting layer 11
Permuting layer 12
Permuting layer 13
Permuting layer 14
Permuting layer 15
Permuting layer 16
Permuting layer 17
Permuting layer 18
Permuting layer 19
Permuting layer 20
Permuting layer 21
Permuting layer 22
Permuting layer 23
Permuting layer 24
Permuting layer 25
Permuting layer 26
Permuting layer 27
Permuting layer 28
Permuting layer 29
Permuting layer 30
Permuting layer 31
model.embed_tokens.weight                        -> token_embd.weight                        | F16    | [32000, 4096]
model.layers.0.self_attn.q_proj.weight           -> blk.0.attn_q.weight                      | F16    | [4096, 4096]
model.layers.0.self_attn.k_proj.weight           -> blk.0.attn_k.weight                      | F16    | [4096, 4096]
model.layers.0.self_attn.v_proj.weight           -> blk.0.attn_v.weight                      | F16    | [4096, 4096]
model.layers.0.self_attn.o_proj.weight           -> blk.0.attn_output.weight                 | F16    | [4096, 4096]
skipping tensor blk.0.attn_rot_embd
model.layers.0.mlp.gate_proj.weight              -> blk.0.ffn_gate.weight                    | F16    | [11008, 4096]
model.layers.0.mlp.up_proj.weight                -> blk.0.ffn_up.weight                      | F16    | [11008, 4096]
model.layers.0.mlp.down_proj.weight              -> blk.0.ffn_down.weight                    | F16    | [4096, 11008]
model.layers.0.input_layernorm.weight            -> blk.0.attn_norm.weight                   | F16    | [4096]
model.layers.0.post_attention_layernorm.weight   -> blk.0.ffn_norm.weight                    | F16    | [4096]
model.layers.1.self_attn.q_proj.weight           -> blk.1.attn_q.weight                      | F16    | [4096, 4096]
model.layers.1.self_attn.k_proj.weight           -> blk.1.attn_k.weight                      | F16    | [4096, 4096]
model.layers.1.self_attn.v_proj.weight           -> blk.1.attn_v.weight                      | F16    | [4096, 4096]
model.layers.1.self_attn.o_proj.weight           -> blk.1.attn_output.weight                 | F16    | [4096, 4096]
skipping tensor blk.1.attn_rot_embd
model.layers.1.mlp.gate_proj.weight              -> blk.1.ffn_gate.weight                    | F16    | [11008, 4096]
model.layers.1.mlp.up_proj.weight                -> blk.1.ffn_up.weight                      | F16    | [11008, 4096]
model.layers.1.mlp.down_proj.weight              -> blk.1.ffn_down.weight                    | F16    | [4096, 11008]
model.layers.1.input_layernorm.weight            -> blk.1.attn_norm.weight                   | F16    | [4096]
model.layers.1.post_attention_layernorm.weight   -> blk.1.ffn_norm.weight                    | F16    | [4096]
model.layers.2.self_attn.q_proj.weight           -> blk.2.attn_q.weight                      | F16    | [4096, 4096]
model.layers.2.self_attn.k_proj.weight           -> blk.2.attn_k.weight                      | F16    | [4096, 4096]
model.layers.2.self_attn.v_proj.weight           -> blk.2.attn_v.weight                      | F16    | [4096, 4096]
model.layers.2.self_attn.o_proj.weight           -> blk.2.attn_output.weight                 | F16    | [4096, 4096]
skipping tensor blk.2.attn_rot_embd
model.layers.2.mlp.gate_proj.weight              -> blk.2.ffn_gate.weight                    | F16    | [11008, 4096]
model.layers.2.mlp.up_proj.weight                -> blk.2.ffn_up.weight                      | F16    | [11008, 4096]
model.layers.2.mlp.down_proj.weight              -> blk.2.ffn_down.weight                    | F16    | [4096, 11008]
model.layers.2.input_layernorm.weight            -> blk.2.attn_norm.weight                   | F16    | [4096]
model.layers.2.post_attention_layernorm.weight   -> blk.2.ffn_norm.weight                    | F16    | [4096]
model.layers.3.self_attn.q_proj.weight           -> blk.3.attn_q.weight                      | F16    | [4096, 4096]
model.layers.3.self_attn.k_proj.weight           -> blk.3.attn_k.weight                      | F16    | [4096, 4096]
model.layers.3.self_attn.v_proj.weight           -> blk.3.attn_v.weight                      | F16    | [4096, 4096]
model.layers.3.self_attn.o_proj.weight           -> blk.3.attn_output.weight                 | F16    | [4096, 4096]
skipping tensor blk.3.attn_rot_embd
model.layers.3.mlp.gate_proj.weight              -> blk.3.ffn_gate.weight                    | F16    | [11008, 4096]
model.layers.3.mlp.up_proj.weight                -> blk.3.ffn_up.weight                      | F16    | [11008, 4096]
model.layers.3.mlp.down_proj.weight              -> blk.3.ffn_down.weight                    | F16    | [4096, 11008]
model.layers.3.input_layernorm.weight            -> blk.3.attn_norm.weight                   | F16    | [4096]
model.layers.3.post_attention_layernorm.weight   -> blk.3.ffn_norm.weight                    | F16    | [4096]
model.layers.4.self_attn.q_proj.weight           -> blk.4.attn_q.weight                      | F16    | [4096, 4096]
model.layers.4.self_attn.k_proj.weight           -> blk.4.attn_k.weight                      | F16    | [4096, 4096]
model.layers.4.self_attn.v_proj.weight           -> blk.4.attn_v.weight                      | F16    | [4096, 4096]
model.layers.4.self_attn.o_proj.weight           -> blk.4.attn_output.weight                 | F16    | [4096, 4096]
skipping tensor blk.4.attn_rot_embd
model.layers.4.mlp.gate_proj.weight              -> blk.4.ffn_gate.weight                    | F16    | [11008, 4096]
model.layers.4.mlp.up_proj.weight                -> blk.4.ffn_up.weight                      | F16    | [11008, 4096]
model.layers.4.mlp.down_proj.weight              -> blk.4.ffn_down.weight                    | F16    | [4096, 11008]
model.layers.4.input_layernorm.weight            -> blk.4.attn_norm.weight                   | F16    | [4096]
model.layers.4.post_attention_layernorm.weight   -> blk.4.ffn_norm.weight                    | F16    | [4096]
model.layers.5.self_attn.q_proj.weight           -> blk.5.attn_q.weight                      | F16    | [4096, 4096]
model.layers.5.self_attn.k_proj.weight           -> blk.5.attn_k.weight                      | F16    | [4096, 4096]
model.layers.5.self_attn.v_proj.weight           -> blk.5.attn_v.weight                      | F16    | [4096, 4096]
model.layers.5.self_attn.o_proj.weight           -> blk.5.attn_output.weight                 | F16    | [4096, 4096]
skipping tensor blk.5.attn_rot_embd
model.layers.5.mlp.gate_proj.weight              -> blk.5.ffn_gate.weight                    | F16    | [11008, 4096]
model.layers.5.mlp.up_proj.weight                -> blk.5.ffn_up.weight                      | F16    | [11008, 4096]
model.layers.5.mlp.down_proj.weight              -> blk.5.ffn_down.weight                    | F16    | [4096, 11008]
model.layers.5.input_layernorm.weight            -> blk.5.attn_norm.weight                   | F16    | [4096]
model.layers.5.post_attention_layernorm.weight   -> blk.5.ffn_norm.weight                    | F16    | [4096]
model.layers.6.self_attn.q_proj.weight           -> blk.6.attn_q.weight                      | F16    | [4096, 4096]
model.layers.6.self_attn.k_proj.weight           -> blk.6.attn_k.weight                      | F16    | [4096, 4096]
model.layers.6.self_attn.v_proj.weight           -> blk.6.attn_v.weight                      | F16    | [4096, 4096]
model.layers.6.self_attn.o_proj.weight           -> blk.6.attn_output.weight                 | F16    | [4096, 4096]
skipping tensor blk.6.attn_rot_embd
model.layers.6.mlp.gate_proj.weight              -> blk.6.ffn_gate.weight                    | F16    | [11008, 4096]
model.layers.6.mlp.up_proj.weight                -> blk.6.ffn_up.weight                      | F16    | [11008, 4096]
model.layers.6.mlp.down_proj.weight              -> blk.6.ffn_down.weight                    | F16    | [4096, 11008]
model.layers.6.input_layernorm.weight            -> blk.6.attn_norm.weight                   | F16    | [4096]
model.layers.6.post_attention_layernorm.weight   -> blk.6.ffn_norm.weight                    | F16    | [4096]
model.layers.7.self_attn.q_proj.weight           -> blk.7.attn_q.weight                      | F16    | [4096, 4096]
model.layers.7.self_attn.k_proj.weight           -> blk.7.attn_k.weight                      | F16    | [4096, 4096]
model.layers.7.self_attn.v_proj.weight           -> blk.7.attn_v.weight                      | F16    | [4096, 4096]
model.layers.7.self_attn.o_proj.weight           -> blk.7.attn_output.weight                 | F16    | [4096, 4096]
skipping tensor blk.7.attn_rot_embd
model.layers.7.mlp.gate_proj.weight              -> blk.7.ffn_gate.weight                    | F16    | [11008, 4096]
model.layers.7.mlp.up_proj.weight                -> blk.7.ffn_up.weight                      | F16    | [11008, 4096]
model.layers.7.mlp.down_proj.weight              -> blk.7.ffn_down.weight                    | F16    | [4096, 11008]
model.layers.7.input_layernorm.weight            -> blk.7.attn_norm.weight                   | F16    | [4096]
model.layers.7.post_attention_layernorm.weight   -> blk.7.ffn_norm.weight                    | F16    | [4096]
model.layers.8.self_attn.q_proj.weight           -> blk.8.attn_q.weight                      | F16    | [4096, 4096]
model.layers.8.self_attn.k_proj.weight           -> blk.8.attn_k.weight                      | F16    | [4096, 4096]
model.layers.8.self_attn.v_proj.weight           -> blk.8.attn_v.weight                      | F16    | [4096, 4096]
model.layers.8.self_attn.o_proj.weight           -> blk.8.attn_output.weight                 | F16    | [4096, 4096]
skipping tensor blk.8.attn_rot_embd
model.layers.8.mlp.gate_proj.weight              -> blk.8.ffn_gate.weight                    | F16    | [11008, 4096]
model.layers.8.mlp.up_proj.weight                -> blk.8.ffn_up.weight                      | F16    | [11008, 4096]
model.layers.8.mlp.down_proj.weight              -> blk.8.ffn_down.weight                    | F16    | [4096, 11008]
model.layers.8.input_layernorm.weight            -> blk.8.attn_norm.weight                   | F16    | [4096]
model.layers.8.post_attention_layernorm.weight   -> blk.8.ffn_norm.weight                    | F16    | [4096]
model.layers.9.self_attn.q_proj.weight           -> blk.9.attn_q.weight                      | F16    | [4096, 4096]
model.layers.9.self_attn.k_proj.weight           -> blk.9.attn_k.weight                      | F16    | [4096, 4096]
model.layers.9.self_attn.v_proj.weight           -> blk.9.attn_v.weight                      | F16    | [4096, 4096]
model.layers.9.self_attn.o_proj.weight           -> blk.9.attn_output.weight                 | F16    | [4096, 4096]
skipping tensor blk.9.attn_rot_embd
model.layers.9.mlp.gate_proj.weight              -> blk.9.ffn_gate.weight                    | F16    | [11008, 4096]
model.layers.9.mlp.up_proj.weight                -> blk.9.ffn_up.weight                      | F16    | [11008, 4096]
model.layers.9.mlp.down_proj.weight              -> blk.9.ffn_down.weight                    | F16    | [4096, 11008]
model.layers.9.input_layernorm.weight            -> blk.9.attn_norm.weight                   | F16    | [4096]
model.layers.9.post_attention_layernorm.weight   -> blk.9.ffn_norm.weight                    | F16    | [4096]
model.layers.10.self_attn.q_proj.weight          -> blk.10.attn_q.weight                     | F16    | [4096, 4096]
model.layers.10.self_attn.k_proj.weight          -> blk.10.attn_k.weight                     | F16    | [4096, 4096]
model.layers.10.self_attn.v_proj.weight          -> blk.10.attn_v.weight                     | F16    | [4096, 4096]
model.layers.10.self_attn.o_proj.weight          -> blk.10.attn_output.weight                | F16    | [4096, 4096]
skipping tensor blk.10.attn_rot_embd
model.layers.10.mlp.gate_proj.weight             -> blk.10.ffn_gate.weight                   | F16    | [11008, 4096]
model.layers.10.mlp.up_proj.weight               -> blk.10.ffn_up.weight                     | F16    | [11008, 4096]
model.layers.10.mlp.down_proj.weight             -> blk.10.ffn_down.weight                   | F16    | [4096, 11008]
model.layers.10.input_layernorm.weight           -> blk.10.attn_norm.weight                  | F16    | [4096]
model.layers.10.post_attention_layernorm.weight  -> blk.10.ffn_norm.weight                   | F16    | [4096]
model.layers.11.self_attn.q_proj.weight          -> blk.11.attn_q.weight                     | F16    | [4096, 4096]
model.layers.11.self_attn.k_proj.weight          -> blk.11.attn_k.weight                     | F16    | [4096, 4096]
model.layers.11.self_attn.v_proj.weight          -> blk.11.attn_v.weight                     | F16    | [4096, 4096]
model.layers.11.self_attn.o_proj.weight          -> blk.11.attn_output.weight                | F16    | [4096, 4096]
skipping tensor blk.11.attn_rot_embd
model.layers.11.mlp.gate_proj.weight             -> blk.11.ffn_gate.weight                   | F16    | [11008, 4096]
model.layers.11.mlp.up_proj.weight               -> blk.11.ffn_up.weight                     | F16    | [11008, 4096]
model.layers.11.mlp.down_proj.weight             -> blk.11.ffn_down.weight                   | F16    | [4096, 11008]
model.layers.11.input_layernorm.weight           -> blk.11.attn_norm.weight                  | F16    | [4096]
model.layers.11.post_attention_layernorm.weight  -> blk.11.ffn_norm.weight                   | F16    | [4096]
model.layers.12.self_attn.q_proj.weight          -> blk.12.attn_q.weight                     | F16    | [4096, 4096]
model.layers.12.self_attn.k_proj.weight          -> blk.12.attn_k.weight                     | F16    | [4096, 4096]
model.layers.12.self_attn.v_proj.weight          -> blk.12.attn_v.weight                     | F16    | [4096, 4096]
model.layers.12.self_attn.o_proj.weight          -> blk.12.attn_output.weight                | F16    | [4096, 4096]
skipping tensor blk.12.attn_rot_embd
model.layers.12.mlp.gate_proj.weight             -> blk.12.ffn_gate.weight                   | F16    | [11008, 4096]
model.layers.12.mlp.up_proj.weight               -> blk.12.ffn_up.weight                     | F16    | [11008, 4096]
model.layers.12.mlp.down_proj.weight             -> blk.12.ffn_down.weight                   | F16    | [4096, 11008]
model.layers.12.input_layernorm.weight           -> blk.12.attn_norm.weight                  | F16    | [4096]
model.layers.12.post_attention_layernorm.weight  -> blk.12.ffn_norm.weight                   | F16    | [4096]
model.layers.13.self_attn.q_proj.weight          -> blk.13.attn_q.weight                     | F16    | [4096, 4096]
model.layers.13.self_attn.k_proj.weight          -> blk.13.attn_k.weight                     | F16    | [4096, 4096]
model.layers.13.self_attn.v_proj.weight          -> blk.13.attn_v.weight                     | F16    | [4096, 4096]
model.layers.13.self_attn.o_proj.weight          -> blk.13.attn_output.weight                | F16    | [4096, 4096]
skipping tensor blk.13.attn_rot_embd
model.layers.13.mlp.gate_proj.weight             -> blk.13.ffn_gate.weight                   | F16    | [11008, 4096]
model.layers.13.mlp.up_proj.weight               -> blk.13.ffn_up.weight                     | F16    | [11008, 4096]
model.layers.13.mlp.down_proj.weight             -> blk.13.ffn_down.weight                   | F16    | [4096, 11008]
model.layers.13.input_layernorm.weight           -> blk.13.attn_norm.weight                  | F16    | [4096]
model.layers.13.post_attention_layernorm.weight  -> blk.13.ffn_norm.weight                   | F16    | [4096]
model.layers.14.self_attn.q_proj.weight          -> blk.14.attn_q.weight                     | F16    | [4096, 4096]
model.layers.14.self_attn.k_proj.weight          -> blk.14.attn_k.weight                     | F16    | [4096, 4096]
model.layers.14.self_attn.v_proj.weight          -> blk.14.attn_v.weight                     | F16    | [4096, 4096]
model.layers.14.self_attn.o_proj.weight          -> blk.14.attn_output.weight                | F16    | [4096, 4096]
skipping tensor blk.14.attn_rot_embd
model.layers.14.mlp.gate_proj.weight             -> blk.14.ffn_gate.weight                   | F16    | [11008, 4096]
model.layers.14.mlp.up_proj.weight               -> blk.14.ffn_up.weight                     | F16    | [11008, 4096]
model.layers.14.mlp.down_proj.weight             -> blk.14.ffn_down.weight                   | F16    | [4096, 11008]
model.layers.14.input_layernorm.weight           -> blk.14.attn_norm.weight                  | F16    | [4096]
model.layers.14.post_attention_layernorm.weight  -> blk.14.ffn_norm.weight                   | F16    | [4096]
model.layers.15.self_attn.q_proj.weight          -> blk.15.attn_q.weight                     | F16    | [4096, 4096]
model.layers.15.self_attn.k_proj.weight          -> blk.15.attn_k.weight                     | F16    | [4096, 4096]
model.layers.15.self_attn.v_proj.weight          -> blk.15.attn_v.weight                     | F16    | [4096, 4096]
model.layers.15.self_attn.o_proj.weight          -> blk.15.attn_output.weight                | F16    | [4096, 4096]
skipping tensor blk.15.attn_rot_embd
model.layers.15.mlp.gate_proj.weight             -> blk.15.ffn_gate.weight                   | F16    | [11008, 4096]
model.layers.15.mlp.up_proj.weight               -> blk.15.ffn_up.weight                     | F16    | [11008, 4096]
model.layers.15.mlp.down_proj.weight             -> blk.15.ffn_down.weight                   | F16    | [4096, 11008]
model.layers.15.input_layernorm.weight           -> blk.15.attn_norm.weight                  | F16    | [4096]
model.layers.15.post_attention_layernorm.weight  -> blk.15.ffn_norm.weight                   | F16    | [4096]
model.layers.16.self_attn.q_proj.weight          -> blk.16.attn_q.weight                     | F16    | [4096, 4096]
model.layers.16.self_attn.k_proj.weight          -> blk.16.attn_k.weight                     | F16    | [4096, 4096]
model.layers.16.self_attn.v_proj.weight          -> blk.16.attn_v.weight                     | F16    | [4096, 4096]
model.layers.16.self_attn.o_proj.weight          -> blk.16.attn_output.weight                | F16    | [4096, 4096]
skipping tensor blk.16.attn_rot_embd
model.layers.16.mlp.gate_proj.weight             -> blk.16.ffn_gate.weight                   | F16    | [11008, 4096]
model.layers.16.mlp.up_proj.weight               -> blk.16.ffn_up.weight                     | F16    | [11008, 4096]
model.layers.16.mlp.down_proj.weight             -> blk.16.ffn_down.weight                   | F16    | [4096, 11008]
model.layers.16.input_layernorm.weight           -> blk.16.attn_norm.weight                  | F16    | [4096]
model.layers.16.post_attention_layernorm.weight  -> blk.16.ffn_norm.weight                   | F16    | [4096]
model.layers.17.self_attn.q_proj.weight          -> blk.17.attn_q.weight                     | F16    | [4096, 4096]
model.layers.17.self_attn.k_proj.weight          -> blk.17.attn_k.weight                     | F16    | [4096, 4096]
model.layers.17.self_attn.v_proj.weight          -> blk.17.attn_v.weight                     | F16    | [4096, 4096]
model.layers.17.self_attn.o_proj.weight          -> blk.17.attn_output.weight                | F16    | [4096, 4096]
skipping tensor blk.17.attn_rot_embd
model.layers.17.mlp.gate_proj.weight             -> blk.17.ffn_gate.weight                   | F16    | [11008, 4096]
model.layers.17.mlp.up_proj.weight               -> blk.17.ffn_up.weight                     | F16    | [11008, 4096]
model.layers.17.mlp.down_proj.weight             -> blk.17.ffn_down.weight                   | F16    | [4096, 11008]
model.layers.17.input_layernorm.weight           -> blk.17.attn_norm.weight                  | F16    | [4096]
model.layers.17.post_attention_layernorm.weight  -> blk.17.ffn_norm.weight                   | F16    | [4096]
model.layers.18.self_attn.q_proj.weight          -> blk.18.attn_q.weight                     | F16    | [4096, 4096]
model.layers.18.self_attn.k_proj.weight          -> blk.18.attn_k.weight                     | F16    | [4096, 4096]
model.layers.18.self_attn.v_proj.weight          -> blk.18.attn_v.weight                     | F16    | [4096, 4096]
model.layers.18.self_attn.o_proj.weight          -> blk.18.attn_output.weight                | F16    | [4096, 4096]
skipping tensor blk.18.attn_rot_embd
model.layers.18.mlp.gate_proj.weight             -> blk.18.ffn_gate.weight                   | F16    | [11008, 4096]
model.layers.18.mlp.up_proj.weight               -> blk.18.ffn_up.weight                     | F16    | [11008, 4096]
model.layers.18.mlp.down_proj.weight             -> blk.18.ffn_down.weight                   | F16    | [4096, 11008]
model.layers.18.input_layernorm.weight           -> blk.18.attn_norm.weight                  | F16    | [4096]
model.layers.18.post_attention_layernorm.weight  -> blk.18.ffn_norm.weight                   | F16    | [4096]
model.layers.19.self_attn.q_proj.weight          -> blk.19.attn_q.weight                     | F16    | [4096, 4096]
model.layers.19.self_attn.k_proj.weight          -> blk.19.attn_k.weight                     | F16    | [4096, 4096]
model.layers.19.self_attn.v_proj.weight          -> blk.19.attn_v.weight                     | F16    | [4096, 4096]
model.layers.19.self_attn.o_proj.weight          -> blk.19.attn_output.weight                | F16    | [4096, 4096]
skipping tensor blk.19.attn_rot_embd
model.layers.19.mlp.gate_proj.weight             -> blk.19.ffn_gate.weight                   | F16    | [11008, 4096]
model.layers.19.mlp.up_proj.weight               -> blk.19.ffn_up.weight                     | F16    | [11008, 4096]
model.layers.19.mlp.down_proj.weight             -> blk.19.ffn_down.weight                   | F16    | [4096, 11008]
model.layers.19.input_layernorm.weight           -> blk.19.attn_norm.weight                  | F16    | [4096]
model.layers.19.post_attention_layernorm.weight  -> blk.19.ffn_norm.weight                   | F16    | [4096]
model.layers.20.self_attn.q_proj.weight          -> blk.20.attn_q.weight                     | F16    | [4096, 4096]
model.layers.20.self_attn.k_proj.weight          -> blk.20.attn_k.weight                     | F16    | [4096, 4096]
model.layers.20.self_attn.v_proj.weight          -> blk.20.attn_v.weight                     | F16    | [4096, 4096]
model.layers.20.self_attn.o_proj.weight          -> blk.20.attn_output.weight                | F16    | [4096, 4096]
skipping tensor blk.20.attn_rot_embd
model.layers.20.mlp.gate_proj.weight             -> blk.20.ffn_gate.weight                   | F16    | [11008, 4096]
model.layers.20.mlp.up_proj.weight               -> blk.20.ffn_up.weight                     | F16    | [11008, 4096]
model.layers.20.mlp.down_proj.weight             -> blk.20.ffn_down.weight                   | F16    | [4096, 11008]
model.layers.20.input_layernorm.weight           -> blk.20.attn_norm.weight                  | F16    | [4096]
model.layers.20.post_attention_layernorm.weight  -> blk.20.ffn_norm.weight                   | F16    | [4096]
model.layers.21.self_attn.q_proj.weight          -> blk.21.attn_q.weight                     | F16    | [4096, 4096]
model.layers.21.self_attn.k_proj.weight          -> blk.21.attn_k.weight                     | F16    | [4096, 4096]
model.layers.21.self_attn.v_proj.weight          -> blk.21.attn_v.weight                     | F16    | [4096, 4096]
model.layers.21.self_attn.o_proj.weight          -> blk.21.attn_output.weight                | F16    | [4096, 4096]
skipping tensor blk.21.attn_rot_embd
model.layers.21.mlp.gate_proj.weight             -> blk.21.ffn_gate.weight                   | F16    | [11008, 4096]
model.layers.21.mlp.up_proj.weight               -> blk.21.ffn_up.weight                     | F16    | [11008, 4096]
model.layers.21.mlp.down_proj.weight             -> blk.21.ffn_down.weight                   | F16    | [4096, 11008]
model.layers.21.input_layernorm.weight           -> blk.21.attn_norm.weight                  | F16    | [4096]
model.layers.21.post_attention_layernorm.weight  -> blk.21.ffn_norm.weight                   | F16    | [4096]
model.layers.22.self_attn.q_proj.weight          -> blk.22.attn_q.weight                     | F16    | [4096, 4096]
model.layers.22.self_attn.k_proj.weight          -> blk.22.attn_k.weight                     | F16    | [4096, 4096]
model.layers.22.self_attn.v_proj.weight          -> blk.22.attn_v.weight                     | F16    | [4096, 4096]
model.layers.22.self_attn.o_proj.weight          -> blk.22.attn_output.weight                | F16    | [4096, 4096]
skipping tensor blk.22.attn_rot_embd
model.layers.22.mlp.gate_proj.weight             -> blk.22.ffn_gate.weight                   | F16    | [11008, 4096]
model.layers.22.mlp.up_proj.weight               -> blk.22.ffn_up.weight                     | F16    | [11008, 4096]
model.layers.22.mlp.down_proj.weight             -> blk.22.ffn_down.weight                   | F16    | [4096, 11008]
model.layers.22.input_layernorm.weight           -> blk.22.attn_norm.weight                  | F16    | [4096]
model.layers.22.post_attention_layernorm.weight  -> blk.22.ffn_norm.weight                   | F16    | [4096]
model.layers.23.self_attn.q_proj.weight          -> blk.23.attn_q.weight                     | F16    | [4096, 4096]
model.layers.23.self_attn.k_proj.weight          -> blk.23.attn_k.weight                     | F16    | [4096, 4096]
model.layers.23.self_attn.v_proj.weight          -> blk.23.attn_v.weight                     | F16    | [4096, 4096]
model.layers.23.self_attn.o_proj.weight          -> blk.23.attn_output.weight                | F16    | [4096, 4096]
skipping tensor blk.23.attn_rot_embd
model.layers.23.mlp.gate_proj.weight             -> blk.23.ffn_gate.weight                   | F16    | [11008, 4096]
model.layers.23.mlp.up_proj.weight               -> blk.23.ffn_up.weight                     | F16    | [11008, 4096]
model.layers.23.mlp.down_proj.weight             -> blk.23.ffn_down.weight                   | F16    | [4096, 11008]
model.layers.23.input_layernorm.weight           -> blk.23.attn_norm.weight                  | F16    | [4096]
model.layers.23.post_attention_layernorm.weight  -> blk.23.ffn_norm.weight                   | F16    | [4096]
model.layers.24.self_attn.q_proj.weight          -> blk.24.attn_q.weight                     | F16    | [4096, 4096]
model.layers.24.self_attn.k_proj.weight          -> blk.24.attn_k.weight                     | F16    | [4096, 4096]
model.layers.24.self_attn.v_proj.weight          -> blk.24.attn_v.weight                     | F16    | [4096, 4096]
model.layers.24.self_attn.o_proj.weight          -> blk.24.attn_output.weight                | F16    | [4096, 4096]
skipping tensor blk.24.attn_rot_embd
model.layers.24.mlp.gate_proj.weight             -> blk.24.ffn_gate.weight                   | F16    | [11008, 4096]
model.layers.24.mlp.up_proj.weight               -> blk.24.ffn_up.weight                     | F16    | [11008, 4096]
model.layers.24.mlp.down_proj.weight             -> blk.24.ffn_down.weight                   | F16    | [4096, 11008]
model.layers.24.input_layernorm.weight           -> blk.24.attn_norm.weight                  | F16    | [4096]
model.layers.24.post_attention_layernorm.weight  -> blk.24.ffn_norm.weight                   | F16    | [4096]
model.layers.25.self_attn.q_proj.weight          -> blk.25.attn_q.weight                     | F16    | [4096, 4096]
model.layers.25.self_attn.k_proj.weight          -> blk.25.attn_k.weight                     | F16    | [4096, 4096]
model.layers.25.self_attn.v_proj.weight          -> blk.25.attn_v.weight                     | F16    | [4096, 4096]
model.layers.25.self_attn.o_proj.weight          -> blk.25.attn_output.weight                | F16    | [4096, 4096]
skipping tensor blk.25.attn_rot_embd
model.layers.25.mlp.gate_proj.weight             -> blk.25.ffn_gate.weight                   | F16    | [11008, 4096]
model.layers.25.mlp.up_proj.weight               -> blk.25.ffn_up.weight                     | F16    | [11008, 4096]
model.layers.25.mlp.down_proj.weight             -> blk.25.ffn_down.weight                   | F16    | [4096, 11008]
model.layers.25.input_layernorm.weight           -> blk.25.attn_norm.weight                  | F16    | [4096]
model.layers.25.post_attention_layernorm.weight  -> blk.25.ffn_norm.weight                   | F16    | [4096]
model.layers.26.self_attn.q_proj.weight          -> blk.26.attn_q.weight                     | F16    | [4096, 4096]
model.layers.26.self_attn.k_proj.weight          -> blk.26.attn_k.weight                     | F16    | [4096, 4096]
model.layers.26.self_attn.v_proj.weight          -> blk.26.attn_v.weight                     | F16    | [4096, 4096]
model.layers.26.self_attn.o_proj.weight          -> blk.26.attn_output.weight                | F16    | [4096, 4096]
skipping tensor blk.26.attn_rot_embd
model.layers.26.mlp.gate_proj.weight             -> blk.26.ffn_gate.weight                   | F16    | [11008, 4096]
model.layers.26.mlp.up_proj.weight               -> blk.26.ffn_up.weight                     | F16    | [11008, 4096]
model.layers.26.mlp.down_proj.weight             -> blk.26.ffn_down.weight                   | F16    | [4096, 11008]
model.layers.26.input_layernorm.weight           -> blk.26.attn_norm.weight                  | F16    | [4096]
model.layers.26.post_attention_layernorm.weight  -> blk.26.ffn_norm.weight                   | F16    | [4096]
model.layers.27.self_attn.q_proj.weight          -> blk.27.attn_q.weight                     | F16    | [4096, 4096]
model.layers.27.self_attn.k_proj.weight          -> blk.27.attn_k.weight                     | F16    | [4096, 4096]
model.layers.27.self_attn.v_proj.weight          -> blk.27.attn_v.weight                     | F16    | [4096, 4096]
model.layers.27.self_attn.o_proj.weight          -> blk.27.attn_output.weight                | F16    | [4096, 4096]
skipping tensor blk.27.attn_rot_embd
model.layers.27.mlp.gate_proj.weight             -> blk.27.ffn_gate.weight                   | F16    | [11008, 4096]
model.layers.27.mlp.up_proj.weight               -> blk.27.ffn_up.weight                     | F16    | [11008, 4096]
model.layers.27.mlp.down_proj.weight             -> blk.27.ffn_down.weight                   | F16    | [4096, 11008]
model.layers.27.input_layernorm.weight           -> blk.27.attn_norm.weight                  | F16    | [4096]
model.layers.27.post_attention_layernorm.weight  -> blk.27.ffn_norm.weight                   | F16    | [4096]
model.layers.28.self_attn.q_proj.weight          -> blk.28.attn_q.weight                     | F16    | [4096, 4096]
model.layers.28.self_attn.k_proj.weight          -> blk.28.attn_k.weight                     | F16    | [4096, 4096]
model.layers.28.self_attn.v_proj.weight          -> blk.28.attn_v.weight                     | F16    | [4096, 4096]
model.layers.28.self_attn.o_proj.weight          -> blk.28.attn_output.weight                | F16    | [4096, 4096]
skipping tensor blk.28.attn_rot_embd
model.layers.28.mlp.gate_proj.weight             -> blk.28.ffn_gate.weight                   | F16    | [11008, 4096]
model.layers.28.mlp.up_proj.weight               -> blk.28.ffn_up.weight                     | F16    | [11008, 4096]
model.layers.28.mlp.down_proj.weight             -> blk.28.ffn_down.weight                   | F16    | [4096, 11008]
model.layers.28.input_layernorm.weight           -> blk.28.attn_norm.weight                  | F16    | [4096]
model.layers.28.post_attention_layernorm.weight  -> blk.28.ffn_norm.weight                   | F16    | [4096]
model.layers.29.self_attn.q_proj.weight          -> blk.29.attn_q.weight                     | F16    | [4096, 4096]
model.layers.29.self_attn.k_proj.weight          -> blk.29.attn_k.weight                     | F16    | [4096, 4096]
model.layers.29.self_attn.v_proj.weight          -> blk.29.attn_v.weight                     | F16    | [4096, 4096]
model.layers.29.self_attn.o_proj.weight          -> blk.29.attn_output.weight                | F16    | [4096, 4096]
skipping tensor blk.29.attn_rot_embd
model.layers.29.mlp.gate_proj.weight             -> blk.29.ffn_gate.weight                   | F16    | [11008, 4096]
model.layers.29.mlp.up_proj.weight               -> blk.29.ffn_up.weight                     | F16    | [11008, 4096]
model.layers.29.mlp.down_proj.weight             -> blk.29.ffn_down.weight                   | F16    | [4096, 11008]
model.layers.29.input_layernorm.weight           -> blk.29.attn_norm.weight                  | F16    | [4096]
model.layers.29.post_attention_layernorm.weight  -> blk.29.ffn_norm.weight                   | F16    | [4096]
model.layers.30.self_attn.q_proj.weight          -> blk.30.attn_q.weight                     | F16    | [4096, 4096]
model.layers.30.self_attn.k_proj.weight          -> blk.30.attn_k.weight                     | F16    | [4096, 4096]
model.layers.30.self_attn.v_proj.weight          -> blk.30.attn_v.weight                     | F16    | [4096, 4096]
model.layers.30.self_attn.o_proj.weight          -> blk.30.attn_output.weight                | F16    | [4096, 4096]
skipping tensor blk.30.attn_rot_embd
model.layers.30.mlp.gate_proj.weight             -> blk.30.ffn_gate.weight                   | F16    | [11008, 4096]
model.layers.30.mlp.up_proj.weight               -> blk.30.ffn_up.weight                     | F16    | [11008, 4096]
model.layers.30.mlp.down_proj.weight             -> blk.30.ffn_down.weight                   | F16    | [4096, 11008]
model.layers.30.input_layernorm.weight           -> blk.30.attn_norm.weight                  | F16    | [4096]
model.layers.30.post_attention_layernorm.weight  -> blk.30.ffn_norm.weight                   | F16    | [4096]
model.layers.31.self_attn.q_proj.weight          -> blk.31.attn_q.weight                     | F16    | [4096, 4096]
model.layers.31.self_attn.k_proj.weight          -> blk.31.attn_k.weight                     | F16    | [4096, 4096]
model.layers.31.self_attn.v_proj.weight          -> blk.31.attn_v.weight                     | F16    | [4096, 4096]
model.layers.31.self_attn.o_proj.weight          -> blk.31.attn_output.weight                | F16    | [4096, 4096]
skipping tensor blk.31.attn_rot_embd
model.layers.31.mlp.gate_proj.weight             -> blk.31.ffn_gate.weight                   | F16    | [11008, 4096]
model.layers.31.mlp.up_proj.weight               -> blk.31.ffn_up.weight                     | F16    | [11008, 4096]
model.layers.31.mlp.down_proj.weight             -> blk.31.ffn_down.weight                   | F16    | [4096, 11008]
model.layers.31.input_layernorm.weight           -> blk.31.attn_norm.weight                  | F16    | [4096]
model.layers.31.post_attention_layernorm.weight  -> blk.31.ffn_norm.weight                   | F16    | [4096]
model.norm.weight                                -> output_norm.weight                       | F16    | [4096]
lm_head.weight                                   -> output.weight                            | F16    | [32000, 4096]
Writing ../llava-v1.5-7b/ggml-model-f16.gguf, format 1
Ignoring added_tokens.json since model matches vocab size without it.
gguf: This GGUF file is for Little Endian only
gguf: Setting special token type bos to 1
gguf: Setting special token type eos to 2
gguf: Setting special token type pad to 0
gguf: Setting add_bos_token to True
gguf: Setting add_eos_token to False
[  1/291] Writing tensor token_embd.weight                      | size  32000 x   4096  | type F16  | T+   1
[  2/291] Writing tensor blk.0.attn_q.weight                    | size   4096 x   4096  | type F16  | T+   1
[  3/291] Writing tensor blk.0.attn_k.weight                    | size   4096 x   4096  | type F16  | T+   1
[  4/291] Writing tensor blk.0.attn_v.weight                    | size   4096 x   4096  | type F16  | T+   1
[  5/291] Writing tensor blk.0.attn_output.weight               | size   4096 x   4096  | type F16  | T+   1
[  6/291] Writing tensor blk.0.ffn_gate.weight                  | size  11008 x   4096  | type F16  | T+   1
[  7/291] Writing tensor blk.0.ffn_up.weight                    | size  11008 x   4096  | type F16  | T+   1
[  8/291] Writing tensor blk.0.ffn_down.weight                  | size   4096 x  11008  | type F16  | T+   1
[  9/291] Writing tensor blk.0.attn_norm.weight                 | size   4096           | type F32  | T+   1
[ 10/291] Writing tensor blk.0.ffn_norm.weight                  | size   4096           | type F32  | T+   1
[ 11/291] Writing tensor blk.1.attn_q.weight                    | size   4096 x   4096  | type F16  | T+   1
[ 12/291] Writing tensor blk.1.attn_k.weight                    | size   4096 x   4096  | type F16  | T+   1
[ 13/291] Writing tensor blk.1.attn_v.weight                    | size   4096 x   4096  | type F16  | T+   1
[ 14/291] Writing tensor blk.1.attn_output.weight               | size   4096 x   4096  | type F16  | T+   1
[ 15/291] Writing tensor blk.1.ffn_gate.weight                  | size  11008 x   4096  | type F16  | T+   1
[ 16/291] Writing tensor blk.1.ffn_up.weight                    | size  11008 x   4096  | type F16  | T+   1
[ 17/291] Writing tensor blk.1.ffn_down.weight                  | size   4096 x  11008  | type F16  | T+   1
[ 18/291] Writing tensor blk.1.attn_norm.weight                 | size   4096           | type F32  | T+   1
[ 19/291] Writing tensor blk.1.ffn_norm.weight                  | size   4096           | type F32  | T+   1
[ 20/291] Writing tensor blk.2.attn_q.weight                    | size   4096 x   4096  | type F16  | T+   1
[ 21/291] Writing tensor blk.2.attn_k.weight                    | size   4096 x   4096  | type F16  | T+   1
[ 22/291] Writing tensor blk.2.attn_v.weight                    | size   4096 x   4096  | type F16  | T+   1
[ 23/291] Writing tensor blk.2.attn_output.weight               | size   4096 x   4096  | type F16  | T+   1
[ 24/291] Writing tensor blk.2.ffn_gate.weight                  | size  11008 x   4096  | type F16  | T+   1
[ 25/291] Writing tensor blk.2.ffn_up.weight                    | size  11008 x   4096  | type F16  | T+   2
[ 26/291] Writing tensor blk.2.ffn_down.weight                  | size   4096 x  11008  | type F16  | T+   2
[ 27/291] Writing tensor blk.2.attn_norm.weight                 | size   4096           | type F32  | T+   2
[ 28/291] Writing tensor blk.2.ffn_norm.weight                  | size   4096           | type F32  | T+   2
[ 29/291] Writing tensor blk.3.attn_q.weight                    | size   4096 x   4096  | type F16  | T+   2
[ 30/291] Writing tensor blk.3.attn_k.weight                    | size   4096 x   4096  | type F16  | T+   2
[ 31/291] Writing tensor blk.3.attn_v.weight                    | size   4096 x   4096  | type F16  | T+   2
[ 32/291] Writing tensor blk.3.attn_output.weight               | size   4096 x   4096  | type F16  | T+   2
[ 33/291] Writing tensor blk.3.ffn_gate.weight                  | size  11008 x   4096  | type F16  | T+   2
[ 34/291] Writing tensor blk.3.ffn_up.weight                    | size  11008 x   4096  | type F16  | T+   2
[ 35/291] Writing tensor blk.3.ffn_down.weight                  | size   4096 x  11008  | type F16  | T+   2
[ 36/291] Writing tensor blk.3.attn_norm.weight                 | size   4096           | type F32  | T+   2
[ 37/291] Writing tensor blk.3.ffn_norm.weight                  | size   4096           | type F32  | T+   2
[ 38/291] Writing tensor blk.4.attn_q.weight                    | size   4096 x   4096  | type F16  | T+   2
[ 39/291] Writing tensor blk.4.attn_k.weight                    | size   4096 x   4096  | type F16  | T+   2
[ 40/291] Writing tensor blk.4.attn_v.weight                    | size   4096 x   4096  | type F16  | T+   2
[ 41/291] Writing tensor blk.4.attn_output.weight               | size   4096 x   4096  | type F16  | T+   2
[ 42/291] Writing tensor blk.4.ffn_gate.weight                  | size  11008 x   4096  | type F16  | T+   2
[ 43/291] Writing tensor blk.4.ffn_up.weight                    | size  11008 x   4096  | type F16  | T+   2
[ 44/291] Writing tensor blk.4.ffn_down.weight                  | size   4096 x  11008  | type F16  | T+   3
[ 45/291] Writing tensor blk.4.attn_norm.weight                 | size   4096           | type F32  | T+   3
[ 46/291] Writing tensor blk.4.ffn_norm.weight                  | size   4096           | type F32  | T+   3
[ 47/291] Writing tensor blk.5.attn_q.weight                    | size   4096 x   4096  | type F16  | T+   3
[ 48/291] Writing tensor blk.5.attn_k.weight                    | size   4096 x   4096  | type F16  | T+   3
[ 49/291] Writing tensor blk.5.attn_v.weight                    | size   4096 x   4096  | type F16  | T+   3
[ 50/291] Writing tensor blk.5.attn_output.weight               | size   4096 x   4096  | type F16  | T+   3
[ 51/291] Writing tensor blk.5.ffn_gate.weight                  | size  11008 x   4096  | type F16  | T+   3
[ 52/291] Writing tensor blk.5.ffn_up.weight                    | size  11008 x   4096  | type F16  | T+   3
[ 53/291] Writing tensor blk.5.ffn_down.weight                  | size   4096 x  11008  | type F16  | T+   3
[ 54/291] Writing tensor blk.5.attn_norm.weight                 | size   4096           | type F32  | T+   3
[ 55/291] Writing tensor blk.5.ffn_norm.weight                  | size   4096           | type F32  | T+   3
[ 56/291] Writing tensor blk.6.attn_q.weight                    | size   4096 x   4096  | type F16  | T+   3
[ 57/291] Writing tensor blk.6.attn_k.weight                    | size   4096 x   4096  | type F16  | T+   3
[ 58/291] Writing tensor blk.6.attn_v.weight                    | size   4096 x   4096  | type F16  | T+   3
[ 59/291] Writing tensor blk.6.attn_output.weight               | size   4096 x   4096  | type F16  | T+   3
[ 60/291] Writing tensor blk.6.ffn_gate.weight                  | size  11008 x   4096  | type F16  | T+   4
[ 61/291] Writing tensor blk.6.ffn_up.weight                    | size  11008 x   4096  | type F16  | T+   4
[ 62/291] Writing tensor blk.6.ffn_down.weight                  | size   4096 x  11008  | type F16  | T+   4
[ 63/291] Writing tensor blk.6.attn_norm.weight                 | size   4096           | type F32  | T+   4
[ 64/291] Writing tensor blk.6.ffn_norm.weight                  | size   4096           | type F32  | T+   4
[ 65/291] Writing tensor blk.7.attn_q.weight                    | size   4096 x   4096  | type F16  | T+   4
[ 66/291] Writing tensor blk.7.attn_k.weight                    | size   4096 x   4096  | type F16  | T+   4
[ 67/291] Writing tensor blk.7.attn_v.weight                    | size   4096 x   4096  | type F16  | T+   4
[ 68/291] Writing tensor blk.7.attn_output.weight               | size   4096 x   4096  | type F16  | T+   4
[ 69/291] Writing tensor blk.7.ffn_gate.weight                  | size  11008 x   4096  | type F16  | T+   4
[ 70/291] Writing tensor blk.7.ffn_up.weight                    | size  11008 x   4096  | type F16  | T+   4
[ 71/291] Writing tensor blk.7.ffn_down.weight                  | size   4096 x  11008  | type F16  | T+   5
[ 72/291] Writing tensor blk.7.attn_norm.weight                 | size   4096           | type F32  | T+   5
[ 73/291] Writing tensor blk.7.ffn_norm.weight                  | size   4096           | type F32  | T+   5
[ 74/291] Writing tensor blk.8.attn_q.weight                    | size   4096 x   4096  | type F16  | T+   5
[ 75/291] Writing tensor blk.8.attn_k.weight                    | size   4096 x   4096  | type F16  | T+   5
[ 76/291] Writing tensor blk.8.attn_v.weight                    | size   4096 x   4096  | type F16  | T+   5
[ 77/291] Writing tensor blk.8.attn_output.weight               | size   4096 x   4096  | type F16  | T+   5
[ 78/291] Writing tensor blk.8.ffn_gate.weight                  | size  11008 x   4096  | type F16  | T+   5
[ 79/291] Writing tensor blk.8.ffn_up.weight                    | size  11008 x   4096  | type F16  | T+   5
[ 80/291] Writing tensor blk.8.ffn_down.weight                  | size   4096 x  11008  | type F16  | T+   6
[ 81/291] Writing tensor blk.8.attn_norm.weight                 | size   4096           | type F32  | T+   6
[ 82/291] Writing tensor blk.8.ffn_norm.weight                  | size   4096           | type F32  | T+   6
[ 83/291] Writing tensor blk.9.attn_q.weight                    | size   4096 x   4096  | type F16  | T+   6
[ 84/291] Writing tensor blk.9.attn_k.weight                    | size   4096 x   4096  | type F16  | T+   6
[ 85/291] Writing tensor blk.9.attn_v.weight                    | size   4096 x   4096  | type F16  | T+   6
[ 86/291] Writing tensor blk.9.attn_output.weight               | size   4096 x   4096  | type F16  | T+   6
[ 87/291] Writing tensor blk.9.ffn_gate.weight                  | size  11008 x   4096  | type F16  | T+   6
[ 88/291] Writing tensor blk.9.ffn_up.weight                    | size  11008 x   4096  | type F16  | T+   6
[ 89/291] Writing tensor blk.9.ffn_down.weight                  | size   4096 x  11008  | type F16  | T+   6
[ 90/291] Writing tensor blk.9.attn_norm.weight                 | size   4096           | type F32  | T+   6
[ 91/291] Writing tensor blk.9.ffn_norm.weight                  | size   4096           | type F32  | T+   6
[ 92/291] Writing tensor blk.10.attn_q.weight                   | size   4096 x   4096  | type F16  | T+   6
[ 93/291] Writing tensor blk.10.attn_k.weight                   | size   4096 x   4096  | type F16  | T+   6
[ 94/291] Writing tensor blk.10.attn_v.weight                   | size   4096 x   4096  | type F16  | T+   6
[ 95/291] Writing tensor blk.10.attn_output.weight              | size   4096 x   4096  | type F16  | T+   7
[ 96/291] Writing tensor blk.10.ffn_gate.weight                 | size  11008 x   4096  | type F16  | T+   7
[ 97/291] Writing tensor blk.10.ffn_up.weight                   | size  11008 x   4096  | type F16  | T+   7
[ 98/291] Writing tensor blk.10.ffn_down.weight                 | size   4096 x  11008  | type F16  | T+   7
[ 99/291] Writing tensor blk.10.attn_norm.weight                | size   4096           | type F32  | T+   7
[100/291] Writing tensor blk.10.ffn_norm.weight                 | size   4096           | type F32  | T+   7
[101/291] Writing tensor blk.11.attn_q.weight                   | size   4096 x   4096  | type F16  | T+   7
[102/291] Writing tensor blk.11.attn_k.weight                   | size   4096 x   4096  | type F16  | T+   7
[103/291] Writing tensor blk.11.attn_v.weight                   | size   4096 x   4096  | type F16  | T+   7
[104/291] Writing tensor blk.11.attn_output.weight              | size   4096 x   4096  | type F16  | T+   7
[105/291] Writing tensor blk.11.ffn_gate.weight                 | size  11008 x   4096  | type F16  | T+   7
[106/291] Writing tensor blk.11.ffn_up.weight                   | size  11008 x   4096  | type F16  | T+   7
[107/291] Writing tensor blk.11.ffn_down.weight                 | size   4096 x  11008  | type F16  | T+   8
[108/291] Writing tensor blk.11.attn_norm.weight                | size   4096           | type F32  | T+   8
[109/291] Writing tensor blk.11.ffn_norm.weight                 | size   4096           | type F32  | T+   8
[110/291] Writing tensor blk.12.attn_q.weight                   | size   4096 x   4096  | type F16  | T+   8
[111/291] Writing tensor blk.12.attn_k.weight                   | size   4096 x   4096  | type F16  | T+   8
[112/291] Writing tensor blk.12.attn_v.weight                   | size   4096 x   4096  | type F16  | T+   8
[113/291] Writing tensor blk.12.attn_output.weight              | size   4096 x   4096  | type F16  | T+   8
[114/291] Writing tensor blk.12.ffn_gate.weight                 | size  11008 x   4096  | type F16  | T+   8
[115/291] Writing tensor blk.12.ffn_up.weight                   | size  11008 x   4096  | type F16  | T+   8
[116/291] Writing tensor blk.12.ffn_down.weight                 | size   4096 x  11008  | type F16  | T+   8
[117/291] Writing tensor blk.12.attn_norm.weight                | size   4096           | type F32  | T+   8
[118/291] Writing tensor blk.12.ffn_norm.weight                 | size   4096           | type F32  | T+   8
[119/291] Writing tensor blk.13.attn_q.weight                   | size   4096 x   4096  | type F16  | T+   8
[120/291] Writing tensor blk.13.attn_k.weight                   | size   4096 x   4096  | type F16  | T+   8
[121/291] Writing tensor blk.13.attn_v.weight                   | size   4096 x   4096  | type F16  | T+   8
[122/291] Writing tensor blk.13.attn_output.weight              | size   4096 x   4096  | type F16  | T+   8
[123/291] Writing tensor blk.13.ffn_gate.weight                 | size  11008 x   4096  | type F16  | T+   8
[124/291] Writing tensor blk.13.ffn_up.weight                   | size  11008 x   4096  | type F16  | T+   8
[125/291] Writing tensor blk.13.ffn_down.weight                 | size   4096 x  11008  | type F16  | T+   9
[126/291] Writing tensor blk.13.attn_norm.weight                | size   4096           | type F32  | T+   9
[127/291] Writing tensor blk.13.ffn_norm.weight                 | size   4096           | type F32  | T+   9
[128/291] Writing tensor blk.14.attn_q.weight                   | size   4096 x   4096  | type F16  | T+   9
[129/291] Writing tensor blk.14.attn_k.weight                   | size   4096 x   4096  | type F16  | T+   9
[130/291] Writing tensor blk.14.attn_v.weight                   | size   4096 x   4096  | type F16  | T+   9
[131/291] Writing tensor blk.14.attn_output.weight              | size   4096 x   4096  | type F16  | T+   9
[132/291] Writing tensor blk.14.ffn_gate.weight                 | size  11008 x   4096  | type F16  | T+   9
[133/291] Writing tensor blk.14.ffn_up.weight                   | size  11008 x   4096  | type F16  | T+   9
[134/291] Writing tensor blk.14.ffn_down.weight                 | size   4096 x  11008  | type F16  | T+   9
[135/291] Writing tensor blk.14.attn_norm.weight                | size   4096           | type F32  | T+   9
[136/291] Writing tensor blk.14.ffn_norm.weight                 | size   4096           | type F32  | T+   9
[137/291] Writing tensor blk.15.attn_q.weight                   | size   4096 x   4096  | type F16  | T+   9
[138/291] Writing tensor blk.15.attn_k.weight                   | size   4096 x   4096  | type F16  | T+   9
[139/291] Writing tensor blk.15.attn_v.weight                   | size   4096 x   4096  | type F16  | T+   9
[140/291] Writing tensor blk.15.attn_output.weight              | size   4096 x   4096  | type F16  | T+   9
[141/291] Writing tensor blk.15.ffn_gate.weight                 | size  11008 x   4096  | type F16  | T+  10
[142/291] Writing tensor blk.15.ffn_up.weight                   | size  11008 x   4096  | type F16  | T+  10
[143/291] Writing tensor blk.15.ffn_down.weight                 | size   4096 x  11008  | type F16  | T+  10
[144/291] Writing tensor blk.15.attn_norm.weight                | size   4096           | type F32  | T+  10
[145/291] Writing tensor blk.15.ffn_norm.weight                 | size   4096           | type F32  | T+  10
[146/291] Writing tensor blk.16.attn_q.weight                   | size   4096 x   4096  | type F16  | T+  10
[147/291] Writing tensor blk.16.attn_k.weight                   | size   4096 x   4096  | type F16  | T+  10
[148/291] Writing tensor blk.16.attn_v.weight                   | size   4096 x   4096  | type F16  | T+  10
[149/291] Writing tensor blk.16.attn_output.weight              | size   4096 x   4096  | type F16  | T+  10
[150/291] Writing tensor blk.16.ffn_gate.weight                 | size  11008 x   4096  | type F16  | T+  10
[151/291] Writing tensor blk.16.ffn_up.weight                   | size  11008 x   4096  | type F16  | T+  10
[152/291] Writing tensor blk.16.ffn_down.weight                 | size   4096 x  11008  | type F16  | T+  11
[153/291] Writing tensor blk.16.attn_norm.weight                | size   4096           | type F32  | T+  11
[154/291] Writing tensor blk.16.ffn_norm.weight                 | size   4096           | type F32  | T+  11
[155/291] Writing tensor blk.17.attn_q.weight                   | size   4096 x   4096  | type F16  | T+  11
[156/291] Writing tensor blk.17.attn_k.weight                   | size   4096 x   4096  | type F16  | T+  11
[157/291] Writing tensor blk.17.attn_v.weight                   | size   4096 x   4096  | type F16  | T+  11
[158/291] Writing tensor blk.17.attn_output.weight              | size   4096 x   4096  | type F16  | T+  11
[159/291] Writing tensor blk.17.ffn_gate.weight                 | size  11008 x   4096  | type F16  | T+  11
[160/291] Writing tensor blk.17.ffn_up.weight                   | size  11008 x   4096  | type F16  | T+  11
[161/291] Writing tensor blk.17.ffn_down.weight                 | size   4096 x  11008  | type F16  | T+  11
[162/291] Writing tensor blk.17.attn_norm.weight                | size   4096           | type F32  | T+  11
[163/291] Writing tensor blk.17.ffn_norm.weight                 | size   4096           | type F32  | T+  11
[164/291] Writing tensor blk.18.attn_q.weight                   | size   4096 x   4096  | type F16  | T+  11
[165/291] Writing tensor blk.18.attn_k.weight                   | size   4096 x   4096  | type F16  | T+  11
[166/291] Writing tensor blk.18.attn_v.weight                   | size   4096 x   4096  | type F16  | T+  11
[167/291] Writing tensor blk.18.attn_output.weight              | size   4096 x   4096  | type F16  | T+  11
[168/291] Writing tensor blk.18.ffn_gate.weight                 | size  11008 x   4096  | type F16  | T+  12
[169/291] Writing tensor blk.18.ffn_up.weight                   | size  11008 x   4096  | type F16  | T+  12
[170/291] Writing tensor blk.18.ffn_down.weight                 | size   4096 x  11008  | type F16  | T+  12
[171/291] Writing tensor blk.18.attn_norm.weight                | size   4096           | type F32  | T+  12
[172/291] Writing tensor blk.18.ffn_norm.weight                 | size   4096           | type F32  | T+  12
[173/291] Writing tensor blk.19.attn_q.weight                   | size   4096 x   4096  | type F16  | T+  12
[174/291] Writing tensor blk.19.attn_k.weight                   | size   4096 x   4096  | type F16  | T+  12
[175/291] Writing tensor blk.19.attn_v.weight                   | size   4096 x   4096  | type F16  | T+  12
[176/291] Writing tensor blk.19.attn_output.weight              | size   4096 x   4096  | type F16  | T+  12
[177/291] Writing tensor blk.19.ffn_gate.weight                 | size  11008 x   4096  | type F16  | T+  12
[178/291] Writing tensor blk.19.ffn_up.weight                   | size  11008 x   4096  | type F16  | T+  12
[179/291] Writing tensor blk.19.ffn_down.weight                 | size   4096 x  11008  | type F16  | T+  13
[180/291] Writing tensor blk.19.attn_norm.weight                | size   4096           | type F32  | T+  13
[181/291] Writing tensor blk.19.ffn_norm.weight                 | size   4096           | type F32  | T+  13
[182/291] Writing tensor blk.20.attn_q.weight                   | size   4096 x   4096  | type F16  | T+  13
[183/291] Writing tensor blk.20.attn_k.weight                   | size   4096 x   4096  | type F16  | T+  13
[184/291] Writing tensor blk.20.attn_v.weight                   | size   4096 x   4096  | type F16  | T+  13
[185/291] Writing tensor blk.20.attn_output.weight              | size   4096 x   4096  | type F16  | T+  13
[186/291] Writing tensor blk.20.ffn_gate.weight                 | size  11008 x   4096  | type F16  | T+  13
[187/291] Writing tensor blk.20.ffn_up.weight                   | size  11008 x   4096  | type F16  | T+  13
[188/291] Writing tensor blk.20.ffn_down.weight                 | size   4096 x  11008  | type F16  | T+  13
[189/291] Writing tensor blk.20.attn_norm.weight                | size   4096           | type F32  | T+  13
[190/291] Writing tensor blk.20.ffn_norm.weight                 | size   4096           | type F32  | T+  13
[191/291] Writing tensor blk.21.attn_q.weight                   | size   4096 x   4096  | type F16  | T+  13
[192/291] Writing tensor blk.21.attn_k.weight                   | size   4096 x   4096  | type F16  | T+  13
[193/291] Writing tensor blk.21.attn_v.weight                   | size   4096 x   4096  | type F16  | T+  13
[194/291] Writing tensor blk.21.attn_output.weight              | size   4096 x   4096  | type F16  | T+  13
[195/291] Writing tensor blk.21.ffn_gate.weight                 | size  11008 x   4096  | type F16  | T+  13
[196/291] Writing tensor blk.21.ffn_up.weight                   | size  11008 x   4096  | type F16  | T+  14
[197/291] Writing tensor blk.21.ffn_down.weight                 | size   4096 x  11008  | type F16  | T+  14
[198/291] Writing tensor blk.21.attn_norm.weight                | size   4096           | type F32  | T+  14
[199/291] Writing tensor blk.21.ffn_norm.weight                 | size   4096           | type F32  | T+  14
[200/291] Writing tensor blk.22.attn_q.weight                   | size   4096 x   4096  | type F16  | T+  14
[201/291] Writing tensor blk.22.attn_k.weight                   | size   4096 x   4096  | type F16  | T+  14
[202/291] Writing tensor blk.22.attn_v.weight                   | size   4096 x   4096  | type F16  | T+  14
[203/291] Writing tensor blk.22.attn_output.weight              | size   4096 x   4096  | type F16  | T+  14
[204/291] Writing tensor blk.22.ffn_gate.weight                 | size  11008 x   4096  | type F16  | T+  14
[205/291] Writing tensor blk.22.ffn_up.weight                   | size  11008 x   4096  | type F16  | T+  14
[206/291] Writing tensor blk.22.ffn_down.weight                 | size   4096 x  11008  | type F16  | T+  14
[207/291] Writing tensor blk.22.attn_norm.weight                | size   4096           | type F32  | T+  14
[208/291] Writing tensor blk.22.ffn_norm.weight                 | size   4096           | type F32  | T+  14
[209/291] Writing tensor blk.23.attn_q.weight                   | size   4096 x   4096  | type F16  | T+  14
[210/291] Writing tensor blk.23.attn_k.weight                   | size   4096 x   4096  | type F16  | T+  14
[211/291] Writing tensor blk.23.attn_v.weight                   | size   4096 x   4096  | type F16  | T+  15
[212/291] Writing tensor blk.23.attn_output.weight              | size   4096 x   4096  | type F16  | T+  15
[213/291] Writing tensor blk.23.ffn_gate.weight                 | size  11008 x   4096  | type F16  | T+  15
[214/291] Writing tensor blk.23.ffn_up.weight                   | size  11008 x   4096  | type F16  | T+  15
[215/291] Writing tensor blk.23.ffn_down.weight                 | size   4096 x  11008  | type F16  | T+  15
[216/291] Writing tensor blk.23.attn_norm.weight                | size   4096           | type F32  | T+  15
[217/291] Writing tensor blk.23.ffn_norm.weight                 | size   4096           | type F32  | T+  15
[218/291] Writing tensor blk.24.attn_q.weight                   | size   4096 x   4096  | type F16  | T+  15
[219/291] Writing tensor blk.24.attn_k.weight                   | size   4096 x   4096  | type F16  | T+  15
[220/291] Writing tensor blk.24.attn_v.weight                   | size   4096 x   4096  | type F16  | T+  15
[221/291] Writing tensor blk.24.attn_output.weight              | size   4096 x   4096  | type F16  | T+  15
[222/291] Writing tensor blk.24.ffn_gate.weight                 | size  11008 x   4096  | type F16  | T+  15
[223/291] Writing tensor blk.24.ffn_up.weight                   | size  11008 x   4096  | type F16  | T+  15
[224/291] Writing tensor blk.24.ffn_down.weight                 | size   4096 x  11008  | type F16  | T+  15
[225/291] Writing tensor blk.24.attn_norm.weight                | size   4096           | type F32  | T+  15
[226/291] Writing tensor blk.24.ffn_norm.weight                 | size   4096           | type F32  | T+  15
[227/291] Writing tensor blk.25.attn_q.weight                   | size   4096 x   4096  | type F16  | T+  15
[228/291] Writing tensor blk.25.attn_k.weight                   | size   4096 x   4096  | type F16  | T+  15
[229/291] Writing tensor blk.25.attn_v.weight                   | size   4096 x   4096  | type F16  | T+  16
[230/291] Writing tensor blk.25.attn_output.weight              | size   4096 x   4096  | type F16  | T+  16
[231/291] Writing tensor blk.25.ffn_gate.weight                 | size  11008 x   4096  | type F16  | T+  16
[232/291] Writing tensor blk.25.ffn_up.weight                   | size  11008 x   4096  | type F16  | T+  16
[233/291] Writing tensor blk.25.ffn_down.weight                 | size   4096 x  11008  | type F16  | T+  16
[234/291] Writing tensor blk.25.attn_norm.weight                | size   4096           | type F32  | T+  16
[235/291] Writing tensor blk.25.ffn_norm.weight                 | size   4096           | type F32  | T+  16
[236/291] Writing tensor blk.26.attn_q.weight                   | size   4096 x   4096  | type F16  | T+  16
[237/291] Writing tensor blk.26.attn_k.weight                   | size   4096 x   4096  | type F16  | T+  16
[238/291] Writing tensor blk.26.attn_v.weight                   | size   4096 x   4096  | type F16  | T+  16
[239/291] Writing tensor blk.26.attn_output.weight              | size   4096 x   4096  | type F16  | T+  16
[240/291] Writing tensor blk.26.ffn_gate.weight                 | size  11008 x   4096  | type F16  | T+  16
[241/291] Writing tensor blk.26.ffn_up.weight                   | size  11008 x   4096  | type F16  | T+  17
[242/291] Writing tensor blk.26.ffn_down.weight                 | size   4096 x  11008  | type F16  | T+  17
[243/291] Writing tensor blk.26.attn_norm.weight                | size   4096           | type F32  | T+  17
[244/291] Writing tensor blk.26.ffn_norm.weight                 | size   4096           | type F32  | T+  17
[245/291] Writing tensor blk.27.attn_q.weight                   | size   4096 x   4096  | type F16  | T+  17
[246/291] Writing tensor blk.27.attn_k.weight                   | size   4096 x   4096  | type F16  | T+  17
[247/291] Writing tensor blk.27.attn_v.weight                   | size   4096 x   4096  | type F16  | T+  17
[248/291] Writing tensor blk.27.attn_output.weight              | size   4096 x   4096  | type F16  | T+  17
[249/291] Writing tensor blk.27.ffn_gate.weight                 | size  11008 x   4096  | type F16  | T+  17
[250/291] Writing tensor blk.27.ffn_up.weight                   | size  11008 x   4096  | type F16  | T+  17
[251/291] Writing tensor blk.27.ffn_down.weight                 | size   4096 x  11008  | type F16  | T+  17
[252/291] Writing tensor blk.27.attn_norm.weight                | size   4096           | type F32  | T+  17
[253/291] Writing tensor blk.27.ffn_norm.weight                 | size   4096           | type F32  | T+  17
[254/291] Writing tensor blk.28.attn_q.weight                   | size   4096 x   4096  | type F16  | T+  18
[255/291] Writing tensor blk.28.attn_k.weight                   | size   4096 x   4096  | type F16  | T+  18
[256/291] Writing tensor blk.28.attn_v.weight                   | size   4096 x   4096  | type F16  | T+  18
[257/291] Writing tensor blk.28.attn_output.weight              | size   4096 x   4096  | type F16  | T+  18
[258/291] Writing tensor blk.28.ffn_gate.weight                 | size  11008 x   4096  | type F16  | T+  18
[259/291] Writing tensor blk.28.ffn_up.weight                   | size  11008 x   4096  | type F16  | T+  18
[260/291] Writing tensor blk.28.ffn_down.weight                 | size   4096 x  11008  | type F16  | T+  18
[261/291] Writing tensor blk.28.attn_norm.weight                | size   4096           | type F32  | T+  18
[262/291] Writing tensor blk.28.ffn_norm.weight                 | size   4096           | type F32  | T+  18
[263/291] Writing tensor blk.29.attn_q.weight                   | size   4096 x   4096  | type F16  | T+  18
[264/291] Writing tensor blk.29.attn_k.weight                   | size   4096 x   4096  | type F16  | T+  18
[265/291] Writing tensor blk.29.attn_v.weight                   | size   4096 x   4096  | type F16  | T+  18
[266/291] Writing tensor blk.29.attn_output.weight              | size   4096 x   4096  | type F16  | T+  18
[267/291] Writing tensor blk.29.ffn_gate.weight                 | size  11008 x   4096  | type F16  | T+  18
[268/291] Writing tensor blk.29.ffn_up.weight                   | size  11008 x   4096  | type F16  | T+  18
[269/291] Writing tensor blk.29.ffn_down.weight                 | size   4096 x  11008  | type F16  | T+  19
[270/291] Writing tensor blk.29.attn_norm.weight                | size   4096           | type F32  | T+  19
[271/291] Writing tensor blk.29.ffn_norm.weight                 | size   4096           | type F32  | T+  19
[272/291] Writing tensor blk.30.attn_q.weight                   | size   4096 x   4096  | type F16  | T+  19
[273/291] Writing tensor blk.30.attn_k.weight                   | size   4096 x   4096  | type F16  | T+  19
[274/291] Writing tensor blk.30.attn_v.weight                   | size   4096 x   4096  | type F16  | T+  19
[275/291] Writing tensor blk.30.attn_output.weight              | size   4096 x   4096  | type F16  | T+  19
[276/291] Writing tensor blk.30.ffn_gate.weight                 | size  11008 x   4096  | type F16  | T+  19
[277/291] Writing tensor blk.30.ffn_up.weight                   | size  11008 x   4096  | type F16  | T+  19
[278/291] Writing tensor blk.30.ffn_down.weight                 | size   4096 x  11008  | type F16  | T+  19
[279/291] Writing tensor blk.30.attn_norm.weight                | size   4096           | type F32  | T+  19
[280/291] Writing tensor blk.30.ffn_norm.weight                 | size   4096           | type F32  | T+  19
[281/291] Writing tensor blk.31.attn_q.weight                   | size   4096 x   4096  | type F16  | T+  19
[282/291] Writing tensor blk.31.attn_k.weight                   | size   4096 x   4096  | type F16  | T+  19
[283/291] Writing tensor blk.31.attn_v.weight                   | size   4096 x   4096  | type F16  | T+  19
[284/291] Writing tensor blk.31.attn_output.weight              | size   4096 x   4096  | type F16  | T+  19
[285/291] Writing tensor blk.31.ffn_gate.weight                 | size  11008 x   4096  | type F16  | T+  19
[286/291] Writing tensor blk.31.ffn_up.weight                   | size  11008 x   4096  | type F16  | T+  20
[287/291] Writing tensor blk.31.ffn_down.weight                 | size   4096 x  11008  | type F16  | T+  20
[288/291] Writing tensor blk.31.attn_norm.weight                | size   4096           | type F32  | T+  20
[289/291] Writing tensor blk.31.ffn_norm.weight                 | size   4096           | type F32  | T+  20
[290/291] Writing tensor output_norm.weight                     | size   4096           | type F32  | T+  20
[291/291] Writing tensor output.weight                          | size  32000 x   4096  | type F16  | T+  20
Wrote ../llava-v1.5-7b/ggml-model-f16.gguf
```

```console
(llava-venv) $ ./llava-cli --n-gpu-layers 27 -m ../llava-v1.5-7b/ggml-model-f16.gguf --mmproj ../llava-v1.5-7b/mmproj-model-f16.gguf --image /home/danielbevenius/Downloads/im-an-expert.jpg
ggml_init_cublas: GGML_CUDA_FORCE_MMQ:   no
ggml_init_cublas: CUDA_USE_TENSOR_CORES: yes
ggml_init_cublas: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4070, compute capability 8.9, VMM: yes
clip_model_load: model name:   openai/clip-vit-large-patch14-336
clip_model_load: description:  image encoder for LLaVA
clip_model_load: GGUF version: 3
clip_model_load: alignment:    32
clip_model_load: n_tensors:    377
clip_model_load: n_kv:         19
clip_model_load: ftype:        f16

clip_model_load: loaded meta data with 19 key-value pairs and 377 tensors from ../llava-v1.5-7b/mmproj-model-f16.gguf
clip_model_load: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
clip_model_load: - kv   0:                       general.architecture str              = clip
clip_model_load: - kv   1:                      clip.has_text_encoder bool             = false
clip_model_load: - kv   2:                    clip.has_vision_encoder bool             = true
clip_model_load: - kv   3:                   clip.has_llava_projector bool             = true
clip_model_load: - kv   4:                          general.file_type u32              = 1
clip_model_load: - kv   5:                               general.name str              = openai/clip-vit-large-patch14-336
clip_model_load: - kv   6:                        general.description str              = image encoder for LLaVA
clip_model_load: - kv   7:                        clip.projector_type str              = mlp
clip_model_load: - kv   8:                     clip.vision.image_size u32              = 336
clip_model_load: - kv   9:                     clip.vision.patch_size u32              = 14
clip_model_load: - kv  10:               clip.vision.embedding_length u32              = 1024
clip_model_load: - kv  11:            clip.vision.feed_forward_length u32              = 4096
clip_model_load: - kv  12:                 clip.vision.projection_dim u32              = 768
clip_model_load: - kv  13:           clip.vision.attention.head_count u32              = 16
clip_model_load: - kv  14:   clip.vision.attention.layer_norm_epsilon f32              = 0.000010
clip_model_load: - kv  15:                    clip.vision.block_count u32              = 23
clip_model_load: - kv  16:                     clip.vision.image_mean arr[f32,3]       = [0.481455, 0.457828, 0.408211]
clip_model_load: - kv  17:                      clip.vision.image_std arr[f32,3]       = [0.268630, 0.261303, 0.275777]
clip_model_load: - kv  18:                              clip.use_gelu bool             = false
clip_model_load: - type  f32:  235 tensors
clip_model_load: - type  f16:  142 tensors
clip_model_load: CLIP using CUDA backend
clip_model_load: text_encoder:   0
clip_model_load: vision_encoder: 1
clip_model_load: llava_projector:  1
clip_model_load: model size:     595.53 MB
clip_model_load: metadata size:  0.14 MB
clip_model_load: params backend buffer size =  595.53 MB (377 tensors)
clip_model_load: compute allocated memory: 36.18 MB
llama_model_loader: loaded meta data with 20 key-value pairs and 291 tensors from ../llava-v1.5-7b/ggml-model-f16.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_model_loader: - kv   1:                               general.name str              = LLaMA v2
llama_model_loader: - kv   2:                       llama.context_length u32              = 4096
llama_model_loader: - kv   3:                     llama.embedding_length u32              = 4096
llama_model_loader: - kv   4:                          llama.block_count u32              = 32
llama_model_loader: - kv   5:                  llama.feed_forward_length u32              = 11008
llama_model_loader: - kv   6:                 llama.rope.dimension_count u32              = 128
llama_model_loader: - kv   7:                 llama.attention.head_count u32              = 32
llama_model_loader: - kv   8:              llama.attention.head_count_kv u32              = 32
llama_model_loader: - kv   9:     llama.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  10:                          general.file_type u32              = 1
llama_model_loader: - kv  11:                       tokenizer.ggml.model str              = llama
llama_model_loader: - kv  12:                      tokenizer.ggml.tokens arr[str,32000]   = ["<unk>", "<s>", "</s>", "<0x00>", "<...
llama_model_loader: - kv  13:                      tokenizer.ggml.scores arr[f32,32000]   = [0.000000, 0.000000, 0.000000, 0.0000...
llama_model_loader: - kv  14:                  tokenizer.ggml.token_type arr[i32,32000]   = [2, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, ...
llama_model_loader: - kv  15:                tokenizer.ggml.bos_token_id u32              = 1
llama_model_loader: - kv  16:                tokenizer.ggml.eos_token_id u32              = 2
llama_model_loader: - kv  17:            tokenizer.ggml.padding_token_id u32              = 0
llama_model_loader: - kv  18:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  19:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - type  f32:   65 tensors
llama_model_loader: - type  f16:  226 tensors
llm_load_vocab: special tokens definition check successful ( 259/32000 ).
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = llama
llm_load_print_meta: vocab type       = SPM
llm_load_print_meta: n_vocab          = 32000
llm_load_print_meta: n_merges         = 0
llm_load_print_meta: n_ctx_train      = 4096
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_head           = 32
llm_load_print_meta: n_head_kv        = 32
llm_load_print_meta: n_layer          = 32
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 1
llm_load_print_meta: n_embd_k_gqa     = 4096
llm_load_print_meta: n_embd_v_gqa     = 4096
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-05
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: n_ff             = 11008
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 10000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_yarn_orig_ctx  = 4096
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: model type       = 7B
llm_load_print_meta: model ftype      = F16
llm_load_print_meta: model params     = 6.74 B
llm_load_print_meta: model size       = 12.55 GiB (16.00 BPW) 
llm_load_print_meta: general.name     = LLaMA v2
llm_load_print_meta: BOS token        = 1 '<s>'
llm_load_print_meta: EOS token        = 2 '</s>'
llm_load_print_meta: UNK token        = 0 '<unk>'
llm_load_print_meta: PAD token        = 0 '<unk>'
llm_load_print_meta: LF token         = 13 '<0x0A>'
llm_load_tensors: ggml ctx size =    0.22 MiB
llm_load_tensors: offloading 27 repeating layers to GPU
llm_load_tensors: offloaded 27/33 layers to GPU
llm_load_tensors:        CPU buffer size = 12853.02 MiB
llm_load_tensors:      CUDA0 buffer size = 10422.84 MiB
...................................................................................................
llama_new_context_with_model: n_ctx      = 2048
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:  CUDA_Host KV buffer size =   160.00 MiB
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 864.00 MiB on device 0: cudaMalloc failed: out of memory
llama_kv_cache_init: failed to allocate buffer for kv cache
llama_new_context_with_model: llama_kv_cache_init() failed for self-attention cache
llava_init: error: failed to create the llama_context
main: error: failed to init llava
(llava-venv) $ ./llava-cli --n-gpu-layers 20 -m ../llava-v1.5-7b/ggml-model-f16.gguf --mmproj ../llava-v1.5-7b/mmproj-model-f16.gguf --image /home/danielbevenius/Downloads/im-an-expert.jpg
ggml_init_cublas: GGML_CUDA_FORCE_MMQ:   no
ggml_init_cublas: CUDA_USE_TENSOR_CORES: yes
ggml_init_cublas: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4070, compute capability 8.9, VMM: yes
clip_model_load: model name:   openai/clip-vit-large-patch14-336
clip_model_load: description:  image encoder for LLaVA
clip_model_load: GGUF version: 3
clip_model_load: alignment:    32
clip_model_load: n_tensors:    377
clip_model_load: n_kv:         19
clip_model_load: ftype:        f16

clip_model_load: loaded meta data with 19 key-value pairs and 377 tensors from ../llava-v1.5-7b/mmproj-model-f16.gguf
clip_model_load: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
clip_model_load: - kv   0:                       general.architecture str              = clip
clip_model_load: - kv   1:                      clip.has_text_encoder bool             = false
clip_model_load: - kv   2:                    clip.has_vision_encoder bool             = true
clip_model_load: - kv   3:                   clip.has_llava_projector bool             = true
clip_model_load: - kv   4:                          general.file_type u32              = 1
clip_model_load: - kv   5:                               general.name str              = openai/clip-vit-large-patch14-336
clip_model_load: - kv   6:                        general.description str              = image encoder for LLaVA
clip_model_load: - kv   7:                        clip.projector_type str              = mlp
clip_model_load: - kv   8:                     clip.vision.image_size u32              = 336
clip_model_load: - kv   9:                     clip.vision.patch_size u32              = 14
clip_model_load: - kv  10:               clip.vision.embedding_length u32              = 1024
clip_model_load: - kv  11:            clip.vision.feed_forward_length u32              = 4096
clip_model_load: - kv  12:                 clip.vision.projection_dim u32              = 768
clip_model_load: - kv  13:           clip.vision.attention.head_count u32              = 16
clip_model_load: - kv  14:   clip.vision.attention.layer_norm_epsilon f32              = 0.000010
clip_model_load: - kv  15:                    clip.vision.block_count u32              = 23
clip_model_load: - kv  16:                     clip.vision.image_mean arr[f32,3]       = [0.481455, 0.457828, 0.408211]
clip_model_load: - kv  17:                      clip.vision.image_std arr[f32,3]       = [0.268630, 0.261303, 0.275777]
clip_model_load: - kv  18:                              clip.use_gelu bool             = false
clip_model_load: - type  f32:  235 tensors
clip_model_load: - type  f16:  142 tensors
clip_model_load: CLIP using CUDA backend
clip_model_load: text_encoder:   0
clip_model_load: vision_encoder: 1
clip_model_load: llava_projector:  1
clip_model_load: model size:     595.53 MB
clip_model_load: metadata size:  0.14 MB
clip_model_load: params backend buffer size =  595.53 MB (377 tensors)
clip_model_load: compute allocated memory: 36.18 MB
llama_model_loader: loaded meta data with 20 key-value pairs and 291 tensors from ../llava-v1.5-7b/ggml-model-f16.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_model_loader: - kv   1:                               general.name str              = LLaMA v2
llama_model_loader: - kv   2:                       llama.context_length u32              = 4096
llama_model_loader: - kv   3:                     llama.embedding_length u32              = 4096
llama_model_loader: - kv   4:                          llama.block_count u32              = 32
llama_model_loader: - kv   5:                  llama.feed_forward_length u32              = 11008
llama_model_loader: - kv   6:                 llama.rope.dimension_count u32              = 128
llama_model_loader: - kv   7:                 llama.attention.head_count u32              = 32
llama_model_loader: - kv   8:              llama.attention.head_count_kv u32              = 32
llama_model_loader: - kv   9:     llama.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  10:                          general.file_type u32              = 1
llama_model_loader: - kv  11:                       tokenizer.ggml.model str              = llama
llama_model_loader: - kv  12:                      tokenizer.ggml.tokens arr[str,32000]   = ["<unk>", "<s>", "</s>", "<0x00>", "<...
llama_model_loader: - kv  13:                      tokenizer.ggml.scores arr[f32,32000]   = [0.000000, 0.000000, 0.000000, 0.0000...
llama_model_loader: - kv  14:                  tokenizer.ggml.token_type arr[i32,32000]   = [2, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, ...
llama_model_loader: - kv  15:                tokenizer.ggml.bos_token_id u32              = 1
llama_model_loader: - kv  16:                tokenizer.ggml.eos_token_id u32              = 2
llama_model_loader: - kv  17:            tokenizer.ggml.padding_token_id u32              = 0
llama_model_loader: - kv  18:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  19:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - type  f32:   65 tensors
llama_model_loader: - type  f16:  226 tensors
llm_load_vocab: special tokens definition check successful ( 259/32000 ).
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = llama
llm_load_print_meta: vocab type       = SPM
llm_load_print_meta: n_vocab          = 32000
llm_load_print_meta: n_merges         = 0
llm_load_print_meta: n_ctx_train      = 4096
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_head           = 32
llm_load_print_meta: n_head_kv        = 32
llm_load_print_meta: n_layer          = 32
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 1
llm_load_print_meta: n_embd_k_gqa     = 4096
llm_load_print_meta: n_embd_v_gqa     = 4096
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-05
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: n_ff             = 11008
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 10000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_yarn_orig_ctx  = 4096
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: model type       = 7B
llm_load_print_meta: model ftype      = F16
llm_load_print_meta: model params     = 6.74 B
llm_load_print_meta: model size       = 12.55 GiB (16.00 BPW) 
llm_load_print_meta: general.name     = LLaMA v2
llm_load_print_meta: BOS token        = 1 '<s>'
llm_load_print_meta: EOS token        = 2 '</s>'
llm_load_print_meta: UNK token        = 0 '<unk>'
llm_load_print_meta: PAD token        = 0 '<unk>'
llm_load_print_meta: LF token         = 13 '<0x0A>'
llm_load_tensors: ggml ctx size =    0.22 MiB
llm_load_tensors: offloading 20 repeating layers to GPU
llm_load_tensors: offloaded 20/33 layers to GPU
llm_load_tensors:        CPU buffer size = 12853.02 MiB
llm_load_tensors:      CUDA0 buffer size =  7720.62 MiB
...................................................................................................
llama_new_context_with_model: n_ctx      = 2048
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:  CUDA_Host KV buffer size =   384.00 MiB
llama_kv_cache_init:      CUDA0 KV buffer size =   640.00 MiB
llama_new_context_with_model: KV self size  = 1024.00 MiB, K (f16):  512.00 MiB, V (f16):  512.00 MiB
llama_new_context_with_model:  CUDA_Host input buffer size   =    12.01 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =   171.60 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   167.20 MiB
llama_new_context_with_model: graph splits (measure): 5

encode_image_with_clip: image encoded in    85.83 ms by CLIP (    0.15 ms per image patch)

 The image shows a cartoon character, resembling an old man with a beard, holding up a finger and giving the thumbs-up sign. This could be a reference to the famous meme "I'm an expert". The background is filled with various school lockers, suggesting that this scene may take place in a school setting or an educational environment.

llama_print_timings:        load time =    6699.59 ms
llama_print_timings:      sample time =      50.83 ms /    76 runs   (    0.67 ms per token,  1495.12 tokens per second)
llama_print_timings: prompt eval time =    9007.51 ms /   616 tokens (   14.62 ms per token,    68.39 tokens per second)
llama_print_timings:        eval time =   22497.33 ms /    76 runs   (  296.02 ms per token,     3.38 tokens per second)
llama_print_timings:       total time =   35969.75 ms /   692 tokens
```
