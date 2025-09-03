## Quantization

## QAT (Quantization Aware Training) quantization
When quantizing a QAT model it might also be good to make sure that the
embeddings weights are also quantized to say Q8_0. By default the `token_embd`
weights migth be left in float32/float16 precision. For example, if we look at
the following model:
```console
(venv) $ ./gguf-py/gguf/scripts/gguf_dump.py ~/Downloads/gemma-3-27b-it-q4_0.gguf 
INFO:gguf-dump:* Loading: /home/danbev/Downloads/gemma-3-27b-it-q4_0.gguf
* File is LITTLE endian, script is running on a LITTLE endian host.
* Dumping 42 key/value pair(s)
      1: UINT32     |        1 | GGUF.version = 3
      2: UINT64     |        1 | GGUF.tensor_count = 808
      3: UINT64     |        1 | GGUF.kv_count = 39
      4: STRING     |        1 | general.architecture = 'gemma3'
      5: UINT32     |        1 | gemma3.context_length = 131072
      6: UINT32     |        1 | gemma3.block_count = 62
      7: UINT32     |        1 | gemma3.embedding_length = 5376
      8: UINT32     |        1 | gemma3.feed_forward_length = 21504
      9: UINT32     |        1 | gemma3.attention.head_count = 32
     10: UINT32     |        1 | gemma3.attention.head_count_kv = 16
     11: UINT32     |        1 | gemma3.attention.key_length = 128
     12: UINT32     |        1 | gemma3.attention.value_length = 128
     13: FLOAT32    |        1 | gemma3.attention.layer_norm_rms_epsilon = 9.999999974752427e-07
     14: STRING     |        1 | gemma3.rope.scaling.type = 'linear'
     15: FLOAT32    |        1 | gemma3.rope.scaling.factor = 8.0
     16: FLOAT32    |        1 | gemma3.rope.freq_base = 1000000.0
     17: UINT32     |        1 | gemma3.attention.sliding_window = 1024
     18: STRING     |        1 | tokenizer.ggml.model = 'llama'
     19: UINT32     |        1 | tokenizer.ggml.bos_token_id = 2
     20: UINT32     |        1 | tokenizer.ggml.eos_token_id = 1
     21: UINT32     |        1 | tokenizer.ggml.padding_token_id = 0
     22: UINT32     |        1 | tokenizer.ggml.unknown_token_id = 3
     23: [STRING]   |   262144 | tokenizer.ggml.tokens = ['<pad>', '<eos>', '<bos>', '<unk>', '<mask>', '[multimodal]', ...]
     24: [FLOAT32]  |   262144 | tokenizer.ggml.scores = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...]
     25: [INT32]    |   262144 | tokenizer.ggml.token_type = [3, 3, 3, 2, 1, 1, ...]
     26: UINT32     |        1 | general.quantization_version = 2
     27: UINT32     |        1 | general.file_type = 2
     28: STRING     |        1 | tokenizer.chat_template = "{{ bos_token }} {%- if messages[0]['role'] == 'system' -%..."
     29: UINT32     |        1 | gemma3.mm.tokens_per_image = 256
     30: UINT32     |        1 | gemma3.vision.attention.head_count = 16
     31: FLOAT32    |        1 | gemma3.vision.attention.layer_norm_epsilon = 9.999999974752427e-07
     32: UINT32     |        1 | gemma3.vision.block_count = 27
     33: UINT32     |        1 | gemma3.vision.embedding_length = 1152
     34: UINT32     |        1 | gemma3.vision.feed_forward_length = 4304
     35: UINT32     |        1 | gemma3.vision.image_size = 896
     36: UINT32     |        1 | gemma3.vision.num_channels = 3
     37: UINT32     |        1 | gemma3.vision.patch_size = 14
     38: BOOL       |        1 | tokenizer.ggml.add_bos_token = True
     39: BOOL       |        1 | tokenizer.ggml.add_eos_token = False
     40: BOOL       |        1 | tokenizer.ggml.add_padding_token = False
     41: BOOL       |        1 | tokenizer.ggml.add_unknown_token = False
     42: STRING     |        1 | tokenizer.ggml.pre = 'default'
* Dumping 808 tensor(s)
      1:       5376 |  5376,     1,     1,     1 | F32     | output_norm.weight
      2: 1409286144 |  5376, 262144,    1,     1 | F16     | token_embd.weight
      3:   11010048 |  5376,  2048,     1,     1 | Q4_0    | blk.0.attn_k.weight
      4:        128 |   128,     1,     1,     1 | F32     | blk.0.attn_k_norm.weight
      5:       5376 |  5376,     1,     1,     1 | F32     | blk.0.attn_norm.weight
      6:   22020096 |  4096,  5376,     1,     1 | Q4_0    | blk.0.attn_output.weight
      7:   22020096 |  5376,  4096,     1,     1 | Q4_0    | blk.0.attn_q.weight
      8:        128 |   128,     1,     1,     1 | F32     | blk.0.attn_q_norm.weight
      9:   11010048 |  5376,  2048,     1,     1 | Q4_0    | blk.0.attn_v.weight
     10:  115605504 | 21504,  5376,     1,     1 | Q4_0    | blk.0.ffn_down.weight
     11:  115605504 |  5376, 21504,     1,     1 | Q4_0    | blk.0.ffn_gate.weight
     12:       5376 |  5376,     1,     1,     1 | F32     | blk.0.ffn_norm.weight
     13:  115605504 |  5376, 21504,     1,     1 | Q4_0    | blk.0.ffn_up.weight
     14:       5376 |  5376,     1,     1,     1 | F32     | blk.0.post_attention_norm.weight
     15:       5376 |  5376,     1,     1,     1 | F32     | blk.0.post_ffw_norm.weight
```
When we quantize the model using `llama-quantize` there is the option to specify
the data type of the token embedding weights using `--token-embedding-type`:
```console
(venv) $ ./build/bin/llama-quantize --help
  ...
  --token-embedding-type ggml_type: use this ggml_type for the token embeddings tensor
  ...
```

So this models original weights are in bf16 so we frist convert to that:
```console
(venv) $ export MODEL_PATH=~/work/ai/models/gemma-3-27b-it-qat-q4_0-unquantized/
(venv) $ make causal-convert-model-bf16
```
```console
(venv) $ make causal-inspect-converted-model
INFO:gguf-dump:* Loading: /home/danbev/work/ai/llama.cpp/models/gemma-3-27b-it-qat-q4_0-unquantized.gguf
* File is LITTLE endian, script is running on a LITTLE endian host.
* Dumping 44 key/value pair(s)
      1: UINT32     |        1 | GGUF.version = 3
      2: UINT64     |        1 | GGUF.tensor_count = 808
      3: UINT64     |        1 | GGUF.kv_count = 41
      4: STRING     |        1 | general.architecture = 'gemma3'
      5: STRING     |        1 | general.type = 'model'
      6: STRING     |        1 | general.name = 'Gemma 3 27b It Qat Q4_0 Unquantized'
      7: STRING     |        1 | general.finetune = 'it-qat-unquantized'
      8: STRING     |        1 | general.basename = 'gemma-3'
      9: STRING     |        1 | general.size_label = '27B'
     10: STRING     |        1 | general.license = 'gemma'
     11: UINT32     |        1 | general.base_model.count = 1
     12: STRING     |        1 | general.base_model.0.name = 'Gemma 3 27b It'
     13: STRING     |        1 | general.base_model.0.organization = 'Google'
     14: STRING     |        1 | general.base_model.0.repo_url = 'https://huggingface.co/google/gemma-3-27b-it'
     15: [STRING]   |        4 | general.tags = ['gemma3', 'gemma', 'google', 'image-text-to-text']
     16: UINT32     |        1 | gemma3.context_length = 131072
     17: UINT32     |        1 | gemma3.embedding_length = 5376
     18: UINT32     |        1 | gemma3.block_count = 62
     19: UINT32     |        1 | gemma3.feed_forward_length = 21504
     20: UINT32     |        1 | gemma3.attention.head_count = 32
     21: FLOAT32    |        1 | gemma3.attention.layer_norm_rms_epsilon = 9.999999974752427e-07
     22: UINT32     |        1 | gemma3.attention.key_length = 128
     23: UINT32     |        1 | gemma3.attention.value_length = 128
     24: UINT32     |        1 | general.file_type = 32
     25: FLOAT32    |        1 | gemma3.rope.freq_base = 1000000.0
     26: UINT32     |        1 | gemma3.attention.sliding_window = 1024
     27: UINT32     |        1 | gemma3.attention.head_count_kv = 16
     28: STRING     |        1 | gemma3.rope.scaling.type = 'linear'
     29: FLOAT32    |        1 | gemma3.rope.scaling.factor = 8.0
     30: UINT32     |        1 | general.quantization_version = 2
     31: STRING     |        1 | tokenizer.ggml.model = 'llama'
     32: STRING     |        1 | tokenizer.ggml.pre = 'default'
     33: [STRING]   |   262208 | tokenizer.ggml.tokens = ['<pad>', '<eos>', '<bos>', '<unk>', '<mask>', '[multimodal]', ...]
     34: [FLOAT32]  |   262208 | tokenizer.ggml.scores = [-1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, ...]
     35: [INT32]    |   262208 | tokenizer.ggml.token_type = [3, 3, 3, 3, 3, 4, ...]
     36: UINT32     |        1 | tokenizer.ggml.bos_token_id = 2
     37: UINT32     |        1 | tokenizer.ggml.eos_token_id = 1
     38: UINT32     |        1 | tokenizer.ggml.unknown_token_id = 3
     39: UINT32     |        1 | tokenizer.ggml.padding_token_id = 0
     40: BOOL       |        1 | tokenizer.ggml.add_bos_token = True
     41: BOOL       |        1 | tokenizer.ggml.add_sep_token = False
     42: BOOL       |        1 | tokenizer.ggml.add_eos_token = False
     43: STRING     |        1 | tokenizer.chat_template = "{{ bos_token }}\n{%- if messages[0]['role'] == 'system' -%..."
     44: BOOL       |        1 | tokenizer.ggml.add_space_prefix = False
* Dumping 808 tensor(s)
      1: 1409630208 |  5376, 262208,     1,     1 | BF16    | token_embd.weight
      2:       5376 |  5376,     1,     1,     1 | F32     | blk.0.attn_norm.weight
      3:  115605504 | 21504,  5376,     1,     1 | BF16    | blk.0.ffn_down.weight
      4:  115605504 |  5376, 21504,     1,     1 | BF16    | blk.0.ffn_gate.weight
      5:  115605504 |  5376, 21504,     1,     1 | BF16    | blk.0.ffn_up.weight
      6:       5376 |  5376,     1,     1,     1 | F32     | blk.0.post_attention_norm.weight
      7:       5376 |  5376,     1,     1,     1 | F32     | blk.0.post_ffw_norm.weight
      8:       5376 |  5376,     1,     1,     1 | F32     | blk.0.ffn_norm.weight
      9:        128 |   128,     1,     1,     1 | F32     | blk.0.attn_k_norm.weight
     10:   11010048 |  5376,  2048,     1,     1 | BF16    | blk.0.attn_k.weight
     11:   22020096 |  4096,  5376,     1,     1 | BF16    | blk.0.attn_output.weight
     12:        128 |   128,     1,     1,     1 | F32     | blk.0.attn_q_norm.weight
     13:   22020096 |  5376,  4096,     1,     1 | BF16    | blk.0.attn_q.weight
     14:   11010048 |  5376,  2048,     1,     1 | BF16    | blk.0.attn_v.weight
     ...
```
So this models has been trained with QAT and specifically Q4_0 quantization in
mind, so we can quantize it using the following command:
```console
(venv) $ make causal-quantize-qat-Q4_0
```

This will produce a model which by defaults will have the token embedding weights
as Q8_0 quantized. We can verify this by inspecting the model:
```console
(venv) $ ../../gguf-py/gguf/scripts/gguf_dump.py ../../models/gemma-3-27b-it-qat-q4_0-unquantized-Q4_0.gguf
INFO:gguf-dump:* Loading: ../../models/gemma-3-27b-it-qat-q4_0-unquantized-Q4_0.gguf
* File is LITTLE endian, script is running on a LITTLE endian host.
* Dumping 44 key/value pair(s)
      1: UINT32     |        1 | GGUF.version = 3
      2: UINT64     |        1 | GGUF.tensor_count = 808
      3: UINT64     |        1 | GGUF.kv_count = 41
      4: STRING     |        1 | general.architecture = 'gemma3'
      5: STRING     |        1 | general.type = 'model'
      6: STRING     |        1 | general.name = 'Gemma 3 27b It Qat Q4_0 Unquantized'
      7: STRING     |        1 | general.finetune = 'it-qat-unquantized'
      8: STRING     |        1 | general.basename = 'gemma-3'
      9: STRING     |        1 | general.size_label = '27B'
     10: STRING     |        1 | general.license = 'gemma'
     11: UINT32     |        1 | general.base_model.count = 1
     12: STRING     |        1 | general.base_model.0.name = 'Gemma 3 27b It'
     13: STRING     |        1 | general.base_model.0.organization = 'Google'
     14: STRING     |        1 | general.base_model.0.repo_url = 'https://huggingface.co/google/gemma-3-27b-it'
     15: [STRING]   |        4 | general.tags = ['gemma3', 'gemma', 'google', 'image-text-to-text']
     16: UINT32     |        1 | gemma3.context_length = 131072
     17: UINT32     |        1 | gemma3.embedding_length = 5376
     18: UINT32     |        1 | gemma3.block_count = 62
     19: UINT32     |        1 | gemma3.feed_forward_length = 21504
     20: UINT32     |        1 | gemma3.attention.head_count = 32
     21: FLOAT32    |        1 | gemma3.attention.layer_norm_rms_epsilon = 9.999999974752427e-07
     22: UINT32     |        1 | gemma3.attention.key_length = 128
     23: UINT32     |        1 | gemma3.attention.value_length = 128
     24: FLOAT32    |        1 | gemma3.rope.freq_base = 1000000.0
     25: UINT32     |        1 | gemma3.attention.sliding_window = 1024
     26: UINT32     |        1 | gemma3.attention.head_count_kv = 16
     27: STRING     |        1 | gemma3.rope.scaling.type = 'linear'
     28: FLOAT32    |        1 | gemma3.rope.scaling.factor = 8.0
     29: STRING     |        1 | tokenizer.ggml.model = 'llama'
     30: STRING     |        1 | tokenizer.ggml.pre = 'default'
     31: [STRING]   |   262208 | tokenizer.ggml.tokens = ['<pad>', '<eos>', '<bos>', '<unk>', '<mask>', '[multimodal]', ...]
     32: [FLOAT32]  |   262208 | tokenizer.ggml.scores = [-1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, ...]
     33: [INT32]    |   262208 | tokenizer.ggml.token_type = [3, 3, 3, 3, 3, 4, ...]
     34: UINT32     |        1 | tokenizer.ggml.bos_token_id = 2
     35: UINT32     |        1 | tokenizer.ggml.eos_token_id = 1
     36: UINT32     |        1 | tokenizer.ggml.unknown_token_id = 3
     37: UINT32     |        1 | tokenizer.ggml.padding_token_id = 0
     38: BOOL       |        1 | tokenizer.ggml.add_bos_token = True
     39: BOOL       |        1 | tokenizer.ggml.add_sep_token = False
     40: BOOL       |        1 | tokenizer.ggml.add_eos_token = False
     41: STRING     |        1 | tokenizer.chat_template = "{{ bos_token }}\n{%- if messages[0]['role'] == 'system' -%..."
     42: BOOL       |        1 | tokenizer.ggml.add_space_prefix = False
     43: UINT32     |        1 | general.quantization_version = 2
     44: UINT32     |        1 | general.file_type = 2
* Dumping 808 tensor(s)
      1:       5376 |  5376,     1,     1,     1 | F32     | output_norm.weight
      2: 1409630208 |  5376, 262208,    1,     1 | Q8_0    | token_embd.weight
      3:   11010048 |  5376,  2048,     1,     1 | Q4_0    | blk.0.attn_k.weight
      4:        128 |   128,     1,     1,     1 | F32     | blk.0.attn_k_norm.weight
      5:       5376 |  5376,     1,     1,     1 | F32     | blk.0.attn_norm.weight
      6:   22020096 |  4096,  5376,     1,     1 | Q4_0    | blk.0.attn_output.weight
      7:   22020096 |  5376,  4096,     1,     1 | Q4_0    | blk.0.attn_q.weight
      8:        128 |   128,     1,     1,     1 | F32     | blk.0.attn_q_norm.weight
      9:   11010048 |  5376,  2048,     1,     1 | Q4_0    | blk.0.attn_v.weight
     10:  115605504 | 21504,  5376,     1,     1 | Q4_0    | blk.0.ffn_down.weight
     11:  115605504 |  5376, 21504,     1,     1 | Q4_0    | blk.0.ffn_gate.weight
     12:       5376 |  5376,     1,     1,     1 | F32     | blk.0.ffn_norm.weight
     13:  115605504 |  5376, 21504,     1,     1 | Q4_0    | blk.0.ffn_up.weight
     14:       5376 |  5376,     1,     1,     1 | F32     | blk.0.post_attention_norm.weight
     15:       5376 |  5376,     1,     1,     1 | F32     | blk.0.post_ffw_norm.weight
     ...
```
The difference in between this and the previous model is that the token embedding
```console
(venv) $ ls -lh ../../models/gemma-3-27b-it-qat-q4_0-unquantized-Q4_0.gguf
-rw-rw-r-- 1 danbev danbev 15G Aug 26 06:29 ../../models/gemma-3-27b-it-qat-q4_0-unquantized-Q4_0.gguf
(venv) $ ls -lh ~/Downloads/gemma-3-27b-it-q4_0.gguf
-rw-rw-r-- 1 danbev danbev 17G Aug 25 20:05 /home/danbev/Downloads/gemma-3-27b-it-q4_0.gguf
```

The default when quantizing to Q4_0 is to actually Q6_K. And while this is
smaller in size having Q8_0 is practically full quanlity with little larger
size. But size is not the only consideration, Q8_0 works with shapes divisable
by 32 while Q6_K works with shapes divisable by 256. Q6_K also requires more
compute to unpack which might not be great for performance.
So for QAT Q4_0 models we should quantize to Q4_0 and then specify Q8_0 for
both the token embeddings and the output layer.


So here is how to think of this: the moment you touch a quantized weight, you
have to unpack/dequantize it before the math can happen.

Lets say we have the prompt “Dan loves ice cream”):

* Token → embedding lookup (input side)
You take each token id and look up one row from the token embedding matrix.
If that embedding matrix is stored as Q8_0 or Q6_K, the kernel unpacks/dequantizes
that row to real numbers.

Q8_0: rows are stored in blocks of 32 int8 values + scale. Easy, fast to unpack.
Q6_K: rows are packed into 256-value superblocks with 6-bit values. More
      bit-twiddling to unpack; slower per row.

Every matmul with quantized weights does the same trick: it reads quantized
blocks and dequantizes on the fly while multiplying with the current activations.
(So the “unpack cost” exists wherever the format is used, not just embeddings.)

* Logits (output side, tied embeddings)
If the model reuses the same embedding matrix as the output projection (common
known as “tied weights”), you now multiply the final hidden state by embedding
to get logits for the entire vocab.

This touches every row of that matrix per token step. If that matrix is Q6_K,
you pay the heavier unpack cost for thousands of rows, which can really hurt
throughput at larger batch sizes. If it’s Q8_0, the unpack is cheaper and
typically faster overall.

For example, lets take the embeddings look up:
```c++
// input embeddings with optional lora
ggml_tensor * llm_graph_context::build_inp_embd(ggml_tensor * tok_embd) const {
    const int64_t n_embd = hparams.n_embd;

    auto inp = std::make_unique<llm_graph_input_embd>();

    ggml_tensor * cur = nullptr;

    if (ubatch.token) {
        inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
        //cb(inp->tokens, "inp_tokens", -1);
        ggml_set_input(inp->tokens);
        res->t_tokens = inp->tokens;

        cur = ggml_get_rows(ctx0, tok_embd, inp->tokens);

        // apply lora for embedding tokens if needed
        for (const auto & lora : *loras) {
            llama_adapter_lora_weight * lw = lora.first->get_weight(tok_embd);
            if (lw == nullptr) {
                continue;
            }

            const float adapter_scale = lora.second;
            const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

            ggml_tensor * inpL_delta = ggml_scale(ctx0, ggml_mul_mat(
                        ctx0, lw->b, // non-transposed lora_b
                        ggml_get_rows(ctx0, lw->a, inp->tokens)
                        ), scale);

            cur = ggml_add(ctx0, cur, inpL_delta);
        }
    } else {
        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, ubatch.n_tokens);
        ggml_set_input(inp->embd);

        cur = inp->embd;
    }

    // For Granite architecture
    if (hparams.f_embedding_scale != 0.0f) {
        cur = ggml_scale(ctx0, cur, hparams.f_embedding_scale);
    }

    cb(cur, "inp_embd", -1);

    res->add_input(std::move(inp));

    return cur;
}
```
And if we look at `ggml_get_rows` which is just returning a tensor operation
but if we look into the cpu implementation of `ggml_get_rows`:
```c++
void ggml_compute_forward_get_rows(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q8_1:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_TQ1_0:
        case GGML_TYPE_TQ2_0:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_get_rows_q(params, dst);
            } break;
```
We can find this is `/work/ai/llama.cpp/ggml/src/ggml-cpu/ops.cpp`:
```c++
static void ggml_compute_forward_get_rows_q(
        const ggml_compute_params * params,
              ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t nc = ne00;
    const int64_t nr = ggml_nelements(src1);

    const ggml_type type = src0->type;
    ggml_to_float_t const dequantize_row_q = ggml_get_type_traits(type)->to_float;

    assert(ne0  == nc);
    assert(ne02 == ne11);
    assert(nb00 == ggml_type_size(type));
    assert(ggml_nrows(dst) == nr);

    const int ith = params->ith;
    const int nth = params->nth;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int64_t i = ir0; i < ir1; ++i) {
        const int64_t i12 = i/(ne11*ne10);
        const int64_t i11 = (i - i12*ne11*ne10)/ne10;
        const int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        const int64_t i01 = *(int32_t *) ((char *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        dequantize_row_q(
                (const void *) ((char *) src0->data + i01*nb01 + i11*nb02 + i12*nb03),
                     (float *) ((char *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}
```
And there we can find the dequantization function for the specific quantization.
This loop runs once per token being looked up. Each iteration calls the
quantization-specific dequantization function to convert one row from quantized
format to float32. For tied weights during output projection, this same pattern
runs for every row in the vocabulary (262K+ times), which is where the Q8_0 vs
Q6_K performance difference becomes significant.
And notice that `dequantize_row_q` is taken/gotten from the type trait:
```c++
ggml_to_float_t const dequantize_row_q = ggml_get_type_traits(type)->to_float;
```
So we can look in `ggml.c` to find the type trait for the type in question,
for example Q4_0:
```c++
    [GGML_TYPE_Q4_0] = {
        .type_name                = "q4_0",
        .blck_size                = QK4_0,
        .type_size                = sizeof(block_q4_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q4_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_q4_0_ref,
    },
```
And we can see that `to_float` is `dequantize_row_q4_0` which is defined in
`ggml/src/ggml-quants.c`:
```c++
void dequantize_row_q4_0(const block_q4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK4_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int j = 0; j < qk/2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >>   4) - 8;

            y[i*qk + j + 0   ] = x0*d;
            y[i*qk + j + qk/2] = x1*d;
        }
    }
}
```
And this should hopefully be familiar as this is what we say earlier in this
document.

```
Input: "Dan loves ice cream" (5 tokens) → 5 embedding lookups
Output: Generate next token → 262,144 dequantizations (entire vocab)
```
