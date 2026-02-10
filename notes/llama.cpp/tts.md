## Text To Speech (TTS)
In llama.cpp there a new feature has been added that enabled text to speech
and there is an accompanying tool named tts.

### Background
The model OuteTTS-0.2-500M, where Oute just stand for the company/org which
produced the model, is described as:
```
An experimental text-to-speech model that uses a pure language modeling approach
to generate speech, without architectural changes to the foundation model itself.
```

Lets take a look at the models (there are two) used in the example, starting
with the LLM model:
```console
INFO:gguf-dump:* Loading: ../llama.cpp-debug/models/outetts-0.2-0.5B-q8_0.gguf
* File is LITTLE endian, script is running on a LITTLE endian host.
* Dumping 43 key/value pair(s)
      1: UINT32     |        1 | GGUF.version = 3
      2: UINT64     |        1 | GGUF.tensor_count = 290
      3: UINT64     |        1 | GGUF.kv_count = 40
      4: STRING     |        1 | general.architecture = 'qwen2'
      5: STRING     |        1 | general.type = 'model'
      6: STRING     |        1 | general.name = 'OuteTTS 0.2 500M'
      7: STRING     |        1 | general.basename = 'OuteTTS-0.2'
      8: STRING     |        1 | general.size_label = '500M'
      9: STRING     |        1 | general.license = 'cc-by-nc-4.0'
     10: UINT32     |        1 | general.dataset.count = 4
     11: STRING     |        1 | general.dataset.0.name = 'Multilingual_Librispeech'
     12: STRING     |        1 | general.dataset.0.organization = 'Facebook'
     13: STRING     |        1 | general.dataset.0.repo_url = 'https://huggingface.co/facebook/multilingual_librispeech'
     14: STRING     |        1 | general.dataset.1.name = 'Libritts_R_Filtered'
     15: STRING     |        1 | general.dataset.1.organization = 'Parler Tts'
     16: STRING     |        1 | general.dataset.1.repo_url = 'https://huggingface.co/parler-tts/libritts_r_filtered'
     17: STRING     |        1 | general.dataset.2.name = 'Emilia Dataset'
     18: STRING     |        1 | general.dataset.2.organization = 'Amphion'
     19: STRING     |        1 | general.dataset.2.repo_url = 'https://huggingface.co/amphion/Emilia-Dataset'
     20: STRING     |        1 | general.dataset.3.name = 'Mls_Eng'
     21: STRING     |        1 | general.dataset.3.organization = 'Parler Tts'
     22: STRING     |        1 | general.dataset.3.repo_url = 'https://huggingface.co/parler-tts/mls_eng'
     23: [STRING]   |        1 | general.tags
     24: [STRING]   |        4 | general.languages
     25: UINT32     |        1 | qwen2.block_count = 24
     26: UINT32     |        1 | qwen2.context_length = 32768
     27: UINT32     |        1 | qwen2.embedding_length = 896
     28: UINT32     |        1 | qwen2.feed_forward_length = 4864
     29: UINT32     |        1 | qwen2.attention.head_count = 14
     30: UINT32     |        1 | qwen2.attention.head_count_kv = 2
     31: FLOAT32    |        1 | qwen2.rope.freq_base = 1000000.0
     32: FLOAT32    |        1 | qwen2.attention.layer_norm_rms_epsilon = 9.999999974752427e-07
     33: STRING     |        1 | tokenizer.ggml.model = 'gpt2'
     34: STRING     |        1 | tokenizer.ggml.pre = 'qwen2'
     35: [STRING]   |   157696 | tokenizer.ggml.tokens
     36: [INT32]    |   157696 | tokenizer.ggml.token_type
     37: [STRING]   |   151387 | tokenizer.ggml.merges
     38: UINT32     |        1 | tokenizer.ggml.eos_token_id = 151645
     39: UINT32     |        1 | tokenizer.ggml.padding_token_id = 151643
     40: UINT32     |        1 | tokenizer.ggml.bos_token_id = 151643
     41: BOOL       |        1 | tokenizer.ggml.add_bos_token = False
     42: UINT32     |        1 | general.quantization_version = 2
     43: UINT32     |        1 | general.file_type = 7
```
So this is a multi-modal LLM model, `qwen2`, it has 157696 tokens in its vocab
which I believe contains text token, special tokens, and audio tokens (and
vision/vision tokens also but those are not related to this example).

For an overview of this we can imagine that if we pass the prompt "Hello World"
this will be processed something like this:
```console
Input: "Hello World" ---> "hello<|text_sep|>world"
Output: logits [0 ... 157695]
```
The output from this model will be a sequence of audio code tokens (integers fro
a codebook. This model is called text-to-codes (model_ttc).

The other model, the WavTokenizer decoder, is a codes-to-speech model (model_cts),
it takes as input the code tokens from above and produces an audio waveform
(PCM samples at 24kHz).

Just to recap one thing about the input to models in llama.cpp, we have two types
of input, tokens and embeddings.
* tokens are token ids (just integers)
* embeddngs are precomputed embeddings (float vectors) used by embedding models
  vision encoders, and in this case the WavTokenizer decoder.

```console
WavTokenizer architecture:
Token ID → tok_embd lookup [512] → conv1d → [768] → posnet/convnext [768] → output [1282]
           ↑ This is n_embd_features          ↑ intermediate                ↑ This is n_embd
```


This token which represents an audio code (I think) will be passed to the
WavTokenizer model and will be looked up, simliar to how text tokens are looked
up to get their embedding, by using the tensor `token_embd` of the WavTokenizer
model.
```console
111:    2097152 |   512,  4096,     1,     1 | F16     | token_embd.weight
```
So the token will act as an index into this tensor, this happens in
`llm_build_inp_embd`.
```c++
    if (ubatch.token) {
        lctx.inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ubatch.n_tokens);
        cb(lctx.inp_tokens, "inp_tokens", -1);
        ggml_set_input(lctx.inp_tokens);

        inpL = ggml_get_rows(ctx, tok_embd, lctx.inp_tokens);
```

The second model is the voice decoder model:
```console
INFO:gguf-dump:* Loading: ../llama.cpp-debug/models/wavtokenizer-large-75-f16.gguf
* File is LITTLE endian, script is running on a LITTLE endian host.
* Dumping 28 key/value pair(s)
      1: UINT32     |        1 | GGUF.version = 3
      2: UINT64     |        1 | GGUF.tensor_count = 161
      3: UINT64     |        1 | GGUF.kv_count = 25
      4: STRING     |        1 | general.architecture = 'wavtokenizer-dec'
      5: STRING     |        1 | general.type = 'model'
      6: STRING     |        1 | general.name = 'WavTokenizer Large Speech 75token'
      7: STRING     |        1 | general.finetune = 'speech-75token'
      8: STRING     |        1 | general.basename = 'WavTokenizer'
      9: STRING     |        1 | general.size_label = 'large'
     10: STRING     |        1 | general.license = 'mit'
     11: UINT32     |        1 | wavtokenizer-dec.block_count = 12
     12: UINT32     |        1 | wavtokenizer-dec.context_length = 8192
     13: UINT32     |        1 | wavtokenizer-dec.embedding_length = 1282
     14: UINT32     |        1 | wavtokenizer-dec.attention.head_count = 1
     15: FLOAT32    |        1 | wavtokenizer-dec.attention.layer_norm_epsilon = 9.999999974752427e-07
     16: UINT32     |        1 | general.file_type = 1
     17: UINT32     |        1 | wavtokenizer-dec.vocab_size = 4096
     18: UINT32     |        1 | wavtokenizer-dec.features_length = 512
     19: UINT32     |        1 | wavtokenizer-dec.feed_forward_length = 2304
     20: FLOAT32    |        1 | wavtokenizer-dec.attention.group_norm_epsilon = 9.999999974752427e-07
     21: UINT32     |        1 | wavtokenizer-dec.attention.group_norm_groups = 32
     22: UINT32     |        1 | wavtokenizer-dec.posnet.embedding_length = 768
     23: UINT32     |        1 | wavtokenizer-dec.posnet.block_count = 6
     24: UINT32     |        1 | wavtokenizer-dec.convnext.embedding_length = 768
     25: UINT32     |        1 | wavtokenizer-dec.convnext.block_count = 12
     26: BOOL       |        1 | wavtokenizer-dec.attention.causal = False
     27: STRING     |        1 | tokenizer.ggml.model = 'none'
     28: UINT32     |        1 | general.quantization_version = 2

* Dumping 161 tensor(s)
      1:        768 |     1,   768,     1,     1 | F32     | convnext.0.dw.bias
      2:       5376 |     7,     1,   768,     1 | F16     | convnext.0.dw.weight
      3:        768 |   768,     1,     1,     1 | F32     | convnext.0.gamma.weight
      4:        768 |   768,     1,     1,     1 | F32     | convnext.0.norm.bias
      5:        768 |   768,     1,     1,     1 | F32     | convnext.0.norm.weight
      6:       2304 |  2304,     1,     1,     1 | F32     | convnext.0.pw1.bias
      7:    1769472 |   768,  2304,     1,     1 | F16     | convnext.0.pw1.weight
      8:        768 |   768,     1,     1,     1 | F32     | convnext.0.pw2.bias
      9:    1769472 |  2304,   768,     1,     1 | F16     | convnext.0.pw2.weight
      ...
    116:        768 |     1,   768,     1,     1 | F32     | posnet.0.conv1.bias
    117:    1769472 |     3,   768,   768,     1 | F16     | posnet.0.conv1.weight
    118:        768 |     1,   768,     1,     1 | F32     | posnet.0.conv2.bias
    119:    1769472 |     3,   768,   768,     1 | F16     | posnet.0.conv2.weight
    120:        768 |     1,   768,     1,     1 | F32     | posnet.0.norm1.bias
    121:        768 |     1,   768,     1,     1 | F32     | posnet.0.norm1.weight
    122:        768 |     1,   768,     1,     1 | F32     | posnet.0.norm2.bias
    123:        768 |     1,   768,     1,     1 | F32     | posnet.0.norm2.weight

    108:    1769472 |  2304,   768,     1,     1 | F16     | convnext.9.pw2.weight
    109:        768 |     1,   768,     1,     1 | F32     | conv1d.bias
    110:    2752512 |     7,   512,   768,     1 | F16     | conv1d.weight
    111:    2097152 |   512,  4096,     1,     1 | F16     | token_embd.weight
    112:        768 |   768,     1,     1,     1 | F32     | output_norm.bias
    113:        768 |   768,     1,     1,     1 | F32     | output_norm.weight
    114:        768 |   768,     1,     1,     1 | F32     | token_embd_norm.bias
    115:        768 |   768,     1,     1,     1 | F32     | token_embd_norm.weight

    160:       1282 |  1282,     1,     1,     1 | F32     | output.bias
    161:     984576 |   768,  1282,     1,     1 | F16     | output.weight

```
So a wav tokenizer model has a vocabulary size of 4096 which are the codes
for the audio. Now a code is a token that represents a piece of audio
information. We can inspect the vocabulary:
```console
gdb) p model_cts.vocab
$14 = {n_vocab = 4096, type = LLAMA_VOCAB_TYPE_NONE, type_pre = LLAMA_VOCAB_PRE_TYPE_DEFAULT, max_token_len = 0,
  token_to_id = std::unordered_map with 0 elements, id_to_token = std::vector of length 0, capacity 0,
  cache_special_tokens = std::vector of length 0, capacity 0, cache_token_to_piece = std::vector of length 0, capacity 0,
  bpe_ranks = std::map with 0 elements, special_bos_id = -1, special_eos_id = -1, special_eot_id = -1, special_eom_id = -1,
  special_unk_id = -1, special_sep_id = -1, special_pad_id = -1, special_cls_id = -1, special_mask_id = -1, linefeed_id = -1,
  special_fim_pre_id = -1, special_fim_suf_id = -1, special_fim_mid_id = -1, special_fim_pad_id = -1, special_fim_rep_id = -1,
  special_fim_sep_id = -1, special_eog_ids = std::set with 0 elements, tokenizer_add_space_prefix = false,
  tokenizer_add_bos = false, tokenizer_add_eos = false, tokenizer_ignore_merges = false, tokenizer_clean_spaces = false,
  tokenizer_remove_extra_whitespaces = false, tokenizer_escape_whitespaces = true, tokenizer_treat_whitespace_as_suffix = false,
  precompiled_charsmap = std::vector of length 0, capacity 0, tokenizer = 0x0}
```
Where cts stands for codes-to-speech.

Notice that it does not contain a vocabulary, like there are no tokens in
id_to_token which there normally are in language models (and there are for
qwen2). Why is this?  Perhaps this is because the id_to_tokens are mostly used
when sampling text tokens and the output of this model is not text but audio.
Hopefully this will become clearer as we continue.

There are 12 convnext blocks and 6 posnet blocks in the voice decoder model.

If we take a look at the example the two models are represented by
`model_ttc` (outetts-0.2-0.5B-q8_0.gguf) and `model_cts` (wavtokenizer-large-75-f16.gguf)
respectively:
 ```c++
    llama_model * model_ttc = NULL; // text-to-codes
    llama_model * model_cts = NULL; // codes-to-speech

    llama_context * ctx_ttc = NULL;
    llama_context * ctx_cts = NULL;

    common_init_result llama_init_ttc = common_init_from_params(params);

    model_ttc = llama_init_ttc.model.get();
    ctx_ttc   = llama_init_ttc.context.get();
```
So we have a model and a context for each of them.

Now, in a "normal" LLM we process a prompt by splitting it into tokens and then
generating embeddings (looking them up) and that is what is passed to the model.
But for a text to speech model it needs to keep the words intact, no subword
splitting, it requires that there is a token to separate the words. For example,
if we have a prompt like "What is LoRA?" this would get changed into:
```
<|text_start|>What<|text_sep|>is<|text_sep|>LoRA<|text_end|>
```

```c++
    std::vector<llama_token> codes;

    // process prompt and generate voice codes
    {
        LOG_INF("%s: constructing prompt ..\n", __func__);

        std::vector<llama_token> prompt_inp;

        prompt_init(prompt_inp, model_ttc);

        prompt_add(prompt_inp, model_ttc, "<|text_start|>the<|text_sep|>overall<|text_sep|>package<|text_sep|>from<|text_sep|>just<|text_sep|>two<|text_sep|>people<|text_sep|>is<|text_sep|>pretty<|text_sep|>remarkable<|text_sep|>sure<|text_sep|>i<|text_sep|>have<|text_sep|>some<|text_sep|>critiques<|text_sep|>about<|text_sep|>some<|text_sep|>of<|text_sep|>the<|text_sep|>gameplay<|text_sep|>aspects<|text_sep|>but<|text_sep|>its<|text_sep|>still<|text_sep|>really<|text_sep|>enjoyable<|text_sep|>and<|text_sep|>it<|text_sep|>looks<|text_sep|>lovely<|text_sep|>", false, true);

        // convert the input text into the necessary format expected by OuteTTS
        {
            std::string prompt_clean = process_text(params.prompt);

            LOG_INF("%s: prompt: '%s'\n", __func__, prompt_clean.c_str());

            prompt_add(prompt_inp, model_ttc, prompt_clean, false, true);
        }

        prompt_add(prompt_inp, model_ttc, "<|text_end|>\n", false, true);
```
So our prompt is added to the `prompt_inp` vector and will contains the
`<|im_start|>` which is the ChatML InterMessage or interactive message character
token.
```console
(gdb) p prompt_inp
$6 = std::vector of length 76, capacity 140 = {151644, 198, 151665, 1782, 151671, 74455, 151671, 1722, 151671, 1499, 151671, 4250,
  151671, 19789, 151671, 16069, 151671, 285, 151671, 32955, 151671, 37448, 480, 151671, 19098, 151671, 72, 151671, 19016, 151671,
  14689, 151671, 36996, 8303, 151671, 9096, 151671, 14689, 151671, 1055, 151671, 1782, 151671, 5804, 1363, 151671, 300, 7973,
  151671, 8088, 151671, 1199, 151671, 43366, 151671, 53660, 151671, 268, 4123, 480, 151671, 437, 151671, 275, 151671, 94273, 151671,
  385, 16239, 151671, 12555, 151671, 285, 151671, 75, 6215}
```
But why the extra tokens?  
This is to give the model some additional context about the flow of messages and
it shows the model how words are separated, the rhythm of speech and various
types of words. This is called prompt conditioning or pattern priming.

Following that we have:
```c++
        const std::string voice_data = R"(<|audio_start|>
the<|t_0.08|><|code_start|><|257|><|740|><|636|><|913|><|788|><|1703|><|code_end|>
overall<|t_0.36|><|code_start|><|127|><|201|><|191|><|774|><|700|><|532|><|1056|><|557|><|798|><|298|><|1741|><|747|><|1662|><|1617|><|1702|><|1527|><|368|><|1588|><|1049|><|1008|><|1625|><|747|><|1576|><|728|><|1019|><|1696|><|1765|><|code_end|>
package<|t_0.56|><|code_start|><|935|><|584|><|1319|><|627|><|1016|><|1491|><|1344|><|1117|><|1526|><|1040|><|239|><|1435|><|951|><|498|><|723|><|1180|><|535|><|789|><|1649|><|1637|><|78|><|465|><|1668|><|901|><|595|><|1675|><|117|><|1009|><|1667|><|320|><|840|><|79|><|507|><|1762|><|1508|><|1228|><|1768|><|802|><|1450|><|1457|><|232|><|639|><|code_end|>
from<|t_0.19|><|code_start|><|604|><|782|><|1682|><|872|><|1532|><|1600|><|1036|><|1761|><|647|><|1554|><|1371|><|653|><|1595|><|950|><|code_end|>
...
        auto tmp = common_tokenize(model_ttc, voice_data, false, true);
        printf("\n\n");
        for (int i = 0; i < tmp.size(); ++i) {
            printf("%d, ", tmp[i]);
        }
        printf("\n\n");
```
So this is what the wav decoder model takes as input, first a token to start the
audio (`<|audio_start|>`) and then the audio tokens in the following format:
```console
word<|t_[duration]|><|code_start|>[audio tokens]<|code_end|>
```
For example for the word 'the' we have:
```console
the<|t_0.08|><|code_start|><|257|><|740|><|636|><|913|><|788|><|1703|><|code_end|>
```
And notice the for `package` we have a lot more code tokens:
```console
package<|t_0.56|><|code_start|><|935|><|584|><|1319|><|627|><|1016|><|1491|><|1344|><|1117|><|1526|><|1040|><|239|><|1435|><|951|><|498|><|723|><|1180|><|535|><|789|><|1649|><|1637|><|78|><|465|><|1668|><|901|><|595|><|1675|><|117|><|1009|><|1667|><|320|><|840|><|79|><|507|><|1762|><|1508|><|1228|><|1768|><|802|><|1450|><|1457|><|232|><|639|><|code_end|>
```
Different words have different durations, for example notice that the word
`package` has a duration of 0.56 seconds which makes sense as the word is longer
and take longer to pronounce.
The code tokens are the audio tokens that are required to generate the audio for
the word in question.

So `<|t_0.08|>` is actually a token and can be found in `tokenizer_config.json`:
```
"155780": {
      "content": "<|t_0.08|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
```
And likewise `<|257|>` is also a token:
```
    "151929": {
      "content": "<|257|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
```


This is also prompt conditioning and it is used to give the model some context
about the rhythm of speech and how words are pronounced. It gives it a sense of
how long each word should take to pronounce. What audio tokens create natural
sounding speech. This is teaching the model about timing and audio generation
patterns rather than just text formatting.
In the code this is actually commented out and the actual tokens are added to
the prompt:
```c++
        prompt_add(prompt_inp, llama_tokens {
            151667, 198, 1782, 155780, 151669, 151929, 152412, 152308, 152585,
            152460, 153375, 151670, 198, 74455, 155808, 151669, 151799,
            151873, 151863, 152446, 152372, 152204, 152728, 152229, 152470,
            151970, 153413, 152419, 153334, 153289, 153374, 153199, 152040,
            ...
```
This does create a pretty large prompt:
```console
main: prompt size: 894
```
Next a batch is create:
```c++
        llama_batch batch = llama_batch_init(std::max(prompt_inp.size(), (size_t) n_parallel), 0, n_parallel);

        std::vector<llama_seq_id> seq_ids(n_parallel, 0);
        for (int32_t i = 0; i < n_parallel; ++i) {
            seq_ids[i] = i;
        }

        // evaluate the initial prompt
        for (size_t i = 0; i < prompt_inp.size(); ++i) {
            common_batch_add(batch, prompt_inp[i], i, seq_ids, false);
        }
        GGML_ASSERT(batch.n_tokens == (int) prompt_inp.size());

        // llama_decode will output logits only for the last token of the prompt
        batch.logits[batch.n_tokens - 1] = true;

        if (llama_decode(ctx_ttc, batch) != 0) {
            LOG_ERR("%s: llama_decode() failed\n", __func__);
            return 1;
        }
```

When this call reaches `llama_build_graph` the following case will be taken:
```c++
        case LLM_ARCH_QWEN2:
            {
                result = llm.build_qwen2();
            } break;
```
So this is just the standard qwen2 model which I think can also be used as a
plain text model.

I'm going to skip ahead to the WaveTokenizer model as this is the part that is
completely new to me. If we take a look at `build_wavtokenizer_dec` (decoder)
we find:
```c++
    struct ggml_cgraph * build_wavtokenizer_dec() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);

        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        cur = ggml_cont(ctx0, ggml_transpose(ctx0, inpL));

        cur = ggml_conv_1d_ph(ctx0, model.conv1d, cur, 1, 1);
        cur = ggml_add(ctx0, cur, model.conv1d_b);
        ...
```
```console
(gdb) p ubatch
$34 = (const llama_ubatch &) @0x7fffffffb3d0: {equal_seqs = true,
n_tokens = 512, n_seq_tokens = 512, n_seqs = 1, token = 0x7fffffffb290,
embd = 0x0, pos = 0x0, n_seq_id = 0x0, seq_id = 0x0, output = 0x0}

(gdb) p model.tok_embd->ne
$33 = {512, 4096, 1, 1}
```
For some reason this being 512 confused me a little as this is also the size
for a max batch when the worst case prefill prompt is generated. But lets
visualize this.

We can check the shape of the tensor:
```console
111:    2097152 |   512,  4096,     1,     1 | F16     | token_embd.weight
```
And this is how it is created:
```c++
            case LLM_ARCH_WAVTOKENIZER_DEC:
                {
                    model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {hparams.n_embd_features, n_vocab}, 0);
```
And if we look at the values:
```console
(gdb) p lctx.model.hparams.n_embd_features
$45 = 512
```
```console
0      [0    511]
       .
       .
       .
4095   [0    511]
```
So we have 4096 rows which represent the audio tokens and each has 512 features.

So after that this `inpL` tensor will be transposed which does nothing to the
shap in this particular case as this is a square matrix but the values are
transposed. After that we have a 1d convolution with horizontal padding (ph)
```console
(gdb) p model.conv1d->name
$48 = "conv1d.weight", '\000' <repeats 50 times>
(gdb) p model.conv1d->ne
$49 = {7, 512, 768, 1}
```
So this convolution kernel has the shape:
```
z_0
    0   [0      6]

    511 [0      6]
.
.
.

z_767
    0   [0      6]

    511 [0      6]
```
And we have the tensor that will be convolved with this kernel:
```console
(gdb) p cur->ne
$50 = {512, 512, 1, 1}

0   [0    6                  511]
    .
    .
    .
511 [0    6                  511]
```

Applying the kernel to the first position (first 7 element):
```console
X = kernel

0   [XXXXXX        +--> [0               767]
     XXXXXX        |
     XXXXXX -------+ 
     XXXXXX
511 [XXXXXX
```
The kernel is then shifted to the right be one position (the second to last
argument to ggml_conv1d_ph). This will produce a new tenor with the shape:
```console
0   [0                             767]
    .
    .
    .
511 [0                             767]

(gdb) p cur->ne
$58 = {512, 768, 1, 1}
```
So that is a tensor with 512 rows and 768 columns but what does this actually
represent? The 512 correspond to a position in the sequence. And each one has
768 feature which encode various aspects of audio like frequency information,
amplitude etc.
The posnet is a way to capture information about the entire sequence using
ResNet blocks, an attention block, finally a normalization block.
The ResNet blocks use a small convolutional kernel, 3 positions,  to capture
local information, how tokens close to each other are related to each other.
The attention block captures global information, how all tokens are related to
each other. The normalization block is used to normalize the output of the
attention block.

Next all the posnet layers, the positional information in the audio data
are processed:
```console
        // posnet
        for (uint32_t il = 0; il < hparams.posnet.n_layer; ++il) {
            const auto & layer = model.layers[il].posnet;

            switch (il) {
                case 0:
                case 1:
                case 3:
                case 4:
                    {
                        cur = llm_build_norm(ctx0, cur, hparams,
                                layer.norm1,
                                layer.norm1_b,
                                LLM_NORM_GROUP, cb, 0);

                        cur = ggml_mul(ctx0, ggml_sigmoid(ctx0, cur), cur);

                        cur = ggml_conv_1d_ph(ctx0, layer.conv1, cur, 1, 1);
                        cur = ggml_add(ctx0, cur, layer.conv1_b);

                        cur = llm_build_norm(ctx0, cur, hparams,
                                layer.norm2,
                                layer.norm2_b,
                                LLM_NORM_GROUP, cb, 0);

                        cur = ggml_mul(ctx0, ggml_sigmoid(ctx0, cur), cur);

                        cur = ggml_conv_1d_ph(ctx0, layer.conv2, cur, 1, 1);
                        cur = ggml_add(ctx0, cur, layer.conv2_b);

                        cur = ggml_add(ctx0, cur, inpL);
                    } break;
```
If I step back and think about this from before the position layers. I there
also have a 512x768 shaped tensor. And each of these 512 rows has a dimension of
768. If I think of this each of these as vectors in that 768 dimensional space,
that the posnet layers are doing is moving them around slightly, or perhaps a
lot sometime, to that they are more appropriately placed in this embedding space
(perhaps if the phonemes for two are similar then one might be moved closer in
the embedding space). So after these layers the vectors point more accurate
positions.
So that is what the posnet layers are doing and this is what will be passed
to the convnext layers.

```console
(gdb) p layer.conv1->ne
$57 = {3, 768, 768, 1}
```


So what is sampled from this model are audio codes. These will be stored in
a vector:
```c++
    std::vector<llama_token> codes;
    ...

                codes.push_back(new_token_id);
```
For example:
```console
(gdb) p new_token_id
$1 = 198
(gdb) p model_ttc.vocab.id_to_token[198]
$2 = {text = "Ċ", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) c
(gdb) p new_token_id
$3 = 12555
(gdb) p model_ttc.vocab.id_to_token[12555]
$4 = {text = "what", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) c
(gdb) p  new_token_id
$7 = 151669
(gdb) p model_ttc.vocab.id_to_token[151669]
$8 = {text = "<|code_start|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) p model_ttc.vocab.id_to_token[152640]
$10 = {text = "<|968|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) c
(gdb) p model_ttc.vocab.id_to_token[152890]
$12 = {text = "<|1218|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) c
(gdb) p model_ttc.vocab.id_to_token[153026]
$14 = {text = "<|1354|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) p model_ttc.vocab.id_to_token[153421]
$16 = {text = "<|1749|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) c
(gdb) p  new_token_id
$17 = 153209
(gdb) p model_ttc.vocab.id_to_token[153209]
$18 = {text = "<|1537|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) c
(gdb) p model_ttc.vocab.id_to_token[152711]
$20 = {text = "<|1039|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) p model_ttc.vocab.id_to_token[152626]
$22 = {text = "<|954|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) p model_ttc.vocab.id_to_token[153439]
$24 = {text = "<|1767|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) p model_ttc.vocab.id_to_token[152137]
$26 = {text = "<|465|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) p model_ttc.vocab.id_to_token[152416]
$28 = {text = "<|744|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) p model_ttc.vocab.id_to_token[153020]
$31 = {text = "<|1348|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) p model_ttc.vocab.id_to_token[152991]
$33 = {text = "<|1319|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) p model_ttc.vocab.id_to_token[152694]
$35 = {text = "<|1022|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) p model_ttc.vocab.id_to_token[152848]
$37 = {text = "<|1176|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) p model_ttc.vocab.id_to_token[152918]
$39 = {text = "<|1246|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
...
(gdb) p codes
$56 = std::vector of length 116, capacity 128 = {198, 12555, 155852, 151669, 152640, 152890, 153026, 153421, 153209, 152711, 152626,
  153439, 152137, 152416, 153020, 152991, 152695, 152848, 152918, 151793, 153146, 152078, 153091, 152934, 152263, 152534, 152263,
  152571, 153146, 151689, 152534, 152945, 152263, 153208, 152263, 152571, 153146, 151689, 153010, 152945, 152263, 152534, 152829,
  153020, 153010, 152399, 153146, 152829, 151793, 153146, 152263, 152920, 152945, 152658, 152049, 152903, 151994, 151856, 152584,
  151782, 152381, 153296, 153049, 152056, 151670, 198, 285, 155785, 151669, 152659, 152028, 153205, 151704, 151719, 152000, 151869,
  151676, 152264, 153190, 151670, 198, 75, 6215, 155807, 151669, 152385, 151694, 153196, 151941, 153267, 151694, 151886, 152035,
  153420, 152197, 152476, 153404, 152144, 153227, 152128, 152903, 152128, 153422, 151817, 152046, 152548, 153343, 153177, 152307,
  153103, 153370, 151670, 198, 151668, 198, 151645}

(gdb) p model_ttc.vocab.id_to_token[151670]
$60 = {text = "<|code_end|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) p model_ttc.vocab.id_to_token[151668]
$58 = {text = "<|audio_end|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) p model_ttc.vocab.id_to_token[151645]
$57 = {text = "<|im_end|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
```
So looking at these tokens we can see that they follow the format we showed
ealier
```console
{198, 12555, 155852, 151669, 152640, 152890, 153026, 153421, 153209, 152711, 152626,
  153439, 152137, 152416, 153020, 152991, 152695, 152848, 152918, 151793, 153146, 152078, 153091, 152934, 152263, 152534, 152263,
  152571, 153146, 151689, 152534, 152945, 152263, 153208, 152263, 152571, 153146, 151689, 153010, 152945, 152263, 152534, 152829,
  153020, 153010, 152399, 153146, 152829, 151793, 153146, 152263, 152920, 152945, 152658, 152049, 152903, 151994, 151856, 152584,
  151782, 152381, 153296, 153049, 152056, 151670,
```
198 is the models tokenizers representation for a newline or a space I think. 
Folling that we have:
```console
what<|t_0.08|><|code_start|><|968|>...<|code_end|>
```
Then we have the space character 198 again. I just wanted to clarify what the
qwen2 is producing.
Oh, this is actually printed a little further down too:
```console
codes: '
what<|t_0.80|><|code_start|><|968|><|1218|><|1354|><|1749|><|1537|><|1039|><|954|><|1767|><|465|><|744|><|1348|><|1319|><|1023|><|1176|><|1246|><|121|><|1474|><|406|><|1419|><|1262|><|591|><|862|><|591|><|899|><|1474|><|17|><|862|><|1273|><|591|><|1536|><|591|><|899|><|1474|><|17|><|1338|><|1273|><|591|><|862|><|1157|><|1348|><|1338|><|727|><|1474|><|1157|><|121|><|1474|><|591|><|1248|><|1273|><|986|><|377|><|1231|><|322|><|184|><|912|><|110|><|709|><|1624|><|1377|><|384|><|code_end|>
is<|t_0.13|><|code_start|><|987|><|356|><|1533|><|32|><|47|><|328|><|197|><|4|><|592|><|1518|><|code_end|>
lora<|t_0.35|><|code_start|><|713|><|22|><|1524|><|269|><|1595|><|22|><|214|><|363|><|1748|><|525|><|804|><|1732|><|472|><|1555|><|456|><|1231|><|456|><|1750|><|145|><|374|><|876|><|1671|><|1505|><|635|><|1431|><|1698|><|code_end|>
<|audio_end|>
<|im_end|>'
```

Next we have:
```c++
    // remove all non-audio tokens (i.e. < 151672 || > 155772)
    codes.erase(std::remove_if(codes.begin(), codes.end(), [](llama_token t) { return t < 151672 || t > 155772; }), codes.end());
```
So this is removing the newline token, the time tokens, the code start and end
tokens etc.

```console
(gdb) p codes
$77 = std::vector of length 116, capacity 128 = {198, 12555, 155852, 151669, 152640, 152890, 153026, 153421, 153209, 152711, 152626,
  153439, 152137, 152416, 153020, 152991, 152695, 152848, 152918, 151793, 153146, 152078, 153091, 152934, 152263, 152534, 152263,
  152571, 153146, 151689, 152534, 152945, 152263, 153208, 152263, 152571, 153146, 151689, 153010, 152945, 152263, 152534, 152829,
  153020, 153010, 152399, 153146, 152829, 151793, 153146, 152263, 152920, 152945, 152658, 152049, 152903, 151994, 151856, 152584,
  151782, 152381, 153296, 153049, 152056, 151670, 198, 285, 155785, 151669, 152659, 152028, 153205, 151704, 151719, 152000, 151869,
  151676, 152264, 153190, 151670, 198, 75, 6215, 155807, 151669, 152385, 151694, 153196, 151941, 153267, 151694, 151886, 152035,
  153420, 152197, 152476, 153404, 152144, 153227, 152128, 152903, 152128, 153422, 151817, 152046, 152548, 153343, 153177, 152307,
  153103, 153370, 151670, 198, 151668, 198, 151645}
(gdb) n
843	        const std::string inp_txt = common_detokenize(ctx_ttc, codes, true);
(gdb) p codes
$78 = std::vector of length 96, capacity 128 = {152640, 152890, 153026, 153421, 153209, 152711, 152626, 153439, 152137, 152416,
  153020, 152991, 152695, 152848, 152918, 151793, 153146, 152078, 153091, 152934, 152263, 152534, 152263, 152571, 153146, 151689,
  152534, 152945, 152263, 153208, 152263, 152571, 153146, 151689, 153010, 152945, 152263, 152534, 152829, 153020, 153010, 152399,
  153146, 152829, 151793, 153146, 152263, 152920, 152945, 152658, 152049, 152903, 151994, 151856, 152584, 151782, 152381, 153296,
  153049, 152056, 152659, 152028, 153205, 151704, 151719, 152000, 151869, 151676, 152264, 153190, 152385, 151694, 153196, 151941,
  153267, 151694, 151886, 152035, 153420, 152197, 152476, 153404, 152144, 153227, 152128, 152903, 152128, 153422, 151817, 152046,
  152548, 153343, 153177, 152307, 153103, 153370}

(gdb) p model_ttc.vocab.id_to_token[152640]
$82 = {text = "<|968|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
```
Then we have the following:
```console
    for (auto & token : codes) {
        token -= 151672;
    }
```
So the first token is 152640, and this is doing 152640 - 151672 = 968. Why
151672?  This is because this is the first audio token which we can see by
looking in tokenizer_config.json:
```console
    "151672": {
      "content": "<|0|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "151673": {
      "content": "<|1|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },

    ...

    "155768": {
      "content": "<|4096|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "155769": {
      "content": "<|4097|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "155770": {
      "content": "<|4098|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "155771": {
      "content": "<|4099|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
```
The vocab size of the WebTokenizer model is 4096 so what this is doing is
basically shifting this down to the range 0-4095.
So after all tokens have been shifted down they look like this:
```console
(gdb) p codes
$88 = std::vector of length 96, capacity 128 = {968, 1218, 1354, 1749, 1537, 1039, 954, 1767, 465, 744, 1348, 1319, 1023, 1176,
  1246, 121, 1474, 406, 1419, 1262, 591, 862, 591, 899, 1474, 17, 862, 1273, 591, 1536, 591, 899, 1474, 17, 1338, 1273, 591, 862,
  1157, 1348, 1338, 727, 1474, 1157, 121, 1474, 591, 1248, 1273, 986, 377, 1231, 322, 184, 912, 110, 709, 1624, 1377, 384, 987, 356,
  1533, 32, 47, 328, 197, 4, 592, 1518, 713, 22, 1524, 269, 1595, 22, 214, 363, 1748, 525, 804, 1732, 472, 1555, 456, 1231, 456,
  1750, 145, 374, 876, 1671, 1505, 635, 1431, 1698}
```
This will then be passed to the wavtokenizer in a batch:
```c++
    const int n_codes = codes.size();

    llama_batch batch = llama_batch_init(n_codes, 0, 1);

    for (size_t i = 0; i < codes.size(); ++i) {
        common_batch_add(batch, codes[i], i, { 0 }, true); // TODO: all logits?
    }
    GGML_ASSERT(batch.n_tokens == n_codes);

    if (llama_decode(ctx_cts, batch) != 0) {
        LOG_ERR("%s: llama_decode() failed\n", __func__);
        return 1;
    }
```
```c++
    const int n_embd = llama_n_embd(model_cts);
    const float * embd = llama_get_embeddings(ctx_cts);

    auto audio = embd_to_audio(embd, n_codes, n_embd, params.cpuparams.n_threads);
```
```console
(gdb) p n_codes
$92 = 96
(gdb) p n_embd
$93 = 1282
```
Lets inspect a few of the embedding values:
```console
(gdb) p embd[0]
$95 = -1.40534163
(gdb) p embd[1]
$96 = -0.891580999
(gdb) p embd[2]
$97 = -1.56436288
(gdb) p embd[3]
$98 = 0.353795171
```
So these value represent compressed audio in the frequency domain. Now, to recap
a little, normal sound when we record we are capturing the pressure the air
waves are hitting the microphone. This is sampled thousands of times per second
and each number, sample, is the amount of how much the microphone membrane is
pushed in or pushed out. These would be numbers like 0.2, 0.3, -0.1 etc.
An audio signal would be store something like this:
```console
Original Audio --> Fourier Transform --> Magnitude and Phase
Time Domain                              Frequency Domain
```
In this case what the wavtokenizer is doing is that it is generating data in
the frequency domain. So the embeddings are the magnitude and phase and they in
a LLM they are generally stored with all the frequency values in the first half
of a vector followed by all the phase values in the second half. This is becuase
that this is more efficient for the neural network to work with. There are still
pairs of values, magnitude and phase, but they are stored in a different way.
This is good to keep in mind later when we transform these values for processing
back to get the audio signal.

The Fourier Transform that says that any signal can be represented as a sum of
sine and cosine waves. And each sine wave has two properties magnitude and
phase. Note that this is magnitude and not amplitude so this is a how far value
differs from zero. The magnitude is always positive (the values above in the
first half of embd are in logarithmic scale so they need to be exponentiated to
get the actual values which we can then see are positive). Imagine the final
audio wave for the word "Hello", it will first have a frequencies for the H and
the transition to the frequencies for e, and it need to know where along then
final audio wave to do this transition, this is what phase is for.

The embeddings in the code, `embd`, are the magnitude and phase of the sine and
cosine waves that make up the audio signal. These are pairs of values but the
are stored in a way that is more efficient for the neural network to work with.

So to produce a sound we need to take these values of magnitude and phase
and sum them up to get the original sound.

So lets look at the magnitude and phase for the first element in embd. Now, the
magnitude values are in logarithmic scale so they need to be exponentiated to
get the actual values:
```console
(gdb) p exp(embd[0])
$108 = 0.24528324719634226
(gdb) p embd[n_embd/2 + 0]
$104 = 1.72218394
```
The forumla for the resulting wave at any given time `t` is:
```console
amplitude = magnitude * cos(frequency_in_hz * 2π * t + phase)

amplitude = 0.24528324719634226 * cos(frequence * t + 1.72218394)
frequence = ?
t = 0
```
The frequence rate is defined as:
```c++
    const int n_sr = 24000; // sampling rate
```
We need to this calculate the frequency for each value:
```console
frequency = (component_index * sample_rate) / n_fft

component_index = 0
frequency = (0 * 24000) / 1280 = 0 Hz
```
So this is the frequence of the first sine wave that, together with the others
will make up the final audio signal. This has a zero frequency meaning that it
does not osscilate at all, it is a constant value.
```console
t = 0
amplitude = 0.24528324719634226 * cos(0 * 0 + 1.72218394)
          = 0.24528324719634226 * cos(1.72218394)
          = 0.24528324719634226 * (-0.15643446...)
          ≈ -0.0383697
```
So this first "wave" will simple offset the signal by -0.0383697 when added
to other waves.

```console
(gdb) p exp(embd[1])
$114 = 0.41000701941844214
(gdb) p embd[n_embd/2 + 1]
$115 = -0.819735587

component_index = 1
frequency = (1 * 24000) / 1280 = 18.75 Hz
```
We need to think about this as a complete wave, how it has a frequency of
18.75 Hz meaning that it osscilates 18.75 times per second.

```
amplitude = 0.410 * cos(18.75 * 2π * t + 0.353795171)

t = 0
amplitude = 0.410 * cos(0 + 0.353795171)
          = 0.410 * 0.938...
          ≈ 0.384

t = 1/18.75  (one cycle, remember that this wave completes 18.75 cycles per second)
amp = 0.410 * cos(18.75 * 2π * 1/18.75 + 0.353795171)
    = 0.410 * cos(2π + 0.353795171)
    = 0.410 * 0.938...
    ≈ 0.384

t = 2/18.75
amp = 0.410 * cos(18.75 * 2π * 2/18.75 + 0.353795171)
    = 0.410 * cos(4π + 0.353795171)
    = 0.410 * 0.938...
    ≈ 0.384
```
And again this would be another wave that is added with all the others.

So to recap, the embeddings are pairs values, the first half represent the
magnitudes (in log scale) and the second half the matching phases.
Each index maps to a specific frequency, (index * 24000/1280 Hz). And the sum
of all these waves will produce the final audio signal. The process of going
from an audio signal (time domain) to the embeddings (frequency domain) looks
something like this:
```console
Original audio → Apply Hann → Forward FFT → [embd data]
```
And in this case we are going the other way:
```console
[embd data] → Inverse FFT → Apply Hann → Overlap-add
```

So lets take a look at the function `embd_to_audio`:
```console
static std::vector<float> embd_to_audio(
        const float * embd,
        const int n_codes,
        const int n_embd,
        const int n_thread) {
    const int n_fft = 1280;   // Fourier transform window.
    const int n_hop = 320;    // How far to "hop" between windows.
    const int n_win = 1280;   // Size of the window.
    const int n_pad = (n_win - n_hop)/2;  // padding before and after the window.
    const int n_out = (n_codes - 1)*n_hop + n_win;
```
`embd` is a vector of size `n_codes * n_embd` and contains the magnitudes first
followed by the phases. The make up pairs of values, the magnitude and phase.

The size 1280 means 1280 samples, likewise 320 means that we hop/skip over 320
samples.
```
          1280
   +------------------+
   |                  |
   |                  |
   |                  | 
   |                  |
   |                  |
   +------------------+
   | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | 
   ↑                  ↑
   0                 1280
   
               1280
       +------------------+
       |               |  |
       |  960 samples  |  |
       |  overlap      |  | 
       |               |  |
       |               |  |
       +------------------+
   | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | 
   ↑   ↑               ↑
   0  320             1280
       [1280-320=960   ]
```
So the above tries to show the window and the overlap.
Next we have the Hann window:
```console
    std::vector<float> hann(n_fft);
    fill_hann_window(hann.size(), true, hann.data());
```
So, the normal Fourier transform requires a continous curve (recall that we are
doing the inverse Fourier transform here but they are very simlar and have the
same requirements that the signal is periodic and continous). When we take slices
or chunks of the signal, like we do above taking 1280 samples at a time. 
So the actual singal might be continious but each 1280 chunk will not be and
each chunk is processed separatly through the Fourier transform. The Hann window
is needed to make each chunk continous.
```
    1280 samples
  [ chunk  1   ] --> inv_fft(chunk)
      ↑
    Repeating for ever is what the inv_fft "sees"

  [ chunk  2   ] --> inv_fft(chunk)
      ↑
    Repeating for ever is what the inv_fft "sees"
```
We take a chunk from a signal placing two of these chunks side by side, this is
what the inverse Fourier transform sees:
```
 [chunk 1][chunk 1][chunk 1][chunk 1][chunk 1][chunk 1][chunk 1][chunk 1]...
```
The chunks will have breaks in the wave form at the edges (unless we are
extreemly lucky). One might "jump" up or down breaking the continous curve. The
Hann window is used to smooth out each chunk so that the edges are not so sharp.
So notice that we do this for each chunk, so the Hann window is applied to each
chunk after it is passed to the fourier transform, and the form of the wave will
be a bell shaped curve so it starts a zero, goes up to the peak and then back
down to zero. This way adding two chunks next to each other will not have a
break and they one chunk ends with zero and the other starts at zero.

The forumla for Hann is the following:
```console
w(n) = 0.5 * (1 - cos(2π * n / (N-1)))

n = the sample number
N = the number of samples in the chunk, the window size that is.
```
So each chunk is multiplied by the Hann window, which we can visualize or think
of as a curve that is multiplied by the chunk's curve.

We can actually do the Hann stuff after the inverse Fourier transform which is
actually what the code is doing.
If we look at the code we can see this:
```console
    std::vector<float> hann(n_fft);
    fill_hann_window(hann.size(), true, hann.data());
```
So this is simply a vector of float values, the size of a chunk/window which is
1280 samples in this cae, and this is filled using:
```console
static void fill_hann_window(int length, bool periodic, float * output) {
    int offset = -1;
    if (periodic) {
        offset = 0;
    }
    for (int i = 0; i < length; i++) {
        output[i] = 0.5 * (1.0 - cosf((2.0 * M_PI * i) / (length + offset)));
                          ↑ 
                    // Forumla for Hann window 
    }
}
```
Recall that the purpose of this is that each chunk, when this vector is used
will smooth out the edges of the chunk so that when two chunks are placed next
to each other there will not be a break in the wave form.

Following that we have:
```console
    int n_spec = n_embd*n_codes;

    std::vector<float> E (n_spec);
    std::vector<float> S (n_spec);
    std::vector<float> ST(n_spec);
```
Now, `n_spec` represents the total number of sample points/values for current
prompt (the output returned by the WaveTokenizer model). Recall that we
mentioned ealier that the embeddings are pairs of values, the magnitude and
phase, and that they are stored in a way that is more efficient for the neural
network to work with. We are now going to transform these values back.

Now `embd` is just a float array which was created by operation:
```console
        cur = ggml_add(ctx0, cur, model.output_b);
        cb(cur, "result_embd", -1);

(gdb) p cur->ne
$79 = {1282, 1, 1, 1}
```
The second dimension is the number of tokens/codes in generated. Now in our
case we have 96:
```
{1282, 96}

0    [0                  1281]
.          .
.          .
.          .
95   [122976           123072]
```
So the embd array will be in this format as well but it is just a flat array
but we can think of it as 2d array like above. That is we have 1282 values for
which the first chunk, then another 1282 values for the second chunk and so.
```console
    for (int l = 0; l < n_codes; ++l) {
        for (int k = 0; k < n_embd; ++k) {
            E[k*n_codes + l] = embd[l*n_embd + k];
        }
    }
```
So this is going to iterate from 0 to 95. And starting with 0 it will then
iterate over 0 to 1281.
```
l=0, k=0     k*n_codes+l=0      l*n_embd+k = 0
l=0, k=1     k*n_codes+l=96     l*n_embd+k = 1
.       .
.       .
.       .
l=0, k=1281  k*n_codes+l=122976 l*n_embd+k = 1281

E[0]      = embd[0]
E[96]     = embd[1]
E[192]    = embd[2]
E[288]    = embd[3]
...
E[122976] = embd[1281]

l=1, k=0     k*n_codes+l=1      l*n_embd+k = 1282
l=1, k=1     k*n_codes+l=97     l*n_embd+k = 1283

E[1]      = embd[1282]
E[97]     = embd[1283]


E will become:
0   [0             95]
1   [96           191]
2   [192          287]
3   [288          383]
.          .
.          .
.          .
1281 [126977    123071]

So this is essentially just transposing the embd array.
```
So now that we have E in the correct order {96, 1282}.

Next the the values are converted to complex numbers which is what the inverse
fourier transformation expects:
```console
    for (int k = 0; k < n_embd/2; ++k) {
        for (int l = 0; l < n_codes; ++l) {
            float mag = E[(k           )*n_codes + l];
            float phi = E[(k + n_embd/2)*n_codes + l];

            mag = exp(mag);

            if (mag > 1e2) {
                mag = 1e2;
            }
            S[2*(k*n_codes + l) + 0] = mag*cosf(phi);
            S[2*(k*n_codes + l) + 1] = mag*sinf(phi);
        }
    }
```

```console
(gdb) p E[(k           )*n_codes + l]
$114 = -1.40534163
(gdb) p E[0]
$115 = -1.40534163
(gdb) p (0 + n_embd/2)*n_codes + 0
$111 = 61536
(gdb) p (n_embd * n_codes) / 2
$117 = 61536
```
So, here we can see that infact the first half of E are the magnitudes and the
second half are the phases. And the above is extracting the pairs into mag and
phi. And like we mentioned earlier the magnitudes are in log scale so they need
to be exponentiated to get the actual values. And these are then converted
from polar form to complex number (cosine and sine) entries in `S`:
```
k_0
  [mag_0, phi_0]     (0, 1)      chunk 0
  ...
  [mag_95, phi_95]               chunk 95

k_1 
  [mag_0, phi_0]     (192,193)   chunk 0
  ...
  [mag_95, phi_95]               chunk 95

k_2 
  [mag_0, phi_0]     (384, 385)  chunk 0
  ...
  [mag_95, phi_95]               chunk 95
.
.
.
k_641 
  [mag_0, phi_0]                  chunk 0
  ...
  [mag_95, phi_95]    (122976, 122977) chunk 95


shape: {2, 96, 641}

(gdb) p 641 * (n_codes*2)
$124 = 123072
```
So, `S` is a 3d array of shape {2, 96, 641} where the first dimension is the
real and imaginary parts of the complex number, the second dimension is the
number of tokens/codes and the third dimension is the number of pairs.
Notice that the chunks are not contiguous in the array, they are interleaved
with the other chunks.

And finally ST is populated and this will store values for each chunk
contiguously so that each one can be processed indpendently by a separate
thread:
```console
    for (int l = 0; l < n_codes; ++l) {
        for (int k = 0; k < n_embd/2; ++k) {
            ST[l*n_embd + 2*k + 0] = S[2*(k*n_codes + l) + 0];
            ST[l*n_embd + 2*k + 1] = S[2*(k*n_codes + l) + 1];
        }
    }
```
So the first chunk will be store like this:
```
chunk 0:
ST[0] = S[0]   (k_0 mag_0)
ST[1] = S[1]   (k_0 phi_0)
ST[2] = S[192] (k_1 mag_0)
ST[3] = S[193] (k_1 phi_0)
ST[4] = S[384] (k_2 mag_0)
ST[5] = S[385] (k_2 phi_0)
...
```
Next two vectors are created:
```console
    std::vector<float> res  (n_codes*n_fft);
    std::vector<float> hann2(n_codes*n_fft);
```
And then n_thread are created:
```console
    std::vector<std::thread> workers(n_thread);
    for (int i = 0; i < n_thread; ++i) {
        workers[i] = std::thread([&, i]() {
            for (int l = i; l < n_codes; l += n_thread) {
                irfft(n_fft, ST.data() + l*n_embd, res.data() + l*n_fft);
                for (int j = 0; j < n_fft; ++j) {
                    res  [l*n_fft + j] *= hann[j];
                    hann2[l*n_fft + j]  = hann[j] * hann[j];
                }
            }
        });
    }
    for (int i = 0; i < n_thread; ++i) {
        workers[i].join();
    }
```
`n_thread` is 4 in this case so there will be 4 worker threads created. And
notice that the lambda takes `i` as a parameter so `0, 1, 2, 3` which is assigned
to l in the loop. And notice the for loop has `l += n_thread`:
```
worker[0] l=0, 4, 8, 12, ... 92
worker[1] l=1, 5, 9, 13, ... 93
worker[2] l=2, 6, 10, 14, ...94
worker[3] l=3, 7, 11, 15, ...95
```
```console
irfft(n_fft, ST.data() + l*n_embd, res.data() + l*n_fft);
```
So each thread will pass a "slice" of the ST array, and a slice of the result
array res to the inverse Fourier transform function.

```console
static void irfft(int n, const float * inp_cplx, float * out_real) {
    int N = n / 2 + 1;

    std::vector<float> real_input(N);
    std::vector<float> imag_input(N);
    for (int i = 0; i < N; ++i) {
        real_input[i] = inp_cplx[2 * i];
        imag_input[i] = inp_cplx[2 * i + 1];
    }

    std::vector<float> real_output(n);
    std::vector<float> imag_output(n);

    for (int k = 0; k < n; ++k) {
        real_output[k] = 0.0f;
        imag_output[k] = 0.0f;
        for (int m = 0; m < N; ++m) {
            float twiddle_real;
            float twiddle_imag;

            twiddle(&twiddle_real, &twiddle_imag, k * m, n);

            real_output[k] += real_input[m] * twiddle_real - imag_input[m] * twiddle_imag;
            imag_output[k] += real_input[m] * twiddle_imag + imag_input[m] * twiddle_real;
        }
    }

    for (int i = 0; i < n; ++i) {
        out_real[i] = real_output[i] / N;
    }
}

// very poor-man fft
static void twiddle(float * real, float * imag, int k, int N) {
    float angle = 2 * M_PI * k / N;
    *real = cos(angle);
    *imag = sin(angle);
}
```

And finally we have the overlap-add process and normalization:
```console
    std::vector<float> audio;
    std::vector<float> env;

    fold(res,   n_out, n_win, n_hop, n_pad, audio);
    fold(hann2, n_out, n_win, n_hop, n_pad, env); // TODO: can be done once

    for (size_t i = 0; i < audio.size(); ++i) {
        audio[i] /= env[i];
    }

    return audio;
}

static void fold(const std::vector<float> & data, int64_t n_out, int64_t n_win, int64_t n_hop, int64_t n_pad, std::vector<float> & output) {
    int64_t output_height = n_out;
    int64_t kernel_w = n_win;
    int64_t stride_w = n_hop;
    int64_t width    = n_out;

    output.resize(width, 0.0f);

    int64_t col_idx = 0;
    for (int64_t w_col = 0; w_col < width; ++w_col) {
        int64_t start = w_col * stride_w - n_pad;
        int64_t end   = start + kernel_w;

        for (int64_t w_im = start; w_im < end; ++w_im) {
            if (w_im >= 0 && w_im < output_height && col_idx < (int64_t) data.size()) {
                output[w_im] += data[col_idx];
            }
            col_idx++;
        }
    }

    output.resize(n_out - 2 * n_pad);
}
```

_wip_ 

### tts example
The example in llama.cpp uses a model from [outeai](https://www.outeai.com/)

Download the model that contains the LLM model:
```console
$ pushd models
$ git clone --branch main --single-branch --depth 1 https://huggingface.co/OuteAI/OuteTTS-0.2-500M
$ cd OuteTTS-0.2-500M && git lfs install && git lfs pull
$ popd
```
Convert the model to .gguf format:
```console
(venv) python convert_hf_to_gguf.py models/OuteTTS-0.2-500M/ --outfile models/outetts-0.2-0.5B-f16.gguf --outtype f16
```
The generated model will be `models/outetts-0.2-0.5B-f16.gguf`.

We can optionally quantize this to Q8_0 using the following command:
```console
$ build/bin/llama-quantize models/outetts-0.2-0.5B-f16.gguf models/outetts-0.2-0.5B-q8_0.gguf q8_0
```
The quantized model will be `models/outetts-0.2-0.5B-q8_0.gguf`.

Next we do something simlar to the audio decoder. Recall that this is text to
audio so the generated tokens from the LLM need to be converted into audio.
First download the model for the voice decoder:
```console
$ pushd models
$ git clone --branch main --single-branch --depth 1 https://huggingface.co/novateur/WavTokenizer-large-speech-75token
$ cd WavTokenizer-large-speech-75token && git lfs install && git lfs pull
$ popd
```
This model file is PyTorch checkpoint (.ckpt) and we first need to convert it to
huggingface format:
```console
(venv) python examples/tts/convert_pt_to_hf.py models/WavTokenizer-large-speech-75token/wavtokenizer_large_speech_320_24k.ckpt
...
Model has been successfully converted and saved to models/WavTokenizer-large-speech-75token/model.safetensors
Metadata has been saved to models/WavTokenizer-large-speech-75token/index.json
Config has been saved to models/WavTokenizer-large-speech-75tokenconfig.json
```
And then we can convert the huggingface format to gguf:
```console
(venv) python convert_hf_to_gguf.py models/WavTokenizer-large-speech-75token/ --outfile models/wavtokenizer-large-75-f16.gguf --outtype f16
...
INFO:hf-to-gguf:Model successfully exported to models/wavtokenizer-large-75-f16.gguf
```

With both of the models generated, the LLM model and the voice decoder model,
we can run the example:
```console
$ build/bin/llama-tts -m  ./models/outetts-0.2-0.5B-q8_0.gguf \
    -mv ./models/wavtokenizer-large-75-f16.gguf \
    -p "Hello world"
...
main: audio written to file 'output.wav'
```
The output.wav file will contain the audio of the prompt. This can be heard
by playing the file with a media player. On Linux the following command will
play the audio:
```console
$ aplay output.wav
```
This section can be removed once https://github.com/ggerganov/llama.cpp/pull/11155
lands.


### Server version of the TTS example
This example requires two server running, one that serves the LLM model and the
other that serves the voice decoder model.

Start the server with the LLM model:
```console
$ ./build/bin/llama-server -m ./models/outetts-0.2-0.5B-q8_0.gguf -ngl 99 --port 8020
```
Start the server with the voice decoder model:
```console
$ ./build/bin/llama-server -m ./models/wavtokenizer-large-75-f16.gguf --port 8021 --embeddings --pooling none
print_info: file format = GGUF V3 (latest)
print_info: file type   = F16
print_info: file size   = 124.15 MiB (16.03 BPW) 
print_info: arch             = wavtokenizer-dec
print_info: vocab_only       = 0
print_info: n_ctx_train      = 8192
print_info: n_embd           = 1282
print_info: n_layer          = 12
print_info: n_head           = 1
print_info: n_head_kv        = 1
print_info: n_rot            = 1282
print_info: n_swa            = 0
print_info: n_embd_head_k    = 1282
print_info: n_embd_head_v    = 1282
print_info: n_gqa            = 1
print_info: n_embd_k_gqa     = 1282
print_info: n_embd_v_gqa     = 1282
print_info: f_norm_eps       = 1.0e-06
print_info: f_norm_rms_eps   = 0.0e+00
print_info: f_clamp_kqv      = 0.0e+00
print_info: f_max_alibi_bias = 0.0e+00
print_info: f_logit_scale    = 0.0e+00
print_info: n_ff             = 2304
print_info: n_expert         = 0
print_info: n_expert_used    = 0
print_info: causal attn      = 0
print_info: pooling type     = 0
print_info: rope type        = -1
print_info: rope scaling     = linear
print_info: freq_base_train  = 10000.0
print_info: freq_scale_train = 1
print_info: n_ctx_orig_yarn  = 8192
print_info: rope_finetuned   = unknown
print_info: ssm_d_conv       = 0
print_info: ssm_d_inner      = 0
print_info: ssm_d_state      = 0
print_info: ssm_dt_rank      = 0
print_info: ssm_dt_b_c_rms   = 0
print_info: model type       = ?B
print_info: model params     = 64.98 M
print_info: general.name     = WavTokenizer Large Speech 75token
print_info: vocab type       = no vocab
print_info: n_vocab          = 0
print_info: n_merges         = 0
print_info: max token length = 0

Thread 1 "llama-server" hit Breakpoint 1, llama_model::load_tensors (this=0x555555e08740, ml=...)
    at /home/danbev/work/ai/llama.cpp-debug/src/llama-model.cpp:3250
3250	                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {hparams.n_embd_features, n_vocab}, 0);
(gdb) p n_vocab
$1 = 0
(gdb) up
#1  0x00007ffff7c5d7ff in llama_model_load (fname="./models/wavtokenizer-large-75-f16.gguf", model=..., params=...)
    at /home/danbev/work/ai/llama.cpp-debug/src/llama.cpp:73
73	        if (!model.load_tensors(ml)) {
(gdb) bt
#0  llama_model::load_tensors (this=0x555555e08740, ml=...) at /home/danbev/work/ai/llama.cpp-debug/src/llama-model.cpp:3250
#1  0x00007ffff7c5d7ff in llama_model_load (fname="./models/wavtokenizer-large-75-f16.gguf", model=..., params=...)
    at /home/danbev/work/ai/llama.cpp-debug/src/llama.cpp:73
#2  0x00007ffff7c664b9 in llama_model_load_from_file (path_model=0x555555e08710 "./models/wavtokenizer-large-75-f16.gguf", 
    params=...) at /home/danbev/work/ai/llama.cpp-debug/src/llama.cpp:9488
#3  0x00005555557464ed in common_init_from_params (params=...) at /home/danbev/work/ai/llama.cpp-debug/common/common.cpp:868
#4  0x000055555562fcd1 in server_context::load_model (this=0x7fffffffc2d0, params=...)
    at /home/danbev/work/ai/llama.cpp-debug/examples/server/server.cpp:1687
#5  0x00005555555e8031 in main (argc=8, argv=0x7fffffffda08) at /home/danbev/work/ai/llama.cpp-debug/examples/server/server.cpp:4228
```
This voice decoder model does not have a vocabulary, it does not have an array
of `tokenizer.ggml.tokens`, so the values returned from `model.vocab.n_tokens()`
will be zero.
But the model model does has the following property:
```console
wavtokenizer-dec.vocab_size
```
And this matches the exptected size of `tok_embd`:
```console
    111:    2097152 |   512,  4096,     1,     1 | F16     | token_embd.weight
```
So perhaps this values should be set a hparam when these types of model are
loaded.

But there will also be a need to a change to `llama_decode_impl` where the
`n_tokens` is also used:
```c++
    if (batch.token) {
        for (uint32_t i = 0; i < n_tokens_all; ++i) {
            if (batch.token[i] < 0 || (uint32_t) batch.token[i] >= model.vocab.n_tokens()) {
                LLAMA_LOG_ERROR("%s: invalid token[%d] = %d\n", __func__, i, batch.token[i]);
                return -1;
            }
        }
    }
```
There might be other places where a similar situation arises as well.

As a quick test to see if this would work the following changes enable the
tts example to run again:
```console
diff --git a/src/llama-hparams.h b/src/llama-hparams.h
index 1fe45410..c641ff2a 100644
--- a/src/llama-hparams.h
+++ b/src/llama-hparams.h
@@ -45,6 +45,7 @@ struct llama_hparams {
     // for WavTokenizer
     struct llama_hparams_posnet   posnet;
     struct llama_hparams_convnext convnext;
+    uint32_t wav_n_vocab = 0;
 
     std::array<uint32_t, LLAMA_MAX_LAYERS> n_head_arr;
     std::array<uint32_t, LLAMA_MAX_LAYERS> n_head_kv_arr;
diff --git a/src/llama-model.cpp b/src/llama-model.cpp
index f90f5e74..5b30154c 100644
--- a/src/llama-model.cpp
+++ b/src/llama-model.cpp
@@ -421,6 +421,7 @@ void llama_model::load_hparams(llama_model_loader & ml) {
 
         ml.get_key(LLM_KV_CONVNEXT_EMBEDDING_LENGTH, hparams.convnext.n_embd);
         ml.get_key(LLM_KV_CONVNEXT_BLOCK_COUNT,      hparams.convnext.n_layer);
+        ml.get_key(LLM_KV_VOCAB_SIZE,                hparams.wav_n_vocab);
     }
 
     GGML_ASSERT(hparams.n_expert <= LLAMA_MAX_EXPERTS);
@@ -3247,7 +3248,7 @@ bool llama_model::load_tensors(llama_model_loader & ml) {
                 } break;
             case LLM_ARCH_WAVTOKENIZER_DEC:
                 {
-                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {hparams.n_embd_features, n_vocab}, 0);
+                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {hparams.n_embd_features, hparams.wav_n_vocab}, 0);
 
                     conv1d   = create_tensor(tn(LLM_TENSOR_CONV1D, "weight"), {7, hparams.n_embd_features, hparams.posnet.n_embd}, 0);
                     conv1d_b = create_tensor(tn(LLM_TENSOR_CONV1D, "bias"),   {1, hparams.posnet.n_embd}, 0);
diff --git a/src/llama.cpp b/src/llama.cpp
index daf1b7c9..15769dc1 100644
--- a/src/llama.cpp
+++ b/src/llama.cpp
@@ -8471,8 +8471,10 @@ static int llama_decode_impl(
     if (batch.token) {
         for (uint32_t i = 0; i < n_tokens_all; ++i) {
             if (batch.token[i] < 0 || (uint32_t) batch.token[i] >= model.vocab.n_tokens()) {
-                LLAMA_LOG_ERROR("%s: invalid token[%d] = %d\n", __func__, i, batch.token[i]);
-                return -1;
+                if (model.arch != LLM_ARCH_WAVTOKENIZER_DEC) {
+                    LLAMA_LOG_ERROR("%s: invalid token[%d] = %d\n", __func__, i, batch.token[i]);
+                    return -1;
+                }
             }
         }
     }

```

```console
llama_model_load: error loading model: check_tensor_dims: tensor 'token_embd.weight' has wrong shape; expected   512,     0, got   512,  4096,     1,     1
llama_model_load_from_file: failed to load model
common_init_from_params: failed to load model './models/wavtokenizer-large-75-f16.gguf'
```
I opened https://github.com/ggerganov/llama.cpp/issues/11229 for this issue
which has now been fixed.

So back to the server example, we need to servers running, one for the LLM model
and the other for the voice decoder model. The LLM model server is started with:
```console
$ ./build/bin/llama-server -m ./models/outetts-0.2-0.5B-q8_0.gguf -ngl 99 --port 8020
```
```console
./build/bin/llama-server -m ./models/wavtokenizer-large-75-f16.gguf --port 8021 --embeddings --pooling none
```

```console
$ python3 -m venv venv
$ source venv/bin/activate
(venv) pip install requests
(venv) python ./examples/tts/tts-outetts.py http://localhost:8020 http://localhost:8021 "Hello world"  
```
This is not completely implements, most notably `embd_to_audio` which we covered
above in c++ is not implemented in the python version.
