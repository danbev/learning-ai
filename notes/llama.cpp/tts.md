## Text To Speech (TTS)
In llama.cpp there a new feature has been added that enabled text to speech
and there is an accompanying example named tts.

### Background
The model OuteTTS-0.2-500M, where Oute just stand for the company/org which
produced the model, is described as:
```
An experimental text-to-speech model that uses a pure language modeling approach
to generate speech, without architectural changes to the foundation model itself
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
So this is a multi-modal LLM model, `qwen2`.

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
```
So a wav tokenizer model has a vocabulary size of 4096 which are the codes
for the audio. More about these codes will be discussed later in this document.

_wip_ continue here.


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

Now, in a "normal" LLM we process a prompt be splittig it into tokens and then
generating embeddings (looking them up) and that is what is passed to the model.
But for a text to speech model it needs to keep the words in tact it requires
that there is a token to separate the words. For example, if we have a prompt
like "What is LoRA?" this would get changed into:
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


This is also prompt conditioning and it is used to give the model some context
about the rhythm of speech and how words are pronounced. It give it a sense of
how long each word should take to pronounce. What audio tokens  create natural
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

