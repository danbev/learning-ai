## whisper.cpp
This is a automatic speech recognition (ASR) library for C/C++.

### Spectrogram
When we process raw audio it is in a wave form which is in the time domain. We
see a single amplitude at each point in time. This amplitude represents the 
total sum of all frequencies at that point in time.

![image](./images/waveform.png)

The wave form is a continuous signal in time and in in amplitude. To represent
this information in a digital form we need to sample it, that is read specific
points (or rather intervals) and store them. These intervalls are ofter denoted
by T, the time between each sample. The number of samples per second is called
the sample rate. The sample rate is often 44.1 kHz or 44,100 samples per second.
Now, if we sample with a low rate we run the risk of losing information so I
though that using a higher sample rate would always be better. But this is not
the case. There is a theorem called the Nyquist-Shannon theorem which states
that the sample rate must be at least twice the highest frequency in the signal
to accurately represent it. So if we have a signal with a maximum frequency of
22 kHz we need a sample rate of at least 44.1 kHz. This is why the sample rate
is often 44.1 kHz. The human ear can hear frequencies up to 20 kHz so this is
why the sample rate is often 44.1 kHz.
Speach is often between 80-8000Hz and music 20-20000Hz.

In this recording I'm just saying my name: "Daniel Bevenius"
I can see that the frequency is not higher than 8000hz which is the limit for
human speech. I start with a frequency at around 100h, which is my fundamental
frequency F0. This is the base vibration rate of my vocal folds.
The "D" sound doesn't have very high amplitude in the waveform because it's what
is call a "stop consonant" - it involves briefly stopping airflow before
releasing it. They tend to be quieter than vowels.
The parallel horizontal lines above the fundamental frequency (at ~200Hz, ~300Hz,
etc.) - these are the harmonics.

"a" in "Daniel" and "e" in "Bevenius" are vowels and they have a lot of energy
in the higher frequencies. The "a" sound has a lot of energy at around 800Hz and

Consonants like "n", "l", "b", "v" each have distinctive patterns. By using
the spectrogram it is actually possible to "read" what letters are being spoken
and this is what the ASR system use. But the ARS systems are trained on a lot
of millions of examples and use statistical models to predict what is being
said.

The Fourier Transform decomposes this signal into its constituent frequencies.
The x-axis is time just like in the wave form, but the y-axis is frequency (not
amplitude). So the spectrogram is showing the whole spectrum of frequencies
possible for each point in time. Now, the color of each point, the point at
time/frequence (x, y) intersection represents the amplitude/energy at that point.

![image](./images/spectrogram.png)

Whisper uses mel spectrogram which is a spectrogram where the frequencies are
converted to the mel scale. The mel scale is a scale of pitches that are
perceived by humans as being equally spaced.

### Mel Spectrograms
In the spectrogram we saw above the y-axis was the frequency in Hz. But this is
not great for humans because we don't perceive frequencies linearly. We perceive
frequencies logarithmically. So what do we mean by that?  
Well, if we have a sound at 1000Hz and another at 2000Hz the difference between
them is 1000Hz. Now, if we have a sound at 10000Hz and another at 11000Hz the
difference is is also 1000Hz. But we perceive the difference between 1000Hz and
2000Hz as being much greater than the difference between 10000Hz and 11000Hz. So
the mel scale is a scale that is designed to mimic the way humans perceive sound.
The mel scale is a logarithmic scale that is linear below 1000Hz and logarithmic
above 1000Hz.

So to recap a little and get an overview of the process:  
1. We start with the original analog signal which is continuous in time and
   amplitude.

2. Digitalize the signal by sampling it at a certain rate to create a digital
   waveform. This is now digital (descrete) but still just amplitude values over
   time.
 
3. Standard spectrogram: The digitized audio undergoes a Short-Time Fourier
   Transform (STFT) to break it into frequency components. Say we sample at a
   rate of 16Khz this would show frequencies up to 8Khz (Nyquist limit).
   A standard spectrogram might have frequency resolution of 10Hz per bin,
   resulting in 800 bins for 8000Hz.

4. Mel spectrogram: This is where the data reduction happens. Instead of keeping
   all the 800 frequency bins, we combine them into just 80 mel scaled bins.
   In this process of converting from the spectrogram to the mel spectrogram we
   apply a set of overlapping triangular filters to the spectrogram which are
   called mel filterbanks. These filters combine multiple frequency bins into
   a single mel bin.

   For example, to create mel bin #10, we might take a weighted average of linear
   frequency bins 50-70, giving us a single value that represents that entire
   frequency range.
   After this process, each time frame now has only 80 amplitude values instead
   of 800.

Visualize this as a matrix:
```
spectrogram:
              800 (frequency bins)
        +-----------------------------+
        |                             |
        |                             |
        |                             |  500 (time frames)
        |                             |
        |                             |
        |                             |
        |                             |
        +-----------------------------+
                       |
                       ↓ 
mel spectrogram:
              80 (frequency bins)
        +------------+
        |            |
        |            |
        |            |  500 (time frames)
        |            |
        |            |
        |            |
        |            |
        +------------+
```
This is like resizing an image from 800x500 to 80x500. Like we are compressing
the data and it will take up less space. But also it will contain more of the
information that is important to humans speech which is what we want for 
whisper.

For example, in our above spectrogram we had 8000Hz as the max frequency. And
these are evenly divided into ranges, lets say there might be 800+ frequency
ranges (or bins) for 8000Hz.

This might get divided into 80 "bins/buckets" in the mel spectrogram:
```
Bin 1-20:     0-1000Hz (about 50Hz per bin)
Bin 21-40: 1000-2500Hz (about 75Hz per bin)
Bin 41-60: 2500-5000Hz (about 125Hz per bin)
Bin 61-80: 5000-8000Hz (about 150Hz per bin)
```

Converting to mel spectrograms reduces the input data dimensionality while
preserving perceptually important information.

### Inference Processing
The raw audio is first split into smaller segment of 30 second chunks. This 30
second limit comes from Whispers training where 30 second segments where used
and its position embeddings are designed for this duration.

Then for each chunk:
The chunk is converted into a mel spectrogram which is then processed by the
encoder.

The decoder starts with initial tokens:
* `<SOT>` (Start of Transcript),
* language token (like `<EN>`),
* task token which will be one of:
  - `<TRANSCRIBE>`
  - `<TRANSLATE>`
* no timestamp token

The decoder generates one token at a time.

For a single 30-second chunk, the result will be the transcribed tokens
corresponding to the speech in that 30-second segment. 

The special tokens (like language markers, timestamp indicators) provide task
control. By intermixing these with the text, Whisper can perform multiple tasks
with a single model, directing it to transcribe, translate, or identify the
language.

In whisper.cpp, these tokens are setup in the `whisper_full_with_state` function:
```c++
int whisper_full_with_state(
        struct whisper_context * ctx,
          struct whisper_state * state,
    struct whisper_full_params   params,
                   const float * samples,
                           int   n_samples) {
    ...

    std::vector<whisper_token> prompt_init = { whisper_token_sot(ctx), };
```
And we can inspect the tokes added when translation is enabled using the
command line option `-tr`:
```console
(std::vector<int>) size=3 {
  [0] = 50258
  [1] = 50259
  [2] = 50358
}
(lldb) p ctx->vocab.id_to_token.at(50258)
(std::map<int, std::string>::mapped_type) "[_SOT_]"
(lldb) p ctx->vocab.id_to_token.at(50259)
(std::map<int, std::string>::mapped_type) "[_LANG_en]"
(lldb) p ctx->vocab.id_to_token.at(50358)
(std::map<int, std::string>::mapped_type) "[_TRANSLATE_]"
```
Transcribe is token:
```console
(lldb) p ctx->vocab.id_to_token.at(50359)
(std::map<int, std::string>::mapped_type) "[_TRANSCRIBE_]"
```

In Whisper models (including implementations like whisper.cpp), the task token (<|transcribe|> or <|translate|>) fundamentally changes what the model does with the audio input. These are mutually exclusive paths in the model's processing:

<|transcribe|> instructs the model to output text in the same language as the audio
<|translate|> instructs the model to translate the speech into English

To get both the original transcription and an English translation, you would indeed need to run two separate inference passes over the same audio:

First pass with <|transcribe|> to get the original language transcription
Second pass with <|translate|> to get the English translation

This is a fundamental limitation of how the model was designed and trained. It's not simply a software limitation that could be worked around in the implementation - the model architecture itself expects to perform one task at a time.

### Diarization
There is a parameter named 'diarize' which indicates if speaker identification
or diarization should be performed. This is about who spoke when. The system
will attempt to identify the speaker and assign a label to each speaker. So
the output will have an identifier like "Speaker 0", "Speaker 1", etc.


### Dynamic Time Warping (DTW)
This is about aligning the transcribed text with precise timestamps in the
audio. When whisper generated text from audio it needs to determine preciely
when each word was spoken. But the encoder-decoder model does not have this
concept of timestamps. 


### Pulse Code Modulation (PCM)
So when we a raw audio signal which is continuous, we sample it at a certain
fixed rate called the sample rate. This is measuring the amplitude of the sound
wave at regular intervals. Each value is then quantied to a fixed number of bits
, the number is determined by the bit depth. This gives us a discrete value.
```
8  bits = 2^8  = 256 levels
16 bits = 2^16 = 65536 levels
24 bits = 2^24 = 16777216 levels
32 bits = 2^32 = 4294967296 levels
```
These quantized values are the codes in Pulse Code Modulation (PCM). Each code
represents the amplitude of the sound wave at that point in time. This data
can then be stored in a file and for a WAV file the header will contain metadata
like the sample rate, bit depth, number of channels, etc.

### Waveform Audio File Format (WAV)
This is a subset of Microsoft's Microsoft’s Resource Interchange File Format (RIFF)
specification for the storage of digital audio. There is no compression involved
in this format.

We can inspect the wav metadata using the `mediainfo` command:
```console
$ hexdump -C -n 500 samples/jfk.wav
00000000  52 49 46 46 46 5f 05 00  57 41 56 45 66 6d 74 20  |RIFFF_..WAVEfmt |
00000010  10 00 00 00 01 00 01 00  80 3e 00 00 00 7d 00 00  |.........>...}..|
00000020  02 00 10 00 4c 49 53 54  1a 00 00 00 49 4e 46 4f  |....LIST....INFO|
00000030  49 53 46 54 0e 00 00 00  4c 61 76 66 35 39 2e 32  |ISFT....Lavf59.2|
00000040  37 2e 31 30 30 00 64 61  74 61 00 5f 05 00 00 00  |7.100.data._....|
00000050  00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00  |................|
```
whisper.cpp uses [miniaudio](https://github.com/mackron/miniaudio) to read wav
files.

### Model

```c++
struct whisper_model {
    e_model type = MODEL_UNKNOWN;

    whisper_hparams hparams;
    whisper_filters filters;

    // encoder.positional_embedding
    struct ggml_tensor * e_pe;

    // encoder.conv1
    struct ggml_tensor * e_conv_1_w;
    struct ggml_tensor * e_conv_1_b;

    // encoder.conv2
    struct ggml_tensor * e_conv_2_w;
    struct ggml_tensor * e_conv_2_b;

    // encoder.ln_post
    struct ggml_tensor * e_ln_w;
    struct ggml_tensor * e_ln_b;

    // decoder.positional_embedding
    struct ggml_tensor * d_pe;

    // decoder.token_embedding
    struct ggml_tensor * d_te;

    // decoder.ln
    struct ggml_tensor * d_ln_w;
    struct ggml_tensor * d_ln_b;

    std::vector<whisper_layer_encoder> layers_encoder;
    std::vector<whisper_layer_decoder> layers_decoder;

    // ggml context that contains all the meta information about the model tensors
    struct ggml_context * ctx = nullptr;

    // the model backend data is read-only and can be shared between processors
    ggml_backend_buffer_t buffer = nullptr;

    // tensors
    int n_loaded;
    std::map<std::string, struct ggml_tensor *> tensors;
};
```
In `whisper_model_load` the model tensor are created:
```c++
        // encoder
        {
            model.e_pe = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_audio_state, n_audio_ctx);
```
This is the encoders position encoder (pe)
```console
(gdb) p model.e_pe.ne
$35 = {384, 1500, 1, 1}

(gdb) p model.hparams.n_audio_state 
$9 = 384
(gdb) p model.hparams.n_audio_ctx 
$10 = 1500
```
So we can see this is a matrix with 384 dimensions and 1500 rows.
```
0     [0                  383]
            .
            .
            .
1499  [0                  383]
```
Audio is processed at 50 frames per second so 1500 frames corresponds to 30
seconds of audio (1500/50 = 30). So each row in this matrix represents a specific
time position the 30 second audio chunk or segment. The positional information
is added so that the model know not only the spectral information (which frequencies
are present) but also when they occur.

Next we have the two 1D convolutions:
```c++
            model.e_conv_1_w     = ggml_new_tensor_3d(ctx, vtype,         3, n_mels,     n_audio_state);
            model.e_conv_1_b     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,         1,     n_audio_state);

            model.e_conv_2_w     = ggml_new_tensor_3d(ctx, vtype,         3, n_audio_state, n_audio_state);
            model.e_conv_2_b     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,                1, n_audio_state);
```
```console
(gdb) p model.e_conv_1_w->ne
$12 = {3, 80, 384, 1}

(gdb) p model.e_conv_2_w->ne
$13 = {3, 384, 384, 1}
```
TODO: Add more information about the convolutions.

Next, we have the encoder layers.
And following that we have the decoder tensors.
```c++
        // decoder
        {
            model.d_pe   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_text_state, n_text_ctx);

            model.d_te   = ggml_new_tensor_2d(ctx, wtype,         n_text_state, n_vocab);
```
These are the decoders positional embedding and token embedding tensors.
```console
(gdb) p model.d_pe->ne
$19 = {384, 448, 1, 1}

(gdb) p model.d_te->ne
$20 = {384, 51864, 1, 1}
```
These tensors are later used in :
```c++
static struct ggml_cgraph * whisper_build_graph_conv(
        whisper_context & wctx,
          whisper_state & wstate) {
    ...
    struct ggml_tensor * mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 2*n_ctx, n_mels);
    ggml_set_name(mel, "mel");
    ggml_set_input(mel);

    struct ggml_tensor * cur = nullptr;

    if (!whisper_encode_external(wstate)) {
        // convolution + gelu
        {
            cur = ggml_conv_1d_ph(ctx0, model.e_conv_1_w, mel, 1, 1);
            cur = ggml_add(ctx0, cur, model.e_conv_1_b);

            cur = ggml_gelu(ctx0, cur);

            cur = ggml_conv_1d_ph(ctx0, model.e_conv_2_w, cur, 2, 1);
            cur = ggml_add(ctx0, cur, model.e_conv_2_b);

            cur = ggml_gelu(ctx0, cur);
        }

        ggml_set_name(cur, "embd_conv");
        wstate.embd_conv = cur;
```
```console
(gdb) p mel->ne
$31 = {3000, 80, 1, 1}
```

### whisper-cli
An initial walk through of the cli example to get familiar with the code.
```console
gdb --args ./build/bin/whisper-cli \
	-m models/ggml-tiny.en.bin \
	-f samples/jfk.wav \
	-di
```

```c++
int main(int argc, char ** argv) {
    whisper_params params;
    ...

    struct whisper_context_params cparams = whisper_context_default_params();
    ...
    struct whisper_context * ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);
}
```
```console
(gdb) set print pretty on
(gdb) p params
$4 = {
  n_threads = 4,
  n_processors = 1,
  offset_t_ms = 0,
  offset_n = 0,
  duration_ms = 0,
  progress_step = 5,
  max_context = -1,
  max_len = 0,
  best_of = 5,
  beam_size = 5,
  audio_ctx = 0,
  word_thold = 0.00999999978,
  entropy_thold = 2.4000001,
  logprob_thold = -1,
  no_speech_thold = 0.600000024,
  grammar_penalty = 100,
  temperature = 0,
  temperature_inc = 0.200000003,
  debug_mode = false,
  translate = false,
  detect_language = false,
  diarize = true,
  tinydiarize = false,
  split_on_word = false,
  no_fallback = false,
  output_txt = false,
  output_vtt = false,
  output_srt = false,
  output_wts = false,
  output_csv = false,
  output_jsn = false,
  output_jsn_full = false,
  output_lrc = false,
  no_prints = false,
  print_special = false,
  print_colors = false,
  print_progress = false,
  no_timestamps = false,
  log_score = false,
  use_gpu = true,
  flash_attn = false,
  suppress_nst = false,
  language = "en",
  prompt = "",
  font_path = "/System/Library/Fonts/Supplemental/Courier New Bold.ttf",
  model = "models/ggml-tiny.en.bin",
  grammar = "",
  grammar_rule = "",
  tdrz_speaker_turn = " [SPEAKER_TURN]",
  suppress_regex = "",
  openvino_encode_device = "CPU",
  dtw = "",
  fname_inp = std::vector of length 1, capacity 1 = {"samples/jfk.wav"},
  fname_out = std::vector of length 0, capacity 0,
  grammar_parsed = {
    symbol_ids = std::map with 0 elements,
    rules = std::vector of length 0, capacity 0
  }
}

gdb) p cparams
$5 = {
  use_gpu = true,
  flash_attn = false,
  gpu_device = 0,
  dtw_token_timestamps = false,
  dtw_aheads_preset = WHISPER_AHEADS_NONE,
  dtw_n_top = -1,
  dtw_aheads = {
    n_heads = 0,
    heads = 0x0
  },
  dtw_mem_size = 134217728
}

```
So lets take a look at the model is loaded:
```console
(gdb) br whisper_init_from_file_with_params_no_state
(gdb) c
```
```c++
struct whisper_context * whisper_init_from_file_with_params_no_state(const char * path_model,
    struct whisper_context_params params) {
    WHISPER_LOG_INFO("%s: loading model from '%s'\n", __func__, path_model);


```

### Beam Search
Lets take the whisper-cli as the example and see how the beam search works.
```c++
int main(int argc, char ** argv) {
    ...
    struct whisper_context * ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);
```
```c++
struct whisper_context * whisper_init_from_file_with_params(const char * path_model, struct whisper_context_params params) {
    whisper_context * ctx = whisper_init_from_file_with_params_no_state(path_model, params);
    if (!ctx) {
        return nullptr;
    }

    ctx->state = whisper_init_state(ctx);
    if (!ctx->state) {
        whisper_free(ctx);
        return nullptr;
    }

    return ctx;
}
```
```c++
struct whisper_context * whisper_init_from_file_with_params_no_state(const char * path_model, struct whisper_context_params params) {
    WHISPER_LOG_INFO("%s: loading model from '%s'\n", __func__, path_model);
#ifdef _MSC_VER
    // Convert UTF-8 path to wide string (UTF-16) for Windows, resolving character encoding issues.
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    std::wstring path_model_wide = converter.from_bytes(path_model);
    auto fin = std::ifstream(path_model_wide, std::ios::binary);
#else
    auto fin = std::ifstream(path_model, std::ios::binary);
#endif
    if (!fin) {
        WHISPER_LOG_ERROR("%s: failed to open '%s'\n", __func__, path_model);
        return nullptr;
    }

    whisper_model_loader loader = {};
    ...

    auto ctx = whisper_init_with_params_no_state(&loader, params);
```
```c++
struct whisper_context * whisper_init_with_params_no_state(struct whisper_model_loader * loader, struct whisper_context_params params) {
    ...

    whisper_context * ctx = new whisper_context;
    ctx->params = params;

    if (!whisper_model_load(loader, *ctx)) {
        loader->close(loader->context);
        WHISPER_LOG_ERROR("%s: failed to load model\n", __func__);
        delete ctx;
        return nullptr;
    }
}
```
```c++
static bool whisper_model_load(struct whisper_model_loader * loader, whisper_context & wctx) {
    WHISPER_LOG_INFO("%s: loading model\n", __func__);

    const int64_t t_start_us = ggml_time_us();

    wctx.t_start_us = t_start_us;

    auto & model = wctx.model;
    auto & vocab = wctx.vocab;

    // verify magic
    {
        uint32_t magic;
        read_safe(loader, magic);
        if (magic != GGML_FILE_MAGIC) {
            WHISPER_LOG_ERROR("%s: invalid model data (bad magic)\n", __func__);
            return false;
        }
    }
```
TODO: take a closer look at the verify magic check in combination with using a Core ML
model. This check forces there to be an ggml model even though there might be cases
where only a Core ML model is used.

The actual inference is started by `cli.cpp`:
```c++
    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    ...
    if (whisper_full_parallel(ctx, wparams, pcmf32.data(), pcmf32.size(), params.n_processors) != 0) {
        fprintf(stderr, "%s: failed to process audio\n", argv[0]);
        return 10;
    }
```
```c++
int whisper_full_parallel(
        struct whisper_context * ctx,
        struct whisper_full_params params,
        const float * samples,
        int n_samples,
        int n_processors) {
    if (n_processors == 1) {
        return whisper_full(ctx, params, samples, n_samples);
    }
```
```c++
int whisper_full(
        struct whisper_context * ctx,
    struct whisper_full_params   params,
                   const float * samples,
                           int   n_samples) {
    return whisper_full_with_state(ctx, ctx->state, params, samples, n_samples);
}
```
```c++
int whisper_full_with_state(
        struct whisper_context * ctx,
          struct whisper_state * state,
    struct whisper_full_params   params,
                   const float * samples,
                           int   n_samples) {
    ...
    if (n_samples > 0) {
        // compute log mel spectrogram
        if (whisper_pcm_to_mel_with_state(ctx, state, samples, n_samples, params.n_threads) != 0) {
            WHISPER_LOG_ERROR("%s: failed to compute log mel spectrogram\n", __func__);
            return -2;
        }
    }
}
```
```console
(lldb) p n_samples
(int) 176000
```

Next we have the temperatures:
```c++
    // a set of temperatures to use
    // [ t0, t0 + delta, t0 + 2*delta, ..., < 1.0f + 1e-6f ]
    std::vector<float> temperatures;
    if (params.temperature_inc > 0.0f) {
        for (float t = params.temperature; t < 1.0f + 1e-6f; t += params.temperature_inc) {
            temperatures.push_back(t);
        }
    } else {
        temperatures.push_back(params.temperature);
    }
```
```console
(std::vector<float>) size=6 {
  [0] = 0
  [1] = 0.200000003
  [2] = 0.400000006
  [3] = 0.600000024
  [4] = 0.800000011
  [5] = 1
}
```
These are described in the whisper paper. So we start with a temperature of 0,
and increment in approximately 0.2 steps up to 1.0.
So for a single inference, from the point of view of the caller, it would potentially cause
multiple inferences to happen if the heuristic (issues discovered in practice) happen.
From the caller's perspective, what appears to be a single transcription request could
potentially trigger multiple inference passes behind the scenes.

Next, the decoders are initalized:
```c++
    // initialize the decoders
    int n_decoders = 1;

    switch (params.strategy) {
        case WHISPER_SAMPLING_GREEDY:
            {
                n_decoders = params.greedy.best_of;
            } break;
        case WHISPER_SAMPLING_BEAM_SEARCH:
            {
                n_decoders = std::max(params.greedy.best_of, params.beam_search.beam_size);
            } break;
    };

    n_decoders = std::max(1, n_decoders);

    if (n_decoders > WHISPER_MAX_DECODERS) {
        WHISPER_LOG_ERROR("%s: too many decoders requested (%d), max = %d\n", __func__, n_decoders, WHISPER_MAX_DECODERS);
        return -4;
    }

    // TAGS: WHISPER_DECODER_INIT
    for (int j = 1; j < n_decoders; j++) {
        auto & decoder = state->decoders[j];

        decoder.sequence.tokens.reserve(state->decoders[0].sequence.tokens.capacity());

        decoder.probs.resize   (ctx->vocab.n_vocab);
        decoder.logits.resize  (ctx->vocab.n_vocab);
        decoder.logprobs.resize(ctx->vocab.n_vocab);
        decoder.logits_id.reserve(ctx->model.hparams.n_vocab);

        decoder.rng = std::mt19937(0);
    }
```
Then the prompt is prepared:
```c++
    // prepare prompt
    {
        std::vector<whisper_token> prompt_tokens;

        // initial prompt
        if (!params.prompt_tokens && params.initial_prompt) {
            prompt_tokens.resize(1024);
            int n_needed = whisper_tokenize(ctx, params.initial_prompt, prompt_tokens.data(), prompt_tokens.size());
            if (n_needed < 0) {
                prompt_tokens.resize(-n_needed);
                n_needed = whisper_tokenize(ctx, params.initial_prompt, prompt_tokens.data(), prompt_tokens.size());
            }
            prompt_tokens.resize(n_needed);
            params.prompt_tokens   = prompt_tokens.data();
            params.prompt_n_tokens = prompt_tokens.size();
        }

        // prepend the prompt tokens to the prompt_past
        if (params.prompt_tokens && params.prompt_n_tokens > 0) {
            // parse tokens from the pointer
            for (int i = 0; i < params.prompt_n_tokens; i++) {
                prompt_past.push_back(params.prompt_tokens[i]);
            }
            std::rotate(prompt_past.begin(), prompt_past.end() - params.prompt_n_tokens, prompt_past.end());
        }
    }
```
Then the tokens are set up:
```c++
    // these tokens determine the task that will be performed
    std::vector<whisper_token> prompt_init = { whisper_token_sot(ctx), };

    if (whisper_is_multilingual(ctx)) {
        const int lang_id = whisper_lang_id(params.language);
        state->lang_id = lang_id;
        prompt_init.push_back(whisper_token_lang(ctx, lang_id));
        if (params.translate) {
            prompt_init.push_back(whisper_token_translate(ctx));
        } else {
            prompt_init.push_back(whisper_token_transcribe(ctx));
        }
    }
```
`sot` is start of transcript. 

`whisper_state` contains a a list of `whisper_decoder`:
```c++
struct whisper_state {
    ...
    whisper_decoder decoders[WHISPER_MAX_DECODERS];
    ...
};
```
```c++
struct whisper_decoder {
    // the currently generated sequence of tokens
    whisper_sequence sequence;

    // grammar parse state of generated sequence of tokens
    whisper_grammar  grammar;

    int i_batch;    // the index of the token in the current batch
    int seek_delta; // the window shift found so far based on the decoded timestamp tokens

    bool failed;    // has the current segment failed to decode?
    bool completed; // has the decoder completed the current segment?
    bool has_ts;    // have we already sampled a non-beg timestamp token for the current segment?

    // new token probs, logits and logprobs after the last whisper_decode (1-dimensional array: [n_vocab])
    std::vector<float> probs;
    std::vector<float> logits;
    std::vector<float> logprobs;

    // work container used to avoid memory allocations
    std::vector<whisper_pair<double, whisper_vocab::id>> logits_id;

    mutable std::mt19937 rng; // used for sampling at t > 0.0
};

struct whisper_sequence {
    std::vector<whisper_token_data> tokens;

    // the accumulated transcription in the current iteration (used to truncate the tokens array)
    int result_len;

    double sum_logprobs_all; // the sum of the log probabilities of the tokens
    double sum_logprobs;     // the sum of the log probabilities of the tokens (first result_len tokens)
    double avg_logprobs;     // the average log probability of the tokens
    double entropy;          // the entropy of the tokens
    double score;            // likelihood rank score
};
```

```console
-bo N,     --best-of N         [5      ] number of best candidates to keep
-bs N,     --beam-size N       [5      ] beam size for beam search
```

```c++
int whisper_full_with_state(
        struct whisper_context * ctx,
          struct whisper_state * state,
    struct whisper_full_params   params,
                   const float * samples,
                           int   n_samples) {
    ...
    // initialize the decoders
    int n_decoders = 1;

    switch (params.strategy) {
        case WHISPER_SAMPLING_GREEDY:
            {
                n_decoders = params.greedy.best_of;
            } break;
        case WHISPER_SAMPLING_BEAM_SEARCH:
            {
                n_decoders = std::max(params.greedy.best_of, params.beam_search.beam_size);
            } break;
    };

    n_decoders = std::max(1, n_decoders);

    if (n_decoders > WHISPER_MAX_DECODERS) {
        WHISPER_LOG_ERROR("%s: too many decoders requested (%d), max = %d\n", __func__, n_decoders, WHISPER_MAX_DECODERS);
        return -4;
    }

    // TAGS: WHISPER_DECODER_INIT
    for (int j = 1; j < n_decoders; j++) {
        auto & decoder = state->decoders[j];

        decoder.sequence.tokens.reserve(state->decoders[0].sequence.tokens.capacity());

        decoder.probs.resize   (ctx->vocab.n_vocab);
        decoder.logits.resize  (ctx->vocab.n_vocab);
        decoder.logprobs.resize(ctx->vocab.n_vocab);
        decoder.logits_id.reserve(ctx->model.hparams.n_vocab);

        decoder.rng = std::mt19937(0);
    }
    ...
    struct beam_candidate {
        int decoder_idx;  // which decoder this candidate came from.
        int seek_delta;   // position in the audio?

        bool has_ts;      // has timestamp information.

        whisper_sequence sequence; // the token sequence for this candidate
        whisper_grammar grammar;   // the grammar for this candidate
    };

    std::vector<std::vector<beam_candidate>> bc_per_dec(n_decoders);
    std::vector<beam_candidate> beam_candidates;
   
    ...

    struct beam_candidate {
        int decoder_idx;
        int seek_delta;

        bool has_ts;

        whisper_sequence sequence;
        whisper_grammar grammar;
    };

    std::vector<std::vector<beam_candidate>> bc_per_dec(n_decoders);
    std::vector<beam_candidate> beam_candidates;

    // main loop
    while (true) {
        if (params.progress_callback) {
            const int progress_cur = (100*(seek - seek_start))/(seek_end - seek_start);

            params.progress_callback(
                ctx, state, progress_cur, params.progress_callback_user_data);
        }

        // if only 1 second left, then stop
        if (seek + 100 >= seek_end) {
            break;
        }

        if (params.encoder_begin_callback) {
            if (params.encoder_begin_callback(ctx, state, params.encoder_begin_callback_user_data) == false) {
                WHISPER_LOG_ERROR("%s: encoder_begin_callback returned false - aborting\n", __func__);
                break;
            }
        }

        // encode audio features starting at offset seek
        if (!whisper_encode_internal(*ctx, *state, seek, params.n_threads, params.abort_callback, params.abort_callback_user_data)) {
            WHISPER_LOG_ERROR("%s: failed to encode\n", __func__);
            return -6;
        }
```
So here we have the inference loop. First the encoder is called which is taking the log mel
spectrogram and passing it to the encoder part of th model. A nice diagram of this can be
found on page 4 of the paper. First there is a Cov1D layer followed by a GELU activation,
and then another Conv1D layer followed by another GELU activation. The output of this then
as position encodings added to it.
This function is also what delegates to the Core ML or OpenVINO external encoders if one
of them are enabled.

```c++
const auto tokens_new = whisper_sample_token_topk(*ctx, decoder, params.beam_search.beam_size);
```
```console
(lldb) p params.beam_search.beam_size
(int) 5

(lldb) p tokens_new
(const std::vector<whisper_token_data>) size=5 {
  [0] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [1] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [2] = (id = 50365, tid = 50365, p = 0.00626884214, plog = -5.07216358, pt = 0.00626884214, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [3] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [4] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
}
```

```c++
// init new transcription with sot, language (opt) and task tokens
prompt.insert(prompt.end(), prompt_init.begin(), prompt_init.end());
```
```console
(lldb) p prompt
(std::vector<int>) size=1 {
  [0] = 50257
}
```
Then we have the decoding part of the model:
```c++
                whisper_kv_cache_clear(state->kv_self);

                whisper_batch_prep_legacy(state->batch, prompt.data(), prompt.size(), 0, 0);

                if (!whisper_decode_internal(*ctx, *state, state->batch, params.n_threads, false, params.abort_callback, params.abort_callback_user_data)) {
                    WHISPER_LOG_ERROR("%s: failed to decode\n", __func__);
                    return -8;
                }
```
```c++
                // Calculate no_speech probability after first decode.
                // This has to be done before any logit filtering. Hence we cannot use the probs from the whisper_process_logits.
                {
                    const int n_logits = ctx->vocab.id_to_token.size();
                    std::vector<float> logprobs(n_logits);
                    std::vector<float> probs(n_logits);

                    whisper_compute_logprobs(state->logits, n_logits, logprobs);
                    whisper_compute_probs(state->logits, n_logits, logprobs, probs);
                    state->no_speech_prob = probs[whisper_token_nosp(ctx)];
                }
```
```c++
                {
                    const int64_t t_start_sample_us = ggml_time_us();

                    state->decoders[0].i_batch = prompt.size() - 1;

                    whisper_process_logits(*ctx, *state, state->decoders[0], params, t_cur);

                    for (int j = 1; j < n_decoders_cur; ++j) {
                        auto & decoder = state->decoders[j];

                        whisper_kv_cache_seq_cp(state->kv_self, 0, j, -1, -1);

                        memcpy(decoder.probs.data(),    state->decoders[0].probs.data(),    decoder.probs.size()*sizeof(decoder.probs[0]));
                        memcpy(decoder.logits.data(),   state->decoders[0].logits.data(),   decoder.logits.size()*sizeof(decoder.logits[0]));
                        memcpy(decoder.logprobs.data(), state->decoders[0].logprobs.data(), decoder.logprobs.size()*sizeof(decoder.logprobs[0]));
                    }

                    state->t_sample_us += ggml_time_us() - t_start_sample_us;
                }
```

Notice that this is passing the first decoder to the `whisper_process_logits` function:
```c++
static void whisper_process_logits(
              struct whisper_context & ctx,
               struct whisper_state  & state,
              struct whisper_decoder & decoder,
    const struct whisper_full_params   params,
                               float   temperature) {
        ...
        if (params.suppress_blank) {
            if (is_initial) {
                logits[vocab.token_eot]           = -INFINITY;
                logits[vocab.token_to_id.at(" ")] = -INFINITY;
            }
        }
```
And in this case the filtering is basically just setting the vocab tokens for these specific
tokens to negative infinity so they will not have any influence on the logprobability calculation?
In this case the filter is preventing the model from starting a transcription with a blank space
or an end-of-transcript token.
This is also how timestamps are not generated:
```c++
        if (params.no_timestamps) {
            for (int i = vocab.token_beg; i < n_logits; ++i) {
                logits[i] = -INFINITY;
            }
        }
```
When we set a logit which recall is the raw values (unnormalized probabilities) from the
final layer of the decoder. They can range from very negative to very positive values.
So they are not constrained to any range like [0,1] and do not sum to 1. There basically
"confidence scores" that indicate how strongly the model believes that a particular token
should be the next token in the sequence.

When we set a logit to negative infinity, we are effectively saying that the probability
of that token is zero. This is a way to filter out tokens that we do not want to consider
in the decoding process.
```c++
        whisper_compute_logprobs(logits, n_logits, logprobs);
```

```c++
static void whisper_compute_logprobs(
                const std::vector<float> & logits,
                              const int    n_logits,
                      std::vector<float> & logprobs) {
    const float logit_max = *std::max_element(logits.begin(), logits.end());
    float logsumexp = 0.0f;
    for (int i = 0; i < n_logits; ++i) {
        if (logits[i] > -INFINITY) {
            logsumexp += expf(logits[i] - logit_max);
        }
    }
    logsumexp = logf(logsumexp) + logit_max;

    for (int i = 0; i < n_logits; ++i) {
        if (logits[i] > -INFINITY) {
            logprobs[i] = logits[i] - logsumexp;
        } else {
            logprobs[i] = -INFINITY;
        }
    }
}
```
So we first get the maximum logit value. This is used for numerical stability, notice
that is is subtracted from the logits before exponentiation. And also notice that
logits with the value of -INFINITY are not considered in the calculation, which is
what the filtering above is for.

After processing the logits the decoders (starting from 1) are updated:
```c++
                    for (int j = 1; j < n_decoders_cur; ++j) {
                        auto & decoder = state->decoders[j];

                        whisper_kv_cache_seq_cp(state->kv_self, 0, j, -1, -1);

                        memcpy(decoder.probs.data(),    state->decoders[0].probs.data(),    decoder.probs.size()*sizeof(decoder.probs[0]));
                        memcpy(decoder.logits.data(),   state->decoders[0].logits.data(),   decoder.logits.size()*sizeof(decoder.logits[0]));
                        memcpy(decoder.logprobs.data(), state->decoders[0].logprobs.data(), decoder.logprobs.size()*sizeof(decoder.logprobs[0]));
                    }
```
So this is setting the logits which are the raw values (unnormalized probabilities). And also
the logprobs which are calculated by:
```c++
logprobs[i] = log(exp(logits[i]) / sum(exp(logits)))
```
Logprobs will always be <= 0.0, with 0 representing a probability of 100% probability and
very negative values representing probilities close to 0. These are useful for working with
scoring when computing scores and ranking in beam search.

And the decoders also have the actual normalized probabilites of each token. This is obtained
by applying the softmax function to the logits.

The we have the process function()
```c++

case whisper_sampling_strategy::WHISPER_SAMPLING_BEAM_SEARCH:
    {
        const auto tokens_new = whisper_sample_token_topk(*ctx, decoder, params.beam_search.beam_size);

        for (const auto & token : tokens_new) {
            bc_per_dec[j].push_back({ j, decoder.seek_delta, decoder.has_ts, decoder.sequence, decoder.grammar, });
            bc_per_dec[j].back().sequence.tokens.push_back(token);
            bc_per_dec[j].back().sequence.sum_logprobs_all += token.plog;
        }
    } break;
```
So for each of the decoders we are going to call `whisper_sample_token_topk`:
```c++
static std::vector<whisper_token_data> whisper_sample_token_topk(
            whisper_context & ctx,
            whisper_decoder & decoder,
                        int   k) {
    ...
    std::discrete_distribution<> dist(probs.begin(), probs.end());

    for (int i = 0; i < k; ++i) {
        const auto id = dist(decoder.rng);
        //printf("XXX %d %d %f %f %f %f\n", id, tid, probs[id], logprobs[id], pt, ptsum);

        result.push_back({ id, tid, probs[id], logprobs[id], pt, ptsum, -1, -1, -1, 0.0f, });

        if (result[i].id >= vocab.token_beg) {
            result[i].tid = result[i].id;
            result[i].pt  = result[i].p;
        }
    }
```
Notice that this is using dist which is a discrete distribution. Each decoder has its own
random number generator so when sampling from this distribution, which is done k times,
the values will be different.
```c++
    std::discrete_distribution<> dist(probs.begin(), probs.end());
```
This creates a distribution where the probability of selecting token i is
proporitional to probs[i], the probability value. Higher probability values are more
likely and lower probability values are less likely.

```console
(lldb) p tokens_new
(const std::vector<whisper_token_data>) size=5 {
  [0] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [1] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [2] = (id = 50365, tid = 50365, p = 0.00626884214, plog = -5.07216358, pt = 0.00626884214, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [3] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [4] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
```
Now, recall that each sample is independent when sampled from the distribution and that the
token 50363 has a probability of 0.837067842. So it is not surprising that it is sampled
multiple times.

This is the third (j=1):
```console
(lldb) p tokens_new
(const std::vector<whisper_token_data>) size=5 {
  [0] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [1] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [2] = (id = 50365, tid = 50365, p = 0.00626884214, plog = -5.07216358, pt = 0.00626884214, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [3] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [4] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
}
```
This is the third (j=2):
```console
(lldb) p tokens_new
(const std::vector<whisper_token_data>) size=5 {
  [0] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [1] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [2] = (id = 50365, tid = 50365, p = 0.00626884214, plog = -5.07216358, pt = 0.00626884214, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [3] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [4] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
}
```
This is the third (j=3):
```console
(lldb) p tokens_new
(const std::vector<whisper_token_data>) size=5 {
  [0] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [1] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [2] = (id = 50365, tid = 50365, p = 0.00626884214, plog = -5.07216358, pt = 0.00626884214, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [3] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [4] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
}
```
This is the third (j=4):
```console
(lldb) p tokens_new
(const std::vector<whisper_token_data>) size=5 {
  [0] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [1] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [2] = (id = 50365, tid = 50365, p = 0.00626884214, plog = -5.07216358, pt = 0.00626884214, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [3] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
  [4] = (id = 50363, tid = 50363, p = 0.837067842, plog = -0.177850127, pt = 0.837067842, ptsum = 0.985369741, t0 = -1, t1 = -1, t_dtw = -1, vlen = 0)
}
```

