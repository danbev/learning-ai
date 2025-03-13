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
The decoder starts with initial tokens: `<SOT>`, language token (like `<EN>`),
task token (like `<TRANSCRIBE>`), and optionally timestamp token.
The decoder generates one token at a time.

For a single 30-second chunk, the result will be the transcribed tokens
corresponding to the speech in that 30-second segment. 

The special tokens (like language markers, timestamp indicators) provide task
control. By intermixing these with the text, Whisper can perform multiple tasks
with a single model, directing it to transcribe, translate, or identify the
language.


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
