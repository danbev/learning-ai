## Conformer-based Automatic Speech Recognition (ASR) Models
These models are hybrid architectures that combine convolutional neural networks
(CNNs) and transformers (conv-formers).


### Model architecture
The flow would look something like this:
```
Input (e.g., 50×512)
    |
    ↓
[1st Feed-Forward Module]  (512 → 2048 → 512)
    |
    | (residual connection + layer norm)
    ↓
[Multi-Head Self-Attention]  (8 heads, key/query/value)
    |
    | (residual connection + layer norm)
    ↓
[Convolution Module]  (depthwise conv1d, kernel size 31)
    |
    | (residual connection + layer norm)
    ↓
[2nd Feed-Forward Module]  (512 → 2048 → 512)
    |
    | (residual connection + layer norm)
    ↓
Output (50×512)
```

### Processing audio input
So lets say we have a hello.wav, which is just a recording of me saying hello
and we would read this in to a buffer:
```cpp
    if (!::read_audio_data(fname_inp, pcmf32, pcmf32s, params.diarize)) {
        fprintf(stderr, "error: failed to read audio file '%s'\n", fname_inp.c_str());
    }
```
This will create a buffer of floats, which is the audio data. The size of the
buffer is the number of samples in the audio file:
```console
(gdb) p pcmf32.size()
$2 = 472373
```

These are then passed to `whisper_full_parallel`:
```cpp

    if (whisper_full_parallel(ctx, wparams, pcmf32.data(), pcmf32.size(), params.n_processors) != 0) {
        fprintf(stderr, "%s: failed to process audio\n", argv[0]);
        return 1;
    }
```

Later the samples will be converted to log mel spectrograms:
```cpp
        // compute log mel spectrogram
        if (whisper_pcm_to_mel_with_state(ctx, state, samples, n_samples, params.n_threads) != 0) {
            WHISPER_LOG_ERROR("%s: failed to compute log mel spectrogram\n", __func__);
            return -2;
        }
```
This will create the mel spectrograms on the whisper state.
```console
(gdb) ptype state->mel
type = struct whisper_mel {
    int n_len;
    int n_len_org;
    int n_mel;
    std::vector<float> data;
}

(gdb) p state->mel.n_len
$7 = 5952

(gdb) p  state->mel.data.size()
$8 = 476160

(gdb) p state->mel
$4 = {n_len = 5952, n_len_org = 2952, n_mel = 80, data = std::vector of length 476160, capacity 476160
```
This would then be set as the input in the encoder which in the case of a
conformer would be a conformer encoder instead of the normal whisper encoder.
