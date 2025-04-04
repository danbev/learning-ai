## Voice Activity Detection (VAD)
Also knowas as speech activity detection (SAD) or speech detection.

Now, keep in mind that this is different than Automatic Speech Recognition (ASR)
which is the process of converting speech into text. VAD is used to determine
whether a segment of audio contains speech or not. It is often used as a
preprocessing step in ASR systems to filter out non-speech segments and reduce
the amount of data that needs to be processed. So it would be like a preprocessor
of an audio signal to remove silence or non-speech segments.
For example ASR systems may struggle with long periods of silence or noise, and
can output strange results if they are not filtered out.

So VAD should tell speech apart from noise and silence. It could be used in
mobil or IoT devices to detace human speech for example.
So the input is a small audio segment/chunk and the output is a probability
that this chunk contains speech.

### Silero-VAD
github: https://github.com/snakers4/silero-vad

The model that Silero-VAD has is not publicly available yet. I found this
discussion:
https://github.com/snakers4/silero-vad/discussions/371

But they do provide their model in two formats, one which I think is in a
PyTorch JIT (Just In Time) format and one in ONNX format.

We can get information from the jit model using the following script:
```console
$ cd audio/silero-vad
$ source venv/bin/activate
(venv) $ python src/jit-info.py
```
The output looks like this (taking it in pieces):
```
RecursiveScriptModule(
  original_name=VADRNNJITMerge
  (_model): RecursiveScriptModule(
    original_name=VADRNNJIT
    (stft): RecursiveScriptModule(
      original_name=STFT
      (padding): RecursiveScriptModule(original_name=ReflectionPad1d)
    )
    (encoder): RecursiveScriptModule(
      original_name=Sequential
      (0): RecursiveScriptModule(
        original_name=SileroVadBlock
        (se): RecursiveScriptModule(original_name=Identity)
        (activation): RecursiveScriptModule(original_name=ReLU)
        (reparam_conv): RecursiveScriptModule(original_name=Conv1d)
      )
      (1): RecursiveScriptModule(
        original_name=SileroVadBlock
        (se): RecursiveScriptModule(original_name=Identity)
        (activation): RecursiveScriptModule(original_name=ReLU)
        (reparam_conv): RecursiveScriptModule(original_name=Conv1d)
      )
      (2): RecursiveScriptModule(
        original_name=SileroVadBlock
        (se): RecursiveScriptModule(original_name=Identity)
        (activation): RecursiveScriptModule(original_name=ReLU)
        (reparam_conv): RecursiveScriptModule(original_name=Conv1d)
      )
      (3): RecursiveScriptModule(
        original_name=SileroVadBlock
        (se): RecursiveScriptModule(original_name=Identity)
        (activation): RecursiveScriptModule(original_name=ReLU)
        (reparam_conv): RecursiveScriptModule(original_name=Conv1d)
      )
    )
    (decoder): RecursiveScriptModule(
      original_name=VADDecoderRNNJIT
      (rnn): RecursiveScriptModule(original_name=LSTMCell)
      (decoder): RecursiveScriptModule(
        original_name=Sequential
        (0): RecursiveScriptModule(original_name=Dropout)
        (1): RecursiveScriptModule(original_name=ReLU)
        (2): RecursiveScriptModule(original_name=Conv1d)
        (3): RecursiveScriptModule(original_name=Sigmoid)
      )
    )
  )
  (_model_8k): RecursiveScriptModule(
    original_name=VADRNNJIT
    (stft): RecursiveScriptModule(
      original_name=STFT
      (padding): RecursiveScriptModule(original_name=ReflectionPad1d)
    )
    (encoder): RecursiveScriptModule(
      original_name=Sequential
      (0): RecursiveScriptModule(
        original_name=SileroVadBlock
        (se): RecursiveScriptModule(original_name=Identity)
        (activation): RecursiveScriptModule(original_name=ReLU)
        (reparam_conv): RecursiveScriptModule(original_name=Conv1d)
      )
      (1): RecursiveScriptModule(
        original_name=SileroVadBlock
        (se): RecursiveScriptModule(original_name=Identity)
        (activation): RecursiveScriptModule(original_name=ReLU)
        (reparam_conv): RecursiveScriptModule(original_name=Conv1d)
      )
      (2): RecursiveScriptModule(
        original_name=SileroVadBlock
        (se): RecursiveScriptModule(original_name=Identity)
        (activation): RecursiveScriptModule(original_name=ReLU)
        (reparam_conv): RecursiveScriptModule(original_name=Conv1d)
      )
      (3): RecursiveScriptModule(
        original_name=SileroVadBlock
        (se): RecursiveScriptModule(original_name=Identity)
        (activation): RecursiveScriptModule(original_name=ReLU)
        (reparam_conv): RecursiveScriptModule(original_name=Conv1d)
      )
    )
    (decoder): RecursiveScriptModule(
      original_name=VADDecoderRNNJIT
      (rnn): RecursiveScriptModule(original_name=LSTMCell)
      (decoder): RecursiveScriptModule(
        original_name=Sequential
        (0): RecursiveScriptModule(original_name=Dropout)
        (1): RecursiveScriptModule(original_name=ReLU)
        (2): RecursiveScriptModule(original_name=Conv1d)
        (3): RecursiveScriptModule(original_name=Sigmoid)
      )
    )
  )
)
```
There are two main components in the model, one named `_model` which takes care
of the 16kHz audio signal and the other one is named `_model_8k` which takes
care of the 8kHz audio signal. 

Both have the same layers but there tensor shapes might be different (more on
this later when we look at them).

#### STFT
So the first layer, `(sftf)` above, will take raw audio samples, 512 samples at
16kHz or 256 samples at 8kHz. The sample rate determines which model is to
be used.
```
audio samples -> STFT -> spectral features (129 frequency bins for 16kHz)
or 
audio samples -> STFT -> spectral features (85 frequency bins for 8kHz)
```

#### Encoder block
The there is an encoder, `(encoder)` above, block which has 4 layers:
```
spectral features → 
Conv1D → ReLU →             Expands to 128 channels
Conv1D → ReLU →             Reduces to 64 channels
Conv1D → ReLU →             Maintains 64 channels
Conv1D → ReLU →             Expands to 128 channels

Kernel size: 3
```

#### Decoder block
Then we have a decoder, `(decoder)` above, block which has 4 layers:
```
encoded features → LSTM Cell → Dropout → ReLU → Conv1D → Sigmoid → speech probability
```
Notice that this is using an LSTM so it is maintaining a hidden state.
The LSTM cell holds state between calls, allowing it to "remember" previous
audio frames. I was a little surprised to see an LSTM here as I read a blog
post prior to looking into Silero-VAD which contained:
```
A few days back we published a new totally reworked Silero VAD. You can try it
on your own voice via interactive demo with a video here or via basic demo here.
We employ a multi-head attention (MHA) based neural network under the hood with
the Short-time Fourier transform as features. This architecture was chosen due
to the fact that MHA-based networks have shown promising results in many
applications ranging from natural language processing to computer vision and
speech processing, but our experiments and recent papers show that you can
achieve good results with any sort of fully feedforward network, you just need
to do enough experiments (i.e. typical choices are MHA-only or transformer
networks, convolutional neural networks or their hybrids) and optimize the
architecture.
```
Perhaps this newer version has not been made available, or have I been looking
at an older version of the model perhaps?
TODO: Look into this and try to figure out what is going on.

The final sigmoid outputs probability (0-1) of speech presence.

For the `_model` we have the following parameters:
```
First encoder layer (input 129 frequency bins, output 128 channels, 3 kernel size),
and the bias for that layer:
_model.encoder.0.reparam_conv.weight, Shape: torch.Size([128, 129, 3])
_model.encoder.0.reparam_conv.bias, Shape: torch.Size([128])

_model.encoder.1.reparam_conv.weight, Shape: torch.Size([64, 128, 3])
_model.encoder.1.reparam_conv.bias, Shape: torch.Size([64])

_model.encoder.2.reparam_conv.weight, Shape: torch.Size([64, 64, 3])
_model.encoder.2.reparam_conv.bias, Shape: torch.Size([64])

_model.encoder.3.reparam_conv.weight, Shape: torch.Size([128, 64, 3])
_model.encoder.3.reparam_conv.bias, Shape: torch.Size([128])

The decoder LSTM cell has the following parameters:
_model.decoder.rnn.weight_ih, Shape: torch.Size([512, 128])
_model.decoder.rnn.weight_hh, Shape: torch.Size([512, 128])
_model.decoder.rnn.bias_ih, Shape: torch.Size([512])
_model.decoder.rnn.bias_hh, Shape: torch.Size([512])

Final output layer:
_model.decoder.decoder.2.weight, Shape: torch.Size([1, 128, 1])
_model.decoder.decoder.2.bias, Shape: torch.Size([1])
```

### Whisper.cpp integration
So the idea is that the raw audio file is processed by a VAD model, simliar to
how an image encoder might process an image into patch embedding tokens and the
pass those to an LLM. In this case what the VAD model does is process the raw
audio and outputs the timestamps where there is speech in the audio:
```
[
    {'start': 0.3, 'end': 2.2},
    {'start': 3.3, 'end': 3.8},
    {'start': 4.0, 'end': 4.3},
    {'start': 5.4, 'end': 7.6},
    {'start': 8.2, 'end': 10.6}
]
```
And using this information we can then extract only those speech segments from
the audio:
```c++
std::vector<float> filter_non_speech(const std::vector<float>& audio, float sample_rate) {
    auto vad_model = load_silero_vad_model();
    auto timestamps = vad_model.detect_speech(audio, sample_rate);

    std::vector<float> speech_only;
    for (const auto& segment : timestamps) {
        int start_sample = segment.start * sample_rate;
        int end_sample = segment.end * sample_rate;
        speech_only.insert(speech_only.end(),
                          audio.begin() + start_sample,
                          audio.begin() + end_sample);
    }

    return speech_only;
}
```
This would be more effient then passing the entire audio file (like if would
have considered a mask of some sort instead) as whisper.cpp only has to process
the actual speech segments.

We need to convert the model to a format that can be uses with whisper.cpp.
Something along the lines of:
```console
(venv) $ python models/convert-solero-vad-to-ggml.py --output models/silero-vad.bin
Converting 16kHz model
Saving GGML Silero-VAD model to models/silero-vad-v5.1.2_16k-ggml.bin
Writing model weights:
  Writing _model.encoder.0.reparam_conv.weight with shape torch.Size([128, 129, 3])
  Writing _model.encoder.0.reparam_conv.bias with shape torch.Size([128])
  Writing _model.encoder.1.reparam_conv.weight with shape torch.Size([64, 128, 3])
  Writing _model.encoder.1.reparam_conv.bias with shape torch.Size([64])
  Writing _model.encoder.2.reparam_conv.weight with shape torch.Size([64, 64, 3])
  Writing _model.encoder.2.reparam_conv.bias with shape torch.Size([64])
  Writing _model.encoder.3.reparam_conv.weight with shape torch.Size([128, 64, 3])
  Writing _model.encoder.3.reparam_conv.bias with shape torch.Size([128])
  Writing lstm_weight_ih with shape torch.Size([512, 128])
  Writing lstm_weight_hh with shape torch.Size([512, 128])
  Writing lstm_bias_ih with shape torch.Size([512])
  Writing lstm_bias_hh with shape torch.Size([512])
  Writing final_conv_weight with shape torch.Size([1, 128, 1])
  Writing final_conv_bias with shape torch.Size([1])
Done! 16kHz model has been converted to GGML format: models/silero-vad-v5.1.2_16k-ggml.bin
```
And the 8kHz model:
```console
(venv) $ python models/convert-solero-vad-to-ggml.py --output models/silero-vad.bin --sample-rate 8000
Converting 8kHz model
Saving GGML Silero-VAD model to models/silero-vad-v5.1.2_8k-ggml.bin
Writing model weights:
  Writing _model_8k.encoder.0.reparam_conv.weight with shape torch.Size([128, 65, 3])
  Writing _model_8k.encoder.0.reparam_conv.bias with shape torch.Size([128])
  Writing _model_8k.encoder.1.reparam_conv.weight with shape torch.Size([64, 128, 3])
  Writing _model_8k.encoder.1.reparam_conv.bias with shape torch.Size([64])
  Writing _model_8k.encoder.2.reparam_conv.weight with shape torch.Size([64, 64, 3])
  Writing _model_8k.encoder.2.reparam_conv.bias with shape torch.Size([64])
  Writing _model_8k.encoder.3.reparam_conv.weight with shape torch.Size([128, 64, 3])
  Writing _model_8k.encoder.3.reparam_conv.bias with shape torch.Size([128])
  Writing lstm_weight_ih with shape torch.Size([512, 128])
  Writing lstm_weight_hh with shape torch.Size([512, 128])
  Writing lstm_bias_ih with shape torch.Size([512])
  Writing lstm_bias_hh with shape torch.Size([512])
  Writing final_conv_weight with shape torch.Size([1, 128, 1])
  Writing final_conv_bias with shape torch.Size([1])
Done! 8kHz model has been converted to GGML format: models/silero-vad-v5.1.2_8k-ggml.bin
```
So it should then be possible to specify a VAD model when using whisper.cpp
in some way. This would then load the this VAD model and process the audio, get
the speech timestamps for each segment. 

_wip_
