## Voice Activity Detection (VAD)
Also know as as speech activity detection (SAD) or speech detection.

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

One thing that I've done for figure out more information about the operations
in the model is use the cpp example and enable profiling:
```c++
    session_options.EnableProfiling("onnxruntime_profile.json");
```
This produces a file that contains information about the operations in the model.
Now since the examples processes a sample audio file in chunks, I'm using jfk.wav
as the audio sample, the actual file is huge. But there patterns are repeating
so we only need to focus on a subsection. 

The inputs to the model are the following, which I collected from https://netron.app/
using the ONNX model:
```
Name   
input: tensor: float32[?,?]

state: tensor: float32[2,?,128]

sr   : tensor: int64    (sampling rate)
```
This is what the model looks like:

![image](./images/silero_vad.onnx.png)

The first node is checking that the sample rate is 16000. The model also
supports 8000Hz so this is probably what this check is about.

In the actual processing there first node is the STFT (Short-Time Fourier
Transform):
```
Padding operation:
Input:  [1, 576]        // 512 sample plus 64 samples from previous frame
Output: [1, 640]        // 567 + 64 = 640

Reshape/squeeze operation:
Reshapes [1,640] to [1,1,640] (adds dimension for batch)

conv1d operation:
Input shape: [1,1,640] (batch, channels, time)
Filter shape: [258,1,256] (num_filters, in_channels, kernel_size)
Output shape: [1,258,4]
Parameters: 264,192 (258 filters × 1 channel × 256 kernel + biases)

slice operation:
Slicing to [1,129,4] suggests extracting real components from complex numbers
```
This is then followed by a 4 encoder layers:
```
First encoder layer:
conv1d operation:
Input shape: [1,129,4] (batch, channels, time)
Filter shape: [128,129,3] (num_filters, in_channels, kernel_size)
Output shape: [1,128,4]
Parameters: 198,656 (128 filters × 129 channels × 3 kernel + biases)

Second encoder layer:
conv1d operation:
Input shape: [1,128,4] (batch, channels, time)
Filter shape: [64,128,3] (num_filters, in_channels, kernel_size)
Output shape: [1,64,2]
Parameters: 98,560 (64 filters × 128 channels × 3 kernel + biases)

Third encoder layer:
conv1d operation:
Input shape: [1,64,2] (batch, channels, time)
Filter shape: [64,64,3] (num_filters, in_channels, kernel_size)
Output shape: [1,64,1] 
Parameters: 49,408 (64 filters × 64 channels × 3 kernel + biases)

Fourth encoder layer:
conv1d operation:
Input shape: [1,64,1] (batch, channels, time)
Filter shape: [128,64,3] (num_filters, in_channels, kernel_size)
Output shape: [1,128,1]
Parameters: 98,816 (128 filters × 64 channels × 3 kernel + biases)
```

Decoder with LSTM:
```
sequeeze operation:
Removing dimension: [1,128,1] → [1,128]

LSTM operation:
- Input: [1,1,128] (sequence of 1 with 128 features)
- Parameters: 1024 weights
- Outputs: [1,1,1,128], [1,1,128], [1,1,128] (output, hidden, cell states)

ReLU operation:

Conv1d operation:
Input shape: [1,128,1] (batch, channels, time)
Filter shape: [1,128,1] (num_filters, in_channels, kernel_size)
Output shape: [1,1,1]
Parameters: 516 (1 filter × 128 channels × 1 kernel + biases)
```

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
)
```
There are two main components in the model, one named `_model` which takes care
of the 16kHz audio signal and the other one is named `_model_8k` which takes
care of the 8kHz audio signal. 

Both have the same layers but there tensor shapes might be different (more on
this later when we look at them).

### Tensors
The following are the tensor that are in the the model (only focusing on the 16kHz
and skipping the 8kHz model for now):
```console
Tensors to be written:
_model.stft.forward_basis_buffer: torch.Size([258, 1, 256])
_model.encoder.0.reparam_conv.weight: torch.Size([128, 129, 3])
_model.encoder.0.reparam_conv.bias: torch.Size([128])
_model.encoder.1.reparam_conv.weight: torch.Size([64, 128, 3])
_model.encoder.1.reparam_conv.bias: torch.Size([64])
_model.encoder.2.reparam_conv.weight: torch.Size([64, 64, 3])
_model.encoder.2.reparam_conv.bias: torch.Size([64])
_model.encoder.3.reparam_conv.weight: torch.Size([128, 64, 3])
_model.encoder.3.reparam_conv.bias: torch.Size([128])
_model.decoder.rnn.weight_ih: torch.Size([512, 128])
_model.decoder.rnn.weight_hh: torch.Size([512, 128])
_model.decoder.rnn.bias_ih: torch.Size([512])
_model.decoder.rnn.bias_hh: torch.Size([512])
_model.decoder.decoder.2.weight: torch.Size([1, 128, 1])
_model.decoder.decoder.2.bias: torch.Size([1])
```

#### Short-Time Fourier Transform (STFT)
So if we start with an raw audio input signal, this will first be sampled and
quantized, which will give us a vector of floats.

Next we divide this into frames/segments of the samples that usually overlap to
avoid spectral leakage, and the size of a frame is usually a power of two so
that we can use the Fast Fourier Transform. 
If we look closely that the node above we find this:
```
    (stft): RecursiveScriptModule(
      original_name=STFT
      (padding): RecursiveScriptModule(original_name=ReflectionPad1d)
    )
```
I missed this initially but this is an `ReflectionPad1d` which is not a simple
zero padding operation. This will add padding to the left and right of the
samples, and will use values from the respective sides.

If we inspect the models tensors (see below for details) we find that the
model contains a precomputed STFT basis buffer:
```console
_model.stft.forward_basis_buffer: torch.Size([258, 1, 256])
```
The first dimension is the filter length, which in this case is 258 and probably
129 complex number (real and imaginary) values for a total of 256 values. These
are the STFT kernel cooefficients (window function * complex exponential terms).
This prepopulated STFT tensor allows us to not have to recompute the STFT basis
every time we want to process an audio. We can use a convolution using this tensor
to get the frequency spectrogram for the segment and then pass it along to the
encoder blocks.

The second dimension is the number of channels, which for audio is a single
channel assuming mono and not stereo.

The third dimension  256 is the number of frequency output bins that we get from the
STFT. These are the frequencies that are of interest for human speach. For a typical
sampling rate of 16000 Hz, 256 bins gives about 31.25.

Now, for whisper/ggml we need to have the convolution tensor in the format:
```
{258, 1, 256},
```
That is a kernel size of 258, 1 channel, and 256 frequency bins (output).


So the first layer, `(sftf)` above, will take raw audio samples, 512 samples at
16kHz which is about 32ms.
```
duration = = 1 / 16000 * 512 = 0.032
```
And like we mentioned above we overlap the frames/segments to avoid spectral
leakage so we add an additional 64 samples from the previous frame which give
us 512+64=576 samples.

In the case of whisper.cpp I think only 16kHz is supported so I'll focus on that
for now.
```
audio samples -> STFT -> spectral features (129 frequency bins for 16kHz)
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
So lets take a look at the first layer:
```
Writing _model.encoder.0.reparam_conv.weight with shape torch.Size([128, 129, 3])

128 output channels
129 input channels
3 kernel size

In ggml this will become a 3D tensor of shape [3, 129, 128]
So this would looks something like this:

0
   0  [0  2]
      ...
 129  [0  2]

...
127
   0  [0  2]
      ...
 129  [0  2]

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

The tensor in the model are the following for the LSTM layer:
```
_model.decoder.rnn.weight_ih, Shape: torch.Size([512, 128])
_model.decoder.rnn.bias_ih, Shape: torch.Size([512])

_model.decoder.rnn.weight_hh, Shape: torch.Size([512, 128])
_model.decoder.rnn.bias_hh, Shape: torch.Size([512])
```
The `ih` stands for `input to hidden` and is used to compute the input gate.
Now, notice that the shape is 512, 128 which might seem odd at first but this
actually contains all the vectors for the 4 gates stacked into a matrix.
So we can perform on matrix multiplication to get the input gate.

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

### Output layer
```
_model.decoder.decoder.2.weight, Shape: torch.Size([1, 128, 1])
_model.decoder.decoder.2.bias, Shape: torch.Size([1])
```

So, if we start with the raw audio input which consists a samples (floats).
We resample this into either 16kHz or 8kHz, which can be done (at least the 16kHz)
by using `examples/common-whisper.cpp`:
```c++
bool read_audio_data(const std::string & fname,
    std::vector<float>& pcmf32,
    std::vector<std::vector<float>>& pcmf32s,
    bool stereo);
```
One thing to not that this uses `WHISPER_SAMPLE_RATE` which is set to 16000 and
perhaps we should only be focusing on the 16kHz model for now and skip the 8kHz
model?  

So with the output from `read_audio_data` we can then pass this to the VAD.
```c++
std::vector<float> pcmf32;               // mono-channel F32 PCM
std::vector<std::vector<float>> pcmf32s; // stereo-channel F32 PCM
```

### Whisper.cpp integration

Branch:  https://github.com/danbev/whisper.cpp/tree/vad

The initial goal is to get the model conversion working and then be able to
load the model and run the graph computation. This currently works and the
test below will run the model and output some results (which don't seem to
be correct).

With this in place I'll start iterating upon this and going through and making
sure that the weights are loaded correctly, and that dimensions for tensors
are correct. Also clean up the code while doing that as I only wanted to get
something working at this stage.

#### Model conversion
To convert silero-vad model first create a virtual environment and install
the version of silero-vad that you want to convert. Then run the conversion:
```console
 $ (venv) pip install silero-vad
 $ (venv) $ python models/convert-silero-vad-to-ggml.py --output models/silero.bin
 Saving GGML Silero-VAD model to models/silero-v5.1.2-ggml.bin

Tensors to be written:
_model.stft.forward_basis_buffer: torch.Size([258, 1, 256])
_model.encoder.0.reparam_conv.weight: torch.Size([128, 129, 3])
_model.encoder.0.reparam_conv.bias: torch.Size([128])
_model.encoder.1.reparam_conv.weight: torch.Size([64, 128, 3])
_model.encoder.1.reparam_conv.bias: torch.Size([64])
_model.encoder.2.reparam_conv.weight: torch.Size([64, 64, 3])
_model.encoder.2.reparam_conv.bias: torch.Size([64])
_model.encoder.3.reparam_conv.weight: torch.Size([128, 64, 3])
_model.encoder.3.reparam_conv.bias: torch.Size([128])
_model.decoder.rnn.weight_ih: torch.Size([512, 128])
_model.decoder.rnn.weight_hh: torch.Size([512, 128])
_model.decoder.rnn.bias_ih: torch.Size([512])
_model.decoder.rnn.bias_hh: torch.Size([512])
_model.decoder.decoder.2.weight: torch.Size([1, 128, 1])
_model.decoder.decoder.2.bias: torch.Size([1])

Writing model weights:
Processing variable: _model.encoder.0.reparam_conv.weight with shape: (128, 129, 3)
  Keeping original convolution weight shape: (128, 129, 3)
Processing variable: _model.encoder.0.reparam_conv.bias with shape: (128,)
  Converting to float32
Processing variable: _model.encoder.1.reparam_conv.weight with shape: (64, 128, 3)
  Keeping original convolution weight shape: (64, 128, 3)
Processing variable: _model.encoder.1.reparam_conv.bias with shape: (64,)
  Converting to float32
Processing variable: _model.encoder.2.reparam_conv.weight with shape: (64, 64, 3)
  Keeping original convolution weight shape: (64, 64, 3)
Processing variable: _model.encoder.2.reparam_conv.bias with shape: (64,)
  Converting to float32
Processing variable: _model.encoder.3.reparam_conv.weight with shape: (128, 64, 3)
  Keeping original convolution weight shape: (128, 64, 3)
Processing variable: _model.encoder.3.reparam_conv.bias with shape: (128,)
  Converting to float32
Processing variable: _model.decoder.rnn.weight_ih with shape: (512, 128)
Processing variable: _model.decoder.rnn.weight_hh with shape: (512, 128)
Processing variable: _model.decoder.rnn.bias_ih with shape: (512,)
  Converting to float32
Processing variable: _model.decoder.rnn.bias_hh with shape: (512,)
  Converting to float32
Processing variable: _model.decoder.decoder.2.weight with shape: (128,)
  Converting to float32
Processing variable: _model.decoder.decoder.2.bias with shape: ()
  Converting to float32
Processing variable: _model.stft.forward_basis_buffer with shape: (258, 256)
Done! Model has been converted to GGML format: models/silero-v5.1.2-ggml.bin
```

#### Running Test
Run the test:
```console
$ cmake --build build --target test-vad && \
    ctest -R test-vad --test-dir build --output-on-failure -VV
    ...
10: whisper_vad_init_from_file_with_params_no_state: loading VAD model from '../../models/silero-v5.1.2-ggml.bin'
10: whisper_vad_init_from_file_with_params_no_state: threshold    = 0.500000
10: whisper_vad_init_from_file_with_params_no_state: min_speech_duration_ms = 100
10: whisper_vad_init_from_file_with_params_no_state: min_silence_duration_ms = 100
10: whisper_vad_init_from_file_with_params_no_state: window_size_samples = 512
10: whisper_vad_init_from_file_with_params_no_state: sample_rate = 16000
10: whisper_vad_init_from_file_with_params_no_state: use_f16 = 1
10: whisper_vad_init_from_file_with_params_no_state: n_encoder_layers = 4
10: whisper_vad_init_from_file_with_params_no_state: encoder_in_channels[0] = 129
10: whisper_vad_init_from_file_with_params_no_state: encoder_in_channels[1] = 128
10: whisper_vad_init_from_file_with_params_no_state: encoder_in_channels[2] = 64
10: whisper_vad_init_from_file_with_params_no_state: encoder_in_channels[3] = 64
10: whisper_vad_init_from_file_with_params_no_state: encoder_out_channels[0] = 128
10: whisper_vad_init_from_file_with_params_no_state: encoder_out_channels[1] = 64
10: whisper_vad_init_from_file_with_params_no_state: encoder_out_channels[2] = 64
10: whisper_vad_init_from_file_with_params_no_state: encoder_out_channels[3] = 128
10: whisper_vad_init_from_file_with_params_no_state: kernel_sizes[0] = 3
10: whisper_vad_init_from_file_with_params_no_state: kernel_sizes[1] = 3
10: whisper_vad_init_from_file_with_params_no_state: kernel_sizes[2] = 3
10: whisper_vad_init_from_file_with_params_no_state: kernel_sizes[3] = 3
10: whisper_vad_init_from_file_with_params_no_state: lstm_input_size = 128
10: whisper_vad_init_from_file_with_params_no_state: lstm_hidden_size = 128
10: whisper_vad_init_from_file_with_params_no_state: final_conv_in = 128
10: whisper_vad_init_from_file_with_params_no_state: final_conv_out = 1
10: register_backend: registered backend CPU (1 devices)
10: register_device: registered device CPU (12th Gen Intel(R) Core(TM) i7-1260P)
10: whisper_vad_init_from_file_with_params_no_state:          CPU total size =     0.62 MB
10: whisper_vad_init_from_file_with_params_no_state: model size    =    0.62 MB
10: whisper_backend_init_gpu: no GPU found
10: whisper_vad_build_graph: Building VAD graph
10: whisper_vad_build_encoder_layer: building encoder layer
10: whisper_vad_build_lstm_layer: building LSTM layer
10: whisper_vad_init_state: compute buffer (VAD)   =    1.58 MB
10: whisper_vad_detect_speech: detecting speech in 176000 samples
10: whisper_vad_build_graph: Building VAD graph
10: whisper_vad_build_encoder_layer: building encoder layer
10: whisper_vad_build_lstm_layer: building LSTM layer
10: whisper_vad_detect_speech: window_with_context.size() = 256
10: whisper_vad_detect_speech: window_sample_size: 192
10: whisper_vad_detect_speech: context_sample_size: 64
10: whisper_vad_detect_speech: effective_window_size: 256
10: whisper_vad_detect_speech: frame tensor size: 256
10: whisper_vad_detect_speech: finished processing 176000 samples
10: whisper_vad_detect_speech: prob[0]: 0.030489
10: whisper_vad_detect_speech: prob[1]: 0.020316
10: whisper_vad_detect_speech: prob[2]: 0.016475
10: whisper_vad_detect_speech: prob[3]: 0.011185
10: whisper_vad_detect_speech: prob[4]: 0.010197
10: whisper_vad_detect_speech: prob[5]: 0.007823
10: whisper_vad_detect_speech: prob[6]: 0.008767
10: whisper_vad_detect_speech: prob[7]: 0.006645
10: whisper_vad_detect_speech: prob[8]: 0.005273
10: whisper_vad_detect_speech: prob[9]: 0.010585
10: whisper_vad_detect_speech: prob[10]: 0.007144
10: whisper_vad_detect_speech: prob[11]: 0.003635
10: whisper_vad_detect_speech: prob[12]: 0.004149
10: whisper_vad_detect_speech: prob[13]: 0.005139
10: whisper_vad_detect_speech: prob[14]: 0.003650
10: whisper_vad_detect_speech: prob[15]: 0.007306
10: whisper_vad_detect_speech: prob[16]: 0.004238
10: whisper_vad_detect_speech: prob[17]: 0.004754
10: whisper_vad_detect_speech: prob[18]: 0.003174
10: whisper_vad_detect_speech: prob[19]: 0.001825
10: whisper_vad_detect_speech: prob[20]: 0.005317
10: whisper_vad_detect_speech: prob[21]: 0.004083
10: whisper_vad_detect_speech: prob[22]: 0.002842
10: whisper_vad_detect_speech: prob[23]: 0.004745
```
When I compare this output to the silaro-vad example the values are
very different:
```console
0.0120120458
0.0106779542
0.1321811974
0.0654894710
0.0445981026
0.0223348271
0.0260702968
0.0116709163
0.0081158215
0.0067158826
0.8111256361
0.9633629322
0.9310814142
0.7854600549
0.8146636486
0.9672259092
```
But that was somewhat expected as this was just an attempt to get the model
up and running. Next step will be to go through and figure out where I might
have gotten things wrong.

So lets start by checking that the weights that we are loading are correct.

Lets start with `_model.stft.forward_basis_buffer`:
```console
Original model:
[
0.0,
0.00015059065481182188,
0.0006022718735039234,
0.0013547716662287712,
0.0024076367262750864,
0.003760232590138912,
0.005411745049059391,
0.007361178752034903,
0.009607359766960144,
0.012148935347795486]

GGML model:
```

### Troubleshooting
I started by looking at the tensor `_model.stft.forward_basis_buffer` and printed
out the value from the original model and the whisper.cpp model. The values
from the original model are:
```console
  [0]: 0.0
    [1]: 0.00015059065481182188
    [2]: 0.0006022718735039234
    [3]: 0.0013547716662287712
    [4]: 0.0024076367262750864
    [5]: 0.003760232590138912
    [6]: 0.005411745049059391
    [7]: 0.007361178752034903
    [8]: 0.009607359766960144
    [9]: 0.012148935347795486
```
And from whisper.cpp:
```console
10: whisper_vad_init_from_file_with_params_no_state: stft_forward_basis[0]: 0.000000
10: whisper_vad_init_from_file_with_params_no_state: stft_forward_basis[1]: 0.000000
10: whisper_vad_init_from_file_with_params_no_state: stft_forward_basis[2]: 0.000000
10: whisper_vad_init_from_file_with_params_no_state: stft_forward_basis[3]: 0.000000
10: whisper_vad_init_from_file_with_params_no_state: stft_forward_basis[4]: 0.000000
10: whisper_vad_init_from_file_with_params_no_state: stft_forward_basis[5]: 0.000000
10: whisper_vad_init_from_file_with_params_no_state: stft_forward_basis[6]: 0.000000
10: whisper_vad_init_from_file_with_params_no_state: stft_forward_basis[7]: 0.000000
10: whisper_vad_init_from_file_with_params_no_state: stft_forward_basis[8]: 0.000000
10: whisper_vad_init_from_file_with_params_no_state: stft_forward_basis[9]: 0.000000
```
This was because I was not using the correct tensor type. I had make this
configurable to use either `float32` or `float16` but I this will not work with
all operations in GGML. So I've updated the script to for f32 for convolution
operations and after that the values are correct:
```console
10: whisper_vad_init_from_file_with_params_no_state: stft_forward_basis[0]: 0.000000
10: whisper_vad_init_from_file_with_params_no_state: stft_forward_basis[1]: 0.000151
10: whisper_vad_init_from_file_with_params_no_state: stft_forward_basis[2]: 0.000602
10: whisper_vad_init_from_file_with_params_no_state: stft_forward_basis[3]: 0.001355
10: whisper_vad_init_from_file_with_params_no_state: stft_forward_basis[4]: 0.002408
10: whisper_vad_init_from_file_with_params_no_state: stft_forward_basis[5]: 0.003760
10: whisper_vad_init_from_file_with_params_no_state: stft_forward_basis[6]: 0.005412
10: whisper_vad_init_from_file_with_params_no_state: stft_forward_basis[7]: 0.007361
10: whisper_vad_init_from_file_with_params_no_state: stft_forward_basis[8]: 0.009607
10: whisper_vad_init_from_file_with_params_no_state: stft_forward_basis[9]: 0.012149
```
But the probabilities are still not the same but I think we can rule out this
tensor (at least how it is read) as the problem here and look at the others.

Now, lets looks at `_model.encoder.0.reparam_conv.weight`
```console
Processing variable: _model.encoder.0.reparam_conv.weight with shape: (128, 129, 3)
  First 10 values for _model.encoder.0.reparam_conv.weight:
    [0]: 0.023059863597154617
    [1]: 0.03755207359790802
    [2]: -0.001536684576421976
    [3]: 0.05659930780529976
    [4]: 0.09177722781896591
    [5]: 0.06459362804889679
    [6]: -0.040349289774894714
    [7]: 0.040909357368946075
    [8]: -0.07200204581022263
    [9]: -0.12808682024478912
  Keeping original convolution weight shape: (128, 129, 3)
  Original tensor dtype: torch.float32
  This tensor will be forced to F16 for GGML im2col compatibility
```
And in whisper.cpp:
```console
0: whisper_vad_init_from_file_with_params_no_state: model.encoder.0.reparam_conv: [0]: 0.023056 (raw: 0x25e7)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.0.reparam_conv: [1]: 0.037567 (raw: 0x28cf)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.0.reparam_conv: [2]: -0.001536 (raw: 0x964b)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.0.reparam_conv: [3]: 0.056610 (raw: 0x2b3f)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.0.reparam_conv: [4]: 0.091797 (raw: 0x2de0)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.0.reparam_conv: [5]: 0.064575 (raw: 0x2c22)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.0.reparam_conv: [6]: -0.040344 (raw: 0xa92a)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.0.reparam_conv: [7]: 0.040924 (raw: 0x293d)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.0.reparam_conv: [8]: -0.072021 (raw: 0xac9c)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.0.reparam_conv: [9]: -0.128052 (raw: 0xb019)
```
Lets also check the bias:
```console
Processing variable: _model.encoder.0.reparam_conv.bias with shape: (128,)
  First 10 values for _model.encoder.0.reparam_conv.bias:
    [0]: 0.20333558320999146
    [1]: -0.24448169767856598
    [2]: -2.1663601398468018
    [3]: 0.3871806859970093
    [4]: 0.055092066526412964
    [5]: 0.05976399779319763
    [6]: 0.0019018948078155518
    [7]: 0.8512471914291382
    [8]: -0.11439383029937744
    [9]: -0.0516715943813324
  Original tensor dtype: torch.float32

```
And the bias in whisper.cpp:
```console
10: whisper_vad_init_from_file_with_params_no_state: encoder_0_bias: [0]: 0.203336
10: whisper_vad_init_from_file_with_params_no_state: encoder_0_bias: [1]: -0.244482
10: whisper_vad_init_from_file_with_params_no_state: encoder_0_bias: [2]: -2.166360
10: whisper_vad_init_from_file_with_params_no_state: encoder_0_bias: [3]: 0.387181
10: whisper_vad_init_from_file_with_params_no_state: encoder_0_bias: [4]: 0.055092
10: whisper_vad_init_from_file_with_params_no_state: encoder_0_bias: [5]: 0.059764
10: whisper_vad_init_from_file_with_params_no_state: encoder_0_bias: [6]: 0.001902
10: whisper_vad_init_from_file_with_params_no_state: encoder_0_bias: [7]: 0.851247
10: whisper_vad_init_from_file_with_params_no_state: encoder_0_bias: [8]: -0.114394
10: whisper_vad_init_from_file_with_params_no_state: encoder_0_bias: [9]: -0.051672
```
So these tensors also look correct

Next we have `_model.encoder.1.reparam_conv.weight` and bias:
```console
Processing variable: _model.encoder.1.reparam_conv.weight with shape: (64, 128, 3)
  First 10 values for _model.encoder.1.reparam_conv.weight:
    [0]: -0.01762554980814457
    [1]: -0.007143480237573385
    [2]: 0.022292815148830414
    [3]: -0.0391620509326458
    [4]: -0.11304397881031036
    [5]: -0.03947301208972931
    [6]: -0.007277275435626507
    [7]: 0.03176437318325043
    [8]: 0.03668201342225075
    [9]: 0.04778497666120529
  Keeping original convolution weight shape: (64, 128, 3)
  Original tensor dtype: torch.float32
  This tensor will be forced to F16 for GGML im2col compatibility
Processing variable: _model.encoder.1.reparam_conv.bias with shape: (64,)
  First 10 values for _model.encoder.1.reparam_conv.bias:
    [0]: 3.2966432571411133
    [1]: 1.6271023750305176
    [2]: -7.954858779907227
    [3]: 2.7928881645202637
    [4]: 0.10639765858650208
    [5]: 1.5769203901290894
    [6]: 1.2196542024612427
    [7]: 1.5114142894744873
    [8]: 0.9804346561431885
    [9]: -7.94569206237793
  Original tensor dtype: torch.float32

```
And from whisper.cpp:
```console
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.1.reparam_conv: [0]: -0.017624 (raw: 0xa483)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.1.reparam_conv: [1]: -0.007145 (raw: 0x9f51)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.1.reparam_conv: [2]: 0.022293 (raw: 0x25b5)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.1.reparam_conv: [3]: -0.039154 (raw: 0xa903)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.1.reparam_conv: [4]: -0.113037 (raw: 0xaf3c)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.1.reparam_conv: [5]: -0.039459 (raw: 0xa90d)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.1.reparam_conv: [6]: -0.007278 (raw: 0x9f74)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.1.reparam_conv: [7]: 0.031769 (raw: 0x2811)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.1.reparam_conv: [8]: 0.036682 (raw: 0x28b2)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.1.reparam_conv: [9]: 0.047791 (raw: 0x2a1e)
10: whisper_vad_init_from_file_with_params_no_state: encoder_1_bias: [0]: 3.296643
10: whisper_vad_init_from_file_with_params_no_state: encoder_1_bias: [1]: 1.627102
10: whisper_vad_init_from_file_with_params_no_state: encoder_1_bias: [2]: -7.954859
10: whisper_vad_init_from_file_with_params_no_state: encoder_1_bias: [3]: 2.792888
10: whisper_vad_init_from_file_with_params_no_state: encoder_1_bias: [4]: 0.106398
10: whisper_vad_init_from_file_with_params_no_state: encoder_1_bias: [5]: 1.576920
10: whisper_vad_init_from_file_with_params_no_state: encoder_1_bias: [6]: 1.219654
10: whisper_vad_init_from_file_with_params_no_state: encoder_1_bias: [7]: 1.511414
10: whisper_vad_init_from_file_with_params_no_state: encoder_1_bias: [8]: 0.980435
10: whisper_vad_init_from_file_with_params_no_state: encoder_1_bias: [9]: -7.945692
```
The look correct as well.

Then we have `_model.encoder.2.reparam_conv.weight` and bias:
```console
Processing variable: _model.encoder.2.reparam_conv.weight with shape: (64, 64, 3)
  First 10 values for _model.encoder.2.reparam_conv.weight:
    [0]: -0.0072915456257760525
    [1]: -0.10136377811431885
    [2]: -0.19760535657405853
    [3]: -0.0005110583733767271
    [4]: -0.01200706698000431
    [5]: -0.0048386408016085625
    [6]: -0.006183745805174112
    [7]: 0.07137007266283035
    [8]: 0.05046859756112099
    [9]: -0.003160792402923107
  Keeping original convolution weight shape: (64, 64, 3)
  Original tensor dtype: torch.float32
  This tensor will be forced to F16 for GGML im2col compatibility
Processing variable: _model.encoder.2.reparam_conv.bias with shape: (64,)
  First 10 values for _model.encoder.2.reparam_conv.bias:
    [0]: 4.060866832733154
    [1]: 3.816256523132324
    [2]: 0.053663045167922974
    [3]: 0.9439471960067749
    [4]: 2.875575065612793
    [5]: 0.27411338686943054
    [6]: 0.8237091302871704
    [7]: -1.587329626083374
    [8]: -0.9315840005874634
    [9]: 1.7247822284698486
  Original tensor dtype: torch.float32

```
And from whisper.cpp:
```console
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.2.reparam_conv: [0]: -0.007290 (raw: 0x9f77)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.2.reparam_conv: [1]: -0.101379 (raw: 0xae7d)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.2.reparam_conv: [2]: -0.197632 (raw: 0xb253)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.2.reparam_conv: [3]: -0.000511 (raw: 0x9030)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.2.reparam_conv: [4]: -0.012009 (raw: 0xa226)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.2.reparam_conv: [5]: -0.004837 (raw: 0x9cf4)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.2.reparam_conv: [6]: -0.006184 (raw: 0x9e55)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.2.reparam_conv: [7]: 0.071350 (raw: 0x2c91)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.2.reparam_conv: [8]: 0.050476 (raw: 0x2a76)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.2.reparam_conv: [9]: -0.003160 (raw: 0x9a79)
10: whisper_vad_init_from_file_with_params_no_state: encoder_2_bias: [0]: 4.060867
10: whisper_vad_init_from_file_with_params_no_state: encoder_2_bias: [1]: 3.816257
10: whisper_vad_init_from_file_with_params_no_state: encoder_2_bias: [2]: 0.053663
10: whisper_vad_init_from_file_with_params_no_state: encoder_2_bias: [3]: 0.943947
10: whisper_vad_init_from_file_with_params_no_state: encoder_2_bias: [4]: 2.875575
10: whisper_vad_init_from_file_with_params_no_state: encoder_2_bias: [5]: 0.274113
10: whisper_vad_init_from_file_with_params_no_state: encoder_2_bias: [6]: 0.823709
10: whisper_vad_init_from_file_with_params_no_state: encoder_2_bias: [7]: -1.587330
10: whisper_vad_init_from_file_with_params_no_state: encoder_2_bias: [8]: -0.931584
10: whisper_vad_init_from_file_with_params_no_state: encoder_2_bias: [9]: 1.724782
```
And these look correct as well.

The we have `_model.encoder.3.reparam_conv.weight` and bias:
```console
Processing variable: _model.encoder.3.reparam_conv.weight with shape: (128, 64, 3)
  First 10 values for _model.encoder.3.reparam_conv.weight:
    [0]: 0.00868716835975647
    [1]: -0.08090031892061234
    [2]: 0.01122092455625534
    [3]: 0.0034291022457182407
    [4]: 0.023257968947291374
    [5]: 0.008206821046769619
    [6]: 0.006397297605872154
    [7]: 0.18601815402507782
    [8]: 0.007254657801240683
    [9]: -0.0012539586750790477
  Keeping original convolution weight shape: (128, 64, 3)
  Original tensor dtype: torch.float32
  This tensor will be forced to F16 for GGML im2col compatibility
Processing variable: _model.encoder.3.reparam_conv.bias with shape: (128,)
  First 10 values for _model.encoder.3.reparam_conv.bias:
    [0]: 0.9335513114929199
    [1]: 0.11157345771789551
    [2]: 0.09006297588348389
    [3]: 0.6109893918037415
    [4]: -0.6373689770698547
    [5]: 0.00609125941991806
    [6]: 1.0473954677581787
    [7]: -0.6057872176170349
    [8]: 1.885377049446106
    [9]: -3.769871711730957
  Original tensor dtype: torch.float32
```
And from whisper.cpp:
```console
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.2.reparam_conv: [0]: 0.008690 (raw: 0x2073)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.2.reparam_conv: [1]: -0.080872 (raw: 0xad2d)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.2.reparam_conv: [2]: 0.011223 (raw: 0x21bf)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.2.reparam_conv: [3]: 0.003429 (raw: 0x1b06)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.2.reparam_conv: [4]: 0.023254 (raw: 0x25f4)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.2.reparam_conv: [5]: 0.008209 (raw: 0x2034)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.2.reparam_conv: [6]: 0.006397 (raw: 0x1e8d)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.2.reparam_conv: [7]: 0.186035 (raw: 0x31f4)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.2.reparam_conv: [8]: 0.007256 (raw: 0x1f6e)
10: whisper_vad_init_from_file_with_params_no_state: model.encoder.2.reparam_conv: [9]: -0.001254 (raw: 0x9523)
10: whisper_vad_init_from_file_with_params_no_state: encoder_3_bias: [0]: 0.933551
10: whisper_vad_init_from_file_with_params_no_state: encoder_3_bias: [1]: 0.111573
10: whisper_vad_init_from_file_with_params_no_state: encoder_3_bias: [2]: 0.090063
10: whisper_vad_init_from_file_with_params_no_state: encoder_3_bias: [3]: 0.610989
10: whisper_vad_init_from_file_with_params_no_state: encoder_3_bias: [4]: -0.637369
10: whisper_vad_init_from_file_with_params_no_state: encoder_3_bias: [5]: 0.006091
10: whisper_vad_init_from_file_with_params_no_state: encoder_3_bias: [6]: 1.047395
10: whisper_vad_init_from_file_with_params_no_state: encoder_3_bias: [7]: -0.605787
10: whisper_vad_init_from_file_with_params_no_state: encoder_3_bias: [8]: 1.885377
10: whisper_vad_init_from_file_with_params_no_state: encoder_3_bias: [9]: -3.769872
```
And these also look correct. 

Next lets check the LSTM/RNN tensors:
```console
Processing variable: _model.decoder.rnn.weight_ih with shape: (512, 128)
  First 10 values for _model.decoder.rnn.weight_ih:
    [0]: -0.1975371241569519
    [1]: -0.13793830573558807
    [2]: 0.16510847210884094
    [3]: 0.007955566048622131
    [4]: 0.029819002375006676
    [5]: -0.3347293436527252
    [6]: 0.019417593255639076
    [7]: 0.00517271226271987
    [8]: -0.08036171644926071
    [9]: 0.14333027601242065
  Original tensor dtype: torch.float32
Processing variable: _model.decoder.rnn.bias_ih with shape: (512,)
  First 10 values for _model.decoder.rnn.bias_ih:
    [0]: -0.1524425894021988
    [1]: -0.12193526327610016
    [2]: -0.08168794959783554
    [3]: -0.29849109053611755
    [4]: -0.2474878579378128
    [5]: 0.03450224548578262
    [6]: -0.08904067426919937
    [7]: -0.06718937307596207
    [8]: -0.12373599410057068
    [9]: -0.392291396856308
  Original tensor dtype: torch.float32

```
And from whisper.cpp:
```console
10: whisper_vad_init_from_file_with_params_no_state: lstm_weight_ih: [0]: -0.197537
10: whisper_vad_init_from_file_with_params_no_state: lstm_weight_ih: [1]: -0.137938
10: whisper_vad_init_from_file_with_params_no_state: lstm_weight_ih: [2]: 0.165108
10: whisper_vad_init_from_file_with_params_no_state: lstm_weight_ih: [3]: 0.007956
10: whisper_vad_init_from_file_with_params_no_state: lstm_weight_ih: [4]: 0.029819
10: whisper_vad_init_from_file_with_params_no_state: lstm_weight_ih: [5]: -0.334729
10: whisper_vad_init_from_file_with_params_no_state: lstm_weight_ih: [6]: 0.019418
10: whisper_vad_init_from_file_with_params_no_state: lstm_weight_ih: [7]: 0.005173
10: whisper_vad_init_from_file_with_params_no_state: lstm_weight_ih: [8]: -0.080362
10: whisper_vad_init_from_file_with_params_no_state: lstm_weight_ih: [9]: 0.143330

10: whisper_vad_init_from_file_with_params_no_state: lstm_bias_ih: [0]: -0.152443
10: whisper_vad_init_from_file_with_params_no_state: lstm_bias_ih: [1]: -0.121935
10: whisper_vad_init_from_file_with_params_no_state: lstm_bias_ih: [2]: -0.081688
10: whisper_vad_init_from_file_with_params_no_state: lstm_bias_ih: [3]: -0.298491
10: whisper_vad_init_from_file_with_params_no_state: lstm_bias_ih: [4]: -0.247488
10: whisper_vad_init_from_file_with_params_no_state: lstm_bias_ih: [5]: 0.034502
10: whisper_vad_init_from_file_with_params_no_state: lstm_bias_ih: [6]: -0.089041
10: whisper_vad_init_from_file_with_params_no_state: lstm_bias_ih: [7]: -0.067189
10: whisper_vad_init_from_file_with_params_no_state: lstm_bias_ih: [8]: -0.123736
10: whisper_vad_init_from_file_with_params_no_state: lstm_bias_ih: [9]: -0.392291
```
These look correct as well (apart for an inconsistency in the nameing of the
tensor in whisper.cpp, I'll fix that).

Next we have `_model.decoder.rnn.weight_hh`:
```console
Processing variable: _model.decoder.rnn.weight_hh with shape: (512, 128)
  First 10 values for _model.decoder.rnn.weight_hh:
    [0]: -0.3621460497379303
    [1]: 0.14502376317977905
    [2]: -0.29783394932746887
    [3]: 0.034422460943460464
    [4]: 0.17480415105819702
    [5]: -0.1250990778207779
    [6]: -0.24738839268684387
    [7]: -0.06837962567806244
    [8]: 0.32639244198799133
    [9]: -0.18058985471725464
  Original tensor dtype: torch.float32
Processing variable: _model.decoder.rnn.bias_hh with shape: (512,)
  First 10 values for _model.decoder.rnn.bias_hh:
    [0]: -0.023373831063508987
    [1]: -0.13415886461734772
    [2]: -0.04436622932553291
    [3]: -0.4029233157634735
    [4]: -0.23194685578346252
    [5]: -0.01958276331424713
    [6]: -0.03060426004230976
    [7]: -0.03582705929875374
    [8]: -0.17606812715530396
    [9]: -0.2881392538547516
  Original tensor dtype: torch.float32
```
And from whisper.cpp:
```console
10: whisper_vad_init_from_file_with_params_no_state: lstm_weight_hh: [0]: -0.362146
10: whisper_vad_init_from_file_with_params_no_state: lstm_weight_hh: [1]: 0.145024
10: whisper_vad_init_from_file_with_params_no_state: lstm_weight_hh: [2]: -0.297834
10: whisper_vad_init_from_file_with_params_no_state: lstm_weight_hh: [3]: 0.034422
10: whisper_vad_init_from_file_with_params_no_state: lstm_weight_hh: [4]: 0.174804
10: whisper_vad_init_from_file_with_params_no_state: lstm_weight_hh: [5]: -0.125099
10: whisper_vad_init_from_file_with_params_no_state: lstm_weight_hh: [6]: -0.247388
10: whisper_vad_init_from_file_with_params_no_state: lstm_weight_hh: [7]: -0.068380
10: whisper_vad_init_from_file_with_params_no_state: lstm_weight_hh: [8]: 0.326392
10: whisper_vad_init_from_file_with_params_no_state: lstm_weight_hh: [9]: -0.180590

10: whisper_vad_init_from_file_with_params_no_state: lstm_bias_hh: [0]: -0.023374
10: whisper_vad_init_from_file_with_params_no_state: lstm_bias_hh: [1]: -0.134159
10: whisper_vad_init_from_file_with_params_no_state: lstm_bias_hh: [2]: -0.044366
10: whisper_vad_init_from_file_with_params_no_state: lstm_bias_hh: [3]: -0.402923
10: whisper_vad_init_from_file_with_params_no_state: lstm_bias_hh: [4]: -0.231947
10: whisper_vad_init_from_file_with_params_no_state: lstm_bias_hh: [5]: -0.019583
10: whisper_vad_init_from_file_with_params_no_state: lstm_bias_hh: [6]: -0.030604
10: whisper_vad_init_from_file_with_params_no_state: lstm_bias_hh: [7]: -0.035827
10: whisper_vad_init_from_file_with_params_no_state: lstm_bias_hh: [8]: -0.176068
10: whisper_vad_init_from_file_with_params_no_state: lstm_bias_hh: [9]: -0.288139
````
And these look correct as well.

And finally we have `_model.decoder.decoder.2.weight` and bias:
```console
Processing variable: _model.decoder.decoder.2.weight with shape: (128,)
  First 10 values for _model.decoder.decoder.2.weight:
    [0]: 0.10062672197818756
    [1]: 0.17330233752727509
    [2]: -0.251087486743927
    [3]: -1.1117055416107178
    [4]: 0.30843374133110046
    [5]: -0.44464311003685
    [6]: -0.45811617374420166
    [7]: -0.027409639209508896
    [8]: 0.3915608525276184
    [9]: 1.2692075967788696
  Original tensor dtype: torch.float32
Processing variable: _model.decoder.decoder.2.bias with shape: ()
  First 10 values for _model.decoder.decoder.2.bias:
    [0]: -0.19063705205917358
  Original tensor dtype: torch.float32
```
And from whisper.cpp:
```console
10: whisper_vad_init_from_file_with_params_no_state: final_conv_weight: [0]: 0.100627
10: whisper_vad_init_from_file_with_params_no_state: final_conv_weight: [1]: 0.173302
10: whisper_vad_init_from_file_with_params_no_state: final_conv_weight: [2]: -0.251087
10: whisper_vad_init_from_file_with_params_no_state: final_conv_weight: [3]: -1.111706
10: whisper_vad_init_from_file_with_params_no_state: final_conv_weight: [4]: 0.308434
10: whisper_vad_init_from_file_with_params_no_state: final_conv_weight: [5]: -0.444643
10: whisper_vad_init_from_file_with_params_no_state: final_conv_weight: [6]: -0.458116
10: whisper_vad_init_from_file_with_params_no_state: final_conv_weight: [7]: -0.027410
10: whisper_vad_init_from_file_with_params_no_state: final_conv_weight: [8]: 0.391561
10: whisper_vad_init_from_file_with_params_no_state: final_conv_weight: [9]: 1.269208

10: whisper_vad_init_from_file_with_params_no_state: final_conv_bias: [0]: -0.190637
```
So the weight seem to be correct for this as well.

Looking little more carefully at the origal model above I noticed a mistake
I've made.
```
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
```
This shows a the LSTM layer, followed by a dropout, a relu, and conv1d and then
a sigmoid. I did not include a drop out or a ReLU, and also instead of a conv1d I
has a simple linear layer (mul_mat plus add bias). This is a mistake, and I've
now updated with these layers.

With those changes in place I get the following probabilities:
```console
10: whisper_vad_detect_speech: prob[0]: 0.433458
10: whisper_vad_detect_speech: prob[1]: 0.231823
10: whisper_vad_detect_speech: prob[2]: 0.180998
10: whisper_vad_detect_speech: prob[3]: 0.151418
10: whisper_vad_detect_speech: prob[4]: 0.129234
10: whisper_vad_detect_speech: prob[5]: 0.118154
10: whisper_vad_detect_speech: prob[6]: 0.117612
10: whisper_vad_detect_speech: prob[7]: 0.103644
10: whisper_vad_detect_speech: prob[8]: 0.100053
10: whisper_vad_detect_speech: prob[9]: 0.095149
10: whisper_vad_detect_speech: prob[10]: 0.081977
10: whisper_vad_detect_speech: prob[11]: 0.080713
10: whisper_vad_detect_speech: prob[12]: 0.083069
10: whisper_vad_detect_speech: prob[13]: 0.081101
10: whisper_vad_detect_speech: prob[14]: 0.081165
10: whisper_vad_detect_speech: prob[15]: 0.069563
10: whisper_vad_detect_speech: prob[16]: 0.059652
10: whisper_vad_detect_speech: prob[17]: 0.060627
10: whisper_vad_detect_speech: prob[18]: 0.060169
```
Which are better but still not the same as the original model.
```console
0.0120120458
0.0106779542
0.1321811974
0.0654894710
0.0445981026
0.0223348271
0.0260702968
0.0116709163
0.0081158215
0.0067158826
0.8111256361
0.9633629322
0.9310814142
0.7854600549
0.8146636486
```

Now, I noticed another thing that in the original model and the python example
and also the C++ example they use a window size of 512 where I've been using
192, which was basically just to allow the matrix multiplication to work with
the stft matrix. Stepping back a little this is how I understand the process
up until this point:
We have samples, in this specific case around 17000 samples. These are split
into chunks for 512 (which if I recall correctly is about 32ms and close to the
threshold of human perception of sound). We add an extra 64 samples which we
take from the previous window to avoid spectral leakage. Now we want to multiple
these sample by the matrix provided by the silero-vad model which contains
precomputed sine and cosine values. But doing this we get the values in the
frequency domain (as complex numbers but we only care about the real part here).
Now, this matrix has the shape {256, 258}, and the frame tensor is {576} and
these cannot be multiplied with each other.

I took a look at the ONNX runtime and it seems to be implementing this as a
convolution operation and not a matrix multiplication which was what I did.

Here is an excert from the onnxruntime log:
```console
   20 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59304,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/padding/Pad_fence_before","args" : {"op_name" : "Pad"}},
   21 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :6,"ts" :59304,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/padding/Pad_kernel_time","args" : {"thread_scheduling_stats" : "","output_size" : "2560","parameter_size" : "32","activation_size" : "2304","output_t      ype_shape" : [{"float":[1,640]}],"exec_plan_index" : "7","graph_index" : "7","input_type_shape" : [{"float":[1,576]},{"int64":[4]}],"provider" : "CPUExecutionProvider","op_name" : "Pad"}},
   22 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59312,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/padding/Pad_fence_after","args" : {"op_name" : "Pad"}},
   23 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59312,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Unsqueeze_fence_before","args" : {"op_name" : "Unsqueeze"}},
   24 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :2,"ts" :59313,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Unsqueeze_kernel_time","args" : {"thread_scheduling_stats" : "","output_size" : "2560","parameter_size" : "8","activation_size" : "2560","output_type      _shape" : [{"float":[1,1,640]}],"exec_plan_index" : "8","graph_index" : "8","input_type_shape" : [{"float":[1,640]},{"int64":[1]}],"provider" : "CPUExecutionProvider","op_name" : "Unsqueeze"}},
   25 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59317,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Unsqueeze_fence_after","args" : {"op_name" : "Unsqueeze"}},
   26 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59318,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Conv_fence_before","args" : {"op_name" : "Conv"}},
   27 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :54,"ts" :59318,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Conv_kernel_time","args" : {"thread_scheduling_stats" : "","output_size" : "4128","parameter_size" : "264192","activation_size" : "2560","output_typ      e_shape" : [{"float":[1,258,4]}],"exec_plan_index" : "9","graph_index" : "9","input_type_shape" : [{"float":[1,1,640]},{"float":[258,1,256]}],"provider" : "CPUExecutionProvider","op_name" : "Conv"}},
   28 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59375,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Conv_fence_after","args" : {"op_name" : "Conv"}},
   29 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59376,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Slice_3_fence_before","args" : {"op_name" : "Slice"}},
   30 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :6,"ts" :59377,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Slice_3_kernel_time","args" : {"thread_scheduling_stats" : "","output_size" : "2064","parameter_size" : "32","activation_size" : "4128","output_type_      shape" : [{"float":[1,129,4]}],"exec_plan_index" : "14","graph_index" : "14","input_type_shape" : [{"float":[1,258,4]},{"int64":[1]},{"int64":[1]},{"int64":[1]},{"int64":[1]}],"provider" : "CPUExecutionProvider","op_name" : "Slice"}},
   31 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59385,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Slice_3_fence_after","args" : {"op_name" : "Slice"}},
   32 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59386,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Pow_1_fence_before","args" : {"op_name" : "Pow"}},
   33 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :3,"ts" :59386,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Pow_1_kernel_time","args" : {"thread_scheduling_stats" : "","output_size" : "2064","parameter_size" : "4","activation_size" : "2064","output_type_sha      pe" : [{"float":[1,129,4]}],"exec_plan_index" : "18","graph_index" : "18","input_type_shape" : [{"float":[1,129,4]},{"float":[]}],"provider" : "CPUExecutionProvider","op_name" : "Pow"}},
   34 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59391,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Pow_1_fence_after","args" : {"op_name" : "Pow"}},
   35 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59393,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Slice_1_fence_before","args" : {"op_name" : "Slice"}},
   36 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :3,"ts" :59393,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Slice_1_kernel_time","args" : {"thread_scheduling_stats" : "","output_size" : "2064","parameter_size" : "32","activation_size" : "4128","output_type_      shape" : [{"float":[1,129,4]}],"exec_plan_index" : "11","graph_index" : "11","input_type_shape" : [{"float":[1,258,4]},{"int64":[1]},{"int64":[1]},{"int64":[1]},{"int64":[1]}],"provider" : "CPUExecutionProvider","op_name" : "Slice"}},
   37 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59398,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Slice_1_fence_after","args" : {"op_name" : "Slice"}},
   38 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59399,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Pow_fence_before","args" : {"op_name" : "Pow"}},
   39 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :2,"ts" :59399,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Pow_kernel_time","args" : {"thread_scheduling_stats" : "","output_size" : "2064","parameter_size" : "4","activation_size" : "2064","output_type_shape      " : [{"float":[1,129,4]}],"exec_plan_index" : "17","graph_index" : "17","input_type_shape" : [{"float":[1,129,4]},{"float":[]}],"provider" : "CPUExecutionProvider","op_name" : "Pow"}},
   40 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59403,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Pow_fence_after","args" : {"op_name" : "Pow"}},
   41 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59404,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Add_fence_before","args" : {"op_name" : "Add"}},
   42 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :3,"ts" :59404,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Add_kernel_time","args" : {"thread_scheduling_stats" : "","output_size" : "2064","parameter_size" : "0","activation_size" : "4128","output_type_shape      " : [{"float":[1,129,4]}],"exec_plan_index" : "19","graph_index" : "19","input_type_shape" : [{"float":[1,129,4]},{"float":[1,129,4]}],"provider" : "CPUExecutionProvider","op_name" : "Add"}},
   43 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59409,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Add_fence_after","args" : {"op_name" : "Add"}},
   44 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59410,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Sqrt_fence_before","args" : {"op_name" : "Sqrt"}},
   45 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :6,"ts" :59411,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Sqrt_kernel_time","args" : {"thread_scheduling_stats" : "","output_size" : "2064","parameter_size" : "0","activation_size" : "2064","output_type_shap      e" : [{"float":[1,129,4]}],"exec_plan_index" : "20","graph_index" : "20","input_type_shape" : [{"float":[1,129,4]}],"provider" : "CPUExecutionProvider","op_name" : "Sqrt"}},
   46 {"cat" : "Node","pid" :34575,"tid" :34575,"dur" :0,"ts" :59418,"ph" : "X","name" :"If_0_then_branch__Inline_0__/stft/Sqrt_fence_after","args" : {"op_name" : "Sqrt"}},
```
So after consulting Claude.ai about this is looks like ONNX is performing a
convolution operation. So I tried this and the results were the exact same I
think:
```console
10: whisper_vad_detect_speech: prob[0]: 0.433458
10: whisper_vad_detect_speech: prob[1]: 0.231823
10: whisper_vad_detect_speech: prob[2]: 0.180998
10: whisper_vad_detect_speech: prob[3]: 0.151459
10: whisper_vad_detect_speech: prob[4]: 0.129263
10: whisper_vad_detect_speech: prob[5]: 0.115693
10: whisper_vad_detect_speech: prob[6]: 0.106706
10: whisper_vad_detect_speech: prob[7]: 0.101227
10: whisper_vad_detect_speech: prob[8]: 0.097039
10: whisper_vad_detect_speech: prob[9]: 0.093106
10: whisper_vad_detect_speech: prob[10]: 0.088892
10: whisper_vad_detect_speech: prob[11]: 0.084535
10: whisper_vad_detect_speech: prob[12]: 0.079914
10: whisper_vad_detect_speech: prob[13]: 0.075425
```


But the probability are not correct. Lets take a look at the
[C++ example](https://github.com/snakers4/silero-vad/tree/master/examples/cpp)
in the silero-vad repository to see if we can extract some information from the
model via the ONNX runtime.

```console
$ git diff
diff --git a/examples/cpp/silero-vad-onnx.cpp b/examples/cpp/silero-vad-onnx.cpp
index 380d76d..99fb289 100644
--- a/examples/cpp/silero-vad-onnx.cpp
+++ b/examples/cpp/silero-vad-onnx.cpp
@@ -131,7 +131,7 @@ private:
     timestamp_t current_speech;

     // Loads the ONNX model.
-    void init_onnx_model(const std::wstring& model_path) {
+    void init_onnx_model(const std::string& model_path) {
         init_engine_threads(1, 1);
         session = std::make_shared<Ort::Session>(env, model_path.c_str(), session_options);
     }
@@ -303,7 +303,7 @@ public:
 public:
     // Constructor: sets model path, sample rate, window size (ms), and other parameters.
     // The parameters are set to match the Python version.
-    VadIterator(const std::wstring ModelPath,
+    VadIterator(const std::string ModelPath,
         int Sample_rate = 16000, int windows_frame_size = 32,
         float Threshold = 0.5, int min_silence_duration_ms = 100,
         int speech_pad_ms = 30, int min_speech_duration_ms = 250,
@@ -329,7 +329,7 @@ public:

 int main() {
     // Read the WAV file (expects 16000 Hz, mono, PCM).
-    wav::WavReader wav_reader("audio/recorder.wav"); // File located in the "audio" folder.
+    wav::WavReader wav_reader("jfk.wav"); // File located in the "audio" folder.
     int numSamples = wav_reader.num_samples();
     std::vector<float> input_wav(static_cast<size_t>(numSamples));
     for (size_t i = 0; i < static_cast<size_t>(numSamples); i++) {
@@ -337,7 +337,7 @@ int main() {
     }

     // Set the ONNX model path (file located in the "model" folder).
-    std::wstring model_path = L"model/silero_vad.onnx";
+    std::string model_path = "../../src/silero_vad/data/silero_vad.onnx";

     // Initialize the VadIterator.
     VadIterator vad(model_path);
```
And a make file to build and run.
```console
ONNX_PATH=./onnxruntime-linux-x64-1.12.1

build:
	g++ silero-vad-onnx.cpp -I ${ONNX_PATH}/include/ \
		-L ${ONNX_PATH}/lib/ \
		-lonnxruntime \
		-Wl,-rpath,${ONNX_PATH}/lib/ \
		-o test

run: build
	./test
```


So I'm getting the following output at the moment:
```console
10: whisper_vad_detect_speech: prob[0]: 0.017022
10: whisper_vad_detect_speech: prob[1]: 0.018527
10: whisper_vad_detect_speech: prob[2]: 0.011746
10: whisper_vad_detect_speech: prob[3]: 0.008625
10: whisper_vad_detect_speech: prob[4]: 0.004357
10: whisper_vad_detect_speech: prob[5]: 0.003329
10: whisper_vad_detect_speech: prob[6]: 0.002859
10: whisper_vad_detect_speech: prob[7]: 0.005444
10: whisper_vad_detect_speech: prob[8]: 0.007293
10: whisper_vad_detect_speech: prob[9]: 0.004256
```
Compared to the output from the silero-vad pythons example:
```console
0.012012
0.010678
0.132181
0.065489
0.044598
0.022335
0.026070
0.011671
0.008116
0.006716
```
One thing I noticed is that if I comment out the actual setting of the pcm32
data, I get the same result.
```c++
        // Copy the current samples from pcmf32 into the window_with_context,
        // starting after the context buffer copied above.
        //std::copy(&pcmf32[i], &pcmf32[i + vctx->window_size_samples], window_with_context.begin() + vctx->context_samples);
```

So it is like the probabilities I'm seeing are
just noise.

```console
10: whisper_vad_detect_speech: prob[0]: 0.017022
10: whisper_vad_detect_speech: prob[1]: 0.018406
10: whisper_vad_detect_speech: prob[2]: 0.051465
10: whisper_vad_detect_speech: prob[3]: 0.028821
10: whisper_vad_detect_speech: prob[4]: 0.012965
10: whisper_vad_detect_speech: prob[5]: 0.022604
10: whisper_vad_detect_speech: prob[6]: 0.056287
10: whisper_vad_detect_speech: prob[7]: 0.020504
10: whisper_vad_detect_speech: prob[8]: 0.015851
10: whisper_vad_detect_speech: prob[9]: 0.018630
10: whisper_vad_detect_speech: prob[10]: 0.037504
10: whisper_vad_detect_speech: prob[11]: 0.141271
10: whisper_vad_detect_speech: prob[12]: 0.051602
10: whisper_vad_detect_speech: prob[13]: 0.029109
10: whisper_vad_detect_speech: prob[14]: 0.110565
10: whisper_vad_detect_speech: prob[15]: 0.047791
10: whisper_vad_detect_speech: prob[16]: 0.031876
10: whisper_vad_detect_speech: prob[17]: 0.086297
10: whisper_vad_detect_speech: prob[18]: 0.041629
10: whisper_vad_detect_speech: prob[19]: 0.097479
10: whisper_vad_detect_speech: prob[20]: 0.073999
10: whisper_vad_detect_speech: prob[21]: 0.063608
10: whisper_vad_detect_speech: prob[22]: 0.078973
10: whisper_vad_detect_speech: prob[23]: 0.486158
10: whisper_vad_detect_speech: prob[24]: 0.609635
10: whisper_vad_detect_speech: prob[25]: 0.028430
```

Inpsecting the silero-vad model a little closer I found:
```console
5030         node {
5031           input: "If_0_then_branch__Inline_0__/stft/Unsqueeze_output_0"
5032           input: "If_0_then_branch__Inline_0__stft.forward_basis_buffer"
5033           output: "If_0_then_branch__Inline_0__/stft/Conv_output_0"
5034           name: "If_0_then_branch__Inline_0__/stft/Conv"
5035           op_type: "Conv"
5036           attribute {
5037             name: "dilations"
5038             ints: 1
5039             type: INTS
5040           }
5041           attribute {
5042             name: "group"
5043             i: 1
5044             type: INT
5045           }
5046           attribute {
5047             name: "kernel_shape"
5048             ints: 256
5049             type: INTS
5050           }
5051           attribute {
5052             name: "pads"
5053             ints: 0
5054             ints: 0
5055             type: INTS
5056           }
5057           attribute {
5058             name: "strides"
5059             ints: 128
5060             type: INTS
5061           }
5062         }
```
So it looks like I was using incorrect padding and stride values. I've updated
this now and the results are much the same. But I'm still only able to inspect
the final probabilities and I don't know for sure if the sftf values are correct
or not. I need to be able to print the intermediate values from the onnx runtime
and compare them. 

After doing throught model once more and looking that the shapes for the
dimensions and stride for some of the convolution operations that I was missing
the probabilities are now:
```console
10: whisper_vad_detect_speech: prob[0]: 0.001881
10: whisper_vad_detect_speech: prob[1]: 0.001317
10: whisper_vad_detect_speech: prob[2]: 0.008114
10: whisper_vad_detect_speech: prob[3]: 0.002658
10: whisper_vad_detect_speech: prob[4]: 0.000671
10: whisper_vad_detect_speech: prob[5]: 0.000244
10: whisper_vad_detect_speech: prob[6]: 0.000759
10: whisper_vad_detect_speech: prob[7]: 0.005907
10: whisper_vad_detect_speech: prob[8]: 0.005715
10: whisper_vad_detect_speech: prob[9]: 0.005150
```

```console
0.0120120458
0.0106779542
0.1321811974
0.0654894710
0.0445981026
0.0223348271
0.0260702968
0.0116709163
0.0081158215
0.0067158826
0.8111256361
0.9633629322
0.9310814142
0.7854600549
0.8146636486
0.9672259092
```
