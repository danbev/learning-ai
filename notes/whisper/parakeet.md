### Parakeet TDT 0.6B V3 support
This documentent contains on the Parakeet model with the goal being to convert
it into a format that can be used with Whisper.cpp. The goal is to identify major
differences which might effect the work.

### Overview
Parakeet offers three different models:
* Parakeet-TDT
* Parakeet-CTC
* Parakeet-RNNT

The Parakeet model uses a Conformer based encoder named
[Fast Conformer](https://arxiv.org/pdf/2305.05084) and a TDT (Token-and-Duration
Transducer) decoder.

### Initial spike/exploration
I've done an initial spike on this to understand how Parakeet works and what might
be need to add support for it in whisper.cpp. For this I've simply added a new
model for Parakeet along side whisper_model in whisper_context just to be able
to explore this model.

Key differences to whisper:
* Pre-processor: Uses a 2D convolutional pre-processor.
* Encoder : Uses Fast-Conformer using 2D depthwise striding convolutions.
* Positional Encoding: Relative positional encodings.
* Decoder: uses a RNN-T/DTD (prediction network + joint network).

What has been done:
* A conversion script has been created to convert the model to GGML format.
* Implemented parakeet_build_graph_conv for encoder pre processor similar to
  whisper_build_graph_conv.

What needs to be done:
* Full encoder (parakeet_build_graph_encoder)
* Decoder (parakeet_build_graph_decoder)
* Joint network (new to me so I can really tell yet what is needed here)
* Refactoring to support an addition model to integrate this cleanly into whisper.cpp.
* ?

Git branch: https://github.com/danbev/whisper.cpp/tree/parakeet-support

Conclusion:
The differences are significant enough that I think a new model, like parakeet_model,
similar to whisper_model might be needed. Or at least a proper look at how to
support this new model in a good way. It does not look like it will be as easy
as we might have thought before we looked at this model.

### Download the model
```console
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install hf
$ hf download nvidia/parakeet-tdt-0.6b-v3 --local-dir parakeet-tdt-0.6b-v3
```

### Model conversion
This following it the output from the initial conversion script:
```console
(venv) $ ./convert-parakeet.sh 
Extracting /home/danbev/work/ai/whisper-models/nvidia/parkeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo to /tmp/tmp215lcam3
Extraction complete
Model configuration:
  Sample rate: 16000
  Encoder layers: 24
  Encoder d_model: 1024
  Mel features: 128

Loading model weights from /tmp/tmp215lcam3/model_weights.ckpt
Loaded 725 tensors

Loading tokenizer...
Loaded 7918 tokens

GGML hyperparameters:
  n_vocab: 7918
  n_audio_ctx: 1500
  n_audio_state: 1024
  n_audio_head: 8
  n_audio_layer: 24
  n_text_ctx: 448
  n_text_state: 640
  n_text_head: 8
  n_text_layer: 0
  n_mels: 128
  n_fft: 512

Writing to models/whisper-parakeet/ggml-model.bin

Generating mel filterbank: n_mels=128, n_fft=512, sample_rate=16000

Converting model weights...
Processing: preprocessor.featurizer.window [400]
Processing: preprocessor.featurizer.fb [128, 257]
2

Processing: encoder.pre_encode.out.weight [1024, 4096]
Processing: encoder.pre_encode.out.bias [1024]
Processing: encoder.pre_encode.conv.0.weight [256, 3, 3]
Processing: encoder.pre_encode.conv.0.bias [256]
Processing: encoder.pre_encode.conv.2.weight [256, 3, 3]
Processing: encoder.pre_encode.conv.2.bias [256]
Processing: encoder.pre_encode.conv.3.weight [256, 256]
Processing: encoder.pre_encode.conv.3.bias [256]
Processing: encoder.pre_encode.conv.5.weight [256, 3, 3]
Processing: encoder.pre_encode.conv.5.bias [256]
Processing: encoder.pre_encode.conv.6.weight [256, 256]
Processing: encoder.pre_encode.conv.6.bias [256]
14

Processing: encoder.layers.0.norm_feed_forward1.weight [1024]
Processing: encoder.layers.0.norm_feed_forward1.bias [1024]
Processing: encoder.layers.0.feed_forward1.linear1.weight [4096, 1024]
Processing: encoder.layers.0.feed_forward1.linear2.weight [1024, 4096]
Processing: encoder.layers.0.norm_conv.weight [1024]
Processing: encoder.layers.0.norm_conv.bias [1024]
Processing: encoder.layers.0.conv.pointwise_conv1.weight [2048, 1024]
Processing: encoder.layers.0.conv.depthwise_conv.weight [1024, 9]
Processing: encoder.layers.0.conv.batch_norm.weight [1024]
Processing: encoder.layers.0.conv.batch_norm.bias [1024]
Processing: encoder.layers.0.conv.batch_norm.running_mean [1024]
Processing: encoder.layers.0.conv.batch_norm.running_var [1024]
Processing: encoder.layers.0.conv.batch_norm.num_batches_tracked []
Processing: encoder.layers.0.conv.pointwise_conv2.weight [1024, 1024]
Processing: encoder.layers.0.norm_self_att.weight [1024]
Processing: encoder.layers.0.norm_self_att.bias [1024]
Processing: encoder.layers.0.self_attn.pos_bias_u [8, 128]
Processing: encoder.layers.0.self_attn.pos_bias_v [8, 128]
Processing: encoder.layers.0.self_attn.linear_q.weight [1024, 1024]
Processing: encoder.layers.0.self_attn.linear_k.weight [1024, 1024]
Processing: encoder.layers.0.self_attn.linear_v.weight [1024, 1024]
Processing: encoder.layers.0.self_attn.linear_out.weight [1024, 1024]
Processing: encoder.layers.0.self_attn.linear_pos.weight [1024, 1024]
Processing: encoder.layers.0.norm_feed_forward2.weight [1024]
Processing: encoder.layers.0.norm_feed_forward2.bias [1024]
Processing: encoder.layers.0.feed_forward2.linear1.weight [4096, 1024]
Processing: encoder.layers.0.feed_forward2.linear2.weight [1024, 4096]
Processing: encoder.layers.0.norm_out.weight [1024]
Processing: encoder.layers.0.norm_out.bias [1024]
29 per layer (and we have 24 layers in this model)
...
Processing: encoder.layers.23.norm_feed_forward1.weight [1024]
Processing: encoder.layers.23.norm_feed_forward1.bias [1024]
Processing: encoder.layers.23.feed_forward1.linear1.weight [4096, 1024]
Processing: encoder.layers.23.feed_forward1.linear2.weight [1024, 4096]
Processing: encoder.layers.23.norm_conv.weight [1024]
Processing: encoder.layers.23.norm_conv.bias [1024]
Processing: encoder.layers.23.conv.pointwise_conv1.weight [2048, 1024]
Processing: encoder.layers.23.conv.depthwise_conv.weight [1024, 9]
Processing: encoder.layers.23.conv.batch_norm.weight [1024]
Processing: encoder.layers.23.conv.batch_norm.bias [1024]
Processing: encoder.layers.23.conv.batch_norm.running_mean [1024]
Processing: encoder.layers.23.conv.batch_norm.running_var [1024]
Processing: encoder.layers.23.conv.batch_norm.num_batches_tracked []
Processing: encoder.layers.23.conv.pointwise_conv2.weight [1024, 1024]
Processing: encoder.layers.23.norm_self_att.weight [1024]
Processing: encoder.layers.23.norm_self_att.bias [1024]
Processing: encoder.layers.23.self_attn.pos_bias_u [8, 128]
Processing: encoder.layers.23.self_attn.pos_bias_v [8, 128]
Processing: encoder.layers.23.self_attn.linear_q.weight [1024, 1024]
Processing: encoder.layers.23.self_attn.linear_k.weight [1024, 1024]
Processing: encoder.layers.23.self_attn.linear_v.weight [1024, 1024]
Processing: encoder.layers.23.self_attn.linear_out.weight [1024, 1024]
Processing: encoder.layers.23.self_attn.linear_pos.weight [1024, 1024]
Processing: encoder.layers.23.norm_feed_forward2.weight [1024]
Processing: encoder.layers.23.norm_feed_forward2.bias [1024]
Processing: encoder.layers.23.feed_forward2.linear1.weight [4096, 1024]
Processing: encoder.layers.23.feed_forward2.linear2.weight [1024, 4096]
Processing: encoder.layers.23.norm_out.weight [1024]
Processing: encoder.layers.23.norm_out.bias [1024]

Processing: decoder.prediction.embed.weight [8193, 640]
Processing: decoder.prediction.dec_rnn.lstm.weight_ih_l0 [2560, 640]
Processing: decoder.prediction.dec_rnn.lstm.weight_hh_l0 [2560, 640]
Processing: decoder.prediction.dec_rnn.lstm.bias_ih_l0 [2560]
Processing: decoder.prediction.dec_rnn.lstm.bias_hh_l0 [2560]
Processing: decoder.prediction.dec_rnn.lstm.weight_ih_l1 [2560, 640]
Processing: decoder.prediction.dec_rnn.lstm.weight_hh_l1 [2560, 640]
Processing: decoder.prediction.dec_rnn.lstm.bias_ih_l1 [2560]
Processing: decoder.prediction.dec_rnn.lstm.bias_hh_l1 [2560]
9

Processing: joint.pred.weight [640, 640]
Processing: joint.pred.bias [640]
Processing: joint.enc.weight [640, 1024]
Processing: joint.enc.bias [640]
Processing: joint.joint_net.2.weight [8198, 640]
Processing: joint.joint_net.2.bias [8198]
6

Conversion complete!
Output file: models/whisper-parakeet/ggml-model.bin
File size: 1197.11 MB
```
So the total number of tensors is:
```console
Total nu2 + 14 + (29 * 24) + 9 + 6 = 727
```
TODO: add this to the output of the conversion script as it might be useful.

### Debug the original model
```console
(venv) $ pip install -U nemo_toolkit['asr']
(venv) $ python -m pdb test-model.py
(Pdb) b /home/danbev/work/ai/whisper-models/nvidia/parkeet-tdt-0.6b-v3/venv/lib/python3.12/site-packages/nemo/collections/asr/models/rnnt_models.py:698
```

### Model Overview
```console
(Pdb) p self
EncDecRNNTBPEModel(
  (preprocessor): AudioToMelSpectrogramPreprocessor(
    (featurizer): FilterbankFeatures()
  )
  (encoder): ConformerEncoder(
    (pre_encode): ConvSubsampling(
      (out): Linear(in_features=4096, out_features=1024, bias=True)
      (conv): MaskedConvSequential(
        (0): Conv2d(1, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (1): ReLU(inplace=True)
        (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        (3): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (4): ReLU(inplace=True)
        (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        (6): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (7): ReLU(inplace=True)
      )
    )
    (pos_enc): RelPositionalEncoding(
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (layers): ModuleList(
      (0-23): 24 x ConformerLayer(
        (norm_feed_forward1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        (feed_forward1): ConformerFeedForward(
          (linear1): Linear(in_features=1024, out_features=4096, bias=False)
          (activation): Swish()
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=4096, out_features=1024, bias=False)
        )
        (norm_conv): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        (conv): ConformerConvolution(
          (pointwise_conv1): Conv1d(1024, 2048, kernel_size=(1,), stride=(1,), bias=False)
          (depthwise_conv): CausalConv1D(1024, 1024, kernel_size=(9,), stride=(1,), groups=1024, bias=False)
          (batch_norm): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activation): Swish()
          (pointwise_conv2): Conv1d(1024, 1024, kernel_size=(1,), stride=(1,), bias=False)
        )
        (norm_self_att): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        (self_attn): RelPositionMultiHeadAttention(
          (linear_q): Linear(in_features=1024, out_features=1024, bias=False)
          (linear_k): Linear(in_features=1024, out_features=1024, bias=False)
          (linear_v): Linear(in_features=1024, out_features=1024, bias=False)
          (linear_out): Linear(in_features=1024, out_features=1024, bias=False)
          (dropout): Dropout(p=0.1, inplace=False)
          (linear_pos): Linear(in_features=1024, out_features=1024, bias=False)
        )
        (norm_feed_forward2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        (feed_forward2): ConformerFeedForward(
          (linear1): Linear(in_features=1024, out_features=4096, bias=False)
          (activation): Swish()
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=4096, out_features=1024, bias=False)
        )
        (dropout): Dropout(p=0.1, inplace=False)
        (norm_out): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      )
    )
  )
  (decoder): RNNTDecoder(
    (prediction): ModuleDict(
      (embed): Embedding(8193, 640, padding_idx=8192)
      (dec_rnn): LSTMDropout(
        (lstm): LSTM(640, 640, num_layers=2, dropout=0.2)
        (dropout): Dropout(p=0.2, inplace=False)
      )
    )
  )
  (joint): RNNTJoint(
    (pred): Linear(in_features=640, out_features=640, bias=True)
    (enc): Linear(in_features=1024, out_features=640, bias=True)
    (joint_net): Sequential(
      (0): ReLU(inplace=True)
      (1): Dropout(p=0.2, inplace=False)
      (2): Linear(in_features=640, out_features=8198, bias=True)
    )
    (_loss): RNNTLoss(
      (_loss): TDTLossNumba()
    )
    (_wer): WER()
  )
  (loss): RNNTLoss(
    (_loss): TDTLossNumba()
  )
  (spec_augmentation): SpectrogramAugmentation(
    (spec_augment): SpecAugment()
  )
  (wer): WER()
)
```

### preprocessor
In Parakeet they have a layer called `preprocessor` which is equivalent to
`whisper_pcm_to_mel_with_state` which convertes the audio signal to a mel
spectrogram.

```console
(Pdb) p x.shape
torch.Size([1, 128, 1101])

(Pdb) p x.view(-1)[:10]
tensor([-2.1009, -2.1009, -2.1009, -2.1009, -2.1009, -2.1006, -0.9609, -0.7740, -1.4154, -1.3690])

(Pdb) p (x.double()**2).mean().item()
0.9981744415301794
```

log melspectrogram in parakeet.cpp:
```console
DEBUG: Mel spectrogram BEFORE normalization:
-16.635532 -16.635532 -16.635532 -16.635532 -16.635532 -16.635498 -16.595291 -14.870002 -16.375238 -15.539862 
Sum of squares = 235881.203697

DEBUG: Mel spectrogram AFTER normalization:
-2.042357 -2.042357 -2.042357 -2.042357 -2.042357 -2.042323 -2.001918 -0.268119 -1.780780 -0.941283 
Sum of squares = 1099.977891
```


### Encoder
Here is an overview of the encoder layers:
```console
(encoder): ConformerEncoder(
    (pre_encode): ConvSubsampling(
      (out): Linear(in_features=4096, out_features=1024, bias=True)
      (conv): MaskedConvSequential(
        (0): Conv2d(1, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (1): ReLU(inplace=True)
        (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        (3): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (4): ReLU(inplace=True)
        (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        (6): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (7): ReLU(inplace=True)
      )
    )
    (pos_enc): RelPositionalEncoding(
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (layers): ModuleList(
      (0-23): 24 x ConformerLayer(
        (norm_feed_forward1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        (feed_forward1): ConformerFeedForward(
          (linear1): Linear(in_features=1024, out_features=4096, bias=False)
          (activation): Swish()
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=4096, out_features=1024, bias=False)
        )
        (norm_conv): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        (conv): ConformerConvolution(
          (pointwise_conv1): Conv1d(1024, 2048, kernel_size=(1,), stride=(1,), bias=False)
          (depthwise_conv): CausalConv1D(1024, 1024, kernel_size=(9,), stride=(1,), groups=1024, bias=False)
          (batch_norm): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activation): Swish()
          (pointwise_conv2): Conv1d(1024, 1024, kernel_size=(1,), stride=(1,), bias=False)
        )
        (norm_self_att): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        (self_attn): RelPositionMultiHeadAttention(
          (linear_q): Linear(in_features=1024, out_features=1024, bias=False)
          (linear_k): Linear(in_features=1024, out_features=1024, bias=False)
          (linear_v): Linear(in_features=1024, out_features=1024, bias=False)
          (linear_out): Linear(in_features=1024, out_features=1024, bias=False)
          (dropout): Dropout(p=0.1, inplace=False)
          (linear_pos): Linear(in_features=1024, out_features=1024, bias=False)
        )
        (norm_feed_forward2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        (feed_forward2): ConformerFeedForward(
          (linear1): Linear(in_features=1024, out_features=4096, bias=False)
          (activation): Swish()
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=4096, out_features=1024, bias=False)
        )
        (dropout): Dropout(p=0.1, inplace=False)
        (norm_out): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      )
    )
  )
```

If we set a break point before the encoder is called:
```console
(Pdb) b /home/danbev/work/ai/whisper-models/nvidia/parkeet-tdt-0.6b-v3/venv/lib/python3.12/site-packages/nemo/collections/asr/models/rnnt_models.py:698
```
```python
class EncDecRNNTModel(ASRModel, ASRModuleMixin, ExportableEncDecModel, ASRTranscriptionMixin):
    """Base class for encoder decoder RNNT-based models."""
    ...

    @typecheck()
    def forward(
        self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        ...
        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal,
                length=input_signal_length,
            )

        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        return encoded, encoded_len
```
We can inspect the `input_signal` shape:
```console
(Pdb) p input_signal.shape
torch.Size([1, 176000])
```
And this matches `n_samples` in whisper.cpp so we can see that these have the
same number of samples as input:
```console
(gdb) p n_samples
$1 = 176000
```
The first thing that happens in the model is that the preprocessor is called so
lets try to figure out where this class is defined:
```console
(Pdb) p self.preprocessor
AudioToMelSpectrogramPreprocessor(
  (featurizer): FilterbankFeatures()
)
```
This is initialized in the constructor:
```python
        self.preprocessor = EncDecRNNTModel.from_config_dict(self.cfg.preprocessor)
```
And there is a section in the model_config.yaml for the preprocessor:
```yaml
preprocessor:
  _target_: nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor
  sample_rate: 16000
  normalize: per_feature
  window_size: 0.025
  window_stride: 0.01
  window: hann
  features: 128
  n_fft: 512
  log: true
  frame_splicing: 1
  dither: 1.0e-05
  pad_to: 0
  pad_value: 0.0
```
And this file can be found in `venv/lib/python3.13/site-packages/nemo/collections/asr/modules/audio_preprocessing.py`:
```python
class AudioToMelSpectrogramPreprocessor(AudioPreprocessor, Exportable):
    """Featurizer module that converts wavs to mel spectrograms.
```
Now, this is very similar if not the same as what is done in whisper by
`whisper_pcm_to_mel_with_state`, apart from the number of mel bins is 128
instead of 80.

After this preprocessing the shape of the processed signal is:
```console
(Pdb) p processed_signal.shape
torch.Size([1, 128, 1101])
```
Now, we we inspect the mel tensor in whisper.cpp we see that it is not the same
size:
```console
(lldb) p mel->ne
(int64_t[4])  ([0] = 3000, [1] = 80, [2] = 1, [3] = 1)
```
Notice that there is a difference in the number of frames (1101 vs 3000) and the
number of mel bins (128 vs 80).
This is because whisper.cpp will always pad the audio to 30s.
At 16kHz and a stride of 160 samples (10ms) we get 3000 frames:
```console
30s * 16000Hz = 480000 samples
48000/160 = 3000 frames
```
But Parakeet does not seems to pad and just takes the 176000 samples and also
it uses a stride of 160:
```console
176000/160 = 1100 frames
+ 1 frame for the initial frame?
```
_Question_: Should we pad the audio or not for Parakeet?

So the actual mel transformation seems to be the same as whisper.cpp, so lets
move on to focus on the convolution subsampling steps that follow this.

### pre_encode
In whisper this is implemented in `whisper_build_graph_conv` and in Parakeet
in the encoder (I think, but we'll find out):
```python
encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
return encoded, encoded_len
```
```python
        self.encoder = EncDecRNNTModel.from_config_dict(self.cfg.encoder)
```
And this also has a section in the config:
```yaml
encoder:
  _target_: nemo.collections.asr.modules.ConformerEncoder
  feat_in: 128
  feat_out: -1
  n_layers: 24
  d_model: 1024
  use_bias: false
  subsampling: dw_striding
  subsampling_factor: 8
  subsampling_conv_channels: 256
  causal_downsampling: false
  reduction: null
  reduction_position: null
  reduction_factor: 1
  ff_expansion_factor: 4
  self_attention_model: rel_pos
  n_heads: 8
  att_context_size:
  - -1
  - -1
  att_context_style: regular
  xscaling: false
  untie_biases: true
  pos_emb_max_len: 5000
  conv_kernel_size: 9
  conv_norm_type: batch_norm
  conv_context_size: null
  dropout: 0.1
  dropout_pre_encoder: 0.1
  dropout_emb: 0.0
  dropout_att: 0.1
  stochastic_depth_drop_prob: 0.0
  stochastic_depth_mode: linear
  stochastic_depth_start_layer: 1
```
Notice that this model uses relative position encoding, and the subsampling method
is `dw_striding` (depthwise striding). And the subsampling factor is 8.

And this can be found in venv/lib/python3.13/site-packages/nemo/collections/asr/modules/conformer_encoder.py and the class is `ConformerEncoder`:
```python
class ConformerEncoder(NeuralModule, StreamingEncoder, Exportable, AccessMixin):
```

And lets set a break point in the in the forward method:
```console
(Pdb) b /home/danbev/work/ai/whisper-models/nvidia/parkeet-tdt-0.6b-v3/venv/lib/python3.12/site-packages/nemo/collections/asr/modules/conformer_encoder.py:569
```
```python
    def forward(
        self,
        audio_signal,
        length,
        cache_last_channel=None,
        cache_last_time=None,
        cache_last_channel_len=None,
        bypass_pre_encode=False,
        ...
        if bypass_pre_encode:
            self.update_max_seq_length(seq_length=audio_signal.size(1), device=audio_signal.device)
        else:
            self.update_max_seq_length(seq_length=audio_signal.size(2), device=audio_signal.device)
        return self.forward_internal(
            audio_signal,
            length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            bypass_pre_encode=bypass_pre_encode,
        )
    ):
```
The audio_signal is the mel spectrogram:
```console
(Pdb) p audio_signal.shape
torch.Size([1, 128, 1101])
(Pdb) p length
tensor([1100])
(Pdb) p bypass_pre_encode
False
```
```python
    def forward_internal(
        self,
        audio_signal,
        length,
        cache_last_channel=None,
        cache_last_time=None,
        cache_last_channel_len=None,
        bypass_pre_encode=False,
    ):
    ...
        if not bypass_pre_encode:
            audio_signal = torch.transpose(audio_signal, 1, 2)

            if isinstance(self.pre_encode, nn.Linear):
                audio_signal = self.pre_encode(audio_signal)
            else:
                audio_signal, length = self.pre_encode(x=audio_signal, lengths=length)
                length = length.to(torch.int64)
                # `self.streaming_cfg` is set by setup_streaming_cfg(), called in the init
                if self.streaming_cfg.drop_extra_pre_encoded > 0 and cache_last_channel is not None:
                    audio_signal = audio_signal[:, self.streaming_cfg.drop_extra_pre_encoded :, :]
                    length = (length - self.streaming_cfg.drop_extra_pre_encoded).clamp(min=0)
```
So first we have a transpose:
```console
(Pdb) p audio_signal.shape
torch.Size([1, 1101, 128])
(Pdb) p isinstance(self.pre_encode, nn.Linear)
False
```
So this will be calling:
```console
                audio_signal, length = self.pre_encode(x=audio_signal, lengths=length)
```
In the constructor we have:
```python
                self.pre_encode = ConvSubsampling(
                    subsampling=subsampling,
                    subsampling_factor=subsampling_factor,
                    feat_in=feat_in,
                    feat_out=d_model,
                    conv_channels=subsampling_conv_channels,
                    subsampling_conv_chunking_factor=subsampling_conv_chunking_factor,
                    activation=nn.ReLU(True),
                    is_causal=causal_downsampling,
                )
```
We can find this class in venv/lib/python3.13/site-packages/nemo/collections/asr/parts/submodules/subsampling.py.
```python
class ConvSubsampling(torch.nn.Module):
    def __init__(
        self,
        subsampling,
        subsampling_factor,
        feat_in,
        feat_out,
        conv_channels,
        subsampling_conv_chunking_factor=1,
        activation=nn.ReLU(),
        is_causal=False,
    ):
        ...
        if subsampling == 'vggnet':
            ...
        elif subsampling == 'dw_striding':
            self._stride = 2
            self._kernel_size = 3
            self._ceil_mode = False

            if self.is_causal:
                self._left_padding = self._kernel_size - 1
                self._right_padding = self._stride - 1
                self._max_cache_len = subsampling_factor + 1
            else:
                self._left_padding = (self._kernel_size - 1) // 2
                self._right_padding = (self._kernel_size - 1) // 2
                self._max_cache_len = 0

            # Layer 1
            if self.is_causal:
                layers.append(
                    CausalConv2D(
                        in_channels=in_channels,
                        out_channels=conv_channels,
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=None,
                    )
                )
            else:
                layers.append(
                    torch.nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=conv_channels,
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=self._left_padding,
                    )
                )
            in_channels = conv_channels
            layers.append(activation)

            for i in range(self._sampling_num - 1):
                if self.is_causal:
                    layers.append(
                        CausalConv2D(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=None,
                            groups=in_channels,
                        )
                    )
                else:
                    layers.append(
                        torch.nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=self._left_padding,
                            groups=in_channels,
                        )
                    )

                layers.append(
                    torch.nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=conv_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        groups=1,
                    )
                )
                layers.append(activation)
                in_channels = conv_channels
                ...

        self.conv = MaskedConvSequential(*layers)
```
Let set a breakpoint int the forward method:
```console
(Pdb) b /Users/danbev/work/ai/whisper-models/nvidia/parakeet-tdt-0.6b-v3/venv/lib/python3.13/site-packages/nemo/collections/asr/parts/submodules/subsampling.py:386
```

```python
    def forward(self, x, lengths):
        out_lengths = calc_length(
            lengths,
            all_paddings=self._left_padding + self._right_padding,
            kernel_size=self._kernel_size,
            stride=self._stride,
            ceil_mode=self._ceil_mode,
            repeat_num=self._sampling_num,
        )
```
```console
(Pdb) p lengths
tensor([1100])
(Pdb) p self._left_padding
1
(Pdb) p self._right_padding
1
(Pdb) p self._kernel_size
3
(Pdb) p self._stride
2
(Pdb) p self._sampling_num
3
(Pdb) p self._ceil_mode
False
(Pdb) p out_lengths
tensor([138], dtype=torch.int32)
```
```python
            if need_to_split:
                ...
            else:
                x, lengths = self.conv(x, lengths)
```
And this is the main convolution for the subsampling I think:
```console
(Pdb) p self.conv
MaskedConvSequential(
  (0): Conv2d(1, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (1): ReLU(inplace=True)
  (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
  (3): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
  (4): ReLU(inplace=True)
  (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
  (6): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
  (7): ReLU(inpla
```
So lets remind ourselves of the input shape which has [batch, time, features]:
```console
(Pdb) p x.shape
torch.Size([1, 1101, 128])
(Pdb) p lengths
tensor([1100])
```

MaskedConvSequential is in the same file, that is subsampling.py.
```console
(Pdb) b /Users/danbev/work/ai/whisper-models/nvidia/parakeet-tdt-0.6b-v3/venv/lib/python3.13/site-packages/nemo/collections/asr/parts/submodules/subsampling.py:728
```
In the constructor of ConvSubsampling we have:
```python
        self.conv = MaskedConvSequential(*layers)
```
And if we look back there actual layers are the ones that were added in the
constructor. This was not obvious to me at first but this is how they can
enumerate this instance:
```python
class MaskedConvSequential(nn.Sequential):
    def forward(self, x, lengths):
        # Convert input (batch, time, features) to conv format
        x = x.unsqueeze(1)  # (batch, 1, time, features)
        current_lengths = lengths.clone().float()
        mask = self._create_mask(x, current_lengths.long())
```
So first x will be unsqueezed to add a channel dimension:
```console
(Pdb) p x.shape
torch.Size([1, 1, 1101, 128])
(Pdb) p mask.shape
torch.Size([1, 1101, 128])
```
The 2D convolution expects a 4D tensor [batch, channels, height, width]. 

Next all the layers will be applied in sequence and the mask will be updated
after each layer to match the new lengths after convolution.
```python
        # Process through each layer with mask propagation
        for i, layer in enumerate(self):
            # Apply current mask before layer
            x = apply_channel_mask(x, mask)

            # Apply layer
            x = layer(x)

            # Update lengths for stride operations with proper padding
            if hasattr(layer, 'stride') and layer.stride != (1, 1):
                if hasattr(layer, "_left_padding"):
                    padding = (layer._left_padding, layer._right_padding)  # CausalConv2D
                else:
                    padding = layer.padding
                current_lengths = calculate_conv_output_size(
                    current_lengths, layer.kernel_size[0], layer.stride[0], padding
                )
                mask = self._create_mask(x, current_lengths.long())

        # Final masking
        x = apply_channel_mask(x, mask)
        return x, current_lengths.long()

    def _create_mask(self, tensor, lengths):
        """Create mask matching tensor dimensions."""
        batch_size, channels, time, features = tensor.shape
        time_mask = torch.arange(time, device=tensor.device).expand(batch_size, time) < lengths.unsqueeze(1)
        return time_mask.unsqueeze(-1).expand(batch_size, time, features).to(tensor.dtype)
```
So the first layer will be:
```console
(Pdb) p layer
Conv2d(1, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
```
This will downsample 2x because the stride is (2, 2), it skips every other time
frame and frequency bin.So we will go from 1101 -> 551 frames, and from 128 ->
64 mel bins.
And the (1, 256) is the number of input and output channels for the convolution.
So one input channel will be projected to 256 different feature maps.
```console
(Pdb) p x.shape
torch.Size([1, 256, 551, 64])
```
Now the weight used for this operation are from the model:
```console
Processing: encoder.pre_encode.conv.0.weight [256, 3, 3]
Processing: encoder.pre_encode.conv.0.bias [256]
```
```console
(Pdb) p layer.weight.shape
torch.Size([256, 1, 3, 3])
```
So we have 256 output channels, like unique feature detectors. 1 is the number
of input channels it looks at. 3 is the vertical size of the kernel, so this will
look accross 3 mel bins at a go. And the last 3 is the horizontal size which is
over the time frames, so this will look accross 3 time frames at a time.

So in ggml this would become [3, 3, 1, 256]
```console
0
   0
       0 [0 1 2]
       1 [0 1 2]
       2 [0 1 2]
...

255
   0
       0 [0 1 2]
       1 [0 1 2]
       2 [0 1 2]
```
So each of these 3x3 are specific feature detectors that the model has been
trained to detect. All 256 will be applied to the first 3x3 "boxes" of the
input. The kernel then moves 2 positions to the right to process an entire row
and then moves 2 positions down to process the next row.
And this will pass over x which has the shape [1, 1, 1101, 128]:
```
pytorch: [1, 1, 1101, 128]:
ggml   : [128, 1101, 1, 1]:

     0 [0 1 2   ...   127]
     1 [0 1 2   ...   127]
     ...
   1100[0 1 2   ...   127]
```
The shape of x will be the following after this layer, which becasuse we used
a step size of 2 for both the mel bins and the time frame these will be havled:
```console
(Pdb) p x.shape
torch.Size([1, 256, 551, 64])
```
And notice that the is padding being applied as 1101/2=550.5, and we can see
the padding is 1 on both sides so that the output is 551.
```console
(Pdb) p layer.padding
(1, 1)
(Pdb) p layer.kernel_size
(3, 3)
(Pdb) p layer.stride
(2, 2)
(Pdb) p layer.padding
(1, 1)
```

The second layer will a non-linear layer which does not change the shape:
```console
(Pdb) p layer
ReLU(inplace=True)

(Pdb) p x.shape
torch.Size([1, 256, 551, 64])
```
The third layer is another 2D convolution but notice that this time it has a
groups parameter which is set to 256::
```console
(Pdb) p layer
Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
```
So we have 256 output and input channels. And like the previous convolution we
have a 3x3 kernel, the stride is also the same, as it the padding.
```console
(Pdb) p layer.weight.shape
torch.Size([256, 1, 3, 3])
```
So we still have the same of the actual weight tensor for this convolution as
the first, that is 246 3x3 kernels (feature detectors). But the groups parameter
changes the operation. So recall that we said that the kernel was like a feature
detector and that convolution created 256 different results for the feature
that it detected accross the entire mel spectrogram. But now there will be 3x3
kernels applied to each individual detecte feature (map) that was created.

So he output from the first convolution was:
```console
x = [1, 256, 551, 64]
x = [64, 551, 256, 1]
0
    0  [0  .... 63]        <--- unique kernel for this channel (feature map)
    1  [0  .... 63]
    ...
    550[0  .... 63]
...
255
    0  [0  .... 63]        <--- unique kernel for this channel (feature map)
    1  [0  .... 63]
    1  [0  .... 63]
    ...
    550[0  .... 63]
```
So the first convolution was more general and looked at the whole input to detect
features. And in this second convolution the kernels are specialized to refine
the features but only for a specific "category", like how this specific feature
evolves over time.

The resulting shape will be:
```console
(Pdb) p x.shape
torch.Size([1, 256, 276, 32])
```
And again we now have approximately half the time frames and half the mel bins.

Next we have another convolution but this time it is a pointwise (the kernel size
is 1x1) and the stride is also (1, 1)::
```console
(Pdb) p layer
Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))

(Pdb) p layer.weight.shape
torch.Size([256, 256, 1, 1])

(Pdb) p x.shape
torch.Size([1, 256, 276, 32])
```console
x = [1,  256,  276, 32]
x = [32, 276,  256,  1]

We could also flatten this to:
x = [(32 * 276),  256]
x = [8832, 256]

    0   [0             ...                          8831]
    1   [0             ...                          8831]
    ...
    255 [0             ...                          8831]
```
And we multiply these columns by the 256x256 matrix. So we are using the same
kernel here for all features. The kernel has learned things like "output feature
42 should be 70% of input feature 7, minus 30% of input feature 183..."

Then we have another non-linear layer:
```console
(Pdb) p layer
ReLU(inplace=True)

(Pdb) p x.shape
torch.Size([1, 256, 276, 32])
```
Following that we have another 2d convolution:
```console
(Pdb) p layer
Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)

(Pdb) p x.shape
torch.Size([1, 256, 138, 16])
```
And following that we will have another pointwise convolution:
```console
(Pdb) p layer
Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))

(Pdb) p x.shape
torch.Size([1, 256, 138, 16])
```
We can think of these various convolutions as a way to progressively reduce the
dimension and at the same time go from more detail to more abstract features.
Each block sees a coarser time resolution but the features it's detecting are
more abstract.

```console
(Pdb) p layer
ReLU(inplace=True)
```
And the final output shape after the convolutional subsampling is:
```console
(Pdb) p x.shape
torch.Size([1, 256, 138, 16])
(Pdb) p current_lengths
tensor([138.])
```

So this will return us to conformer_encoder.py and its forward_internal method:
```python
           audio_signal, length = self.pre_encode(x=audio_signal, lengths=length)
           ...
           max_audio_length = audio_signal.size(1)

           padding_length = length
           cache_last_channel_next = None
           cache_len = 0
           offset = None

        audio_signal, pos_emb = self.pos_enc(x=audio_signal, cache_len=cache_len)
```

```console
(Pdb) b /home/danbev/work/ai/whisper-models/nvidia/parkeet-tdt-0.6b-v3/venv/lib/python3.12/site-packages/nemo/collections/asr/modules/conformer_encoder.py:656
(Pdb) c
> /home/danbev/work/ai/whisper-models/nvidia/parkeet-tdt-0.6b-v3/venv/lib/python3.12/site-packages/nemo/collections/asr/modules/conformer_encoder.py(656)forward_internal()
-> audio_signal, pos_emb = self.pos_enc(x=audio_signal, cache_len=cache_len)
```

And the shape of the audio_signal is now:
```console
(Pdb) p audio_signal.shape
torch.Size([1, 138, 1024])
(Pdb) p audio_signal.size(1)
138
```
So this brings us to the position encoding:
```console
(Pdb) p self.pos_enc
RelPositionalEncoding(
  (dropout): Dropout(p=0.1, inplace=False)
)

```
Looking back at the constructor we find the following:
```console
        # Positional encodings
        self.pos_emb_max_len = pos_emb_max_len
        if self_attention_model == "rel_pos":
            self.pos_enc = RelPositionalEncoding(
                d_model=d_model,
                dropout_rate=dropout_pre_encoder,
                max_len=pos_emb_max_len,
                xscale=self.xscale,
                dropout_rate_emb=dropout_emb,
            )
```
We can find this class in venv/lib/python3.12/site-packages/nemo/collections/asr/parts/submodules/multi_head_attention.py
```python
class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding for TransformerXL's layers
    See : Appendix B in https://arxiv.org/abs/1901.02860
    ...

    def forward(self, x, cache_len=0):
        """Compute positional encoding.
        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, feature_size)
            cache_len (int): the size of the cache which is used to shift positions
        Returns:
            x (torch.Tensor): Its shape is (batch, time, feature_size)
            pos_emb (torch.Tensor): Its shape is (1, time, feature_size)
        """

        if self.xscale:
            x = x * self.xscale

        # center_pos would be the index of position 0
        # negative positions would be used for right and positive for left tokens
        # for input of length L, 2*L-1 positions are needed, positions from (L-1) to -(L-1)
        input_len = x.size(1) + cache_len
        center_pos = self.pe.size(1) // 2 + 1
        start_pos = center_pos - input_len
        end_pos = center_pos + input_len - 1
        pos_emb = self.pe[:, start_pos:end_pos]
        if self.dropout_emb:
            pos_emb = self.dropout_emb(pos_emb)
        return self.dropout(x), pos_emb
```
We can see that there is a pe tensor in this class:
```console
(Pdb) p self.pe.shape
torch.Size([1, 9999, 1024])
```
But it does not exist in the model. This is because it is generated. If we look
in the super class we find:

So RelPositionalEncoding extends PositionalEncoding. The
```python
class PositionalEncoding(torch.nn.Module):
    """Fixed sinusoidal positional encoding.
    Args:
        d_model (int): embedding dim
        dropout_rate (float): dropout rate
        max_len (int): maximum input length
        xscale (bool): whether to scale the input by sqrt(d_model)
        dropout_rate_emb (float): dropout rate for the positional embeddings
    """
    def create_pe(self, positions, dtype):
        pos_length = positions.size(0)
        pe = torch.zeros(pos_length, self.d_model, device=positions.device)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=positions.device)
            * -(math.log(INF_VAL) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        pe = pe.unsqueeze(0).to(dtype)
        if hasattr(self, 'pe'):
            self.pe = pe
        else:
            self.register_buffer('pe', pe, persistent=False)

    def extend_pe(self, length, device, dtype):
        """Reset and extend the positional encodings if needed."""
        if hasattr(self, 'pe') and self.pe.size(1) >= length:
            return
        positions = torch.arange(0, length, dtype=torch.float32, device=device).unsqueeze(1)
        self.create_pe(positions=positions, dtype=dtype)
```
The `extend_pe` function is called from the `set_max_audio_length` function in
conformer_encoder.py:
```python
    def set_max_audio_length(self, max_audio_length):
        self.max_audio_length = max_audio_length
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        self.pos_enc.extend_pe(max_audio_length, device, dtype)
```
```console
(Pdb) p self.max_audio_length
5000

(Pdb) b venv/lib/python3.12/site-packages/nemo/collections/asr/parts/submodules/multi_head_attention.py:1016
Breakpoint 5 at /home/danbev/work/ai/whisper-models/nvidia/parkeet-tdt-0.6b-v3/venv/lib/python3.12/site-packages/nemo/collections/asr/parts/submodules/multi_head_attention.py:1016

(Pdb) p pe
tensor([[[-0.6639, -0.7478,  0.4186,  ...,  0.8687,  0.4873,  0.8732],
         [ 0.2705, -0.9627,  0.9878,  ...,  0.8688,  0.4872,  0.8733],
         [ 0.9563, -0.2925,  0.6782,  ...,  0.8688,  0.4871,  0.8733],
         ...,
         [-0.9563, -0.2925, -0.6782,  ...,  0.8688, -0.4871,  0.8733],
         [-0.2705, -0.9627, -0.9878,  ...,  0.8688, -0.4872,  0.8733],
         [ 0.6639, -0.7478, -0.4186,  ...,  0.8687, -0.4873,  0.8732]]])
```
The above generates the position encoding matrix, which is then stored in the model:
```python
            self.register_buffer('pe', pe, persistent=False)
```
Which I think is similar to a parameter tensor but does not take place in traning
of the model, it does not get updated during training. Also buffers are moved
automatically when the model is moved to a device. And notice that it is not
stored to the model when torch.save is called.

So in parakeet.cpp we will need to generate this matrix. Comparing the values
it looks like the match quite well:
```console
(Pdb) p pe[0, 0, :10]
tensor([-0.6639,  -0.7478,  0.4186,   -0.9082,  0.0015,  -1.0000,  -0.9134,  0.4070,  0.6954,  -0.7186])

enc_pos_enc:  type: f32, shape: [1024, 9999, 1, 1]. First 10 values
        -0.663950 -0.747777 0.418575  -0.908182 0.001462 -0.999999 -0.913418 0.407022 0.695440 -0.718584 
```

So that was the relative positional encoding, this the breakpoint to set:
```console
(Pdb) b /home/danbev/work/ai/whisper-models/nvidia/parkeet-tdt-0.6b-v3/venv/lib/python3.12/site-packages/nemo/collections/asr/modules/conformer_encoder.py:656
```
We have the self-attention mask and padding:
```python
        audio_signal, pos_emb = self.pos_enc(x=audio_signal, cache_len=cache_len)

        # Create the self-attention and padding masks
        pad_mask, att_mask = self._create_masks(
            att_context_size=cur_att_context_size,
            padding_length=padding_length,
            max_audio_length=max_audio_length,
            offset=offset,
            device=audio_signal.device,
        )
```
Notice that there are two masks here 'pad_mask' and 'att_mask'. The pad_mask
is for the end of the buffer to determine real audio from empty padding (I think)
The att_mask is for the self attention specifying which time frames are allowed
to see (like not looking into the future or only look at the last 2 seconds).

```python
    def _create_masks(self, att_context_size, padding_length, max_audio_length, offset, device):
        if self.self_attention_model != "rel_pos_local_attn":
            att_mask = torch.ones(1, max_audio_length, max_audio_length, dtype=torch.bool, device=device)

            ...

        # pad_mask is the masking to be used to ignore paddings
        pad_mask = torch.arange(0, max_audio_length, device=device).expand(
            padding_length.size(0), -1
        ) < padding_length.unsqueeze(-1)
```
```console
(Pdb) p self.self_attention_model
'rel_pos'
(Pdb) p self.att_context_style
'regular'
```
```
(Pdb) p att_mask.shape
torch.Size([1, 138, 138])
```
```python
        # pad_mask is the masking to be used to ignore paddings
        pad_mask = torch.arange(0, max_audio_length, device=device).expand(
            padding_length.size(0), -1
        ) < padding_length.unsqueeze(-1)
```
So the first is just creating a tensor using a range from 0 to 138:
```console
(Pdb) p torch.arange(0, max_audio_length, device=device)
tensor([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
         14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
         28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
         42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
         56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
         70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
         84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,
         98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
        112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
        126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137])
```
This is then expended, that is we add a dimension to it:
```console
(Pdb) p torch.arange(0, max_audio_length, device=device).expand(1, -1)
tensor([[  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
          14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
          28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
          42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
          56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
          70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
          84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,
          98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
         112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
         126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137]])

(Pdb) p torch.arange(0, max_audio_length, device=device).expand(1, -1).shape
torch.Size([1, 138])
```
The last step which is `< padding_length.unsqueeze(-1)`:
```console
(Pdb) p padding_length.unsqueeze(-1)
tensor([[138]])
```
This is checking every element if it is smaller than the actual lenght:
```console
(Pdb) p torch.arange(0, max_audio_length, device=device).expand(1, -1) < padding_length.unsqueeze(-1)
tensor([[True, True, True, True, True, True, True, True, True, True, True, True,
         True, True, True, True, True, True, True, True, True, True, True, True,
         True, True, True, True, True, True, True, True, True, True, True, True,
         True, True, True, True, True, True, True, True, True, True, True, True,
         True, True, True, True, True, True, True, True, True, True, True, True,
         True, True, True, True, True, True, True, True, True, True, True, True,
         True, True, True, True, True, True, True, True, True, True, True, True,
         True, True, True, True, True, True, True, True, True, True, True, True,
         True, True, True, True, True, True, True, True, True, True, True, True,
         True, True, True, True, True, True, True, True, True, True, True, True,
         True, True, True, True, True, True, True, True, True, True, True, True,
         True, True, True, True, True, True]])
```
```python
        if att_mask is not None:
            # pad_mask_for_att_mask is the mask which helps to ignore paddings
            pad_mask_for_att_mask = pad_mask.unsqueeze(1).repeat([1, max_audio_length, 1])
            pad_mask_for_att_mask = torch.logical_and(pad_mask_for_att_mask, pad_mask_for_att_mask.transpose(1, 2))
            # att_mask is the masking to be used by the MHA layers to ignore the tokens not supposed to be visible
            att_mask = att_mask[:, :max_audio_length, :max_audio_length]
            # paddings should also get ignored, so pad_mask_for_att_mask is used to ignore their corresponding scores
            att_mask = torch.logical_and(pad_mask_for_att_mask, att_mask.to(pad_mask_for_att_mask.device))
            att_mask = ~att_mask
```
```console
(Pdb) p pad_mask.unsqueeze(0).repeat([1, max_audio_length, 1])
tensor([[[True, True, True,  ..., True, True, True],
         [True, True, True,  ..., True, True, True],
         [True, True, True,  ..., True, True, True],
         ...,
         [True, True, True,  ..., True, True, True],
         [True, True, True,  ..., True, True, True],
         [True, True, True,  ..., True, True, True]]])
(Pdb) p pad_mask.unsqueeze(0).repeat([1, max_audio_length, 1]).shape
torch.Size([1, 138, 138])
```
Following that we have the layers of the model:
```python
        for lth, (drop_prob, layer) in enumerate(zip(self.layer_drop_probs, self.layers)):
            original_signal = audio_signal
```
The drop_probs are just for training and we can ignore them for inference.
```console
(Pdb) p lth
0
(Pdb) p layer
ConformerLayer(
  (norm_feed_forward1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
  (feed_forward1): ConformerFeedForward(
    (linear1): Linear(in_features=1024, out_features=4096, bias=False)
    (activation): Swish()
    (dropout): Dropout(p=0.1, inplace=False)
    (linear2): Linear(in_features=4096, out_features=1024, bias=False)
  )
  (norm_conv): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
  (conv): ConformerConvolution(
    (pointwise_conv1): Conv1d(1024, 2048, kernel_size=(1,), stride=(1,), bias=False)
    (depthwise_conv): CausalConv1D(1024, 1024, kernel_size=(9,), stride=(1,), groups=1024, bias=False)
    (batch_norm): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (activation): Swish()
    (pointwise_conv2): Conv1d(1024, 1024, kernel_size=(1,), stride=(1,), bias=False)
  )
  (norm_self_att): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
  (self_attn): RelPositionMultiHeadAttention(
    (linear_q): Linear(in_features=1024, out_features=1024, bias=False)
    (linear_k): Linear(in_features=1024, out_features=1024, bias=False)
    (linear_v): Linear(in_features=1024, out_features=1024, bias=False)
    (linear_out): Linear(in_features=1024, out_features=1024, bias=False)
    (dropout): Dropout(p=0.1, inplace=False)
    (linear_pos): Linear(in_features=1024, out_features=1024, bias=False)
  )
  (norm_feed_forward2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
  (feed_forward2): ConformerFeedForward(
    (linear1): Linear(in_features=1024, out_features=4096, bias=False)
    (activation): Swish()
    (dropout): Dropout(p=0.1, inplace=False)
    (linear2): Linear(in_features=4096, out_features=1024, bias=False)
  )
  (dropout): Dropout(p=0.1, inplace=False)
  (norm_out): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
)
```
A layer is called using:
```python
            audio_signal = layer(
                x=audio_signal,
                att_mask=att_mask,
                pos_emb=pos_emb,
                pad_mask=pad_mask,
                cache_last_channel=cache_last_channel_cur,
                cache_last_time=cache_last_time_cur,
            )
```
Notice that the masks are passed in.
```console
(Pdb) b venv/lib/python3.12/site-packages/nemo/collections/asr/parts/submodules/conformer_modules.py:174
Breakpoint 2 at /home/danbev/work/ai/whisper-models/nvidia/parkeet-tdt-0.6b-v3/venv/lib/python3.12/site-packages/nemo/collections/asr/parts/submodules/conformer_modules.py:174
```
And this is what the layers forward method looks like:
```python
    def forward(self, x, att_mask=None, pos_emb=None, pad_mask=None, cache_last_channel=None, cache_last_time=None):
        """
        Args:
            x (torch.Tensor): input signals (B, T, d_model)
            att_mask (torch.Tensor): attention masks(B, T, T)
            pos_emb (torch.Tensor): (L, 1, d_model)
            pad_mask (torch.tensor): padding mask
            cache_last_channel (torch.tensor) : cache for MHA layers (B, T_cache, d_model)
            cache_last_time (torch.tensor) : cache for convolutional layers (B, d_model, T_cache)
        Returns:
            x (torch.Tensor): (B, T, d_model)
            cache_last_channel (torch.tensor) : next cache for MHA layers (B, T_cache, d_model)
            cache_last_time (torch.tensor) : next cache for convolutional layers (B, d_model, T_cache)
        """
        residual = x
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        residual = residual + self.dropout(x) * self.fc_factor

        x = self.norm_self_att(residual)
        if self.self_attention_model == 'rel_pos':
            x = self.self_attn(query=x, key=x, value=x, mask=att_mask, pos_emb=pos_emb, cache=cache_last_channel)
        elif self.self_attention_model == 'rel_pos_local_attn':
            x = self.self_attn(query=x, key=x, value=x, pad_mask=pad_mask, pos_emb=pos_emb, cache=cache_last_channel)
        elif self.self_attention_model == 'abs_pos':
            x = self.self_attn(query=x, key=x, value=x, mask=att_mask, cache=cache_last_channel)
        else:
            x = None

        if x is not None and cache_last_channel is not None:
            (x, cache_last_channel) = x

        residual = residual + self.dropout(x)

        if self.is_adapter_available():
            # Call the MHA adapters
            pack_input = {
                'x': residual,
                'loc': 'mha',
                'att_mask': att_mask,
                'pos_emb': pos_emb,
            }
            pack_input = self.forward_enabled_adapters(pack_input)
            residual = pack_input['x']

        x = self.norm_conv(residual)
        x = self.conv(x, pad_mask=pad_mask, cache=cache_last_time)
        if cache_last_time is not None:
            (x, cache_last_time) = x
        residual = residual + self.dropout(x)

        x = self.norm_feed_forward2(residual)
        x = self.feed_forward2(x)
        residual = residual + self.dropout(x) * self.fc_factor

        x = self.norm_out(residual)

        if self.is_adapter_available():
            # Call the adapters
            pack_input = {
                'x': x,
                'loc': 'post',
            }
            pack_input = self.forward_enabled_adapters(pack_input)
            x = pack_input['x']

        if self.is_access_enabled(getattr(self, "model_guid", None)) and self.access_cfg.get(
            'save_encoder_tensors', False
        ):
            self.register_accessible_tensor(name='encoder', tensor=x)
        if cache_last_channel is None:
            return x
        else:
            return x, cache_last_channel, cache_last_time
```

```python
        pad_mask = ~pad_mask
        return pad_mask, att_mask
```
Continuing with the layers, I've implemented this block:
```console
    (layers): ModuleList(
      (0-23): 24 x ConformerLayer(
        (norm_feed_forward1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        (feed_forward1): ConformerFeedForward(
          (linear1): Linear(in_features=1024, out_features=4096, bias=False)
          (activation): Swish()
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=4096, out_features=1024, bias=False)
        )
        // ---- this is what is to be implemented now ---
        (conv): ConformerConvolution(
          (pointwise_conv1): Conv1d(1024, 2048, kernel_size=(1,), stride=(1,), bias=False)
          (depthwise_conv): CausalConv1D(1024, 1024, kernel_size=(9,), stride=(1,), groups=1024, bias=False)
          (batch_norm): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activation): Swish()
          (pointwise_conv2): Conv1d(1024, 1024, kernel_size=(1,), stride=(1,), bias=False)
        )
```
So we can see that this is a convolution, it start with pointwise convolution
which is something we did in the subsampling encoder graph.
```console
(Pdb) b venv/lib/python3.12/site-packages/nemo/collections/asr/parts/submodules/conformer_modules.py:174
```
```python
        x = self.norm_conv(residual)
        x = self.conv(x, pad_mask=pad_mask, cache=cache_last_time)
```
And the convolution layer looks like this (its in the same file): 
```python
class ConformerConvolution(nn.Module):
    ...

    def forward(self, x, pad_mask=None, cache=None):
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)

        # Compute the activation function or use GLU for original Conformer
        if self.pointwise_activation == 'glu_':
            x = nn.functional.glu(x, dim=1)
        else:
            x = self.pointwise_activation(x)

        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)

        x = self.depthwise_conv(x, cache=cache)
        if cache is not None:
            x, cache = x

        if self.norm_type == "layer_norm":
            x = x.transpose(1, 2)
            x = self.batch_norm(x)
            x = x.transpose(1, 2)
        else:
            x = self.batch_norm(x)

        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)
        if cache is None:
            return x
        else:
            return x, cache
```
```console
(Pdb) p x.shape
torch.Size([1, 138, 1024])

(Pdb) p self.pointwise_conv1
Conv1d(1024, 2048, kernel_size=(1,), stride=(1,), bias=False)

(Pdb) p self.pointwise_activation
'glu_'
```
After transpose:
```console
(Pdb) p x.shape
torch.Size([1, 1024, 138])
```
After pointwise_conv1:
```console
(Pdb) p x.shape
torch.Size([1, 2048, 138])
```
In this case batch normalization is used:
```console
(Pdb) p self.norm_type
'batch_norm'
```

```console
(Pdb) p self.depthwise_conv
CausalConv1D(1024, 1024, kernel_size=(9,), stride=(1,), groups=1024, bias=False)
```

_wip_


The shape of the subsampling compution is
```c++
(gdb) p cur->ne
$4 = {1024, 188, 1, 1}

   0 [0   ...           1023]
     ...
 187 [0   ...           1023]
```
So we have an feature/hidden state of 1024, and we have 188 time or sequence
length.
This what the layers of the model are going to operate on, they will project
them to higher dimensions (ffn1/ffn2) and this is also what attention is run on.
The attention is multihead attention and there are 8 heads:
```console
1985	            const int d_head = n_state / n_head;
(gdb) p n_head
$9 = 8
```
And this matches the model yml:
```yml
  n_heads: 8
```
So we divide the 1024 features into groups of 128 each (1024/8=128).

We know that n_time is 188:
```c++
            const int p_len = 2 * n_time - 1;
            const int p_offset = (9999 / 2) - (n_time - 1);
            struct ggml_tensor * pos_raw = ggml_view_2d(ctx0, model.pe,
                n_state, p_len,
                model.pe->nb[1],
                p_offset * model.pe->nb[1]);
```
```console
(gdb) p model.pe->ne
(gdb) p p_offset
$22 = 4812
(gdb) p p_len
$20 = 375

(gdb) p model.pe->ne
$17 = {1024, 9999, 1, 1}
   0 [0   ...            1023]
     ...
     ...
     ...
     ...
     ...
9999 [0   ...            1023]

(gdb) p pos_raw->ne
$23 = {1024, 375, 1, 1}
   0 [0   ...            1023]
     ...
     ...
 374 [0   ...            1023]
```
And recall that pe is the computed relative distances scores. We are creating
a few into that matrix for the current 188 frame of audio (from -187 to +187)
But these are just "raw" sin/cos values:
```c++
        const int d_model = model.hparams.n_audio_state;
        const int max_len = 9999;

        // Positions range from +4999 to -4999 (centered at 0)
        std::vector<float> pe_buf(d_model * max_len);
        for (int idx = 0; idx < max_len; idx++) {
            int position = (max_len / 2) - idx;
            for (int i = 0; i < d_model; i++) {
                // div_term for this dimension: 10000^(-2k/d_model) where k = i/2
                float div_term = expf(-(i / 2) * 2.0f * logf(10000.0f) / d_model);
                float angle = position * div_term;

                if (i % 2 == 0) {
                    pe_buf[idx * d_model + i] = sinf(angle);
                } else {
                    pe_buf[idx * d_model + i] = cosf(angle);
                }
            }
        }
```
Then model does not understand them directly, but it has been trained to do so,
and we use the att_pos_w to get the model understand. Like which frequencies
matter for identifiying word boundries etc:
```c++
    struct ggml_tensor * pos = ggml_mul_mat(ctx0, model.layers[il].attn_pos_w, pos_raw);
```
At this point Q_cur has the shape:
```console
(gdb) p Q_cur->ne
$28 = {1024, 188, 1, 1}
```
But we are using multihead attention so we need to reshape this to
```console
{128, 8, 188, 1}
0
   0 [0  ... 127]
     ...
   7 [0  ... 127]

1
   0 [0  ... 127]
     ...
   7 [0  ... 127]
...
187
   0 [0  ... 127]
     ...
   7 [0  ... 127]
```

```c++
    Q_cur = ggml_reshape_3d(ctx0, Q_cur, d_head, n_head, n_time);
```
We also reshape the position tensor which recall was:
```console
(gdb) p pos->ne
$33 = {1024, 375, 1, 1}
```
```c++
        pos   = ggml_reshape_3d(ctx0,   pos, d_head, n_head, p_len);
(gdb) p pos->ne
$35 = {128, 8, 375, 1}
```
Next we have the this layers learned content bias which allows the model to ensure
that certain query features should be important regardless if the attention score
is high or not. It is like something that the model is always looking for.
So by adding this to the query it is like telling it to "look for things that
match my current audio `and` look for things that i've learned that are generally
important in every sentence.
```c++
    struct ggml_tensor * Q_u = ggml_add(ctx0, Q_cur, model.layers[il].attn_pos_bias_u);
```
```console
(gdb) p model.layers[il].attn_pos_bias_u->ne
$36 = {128, 8, 1, 1}
(gdb) p Q_u->ne
$37 = {128, 8, 188, 1}
```
And we also have something similar but this time it is for positional scores, so
this help the model decide which distances are important.
```c++
    struct ggml_tensor * Q_v = ggml_add(ctx0, Q_cur, model.layers[il].attn_pos_bias_v);
```
```console
(gdb) p model.layers[il].attn_pos_bias_v->ne
$38 = {128, 8, 1, 1}
```
```console
(gdb) p K_prep->ne
$9 = {128, 188, 8, 1}
(gdb) p Q_prep->ne
$10 = {128, 188, 8, 1}
(gdb) p matrix_ac->ne
$11 = {188, 188, 8, 1}
```
Next we have:
```c++
            struct ggml_tensor * P_T = ggml_permute(ctx0, pos, 0, 2, 1, 3);
            struct ggml_tensor * matrix_bd = ggml_mul_mat(ctx0, P_T, Q_v);
```
So this is transposing the position tensor, then this is multiplied by Q_v, the
query with the position bias added.
```console
(gdb) p matrix_bd->ne
$1 = {188, 375, 8, 1}
```
Lets walk through this relative shift:
```console
(gdb) p content_scores->ne
$10 = {188, 188, 8, 1}         // Audio i vs Audio j
(gdb) p rel_pos_scores->ne     // Audio i vs every possible distance
$9 = {375, 188, 8, 1}
```

So the starting state is the following:
```console
(gdb) p rel_pos_scores->ne
$9 = {375, 188, 8, 1}
```
So we have 188 audio frames and each has 375 distance values. And we have 8
heads.

We will swap the first two dimensions:
```c++
            rel_pos_scores = ggml_cont(ctx0, ggml_permute(ctx0, rel_pos_scores, 1, 0, 2, 3));
            // rel shift
            {
                const auto pos_len = rel_pos_scores->ne[0];
                const auto q_len = rel_pos_scores->ne[1];
                const auto h  = rel_pos_scores->ne[2];
```

```console
(gdb) p rel_pos_scores->ne
$20 = {188, 375, 8, 1}
```

```
(gdb) p pos_len
$13 = 188
(gdb) p q_len
$14 = 375
(gdb) p h
$15 = 8
```
Next we us pad to add a column of spacing, by using ggml_pad and specifying the
first dimension:
```c++
                rel_pos_scores = ggml_pad(ctx0, rel_pos_scores, 1, 0, 0, 0);
```
```console
(gdb) p rel_pos_scores->ne
$17 = {189, 375, 8, 1}
```
Next we are using ggml_roll and we are shifting the first dimension by one and
entries will roll over at the end to the beginning.
```c++
                rel_pos_scores = ggml_roll(ctx0, rel_pos_scores, 1, 0, 0, 0);
```
So the first dimension is now 189 after the padding. So is this moving the
padding to the first column. So we padded by one and then moved that empty column
to be the first.

Next we rehape:
```c++
                rel_pos_scores = ggml_reshape_3d(ctx0, rel_pos_scores, q_len, pos_len + 1, h);
```
```console
(gdb) p rel_pos_scores->ne
$24 = {375, 189, 8, 1}
```

```c++
                rel_pos_scores = ggml_view_3d(ctx0, rel_pos_scores, q_len, pos_len, h, rel_pos_scores->nb[1],
                                              rel_pos_scores->nb[2], rel_pos_scores->nb[0] * q_len);
```
```console
(gdb) p rel_pos_scores->ne
$25 = {375, 188, 8, 1}
```

```c++
                rel_pos_scores = ggml_cont_3d(ctx0, rel_pos_scores, pos_len, q_len, h);
```
```console
(gdb) p rel_pos_scores->ne
$26 = {188, 375, 8, 1}
```

```c++
            }
            rel_pos_scores = ggml_view_3d(ctx0, rel_pos_scores, content_scores->ne[0], rel_pos_scores->ne[1],
                                          rel_pos_scores->ne[2], rel_pos_scores->nb[1], rel_pos_scores->nb[2], 0);
```

So lets imagine that the relative position matrix is a [5, 3, 1, 1]:
```console
  Row 0:   -2.0  -1.0   0.0   1.0   2.0
  Row 1:   -2.0  -1.0   0.0   1.0   2.0
  Row 2:   -2.0  -1.0   0.0   1.0   2.0
```
And our content score might look something like this ([3, 3, 1, 1]):
```console
  Row 0:    0.1   0.2   0.3
  Row 1:    0.4   0.5   0.6
  Row 2:    0.7   0.8   0.9
```

We want to produce a relative position matrix that looks like this:
```console
  Row 0:    0.0   1.0   2.0
  Row 1:   -1.0   0.0   1.0
  Row 2:   -2.0  -1.0   0.0
```
With this is is possible for Row 0 (frame 0) can look at Frame 0: Distance is 0
because (0,0) in the position matrix is 0. To look up frame 1 it uses (0,1) which
is 1 so distance is +1.
```console
  Row 0:    0.1   0.2   0.3
  Row 1:    0.4   0.5   0.6
  Row 2:    0.7   0.8   0.9
```


```console
(Pdb) p pos_emb.shape
torch.Size([1, 275, 1024])
```
```console
(Pdb) b venv/lib/python3.12/site-packages/nemo/collections/asr/parts/submodules/multi_head_attention.py:1087
```


```console
(Pdb) b /home/danbev/work/ai/whisper-models/nvidia/parkeet-tdt-0.6b-v3/venv/lib/python3.12/site-packages/nemo/collections/asr/modules/conformer_encoder.py:627
Breakpoint 1 at /home/danbev/work/ai/whisper-models/nvidia/parkeet-tdt-0.6b-v3/venv/lib/python3.12/site-packages/nemo/collections/asr/modules/conformer_encoder.py:627

```


### Encoder
The following is taken from the model_config.json file:
```console
encoder:
  _target_: nemo.collections.asr.modules.ConformerEncoder
  feat_in: 128
  feat_out: -1
  n_layers: 24
  d_model: 1024
  use_bias: false
  subsampling: dw_striding
  subsampling_factor: 8
  subsampling_conv_channels: 256
  causal_downsampling: false
  reduction: null
  reduction_position: null
  reduction_factor: 1
  ff_expansion_factor: 4
  self_attention_model: rel_pos
  n_heads: 8
  att_context_size:
  - -1
  - -1
  att_context_style: regular
  xscaling: false
  untie_biases: true
  pos_emb_max_len: 5000
  conv_kernel_size: 9
  conv_norm_type: batch_norm
  conv_context_size: null
  dropout: 0.1
  dropout_pre_encoder: 0.1
  dropout_emb: 0.0
  dropout_att: 0.1
  stochastic_depth_drop_prob: 0.0
  stochastic_depth_mode: linear
  stochastic_depth_start_layer: 1
```
So we have our input audio which consists of samples and the number of samples,
the data points depends on the sampling rate. For 30 seconds of audio and a sampling
rate of 16kHz we have 480000 samples (data points). That is 16000 samples per
second.
We don't pass the full 480000 data points to the model, instead we first convert
them into frames/windows using STFT. A standard frame size is 25ms which gives
0.25*16000 = 400 samples per frame. We move this window forward by 10ms each time.
```console
Total duration / stride = number of frames
30000 ms      / 10 ms  = 3000 frames
```
So we have 3000 frames and each frame has 400 samples. But this is also a lot
for the model so we subsample by grouping frames together. We might group 4
frames together which gives us 3000/4 = 750 frames and now each vector respresents
40ms of audio.
Or we might group 8 frames together which gives us 3000/8 = 375 frames which is
80ms of audio per vector. This is the `subsampling_factor` of 8 in the config.
```
Raw samples:    Every single vibration   480000 data points
SFTF frames:    10ms snapshots           3000 frames
Whisper input:  40ms combined blocks     750 vectors
Parakeet input: 80ms combined blocks     375 vectors
```
So both whisper.cpp and Parakeet do subsampling but Parakeet does more aggressive
subsampling (8x vs 4x), and they also use different subsampling methods. Both do
convolutions subsampling but Parakeet does a VGGNet style while whisper.cpp does
a strided convolution approach.

So the encoder flow for Parakeet is as follows (where we discussed the pre_endcode
layer previously):
```console
  (encoder): ConformerEncoder(
    (pre_encode): ConvSubsampling(
      (out): Linear(in_features=4096, out_features=1024, bias=True)
      (conv): MaskedConvSequential(
        (0): Conv2d(1, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (1): ReLU(inplace=True)
        (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        (3): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (4): ReLU(inplace=True)
        (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        (6): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (7): ReLU(inplace=True)
      )
    )
    (pos_enc): RelPositionalEncoding(
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (layers): ModuleList(
      (0-23): 24 x ConformerLayer(
        (norm_feed_forward1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        (feed_forward1): ConformerFeedForward(
          (linear1): Linear(in_features=1024, out_features=4096, bias=False)
          (activation): Swish()
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=4096, out_features=1024, bias=False)
        )
        (norm_conv): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        (conv): ConformerConvolution(
          (pointwise_conv1): Conv1d(1024, 2048, kernel_size=(1,), stride=(1,), bias=False)
          (depthwise_conv): CausalConv1D(1024, 1024, kernel_size=(9,), stride=(1,), groups=1024, bias=False)
          (batch_norm): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activation): Swish()
          (pointwise_conv2): Conv1d(1024, 1024, kernel_size=(1,), stride=(1,), bias=False)
        )
        (norm_self_att): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        (self_attn): RelPositionMultiHeadAttention(
          (linear_q): Linear(in_features=1024, out_features=1024, bias=False)
          (linear_k): Linear(in_features=1024, out_features=1024, bias=False)
          (linear_v): Linear(in_features=1024, out_features=1024, bias=False)
          (linear_out): Linear(in_features=1024, out_features=1024, bias=False)
          (dropout): Dropout(p=0.1, inplace=False)
          (linear_pos): Linear(in_features=1024, out_features=1024, bias=False)
        )
        (norm_feed_forward2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        (feed_forward2): ConformerFeedForward(
          (linear1): Linear(in_features=1024, out_features=4096, bias=False)
          (activation): Swish()
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=4096, out_features=1024, bias=False)
        )
        (dropout): Dropout(p=0.1, inplace=False)
        (norm_out): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      )
    )
  )
```

```console
  (decoder): RNNTDecoder(
    (prediction): ModuleDict(
      (embed): Embedding(8193, 640, padding_idx=8192)
      (dec_rnn): LSTMDropout(
        (lstm): LSTM(640, 640, num_layers=2, dropout=0.2)
        (dropout): Dropout(p=0.2, inplace=False)
      )
    )
  )
  (joint): RNNTJoint(
    (pred): Linear(in_features=640, out_features=640, bias=True)
    (enc): Linear(in_features=1024, out_features=640, bias=True)
    (joint_net): Sequential(
      (0): ReLU(inplace=True)
      (1): Dropout(p=0.2, inplace=False)
      (2): Linear(in_features=640, out_features=8198, bias=True)
    )
    (_loss): RNNTLoss(
      (_loss): TDTLossNumba()
    )
    (_wer): WER()
  )
  (loss): RNNTLoss(
    (_loss): TDTLossNumba()
  )
  (spec_augmentation): SpectrogramAugmentation(
    (spec_augment): SpecAugment()
  )
  (wer): WER()
)
```

### Decoder
So we have first process the audio to the the log mel spectrogram, which is
passed through the pre-encoder which filters, subsamples the time dimension, and
projects the features into the model's abstract feature space. The encoder then
processes this information through a series of conformer layers. 

In the decoder we first have the network, this takes the input token and looks
up the embedding and then passes it through the LSTM layers. The output of this
is then passed to the joint network which combines it with the encoder output to
produce the final output.

### Debugging
```console
(venv) $ python -m pdb test-model.py
(Pdb) b venv/lib/python3.12/site-packages/nemo/collections/asr/models/rnnt_models.py:698
(Pdb) r
Transcribing: 0it [00:00, ?it/s]> /home/danbev/work/ai/whisper-models/nvidia/parkeet-tdt-0.6b-v3/venv/lib/python3.12/site-packages/nemo/collections/asr/models/rnnt_models.py(698)forward()
-> has_input_signal = input_signal is not None and input_signal_length is not None
(Pdb) l
693  	        Returns:
694  	            A tuple of 2 elements -
695  	            1) The log probabilities tensor of shape [B, T, D].
696  	            2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
697  	        """
698 B->	        has_input_signal = input_signal is not None and input_signal_length is not None
699  	        has_processed_signal = processed_signal is not None and processed_signal_length is not None
700  	        if (has_input_signal ^ has_processed_signal) is False:
701  	            raise ValueError(
702  	                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
703  	                " with ``processed_signal`` and ``processed_signal_len`` arguments."
```
```python
    def forward(
        self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None
    ):
    ...
        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal,
                length=input_signal_length,
            )
            ...

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        return encoded, encoded_len
```
So we first have the audio samples.

/home/danbev/work/ai/whisper-models/nvidia/parkeet-tdt-0.6b-v3/venv/lib/python3.12/site-packages/nemo/collections/asr/modules/audio_preprocessing.py

```console
(Pdb) b /home/danbev/work/ai/whisper-models/nvidia/parkeet-tdt-0.6b-v3/venv/lib/python3.12/site-packages/nemo/collections/asr/modules/audio_preprocessing.py:84
```

```python
class AudioPreprocessor(NeuralModule, ABC):
    """
    An interface for Neural Modules that performs audio pre-processing,
    transforming the wav files to features.
    """
```

Then the encoder will be called which is a Conformer.
```console
(Pdb) b venv/lib/python3.12/site-packages/nemo/collections/asr/models/rnnt_models.py:716
```

```console
/home/danbev/work/ai/whisper-models/nvidia/parkeet-tdt-0.6b-v3/venv/lib/python3.12/site-packages/nemo/collections/asr/modules/conformer_encoder.py
```

```python
class ConformerEncoder(NeuralModule, StreamingEncoder, Exportable, AccessMixin):
    """
    The encoder for ASR model of Conformer.
    Based on this paper:
    'Conformer: Convolution-augmented Transformer for Speech Recognition' by Anmol Gulati et al.
    https://arxiv.org/abs/2005.08100
```

### Python Example
This is the python example that I've used for debugging/exploring the original
model.
```python
import nemo.collections.asr as nemo_asr

asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3")
output = asr_model.transcribe(['jfk.wav'])
print(output[0].text)

```

### Connectionist Temporal Classification (CTC)
In ASR (Automatic Speech Recognition) the audio signal is much longer than the
resulting transcript. Like 10s of audio might produce 1000 frames by the encoder
but only 15 letters of text.

Process flow:
1. Preprocessing: convert .wav to mel-spectrograms
2. Encoder: series of convolutions or transformers to compress the time dimension.
3. Linear layer: Takes the encoder output and passes it through a linear layer
   to produce a probability distribution over the vocabulary for each time step.

So instead of feeding the mel spectrogram to a decoder like we do in whisper.cpp
the CTC model will process the mel spectrogram directly into a probability
distribution over the vocabulary.

Lets say that the linear layer for the word "Hi" produced:
```console
Frame   Top prediction           Probability
1          H                        0.9
2          H                        0.7
3          - (blank)                0.8
4          I                        0.9
5          I                        0.8
```
The greedy decoder would:
* Iterate through each frame [H, H, -, I, I]
* Collapse repeats [H, -, I]
* Remove blanks [H I]

So where whisper.cpp would generate a sequence token by token, CTC models
generate the logits for all time frames at once, there is no loop. And then it
performs the above steps on the logits to produce the final text.
output:
```console
frame 0 [0           n_vocab]
frame 1 [0           n_vocab]
...
frame N [0           n_vocab]
```
A CTC depends on the Encoder to a good job on determining the features, to
"compensate" for not having attention or being able looking back as it processes
the whole audio chunk at once.

### Token-and-Duration Transducer (TDT)
The TDT model is a variant of the RNN-T (Recurrent Neural Network Transducer). A
RNN-T would process the output frames from the encoder one by one, checking if
there is a letter here? But it might spend a considerable amount of time for blank
spaces (no sound because human speech is sparse).
When a TDT processes a frame it will also inspect one by one but it will produce
two outputs:
1. The token (letter) to predict
2. The duration to predict (how long to hold that token)

It then uses the duration to skip ahead that number of frames in the mel
spectrogram before processing the next frame.

The output will be a shorted list:
```console
Index   Source Frame   Predicted Token   Duration   Logit/Prob
0       frame 0        'H'               4          0.98
1       frame 4        'E'               2          0.95
2       frame 6        'L'               3          0.92
...
```

### Joint-Network
So in a TDT we have three components:
* Encoder       : processes the mel spectrogram
* Predictor     : like a small LLM that looks at the previous text that was transcribed
* Joint Network : takes one vector from the encoder and one from the predictor and combines them

The joint network is a simple neural network that combines the encoder and
predictor, something like this:
```console
enc_proj + pred_proj → ReLU → linear → [frames+vocab]
```
In whisper.cpp this is taken care of by the decoder so there is no separate
component/layer for this.

### pre-emphasis
TODO:


### impl notes
So after the attention we have the following tensor:
```console
(gdb) p cur->ne
$6 = {1024, 188, 1, 1}
```
So we have 188 time frames each with an embedding dimension of 1024:
```console
  0 [0   ...   1023]
         ...
 187[0   ...   1023]
```
This will go through a normalization and then a pointwise convolution:
```console
{1024, 188, 1, 1} -> {2048, 188, 1, 1}
```
So we have now doubles the hidden space.

And then we have the glu which and after that we have:
```console
(gdb) p cur->ne
$8 = {188, 1024, 1, 1}
```

```console
{9, 1024, 1, 1}

    0 [0  ... 8]
          ...
          ...
          ...
          ...
 1023 [0  ... 8]
 ```
So we have 1024 different 9 element kernels. Each of there were trained
to detect different things from the hidden embeddings for this layer.

And at this point we have 188 + 9 padding audio frames (time sequence), and
the embedding dimension/feature dimension i
```console
 {196, 1024, 1, 1}

   0 [0    ...    195]
           ...
           ...
           ...
           ...
 1023[0    ...    195]
```
 For each embedding dimension we want to slide the kernel over them, one for
 each row. For example kernel 3 only slides over feature 3, they are specific
 to a feature. To achive this we need to reshape the tensors.
```console
{9, 1, 1024, 1}

0
    0 [0  ... 8]
1 
    0 [0  ... 8]

...

1023
    0 [0  ... 8]
 ```
And likewise for the input:
```console
 {196, 1, 1024, 1}
0
   0 [0    ...    195]
1
   0 [0    ...    195]
...
1023
   0 [0    ...    195]
```

 ```console
    struct ggml_tensor * dw_weights = ggml_reshape_3d(ctx0, model.layers[il].conv_dw_w, 9, 1, 1024);
    // {196, 1024, 1, 1} -> {196, 1, 1024, 1}
    // We have 196 audio frames (time sequences) 188 + 9 padding.
    struct ggml_tensor * dw_input = ggml_reshape_3d(ctx0, cur, 196, 1, 1024);
```

### Troubleshooting
Check the hann window:
```console
(Pdb) p self.preprocessor.featurizer.window
tensor([0.0000e+00, 6.1989e-05, 2.4796e-04, 5.5784e-04, 9.9158e-04, 1.5491e-03,
        2.2301e-03, 3.0347e-03, 3.9624e-03, 5.0132e-03, 6.1867e-03, 7.4826e-03,
        8.9007e-03, 1.0441e-02, 1.2102e-02, 1.3884e-02, 1.5787e-02, 1.7810e-02,
        1.9952e-02, 2.2214e-02, 2.4594e-02, 2.7091e-02, 2.9706e-02, 3.2438e-02,
        3.5286e-02, 3.8249e-02, 4.1326e-02, 4.4517e-02, 4.7821e-02, 5.1238e-02,
```
```console
DEBUG: Using window size=400, model window available=1
DEBUG: First 10 window values: 0.000000 0.000062 0.000248 0.000558 0.000992 0.001549 0.002230 0.003035 0.003962 0.005013
```
This looks alright.

```console
(Pdb) p self.preprocessor.featurizer.normalize
'per_feature
```



Print out the log mel spectrogram:
Original model:
```console
(Pdb) p processed_signal.shape
torch.Size([1, 128, 1101])
(Pdb) p processed_signal_length
tensor([1100])

(Pdb) p processed_signal.flatten()[:10]
tensor([-2.1009, -2.1009, -2.1009, -2.1009, -2.1009, -2.1006, -0.9609, -0.7740, -1.4154, -1.3690])
```
Converted model:
```console
$1 = std::vector of length 140928, capacity 140928 = {-2.05021167, -2.0502038, -2.05019426, -2.0502038, -2.05020976, -2.05017686,
  -2.01004934, -0.266436309, -1.78740978, -0.941619873, 0.0673485398, 0.504330456, -1.34932756, -0.418367326, -0.718491852,
  -0.104035422, -1.82856262, -0.803638935, -0.167220548, -1.65037513, 0.787270248, 0.189204603, -1.20511734, -0.908368051,
  -0.855753958, -0.605012834, -0.384045064, -0.134049699, -1.33701825, -0.594381094, -0.104019038, -1.60528338, -1.29771447,
  -0.54337126, 0.71394676, -1.04538357, -0.327619195, -1.06872046, -1.03802013, -1.74886429, -1.19933796, -1.18783236, -1.70146394,
```


(Pdb) p sample_rate
16000
(Pdb) p n_window_size
400
(Pdb) p n_window_stride
160
(Pdb) p window
'hann'
(Pdb) p normalize
'per_feature'
(Pdb) p n_fft
512
(Pdb) p preemph
0.97
(Pdb) p nfilt
128
(Pdb) p lowfreq
0
(Pdb) p highfreq
None
(Pdb) p log
True
(Pdb) p log_zero_guard_type
'add'
(Pdb) p log_zero_guard_value
5.960464477539063e-08
(Pdb) p dither
1e-05
(Pdb) p pad_to
0
(Pdb) p max_duration
16.7
(Pdb) p frame_splicing
1
(Pdb) p exact_pad
False
(Pdb) p pad_value
0.0
(Pdb) p mag_power
2.0
(Pdb) p use_grads
False
(Pdb) p rng
None
(Pdb) p nb_augmentation_prob
0.0
(Pdb) p nb_max_freq
4000
(Pdb) p mel_norm
'slaney'


------------------------------
### Checking tensors

Window function:
```console
(Pdb) pp self.window.shape
torch.Size([400])

(Pdb) pp self.window.view(-1)[:10]
tensor([0.0000e+00, 6.1989e-05, 2.4796e-04, 5.5784e-04, 9.9158e-04, 1.5491e-03, 2.2301e-03, 3.0347e-03, 3.9624e-03, 5.0132e-03])
(Pdb) p (self.window.double()**2).mean().item()
0.374062509671165
```

```console
window func size: 400:
0.000000 0.000062 0.000248 0.000558 0.000992 0.001549 0.002230 0.003035 0.003962 0.005013
Sum of squares = 0.374063
```

Log melspectrogram in python:
```console
(Pdb) p x.shape
torch.Size([1, 128, 1101])

(Pdb) p x.view(-1)[:10]
tensor([-2.1009, -2.1009, -2.1009, -2.1009, -2.1009, -2.1006, -0.9609, -0.7740, -1.4154, -1.3690])

(Pdb) p (x.double()**2).mean().item()
0.9981744415301794
```

```console
DEBUG: Mel spectrogram AFTER normalization:
-2.050006 -2.050006 -2.050006 -2.050006 -2.050006 -2.049972 -2.009351 -0.266303 -1.787033 -0.943058
Mean of squares (all values) = 0.9990137367
```

And the same tensor in parakeet.cpp look like this:
```console
(gdb) p model.enc_pre_conv_0_w->ne
$1 = {3, 3, 1, 256}

Tensor 'enc_pre_conv_0_w', type: f32
ne = [3 3 1 256]
Tensor value at [0, 0, 0, 0]: 0.096362
Tensor value at [1, 0, 0, 0]: 0.025928
Tensor value at [2, 0, 0, 0]: -0.486745
Tensor value at [0, 1, 0, 0]: -0.226291
Tensor value at [1, 1, 0, 0]: -0.319316
Tensor value at [2, 1, 0, 0]: 1.150509
Tensor value at [0, 2, 0, 0]: -0.009058
Tensor value at [1, 2, 0, 0]: 0.119550
Tensor value at [2, 2, 0, 0]: 0.008535
Tensor value at [0, 0, 0, 1]: 0.058622
enc_pre_conv_0_w mean_sq = 0.1960858460
```
So we have a pretty close match and I think we can say that the weights are
correct for this tensor. Now lets look at the output of the convolution using
this tensor.

And the input x tensor is:
```console
(Pdb) p x
tensor([[[[-2.1009, -3.3455, -2.7749,  ..., -1.6735, -1.6589, -1.7144],
          [-2.1009, -3.3455, -2.7749,  ..., -1.6735, -1.6589, -1.7144],
          [-2.1009, -3.3455, -2.7749,  ..., -1.6735, -1.6589, -1.7144],
          ...,
          [ 0.5293,  0.3247,  0.3585,  ...,  1.1187,  0.5583,  0.5893],
          [ 0.4928, -0.4181, -0.5930,  ...,  1.0809,  1.1783,  1.1371],
          [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]]]])
(Pdb) p x.is_contiguous()
False
(Pdb) p x.contiguous().view(-1)[:10]
tensor([-2.1009, -3.3455, -2.7749, -4.7532, -4.3730, -5.6027, -4.7664, -4.5884, -3.5901, -3.6141])
```

After the convolution operation, which also adds the bias:
```console
(Pdb) p layer.bias.shape
torch.Size([256])

(Pdb) p layer.bias.view(-1)[:10]
tensor([-3.4104e-01,  2.1441e-01, -7.0875e-01, -5.4349e-01, -4.5362e-01, -6.2717e-02, -5.1516e-01, -2.3044e-01,  9.4809e-06, -3.9431e-03])

(Pdb) p (layer.bias.double()**2).mean().item()
0.09868425240328094
```

And the bias in parakeet.cpp:
```console
Tensor 'enc_pre_conv_0_b', type: f32
ne = [1 1 256 1]
Tensor value at [0, 0, 0, 0]: -0.341044
Tensor value at [0, 0, 1, 0]: 0.214413
Tensor value at [0, 0, 2, 0]: -0.708747
Tensor value at [0, 0, 3, 0]: -0.543491
Tensor value at [0, 0, 4, 0]: -0.453616
Tensor value at [0, 0, 5, 0]: -0.062717
Tensor value at [0, 0, 6, 0]: -0.515164
Tensor value at [0, 0, 7, 0]: -0.230438
Tensor value at [0, 0, 8, 0]: 0.000009
Tensor value at [0, 0, 9, 0]: -0.003943
enc_pre_conv_0_b mean_sq = 0.0986842517
```

output in python:
```console
(Pdb) p x.contiguous().view(-1)[:10]
tensor([-3.7990, -4.5085, -4.8426, -3.3884, -2.7329, -3.0468, -3.6646, -2.8045, -2.7402, -2.3244])
(Pdb) p (x.double()**2).mean().item()
0.6077475048109069
```

output in parakeet.cpp:
```console
Tensor 'pre_conv_0_bias', type: f32
ne = [551 64 256 1]
Tensor value at [0, 0, 0, 0]: -2.477598
Tensor value at [1, 0, 0, 0]: -1.983089
Tensor value at [2, 0, 0, 0]: -1.983040
Tensor value at [3, 0, 0, 0]: 0.105437
Tensor value at [4, 0, 0, 0]: -0.858916
Tensor value at [5, 0, 0, 0]: 0.364040
Tensor value at [6, 0, 0, 0]: -0.588580
Tensor value at [7, 0, 0, 0]: -0.093665
Tensor value at [8, 0, 0, 0]: -0.762478
Tensor value at [9, 0, 0, 0]: -2.190899
pre_conv_0_bias mean_sq = 0.6100155181
```
So there is a mismatch in the output values. Lets just double check that the
inputs to the operations are the same.
Python:
```console
(Pdb) p x.shape
torch.Size([1, 1101, 128])

(Pdb) p x.contiguous().view(-1)[:10]
tensor([-2.1009, -3.3455, -2.7749, -4.7532, -4.3730, -5.6027, -4.7664, -4.5884, -3.5901, -3.6141])

(Pdb) p (x.double()**2).mean().item()
0.9981744415301794
```
Then there is an unsqueeze:
```console
(Pdb) p x.shape
torch.Size([1, 1, 1101, 128])
```
Then we have this mask:
```console
(Pdb) l
724
725  	class MaskedConvSequential(nn.Sequential):
726  	    def forward(self, x, lengths):
727  	        # Convert input (batch, time, features) to conv format
728 B	        x = x.unsqueeze(1)  # (batch, 1, time, features)
729  ->	        current_lengths = lengths.clone().float()
730  	        mask = self._create_mask(x, current_lengths.long())
```
```console
(Pdb) p mask.contiguous().view(-1)[:10]
tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

(Pdb) p (mask.double()**2).mean().item()
0.9990917347865577
```
And this mask is applied:
```console
(Pdb) l .
730  	        mask = self._create_mask(x, current_lengths.long())
731
732  	        # Process through each layer with mask propagation
733  	        for i, layer in enumerate(self):
734  	            # Apply current mask before layer
735  ->	            x = apply_channel_mask(x, mask)
736
737  	            # Apply layer
738  	            x = layer(x)
```
```console
(Pdb) p x.contiguous().view(-1)[:10]
tensor([-2.1009, -3.3455, -2.7749, -4.7532, -4.3730, -5.6027, -4.7664, -4.5884, -3.5901, -3.6141])
(Pdb) p (x.double()**2).mean().item()
0.9981744415301794
```
But this does not change the data.

In parakeet.cpp:
```console
Tensor 'mel', type: f32
ne = [1101 128 1 1]
Tensor value at [0, 0, 0, 0]: -2.049985
Tensor value at [1, 0, 0, 0]: -2.049981
Tensor value at [2, 0, 0, 0]: -2.049985
Tensor value at [3, 0, 0, 0]: -2.049984
Tensor value at [4, 0, 0, 0]: -2.049968
Tensor value at [5, 0, 0, 0]: -2.049945
Tensor value at [6, 0, 0, 0]: -2.008003
Tensor value at [7, 0, 0, 0]: -0.266578
Tensor value at [8, 0, 0, 0]: -1.787163
Tensor value at [9, 0, 0, 0]: -0.942764
mel mean_sq = 0.9986481456
```


### conv tensor conversion
```console
(Pdb) p layer.weight.shape
torch.Size([256, 1, 3, 3])
```
```c++
model.enc_pre_conv_0_w = create_tensor(PARAKEET_TENSOR_ENC_PRE_CONV_0_WEIGHT, ggml_new_tensor_4d(ctx, vtype, 3, 3, 1, n_subsampling_channels));
```



Perfect! Now I can see the exact problem. Comparing the mel spectrum values before log:

  Python:
  - Frame 6: 1.2174e-07
  - Frame 7: 1.5803e-07
  - Frame 8: 5.6772e-08
  - Frame 9: 6.2160e-08

  C++:
  - Frame 6: 2.4475e-09 (50x smaller!)
  - Frame 7: 2.8876e-07 (1.8x larger)
  - Frame 8: 1.7721e-08 (3.2x smaller)
  - Frame 9: 1.1868e-07 (1.9x larger)

  The values are significantly different, meaning the issue is in the power spectrum calculation or mel filterbank application,
  not in normalization.

  Since the window function matches perfectly and padding looks correct, let me check the mel filterbank weights. Can you verify
  in Python:

  # Check filterbank values
  print(f"\nDEBUG: Mel filterbank first bin (first 10 weights):")
  print(f"{self.fb[0, 0, :10]}")
  print(f"Filterbank mean squared: {(self.fb.double()**2).mean().item()}")

  Also, I need to check if there's a difference in how the power spectrum is computed. In your earlier Python output, you showed
  the mean squared of the power spectrum before mel filterbank was 36.698161585839.

  Can you add debug to C++ to print the power spectrum values for a specific frame (like frame 7 where audio starts)?


### Time-restricted Durations
In the model configuration we have:
```console
model_defaults:
  ...
  tdt_durations:
  - 0
  - 1
  - 2
  - 3
  - 4
  num_tdt_durations: 5
```
The joint network outputs not just log probs for the vocabulary which is 8192
but it outputs 8198:
```console
Tensor 'log_probs', type: f32
ne = [8198 1 1 1]
nb = [4 32792 32792 32792]
Tensor value at [0, 0, 0, 0]: -30.037724
Tensor value at [1, 0, 0, 0]: -41.217815
Tensor value at [2, 0, 0, 0]: -41.076900
Tensor value at [3, 0, 0, 0]: -41.158520
Tensor value at [4, 0, 0, 0]: -41.102310
Tensor value at [5, 0, 0, 0]: -41.144932
Tensor value at [6, 0, 0, 0]: -41.136822
Tensor value at [7, 0, 0, 0]: -41.068092
Tensor value at [8, 0, 0, 0]: -41.137356
Tensor value at [9, 0, 0, 0]: -41.134548
log_probs mean_sq = 2587.2295275343
```
* 0-8191 are the vocabulary tokens (8192)
* 8192 is the Blank token
* 8193-8197 are the duration predictions.

This means:
* Duration 0: Stay at current frame (emit another token at same timestep)
* Duration 1: Advance by 1 frame
* Duration 2: Skip ahead 2 frames
* Duration 3: Skip ahead 3 frames
* Duration 4: Skip ahead 4 frames

I was not aware of this initially and I need to update the conversion script
to include these fields and update the model loading code and the model in
parakeet.cpp to handle the duration predictions. I've hardcoded this at the
moment to 5 which matches the original model.


### Chunking 
So after working on the original implementation of parakeet.cpp I was able to
got through a simple audio file and comparing the operation outputs and verify
that they closely match the original model.

This is from the original model transcribing:
```console
[1976,  547, 7877, 1103,  309,  530,  596, 3213,  404,  667, 7877,  279,
583, 1491, 3470, 3629,  867,  331,  958, 7893, 2059,  458,  509, 1180,
7877,  279,  583, 3470, 1180, 2059,  458,  509, 3629,  867,  331,  958,
7893, 7883]),
text='And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.'
```

And this is the output of the converted parakeet.cpp:
```console
Processing audio: total_frames=1101, chunk_size=1101
parakeet_decode: starting decode with n_frames=138
Decoded 38 new tokens (total: 38). First 20 new:
1976 547 7877 1103 309 530 596 3213 404 667 7877 279 583 1491 3470 3629 867 331 958 7893 2059 458 509 1180 7877 279 583 3470 1180 2059 458 509 3629 867 331 958 7893 7883
Segment [     0 ->   1101]: And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
```
But for a longer audio file we need to split the longer audio into chunks. So we
would process each chunk through the pre-encoder and the encoder (the subsampling)
and then we can use the same prediction network state, this is the LSTM state, 
and then continue with the joint network to decode the tokens. But just naively
splitting the audio will not work as there will be cut offs which can cause
incorrect transcriptions. And we also need to take the duration into consideration
when doing this. I'll take a look at how the original model handles this to se
how it should be done.

In audio the end of one chunk needs to understand the context of the previous
chunk. For example if we cut off the middle of the sounds "b" this might be mis-
interpreted by the model.

We can use a context buffer for this so that we don't just cut off and sent
the new chunk, but we include some of the end of the last chunk as well.

we also need to take into consideration that the joint network in Parakeet also
produces duration tokens/values. These are used with argmax to get a index into
the models duration array which has values indicating how many time frames to
skip a ahead.

So when processing a chunk, we actually need to feed the end of the previous
chunk plus the current chunk:
```console
   [end of previous chunk] + [current chunk]
```
We can store this in the parakeet_state.
```c++
    // This vector stores the previous n audio frames to enable chunk.
    // So this should store the samples I think.
    std::vector<float> audio_context_buffer;
```

```console
$ ffprobe -i samples/gb1.ogg -show_entries format=duration,bit_rate -show_entries stream=sample_rate,channels -of compact=p=0:nk=1

Input #0, ogg, from 'samples/gb1.ogg':
  Duration: 00:03:18.73, start: 0.000000, bitrate: 67 kb/s
  Stream #0:0: Audio: vorbis, 22050 Hz, stereo, fltp, 88 kb/s
22050|2
198.734331|67131

$ ffmpeg -i samples/gb1.ogg -ar 16000 -ac 1 samples/gb1_16k.wav

Input #0, ogg, from 'samples/gb1.ogg':
  Duration: 00:03:18.73, start: 0.000000, bitrate: 67 kb/s
  Stream #0:0: Audio: vorbis, 22050 Hz, stereo, fltp, 88 kb/s
Stream mapping:
  Stream #0:0 -> #0:0 (vorbis (native) -> pcm_s16le (native))
Press [q] to stop, [?] for help
Output #0, wav, to 'samples/gb1_16k.wav':
  Metadata:
    ISFT            : Lavf61.7.100
  Stream #0:0: Audio: pcm_s16le ([1][0][0][0] / 0x0001), 16000 Hz, mono, s16, 256 kb/s
      Metadata:
        encoder         : Lavc61.19.101 pcm_s16le
[out#0/wav @ 0x6000035dc000] video:0KiB audio:6210KiB subtitle:0KiB other streams:0KiB global headers:0KiB muxing overhead: 0.001227%
size=    6211KiB time=00:03:18.73 bitrate= 256.0kbits/s speed=2.51e+03x

(venv) $ ffprobe -i samples/gb1_16k.wav -show_entries format=duration,bit_rate -show_entries stream=sample_rate,channels -of compact=p=0:nk=1
Input #0, wav, from 'samples/gb1_16k.wav':
  Metadata:
    encoder         : Lavf61.7.100
  Duration: 00:03:18.73, bitrate: 256 kb/s
  Stream #0:0: Audio: pcm_s16le ([1][0][0][0] / 0x0001), 16000 Hz, 1 channels, s16, 256 kb/s
16000|1
198.734375|256003
```
```console
$ ffprobe -i samples/gb1.wav -show_entries format=duration,bit_rate -show_entries stream=sample_rate,channels -of compact=p=0:nk=1
Input #0, wav, from 'samples/gb1.wav':
  Metadata:
    encoder         : Lavf61.7.100
  Duration: 00:03:18.73, bitrate: 256 kb/s
  Stream #0:0: Audio: pcm_s16le ([1][0][0][0] / 0x0001), 16000 Hz, 1 channels, s16, 256 kb/s
16000|1
198.734375|256003
```
So we have 198.73 seconds of audio, at 16000 samples per second, which gives
198.73 x 16000 = 3179750 samples:
```console
(gdb) p n_samples
$1 = 3179750
```
And after this has be processed by the logmel spectrogram generation we have:
```console
(gdb) p state.mel
$5 = {n_len = 19874, n_len_org = 19874, n_mel = 128, data = std::vector of length 2543872
```
So we have 19874 time frames, and 128 mel bins (for each time frame).
```console
0     [0  ...     127]
...
...
...
...
...
...
19873 [0  ...     127]
```
So we need to split this into chunks to be processed. What size of the chunks
should we use? Perhaps 1024 would be a good choice.

So
```c++
    state->n_audio_ctx = 1024;

    int total_mel_frames = state->mel.n_len;
    int step_size        = 800;
    int window_size      = state->n_audio_ctx;

    for (int t = 0; t < total_mel_frames; t += step_size) {
        int current_window = std::min(window_size, total_mel_frames - t);
    }
```
```c++
            float * dst = pstate.inp_mel.data();
            // zero out the mel data.
            memset(dst, 0, ggml_nbytes(mel));

            const int i0 = std::min(mel_offset,         mel_inp.n_len);
            const int i1 = std::min(mel_offset + n_ctx, mel_inp.n_len);

            for (int j = 0; j < mel_inp.n_mel; ++j) {
                for (int i = i0; i < i1; ++i) {
                    dst[(i - i0) * mel_inp.n_mel + j] = mel_inp.data[j * mel_inp.n_len + i];
                }
            }
            ggml_backend_tensor_set(mel, pstate.inp_mel.data(), 0, ggml_nelements(mel)*sizeof(float));
```
Currently, that is before the copying above, the mel data exist in memory like
the image shows above, we first have time frame 0 with its bins, then row 1 etc.
```
[Mel 0 , t0, t1, t2...], [Mel 1, t0, t1, t2...], ...
```
But we want this data in interleaved format:
```
[T0, Mel 0, Mel 1, Mel 2 ...], [T1, Mel 0, Mel 1, ...], ...
```

I've tried a few different approaches to this, but the most promising this far
is to first process the entire audio file by encoder and then run a single
decode process. This produces a pretty accurate output:
```console
My fellow Americans, this day has brought terrible news and great sadness to our country. At nine o'clock this morning, mission control in Houston lost contact with our space shuttle Columbia. A short time later, debris was seen falling from the skies above Texas. The Columbia's lost. There are no survivors. On board was a crew of seven, Colonel Rick Husband, Lieutenant Colonel Michael Anderson, Commander Laurel Clark, Captain David Brown, Commander William McCool, Dr. Kulpna Chavla, and Ilan Ramon, a colonel. These men and women assumed great risk in the service to all humanity. In an age when spaceflight has come to seem almost routine. It is easy to overlook the dangers of travel by rocket and the difficulties of navigating the fierce outer atmosphere of the Earth. These astronauts knew the dangers, and they faced them willingly, knowing they had a high and noble purpose in life. Because of their courage and daring and idealism, we will miss them all the more. All Americans today are thinking as well of the families of these men and women who have been given this sudden shock and grief. Our entire nation grieves with you. And those you loved will always have the respect and gratitude of this country. The cause in which they died will continue. Mankind is led into the darkness beyond our world by the inspiration of discovery and the longing to understand. Our journey into space will go on. In the skies today, we saw destruction and tragedy. Yet farther than we can see, there is comfort and hope. In the words of the prophet Isaiah, lift your eyes and look to the heavens. Who created all this? Yet we can pray that all are safely home. May God bless the grieving families, and may God continue to bless America.
Test passed: Parakeet model loaded and freed successfully
```
The original models output:
```console
My fellow Americans, this day has brought terrible news and great sadness to our country. At nine o'clock this morning, mission control in Houston lost contact with our space shuttle Columbia. A short time later, debris was seen falling from the skies above Texas. The Columbia's lost. There are no survivors. On board was a crew of seven, Colonel Rick Husband, Lieutenant Colonel Michael Anderson, Commander Laurel Clark, Captain David Brown, Commander William McCool, Dr. Kulpna Shavla, and Ilan Ramon, a colonel in the Israeli Air Force. These men and women assumed great risk in the service to all humanity. In an age when spaceflight has come to seem almost routine, it is easy to overlook the dangers of travel by rocket and the difficulties of navigating the fierce outer atmosphere of the Earth. Because of their courage and daring and idealism, we will miss them all the more. All Americans today are thinking as well of the families of these men and women who have been given this sudden shock and grief. You're not alone. Our entire nation grieves with you, and those you love will always have the respect and gratitude of this country. The cause in which they died will continue. Mankind is led into the darkness beyond our world by the inspiration of discovery and the longing to understand. Our journey into space will go on. In the skies today, we saw destruction and tragedy. Yet farther than we can see, there is comfort and hope. In the words of the prophet Isaiah, lift your eyes and look to the heavens. Who created all these? He who brings out the starry hosts one by one and calls them each by name. Because of his great power and mighty strength, not one of them is missing. The crew of the shuttle Columbia did not return safely to Earth. Yet we can pray that all are safely home. May God bless the grieving families, and may God continue to bless America.
```
This does not feel like an optimal solution but it works and hopefully will allow
me to try other approaches.

_wip_

Total input frames if we have 10ms chunks is 19873 frames, and with 8x
subsampling this becomes 2484 frames. This is the number of frames that the
encoder will output.

So to recap that process a bit here. We pass in the samples and the number of
samples. This will go throught the logmel spectrogram generation and this is what
is the input to the subsampling.

### Timestamps
The original parakeet model can be used with a Timestamp option which will
generate information like the following:
```console
[Hypothesis(
score=-622.7587890625,
y_sequence=tensor([1976,  547, 7877, 1103,  309,  530,  596, 3213,  404,  667, 7877,  279,
 583, 1491, 3470, 3629,  867,  331,  958, 7893, 2059,  458,  509, 1180,
7877,  279,  583, 3470, 1180, 2059,  458,  509, 3629,  867,  331,  958,
 7893, 7883]),
 text='And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.',
 dec_out=None,
 dec_state=LabelLoopingStateItem(predictor_state=(tensor([[-7.2654e-08,  6.2677e-06, -1.5610e-03,  ..., -7.7586e-08,
 ...-1.2105e-01, -2.6987e-01, -3.1783e-01, -1.6972e+00,  4.8569e-03]]),
 label=tensor(7883),
 decoded_length=tensor(138),
 fusion_state_list=[],
 time_jump=tensor(0)),
 timestamp={'timestep':
   [3, 7, 11, 13, 16, 17, 19, 22, 26, 30, 37, 41, 45, 53, 65, 69, 75, 76, 78, 80, 82, 85, 89, 93, 98, 102, 105, 109, 112, 116, 119, 121, 124, 126, 128, 129, 130, 132],
'char': [
   {'char': ['And'], 'start_offset': 3, 'end_offset': 7, 'start': 0.24, 'end': 0.56},
   {'char': ['so'], 'start_offset': 7, 'end_offset': 11, 'start': 0.56, 'end': 0.88},
   {'char': [','], 'start_offset': 11, 'end_offset': 11, 'start': 0.88, 'end': 0.88},
   {'char': ['my'], 'start_offset': 13, 'end_offset': 16, 'start': 1.04, 'end': 1.28},
   {'char': ['f'], 'start_offset': 16, 'end_offset': 17, 'start': 1.28, 'end': 1.36},
   {'char': ['ell'], 'start_offset': 17, 'end_offset': 19, 'start': 1.36, 'end': 1.52},
   {'char': ['ow'], 'start_offset': 19, 'end_offset': 22, 'start': 1.52, 'end': 1.76},
   {'char': ['Amer'], 'start_offset': 22, 'end_offset': 26, 'start': 1.76, 'end': 2.08},
   {'char': ['ic'], 'start_offset': 26, 'end_offset': 30, 'start': 2.08, 'end': 2.4},
   {'char': ['ans'], 'start_offset': 30, 'end_offset': 34, 'start': 2.4, 'end': 2.72},
   {'char': [','], 'start_offset': 34, 'end_offset': 34, 'start': 2.72, 'end': 2.72},
   {'char': ['a'], 'start_offset': 41, 'end_offset': 45, 'start': 3.2800000000000002, 'end': 3.6},
   {'char': ['sk'], 'start_offset': 45, 'end_offset': 49, 'start': 3.6, 'end': 3.92},
   {'char': ['not'], 'start_offset': 53, 'end_offset': 57, 'start': 4.24, 'end': 4.5600000000000005},
   {'char': ['what'], 'start_offset': 65, 'end_offset': 69, 'start': 5.2, 'end': 5.5200000000000005},
   {'char': ['your'], 'start_offset': 69, 'end_offset': 71, 'start': 5.5200000000000005, 'end': 5.68},
   {'char': ['co'], 'start_offset': 75, 'end_offset': 76, 'start': 6.0, 'end': 6.08},
   {'char': ['un'], 'start_offset': 76, 'end_offset': 78, 'start': 6.08, 'end': 6.24},
   {'char': ['tr'], 'start_offset': 78, 'end_offset': 80, 'start': 6.24, 'end': 6.4},
   {'char': ['y'], 'start_offset': 80, 'end_offset': 82, 'start': 6.4, 'end': 6.5600000000000005},
   {'char': ['can'], 'start_offset': 82, 'end_offset': 85, 'start': 6.5600000000000005, 'end': 6.8},
   {'char': ['do'], 'start_offset': 85, 'end_offset': 89, 'start': 6.8, 'end': 7.12},
   {'char': ['for'], 'start_offset': 89, 'end_offset': 93, 'start': 7.12, 'end': 7.44},
   {'char': ['you'], 'start_offset': 93, 'end_offset': 97, 'start': 7.44, 'end': 7.76},
   {'char': [','], 'start_offset': 97, 'end_offset': 97, 'start': 7.76, 'end': 7.76},
   {'char': ['a'], 'start_offset': 102, 'end_offset': 105, 'start': 8.16, 'end': 8.4},
   {'char': ['sk'], 'start_offset': 105, 'end_offset': 109, 'start': 8.4, 'end': 8.72},
   {'char': ['what'], 'start_offset': 109, 'end_offset': 112, 'start': 8.72, 'end': 8.96},
   {'char': ['you'], 'start_offset': 112, 'end_offset': 116, 'start': 8.96, 'end': 9.28},
   {'char': ['can'], 'start_offset': 116, 'end_offset': 119, 'start': 9.28, 'end': 9.52},
   {'char': ['do'], 'start_offset': 119, 'end_offset': 121, 'start': 9.52, 'end': 9.68},
   {'char': ['for'], 'start_offset': 121, 'end_offset': 124, 'start': 9.68, 'end': 9.92},
   {'char': ['your'], 'start_offset': 124, 'end_offset': 126, 'start': 9.92, 'end': 10.08},
   {'char': ['co'], 'start_offset': 126, 'end_offset': 128, 'start': 10.08, 'end': 10.24},
   {'char': ['un'], 'start_offset': 128, 'end_offset': 129, 'start': 10.24, 'end': 10.32},
   {'char': ['tr'], 'start_offset': 129, 'end_offset': 130, 'start': 10.32, 'end': 10.4},
   {'char': ['y'], 'start_offset': 130, 'end_offset': 132, 'start': 10.4, 'end': 10.56},
   {'char': ['.'], 'start_offset': 132, 'end_offset': 132, 'start': 10.56, 'end': 10.56}],
'word': [
   {'word': 'And', 'start_offset': 3, 'end_offset': 7, 'start': 0.24, 'end': 0.56},
   {'word': 'so,', 'start_offset': 7, 'end_offset': 11, 'start': 0.56, 'end': 0.88},
   {'word': 'my', 'start_offset': 13, 'end_offset': 16, 'start': 1.04, 'end': 1.28},
   {'word': 'fellow', 'start_offset': 16, 'end_offset': 22, 'start': 1.28, 'end': 1.76},
   {'word': 'Americans,', 'start_offset': 22, 'end_offset': 34, 'start': 1.76, 'end': 2.72},
   {'word': 'ask', 'start_offset': 41, 'end_offset': 49, 'start': 3.2800000000000002, 'end': 3.92},
   {'word': 'not', 'start_offset': 53, 'end_offset': 57, 'start': 4.24, 'end': 4.5600000000000005},
   {'word': 'what', 'start_offset': 65, 'end_offset': 69, 'start': 5.2, 'end': 5.5200000000000005},
   {'word': 'your', 'start_offset': 69, 'end_offset': 71, 'start': 5.5200000000000005, 'end': 5.68},
   {'word': 'country', 'start_offset': 75, 'end_offset': 82, 'start': 6.0, 'end': 6.5600000000000005},
   {'word': 'can', 'start_offset': 82, 'end_offset': 85, 'start': 6.5600000000000005, 'end': 6.8},
   {'word': 'do', 'start_offset': 85, 'end_offset': 89, 'start': 6.8, 'end': 7.12},
   {'word': 'for', 'start_offset': 89, 'end_offset': 93, 'start': 7.12, 'end': 7.44},
   {'word': 'you,', 'start_offset': 93, 'end_offset': 97, 'start': 7.44, 'end': 7.76},
   {'word': 'ask', 'start_offset': 102, 'end_offset': 109, 'start': 8.16, 'end': 8.72},
   {'word': 'what', 'start_offset': 109, 'end_offset': 112, 'start': 8.72, 'end': 8.96},
   {'word': 'you', 'start_offset': 112, 'end_offset': 116, 'start': 8.96, 'end': 9.28},
   {'word': 'can', 'start_offset': 116, 'end_offset': 119, 'start': 9.28, 'end': 9.52},
   {'word': 'do', 'start_offset': 119, 'end_offset': 121, 'start': 9.52, 'end': 9.68},
   {'word': 'for', 'start_offset': 121, 'end_offset': 124, 'start': 9.68, 'end': 9.92},
   {'word': 'your', 'start_offset': 124, 'end_offset': 126, 'start': 9.92, 'end': 10.08},
   {'word': 'country.', 'start_offset': 126, 'end_offset': 132, 'start': 10.08, 'end': 10.56}
],
'segment': [
    {'segment': 'And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.',
     'start_offset': 3, 'end_offset': 132, 'start': 0.24, 'end': 10.56}
 ]}, alignments=None, frame_confidence=None, token_confidence=None, word_confidence=None, length=tensor(138), y=None, lm_state=None, lm_scores=None, ngram_lm_state=None, tokens=None, last_token=None, token_duration=[4, 4, 2, 3, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 2, 1, 2, 2, 2, 3, 4, 4, 4, 4, 3, 4, 3, 4, 3, 2, 3, 2, 2, 1, 1, 2, 4], last_frame=None, biasing_cfg=None, xatt_scores=None)]
```

To match the above, though not exactly (still keeping this somewhat similar to
whisper.cpp but perhaps this should change to be more like the original):
```console
Segment 0: [0 -> 1101] "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country."
Tokens:
  [ 0] id= 1976 frame=  3 dur_idx= 4 dur_val= 4 p=0.9996 plog=-15.6162 t0=  24 t1=  56 word_start=1 "▁And"
  [ 1] id=  547 frame=  7 dur_idx= 4 dur_val= 4 p=0.9999 plog=-18.7914 t0=  56 t1=  88 word_start=1 "▁so"
  [ 2] id= 7877 frame= 11 dur_idx= 2 dur_val= 2 p=0.8453 plog=-14.5968 t0=  88 t1= 104 word_start=0 ","
  [ 3] id= 1103 frame= 13 dur_idx= 3 dur_val= 3 p=0.9996 plog=-15.6151 t0= 104 t1= 128 word_start=1 "▁my"
  [ 4] id=  309 frame= 16 dur_idx= 1 dur_val= 1 p=0.9913 plog=-11.9667 t0= 128 t1= 136 word_start=1 "▁f"
  [ 5] id=  530 frame= 17 dur_idx= 2 dur_val= 2 p=1.0000 plog=-13.5254 t0= 136 t1= 152 word_start=0 "ell"
  [ 6] id=  596 frame= 19 dur_idx= 3 dur_val= 3 p=1.0000 plog=-16.3210 t0= 152 t1= 176 word_start=0 "ow"
  [ 7] id= 3213 frame= 22 dur_idx= 4 dur_val= 4 p=0.9999 plog=-10.1507 t0= 176 t1= 208 word_start=1 "▁Amer"
  [ 8] id=  404 frame= 26 dur_idx= 4 dur_val= 4 p=1.0000 plog=-25.1000 t0= 208 t1= 240 word_start=0 "ic"
  [ 9] id=  667 frame= 30 dur_idx= 4 dur_val= 4 p=1.0000 plog=-27.1773 t0= 240 t1= 272 word_start=0 "ans"
  [10] id= 7877 frame= 37 dur_idx= 4 dur_val= 4 p=0.9093 plog=-16.3398 t0= 296 t1= 328 word_start=0 ","
  [11] id=  279 frame= 41 dur_idx= 4 dur_val= 4 p=0.9980 plog=-19.7236 t0= 328 t1= 360 word_start=1 "▁a"
  [12] id=  583 frame= 45 dur_idx= 4 dur_val= 4 p=1.0000 plog=-24.5318 t0= 360 t1= 392 word_start=0 "sk"
  [13] id= 1491 frame= 53 dur_idx= 4 dur_val= 4 p=1.0000 plog=-23.2955 t0= 424 t1= 456 word_start=1 "▁not"
  [14] id= 3470 frame= 65 dur_idx= 4 dur_val= 4 p=0.9995 plog=-16.7244 t0= 520 t1= 552 word_start=1 "▁what"
  [15] id= 3629 frame= 69 dur_idx= 2 dur_val= 2 p=0.8168 plog=-11.6476 t0= 552 t1= 568 word_start=1 "▁your"
  [16] id=  867 frame= 75 dur_idx= 1 dur_val= 1 p=0.9980 plog=-12.5256 t0= 600 t1= 608 word_start=1 "▁co"
  [17] id=  331 frame= 76 dur_idx= 2 dur_val= 2 p=1.0000 plog=-11.6734 t0= 608 t1= 624 word_start=0 "un"
  [18] id=  958 frame= 78 dur_idx= 2 dur_val= 2 p=1.0000 plog=-11.3656 t0= 624 t1= 640 word_start=0 "tr"
  [19] id= 7893 frame= 80 dur_idx= 2 dur_val= 2 p=1.0000 plog=-14.3272 t0= 640 t1= 656 word_start=0 "y"
  [20] id= 2059 frame= 82 dur_idx= 3 dur_val= 3 p=1.0000 plog=-17.7691 t0= 656 t1= 680 word_start=1 "▁can"
  [21] id=  458 frame= 85 dur_idx= 4 dur_val= 4 p=1.0000 plog=-23.2535 t0= 680 t1= 712 word_start=1 "▁do"
  [22] id=  509 frame= 89 dur_idx= 4 dur_val= 4 p=1.0000 plog=-23.0690 t0= 712 t1= 744 word_start=1 "▁for"
  [23] id= 1180 frame= 93 dur_idx= 4 dur_val= 4 p=0.9999 plog=-25.0585 t0= 744 t1= 776 word_start=1 "▁you"
  [24] id= 7877 frame= 98 dur_idx= 4 dur_val= 4 p=0.8822 plog=-14.2621 t0= 784 t1= 816 word_start=0 ","
  [25] id=  279 frame=102 dur_idx= 3 dur_val= 3 p=0.9992 plog=-16.8122 t0= 816 t1= 840 word_start=1 "▁a"
  [26] id=  583 frame=105 dur_idx= 4 dur_val= 4 p=1.0000 plog=-21.0343 t0= 840 t1= 872 word_start=0 "sk"
  [27] id= 3470 frame=109 dur_idx= 3 dur_val= 3 p=0.9999 plog=-15.4671 t0= 872 t1= 896 word_start=1 "▁what"
  [28] id= 1180 frame=112 dur_idx= 4 dur_val= 4 p=0.9997 plog=-17.6853 t0= 896 t1= 928 word_start=1 "▁you"
  [29] id= 2059 frame=116 dur_idx= 3 dur_val= 3 p=0.9999 plog=-15.5379 t0= 928 t1= 952 word_start=1 "▁can"
  [30] id=  458 frame=119 dur_idx= 2 dur_val= 2 p=1.0000 plog=-15.9920 t0= 952 t1= 968 word_start=1 "▁do"
  [31] id=  509 frame=121 dur_idx= 3 dur_val= 3 p=1.0000 plog=-15.9604 t0= 968 t1= 992 word_start=1 "▁for"
  [32] id= 3629 frame=124 dur_idx= 2 dur_val= 2 p=0.9994 plog=-12.2084 t0= 992 t1=1008 word_start=1 "▁your"
  [33] id=  867 frame=126 dur_idx= 2 dur_val= 2 p=0.9969 plog=-9.1232 t0=1008 t1=1024 word_start=1 "▁co"
  [34] id=  331 frame=128 dur_idx= 1 dur_val= 1 p=0.9999 plog=-12.6941 t0=1024 t1=1032 word_start=0 "un"
  [35] id=  958 frame=129 dur_idx= 1 dur_val= 1 p=1.0000 plog=-8.8891 t0=1032 t1=1040 word_start=0 "tr"
  [36] id= 7893 frame=130 dur_idx= 2 dur_val= 2 p=1.0000 plog=-14.1431 t0=1040 t1=1056 word_start=0 "y"
  [37] id= 7883 frame=132 dur_idx= 4 dur_val= 4 p=0.9567 plog=-11.5198 t0=1056 t1=1088 word_start=0 "."
```
