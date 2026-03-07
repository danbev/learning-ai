### Parakeet TDT 0.6B V3 support
This documentent contains on the Parakeet model with the goal being to convert
it into a format that can be used with Whisper.cpp. The goal is to identify major
differences which might effect the work.

### Overview
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

So this will return us to conformer_encoder.py and its forward_internal method::
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
```console
(Pdb) p pos_emb.shape
torch.Size([1, 275, 1024])
```
So this does not add potional encodings to the input but rather creates a separate
tensor with a table of positional encodings.

_wip_

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
