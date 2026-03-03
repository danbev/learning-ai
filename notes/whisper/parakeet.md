### Parakeet TDT 0.6B V3 support
This documentent contains on the Parakeet model with the goal being to convert
it into a format that can be used with Whisper.cpp. The goal is to identify major
differences which might effect the work.

### Overview
The Parakeet model uses a Conformer based encoder named
[Fast Conformer](https://arxiv.org/pdf/2305.05084) and a TDT (Token-and-Duration
Transducer) decoder.

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
```

### preprosssor (mel spectrogram transformation)
In Parakeet they have a layer called `pre_encode` which is equivalent to
`whisper_build_graph_conv` but the operations differ and this is what this
section is focused on.

If we set a break point in:
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
Now, this is very similar if not the same as what is done in whisper, apart from
the number of mel bins is 128 instead of 80.

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
uses a stride of 160:
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
frame and frequency bin. So we will go from 1101 -> 551 frames, and from 128 ->
64 mel bins. And the (1, 256) is the number of input and output channels for the
convolution. So one input channel will be projected to 256 different feature
maps.
```console
(Pdb) p x.shape
torch.Size([1, 256, 551, 64])
```
The second layer will a non-linear layer which does not change the shape:
```console
(Pdb) p layer
ReLU(inplace=True)

(Pdb) p x.shape
torch.Size([1, 256, 551, 64])
```
The next layer is:
```console
(Pdb) p layer
Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)

(Pdb) p x.shape
torch.Size([1, 256, 276, 32])
```
Next we have:
```console
(Pdb) p layer
Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))

(Pdb) p x.shape
torch.Size([1, 256, 276, 32])
```
Then we have another non-linear layer:
```console
(Pdb) p layer
ReLU(inplace=True)

(Pdb) p x.shape
torch.Size([1, 256, 276, 32])
```
Following that we have:
```console
(Pdb) p layer
Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)

(Pdb) p x.shape
torch.Size([1, 256, 138, 16])
```
```console
(Pdb) p layer
Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))

(Pdb) p x.shape
torch.Size([1, 256, 138, 16])
```
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
_wip_

(Pdb) b ~/work/ai/whisper-models/nvidia/parkeet-tdt-0.6b-v3/venv/lib/python3.12/site-packages/nemo/collections/asr/parts/submodules/subsampling.py:89
(Pdb) c
```
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
```

```console
(Pdb) p self.pre_encode
ConvSubsampling(
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
```
In whisper.cpp we have:
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
So for the Parakeet model we would need to enable it to handle this in a was
specific to that model.

So the tensor that are involved in this step are:

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
```
~/work/whisper-models/parkeet-tdt-0.6b-v3/venv/lib/python3.12/site-packages/nemo/collections/asr/parts/submodules/subsampling.py
```

```console
(Pdb) b /home/danbev/work/ai/whisper-models/nvidia/parkeet-tdt-0.6b-v3/venv/lib/python3.12/site-packages/nemo/collections/asr/modules/conformer_encoder.py:615
(Pdb) c

(Pdb) p self.pre_encode
ConvSubsampling(
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

(Pdb) l
627  	        if not bypass_pre_encode:
628  ->	            audio_signal = torch.transpose(audio_signal, 1, 2)
629
630  	            if isinstance(self.pre_encode, nn.Linear):
631  	                audio_signal = self.pre_encode(audio_signal)
632  	            else:
633  	                audio_signal, length = self.pre_encode(x=audio_signal, lengths=length)

(Pdb) p audio_signal.shape
torch.Size([1, 128, 1101])

(Pdb) p audio_signal.shape
torch.Size([1, 1101, 128])
```

```console 
(Pdb) b /home/danbev/work/ai/whisper-models/nvidia/parkeet-tdt-0.6b-v3/venv/lib/python3.12/site-packages/nemo/collections/asr/parts/submodules/subsampling.py:385
```
```console
(Pdb) l
380  	        return [1, self.subsampling_factor]
381  	
382  	    def get_streaming_cache_size(self):
383  	        return [0, self.subsampling_factor + 1]
384  	
385  ->	    def forward(self, x, lengths):
386  	        out_lengths = calc_length(
387  	            lengths,
388  	            all_paddings=self._left_padding + self._right_padding,
389  	            kernel_size=self._kernel_size,
390  	            stride=self._stride,

```
```python
                x, lengths = self.conv(x, lengths)
```
```console
-> x, lengths = self.conv(x, lengths)
(Pdb) p lengths
tensor([1100])
(Pdb) p x.shape
torch.Size([1, 1101, 128])

(Pdb) p self.conv
MaskedConvSequential(
  (0): Conv2d(1, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (1): ReLU(inplace=True)
  (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
  (3): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
  (4): ReLU(inplace=True)
  (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
  (6): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
  (7): ReLU(inplace=True)
)
```


```console
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
