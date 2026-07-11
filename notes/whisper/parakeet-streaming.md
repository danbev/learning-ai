### Parakeet Streaming Implementation
Just stashing some notes about Parakeet Stream Models in case we decide to
support them in the future.

### nemotron-3.5-asr-streaming-0.6b.nemo model
This model is described as a `Cache-Aware FastConformer-RNNT architecture`.
To understand what this means lets think of "offline", which is where we pass the
entire sample through the network. When we do streaming, we process small chunks
of audio at a time. If we try to process each chunk in isolation it will loose
context and the accuracy will collapse.

A Cache-Aware architecture fixes this. It means the model layers are explicitly
designed to accept, update, and pass forward historical state tensors (caches)
right alongside the fresh incoming audio chunk. And this is different from trying
to do this externally from the model, it looses accuracy at the borders. Because
the model was trained using this same structure, this chunking, it will not loose
accuracy in the same way.


```console
EncDecRNNTBPEModelWithPrompt(
  (preprocessor): AudioToMelSpectrogramPreprocessor(
    (featurizer): FilterbankFeatures()
  )
  (encoder): ConformerEncoder(
    (pre_encode): ConvSubsampling(
      (out): Linear(in_features=4352, out_features=1024, bias=True)
      (conv): MaskedConvSequential(
        (0): CausalConv2D(1, 256, kernel_size=(3, 3), stride=(2, 2))
        (1): ReLU(inplace=True)
        (2): CausalConv2D(256, 256, kernel_size=(3, 3), stride=(2, 2), groups=256)
        (3): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (4): ReLU(inplace=True)
        (5): CausalConv2D(256, 256, kernel_size=(3, 3), stride=(2, 2), groups=256)
        (6): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (7): ReLU(inplace=True)
      )
    )
    (pos_enc): RelPositionalEncoding(
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (layers): ModuleList(
      (0-23): 24 x ConformerLayer(
        (norm_feed_forward1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True, bias=True)
        (feed_forward1): ConformerFeedForward(
          (linear1): Linear(in_features=1024, out_features=4096, bias=False)
          (activation): Swish()
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=4096, out_features=1024, bias=False)
        )
        (norm_conv): LayerNorm((1024,), eps=1e-05, elementwise_affine=True, bias=True)
        (conv): ConformerConvolution(
          (pointwise_conv1): Conv1d(1024, 2048, kernel_size=(1,), stride=(1,), bias=False)
          (depthwise_conv): CausalConv1D(1024, 1024, kernel_size=(9,), stride=(1,), groups=1024, bias=False)
          (batch_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True, bias=True)
          (activation): Swish()
          (pointwise_conv2): Conv1d(1024, 1024, kernel_size=(1,), stride=(1,), bias=False)
        )
        (norm_self_att): LayerNorm((1024,), eps=1e-05, elementwise_affine=True, bias=True)
        (self_attn): RelPositionMultiHeadAttention(
          (linear_q): Linear(in_features=1024, out_features=1024, bias=False)
          (linear_k): Linear(in_features=1024, out_features=1024, bias=False)
          (linear_v): Linear(in_features=1024, out_features=1024, bias=False)
          (linear_out): Linear(in_features=1024, out_features=1024, bias=False)
          (dropout): Dropout(p=0.1, inplace=False)
          (linear_pos): Linear(in_features=1024, out_features=1024, bias=False)
        )
        (norm_feed_forward2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True, bias=True)
        (feed_forward2): ConformerFeedForward(
          (linear1): Linear(in_features=1024, out_features=4096, bias=False)
          (activation): Swish()
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=4096, out_features=1024, bias=False)
        )
        (dropout): Dropout(p=0.1, inplace=False)
        (norm_out): LayerNorm((1024,), eps=1e-05, elementwise_affine=True, bias=True)
      )
    )
  )
  (decoder): RNNTDecoder(
    (prediction): ModuleDict(
      (embed): Embedding(13088, 640, padding_idx=13087)
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
      (2): Linear(in_features=640, out_features=13088, bias=True)
    )
    (_loss): RNNTLoss(
      (_loss): RNNTLossNumba()
    )
    (_wer): WER()
  )
  (loss): RNNTLoss(
    (_loss): RNNTLossNumba()
  )
  (spec_augmentation): SpectrogramAugmentation(
    (spec_augment): SpecAugment()
  )
  (wer): WER()
  (prompt_kernel): Sequential(
    (0): Linear(in_features=1152, out_features=2048, bias=True)
    (1): ReLU()
    (2): Linear(in_features=2048, out_features=1024, bias=True)
  )
)
```
I've not see the `prompt_kernel` before and the shapes of the tensors look
like this:
```console
Processing: prompt_kernel.0.weight [2048, 1152], dtype: float32, n_dims: 2, reversed: [1152, 2048]
Processing: prompt_kernel.0.bias   [2048],       dtype: float32, n_dims: 1, reversed: [2048]
Processing: prompt_kernel.2.weight [1024, 2048], dtype: float32, n_dims: 2, reversed: [2048, 1024]
Processing: prompt_kernel.2.bias   [1024],       dtype: float32, n_dims: 1, reversed: [1024]
```
And if we look `model_config.yaml` we can see that we have a `num_prompts` and
a `prompts_dictionary`:
```console
sample_rate: 16000
compute_eval_loss: false
log_prediction: true
skip_nan_grad: true
model_defaults:
  enc_hidden: 1024
  pred_hidden: 640
  joint_hidden: 640
  initialize_prompt_feature: true
  num_prompts: 128
  norm: None
  prompt_dictionary:
    en-US: 0
    en: 0
    en-GB: 1
    enGB: 1
    es-ES: 2
    esES: 2
    es-US: 3
    es: 3
    zh-CN: 4
    ...
    mt-MT: 102
    auto: 101
```
So we have `num_prompts` which is 128, this is the number of possible supported
languaged. The `prompt_dictionary` is a mapping between language codes the index
into a one-hot vector. This one-hot vector is 128 dims and is concatented with
the encoders output, 1024 dims before it passed to the MPL (the kernel_prompt
tensors we saw above).
```console
encoder_output  [B x T x 1024]
one_hot         [B x T x  128]   < same vector repeated for every time step T
                --------------
concat          [B x T x 1152]   < this is what goes into prompt_kernel.0
```
```console
prompt_kernel.0.weight [2048, 1152]
```
So the encoder itself just produces acoustic features which are language agnostic,
and the MLP will is moving these points/vector into a region of the hidden embedding
space that the model has learned is for the language in question, the language
specified in the one-hot vector.

Example transciption:
```console
text="My fellow Americans this day has brought terrible news, and great sadness to our country at nine o'clock this morning mission control in Houston lost contact with our space shuttle Columbia a short time later, debris was seen falling from the skies above Texas. <en-US> The Columbia is lost there are no survivors on board was a crew of seven, Colonel Rick Husband, Lieutenant Colonel Michael Anderson, Commander Laurel Clark, Captain David Brown, Commander William McCool Drive Coolbna Shavla Annie Lan Ramon, a colonel in the Israeli Air Force. <en-US> These men and women assumed great risk in the service to all humanity in an age when spaceflight has come to seem almost routine, it is easy to overlook the dangers of travel by rocket, and the difficulties of navigating the fierce outer atmosphere of the Earth. <en-US> These astronauts knew the dangers, and they faced them willingly, knowing they had a high and noble purpose in life because of their courage and daring and idealism we will miss them all the more. <en-US> All Americans today are thinking as well of the families of these men and women who have been given this sudden shock in grief. <en-US> You're not alone. <en-US> Our entire nation griefs with you and those you loved will always have the respect and gratitude of this country. <en-US> The cause in which they died will continue mankind is led into the darkness beyond our world by the inspiration of discovery and the longing to understand our journey into space will go on in the skies today we saw destruction and tragedy it's farther than we can see there is comfort and hope in the words of the Prophet Isaiah Lift your eyes and look to the heavens who created all these He who brings out the starry hosts one by one and calls them each by name because of His great power and mighty strength, not one of them is missing. <en-US> The same Creator who names the stars also knows the names of the seven souls we mourn today. <en-US> The crew of the shuttle Columbia did not return safely to Earth, yet we can pray that all are safely home. <en-US> May God bless the grieving families and make God continue to bless America. <en-US>", dec_out=None, dec_state=None, timestamp=[], alignments=None, frame_confidence=None, token_confidence=None, word_confidence=None, length=0, y=None, lm_state=None, lm_scores=None, ngram_lm_state=None, tokens=None, last_token=None, token_duration=None, last_frame=None, biasing_cfg=None, non_blank_step_confidence_precomputed=None, xatt_scores=None)]
```

Transcription configuration options:
```console
use_lhotse: True  (type: <class 'bool'>)
batch_size: 4  (type: <class 'int'>)
return_hypotheses: False  (type: <class 'bool'>)
num_workers: None  (type: typing.Optional[int])
channel_selector: None  (type: typing.Union[int, typing.Iterable[int], str])
augmentor: None  (type: typing.Optional[omegaconf.dictconfig.DictConfig])
timestamps: None  (type: typing.Optional[bool])
verbose: True  (type: <class 'bool'>)
partial_hypothesis: None  (type: typing.Optional[typing.List[typing.Any]])
_internal: None  (type: typing.Optional[nemo.collections.asr.parts.mixins.transcription.InternalTranscribeConfig])
target_lang: 'auto'  (type: <class 'str'>)
```

### Encoder configuration
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
  causal_downsampling: true
  reduction: null
  reduction_position: null
  reduction_factor: 1
  ff_expansion_factor: 4
  self_attention_model: rel_pos
  n_heads: 8
  att_context_size:
  - - 56
    - 3
  - - 56
    - 0
  - - 56
    - 6
  - - 56
    - 13
  att_context_style: chunked_limited
  xscaling: false
  untie_biases: true
  pos_emb_max_len: 5000
  conv_kernel_size: 9
  conv_norm_type: layer_norm
  conv_context_size: causal
  dropout: 0.1
  dropout_pre_encoder: 0.1
  dropout_emb: 0.0
  dropout_att: 0.1
  stochastic_depth_drop_prob: 0.0
  stochastic_depth_mode: linear
  stochastic_depth_start_layer: 1
```

### att_context_size
So recall that this model was trained with chunked attention and this section
specifies the boundries of the chunks it used during training. These are pairs
of [left context, right context], they decribe the tokens ability to see into
the past and future tokens:
```console
  att_context_size:
  - - 56   Pair 1: left=56 frames (~4.5s in the past)
    - 3            right=3 frames (~0.25s in the future)

  - - 56   Pair 2: left=56 frames (~4.5s)
    - 0

  - - 56   Pair 3: left=56 frames (~4.5s)
    - 6

  - - 56   Pair 4: left=56 frames (~4.5s)
    - 13
```
So all have the same left context of 56 frames so that can all look backward into
the past by 56 frames (56 x 80ms = 4.48s), but the right context varies. And the
right is how much the model can look forward. So it was trained to look 3 frames
into the future, 0 so just a normal causal mask where the future is completely
masked out. This is all related to the attention layer.

And notice that the `att_context_style: chunked_limited` which means it was
it was trained using those attention sizes.

Now, there are 4 different pairs or attention context here which is why this is
called multi-lookahead, it was not just trained with one fixed attention window
but multiple. And I think this can be configured/specified by choosing a specific
pair from the above list.

Also note that the `conv_context_size: causal` which means that the
And that this use `layer_norm` and not `batch_norm`.


### causal_downsampling
For this model `causal_downsampling: true` which is different from the TDT model
where this is false. This configuration options is related to the pre-processing
stage where the downsampling happens. Both models down sample by a factor of 8
which is done by 3 convolutions if I recall correctly.

For offline mode (which is what the TDT model does) when calculating the
downsampled feature for time t, the convolution kernel centers itself on t and
looks both backward into the past (t - k) and forward into the future (t + k).
This is perfectly fine for offline transcription because the entire audio file
is loaded into memory at once. The model is free to look ahead.

But in a live streaming environment, looking ahead at the subsampling stage is
impossible because those future audio frames haven't been recorded yet.
Setting this to true forces the subsampling convolutions to use asymmetric
(left-only) padding.
So we need to restrict convolution kernels so that when they evaluate time t,
they are strictly restricted to looking at the current frame and past frames.
The future side of the kernel is completely blinded.

```console
(Pdb) p model.encoder.streaming_cfg
CacheAwareStreamingConfig(
chunk_size=[25, 32],
shift_size=[25, 32],
cache_drop_size=0,
last_channel_cache_size=56,
valid_out_len=4,
pre_encode_cache_size=[0, 9],
drop_extra_pre_encoded=2,
last_channel_num=0,
last_time_num=0)
```
Notice that we have `[0, 9]` as the value for `pre_encode_cache_size`, which means
that 0 is used for the first chunk and 9 for the following chunks.


### prompt
The NeMo model has internal classes named prompt, and there is also tensors name
like `prompt_kernel.0.weight` which I found somewhat confusing. I think of a
prompt as a text prompt to an LLM. In this case it is a language prompt. In audio
the features for someone saying a phrase in English has a simlar acustic
characteristic to someone speaking Swedish. We need to tell the decoding head
what language we are expecting. This is done using a one-hot vector which has
a 1 in the position of the language we are expecting. We can think of this as
a text prompt, "Respond only in Swedish". So in this way it is like a prompt, 
hence the name.


### Subsampling (pre_encode)
So in an offline model we have a subsampling and the streaming model will be 
similar, that is it will have the same 3 convolutions but with some differences.

If we consider the input to this layer it is the log mel-spectrogram where we
have 128 mel bins and 1101 time steps.
```console
(gdb) p mel->ne
$2 = {128, 1101, 1, 1}

     mel bins →
0    [0 1 2 3 4 5 6 ...  127]  time
1    [0 1 2 3 4 5 6 ...  127]   ↓
2    [0 1 2 3 4 5 6 ...  127]
3    [0 1 2 3 4 5 6 ...  127]
       ...
1100 [0 1 2 3 4 5 6 ...  127]

(gdb) p model.enc_pre_conv_0_w->ne
$3 = {3, 3, 1, 256}
```
So we have a kernel of size 3x3.
And this is what the first convolution looks like:
```c++
    struct ggml_tensor * cur = ggml_conv_2d(ctx0, model.enc_pre_conv_0_w, mel, 2, 2, 1, 1, 1, 1);
```
This has a x stride of 2 and a y stride of 2 as well. And we have one frame of
padding on the left and the right.
So the first convolution will start at time 0 and look at the first 3 frames,
and the first 3 mel bins like this.

If we did not have any padding (0) then our starting point would be like this:
```console
     mel bins →
0    [x x x 3 4 5 6 ...  127]  time
1    [x x x 3 4 5 6 ...  127]   ↓
2    [x x x 3 4 5 6 ...  127]
3    [0 1 2 3 4 5 6 ...  127]
       ...
1100 [0 1 2 3 4 5 6 ...  127]
```
But with a padding of 1 we get:
```console

     -1    0    1    2    3    4  ...  125  126  127  128
-1  [ 0    0    0    0    0    0  ...    0    0    0    0 ]  <-- Virtual Row -1
 0  [ 0    x    x    x    x    x  ...    x    x    x    0 ]  <-- Time 0
 1  [ 0    x    x    x    x    x  ...    x    x    x    0 ]  <-- Time 1
 2  [ 0    x    x    x    x    x  ...    x    x    x    0 ]  <-- Time 2
    ...
1100[ 0    x    x    x    x    x  ...    x    x    x    0 ]  <-- Time 1100 (Last actual frame)
1101[ 0    0    0    0    0    0  ...    0    0    0    0 ]  <-- Virtual Row 1101
```

```console
               mel bins →
      -1   0   1   2   3   4   5 ... 127 128
 -1  [ x   x   x ] 0   0   0   0              ← Virtual padding row (zeros)
  0  [ x   x   x ] 2   3   4   5 ... 127   0  ← time 0
  1  [ x   x   x ] 2   3   4   5 ... 127   0  ← time 1
  2    0   1   2   3   4   5   6 ... 127   0  ← time 2
  3    0   1   2   3   4   5   6 ... 127   0  ← time 3
       ...
1100   0   1   2   3   4   5   6 ... 127   0
1101   0   0   0   0   0   0   0 ... 127   0
       ↑
 Virtual padding column (zeros)
```
For the next jump, since we have a stride of 2 we jump to mel bins while still
on the same time steps (-1, 0, 1):
```console
               mel bins →
      -1   0   1   2   3   4   5 ... 127 128
 -1    0   0 [ x   x   x ]                
  0    0   x [ x   x   x ] 4   5 ... 127   0 ← time 0
  1    0   x [ x   x   x ] 4   5 ... 127   0 ← time 1
  2    0   1   2   3   4   5   6 ... 127   0 ← time 2
  3    0   1   2   3   4   5   6 ... 127   0 ← time 3
       ...
1100   0   1   2   3   4   5   6 ... 127   0
1101   0   0   0   0   0   0   0 ...   0   0
```
Notice that we get an overlapping of the previous mel bin and the next mel bin
for each convolution operation, as well as an overlapp in time.
And when we hit the end of the x-axis we will have processed 3 time frames and
all the mel bins for them. We then shift the kernel by 2 in the y direction and
continue. This again allows for an overlap in the time dimension as well.

Now, this works fine for offline mode as we can add padding to the start, a column
of zero frequencies, and a row of zero time at the top as well. Lets say we have
chunked a larger audio file, then this above approach would work for the first
chunk, but for the next chunk, the padding should not be zeros, well it should
not be there at all. What this chunk really needs is the last 2 time steps
from the previous chunk.

When we cut a long audio stream into isolated blocks (let’s say 80ms chunks,
which give us 8 log-mel frames at a time), a standard offline Conv2d module
encounters a crisis at both ends of the chunk.

When Chunk 1 (frames 0–7) finishes, Chunk 2 (frames 8–15) arrives.
This is what would happend if we did this:
```console
         mel bins →
   7   [ 0   0   0 ... 0 ]  ← Standard padding inserts ZEROS here for Chunk 2!
-------------------------
   8   [ x   x   x ... x ]  ← Frame 8 (Actual start of Chunk 2)
   9   [ x   x   x ... x ]  ← Frame 9
```
We would be telling the neural network that the speaker was completely silent a
split second ago might not have been the case at all.

And we also have an problem with the end. When the offline model reaches the
absolute end of the file, it applies its standard symmetric padding. It creates
a virtual frame filled with zeros symmetrically on both sides like we saw above.
This is perfectly fine for offline because the speaker has actually stopped
talking. There is no future audio missing, silence is the ground truth. But for
streaming this is wrong.
What we have to do is to change the the centering of the kernel so that instead
of looking at [t-1, 0, t+1] we would look at [t-2, t-1, t]. SO this never looks
to the right so this padding is not an issue now. So we can't use this type of
symmatric padding. We actually can't specify any padding at all for these type
of model. What we have to do is to add the padding manually to the tensor before
the we call the convolution operation. So we handle the padding manually.
So this is what we want to achive (and recall this is the mel tensor that we
pass into the convolution operation, after applying our manual padding):
```console
                    mel bins →
      -1    0    1    2    3    4  ...  127  128
 -2  [ 0    0    0    0    0    0  ...    0    0 ]  <-- Virtual Row -2 (Past)
 -1  [ 0    0    0    0    0    0  ...    0    0 ]  <-- Virtual Row -1 (Past)
-------------------------------------------------
  0  [ 0    x    x    x    x    x  ...    x    0 ]  <-- Time 0 (First actual frame)
  1  [ 0    x    x    x    x    x  ...    x    0 ]  <-- Time 1
  2  [ 0    x    x    x    x    x  ...    x    0 ]  <-- Time 2
    ...
1100 [ 0    x    x    x    x    x  ...    x    0 ]  <-- Time 1100 (Last frame)
                                                    <-- NO BOTTOM PADDING ROW HERE!
```
So we still have the padding for the mel bins which it not an issue.
```c++
struct ggml_tensor * padded_mel = ggml_pad_ext(ctx0, mel,
    1, 1, // X-axis padding (Left, Right)
    2, 0, // Y-axis padding (Top/Past, Bottom/Future)
    0, 0
);

struct ggml_tensor * cur = ggml_conv_2d(ctx0, model.enc_pre_conv_0_w, padded_mel,
    2, 2, // Strides (s0, s1)
    0, 0, // Padding parameters set to 0 because we manual padded!
    1, 1  // Dilation
);
```

### Python walkthrough
```console
$ source venv/bin/activate
(venv) $ python -m pdb test-model.py

(Pdb) b ../NeMo/nemo/collections/asr/models/rnnt_bpe_models_prompt.py:324
```
```python
    @typecheck()
    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
        prompt_indices=None,
    ):
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) is False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal,
                length=input_signal_length,
            )
```
So in this case has_processed_signal is False so this block will be executed.
This is what is going to proprocess the audio into log mel-spectrograms.


_wip_
