### Parakeet Streaming Implementation
Just stashing some notes about Parakeet Stream Models in case we decide to
support them in the future.

### nemotron-3.5-asr-streaming-0.6b.nemo model
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
    zh-ZH: 4
    zh-TW: 5
    hi-IN: 6
    hi: 6
    hi-HI: 6
    ar-AR: 7
    ar: 7
    fr-FR: 8
    fr: 8
    de-DE: 9
    de: 9
    ja-JP: 10
    ja-JA: 10
    ru-RU: 11
    ru: 11
    pt-BR: 12
    pt-PT: 13
    pt: 13
    ko-KR: 14
    ko: 14
    ko-KO: 14
    it-IT: 15
    it: 15
    nl-NL: 16
    nl: 16
    pl-PL: 17
    pl: 17
    tr-TR: 18
    tr: 18
    uk-UA: 19
    uk: 19
    ro-RO: 20
    ro: 20
    el-GR: 21
    el: 21
    cs-CZ: 22
    cs: 22
    hu-HU: 23
    hu: 23
    sv-SE: 24
    sv: 24
    da-DK: 25
    da: 25
    fi-FI: 26
    fi: 26
    no-NO: 27
    'no': 27
    nb-NO: 103
    nb: 103
    nn-NO: 104
    nn: 104
    sk-SK: 28
    sk: 28
    hr-HR: 29
    hr: 29
    bg-BG: 30
    bg: 30
    lt-LT: 31
    lt: 31
    et-EE: 60
    et: 60
    lv-LV: 61
    lv: 61
    sl-SI: 62
    sl: 62
    th-TH: 32
    vi-VN: 33
    id-ID: 34
    ms-MY: 35
    bn-IN: 36
    ur-PK: 37
    fa-IR: 38
    ta-IN: 39
    te-IN: 40
    mr-IN: 41
    gu-IN: 42
    kn-IN: 43
    ml-IN: 44
    si-LK: 45
    ne-NP: 46
    km-KH: 47
    sw-KE: 48
    am-ET: 49
    ha-NG: 50
    zu-ZA: 51
    yo-NG: 52
    ig-NG: 53
    af-ZA: 54
    rw-RW: 55
    so-SO: 56
    ny-MW: 57
    ln-CD: 58
    or-KE: 59
    he-IL: 64
    ku-TR: 65
    az-AZ: 66
    ka-GE: 67
    hy-AM: 68
    uz-UZ: 69
    tg-TJ: 70
    ky-KG: 71
    qu-PE: 80
    ay-BO: 81
    gn-PY: 82
    nah-MX: 83
    mi-NZ: 96
    haw-US: 97
    sm-WS: 98
    to-TO: 99
    fr-CA: 100
    mt-MT: 102
    auto: 101
...
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

_wip_
