### Parakeet TDT 0.6B V3 support
This documentent contains on the Parakeet model with the goal being to convert
it into a format that can be used with Whisper.cpp. The goal is to identify major
differences which might effect the work.

### Overview
The Parakeet model uses a Conformer based encoder named
[Fast Conformer](https://arxiv.org/pdf/2305.05084) and a TDT (Token-and-Duration
Transducer) decoder.

### Model conversion
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
Processing: encoder.layers.1.norm_feed_forward1.weight [1024]
Processing: encoder.layers.1.norm_feed_forward1.bias [1024]
Processing: encoder.layers.1.feed_forward1.linear1.weight [4096, 1024]
Processing: encoder.layers.1.feed_forward1.linear2.weight [1024, 4096]
Processing: encoder.layers.1.norm_conv.weight [1024]
Processing: encoder.layers.1.norm_conv.bias [1024]
Processing: encoder.layers.1.conv.pointwise_conv1.weight [2048, 1024]
Processing: encoder.layers.1.conv.depthwise_conv.weight [1024, 9]
Processing: encoder.layers.1.conv.batch_norm.weight [1024]
Processing: encoder.layers.1.conv.batch_norm.bias [1024]
Processing: encoder.layers.1.conv.batch_norm.running_mean [1024]
Processing: encoder.layers.1.conv.batch_norm.running_var [1024]
Processing: encoder.layers.1.conv.batch_norm.num_batches_tracked []
Processing: encoder.layers.1.conv.pointwise_conv2.weight [1024, 1024]
Processing: encoder.layers.1.norm_self_att.weight [1024]
Processing: encoder.layers.1.norm_self_att.bias [1024]
Processing: encoder.layers.1.self_attn.pos_bias_u [8, 128]
Processing: encoder.layers.1.self_attn.pos_bias_v [8, 128]
Processing: encoder.layers.1.self_attn.linear_q.weight [1024, 1024]
Processing: encoder.layers.1.self_attn.linear_k.weight [1024, 1024]
Processing: encoder.layers.1.self_attn.linear_v.weight [1024, 1024]
Processing: encoder.layers.1.self_attn.linear_out.weight [1024, 1024]
Processing: encoder.layers.1.self_attn.linear_pos.weight [1024, 1024]
Processing: encoder.layers.1.norm_feed_forward2.weight [1024]
Processing: encoder.layers.1.norm_feed_forward2.bias [1024]
Processing: encoder.layers.1.feed_forward2.linear1.weight [4096, 1024]
Processing: encoder.layers.1.feed_forward2.linear2.weight [1024, 4096]
Processing: encoder.layers.1.norm_out.weight [1024]
Processing: encoder.layers.1.norm_out.bias [1024]
Processing: encoder.layers.2.norm_feed_forward1.weight [1024]
Processing: encoder.layers.2.norm_feed_forward1.bias [1024]
Processing: encoder.layers.2.feed_forward1.linear1.weight [4096, 1024]
Processing: encoder.layers.2.feed_forward1.linear2.weight [1024, 4096]
Processing: encoder.layers.2.norm_conv.weight [1024]
Processing: encoder.layers.2.norm_conv.bias [1024]
Processing: encoder.layers.2.conv.pointwise_conv1.weight [2048, 1024]
Processing: encoder.layers.2.conv.depthwise_conv.weight [1024, 9]
Processing: encoder.layers.2.conv.batch_norm.weight [1024]
Processing: encoder.layers.2.conv.batch_norm.bias [1024]
Processing: encoder.layers.2.conv.batch_norm.running_mean [1024]
Processing: encoder.layers.2.conv.batch_norm.running_var [1024]
Processing: encoder.layers.2.conv.batch_norm.num_batches_tracked []
Processing: encoder.layers.2.conv.pointwise_conv2.weight [1024, 1024]
Processing: encoder.layers.2.norm_self_att.weight [1024]
Processing: encoder.layers.2.norm_self_att.bias [1024]
Processing: encoder.layers.2.self_attn.pos_bias_u [8, 128]
Processing: encoder.layers.2.self_attn.pos_bias_v [8, 128]
Processing: encoder.layers.2.self_attn.linear_q.weight [1024, 1024]
Processing: encoder.layers.2.self_attn.linear_k.weight [1024, 1024]
Processing: encoder.layers.2.self_attn.linear_v.weight [1024, 1024]
Processing: encoder.layers.2.self_attn.linear_out.weight [1024, 1024]
Processing: encoder.layers.2.self_attn.linear_pos.weight [1024, 1024]
Processing: encoder.layers.2.norm_feed_forward2.weight [1024]
Processing: encoder.layers.2.norm_feed_forward2.bias [1024]
Processing: encoder.layers.2.feed_forward2.linear1.weight [4096, 1024]
Processing: encoder.layers.2.feed_forward2.linear2.weight [1024, 4096]
Processing: encoder.layers.2.norm_out.weight [1024]
Processing: encoder.layers.2.norm_out.bias [1024]
Processing: encoder.layers.3.norm_feed_forward1.weight [1024]
Processing: encoder.layers.3.norm_feed_forward1.bias [1024]
Processing: encoder.layers.3.feed_forward1.linear1.weight [4096, 1024]
Processing: encoder.layers.3.feed_forward1.linear2.weight [1024, 4096]
Processing: encoder.layers.3.norm_conv.weight [1024]
Processing: encoder.layers.3.norm_conv.bias [1024]
Processing: encoder.layers.3.conv.pointwise_conv1.weight [2048, 1024]
Processing: encoder.layers.3.conv.depthwise_conv.weight [1024, 9]
Processing: encoder.layers.3.conv.batch_norm.weight [1024]
Processing: encoder.layers.3.conv.batch_norm.bias [1024]
Processing: encoder.layers.3.conv.batch_norm.running_mean [1024]
Processing: encoder.layers.3.conv.batch_norm.running_var [1024]
Processing: encoder.layers.3.conv.batch_norm.num_batches_tracked []
Processing: encoder.layers.3.conv.pointwise_conv2.weight [1024, 1024]
Processing: encoder.layers.3.norm_self_att.weight [1024]
Processing: encoder.layers.3.norm_self_att.bias [1024]
Processing: encoder.layers.3.self_attn.pos_bias_u [8, 128]
Processing: encoder.layers.3.self_attn.pos_bias_v [8, 128]
Processing: encoder.layers.3.self_attn.linear_q.weight [1024, 1024]
Processing: encoder.layers.3.self_attn.linear_k.weight [1024, 1024]
Processing: encoder.layers.3.self_attn.linear_v.weight [1024, 1024]
Processing: encoder.layers.3.self_attn.linear_out.weight [1024, 1024]
Processing: encoder.layers.3.self_attn.linear_pos.weight [1024, 1024]
Processing: encoder.layers.3.norm_feed_forward2.weight [1024]
Processing: encoder.layers.3.norm_feed_forward2.bias [1024]
Processing: encoder.layers.3.feed_forward2.linear1.weight [4096, 1024]
Processing: encoder.layers.3.feed_forward2.linear2.weight [1024, 4096]
Processing: encoder.layers.3.norm_out.weight [1024]
Processing: encoder.layers.3.norm_out.bias [1024]
Processing: encoder.layers.4.norm_feed_forward1.weight [1024]
Processing: encoder.layers.4.norm_feed_forward1.bias [1024]
Processing: encoder.layers.4.feed_forward1.linear1.weight [4096, 1024]
Processing: encoder.layers.4.feed_forward1.linear2.weight [1024, 4096]
Processing: encoder.layers.4.norm_conv.weight [1024]
Processing: encoder.layers.4.norm_conv.bias [1024]
Processing: encoder.layers.4.conv.pointwise_conv1.weight [2048, 1024]
Processing: encoder.layers.4.conv.depthwise_conv.weight [1024, 9]
Processing: encoder.layers.4.conv.batch_norm.weight [1024]
Processing: encoder.layers.4.conv.batch_norm.bias [1024]
Processing: encoder.layers.4.conv.batch_norm.running_mean [1024]
Processing: encoder.layers.4.conv.batch_norm.running_var [1024]
Processing: encoder.layers.4.conv.batch_norm.num_batches_tracked []
Processing: encoder.layers.4.conv.pointwise_conv2.weight [1024, 1024]
Processing: encoder.layers.4.norm_self_att.weight [1024]
Processing: encoder.layers.4.norm_self_att.bias [1024]
Processing: encoder.layers.4.self_attn.pos_bias_u [8, 128]
Processing: encoder.layers.4.self_attn.pos_bias_v [8, 128]
Processing: encoder.layers.4.self_attn.linear_q.weight [1024, 1024]
Processing: encoder.layers.4.self_attn.linear_k.weight [1024, 1024]
Processing: encoder.layers.4.self_attn.linear_v.weight [1024, 1024]
Processing: encoder.layers.4.self_attn.linear_out.weight [1024, 1024]
Processing: encoder.layers.4.self_attn.linear_pos.weight [1024, 1024]
Processing: encoder.layers.4.norm_feed_forward2.weight [1024]
Processing: encoder.layers.4.norm_feed_forward2.bias [1024]
Processing: encoder.layers.4.feed_forward2.linear1.weight [4096, 1024]
Processing: encoder.layers.4.feed_forward2.linear2.weight [1024, 4096]
Processing: encoder.layers.4.norm_out.weight [1024]
Processing: encoder.layers.4.norm_out.bias [1024]
Processing: encoder.layers.5.norm_feed_forward1.weight [1024]
Processing: encoder.layers.5.norm_feed_forward1.bias [1024]
Processing: encoder.layers.5.feed_forward1.linear1.weight [4096, 1024]
Processing: encoder.layers.5.feed_forward1.linear2.weight [1024, 4096]
Processing: encoder.layers.5.norm_conv.weight [1024]
Processing: encoder.layers.5.norm_conv.bias [1024]
Processing: encoder.layers.5.conv.pointwise_conv1.weight [2048, 1024]
Processing: encoder.layers.5.conv.depthwise_conv.weight [1024, 9]
Processing: encoder.layers.5.conv.batch_norm.weight [1024]
Processing: encoder.layers.5.conv.batch_norm.bias [1024]
Processing: encoder.layers.5.conv.batch_norm.running_mean [1024]
Processing: encoder.layers.5.conv.batch_norm.running_var [1024]
Processing: encoder.layers.5.conv.batch_norm.num_batches_tracked []
Processing: encoder.layers.5.conv.pointwise_conv2.weight [1024, 1024]
Processing: encoder.layers.5.norm_self_att.weight [1024]
Processing: encoder.layers.5.norm_self_att.bias [1024]
Processing: encoder.layers.5.self_attn.pos_bias_u [8, 128]
Processing: encoder.layers.5.self_attn.pos_bias_v [8, 128]
Processing: encoder.layers.5.self_attn.linear_q.weight [1024, 1024]
Processing: encoder.layers.5.self_attn.linear_k.weight [1024, 1024]
Processing: encoder.layers.5.self_attn.linear_v.weight [1024, 1024]
Processing: encoder.layers.5.self_attn.linear_out.weight [1024, 1024]
Processing: encoder.layers.5.self_attn.linear_pos.weight [1024, 1024]
Processing: encoder.layers.5.norm_feed_forward2.weight [1024]
Processing: encoder.layers.5.norm_feed_forward2.bias [1024]
Processing: encoder.layers.5.feed_forward2.linear1.weight [4096, 1024]
Processing: encoder.layers.5.feed_forward2.linear2.weight [1024, 4096]
Processing: encoder.layers.5.norm_out.weight [1024]
Processing: encoder.layers.5.norm_out.bias [1024]
Processing: encoder.layers.6.norm_feed_forward1.weight [1024]
Processing: encoder.layers.6.norm_feed_forward1.bias [1024]
Processing: encoder.layers.6.feed_forward1.linear1.weight [4096, 1024]
Processing: encoder.layers.6.feed_forward1.linear2.weight [1024, 4096]
Processing: encoder.layers.6.norm_conv.weight [1024]
Processing: encoder.layers.6.norm_conv.bias [1024]
Processing: encoder.layers.6.conv.pointwise_conv1.weight [2048, 1024]
Processing: encoder.layers.6.conv.depthwise_conv.weight [1024, 9]
Processing: encoder.layers.6.conv.batch_norm.weight [1024]
Processing: encoder.layers.6.conv.batch_norm.bias [1024]
Processing: encoder.layers.6.conv.batch_norm.running_mean [1024]
Processing: encoder.layers.6.conv.batch_norm.running_var [1024]
Processing: encoder.layers.6.conv.batch_norm.num_batches_tracked []
Processing: encoder.layers.6.conv.pointwise_conv2.weight [1024, 1024]
Processing: encoder.layers.6.norm_self_att.weight [1024]
Processing: encoder.layers.6.norm_self_att.bias [1024]
Processing: encoder.layers.6.self_attn.pos_bias_u [8, 128]
Processing: encoder.layers.6.self_attn.pos_bias_v [8, 128]
Processing: encoder.layers.6.self_attn.linear_q.weight [1024, 1024]
Processing: encoder.layers.6.self_attn.linear_k.weight [1024, 1024]
Processing: encoder.layers.6.self_attn.linear_v.weight [1024, 1024]
Processing: encoder.layers.6.self_attn.linear_out.weight [1024, 1024]
Processing: encoder.layers.6.self_attn.linear_pos.weight [1024, 1024]
Processing: encoder.layers.6.norm_feed_forward2.weight [1024]
Processing: encoder.layers.6.norm_feed_forward2.bias [1024]
Processing: encoder.layers.6.feed_forward2.linear1.weight [4096, 1024]
Processing: encoder.layers.6.feed_forward2.linear2.weight [1024, 4096]
Processing: encoder.layers.6.norm_out.weight [1024]
Processing: encoder.layers.6.norm_out.bias [1024]
Processing: encoder.layers.7.norm_feed_forward1.weight [1024]
Processing: encoder.layers.7.norm_feed_forward1.bias [1024]
Processing: encoder.layers.7.feed_forward1.linear1.weight [4096, 1024]
Processing: encoder.layers.7.feed_forward1.linear2.weight [1024, 4096]
Processing: encoder.layers.7.norm_conv.weight [1024]
Processing: encoder.layers.7.norm_conv.bias [1024]
Processing: encoder.layers.7.conv.pointwise_conv1.weight [2048, 1024]
Processing: encoder.layers.7.conv.depthwise_conv.weight [1024, 9]
Processing: encoder.layers.7.conv.batch_norm.weight [1024]
Processing: encoder.layers.7.conv.batch_norm.bias [1024]
Processing: encoder.layers.7.conv.batch_norm.running_mean [1024]
Processing: encoder.layers.7.conv.batch_norm.running_var [1024]
Processing: encoder.layers.7.conv.batch_norm.num_batches_tracked []
Processing: encoder.layers.7.conv.pointwise_conv2.weight [1024, 1024]
Processing: encoder.layers.7.norm_self_att.weight [1024]
Processing: encoder.layers.7.norm_self_att.bias [1024]
Processing: encoder.layers.7.self_attn.pos_bias_u [8, 128]
Processing: encoder.layers.7.self_attn.pos_bias_v [8, 128]
Processing: encoder.layers.7.self_attn.linear_q.weight [1024, 1024]
Processing: encoder.layers.7.self_attn.linear_k.weight [1024, 1024]
Processing: encoder.layers.7.self_attn.linear_v.weight [1024, 1024]
Processing: encoder.layers.7.self_attn.linear_out.weight [1024, 1024]
Processing: encoder.layers.7.self_attn.linear_pos.weight [1024, 1024]
Processing: encoder.layers.7.norm_feed_forward2.weight [1024]
Processing: encoder.layers.7.norm_feed_forward2.bias [1024]
Processing: encoder.layers.7.feed_forward2.linear1.weight [4096, 1024]
Processing: encoder.layers.7.feed_forward2.linear2.weight [1024, 4096]
Processing: encoder.layers.7.norm_out.weight [1024]
Processing: encoder.layers.7.norm_out.bias [1024]
Processing: encoder.layers.8.norm_feed_forward1.weight [1024]
Processing: encoder.layers.8.norm_feed_forward1.bias [1024]
Processing: encoder.layers.8.feed_forward1.linear1.weight [4096, 1024]
Processing: encoder.layers.8.feed_forward1.linear2.weight [1024, 4096]
Processing: encoder.layers.8.norm_conv.weight [1024]
Processing: encoder.layers.8.norm_conv.bias [1024]
Processing: encoder.layers.8.conv.pointwise_conv1.weight [2048, 1024]
Processing: encoder.layers.8.conv.depthwise_conv.weight [1024, 9]
Processing: encoder.layers.8.conv.batch_norm.weight [1024]
Processing: encoder.layers.8.conv.batch_norm.bias [1024]
Processing: encoder.layers.8.conv.batch_norm.running_mean [1024]
Processing: encoder.layers.8.conv.batch_norm.running_var [1024]
Processing: encoder.layers.8.conv.batch_norm.num_batches_tracked []
Processing: encoder.layers.8.conv.pointwise_conv2.weight [1024, 1024]
Processing: encoder.layers.8.norm_self_att.weight [1024]
Processing: encoder.layers.8.norm_self_att.bias [1024]
Processing: encoder.layers.8.self_attn.pos_bias_u [8, 128]
Processing: encoder.layers.8.self_attn.pos_bias_v [8, 128]
Processing: encoder.layers.8.self_attn.linear_q.weight [1024, 1024]
Processing: encoder.layers.8.self_attn.linear_k.weight [1024, 1024]
Processing: encoder.layers.8.self_attn.linear_v.weight [1024, 1024]
Processing: encoder.layers.8.self_attn.linear_out.weight [1024, 1024]
Processing: encoder.layers.8.self_attn.linear_pos.weight [1024, 1024]
Processing: encoder.layers.8.norm_feed_forward2.weight [1024]
Processing: encoder.layers.8.norm_feed_forward2.bias [1024]
Processing: encoder.layers.8.feed_forward2.linear1.weight [4096, 1024]
Processing: encoder.layers.8.feed_forward2.linear2.weight [1024, 4096]
Processing: encoder.layers.8.norm_out.weight [1024]
Processing: encoder.layers.8.norm_out.bias [1024]
Processing: encoder.layers.9.norm_feed_forward1.weight [1024]
Processing: encoder.layers.9.norm_feed_forward1.bias [1024]
Processing: encoder.layers.9.feed_forward1.linear1.weight [4096, 1024]
Processing: encoder.layers.9.feed_forward1.linear2.weight [1024, 4096]
Processing: encoder.layers.9.norm_conv.weight [1024]
Processing: encoder.layers.9.norm_conv.bias [1024]
Processing: encoder.layers.9.conv.pointwise_conv1.weight [2048, 1024]
Processing: encoder.layers.9.conv.depthwise_conv.weight [1024, 9]
Processing: encoder.layers.9.conv.batch_norm.weight [1024]
Processing: encoder.layers.9.conv.batch_norm.bias [1024]
Processing: encoder.layers.9.conv.batch_norm.running_mean [1024]
Processing: encoder.layers.9.conv.batch_norm.running_var [1024]
Processing: encoder.layers.9.conv.batch_norm.num_batches_tracked []
Processing: encoder.layers.9.conv.pointwise_conv2.weight [1024, 1024]
Processing: encoder.layers.9.norm_self_att.weight [1024]
Processing: encoder.layers.9.norm_self_att.bias [1024]
Processing: encoder.layers.9.self_attn.pos_bias_u [8, 128]
Processing: encoder.layers.9.self_attn.pos_bias_v [8, 128]
Processing: encoder.layers.9.self_attn.linear_q.weight [1024, 1024]
Processing: encoder.layers.9.self_attn.linear_k.weight [1024, 1024]
Processing: encoder.layers.9.self_attn.linear_v.weight [1024, 1024]
Processing: encoder.layers.9.self_attn.linear_out.weight [1024, 1024]
Processing: encoder.layers.9.self_attn.linear_pos.weight [1024, 1024]
Processing: encoder.layers.9.norm_feed_forward2.weight [1024]
Processing: encoder.layers.9.norm_feed_forward2.bias [1024]
Processing: encoder.layers.9.feed_forward2.linear1.weight [4096, 1024]
Processing: encoder.layers.9.feed_forward2.linear2.weight [1024, 4096]
Processing: encoder.layers.9.norm_out.weight [1024]
Processing: encoder.layers.9.norm_out.bias [1024]
Processing: encoder.layers.10.norm_feed_forward1.weight [1024]
Processing: encoder.layers.10.norm_feed_forward1.bias [1024]
Processing: encoder.layers.10.feed_forward1.linear1.weight [4096, 1024]
Processing: encoder.layers.10.feed_forward1.linear2.weight [1024, 4096]
Processing: encoder.layers.10.norm_conv.weight [1024]
Processing: encoder.layers.10.norm_conv.bias [1024]
Processing: encoder.layers.10.conv.pointwise_conv1.weight [2048, 1024]
Processing: encoder.layers.10.conv.depthwise_conv.weight [1024, 9]
Processing: encoder.layers.10.conv.batch_norm.weight [1024]
Processing: encoder.layers.10.conv.batch_norm.bias [1024]
Processing: encoder.layers.10.conv.batch_norm.running_mean [1024]
Processing: encoder.layers.10.conv.batch_norm.running_var [1024]
Processing: encoder.layers.10.conv.batch_norm.num_batches_tracked []
Processing: encoder.layers.10.conv.pointwise_conv2.weight [1024, 1024]
Processing: encoder.layers.10.norm_self_att.weight [1024]
Processing: encoder.layers.10.norm_self_att.bias [1024]
Processing: encoder.layers.10.self_attn.pos_bias_u [8, 128]
Processing: encoder.layers.10.self_attn.pos_bias_v [8, 128]
Processing: encoder.layers.10.self_attn.linear_q.weight [1024, 1024]
Processing: encoder.layers.10.self_attn.linear_k.weight [1024, 1024]
Processing: encoder.layers.10.self_attn.linear_v.weight [1024, 1024]
Processing: encoder.layers.10.self_attn.linear_out.weight [1024, 1024]
Processing: encoder.layers.10.self_attn.linear_pos.weight [1024, 1024]
Processing: encoder.layers.10.norm_feed_forward2.weight [1024]
Processing: encoder.layers.10.norm_feed_forward2.bias [1024]
Processing: encoder.layers.10.feed_forward2.linear1.weight [4096, 1024]
Processing: encoder.layers.10.feed_forward2.linear2.weight [1024, 4096]
Processing: encoder.layers.10.norm_out.weight [1024]
Processing: encoder.layers.10.norm_out.bias [1024]
Processing: encoder.layers.11.norm_feed_forward1.weight [1024]
Processing: encoder.layers.11.norm_feed_forward1.bias [1024]
Processing: encoder.layers.11.feed_forward1.linear1.weight [4096, 1024]
Processing: encoder.layers.11.feed_forward1.linear2.weight [1024, 4096]
Processing: encoder.layers.11.norm_conv.weight [1024]
Processing: encoder.layers.11.norm_conv.bias [1024]
Processing: encoder.layers.11.conv.pointwise_conv1.weight [2048, 1024]
Processing: encoder.layers.11.conv.depthwise_conv.weight [1024, 9]
Processing: encoder.layers.11.conv.batch_norm.weight [1024]
Processing: encoder.layers.11.conv.batch_norm.bias [1024]
Processing: encoder.layers.11.conv.batch_norm.running_mean [1024]
Processing: encoder.layers.11.conv.batch_norm.running_var [1024]
Processing: encoder.layers.11.conv.batch_norm.num_batches_tracked []
Processing: encoder.layers.11.conv.pointwise_conv2.weight [1024, 1024]
Processing: encoder.layers.11.norm_self_att.weight [1024]
Processing: encoder.layers.11.norm_self_att.bias [1024]
Processing: encoder.layers.11.self_attn.pos_bias_u [8, 128]
Processing: encoder.layers.11.self_attn.pos_bias_v [8, 128]
Processing: encoder.layers.11.self_attn.linear_q.weight [1024, 1024]
Processing: encoder.layers.11.self_attn.linear_k.weight [1024, 1024]
Processing: encoder.layers.11.self_attn.linear_v.weight [1024, 1024]
Processing: encoder.layers.11.self_attn.linear_out.weight [1024, 1024]
Processing: encoder.layers.11.self_attn.linear_pos.weight [1024, 1024]
Processing: encoder.layers.11.norm_feed_forward2.weight [1024]
Processing: encoder.layers.11.norm_feed_forward2.bias [1024]
Processing: encoder.layers.11.feed_forward2.linear1.weight [4096, 1024]
Processing: encoder.layers.11.feed_forward2.linear2.weight [1024, 4096]
Processing: encoder.layers.11.norm_out.weight [1024]
Processing: encoder.layers.11.norm_out.bias [1024]
Processing: encoder.layers.12.norm_feed_forward1.weight [1024]
Processing: encoder.layers.12.norm_feed_forward1.bias [1024]
Processing: encoder.layers.12.feed_forward1.linear1.weight [4096, 1024]
Processing: encoder.layers.12.feed_forward1.linear2.weight [1024, 4096]
Processing: encoder.layers.12.norm_conv.weight [1024]
Processing: encoder.layers.12.norm_conv.bias [1024]
Processing: encoder.layers.12.conv.pointwise_conv1.weight [2048, 1024]
Processing: encoder.layers.12.conv.depthwise_conv.weight [1024, 9]
Processing: encoder.layers.12.conv.batch_norm.weight [1024]
Processing: encoder.layers.12.conv.batch_norm.bias [1024]
Processing: encoder.layers.12.conv.batch_norm.running_mean [1024]
Processing: encoder.layers.12.conv.batch_norm.running_var [1024]
Processing: encoder.layers.12.conv.batch_norm.num_batches_tracked []
Processing: encoder.layers.12.conv.pointwise_conv2.weight [1024, 1024]
Processing: encoder.layers.12.norm_self_att.weight [1024]
Processing: encoder.layers.12.norm_self_att.bias [1024]
Processing: encoder.layers.12.self_attn.pos_bias_u [8, 128]
Processing: encoder.layers.12.self_attn.pos_bias_v [8, 128]
Processing: encoder.layers.12.self_attn.linear_q.weight [1024, 1024]
Processing: encoder.layers.12.self_attn.linear_k.weight [1024, 1024]
Processing: encoder.layers.12.self_attn.linear_v.weight [1024, 1024]
Processing: encoder.layers.12.self_attn.linear_out.weight [1024, 1024]
Processing: encoder.layers.12.self_attn.linear_pos.weight [1024, 1024]
Processing: encoder.layers.12.norm_feed_forward2.weight [1024]
Processing: encoder.layers.12.norm_feed_forward2.bias [1024]
Processing: encoder.layers.12.feed_forward2.linear1.weight [4096, 1024]
Processing: encoder.layers.12.feed_forward2.linear2.weight [1024, 4096]
Processing: encoder.layers.12.norm_out.weight [1024]
Processing: encoder.layers.12.norm_out.bias [1024]
Processing: encoder.layers.13.norm_feed_forward1.weight [1024]
Processing: encoder.layers.13.norm_feed_forward1.bias [1024]
Processing: encoder.layers.13.feed_forward1.linear1.weight [4096, 1024]
Processing: encoder.layers.13.feed_forward1.linear2.weight [1024, 4096]
Processing: encoder.layers.13.norm_conv.weight [1024]
Processing: encoder.layers.13.norm_conv.bias [1024]
Processing: encoder.layers.13.conv.pointwise_conv1.weight [2048, 1024]
Processing: encoder.layers.13.conv.depthwise_conv.weight [1024, 9]
Processing: encoder.layers.13.conv.batch_norm.weight [1024]
Processing: encoder.layers.13.conv.batch_norm.bias [1024]
Processing: encoder.layers.13.conv.batch_norm.running_mean [1024]
Processing: encoder.layers.13.conv.batch_norm.running_var [1024]
Processing: encoder.layers.13.conv.batch_norm.num_batches_tracked []
Processing: encoder.layers.13.conv.pointwise_conv2.weight [1024, 1024]
Processing: encoder.layers.13.norm_self_att.weight [1024]
Processing: encoder.layers.13.norm_self_att.bias [1024]
Processing: encoder.layers.13.self_attn.pos_bias_u [8, 128]
Processing: encoder.layers.13.self_attn.pos_bias_v [8, 128]
Processing: encoder.layers.13.self_attn.linear_q.weight [1024, 1024]
Processing: encoder.layers.13.self_attn.linear_k.weight [1024, 1024]
Processing: encoder.layers.13.self_attn.linear_v.weight [1024, 1024]
Processing: encoder.layers.13.self_attn.linear_out.weight [1024, 1024]
Processing: encoder.layers.13.self_attn.linear_pos.weight [1024, 1024]
Processing: encoder.layers.13.norm_feed_forward2.weight [1024]
Processing: encoder.layers.13.norm_feed_forward2.bias [1024]
Processing: encoder.layers.13.feed_forward2.linear1.weight [4096, 1024]
Processing: encoder.layers.13.feed_forward2.linear2.weight [1024, 4096]
Processing: encoder.layers.13.norm_out.weight [1024]
Processing: encoder.layers.13.norm_out.bias [1024]
Processing: encoder.layers.14.norm_feed_forward1.weight [1024]
Processing: encoder.layers.14.norm_feed_forward1.bias [1024]
Processing: encoder.layers.14.feed_forward1.linear1.weight [4096, 1024]
Processing: encoder.layers.14.feed_forward1.linear2.weight [1024, 4096]
Processing: encoder.layers.14.norm_conv.weight [1024]
Processing: encoder.layers.14.norm_conv.bias [1024]
Processing: encoder.layers.14.conv.pointwise_conv1.weight [2048, 1024]
Processing: encoder.layers.14.conv.depthwise_conv.weight [1024, 9]
Processing: encoder.layers.14.conv.batch_norm.weight [1024]
Processing: encoder.layers.14.conv.batch_norm.bias [1024]
Processing: encoder.layers.14.conv.batch_norm.running_mean [1024]
Processing: encoder.layers.14.conv.batch_norm.running_var [1024]
Processing: encoder.layers.14.conv.batch_norm.num_batches_tracked []
Processing: encoder.layers.14.conv.pointwise_conv2.weight [1024, 1024]
Processing: encoder.layers.14.norm_self_att.weight [1024]
Processing: encoder.layers.14.norm_self_att.bias [1024]
Processing: encoder.layers.14.self_attn.pos_bias_u [8, 128]
Processing: encoder.layers.14.self_attn.pos_bias_v [8, 128]
Processing: encoder.layers.14.self_attn.linear_q.weight [1024, 1024]
Processing: encoder.layers.14.self_attn.linear_k.weight [1024, 1024]
Processing: encoder.layers.14.self_attn.linear_v.weight [1024, 1024]
Processing: encoder.layers.14.self_attn.linear_out.weight [1024, 1024]
Processing: encoder.layers.14.self_attn.linear_pos.weight [1024, 1024]
Processing: encoder.layers.14.norm_feed_forward2.weight [1024]
Processing: encoder.layers.14.norm_feed_forward2.bias [1024]
Processing: encoder.layers.14.feed_forward2.linear1.weight [4096, 1024]
Processing: encoder.layers.14.feed_forward2.linear2.weight [1024, 4096]
Processing: encoder.layers.14.norm_out.weight [1024]
Processing: encoder.layers.14.norm_out.bias [1024]
Processing: encoder.layers.15.norm_feed_forward1.weight [1024]
Processing: encoder.layers.15.norm_feed_forward1.bias [1024]
Processing: encoder.layers.15.feed_forward1.linear1.weight [4096, 1024]
Processing: encoder.layers.15.feed_forward1.linear2.weight [1024, 4096]
Processing: encoder.layers.15.norm_conv.weight [1024]
Processing: encoder.layers.15.norm_conv.bias [1024]
Processing: encoder.layers.15.conv.pointwise_conv1.weight [2048, 1024]
Processing: encoder.layers.15.conv.depthwise_conv.weight [1024, 9]
Processing: encoder.layers.15.conv.batch_norm.weight [1024]
Processing: encoder.layers.15.conv.batch_norm.bias [1024]
Processing: encoder.layers.15.conv.batch_norm.running_mean [1024]
Processing: encoder.layers.15.conv.batch_norm.running_var [1024]
Processing: encoder.layers.15.conv.batch_norm.num_batches_tracked []
Processing: encoder.layers.15.conv.pointwise_conv2.weight [1024, 1024]
Processing: encoder.layers.15.norm_self_att.weight [1024]
Processing: encoder.layers.15.norm_self_att.bias [1024]
Processing: encoder.layers.15.self_attn.pos_bias_u [8, 128]
Processing: encoder.layers.15.self_attn.pos_bias_v [8, 128]
Processing: encoder.layers.15.self_attn.linear_q.weight [1024, 1024]
Processing: encoder.layers.15.self_attn.linear_k.weight [1024, 1024]
Processing: encoder.layers.15.self_attn.linear_v.weight [1024, 1024]
Processing: encoder.layers.15.self_attn.linear_out.weight [1024, 1024]
Processing: encoder.layers.15.self_attn.linear_pos.weight [1024, 1024]
Processing: encoder.layers.15.norm_feed_forward2.weight [1024]
Processing: encoder.layers.15.norm_feed_forward2.bias [1024]
Processing: encoder.layers.15.feed_forward2.linear1.weight [4096, 1024]
Processing: encoder.layers.15.feed_forward2.linear2.weight [1024, 4096]
Processing: encoder.layers.15.norm_out.weight [1024]
Processing: encoder.layers.15.norm_out.bias [1024]
Processing: encoder.layers.16.norm_feed_forward1.weight [1024]
Processing: encoder.layers.16.norm_feed_forward1.bias [1024]
Processing: encoder.layers.16.feed_forward1.linear1.weight [4096, 1024]
Processing: encoder.layers.16.feed_forward1.linear2.weight [1024, 4096]
Processing: encoder.layers.16.norm_conv.weight [1024]
Processing: encoder.layers.16.norm_conv.bias [1024]
Processing: encoder.layers.16.conv.pointwise_conv1.weight [2048, 1024]
Processing: encoder.layers.16.conv.depthwise_conv.weight [1024, 9]
Processing: encoder.layers.16.conv.batch_norm.weight [1024]
Processing: encoder.layers.16.conv.batch_norm.bias [1024]
Processing: encoder.layers.16.conv.batch_norm.running_mean [1024]
Processing: encoder.layers.16.conv.batch_norm.running_var [1024]
Processing: encoder.layers.16.conv.batch_norm.num_batches_tracked []
Processing: encoder.layers.16.conv.pointwise_conv2.weight [1024, 1024]
Processing: encoder.layers.16.norm_self_att.weight [1024]
Processing: encoder.layers.16.norm_self_att.bias [1024]
Processing: encoder.layers.16.self_attn.pos_bias_u [8, 128]
Processing: encoder.layers.16.self_attn.pos_bias_v [8, 128]
Processing: encoder.layers.16.self_attn.linear_q.weight [1024, 1024]
Processing: encoder.layers.16.self_attn.linear_k.weight [1024, 1024]
Processing: encoder.layers.16.self_attn.linear_v.weight [1024, 1024]
Processing: encoder.layers.16.self_attn.linear_out.weight [1024, 1024]
Processing: encoder.layers.16.self_attn.linear_pos.weight [1024, 1024]
Processing: encoder.layers.16.norm_feed_forward2.weight [1024]
Processing: encoder.layers.16.norm_feed_forward2.bias [1024]
Processing: encoder.layers.16.feed_forward2.linear1.weight [4096, 1024]
Processing: encoder.layers.16.feed_forward2.linear2.weight [1024, 4096]
Processing: encoder.layers.16.norm_out.weight [1024]
Processing: encoder.layers.16.norm_out.bias [1024]
Processing: encoder.layers.17.norm_feed_forward1.weight [1024]
Processing: encoder.layers.17.norm_feed_forward1.bias [1024]
Processing: encoder.layers.17.feed_forward1.linear1.weight [4096, 1024]
Processing: encoder.layers.17.feed_forward1.linear2.weight [1024, 4096]
Processing: encoder.layers.17.norm_conv.weight [1024]
Processing: encoder.layers.17.norm_conv.bias [1024]
Processing: encoder.layers.17.conv.pointwise_conv1.weight [2048, 1024]
Processing: encoder.layers.17.conv.depthwise_conv.weight [1024, 9]
Processing: encoder.layers.17.conv.batch_norm.weight [1024]
Processing: encoder.layers.17.conv.batch_norm.bias [1024]
Processing: encoder.layers.17.conv.batch_norm.running_mean [1024]
Processing: encoder.layers.17.conv.batch_norm.running_var [1024]
Processing: encoder.layers.17.conv.batch_norm.num_batches_tracked []
Processing: encoder.layers.17.conv.pointwise_conv2.weight [1024, 1024]
Processing: encoder.layers.17.norm_self_att.weight [1024]
Processing: encoder.layers.17.norm_self_att.bias [1024]
Processing: encoder.layers.17.self_attn.pos_bias_u [8, 128]
Processing: encoder.layers.17.self_attn.pos_bias_v [8, 128]
Processing: encoder.layers.17.self_attn.linear_q.weight [1024, 1024]
Processing: encoder.layers.17.self_attn.linear_k.weight [1024, 1024]
Processing: encoder.layers.17.self_attn.linear_v.weight [1024, 1024]
Processing: encoder.layers.17.self_attn.linear_out.weight [1024, 1024]
Processing: encoder.layers.17.self_attn.linear_pos.weight [1024, 1024]
Processing: encoder.layers.17.norm_feed_forward2.weight [1024]
Processing: encoder.layers.17.norm_feed_forward2.bias [1024]
Processing: encoder.layers.17.feed_forward2.linear1.weight [4096, 1024]
Processing: encoder.layers.17.feed_forward2.linear2.weight [1024, 4096]
Processing: encoder.layers.17.norm_out.weight [1024]
Processing: encoder.layers.17.norm_out.bias [1024]
Processing: encoder.layers.18.norm_feed_forward1.weight [1024]
Processing: encoder.layers.18.norm_feed_forward1.bias [1024]
Processing: encoder.layers.18.feed_forward1.linear1.weight [4096, 1024]
Processing: encoder.layers.18.feed_forward1.linear2.weight [1024, 4096]
Processing: encoder.layers.18.norm_conv.weight [1024]
Processing: encoder.layers.18.norm_conv.bias [1024]
Processing: encoder.layers.18.conv.pointwise_conv1.weight [2048, 1024]
Processing: encoder.layers.18.conv.depthwise_conv.weight [1024, 9]
Processing: encoder.layers.18.conv.batch_norm.weight [1024]
Processing: encoder.layers.18.conv.batch_norm.bias [1024]
Processing: encoder.layers.18.conv.batch_norm.running_mean [1024]
Processing: encoder.layers.18.conv.batch_norm.running_var [1024]
Processing: encoder.layers.18.conv.batch_norm.num_batches_tracked []
Processing: encoder.layers.18.conv.pointwise_conv2.weight [1024, 1024]
Processing: encoder.layers.18.norm_self_att.weight [1024]
Processing: encoder.layers.18.norm_self_att.bias [1024]
Processing: encoder.layers.18.self_attn.pos_bias_u [8, 128]
Processing: encoder.layers.18.self_attn.pos_bias_v [8, 128]
Processing: encoder.layers.18.self_attn.linear_q.weight [1024, 1024]
Processing: encoder.layers.18.self_attn.linear_k.weight [1024, 1024]
Processing: encoder.layers.18.self_attn.linear_v.weight [1024, 1024]
Processing: encoder.layers.18.self_attn.linear_out.weight [1024, 1024]
Processing: encoder.layers.18.self_attn.linear_pos.weight [1024, 1024]
Processing: encoder.layers.18.norm_feed_forward2.weight [1024]
Processing: encoder.layers.18.norm_feed_forward2.bias [1024]
Processing: encoder.layers.18.feed_forward2.linear1.weight [4096, 1024]
Processing: encoder.layers.18.feed_forward2.linear2.weight [1024, 4096]
Processing: encoder.layers.18.norm_out.weight [1024]
Processing: encoder.layers.18.norm_out.bias [1024]
Processing: encoder.layers.19.norm_feed_forward1.weight [1024]
Processing: encoder.layers.19.norm_feed_forward1.bias [1024]
Processing: encoder.layers.19.feed_forward1.linear1.weight [4096, 1024]
Processing: encoder.layers.19.feed_forward1.linear2.weight [1024, 4096]
Processing: encoder.layers.19.norm_conv.weight [1024]
Processing: encoder.layers.19.norm_conv.bias [1024]
Processing: encoder.layers.19.conv.pointwise_conv1.weight [2048, 1024]
Processing: encoder.layers.19.conv.depthwise_conv.weight [1024, 9]
Processing: encoder.layers.19.conv.batch_norm.weight [1024]
Processing: encoder.layers.19.conv.batch_norm.bias [1024]
Processing: encoder.layers.19.conv.batch_norm.running_mean [1024]
Processing: encoder.layers.19.conv.batch_norm.running_var [1024]
Processing: encoder.layers.19.conv.batch_norm.num_batches_tracked []
Processing: encoder.layers.19.conv.pointwise_conv2.weight [1024, 1024]
Processing: encoder.layers.19.norm_self_att.weight [1024]
Processing: encoder.layers.19.norm_self_att.bias [1024]
Processing: encoder.layers.19.self_attn.pos_bias_u [8, 128]
Processing: encoder.layers.19.self_attn.pos_bias_v [8, 128]
Processing: encoder.layers.19.self_attn.linear_q.weight [1024, 1024]
Processing: encoder.layers.19.self_attn.linear_k.weight [1024, 1024]
Processing: encoder.layers.19.self_attn.linear_v.weight [1024, 1024]
Processing: encoder.layers.19.self_attn.linear_out.weight [1024, 1024]
Processing: encoder.layers.19.self_attn.linear_pos.weight [1024, 1024]
Processing: encoder.layers.19.norm_feed_forward2.weight [1024]
Processing: encoder.layers.19.norm_feed_forward2.bias [1024]
Processing: encoder.layers.19.feed_forward2.linear1.weight [4096, 1024]
Processing: encoder.layers.19.feed_forward2.linear2.weight [1024, 4096]
Processing: encoder.layers.19.norm_out.weight [1024]
Processing: encoder.layers.19.norm_out.bias [1024]
Processing: encoder.layers.20.norm_feed_forward1.weight [1024]
Processing: encoder.layers.20.norm_feed_forward1.bias [1024]
Processing: encoder.layers.20.feed_forward1.linear1.weight [4096, 1024]
Processing: encoder.layers.20.feed_forward1.linear2.weight [1024, 4096]
Processing: encoder.layers.20.norm_conv.weight [1024]
Processing: encoder.layers.20.norm_conv.bias [1024]
Processing: encoder.layers.20.conv.pointwise_conv1.weight [2048, 1024]
Processing: encoder.layers.20.conv.depthwise_conv.weight [1024, 9]
Processing: encoder.layers.20.conv.batch_norm.weight [1024]
Processing: encoder.layers.20.conv.batch_norm.bias [1024]
Processing: encoder.layers.20.conv.batch_norm.running_mean [1024]
Processing: encoder.layers.20.conv.batch_norm.running_var [1024]
Processing: encoder.layers.20.conv.batch_norm.num_batches_tracked []
Processing: encoder.layers.20.conv.pointwise_conv2.weight [1024, 1024]
Processing: encoder.layers.20.norm_self_att.weight [1024]
Processing: encoder.layers.20.norm_self_att.bias [1024]
Processing: encoder.layers.20.self_attn.pos_bias_u [8, 128]
Processing: encoder.layers.20.self_attn.pos_bias_v [8, 128]
Processing: encoder.layers.20.self_attn.linear_q.weight [1024, 1024]
Processing: encoder.layers.20.self_attn.linear_k.weight [1024, 1024]
Processing: encoder.layers.20.self_attn.linear_v.weight [1024, 1024]
Processing: encoder.layers.20.self_attn.linear_out.weight [1024, 1024]
Processing: encoder.layers.20.self_attn.linear_pos.weight [1024, 1024]
Processing: encoder.layers.20.norm_feed_forward2.weight [1024]
Processing: encoder.layers.20.norm_feed_forward2.bias [1024]
Processing: encoder.layers.20.feed_forward2.linear1.weight [4096, 1024]
Processing: encoder.layers.20.feed_forward2.linear2.weight [1024, 4096]
Processing: encoder.layers.20.norm_out.weight [1024]
Processing: encoder.layers.20.norm_out.bias [1024]
Processing: encoder.layers.21.norm_feed_forward1.weight [1024]
Processing: encoder.layers.21.norm_feed_forward1.bias [1024]
Processing: encoder.layers.21.feed_forward1.linear1.weight [4096, 1024]
Processing: encoder.layers.21.feed_forward1.linear2.weight [1024, 4096]
Processing: encoder.layers.21.norm_conv.weight [1024]
Processing: encoder.layers.21.norm_conv.bias [1024]
Processing: encoder.layers.21.conv.pointwise_conv1.weight [2048, 1024]
Processing: encoder.layers.21.conv.depthwise_conv.weight [1024, 9]
Processing: encoder.layers.21.conv.batch_norm.weight [1024]
Processing: encoder.layers.21.conv.batch_norm.bias [1024]
Processing: encoder.layers.21.conv.batch_norm.running_mean [1024]
Processing: encoder.layers.21.conv.batch_norm.running_var [1024]
Processing: encoder.layers.21.conv.batch_norm.num_batches_tracked []
Processing: encoder.layers.21.conv.pointwise_conv2.weight [1024, 1024]
Processing: encoder.layers.21.norm_self_att.weight [1024]
Processing: encoder.layers.21.norm_self_att.bias [1024]
Processing: encoder.layers.21.self_attn.pos_bias_u [8, 128]
Processing: encoder.layers.21.self_attn.pos_bias_v [8, 128]
Processing: encoder.layers.21.self_attn.linear_q.weight [1024, 1024]
Processing: encoder.layers.21.self_attn.linear_k.weight [1024, 1024]
Processing: encoder.layers.21.self_attn.linear_v.weight [1024, 1024]
Processing: encoder.layers.21.self_attn.linear_out.weight [1024, 1024]
Processing: encoder.layers.21.self_attn.linear_pos.weight [1024, 1024]
Processing: encoder.layers.21.norm_feed_forward2.weight [1024]
Processing: encoder.layers.21.norm_feed_forward2.bias [1024]
Processing: encoder.layers.21.feed_forward2.linear1.weight [4096, 1024]
Processing: encoder.layers.21.feed_forward2.linear2.weight [1024, 4096]
Processing: encoder.layers.21.norm_out.weight [1024]
Processing: encoder.layers.21.norm_out.bias [1024]
Processing: encoder.layers.22.norm_feed_forward1.weight [1024]
Processing: encoder.layers.22.norm_feed_forward1.bias [1024]
Processing: encoder.layers.22.feed_forward1.linear1.weight [4096, 1024]
Processing: encoder.layers.22.feed_forward1.linear2.weight [1024, 4096]
Processing: encoder.layers.22.norm_conv.weight [1024]
Processing: encoder.layers.22.norm_conv.bias [1024]
Processing: encoder.layers.22.conv.pointwise_conv1.weight [2048, 1024]
Processing: encoder.layers.22.conv.depthwise_conv.weight [1024, 9]
Processing: encoder.layers.22.conv.batch_norm.weight [1024]
Processing: encoder.layers.22.conv.batch_norm.bias [1024]
Processing: encoder.layers.22.conv.batch_norm.running_mean [1024]
Processing: encoder.layers.22.conv.batch_norm.running_var [1024]
Processing: encoder.layers.22.conv.batch_norm.num_batches_tracked []
Processing: encoder.layers.22.conv.pointwise_conv2.weight [1024, 1024]
Processing: encoder.layers.22.norm_self_att.weight [1024]
Processing: encoder.layers.22.norm_self_att.bias [1024]
Processing: encoder.layers.22.self_attn.pos_bias_u [8, 128]
Processing: encoder.layers.22.self_attn.pos_bias_v [8, 128]
Processing: encoder.layers.22.self_attn.linear_q.weight [1024, 1024]
Processing: encoder.layers.22.self_attn.linear_k.weight [1024, 1024]
Processing: encoder.layers.22.self_attn.linear_v.weight [1024, 1024]
Processing: encoder.layers.22.self_attn.linear_out.weight [1024, 1024]
Processing: encoder.layers.22.self_attn.linear_pos.weight [1024, 1024]
Processing: encoder.layers.22.norm_feed_forward2.weight [1024]
Processing: encoder.layers.22.norm_feed_forward2.bias [1024]
Processing: encoder.layers.22.feed_forward2.linear1.weight [4096, 1024]
Processing: encoder.layers.22.feed_forward2.linear2.weight [1024, 4096]
Processing: encoder.layers.22.norm_out.weight [1024]
Processing: encoder.layers.22.norm_out.bias [1024]
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
Processing: joint.pred.weight [640, 640]
Processing: joint.pred.bias [640]
Processing: joint.enc.weight [640, 1024]
Processing: joint.enc.bias [640]
Processing: joint.joint_net.2.weight [8198, 640]
Processing: joint.joint_net.2.bias [8198]

Conversion complete!
Output file: models/whisper-parakeet/ggml-model.bin
File size: 1197.11 MB
```

### pre_encode
In Parakeet they have the following layer called `pre_encode`:
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

Perfect! Now I can explain the encoder flow. The encoder being called is ConformerEncoder located at
/home/danbev/work/ai/whisper-models/nvidia/parkeet-tdt-0.6b-v3/venv/lib/python3.12/site-packages/nemo/collections/asr/modules/conformer_encoder.py:62

  Call chain for inference:

  1. rnnt_models.py:716 - self.encoder(audio_signal=processed_signal, length=processed_signal_length)
  2. conformer_encoder.py:549-591 - ConformerEncoder.forward()
    - Validates input shape
    - Calls forward_internal()
  3. conformer_encoder.py:593-759 - forward_internal() - Main encoder processing:
    - Line 628: Transpose input from (B, D, T) → (B, T, D)
    - Line 630-633: Pre-encode/subsampling via self.pre_encode (your model has ConvSubsampling)
    - Line 656: Positional encoding via self.pos_enc (RelPositionalEncoding)
    - Line 659-665: Create attention masks
    - Line 675-708: Loop through 24 ConformerLayers (self.layers)
    - Line 738-739: Optional output projection
    - Line 745: Transpose back to (B, D, T)
    - Line 759: Return audio_signal, length

  For debugging in pdb, set breakpoints at:
  - conformer_encoder.py:549 - encoder entry point
  - conformer_encoder.py:633 - after subsampling (reduces time dimension by 8x)
  - conformer_encoder.py:656 - positional encoding
  - conformer_encoder.py:683 - inside layer loop to step through each of the 24 ConformerLayers

