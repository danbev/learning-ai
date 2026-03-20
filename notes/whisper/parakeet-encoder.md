### Parakeet encoder layer
```console
-------------- encoder  ----------------------

audio_signal shape: torch.Size([1, 138, 1024])
audio_signal: tensor([ 487.9199, -313.3978,  109.8925,  422.0488,   20.6455,  106.9309,
         467.3413, -125.1918,  108.5956,  317.1499])
audio_signal ms: 139228.13879601416

pe shape: torch.Size([1, 9999, 1024])
pe: tensor([-0.6639, -0.7478,  0.4186, -0.9082,  0.0015, -1.0000, -0.9134,  0.4070,
         0.6954, -0.7186])
pe ms: 0.49999999986975796

pos_emb shape: torch.Size([1, 275, 1024])
pos_emb: tensor([-0.9425,  0.3342,  0.5063, -0.8623,  0.2102,  0.9777, -0.8400, -0.5427,
         0.9679, -0.2511])
pos_emb ms: 0.4999999996931155
audio_signal shape: torch.Size([1, 138, 1024])
audio_signal: tensor([ 487.9199, -313.3978,  109.8925,  422.0488,   20.6455,  106.9309,
         467.3413, -125.1918,  108.5956,  317.1499])
audio_signal ms: 139228.13879601416

att_mask shape: torch.Size([1, 138, 138])
att_mask: tensor([False, False, False, False, False, False, False, False, False, False])
att_mask ms: 0.0

pad_mask shape: torch.Size([1, 138])
pad_mask: tensor([False, False, False, False, False, False, False, False, False, False])
pad_mask ms: 0.0


-----------------  Layer 0 ConformerLayer --------------
layer.self_attn.linear_pos shape: torch.Size([1024, 1024])
layer.self_attn.linear_pos: tensor([ 0.1493,  0.2252, -0.0377, -0.0243, -0.0554,  0.0214,  0.0899,  0.0771,
         0.0791,  0.0087])
layer.self_attn.linear_pos ms: 0.007167879016591754

input shape: torch.Size([1, 138, 1024])
input: tensor([ 487.9199, -313.3978,  109.8925,  422.0488,   20.6455,  106.9309,
         467.3413, -125.1918,  108.5956,  317.1499])
input ms: 139228.13879601416
norm ffn shape: torch.Size([1, 138, 1024])
norm ffn: tensor([ 0.2587, -0.2806,  0.1463,  0.2680,  0.0485,  0.0919,  0.2121, -0.0258,
         0.1462,  0.1704])
norm ffn ms: 0.13132205009580086

ffn shape: torch.Size([1, 138, 1024])
ffn: tensor([   6.6155,  197.2119, -150.4610,  123.4853,   73.6088, -102.0205,
         127.4773,   80.8189, -202.6328,   43.0125])
ffn ms: 12017.2921508273
fc_factor: 0.5

residual shape: torch.Size([1, 138, 1024])
residual: tensor([ 491.2276, -214.7918,   34.6620,  483.7915,   57.4499,   55.9207,
         531.0800,  -84.7824,    7.2793,  338.6562])
residual ms: 153346.22806187702


-----------------  RelPositionMultiHeadAttention --------------
q_with_bias_u shape: torch.Size([1, 8, 138, 128])
q_with_bias_u : tensor([-7.9274e-01,  4.0974e-02, -5.4027e-03,  2.6341e-02,  2.2412e-01,
         1.3702e-03,  1.2226e-01, -1.4372e+00,  1.2985e-01, -4.9004e-03])
q_with_bias_u ms: 0.10756340975634

q_with_bias_v shape: torch.Size([1, 8, 138, 128])
q_with_bias_v : tensor([-1.1287,  0.0416,  0.0040, -0.0208,  0.2296, -0.0070,  0.1282, -1.1946,
         0.1283, -0.0017])
q_with_bias_v ms: 0.10502276753826272

matrix_bd: torch.Size([1, 8, 138, 275])
matrix_bd : tensor([-25.6702, -25.3171, -23.9058, -22.1954, -21.0549, -20.9036, -21.4795,
        -22.0941, -22.1657, -21.6107])
matrix_bd ms: 23888.763351239817

matrix_ac: torch.Size([1, 8, 138, 138])
matrix_ac : tensor([ 59.4513,   5.7520, -19.7675, -20.4597, -18.0447, -18.3218, -14.0909,
        -23.6188, -13.0872,  -7.2065])
matrix_ac ms: 244.87842905949648

matrix_bd final: torch.Size([1, 8, 138, 138])
matrix_bd final : tensor([-25.6702, -25.3171, -23.9058, -22.1954, -21.0549, -20.9036, -21.4795,
        -22.0941, -22.1657, -21.6107])
matrix_bd final ms: 24059.37315205672
scores final: torch.Size([1, 8, 138, 138])
scores final : tensor([ 2.9859, -1.7293, -3.8602, -3.7702, -3.4559, -3.4671, -3.1440, -4.0405,
        -3.1160, -2.5471])
scores final ms: 191.0932123147937

linear_out.weight shape: torch.Size([1024, 1024])
self.linear_out.weight : tensor([-0.0143, -0.1548,  0.6234,  0.7259,  0.0368, -0.6819, -0.7607, -0.5111,
        -1.8123,  0.7853])
self.linear_out.weight ms: 0.4327764243357476

out shape: torch.Size([1, 138, 1024])
out : tensor([-60.3288,  64.8704,   5.5564,  21.3864,  86.9383,  19.3534, -56.6037,
        -65.3306,   9.2008,  -6.8365])
out ms: 4918.294794905443
```

```console
--------------------- encoder ------------------------
Tensor 'embd_conv', type: f32
ne = [1024 138 1 1]
Tensor value at [0, 0, 0, 0]: 487.920258
Tensor value at [1, 0, 0, 0]: -313.397949
Tensor value at [2, 0, 0, 0]: 109.892563
Tensor value at [3, 0, 0, 0]: 422.048706
Tensor value at [4, 0, 0, 0]: 20.645481
Tensor value at [5, 0, 0, 0]: 106.930923
Tensor value at [6, 0, 0, 0]: 467.341736
Tensor value at [7, 0, 0, 0]: -125.191956
Tensor value at [8, 0, 0, 0]: 108.595947
Tensor value at [9, 0, 0, 0]: 317.150665
embd_conv mean_sq = 139383.4975138433
Tensor 'attn_mask', type: f32
ne = [138 1 1 1]
Tensor value at [0, 0, 0, 0]: 0.000000
Tensor value at [1, 0, 0, 0]: 0.000000
Tensor value at [2, 0, 0, 0]: 0.000000
Tensor value at [3, 0, 0, 0]: 0.000000
Tensor value at [4, 0, 0, 0]: 0.000000
Tensor value at [5, 0, 0, 0]: 0.000000
Tensor value at [6, 0, 0, 0]: 0.000000
Tensor value at [7, 0, 0, 0]: 0.000000
Tensor value at [8, 0, 0, 0]: 0.000000
Tensor value at [9, 0, 0, 0]: 0.000000
attn_mask mean_sq = 0.0000000000
Tensor 'enc_0_res', type: f32
ne = [1024 138 1 1]
Tensor value at [0, 0, 0, 0]: 487.920258
Tensor value at [1, 0, 0, 0]: -313.397949
Tensor value at [2, 0, 0, 0]: 109.892563
Tensor value at [3, 0, 0, 0]: 422.048706
Tensor value at [4, 0, 0, 0]: 20.645481
Tensor value at [5, 0, 0, 0]: 106.930923
Tensor value at [6, 0, 0, 0]: 467.341736
Tensor value at [7, 0, 0, 0]: -125.191956
Tensor value at [8, 0, 0, 0]: 108.595947
Tensor value at [9, 0, 0, 0]: 317.150665
enc_0_res mean_sq = 139383.4975138433
Tensor 'enc_0_ffn_norm_1', type: f32
ne = [1024 138 1 1]
Tensor value at [0, 0, 0, 0]: 0.258695
Tensor value at [1, 0, 0, 0]: -0.280570
Tensor value at [2, 0, 0, 0]: 0.146308
Tensor value at [3, 0, 0, 0]: 0.267969
Tensor value at [4, 0, 0, 0]: 0.048496
Tensor value at [5, 0, 0, 0]: 0.091927
Tensor value at [6, 0, 0, 0]: 0.212106
Tensor value at [7, 0, 0, 0]: -0.025771
Tensor value at [8, 0, 0, 0]: 0.146185
Tensor value at [9, 0, 0, 0]: 0.170360
enc_0_ffn_norm_1 mean_sq = 0.1313311797
Tensor 'enc_0_ffn_1', type: f32
ne = [1024 138 1 1]
Tensor value at [0, 0, 0, 0]: 6.615574
Tensor value at [1, 0, 0, 0]: 197.211914
Tensor value at [2, 0, 0, 0]: -150.460953
Tensor value at [3, 0, 0, 0]: 123.485283
Tensor value at [4, 0, 0, 0]: 73.608658
Tensor value at [5, 0, 0, 0]: -102.020386
Tensor value at [6, 0, 0, 0]: 127.477524
Tensor value at [7, 0, 0, 0]: 80.818802
Tensor value at [8, 0, 0, 0]: -202.632614
Tensor value at [9, 0, 0, 0]: 43.012436
enc_0_ffn_1 mean_sq = 12027.1049737595
Tensor 'enc_0_res_ffn', type: f32
ne = [1024 138 1 1]
Tensor value at [0, 0, 0, 0]: 491.228058
Tensor value at [1, 0, 0, 0]: -214.791992
Tensor value at [2, 0, 0, 0]: 34.662086
Tensor value at [3, 0, 0, 0]: 483.791351
Tensor value at [4, 0, 0, 0]: 57.449810
Tensor value at [5, 0, 0, 0]: 55.920731
Tensor value at [6, 0, 0, 0]: 531.080505
Tensor value at [7, 0, 0, 0]: -84.782555
Tensor value at [8, 0, 0, 0]: 7.279640
Tensor value at [9, 0, 0, 0]: 338.656891
enc_0_res_ffn mean_sq = 153515.4263211748
```
Attention block:
```console
Tensor 'enc_0_attn_norm', type: f32
ne = [1024 138 1 1]
Tensor value at [0, 0, 0, 0]: 0.082301
Tensor value at [1, 0, 0, 0]: -0.034343
Tensor value at [2, 0, 0, 0]: 0.007430
Tensor value at [3, 0, 0, 0]: 0.067338
Tensor value at [4, 0, 0, 0]: 0.004397
Tensor value at [5, 0, 0, 0]: -0.000563
Tensor value at [6, 0, 0, 0]: 0.052343
Tensor value at [7, 0, 0, 0]: -0.008798
Tensor value at [8, 0, 0, 0]: -0.006785
Tensor value at [9, 0, 0, 0]: 0.046064
enc_0_attn_norm mean_sq = 0.0065364573

Tensor 'pe', type: f32
ne = [1024 9999 1 1]
Tensor value at [0, 0, 0, 0]: -0.663950
Tensor value at [1, 0, 0, 0]: -0.747777
Tensor value at [2, 0, 0, 0]: 0.418575
Tensor value at [3, 0, 0, 0]: -0.908182
Tensor value at [4, 0, 0, 0]: 0.001462
Tensor value at [5, 0, 0, 0]: -0.999999
Tensor value at [6, 0, 0, 0]: -0.913418
Tensor value at [7, 0, 0, 0]: 0.407022
Tensor value at [8, 0, 0, 0]: 0.695440
Tensor value at [9, 0, 0, 0]: -0.718584
pe mean_sq = 0.5000000000

Tensor 'enc_0_attn_pos_emb', type: f32
ne = [1024 275 1 1]
Tensor value at [0, 0, 0, 0]: -0.942514
Tensor value at [1, 0, 0, 0]: 0.334165
Tensor value at [2, 0, 0, 0]: 0.506334
Tensor value at [3, 0, 0, 0]: -0.862338
Tensor value at [4, 0, 0, 0]: 0.210175
Tensor value at [5, 0, 0, 0]: 0.977664
Tensor value at [6, 0, 0, 0]: -0.839958
Tensor value at [7, 0, 0, 0]: -0.542651
Tensor value at [8, 0, 0, 0]: 0.967950
Tensor value at [9, 0, 0, 0]: -0.251145
enc_0_attn_pos_emb mean_sq = 0.4999999998

Tensor 'enc_0_attn_pos_w', type: f32
ne = [1024 1024 1 1]
Tensor value at [0, 0, 0, 0]: 0.149347
Tensor value at [1, 0, 0, 0]: 0.225230
Tensor value at [2, 0, 0, 0]: -0.037711
Tensor value at [3, 0, 0, 0]: -0.024281
Tensor value at [4, 0, 0, 0]: -0.055403
Tensor value at [5, 0, 0, 0]: 0.021391
Tensor value at [6, 0, 0, 0]: 0.089919
Tensor value at [7, 0, 0, 0]: 0.077065
Tensor value at [8, 0, 0, 0]: 0.079095
Tensor value at [9, 0, 0, 0]: 0.008742
enc_0_attn_pos_w mean_sq = 0.0071678790

Tensor 'enc_0_attn_q_u', type: f32
ne = [128 8 138 1]
Tensor value at [0, 0, 0, 0]: -0.792737
Tensor value at [1, 0, 0, 0]: 0.040973
Tensor value at [2, 0, 0, 0]: -0.005403
Tensor value at [3, 0, 0, 0]: 0.026341
Tensor value at [4, 0, 0, 0]: 0.224118
Tensor value at [5, 0, 0, 0]: 0.001370
Tensor value at [6, 0, 0, 0]: 0.122264
Tensor value at [7, 0, 0, 0]: -1.437167
Tensor value at [8, 0, 0, 0]: 0.129851
Tensor value at [9, 0, 0, 0]: -0.004900
enc_0_attn_q_u mean_sq = 0.1075830293

Tensor 'enc_0_attn_q_v', type: f32
ne = [128 8 138 1]
Tensor value at [0, 0, 0, 0]: -1.128747
Tensor value at [1, 0, 0, 0]: 0.041612
Tensor value at [2, 0, 0, 0]: 0.003982
Tensor value at [3, 0, 0, 0]: -0.020817
Tensor value at [4, 0, 0, 0]: 0.229579
Tensor value at [5, 0, 0, 0]: -0.006974
Tensor value at [6, 0, 0, 0]: 0.128156
Tensor value at [7, 0, 0, 0]: -1.194555
Tensor value at [8, 0, 0, 0]: 0.128322
Tensor value at [9, 0, 0, 0]: -0.001726
enc_0_attn_q_v mean_sq = 0.1050378194


enc_0_attn_content_scores mean_sq = 244.9378855753
Tensor 'enc_0_attn_rel_pos', type: f32
ne = [275 138 8 1]
Tensor value at [0, 0, 0, 0]: -26.429682
Tensor value at [1, 0, 0, 0]: -26.967243
Tensor value at [2, 0, 0, 0]: -27.641254
Tensor value at [3, 0, 0, 0]: -28.016020
Tensor value at [4, 0, 0, 0]: -27.315594
Tensor value at [5, 0, 0, 0]: -25.158445
Tensor value at [6, 0, 0, 0]: -22.097214
Tensor value at [7, 0, 0, 0]: -19.421661
Tensor value at [8, 0, 0, 0]: -18.295012
Tensor value at [9, 0, 0, 0]: -18.898653
enc_0_attn_rel_pos mean_sq = 23893.3524661032

Tensor 'enc_0_attn_rel_pos_shifted', type: f32
ne = [138 138 8 1]
Tensor value at [0, 0, 0, 0]: -25.317028
Tensor value at [1, 0, 0, 0]: -23.905769
Tensor value at [2, 0, 0, 0]: -22.195379
Tensor value at [3, 0, 0, 0]: -21.054802
Tensor value at [4, 0, 0, 0]: -20.903545
Tensor value at [5, 0, 0, 0]: -21.479376
Tensor value at [6, 0, 0, 0]: -22.094097
Tensor value at [7, 0, 0, 0]: -22.165684
Tensor value at [8, 0, 0, 0]: -21.610710
Tensor value at [9, 0, 0, 0]: -20.812128
enc_0_attn_rel_pos_shifted mean_sq = 23716.2931996744

Tensor 'enc_0_attn_scores', type: f32
ne = [138 138 8 1]
Tensor value at [0, 0, 0, 0]: 3.017072
Tensor value at [1, 0, 0, 0]: -1.604583
Tensor value at [2, 0, 0, 0]: -3.709028
Tensor value at [3, 0, 0, 0]: -3.669397
Tensor value at [4, 0, 0, 0]: -3.442569
Tensor value at [5, 0, 0, 0]: -3.517963
Tensor value at [6, 0, 0, 0]: -3.198330
Tensor value at [7, 0, 0, 0]: -4.046814
Tensor value at [8, 0, 0, 0]: -3.066896
Tensor value at [9, 0, 0, 0]: -2.476525
enc_0_attn_scores mean_sq = 188.0488475351

Tensor 'leaf_5', type: f32
ne = [1024 1024 1 1]
Tensor value at [0, 0, 0, 0]: -0.014282
Tensor value at [1, 0, 0, 0]: -0.154809
Tensor value at [2, 0, 0, 0]: 0.623413
Tensor value at [3, 0, 0, 0]: 0.725945
Tensor value at [4, 0, 0, 0]: 0.036810
Tensor value at [5, 0, 0, 0]: -0.681947
Tensor value at [6, 0, 0, 0]: -0.760667
Tensor value at [7, 0, 0, 0]: -0.511071
Tensor value at [8, 0, 0, 0]: -1.812253
Tensor value at [9, 0, 0, 0]: 0.785316
leaf_5 mean_sq = 0.4327764243

Tensor 'enc_0_attn_out', type: f32
ne = [1024 138 1 1]
Tensor value at [0, 0, 0, 0]: 23.563305
Tensor value at [1, 0, 0, 0]: -10.230977
Tensor value at [2, 0, 0, 0]: -44.699120
Tensor value at [3, 0, 0, 0]: -49.348930
Tensor value at [4, 0, 0, 0]: -31.713549
Tensor value at [5, 0, 0, 0]: -7.070933
Tensor value at [6, 0, 0, 0]: 45.298119
Tensor value at [7, 0, 0, 0]: 16.056097
Tensor value at [8, 0, 0, 0]: -90.243919
Tensor value at [9, 0, 0, 0]: -49.085518
enc_0_attn_out mean_sq = 2151.0597265859

```

###
```console
(Pdb) b ../NeMo/nemo/collections/asr/parts/submodules/conformer_modules.py:174
Breakpoint 3 at /home/danbev/work/ai/whisper-models/nvidia/NeMo/nemo/collections/asr/parts/submodules/conformer_modules.py:174
```

So we know that:
Tensor 'enc_0_attn_q_u', type: f32
ne = [128 8 138 1]
Tensor value at [0, 0, 0, 0]: -0.792737
Tensor value at [1, 0, 0, 0]: 0.040973
Tensor value at [2, 0, 0, 0]: -0.005403
Tensor value at [3, 0, 0, 0]: 0.026341
Tensor value at [4, 0, 0, 0]: 0.224118
Tensor value at [5, 0, 0, 0]: 0.001370
Tensor value at [6, 0, 0, 0]: 0.122264
Tensor value at [7, 0, 0, 0]: -1.437167
Tensor value at [8, 0, 0, 0]: 0.129851
Tensor value at [9, 0, 0, 0]: -0.004900
enc_0_attn_q_u mean_sq = 0.1075830293

q_with_bias_u shape: torch.Size([1, 8, 138, 128])
q_with_bias_u : tensor([-7.9274e-01,  4.0974e-02, -5.4027e-03,  2.6341e-02,  2.2412e-01,
         1.3702e-03,  1.2226e-01, -1.4372e+00,  1.2985e-01, -4.9004e-03])
q_with_bias_u ms: 0.10756340975634


Tensor 'enc_0_attn_q_v', type: f32
ne = [128 8 138 1]
Tensor value at [0, 0, 0, 0]: -1.128747
Tensor value at [1, 0, 0, 0]: 0.041612
Tensor value at [2, 0, 0, 0]: 0.003982
Tensor value at [3, 0, 0, 0]: -0.020817
Tensor value at [4, 0, 0, 0]: 0.229579
Tensor value at [5, 0, 0, 0]: -0.006974
Tensor value at [6, 0, 0, 0]: 0.128156
Tensor value at [7, 0, 0, 0]: -1.194555
Tensor value at [8, 0, 0, 0]: 0.128322
Tensor value at [9, 0, 0, 0]: -0.001726
enc_0_attn_q_v mean_sq = 0.1050378194

q_with_bias_v shape: torch.Size([1, 8, 138, 128])
q_with_bias_v : tensor([-1.1287,  0.0416,  0.0040, -0.0208,  0.2296, -0.0070,  0.1282, -1.1946,
         0.1283, -0.0017])
q_with_bias_v ms: 0.10502276753826272


And the positional encodings:

Tensor 'enc_0_attn_rel_pos', type: f32
ne = [275 138 8 1]
Tensor value at [0, 0, 0, 0]: -26.429682
Tensor value at [1, 0, 0, 0]: -26.967243
Tensor value at [2, 0, 0, 0]: -27.641254
Tensor value at [3, 0, 0, 0]: -28.016020
Tensor value at [4, 0, 0, 0]: -27.315594
Tensor value at [5, 0, 0, 0]: -25.158445
Tensor value at [6, 0, 0, 0]: -22.097214
Tensor value at [7, 0, 0, 0]: -19.421661
Tensor value at [8, 0, 0, 0]: -18.295012
Tensor value at [9, 0, 0, 0]: -18.898653
enc_0_attn_rel_pos mean_sq = 23893.3524661032

Tensor 'enc_0_attn_rel_pos_shifted', type: f32
ne = [138 138 8 1]
Tensor value at [0, 0, 0, 0]: -25.317028
Tensor value at [1, 0, 0, 0]: -23.905769
Tensor value at [2, 0, 0, 0]: -22.195379
Tensor value at [3, 0, 0, 0]: -21.054802
Tensor value at [4, 0, 0, 0]: -20.903545
Tensor value at [5, 0, 0, 0]: -21.479376
Tensor value at [6, 0, 0, 0]: -22.094097
Tensor value at [7, 0, 0, 0]: -22.165684
Tensor value at [8, 0, 0, 0]: -21.610710
Tensor value at [9, 0, 0, 0]: -20.812128
enc_0_attn_rel_pos_shifted mean_sq = 23716.2931996744

And before shifted in python:
matrix_bd shape: torch.Size([1, 8, 138, 275])
matrix_bd : tensor([-26.4298, -26.9673, -27.6413, -28.0161, -27.3157, -25.1585, -22.0973, -19.4217, -18.2950, -18.8987])
matrix_bd ms: 23960.70181130967

matrix_bd shifted shape: torch.Size([1, 8, 138, 275])
matrix_bd shifted: tensor([-25.6702, -25.3171, -23.9058, -22.1954, -21.0549, -20.9036, -21.4795, -22.0941, -22.1657, -21.6107])
matrix_bd shifted ms: 23888.763351239817

Notice that we have `
Tensor value at [0, 0, 0, 0]: -25.317028
Tensor value at [1, 0, 0, 0]: -23.905769
Tensor value at [2, 0, 0, 0]: -22.195379
But in python we have
[-25.6702, -25.3171, -23.9058, -22.1954, ...
We are missing the first values!
