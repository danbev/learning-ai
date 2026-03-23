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
ffn linear1 weight shape: torch.Size([4096, 1024])
ffn linear1 weight: tensor([-0.2274, -0.4161, -0.5412,  0.0456, -0.5201,  0.1141,  0.0055, -0.5780,
        -0.4342, -0.0625])
ffn linear1 weight ms: 0.22115055122801078
ffn linear2 weight shape: torch.Size([1024, 4096])
ffn linear2 weight: tensor([-0.0072,  0.0099,  0.1530, -0.5591, -0.1071,  0.7194,  0.2407, -0.9738,
         0.2891, -0.3258])
ffn linear2 weight ms: 0.23082542097280803
ffn shape: torch.Size([1, 138, 1024])
ffn: tensor([   6.6155,  197.2119, -150.4610,  123.4853,   73.6088, -102.0205,
         127.4773,   80.8189, -202.6328,   43.0125])
ffn ms: 12017.2921508273
fc_factor: 0.5
residual shape: torch.Size([1, 138, 1024])
residual: tensor([ 491.2276, -214.7918,   34.6620,  483.7915,   57.4499,   55.9207,
         531.0800,  -84.7824,    7.2793,  338.6562])
residual ms: 153346.22806187702
 Attention block
norm shape: torch.Size([1, 138, 1024])
norm: tensor([ 0.0823, -0.0343,  0.0074,  0.0673,  0.0044, -0.0006,  0.0523, -0.0088,
        -0.0068,  0.0461])
norm ms: 0.006534526137260072
-----------------  RelPositionMultiHeadAttention --------------
q_with_bias_u shape: torch.Size([1, 8, 138, 128])
q_with_bias_u : tensor([-7.9274e-01,  4.0974e-02, -5.4027e-03,  2.6341e-02,  2.2412e-01,
         1.3702e-03,  1.2226e-01, -1.4372e+00,  1.2985e-01, -4.9004e-03])
q_with_bias_u ms: 0.10756340975634
q_with_bias_v shape: torch.Size([1, 8, 138, 128])
q_with_bias_v : tensor([-1.1287,  0.0416,  0.0040, -0.0208,  0.2296, -0.0070,  0.1282, -1.1946,
         0.1283, -0.0017])
q_with_bias_v ms: 0.10502276753826272
matrix_bd shape: torch.Size([1, 8, 138, 275])
matrix_bd : tensor([-26.4298, -26.9673, -27.6413, -28.0161, -27.3157, -25.1585, -22.0973,
        -19.4217, -18.2950, -18.8987])
matrix_bd ms: 23960.70181130967
matrix_bd shifted shape: torch.Size([1, 8, 138, 275])
matrix_bd shifted: tensor([-25.6702, -25.3171, -23.9058, -22.1954, -21.0549, -20.9036, -21.4795,
        -22.0941, -22.1657, -21.6107])
matrix_bd shifted ms: 23888.763351239817
matrix_ac shape: torch.Size([1, 8, 138, 138])
matrix_ac : tensor([ 59.4513,   5.7520, -19.7675, -20.4597, -18.0447, -18.3218, -14.0909,
        -23.6188, -13.0872,  -7.2065])
matrix_ac ms: 244.87842905949648
matrix_bd final shape: torch.Size([1, 8, 138, 138])
matrix_bd final : tensor([-25.6702, -25.3171, -23.9058, -22.1954, -21.0549, -20.9036, -21.4795,
        -22.0941, -22.1657, -21.6107])
matrix_bd final ms: 24059.37315205672
scores_pre final shape: torch.Size([1, 8, 138, 138])
scores_pre final : tensor([ 33.7811, -19.5651, -43.6733, -42.6551, -39.0996, -39.2254, -35.5703,
        -45.7129, -35.2530, -28.8172])
scores_pre final ms: 24459.93033991231
scores final shape: torch.Size([1, 8, 138, 138])
scores final : tensor([ 2.9859, -1.7293, -3.8602, -3.7702, -3.4559, -3.4671, -3.1440, -4.0405,
        -3.1160, -2.5471])
scores final ms: 191.0932123147937
linear_out.weight shape: torch.Size([1024, 1024])
self.linear_out.weight : tensor([-0.0143, -0.1548,  0.6234,  0.7259,  0.0368, -0.6819, -0.7607, -0.5111,
        -1.8123,  0.7853])
self.linear_out.weight ms: 0.4327764243357476
mask sum: 0
scores before attn shape: torch.Size([1, 8, 138, 138])
scores before attn : tensor([ 2.9859, -1.7293, -3.8602, -3.7702, -3.4559, -3.4671, -3.1440, -4.0405,
        -3.1160, -2.5471])
scores before attn ms: 191.0932123147937
attn before dropout shape: torch.Size([1, 8, 138, 138])
attn before dropout : tensor([8.0209e-01, 7.1853e-03, 8.5313e-04, 9.3347e-04, 1.2782e-03, 1.2640e-03,
        1.7461e-03, 7.1240e-04, 1.7957e-03, 3.1717e-03])
attn before dropout ms: 0.0002531075320057071
attn V shape: torch.Size([1, 8, 138, 128])
attn V : tensor([-4.4337, -2.2749,  2.5360,  4.4929, -6.2656,  1.3339, -1.9875,  2.5671,
        -1.9471, -4.2665])
attn V ms: 19.189832810281967
attn raw x shape: torch.Size([1, 8, 138, 128])
attn raw x: tensor([-3.5839, -1.4587,  2.1541,  3.2880, -5.2931,  1.4242, -1.5868,  2.1512,
        -1.3767, -3.6639])
attn raw ms: 3.027635145334781
attn after matmul shape: torch.Size([1, 138, 1024])
attn after matmul : tensor([-3.5839, -1.4587,  2.1541,  3.2880, -5.2931,  1.4242, -1.5868,  2.1512,
        -1.3767, -3.6639])
attn after matmul : 3.0276351453347816
before linear_out projection....
out shape: torch.Size([1, 138, 1024])
out : tensor([-60.3288,  64.8704,   5.5564,  21.3864,  86.9383,  19.3534, -56.6037,
        -65.3306,   9.2008,  -6.8365])
out ms: 4918.294794905443
self attn: torch.Size([1, 138, 1024])
self attn: tensor([-60.3288,  64.8704,   5.5564,  21.3864,  86.9383,  19.3534, -56.6037,
        -65.3306,   9.2008,  -6.8365])
self attn ms: 4918.294794905443
self attn res: torch.Size([1, 138, 1024])
self attn res: tensor([ 430.8988, -149.9214,   40.2184,  505.1778,  144.3882,   75.2741,
         474.4763, -150.1130,   16.4801,  331.8197])
self attn res ms: 141916.60804585845
conv norm shape: torch.Size([1, 138, 1024])
conv norm: tensor([ 0.0618, -0.0422,  0.0008,  0.1547,  0.0518,  0.0121, -0.0024,  0.0106,
        -0.0124,  0.0117])
conv norm first timestep: tensor([ 0.0618,  0.0067, -0.0252, -0.0147, -0.1410, -0.0493,  0.0726,  0.0451,
        -0.0204,  0.0438])
conv norm ms: 0.01890855922574891
    self conv.....
pointwise_conv1.weight shape: torch.Size([2048, 1024, 1])
pointwise_conv1: tensor([ 0.1812, -0.4557,  0.2351,  0.1232, -0.3855,  0.0109,  0.8278,  0.0904,
        -0.0076, -0.1848])
pointwise_conv1 ms: 0.2386222757676761
pointwise_conv1 shape: torch.Size([1, 2048, 138])
pointwise_conv1: tensor([11.5671,  1.7493,  5.0826,  5.1614,  6.5431,  8.0117,  1.6655,  5.8726,
         1.4569,  1.3355])
pointwise_conv1 first timestep: tensor([ 11.5671,   0.5456,  -2.7401,   0.4333,   1.8034, -10.0960,  -0.4434,
          5.9441,   9.7434,  -2.7544])
pointwise_conv1 ms: 16.945098720751723
glu shape: torch.Size([1, 1024, 138])
glu: tensor([2.6857, 0.0458, 0.1056, 0.0458, 0.0128, 0.4328, 0.0180, 0.1499, 0.0246,
        0.0238])
glu first timestep: tensor([ 2.6857,  0.0693, -0.2843,  0.0476,  1.5518, -0.6870, -0.0412,  5.7420,
         8.1806, -1.6766])
glu ms: 6.7073230921849465
pad_mask shape: torch.Size([1, 1024, 138])
pad_mask : tensor([2.6857, 0.0458, 0.1056, 0.0458, 0.0128, 0.4328, 0.0180, 0.1499, 0.0246,
        0.0238])
pad_mask first timestep: tensor([ 2.6857,  0.0693, -0.2843,  0.0476,  1.5518, -0.6870, -0.0412,  5.7420,
         8.1806, -1.6766])
pad_mask ms: 6.7073230921849465
depthwise_conv weight shape: torch.Size([1024, 1, 9])
depthwise_conv weight : tensor([ 1.0333e-01,  9.8377e-02, -4.1884e-02, -1.0546e-03, -4.3881e-01,
         1.8495e+00,  2.1246e-01,  9.1525e-02,  2.4795e-02,  3.8878e-02])
depthwise_conv weight ms: 0.7115477333981247
depthwise_conv shape: torch.Size([1, 1024, 138])
depthwise_conv : tensor([-1.0668,  0.1940, -0.0314,  0.3630,  1.0905, -0.1088,  0.2936, -0.0168,
         0.1042,  0.2302])
depthwise first timestep: tensor([-1.0668,  0.0491, -0.1965,  0.1021,  1.7261,  0.2110,  0.1113,  6.8065,
        48.6207,  5.2140])
depthwise_conv ms: 27.07695456478801
batch_norm weight shape: torch.Size([1024])
batch_norm weight : tensor([1.8290, 3.3098, 3.6927, 3.5019, 2.9346, 4.7947, 4.6541, 3.1147, 4.9905,
        3.9719])
batch_norm weight ms: 12.385712789912004
batch_norm bias shape: torch.Size([1024])
batch_norm bias : tensor([-0.7994,  0.7474,  1.8876, -0.1093, -0.1495, -1.4259,  1.8639, -0.8108,
         0.9187,  1.1829])
batch_norm bias ms: 2.0716094014926023
batch norm shape: torch.Size([1, 1024, 138])
batch norm : tensor([-2.5621, -1.0566, -1.3257, -0.8548,  0.0139, -1.4181, -0.9376, -1.3082,
        -1.1638, -1.0134])
batch norm first timestep: tensor([-2.5621,  0.2288,  0.1080,  0.4291,  0.7375,  1.7004, -1.0848,  5.0396,
        13.4760,  4.0654])
batch norm ms: 19.28164988228713
swish shape: torch.Size([1, 1024, 138])
swish: tensor([-0.1835, -0.2726, -0.2782, -0.2551,  0.0070, -0.2765, -0.2638, -0.2784,
        -0.2770, -0.2699])
switsh first timestep: tensor([-0.1835,  0.1275,  0.0569,  0.2599,  0.4989,  1.4378, -0.2740,  5.0072,
        13.4759,  3.9968])
swish ms: 13.717630440419418
pointwise_conv2 shape: torch.Size([1, 1024, 138])
pointwise_conv2: tensor([ 14.9262, -35.1995, -88.5250, -65.0133,  61.1004, -12.4325, -11.3220,
        129.8542,  60.5439, -61.1794])
pointwise_conv2 first timestep: tensor([  14.9262,   40.0671,   27.8411,   -4.9402,  -27.5008,  -64.8882,
          80.3789,  119.9014,   85.4843, -106.0588])
pointwise_conv2 ms: 7194.609276554608
transpose shape: torch.Size([1, 138, 1024])
transpose: tensor([  14.9262,   40.0671,   27.8411,   -4.9402,  -27.5008,  -64.8882,
          80.3789,  119.9014,   85.4843, -106.0588])
transpose first timestep: tensor([ 14.9262, -35.1995, -88.5250, -65.0133,  61.1004, -12.4325, -11.3220,
        129.8542,  60.5439, -61.1794])
transpose ms: 7194.609276554608
conv shape: torch.Size([1, 138, 1024])
conv: tensor([  14.9262,   40.0671,   27.8411,   -4.9402,  -27.5008,  -64.8882,
          80.3789,  119.9014,   85.4843, -106.0588])
conv first timestep: tensor([ 14.9262, -35.1995, -88.5250, -65.0133,  61.1004, -12.4325, -11.3220,
        129.8542,  60.5439, -61.1794])
conv ms: 7194.609276554608
conv dropout(residual): torch.Size([1, 138, 1024])
conv dropout: tensor([ 445.8250, -109.8543,   68.0595,  500.2376,  116.8873,   10.3859,
         554.8551,  -30.2116,  101.9644,  225.7609])
conv dropout first timestep: tensor([ 14.9262, -35.1995, -88.5250, -65.0133,  61.1004, -12.4325, -11.3220,
        129.8542,  60.5439, -61.1794])
conv dropout ms: 142102.92381863206
conv norm ffn2 shape: torch.Size([1, 138, 1024])
conv norm ffn2: tensor([ 0.4610, -0.3495,  0.0989,  0.5687,  0.2298,  0.0773,  0.3931, -0.0310,
         0.3148,  0.1954])
conv norm ffn2 first timestep: tensor([ 0.4610,  0.0270, -0.2790, -0.1752, -0.6969, -0.2704,  0.4648,  0.4882,
         0.0022,  0.2318])
conv norm ffn2 ms: 0.5339882078505828
self.fc_factor: 0.5
conv dropout(residual): torch.Size([1, 138, 1024])
conv dropout: tensor([ 549.2919,   15.9465,  -21.4161,  525.2866,  -86.2817,  -58.1708,
         613.5366, -150.3389,  158.8730,  178.1915])
conv dropout first timestep: tensor([ 206.9339,   29.0183,   42.6313,    1.3109, -234.0870, -248.8197,
           7.7147,  199.5882,  315.7012, -167.7424])
conv dropout ms: 135156.94079037494
conv norm out shape: torch.Size([1, 138, 1024])
conv norm out: tensor([ 7.0499, -0.0602, -0.3672,  6.8621, -1.0881,  0.9099, 11.5434, -4.7990,
         1.5167,  2.9011])
conv norm first timestep: tensor([  7.0499,  -0.3384,  -4.6343,  -3.7465, -13.3748,  -7.0445,   5.6967,
          7.8999,   1.9651,   0.8847])
conv norm out ms: 100.31197063284363
```

```console
--------------------- ffn block ------------------------
Tensor 'embd_conv', type: f32
ne = [1024 138 1 1]
nb = [4 4096 565248 565248]
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
ne = [138 138 1 1]
nb = [4 552 76176 76176]
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
nb = [4 4096 565248 565248]
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
nb = [4 4096 565248 565248]
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
Tensor 'enc_0_ff1_linear1_w', type: f32
ne = [1024 4096 1 1]
nb = [4 4096 16777216 16777216]
Tensor value at [0, 0, 0, 0]: -0.227391
Tensor value at [1, 0, 0, 0]: -0.416074
Tensor value at [2, 0, 0, 0]: -0.541159
Tensor value at [3, 0, 0, 0]: 0.045618
Tensor value at [4, 0, 0, 0]: -0.520147
Tensor value at [5, 0, 0, 0]: 0.114125
Tensor value at [6, 0, 0, 0]: 0.005541
Tensor value at [7, 0, 0, 0]: -0.578040
Tensor value at [8, 0, 0, 0]: -0.434211
Tensor value at [9, 0, 0, 0]: -0.062450
enc_0_ff1_linear1_w mean_sq = 0.2211505512
Tensor 'enc_0_ff1_linear2_w', type: f32
ne = [4096 1024 1 1]
nb = [4 16384 16777216 16777216]
Tensor value at [0, 0, 0, 0]: -0.007173
Tensor value at [1, 0, 0, 0]: 0.009939
Tensor value at [2, 0, 0, 0]: 0.153050
Tensor value at [3, 0, 0, 0]: -0.559111
Tensor value at [4, 0, 0, 0]: -0.107072
Tensor value at [5, 0, 0, 0]: 0.719427
Tensor value at [6, 0, 0, 0]: 0.240691
Tensor value at [7, 0, 0, 0]: -0.973804
Tensor value at [8, 0, 0, 0]: 0.289116
Tensor value at [9, 0, 0, 0]: -0.325829
enc_0_ff1_linear2_w mean_sq = 0.2308254209
Tensor 'enc_0_ffn_1', type: f32
ne = [1024 138 1 1]
nb = [4 4096 565248 565248]
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
Tensor 'enc_0_ffn_1', type: f32
ne = [1024 138 1 1]
nb = [4 4096 565248 565248]
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
nb = [4 4096 565248 565248]
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
--------------------- attention block ------------------------
Tensor 'enc_0_attn_q' not found in graph.
Tensor 'enc_0_attn_norm', type: f32
ne = [1024 138 1 1]
nb = [4 4096 565248 565248]
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
nb = [4 4096 40955904 40955904]
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
nb = [4 4096 1126400 1126400]
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
nb = [4 4096 4194304 4194304]
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
Tensor 'enc_0_attn_pos', type: f32
ne = [1024 275 1 1]
nb = [4 4096 1126400 1126400]
Tensor value at [0, 0, 0, 0]: 29.045818
Tensor value at [1, 0, 0, 0]: -18.000935
Tensor value at [2, 0, 0, 0]: 7.388755
Tensor value at [3, 0, 0, 0]: 75.795540
Tensor value at [4, 0, 0, 0]: 78.194031
Tensor value at [5, 0, 0, 0]: -30.121264
Tensor value at [6, 0, 0, 0]: -2.113738
Tensor value at [7, 0, 0, 0]: 23.785250
Tensor value at [8, 0, 0, 0]: 76.862717
Tensor value at [9, 0, 0, 0]: 2.455283
enc_0_attn_pos mean_sq = 806.2867870198
Tensor 'enc_0_attn_pos_perm', type: f32
ne = [128 275 8 1]
nb = [4 512 140800 1126400]
Tensor value at [0, 0, 0, 0]: -25.670147
Tensor value at [1, 0, 0, 0]: -25.317028
Tensor value at [2, 0, 0, 0]: -23.905769
Tensor value at [3, 0, 0, 0]: -22.195379
Tensor value at [4, 0, 0, 0]: -21.054802
Tensor value at [5, 0, 0, 0]: -20.903545
Tensor value at [6, 0, 0, 0]: -21.479376
Tensor value at [7, 0, 0, 0]: -22.094097
Tensor value at [8, 0, 0, 0]: -22.165684
Tensor value at [9, 0, 0, 0]: -21.610710
enc_0_attn_pos_perm mean_sq = 13257.7836307530
Tensor 'enc_0_attn_q_v_perm', type: f32
ne = [128 138 8 1]
nb = [4 512 70656 565248]
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
enc_0_attn_q_v_perm mean_sq = 0.1050378194
Tensor 'attn_mask', type: f32
ne = [138 138 1 1]
nb = [4 552 76176 76176]
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
Tensor 'enc_0_attn_q_u', type: f32
ne = [128 8 138 1]
nb = [4 512 4096 565248]
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
nb = [4 512 4096 565248]
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
Tensor 'enc_0_attn_content_scores', type: f32
ne = [138 138 8 1]
nb = [4 552 76176 609408]
Tensor value at [0, 0, 0, 0]: 59.451302
Tensor value at [1, 0, 0, 0]: 5.751985
Tensor value at [2, 0, 0, 0]: -19.767485
Tensor value at [3, 0, 0, 0]: -20.459682
Tensor value at [4, 0, 0, 0]: -18.044676
Tensor value at [5, 0, 0, 0]: -18.321831
Tensor value at [6, 0, 0, 0]: -14.090880
Tensor value at [7, 0, 0, 0]: -23.618797
Tensor value at [8, 0, 0, 0]: -13.087257
Tensor value at [9, 0, 0, 0]: -7.206554
enc_0_attn_content_scores mean_sq = 244.9378855753
Tensor 'enc_0_attn_rel_pos', type: f32
ne = [275 138 8 1]
nb = [4 1100 151800 1214400]
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
Tensor 'enc_0_attn_rel_pos_reshaped', type: f32
ne = [138 276 8 1]
nb = [4 552 152352 1218816]
Tensor value at [0, 0, 0, 0]: 0.000000
Tensor value at [1, 0, 0, 0]: -26.429682
Tensor value at [2, 0, 0, 0]: -26.967243
Tensor value at [3, 0, 0, 0]: -27.641254
Tensor value at [4, 0, 0, 0]: -28.016020
Tensor value at [5, 0, 0, 0]: -27.315594
Tensor value at [6, 0, 0, 0]: -25.158445
Tensor value at [7, 0, 0, 0]: -22.097214
Tensor value at [8, 0, 0, 0]: -19.421661
Tensor value at [9, 0, 0, 0]: -18.295012
enc_0_attn_rel_pos_reshaped mean_sq = 23806.7823484724
Tensor 'enc_0_attn_rel_pos_shifted', type: f32
ne = [138 275 8 1]
nb = [4 552 151800 1214400]
Tensor value at [0, 0, 0, 0]: -25.670147
Tensor value at [1, 0, 0, 0]: -25.317028
Tensor value at [2, 0, 0, 0]: -23.905769
Tensor value at [3, 0, 0, 0]: -22.195379
Tensor value at [4, 0, 0, 0]: -21.054802
Tensor value at [5, 0, 0, 0]: -20.903545
Tensor value at [6, 0, 0, 0]: -21.479376
Tensor value at [7, 0, 0, 0]: -22.094097
Tensor value at [8, 0, 0, 0]: -22.165684
Tensor value at [9, 0, 0, 0]: -21.610710
enc_0_attn_rel_pos_shifted mean_sq = 22601.1715295159
Tensor 'enc_0_attn_rel_pos_shifted_view', type: f32
ne = [138 138 8 1]
nb = [4 552 76176 609408]
Tensor value at [0, 0, 0, 0]: -25.670147
Tensor value at [1, 0, 0, 0]: -25.317028
Tensor value at [2, 0, 0, 0]: -23.905769
Tensor value at [3, 0, 0, 0]: -22.195379
Tensor value at [4, 0, 0, 0]: -21.054802
Tensor value at [5, 0, 0, 0]: -20.903545
Tensor value at [6, 0, 0, 0]: -21.479376
Tensor value at [7, 0, 0, 0]: -22.094097
Tensor value at [8, 0, 0, 0]: -22.165684
Tensor value at [9, 0, 0, 0]: -21.610710
enc_0_attn_rel_pos_shifted_view mean_sq = 23871.0724275297
Tensor 'enc_0_attn_scores', type: f32
ne = [138 138 8 1]
nb = [4 552 76176 609408]
Tensor value at [0, 0, 0, 0]: 33.781155
Tensor value at [1, 0, 0, 0]: -19.565044
Tensor value at [2, 0, 0, 0]: -43.673256
Tensor value at [3, 0, 0, 0]: -42.655060
Tensor value at [4, 0, 0, 0]: -39.099480
Tensor value at [5, 0, 0, 0]: -39.225376
Tensor value at [6, 0, 0, 0]: -35.570255
Tensor value at [7, 0, 0, 0]: -45.712894
Tensor value at [8, 0, 0, 0]: -35.252941
Tensor value at [9, 0, 0, 0]: -28.817265
enc_0_attn_scores mean_sq = 24268.6895794422
Tensor 'enc_0_attn_scores_scaled', type: f32
ne = [138 138 8 1]
nb = [4 552 76176 609408]
Tensor value at [0, 0, 0, 0]: 2.985860
Tensor value at [1, 0, 0, 0]: -1.729322
Tensor value at [2, 0, 0, 0]: -3.860207
Tensor value at [3, 0, 0, 0]: -3.770210
Tensor value at [4, 0, 0, 0]: -3.455938
Tensor value at [5, 0, 0, 0]: -3.467066
Tensor value at [6, 0, 0, 0]: -3.143996
Tensor value at [7, 0, 0, 0]: -4.040487
Tensor value at [8, 0, 0, 0]: -3.115949
Tensor value at [9, 0, 0, 0]: -2.547110
enc_0_attn_scores_scaled mean_sq = 189.5991308536
Tensor 'enc_0_attn_probs', type: f32
ne = [138 138 8 1]
nb = [4 552 76176 609408]
Tensor value at [0, 0, 0, 0]: 0.802226
Tensor value at [1, 0, 0, 0]: 0.007187
Tensor value at [2, 0, 0, 0]: 0.000853
Tensor value at [3, 0, 0, 0]: 0.000934
Tensor value at [4, 0, 0, 0]: 0.001278
Tensor value at [5, 0, 0, 0]: 0.001264
Tensor value at [6, 0, 0, 0]: 0.001746
Tensor value at [7, 0, 0, 0]: 0.000713
Tensor value at [8, 0, 0, 0]: 0.001796
Tensor value at [9, 0, 0, 0]: 0.003172
enc_0_attn_probs mean_sq = 0.0002053219
Tensor 'enc_0_attn_v' not found in graph.
Tensor 'enc_0_attn_inp', type: f32
ne = [138 128 8 1]
nb = [4 552 70656 565248]
Tensor value at [0, 0, 0, 0]: -3.583522
Tensor value at [1, 0, 0, 0]: -0.731219
Tensor value at [2, 0, 0, 0]: 1.137398
Tensor value at [3, 0, 0, 0]: 0.620410
Tensor value at [4, 0, 0, 0]: 0.839996
Tensor value at [5, 0, 0, 0]: 0.945457
Tensor value at [6, 0, 0, 0]: 0.943489
Tensor value at [7, 0, 0, 0]: 2.052510
Tensor value at [8, 0, 0, 0]: 0.892313
Tensor value at [9, 0, 0, 0]: 2.360969
enc_0_attn_inp mean_sq = 2.7829171814
Tensor 'enc_0_attn_out', type: f32
ne = [1024 138 1 1]
nb = [4 4096 565248 565248]
Tensor value at [0, 0, 0, 0]: -60.415253
Tensor value at [1, 0, 0, 0]: 64.783768
Tensor value at [2, 0, 0, 0]: 5.617332
Tensor value at [3, 0, 0, 0]: 21.489803
Tensor value at [4, 0, 0, 0]: 87.459686
Tensor value at [5, 0, 0, 0]: 19.067219
Tensor value at [6, 0, 0, 0]: -56.190159
Tensor value at [7, 0, 0, 0]: -65.619751
Tensor value at [8, 0, 0, 0]: 9.149857
Tensor value at [9, 0, 0, 0]: -7.055437
enc_0_attn_out mean_sq = 4452.4519602286
Tensor 'enc_0_attn_res', type: f32
ne = [1024 138 1 1]
nb = [4 4096 565248 565248]
Tensor value at [0, 0, 0, 0]: 430.812805
Tensor value at [1, 0, 0, 0]: -150.008224
Tensor value at [2, 0, 0, 0]: 40.279419
Tensor value at [3, 0, 0, 0]: 505.281158
Tensor value at [4, 0, 0, 0]: 144.909500
Tensor value at [5, 0, 0, 0]: 74.987946
Tensor value at [6, 0, 0, 0]: 474.890350
Tensor value at [7, 0, 0, 0]: -150.402313
Tensor value at [8, 0, 0, 0]: 16.429497
Tensor value at [9, 0, 0, 0]: 331.601440
enc_0_attn_res mean_sq = 144377.3928269203
--------------------- convolution block ------------------------
Tensor 'enc_0_residual_conv', type: f32
ne = [1024 138 1 1]
nb = [4 4096 565248 565248]
Tensor value at [0, 0, 0, 0]: 430.812805
Tensor value at [1, 0, 0, 0]: -150.008224
Tensor value at [2, 0, 0, 0]: 40.279419
Tensor value at [3, 0, 0, 0]: 505.281158
Tensor value at [4, 0, 0, 0]: 144.909500
Tensor value at [5, 0, 0, 0]: 74.987946
Tensor value at [6, 0, 0, 0]: 474.890350
Tensor value at [7, 0, 0, 0]: -150.402313
Tensor value at [8, 0, 0, 0]: 16.429497
Tensor value at [9, 0, 0, 0]: 331.601440
enc_0_residual_conv mean_sq = 144377.3928269203
Tensor 'enc_0_norm_conv_w', type: f32
ne = [1024 1 1 1]
nb = [4 4096 4096 4096]
Tensor value at [0, 0, 0, 0]: 0.088132
Tensor value at [1, 0, 0, 0]: 0.165619
Tensor value at [2, 0, 0, 0]: 0.153596
Tensor value at [3, 0, 0, 0]: 0.191302
Tensor value at [4, 0, 0, 0]: 0.153157
Tensor value at [5, 0, 0, 0]: 0.086507
Tensor value at [6, 0, 0, 0]: 0.061375
Tensor value at [7, 0, 0, 0]: 0.070593
Tensor value at [8, 0, 0, 0]: 0.222852
Tensor value at [9, 0, 0, 0]: 0.096418
enc_0_norm_conv_w mean_sq = 0.0164449651
Tensor 'enc_0_norm_conv_b', type: f32
ne = [1024 1 1 1]
nb = [4 4096 4096 4096]
Tensor value at [0, 0, 0, 0]: 0.006895
Tensor value at [1, 0, 0, 0]: -0.006137
Tensor value at [2, 0, 0, 0]: -0.008066
Tensor value at [3, 0, 0, 0]: 0.014847
Tensor value at [4, 0, 0, 0]: 0.019850
Tensor value at [5, 0, 0, 0]: 0.002688
Tensor value at [6, 0, 0, 0]: -0.044522
Tensor value at [7, 0, 0, 0]: 0.025966
Tensor value at [8, 0, 0, 0]: -0.017648
Tensor value at [9, 0, 0, 0]: -0.034562
enc_0_norm_conv_b mean_sq = 0.0035098246
Tensor 'enc_0_norm_conv', type: f32
ne = [1024 138 1 1]
nb = [4 4096 565248 565248]
Tensor value at [0, 0, 0, 0]: 0.061833
Tensor value at [1, 0, 0, 0]: -0.042184
Tensor value at [2, 0, 0, 0]: 0.000824
Tensor value at [3, 0, 0, 0]: 0.154724
Tensor value at [4, 0, 0, 0]: 0.051917
Tensor value at [5, 0, 0, 0]: 0.012043
Tensor value at [6, 0, 0, 0]: -0.002346
Tensor value at [7, 0, 0, 0]: 0.010561
Tensor value at [8, 0, 0, 0]: -0.012446
Tensor value at [9, 0, 0, 0]: 0.011690
enc_0_norm_conv mean_sq = 0.0189186877
Tensor 'enc_0_conv_pw1_w', type: f32
ne = [1024 2048 1 1]
nb = [4 4096 8388608 8388608]
Tensor value at [0, 0, 0, 0]: 0.181187
Tensor value at [1, 0, 0, 0]: -0.455710
Tensor value at [2, 0, 0, 0]: 0.235130
Tensor value at [3, 0, 0, 0]: 0.123168
Tensor value at [4, 0, 0, 0]: -0.385514
Tensor value at [5, 0, 0, 0]: 0.010906
Tensor value at [6, 0, 0, 0]: 0.827813
Tensor value at [7, 0, 0, 0]: 0.090416
Tensor value at [8, 0, 0, 0]: -0.007626
Tensor value at [9, 0, 0, 0]: -0.184768
enc_0_conv_pw1_w mean_sq = 0.2386222758
Tensor 'enc_0_conv_pw1', type: f32
ne = [2048 138 1 1]
nb = [4 8192 1130496 1130496]
Tensor value at [0, 0, 0, 0]: 11.568198
Tensor value at [1, 0, 0, 0]: 0.546999
Tensor value at [2, 0, 0, 0]: -2.741590
Tensor value at [3, 0, 0, 0]: 0.431667
Tensor value at [4, 0, 0, 0]: 1.803554
Tensor value at [5, 0, 0, 0]: -10.093614
Tensor value at [6, 0, 0, 0]: -0.443020
Tensor value at [7, 0, 0, 0]: 5.943117
Tensor value at [8, 0, 0, 0]: 9.744120
Tensor value at [9, 0, 0, 0]: -2.755332
enc_0_conv_pw1 mean_sq = 17.0137245494
Tensor 'enc_0_conv_glu', type: f32
ne = [1024 138 1 1]
nb = [4 4096 565248 565248]
Tensor value at [0, 0, 0, 0]: 2.685092
Tensor value at [1, 0, 0, 0]: 0.069481
Tensor value at [2, 0, 0, 0]: -0.284969
Tensor value at [3, 0, 0, 0]: 0.047531
Tensor value at [4, 0, 0, 0]: 1.552031
Tensor value at [5, 0, 0, 0]: -0.687057
Tensor value at [6, 0, 0, 0]: -0.041173
Tensor value at [7, 0, 0, 0]: 5.740987
Tensor value at [8, 0, 0, 0]: 8.179447
Tensor value at [9, 0, 0, 0]: -1.678105
enc_0_conv_glu mean_sq = 6.7612585152
Tensor 'enc_0_conv_dw_pad', type: f32
ne = [146 1024 1 1]
nb = [4 584 598016 598016]
Tensor value at [0, 0, 0, 0]: 0.000000
Tensor value at [1, 0, 0, 0]: 0.000000
Tensor value at [2, 0, 0, 0]: 0.000000
Tensor value at [3, 0, 0, 0]: 0.000000
Tensor value at [4, 0, 0, 0]: 2.685092
Tensor value at [5, 0, 0, 0]: 0.047974
Tensor value at [6, 0, 0, 0]: 0.117030
Tensor value at [7, 0, 0, 0]: 0.048753
Tensor value at [8, 0, 0, 0]: 0.014454
Tensor value at [9, 0, 0, 0]: 0.460651
enc_0_conv_dw_pad mean_sq = 6.3907785966
Tensor 'enc_0_conv_dw_w', type: f32
ne = [9 1024 1 1]
nb = [4 36 36864 36864]
Tensor value at [0, 0, 0, 0]: 0.103328
Tensor value at [1, 0, 0, 0]: 0.098377
Tensor value at [2, 0, 0, 0]: -0.041884
Tensor value at [3, 0, 0, 0]: -0.001055
Tensor value at [4, 0, 0, 0]: -0.438810
Tensor value at [5, 0, 0, 0]: 1.849534
Tensor value at [6, 0, 0, 0]: 0.212458
Tensor value at [7, 0, 0, 0]: 0.091525
Tensor value at [8, 0, 0, 0]: 0.024795
Tensor value at [0, 1, 0, 0]: 0.038878
enc_0_conv_dw_w mean_sq = 0.7115477345
Tensor 'enc_0_conv_dw_weights' not found in graph.
Tensor 'enc_0_conv_1d_dw', type: f32
ne = [1024 138 1 1]
nb = [4 4096 565248 565248]
Tensor value at [0, 0, 0, 0]: -1.059831
Tensor value at [1, 0, 0, 0]: 0.050381
Tensor value at [2, 0, 0, 0]: -0.211476
Tensor value at [3, 0, 0, 0]: 0.098426
Tensor value at [4, 0, 0, 0]: 1.742875
Tensor value at [5, 0, 0, 0]: 0.204349
Tensor value at [6, 0, 0, 0]: 0.111458
Tensor value at [7, 0, 0, 0]: 6.683582
Tensor value at [8, 0, 0, 0]: 48.647888
Tensor value at [9, 0, 0, 0]: 5.232388
enc_0_conv_1d_dw mean_sq = 27.3373583406
Tensor 'enc_0_conv_bn_w', type: f32
ne = [1024 1 1 1]
nb = [4 4096 4096 4096]
Tensor value at [0, 0, 0, 0]: 1.829028
Tensor value at [1, 0, 0, 0]: 3.309807
Tensor value at [2, 0, 0, 0]: 3.692689
Tensor value at [3, 0, 0, 0]: 3.501884
Tensor value at [4, 0, 0, 0]: 2.934583
Tensor value at [5, 0, 0, 0]: 4.794742
Tensor value at [6, 0, 0, 0]: 4.654086
Tensor value at [7, 0, 0, 0]: 3.114681
Tensor value at [8, 0, 0, 0]: 4.990479
Tensor value at [9, 0, 0, 0]: 3.971935
enc_0_conv_bn_w mean_sq = 12.3857127957
Tensor 'enc_0_conv_bn_b', type: f32
ne = [1024 1 1 1]
nb = [4 4096 4096 4096]
Tensor value at [0, 0, 0, 0]: -0.799362
Tensor value at [1, 0, 0, 0]: 0.747423
Tensor value at [2, 0, 0, 0]: 1.887580
Tensor value at [3, 0, 0, 0]: -0.109272
Tensor value at [4, 0, 0, 0]: -0.149506
Tensor value at [5, 0, 0, 0]: -1.425901
Tensor value at [6, 0, 0, 0]: 1.863889
Tensor value at [7, 0, 0, 0]: -0.810838
Tensor value at [8, 0, 0, 0]: 0.918657
Tensor value at [9, 0, 0, 0]: 1.182906
enc_0_conv_bn_b mean_sq = 2.0716094031
Tensor 'enc_0_conv_bn', type: f32
ne = [1024 138 1 1]
nb = [4 4096 565248 565248]
Tensor value at [0, 0, 0, 0]: -2.553764
Tensor value at [1, 0, 0, 0]: 0.231162
Tensor value at [2, 0, 0, 0]: 0.096240
Tensor value at [3, 0, 0, 0]: 0.424527
Tensor value at [4, 0, 0, 0]: 0.749338
Tensor value at [5, 0, 0, 0]: 1.684467
Tensor value at [6, 0, 0, 0]: -1.084650
Tensor value at [7, 0, 0, 0]: 4.939498
Tensor value at [8, 0, 0, 0]: 13.484200
Tensor value at [9, 0, 0, 0]: 4.082446
enc_0_conv_bn mean_sq = 19.4521985282
Tensor 'enc_0_conv_silu', type: f32
ne = [1024 138 1 1]
nb = [4 4096 565248 565248]
Tensor value at [0, 0, 0, 0]: -0.184315
Tensor value at [1, 0, 0, 0]: 0.128881
Tensor value at [2, 0, 0, 0]: 0.050434
Tensor value at [3, 0, 0, 0]: 0.256654
Tensor value at [4, 0, 0, 0]: 0.508826
Tensor value at [5, 0, 0, 0]: 1.420840
Tensor value at [6, 0, 0, 0]: -0.274012
Tensor value at [7, 0, 0, 0]: 4.904392
Tensor value at [8, 0, 0, 0]: 13.484181
Tensor value at [9, 0, 0, 0]: 4.014732
enc_0_conv_silu mean_sq = 13.8447128000
Tensor 'enc_0_conv_pw2', type: f32
ne = [1024 138 1 1]
nb = [4 4096 565248 565248]
Tensor value at [0, 0, 0, 0]: 15.265339
Tensor value at [1, 0, 0, 0]: 41.890793
Tensor value at [2, 0, 0, 0]: 29.166378
Tensor value at [3, 0, 0, 0]: -4.641632
Tensor value at [4, 0, 0, 0]: -28.494171
Tensor value at [5, 0, 0, 0]: -64.793800
Tensor value at [6, 0, 0, 0]: 80.337692
Tensor value at [7, 0, 0, 0]: 118.784485
Tensor value at [8, 0, 0, 0]: 85.281563
Tensor value at [9, 0, 0, 0]: -105.220612
enc_0_conv_pw2 mean_sq = 7355.9159808887
Tensor 'enc_0_conv_res', type: f32
ne = [1024 138 1 1]
nb = [4 4096 565248 565248]
Tensor value at [0, 0, 0, 0]: 446.078156
Tensor value at [1, 0, 0, 0]: -108.117432
Tensor value at [2, 0, 0, 0]: 69.445801
Tensor value at [3, 0, 0, 0]: 500.639526
Tensor value at [4, 0, 0, 0]: 116.415329
Tensor value at [5, 0, 0, 0]: 10.194145
Tensor value at [6, 0, 0, 0]: 555.228027
Tensor value at [7, 0, 0, 0]: -31.617828
Tensor value at [8, 0, 0, 0]: 101.711060
Tensor value at [9, 0, 0, 0]: 226.380829
enc_0_conv_res mean_sq = 144050.7795043931
--------------------- ffn2 block ------------------------
Tensor 'enc_0_ffn_norm_2', type: f32
ne = [1024 138 1 1]
nb = [4 4096 565248 565248]
Tensor value at [0, 0, 0, 0]: 0.461161
Tensor value at [1, 0, 0, 0]: -0.346955
Tensor value at [2, 0, 0, 0]: 0.100777
Tensor value at [3, 0, 0, 0]: 0.569235
Tensor value at [4, 0, 0, 0]: 0.229204
Tensor value at [5, 0, 0, 0]: 0.077152
Tensor value at [6, 0, 0, 0]: 0.393244
Tensor value at [7, 0, 0, 0]: -0.031368
Tensor value at [8, 0, 0, 0]: 0.314326
Tensor value at [9, 0, 0, 0]: 0.195936
enc_0_ffn_norm_2 mean_sq = 0.5367008923
Tensor 'enc_0_ffn_res', type: f32
ne = [1024 138 1 1]
nb = [4 4096 565248 565248]
Tensor value at [0, 0, 0, 0]: 549.468506
Tensor value at [1, 0, 0, 0]: 17.761490
Tensor value at [2, 0, 0, 0]: -20.095726
Tensor value at [3, 0, 0, 0]: 526.312988
Tensor value at [4, 0, 0, 0]: -86.670700
Tensor value at [5, 0, 0, 0]: -58.718094
Tensor value at [6, 0, 0, 0]: 614.930542
Tensor value at [7, 0, 0, 0]: -151.191513
Tensor value at [8, 0, 0, 0]: 158.742523
Tensor value at [9, 0, 0, 0]: 178.830414
enc_0_ffn_res mean_sq = 136927.2894156699
--------------------- norm out block ------------------------
Tensor 'enc_1_res', type: f32
ne = [1024 138 1 1]
nb = [4 4096 565248 565248]
Tensor value at [0, 0, 0, 0]: 7.050279
Tensor value at [1, 0, 0, 0]: -0.037476
Tensor value at [2, 0, 0, 0]: -0.353877
Tensor value at [3, 0, 0, 0]: 6.873757
Tensor value at [4, 0, 0, 0]: -1.092907
Tensor value at [5, 0, 0, 0]: 0.899727
Tensor value at [6, 0, 0, 0]: 11.566944
Tensor value at [7, 0, 0, 0]: -4.812798
Tensor value at [8, 0, 0, 0]: 1.515178
Tensor value at [9, 0, 0, 0]: 2.909692
enc_1_res mean_sq = 100.6613881528
```
