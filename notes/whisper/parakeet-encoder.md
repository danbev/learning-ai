### Parakeet encoder layer
```console
-------------- encoder  ----------------------

audio_signal shape: torch.Size([1, 138, 1024])
audio_signal: tensor([ 487.9199, -313.3978,  109.8925,  422.0488,   20.6455,  106.9309,
         467.3413, -125.1918,  108.5956,  317.1499])
audio_signal ms: 139228.13879601416
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

###
```console
(Pdb) b ../NeMo/nemo/collections/asr/parts/submodules/conformer_modules.py:174
Breakpoint 3 at /home/danbev/work/ai/whisper-models/nvidia/NeMo/nemo/collections/asr/parts/submodules/conformer_modules.py:174
```
