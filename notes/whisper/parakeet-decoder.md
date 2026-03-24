## Parakeet decoder notes

### Decoder implementation
```console
(Pdb) b ../NeMo/nemo/collections/common/parts/rnn.py:247
Breakpoint 1 at /home/danbev/work/ai/whisper-models/nvidia/NeMo/nemo/collections/common/parts/rnn.py:247
```

### Tensors operations
Original model:
Note: in the original model there are two LSTM layers and therefor in python
we see 2 weight_ih and 2 weight_hh tensors. We also have this in our converted
model by they are in a vector so it might not obvious looking at the converted
tensor names to see a direct match.

```console
--------- LSTM ------------
weight_ih_l0 weight shape: torch.Size([2560, 640])
weight_ih_l0: tensor([ 0.1676,  0.1051, -0.2337, -0.1759,  0.2805, -0.0289,  0.3011,  0.2723,
         0.4243,  0.3569])
weight_ih_l0 ms: 0.0863738453256456

weight_ih_l1 weight shape: torch.Size([2560, 640])
weight_ih_l1: tensor([ 0.0751,  0.0539,  0.0842, -0.0482,  0.0227,  0.0177, -0.0183, -0.0230,
        -0.1247, -0.0111])
weight_ih_l1 ms: 0.01959334758041556
```
```console
Tensor 'pred_0_ih_w', type: f32
ne = [640 2560 1 1]
nb = [4 2560 6553600 6553600]
Tensor value at [0, 0, 0, 0]: 0.167618
Tensor value at [1, 0, 0, 0]: 0.105139
Tensor value at [2, 0, 0, 0]: -0.233656
Tensor value at [3, 0, 0, 0]: -0.175924
Tensor value at [4, 0, 0, 0]: 0.280509
Tensor value at [5, 0, 0, 0]: -0.028884
Tensor value at [6, 0, 0, 0]: 0.301097
Tensor value at [7, 0, 0, 0]: 0.272333
Tensor value at [8, 0, 0, 0]: 0.424270
Tensor value at [9, 0, 0, 0]: 0.356870
pred_0_ih_w mean_sq = 0.0863738453

Tensor 'pred_1_ih_w', type: f32
ne = [640 2560 1 1]
nb = [4 2560 6553600 6553600]
Tensor value at [0, 0, 0, 0]: 0.075094
Tensor value at [1, 0, 0, 0]: 0.053861
Tensor value at [2, 0, 0, 0]: 0.084203
Tensor value at [3, 0, 0, 0]: -0.048229
Tensor value at [4, 0, 0, 0]: 0.022730
Tensor value at [5, 0, 0, 0]: 0.017651
Tensor value at [6, 0, 0, 0]: -0.018252
Tensor value at [7, 0, 0, 0]: -0.023009
Tensor value at [8, 0, 0, 0]: -0.124707
Tensor value at [9, 0, 0, 0]: -0.011123
pred_1_ih_w mean_sq = 0.0195933476
```
So the two input to hidden weights look good.

And the biases:
```console
bias_ih_l0 shape: torch.Size([2560])
bias_ih_l0: tensor([ 0.6148,  0.0201, -0.3556,  0.4211,  0.5016,  0.0922,  0.1749,  0.2933,
         0.3930,  0.2531])
bias_ih_l0 ms: 0.15821436759665083

bias_ih_l1 shape: torch.Size([2560])
bias_ih_l1: tensor([-1.0063,  0.1899, -0.0338, -0.1681, -0.6291, -1.4161, -1.8615, -0.7408,
        -0.6594,  0.1817])
bias_ih_l1 ms: 0.5914398869601122
```
```console
Tensor 'pred_0_ih_b', type: f32
ne = [2560 1 1 1]
nb = [4 10240 10240 10240]
Tensor value at [0, 0, 0, 0]: 0.614826
Tensor value at [1, 0, 0, 0]: 0.020149
Tensor value at [2, 0, 0, 0]: -0.355560
Tensor value at [3, 0, 0, 0]: 0.421130
Tensor value at [4, 0, 0, 0]: 0.501597
Tensor value at [5, 0, 0, 0]: 0.092152
Tensor value at [6, 0, 0, 0]: 0.174912
Tensor value at [7, 0, 0, 0]: 0.293331
Tensor value at [8, 0, 0, 0]: 0.393027
Tensor value at [9, 0, 0, 0]: 0.253118
pred_0_ih_b mean_sq = 0.1582143676
Tensor 'pred_1_ih_b', type: f32
ne = [2560 1 1 1]
nb = [4 10240 10240 10240]
Tensor value at [0, 0, 0, 0]: -1.006311
Tensor value at [1, 0, 0, 0]: 0.189915
Tensor value at [2, 0, 0, 0]: -0.033847
Tensor value at [3, 0, 0, 0]: -0.168075
Tensor value at [4, 0, 0, 0]: -0.629074
Tensor value at [5, 0, 0, 0]: -1.416134
Tensor value at [6, 0, 0, 0]: -1.861542
Tensor value at [7, 0, 0, 0]: -0.740779
Tensor value at [8, 0, 0, 0]: -0.659427
Tensor value at [9, 0, 0, 0]: 0.181663
pred_1_ih_b mean_sq = 0.5914398877
```

```console
bias_hh_l0 shape: torch.Size([2560])
bias_hh_l0: tensor([ 0.6425,  0.0227, -0.3591,  0.3803,  0.4799,  0.1192,  0.1688,  0.3171,
         0.4005,  0.2199])
bias_hh_l0 ms: 0.48959604766099324

bias_hh_l1 shape: torch.Size([2560])
bias_hh_l1: tensor([-1.0186,  0.2159,  0.0159, -0.1612, -0.6269, -1.4517, -1.8947, -0.7625,
        -0.6602,  0.1272])
bias_hh_l1 ms: 0.7729063754253067
```
```console
Tensor 'pred_0_hh_b', type: f32
ne = [2560 1 1 1]
nb = [4 10240 10240 10240]
Tensor value at [0, 0, 0, 0]: 0.642521
Tensor value at [1, 0, 0, 0]: 0.022742
Tensor value at [2, 0, 0, 0]: -0.359071
Tensor value at [3, 0, 0, 0]: 0.380312
Tensor value at [4, 0, 0, 0]: 0.479876
Tensor value at [5, 0, 0, 0]: 0.119183
Tensor value at [6, 0, 0, 0]: 0.168764
Tensor value at [7, 0, 0, 0]: 0.317118
Tensor value at [8, 0, 0, 0]: 0.400493
Tensor value at [9, 0, 0, 0]: 0.219857
pred_0_hh_b mean_sq = 0.4895960480

Tensor 'pred_1_hh_b', type: f32
ne = [2560 1 1 1]
nb = [4 10240 10240 10240]
Tensor value at [0, 0, 0, 0]: -1.018631
Tensor value at [1, 0, 0, 0]: 0.215855
Tensor value at [2, 0, 0, 0]: 0.015897
Tensor value at [3, 0, 0, 0]: -0.161164
Tensor value at [4, 0, 0, 0]: -0.626932
Tensor value at [5, 0, 0, 0]: -1.451684
Tensor value at [6, 0, 0, 0]: -1.894704
Tensor value at [7, 0, 0, 0]: -0.762498
Tensor value at [8, 0, 0, 0]: -0.660172
Tensor value at [9, 0, 0, 0]: 0.127235
pred_1_hh_b mean_sq = 0.7729063755
```

```console
weight_hh_l0 weight shape: torch.Size([2560, 640])
weight_hh_l0: tensor([-0.4790, -0.5023, -0.1944, -0.4421,  0.0749, -0.0917,  0.1450, -0.5983,
        -0.0511,  0.3650])
weight_hh_l0 ms: 0.09143777953907747

weight_hh_l1 weight shape: torch.Size([2560, 640])
weight_hh_l1: tensor([ 0.8828,  0.5131,  0.0236, -0.6477, -0.5317,  0.2694,  0.2595,  0.5725,
         0.3968,  0.0963])
weight_hh_l1 ms: 0.27883419977913043
```
```console
Tensor 'pred_0_hh_w', type: f32
ne = [640 2560 1 1]
nb = [4 2560 6553600 6553600]
Tensor value at [0, 0, 0, 0]: -0.479023
Tensor value at [1, 0, 0, 0]: -0.502337
Tensor value at [2, 0, 0, 0]: -0.194352
Tensor value at [3, 0, 0, 0]: -0.442080
Tensor value at [4, 0, 0, 0]: 0.074884
Tensor value at [5, 0, 0, 0]: -0.091718
Tensor value at [6, 0, 0, 0]: 0.145002
Tensor value at [7, 0, 0, 0]: -0.598290
Tensor value at [8, 0, 0, 0]: -0.051108
Tensor value at [9, 0, 0, 0]: 0.365014
pred_0_hh_w mean_sq = 0.0914377795

Tensor 'pred_1_hh_w', type: f32
ne = [640 2560 1 1]
nb = [4 2560 6553600 6553600]
Tensor value at [0, 0, 0, 0]: 0.882829
Tensor value at [1, 0, 0, 0]: 0.513058
Tensor value at [2, 0, 0, 0]: 0.023595
Tensor value at [3, 0, 0, 0]: -0.647707
Tensor value at [4, 0, 0, 0]: -0.531749
Tensor value at [5, 0, 0, 0]: 0.269398
Tensor value at [6, 0, 0, 0]: 0.259524
Tensor value at [7, 0, 0, 0]: 0.572463
Tensor value at [8, 0, 0, 0]: 0.396843
Tensor value at [9, 0, 0, 0]: 0.096255
pred_1_hh_w mean_sq = 0.2788341998
```
And the hidden to hidden weights also look good.


```console

```
