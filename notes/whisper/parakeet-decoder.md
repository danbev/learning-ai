## Parakeet decoder notes

### Decoder implementation
```console
(Pdb) b ../NeMo/nemo/collections/common/parts/rnn.py:247
Breakpoint 1 at /home/danbev/work/ai/whisper-models/nvidia/NeMo/nemo/collections/common/parts/rnn.py:247
```

### Tensors
Original model:
```console
--------- LSTM ------------
weight_hh_l0 weight shape: torch.Size([2560, 640])
weight_hh_l0: tensor([-0.4790, -0.5023, -0.1944, -0.4421,  0.0749, -0.0917,  0.1450, -0.5983,
        -0.0511,  0.3650])
weight_hh_l0 ms: 0.09143777953907747

weight_hh_l1 weight shape: torch.Size([2560, 640])
weight_hh_l1: tensor([ 0.8828,  0.5131,  0.0236, -0.6477, -0.5317,  0.2694,  0.2595,  0.5725,
         0.3968,  0.0963])
weight_hh_l1 ms: 0.27883419977913043

weight_ih_l0 weight shape: torch.Size([2560, 640])
weight_ih_l0: tensor([ 0.1676,  0.1051, -0.2337, -0.1759,  0.2805, -0.0289,  0.3011,  0.2723,
         0.4243,  0.3569])
weight_ih_l0 ms: 0.0863738453256456

weight_ih_l1 weight shape: torch.Size([2560, 640])
weight_ih_l1: tensor([ 0.0751,  0.0539,  0.0842, -0.0482,  0.0227,  0.0177, -0.0183, -0.0230,
        -0.1247, -0.0111])
weight_ih_l1 ms: 0.01959334758041556

x shape: torch.Size([1, 1, 640])
x: tensor([-0.0059, -0.0164, -0.0030, -0.0016,  0.0003,  0.0064,  0.0060, -0.0049,
        -0.0020, -0.0124])
x ms: 0.0015669984467194518
h shape: (tensor([[[-0.0321,  0.0684, -0.0235,  ...,  0.1899,  0.1759, -0.0167]],

        [[-0.0059, -0.0164, -0.0030,  ...,  0.0092, -0.0012, -0.0015]]]), tensor([[[-0.0639,  0.1335, -0.0534,  ...,  0.3910,  0.4361, -0.0316]],

        [[-0.0473, -0.4182, -0.0692,  ...,  0.2822, -0.7955, -0.0050]]]))
g (embed) shape: torch.Size([1, 1, 640])
g: tensor([-0.0059, -0.0164, -0.0030, -0.0016,  0.0003,  0.0064,  0.0060, -0.0049,
        -0.0020, -0.0124])
g ms: 0.0015669984467194518
```
Converted model:
```console
Tensor 'lstm_0_hh_w', type: f32
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
lstm_0_hh_w mean_sq = 0.0914377795

Tensor 'lstm_0_ih_b', type: f32
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
lstm_0_ih_b mean_sq = 0.4895960480

Tensor 'lstm_0_ih_w', type: f32
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
lstm_0_ih_w mean_sq = 0.0863738453

Tensor 'lstm_0_ih_b', type: f32
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
lstm_0_ih_b mean_sq = 0.1582143676

Tensor 'layer_0_input', type: f32
ne = [640 1 1 1]
nb = [4 2560 2560 2560]
Tensor value at [0, 0, 0, 0]: 0.127817
Tensor value at [1, 0, 0, 0]: -0.428602
Tensor value at [2, 0, 0, 0]: 0.159392
Tensor value at [3, 0, 0, 0]: 0.531440
Tensor value at [4, 0, 0, 0]: 0.177826
Tensor value at [5, 0, 0, 0]: -0.468060
Tensor value at [6, 0, 0, 0]: 0.650171
Tensor value at [7, 0, 0, 0]: 0.909023
Tensor value at [8, 0, 0, 0]: -0.437073
Tensor value at [9, 0, 0, 0]: 0.954400
layer_0_input mean_sq = 0.7234731143

Tensor 'layer_0_input_gates', type: f32
ne = [2560 1 1 1]
nb = [4 10240 10240 10240]
Tensor value at [0, 0, 0, 0]: -0.604602
Tensor value at [1, 0, 0, 0]: 8.241313
Tensor value at [2, 0, 0, 0]: -0.446679
Tensor value at [3, 0, 0, 0]: -2.662867
Tensor value at [4, 0, 0, 0]: -11.161223
Tensor value at [5, 0, 0, 0]: -0.801732
Tensor value at [6, 0, 0, 0]: 3.463732
Tensor value at [7, 0, 0, 0]: 1.520854
Tensor value at [8, 0, 0, 0]: -12.058603
Tensor value at [9, 0, 0, 0]: 5.774023
layer_0_input_gates mean_sq = 38.7930748049
```
_wip_

