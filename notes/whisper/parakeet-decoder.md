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
