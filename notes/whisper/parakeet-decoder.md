## Parakeet decoder notes

### Decoder implementation
```console
(Pdb) b ../NeMo/nemo/collections/common/parts/rnn.py:247
Breakpoint 1 at /home/danbev/work/ai/whisper-models/nvidia/NeMo/nemo/collections/common/parts/rnn.py:247
```

### Tensors weights
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

### Tensor operations
The first input in the original model is initial token is the blank token, which
has token id 8192:
This can be seen in nemo/collections/asr/modules/rnnt.py
```python
    def predict(
        self,
        y: Optional[torch.Tensor] = None,
        state: Optional[List[torch.Tensor]] = None,
        add_sos: bool = True,
        batch_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    ...
        if y is not None:
            if y.device != device:
                y = y.to(device)

            # (B, U) -> (B, U, H)
            y = self.prediction["embed"](y)
```
```console
(Pdb) p y
tensor([[8192]])
(Pdb) n
> /home/danbev/work/ai/whisper-models/nvidia/NeMo/nemo/collections/asr/modules/rnnt.py(775)predict()
-> print(f"y (embed) shape: {y.shape}")
(Pdb) p y
tensor([[[0., 0., 0., 0., 0...]]])
(Pdb) p y.shape
torch.Size([1, 1, 640])
```
We also set use the blank token and our initial embedding vector is also all
zeros:
```console
Tensor 'lstm_layer_0_x_t', type: f32
ne = [640 1 1 1]
nb = [4 2560 2560 2560]
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
lstm_layer_0_x_t mean_sq = 0.0000000000
```
After the first LSTM layers have been processed the hidden states look like this:
```console
updated h_state_0 shape: torch.Size([2, 1, 640])
updated h_state: tensor([-0.0321,  0.0684, -0.0235, -0.2065,  0.1984,  0.1661,  0.1775,  0.2153, 0.0759, -0.2032])
updated h_state_0: tensor([-0.0321,  0.0684, -0.0235, -0.2065,  0.1984,  0.1661,  0.1775,  0.2153, 0.0759, -0.2032])
updated h_state_1: tensor([-0.0059, -0.0164, -0.0030, -0.0016,  0.0003,  0.0064,  0.0060, -0.0049, -0.0020, -0.0124])
updated h_state_0: ms: tensor([[0.0349], [0.0016]], dtype=torch.float64)
```
```console
Tensor 'lstm_layer_0_h_state', type: f32
ne = [640 1 1 1]
nb = [4 2560 2560 2560]
Tensor value at [0, 0, 0, 0]: -0.032146
Tensor value at [1, 0, 0, 0]: 0.068369
Tensor value at [2, 0, 0, 0]: -0.023465
Tensor value at [3, 0, 0, 0]: -0.206481
Tensor value at [4, 0, 0, 0]: 0.198382
Tensor value at [5, 0, 0, 0]: 0.166126
Tensor value at [6, 0, 0, 0]: 0.177529
Tensor value at [7, 0, 0, 0]: 0.215326
Tensor value at [8, 0, 0, 0]: 0.075932
Tensor value at [9, 0, 0, 0]: -0.203249
lstm_layer_0_h_state mean_sq = 0.0348651816

Tensor 'lstm_layer_1_h_state', type: f32
ne = [640 1 1 1]
nb = [4 2560 2560 2560]
Tensor value at [0, 0, 0, 0]: -0.005916
Tensor value at [1, 0, 0, 0]: -0.016424
Tensor value at [2, 0, 0, 0]: -0.002994
Tensor value at [3, 0, 0, 0]: -0.001601
Tensor value at [4, 0, 0, 0]: 0.000315
Tensor value at [5, 0, 0, 0]: 0.006404
Tensor value at [6, 0, 0, 0]: 0.005976
Tensor value at [7, 0, 0, 0]: -0.004926
Tensor value at [8, 0, 0, 0]: -0.002023
Tensor value at [9, 0, 0, 0]: -0.012439
lstm_layer_1_h_state mean_sq = 0.0015669989
```

```console
updated c_state shape: torch.Size([2, 1, 640])
updated c_state: tensor([-0.0639,  0.1335, -0.0534, -0.4498,  0.3495,  0.3189,  0.2865,  0.3390, 0.1589, -0.3680])
updated c_state: tensor([-0.0473, -0.4182, -0.0692, -0.0091,  0.0013,  0.0064,  0.0060, -0.0203, -0.2083, -0.0854])
updated c_state: ms: tensor([[0.1245], [0.1986]], dtype=torch.float64)
```

```console
Tensor 'lstm_layer_0_c_state', type: f32
ne = [640 1 1 1]
nb = [4 2560 2560 2560]
Tensor value at [0, 0, 0, 0]: -0.063912
Tensor value at [1, 0, 0, 0]: 0.133550
Tensor value at [2, 0, 0, 0]: -0.053388
Tensor value at [3, 0, 0, 0]: -0.449848
Tensor value at [4, 0, 0, 0]: 0.349465
Tensor value at [5, 0, 0, 0]: 0.318875
Tensor value at [6, 0, 0, 0]: 0.286453
Tensor value at [7, 0, 0, 0]: 0.339033
Tensor value at [8, 0, 0, 0]: 0.158884
Tensor value at [9, 0, 0, 0]: -0.368004
lstm_layer_0_c_state mean_sq = 0.1245070663
Tensor 'lstm_layer_1_c_state', type: f32

ne = [640 1 1 1]
nb = [4 2560 2560 2560]
Tensor value at [0, 0, 0, 0]: -0.047322
Tensor value at [1, 0, 0, 0]: -0.418231
Tensor value at [2, 0, 0, 0]: -0.069170
Tensor value at [3, 0, 0, 0]: -0.009116
Tensor value at [4, 0, 0, 0]: 0.001291
Tensor value at [5, 0, 0, 0]: 0.006406
Tensor value at [6, 0, 0, 0]: 0.005984
Tensor value at [7, 0, 0, 0]: -0.020290
Tensor value at [8, 0, 0, 0]: -0.208279
Tensor value at [9, 0, 0, 0]: -0.085378
lstm_layer_1_c_state mean_sq = 0.1985601852
```
So these don't match either. Lets add some logging to the lstm layer to see
the intermediate values:
```console
gates (combined): tensor([ 1.2573,  0.0429, -0.7146,  0.8014,  0.9815,  0.2113,  0.3437,  0.6104, 0.7935,  0.4730])
i_t: tensor([0.7786, 0.5107, 0.3286, 0.6903, 0.7274, 0.5526, 0.5851, 0.6480, 0.6886, 0.6161])
f_t: tensor([0.1535, 0.3506, 0.1740, 0.3476, 0.1536, 0.1700, 0.1236, 0.2296, 0.3165, 0.1158])
g_t: tensor([-0.0821,  0.2615, -0.1625, -0.6517,  0.4804,  0.5770,  0.4896,  0.5232, 0.2307, -0.5973])
o_t: tensor([0.5037, 0.5150, 0.4399, 0.4896, 0.5906, 0.5385, 0.6366, 0.6593, 0.4819, 0.5770])
c_new: tensor([-0.0639,  0.1335, -0.0534, -0.4498,  0.3495,  0.3189,  0.2865,  0.3390, 0.1589, -0.3680])
h_new: tensor([-0.0321,  0.0684, -0.0235, -0.2065,  0.1984,  0.1661,  0.1775,  0.2153, 0.0759, -0.2032])

gates ms: 1.10931396484375
i_t ms: 0.39431747794151306
f_t ms: 0.05079362541437149
g_t ms: 0.2744366526603699
o_t ms: 0.28297215700149536
```
```console
Tensor 'lstm_layer_0_gates', type: f32
ne = [2560 1 1 1]
nb = [4 10240 10240 10240]
Tensor value at [0, 0, 0, 0]: 1.257346
Tensor value at [1, 0, 0, 0]: 0.042891
Tensor value at [2, 0, 0, 0]: -0.714631
Tensor value at [3, 0, 0, 0]: 0.801442
Tensor value at [4, 0, 0, 0]: 0.981473
Tensor value at [5, 0, 0, 0]: 0.211334
Tensor value at [6, 0, 0, 0]: 0.343676
Tensor value at [7, 0, 0, 0]: 0.610449
Tensor value at [8, 0, 0, 0]: 0.793519
Tensor value at [9, 0, 0, 0]: 0.472975
lstm_layer_0_gates mean_sq = 1.1093139611

Tensor 'lstm_layer_0_i_t', type: f32
ne = [640 1 1 1]
nb = [4 2560 2560 2560]
Tensor value at [0, 0, 0, 0]: 0.778569
Tensor value at [1, 0, 0, 0]: 0.510721
Tensor value at [2, 0, 0, 0]: 0.328576
Tensor value at [3, 0, 0, 0]: 0.690283
Tensor value at [4, 0, 0, 0]: 0.727400
Tensor value at [5, 0, 0, 0]: 0.552638
Tensor value at [6, 0, 0, 0]: 0.585083
Tensor value at [7, 0, 0, 0]: 0.648043
Tensor value at [8, 0, 0, 0]: 0.688587
Tensor value at [9, 0, 0, 0]: 0.616088
lstm_layer_0_i_t mean_sq = 0.3943174750

Tensor 'lstm_layer_0_f_t', type: f32
ne = [640 1 1 1]
nb = [4 2560 2560 2560]
Tensor value at [0, 0, 0, 0]: 0.153477
Tensor value at [1, 0, 0, 0]: 0.350646
Tensor value at [2, 0, 0, 0]: 0.174042
Tensor value at [3, 0, 0, 0]: 0.347595
Tensor value at [4, 0, 0, 0]: 0.153600
Tensor value at [5, 0, 0, 0]: 0.169978
Tensor value at [6, 0, 0, 0]: 0.123645
Tensor value at [7, 0, 0, 0]: 0.229634
Tensor value at [8, 0, 0, 0]: 0.316538
Tensor value at [9, 0, 0, 0]: 0.115825
lstm_layer_0_f_t mean_sq = 0.0507936280

Tensor 'lstm_layer_0_c_t', type: f32
ne = [640 1 1 1]
nb = [4 2560 2560 2560]
Tensor value at [0, 0, 0, 0]: -0.082089
Tensor value at [1, 0, 0, 0]: 0.261492
Tensor value at [2, 0, 0, 0]: -0.162481
Tensor value at [3, 0, 0, 0]: -0.651686
Tensor value at [4, 0, 0, 0]: 0.480430
Tensor value at [5, 0, 0, 0]: 0.577005
Tensor value at [6, 0, 0, 0]: 0.489593
Tensor value at [7, 0, 0, 0]: 0.523164
Tensor value at [8, 0, 0, 0]: 0.230740
Tensor value at [9, 0, 0, 0]: -0.597324
lstm_layer_0_c_t mean_sq = 0.2744366045

Tensor 'lstm_layer_0_o_t', type: f32
ne = [640 1 1 1]
nb = [4 2560 2560 2560]
Tensor value at [0, 0, 0, 0]: 0.503661
Tensor value at [1, 0, 0, 0]: 0.514978
Tensor value at [2, 0, 0, 0]: 0.439947
Tensor value at [3, 0, 0, 0]: 0.489554
Tensor value at [4, 0, 0, 0]: 0.590596
Tensor value at [5, 0, 0, 0]: 0.538513
Tensor value at [6, 0, 0, 0]: 0.636610
Tensor value at [7, 0, 0, 0]: 0.659268
Tensor value at [8, 0, 0, 0]: 0.481923
Tensor value at [9, 0, 0, 0]: 0.577011
lstm_layer_0_o_t mean_sq = 0.2829721501
```

And the new embedding vector is:
```console
new x shape: torch.Size([1, 1, 640])
new x: tensor([-0.0059, -0.0164, -0.0030, -0.0016,  0.0003,  0.0064,  0.0060, -0.0049, -0.0020, -0.0124])
new x: ms: 0.0015669984467194518
```
```console
(gdb) n
Tensor 'lstm_pred_out', type: f32
ne = [640 1 1 1]
nb = [4 2560 2560 2560]
Tensor value at [0, 0, 0, 0]: -0.005916
Tensor value at [1, 0, 0, 0]: -0.016424
Tensor value at [2, 0, 0, 0]: -0.002994
Tensor value at [3, 0, 0, 0]: -0.001601
Tensor value at [4, 0, 0, 0]: 0.000315
Tensor value at [5, 0, 0, 0]: 0.006404
Tensor value at [6, 0, 0, 0]: 0.005976
Tensor value at [7, 0, 0, 0]: -0.004926
Tensor value at [8, 0, 0, 0]: -0.002023
Tensor value at [9, 0, 0, 0]: -0.012439
lstm_pred_out mean_sq = 0.0015669989
```
So the lstm network looks like it is working correctly.
