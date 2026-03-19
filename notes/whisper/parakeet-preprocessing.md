## Parakeet preprocessing notes


### Tensor/Operation comparision
Pytorch:
```console
input shape: torch.Size([1, 1101, 128])

input: tensor([-2.1009, -3.3455, -2.7749, -4.7532, -4.3730, -5.6027, -4.7664, -4.5884,
        -3.5901, -3.6141])
input ms: 0.9981744415301794

---------------------------------------------
Before Layer 0 Conv2d: x.shape: torch.Size([1, 1, 1101, 128]), op: Conv2d(1, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
x shape: torch.Size([1, 1, 1101, 128])
x: tensor([-2.1009, -3.3455, -2.7749, -4.7532, -4.3730, -5.6027, -4.7664, -4.5884,
        -3.5901, -3.6141])
x ms: 0.9981744415301794
torch.Size([256, 1, 3, 3])
tensor([ 0.0964,  0.0259, -0.4867, -0.2263, -0.3193,  1.1505, -0.0091,  0.1196,
         0.0085,  0.0586])
0.19608584626767314
tensor([-3.4104e-01,  2.1441e-01, -7.0875e-01, -5.4349e-01, -4.5362e-01,
        -6.2717e-02, -5.1516e-01, -2.3044e-01,  9.4809e-06, -3.9431e-03])
0.09868425240328094
After Layer 0: x shape torch.Size([1, 256, 551, 64])
x shape: torch.Size([1, 256, 551, 64])
x: tensor([-3.7990, -4.5085, -4.8426, -3.3884, -2.7329, -3.0468, -3.6646, -2.8045,
        -2.7402, -2.3244])
x ms: 0.6077475048109069
---------------------------------------------


---------------------------------------------
Before Layer 1 ReLU: x.shape: torch.Size([1, 256, 551, 64]), op: ReLU(inplace=True)
x shape: torch.Size([1, 256, 551, 64])
x: tensor([-3.7990, -4.5085, -4.8426, -3.3884, -2.7329, -3.0468, -3.6646, -2.8045,
        -2.7402, -2.3244])
x ms: 0.6072816928122464
After Layer 1: x shape torch.Size([1, 256, 551, 64])
x shape: torch.Size([1, 256, 551, 64])
x: tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
x ms: 0.24929871292547623
---------------------------------------------


---------------------------------------------
Before Layer 2 Conv2d: x.shape: torch.Size([1, 256, 551, 64]), op: Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
x shape: torch.Size([1, 256, 551, 64])
x: tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
x ms: 0.24929871292547623
torch.Size([256, 1, 3, 3])
tensor([ 0.0998,  0.2409,  0.0960, -0.2674, -0.9655, -0.8366,  0.2090,  0.2630,
         0.0761, -0.3445])
0.2360849789271445
tensor([ 0.0029,  0.2550, -0.0309,  0.0025,  0.0747,  0.0200,  0.0216, -0.0125,
        -0.0579,  0.0660])
0.008804818570572236
After Layer 2: x shape torch.Size([1, 256, 276, 32])
x shape: torch.Size([1, 256, 276, 32])
x: tensor([0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029,
        0.0029])
x ms: 0.5077442316707769
---------------------------------------------


---------------------------------------------
Before Layer 3 Conv2d: x.shape: torch.Size([1, 256, 276, 32]), op: Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
x shape: torch.Size([1, 256, 276, 32])
x: tensor([0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029,
        0.0029])
x ms: 0.5073748530508311
torch.Size([256, 256, 1, 1])
tensor([ 0.0406,  0.4173,  0.5180,  1.5758,  1.1877, -0.9608, -0.5145, -0.1828,
        -0.0264, -0.0442])
0.25640109406762873
tensor([ 0.6143,  1.5695,  0.1309,  0.0189,  0.1805, -0.1099,  0.1983,  0.2063,
        -0.0342, -0.3553])
0.8843241577090166
After Layer 3: x shape torch.Size([1, 256, 276, 32])
x shape: torch.Size([1, 256, 276, 32])
x: tensor([ -8.3362, -25.2448, -26.4410, -23.6129, -21.3224, -18.5399, -17.5652,
        -18.3385, -17.7654, -17.1464])
x ms: 149.37329453965867
---------------------------------------------


---------------------------------------------
Before Layer 4 ReLU: x.shape: torch.Size([1, 256, 276, 32]), op: ReLU(inplace=True)
x shape: torch.Size([1, 256, 276, 32])
x: tensor([ -8.3362, -25.2448, -26.4410, -23.6129, -21.3224, -18.5399, -17.5652,
        -18.3385, -17.7654, -17.1464])
x ms: 149.37009046662348
After Layer 4: x shape torch.Size([1, 256, 276, 32])
x shape: torch.Size([1, 256, 276, 32])
x: tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
x ms: 12.686015406319461
---------------------------------------------


---------------------------------------------
Before Layer 5 Conv2d: x.shape: torch.Size([1, 256, 276, 32]), op: Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
x shape: torch.Size([1, 256, 276, 32])
x: tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
x ms: 12.686015406319461
torch.Size([256, 1, 3, 3])
tensor([ 0.4176,  0.2623, -0.0297,  0.1375, -0.0719, -0.1959, -0.0702, -0.1433,
        -0.1219,  0.0293])
0.15548679877001184
tensor([-0.1529,  1.0856,  0.9887,  0.7857,  0.1514,  0.4395, -0.0319, -0.3846,
        -0.3217,  0.2238])
0.3769830744735316
After Layer 5: x shape torch.Size([1, 256, 138, 16])
x shape: torch.Size([1, 256, 138, 16])
x: tensor([-0.1529, -0.1529, -0.1529, -0.1529, -0.1529, -0.1529, -0.1529, -0.1529,
        -0.1529, -0.1529])
x ms: 8.025787332469191
---------------------------------------------


---------------------------------------------
Before Layer 6 Conv2d: x.shape: torch.Size([1, 256, 138, 16]), op: Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
x shape: torch.Size([1, 256, 138, 16])
x: tensor([-0.1529, -0.1529, -0.1529, -0.1529, -0.1529, -0.1529, -0.1529, -0.1529,
        -0.1529, -0.1529])
x ms: 8.025787332469191
torch.Size([256, 256, 1, 1])
tensor([-0.9037, -0.1159,  0.0021, -0.5367, -0.1100, -0.0310,  0.6205, -0.0788,
         0.0912,  0.0887])
0.23345056412277246
tensor([ 0.5281, -0.0186, -1.5426, -0.4881,  1.1282, -0.0297,  0.4590, -0.8397,
         0.0881, -0.4935])
0.5569255296500295
After Layer 6: x shape torch.Size([1, 256, 138, 16])
x shape: torch.Size([1, 256, 138, 16])
x: tensor([-34.3560, -42.0583, -27.8299, -23.4590, -22.6265, -21.5393, -21.6146,
        -21.7026, -20.6098, -21.3324])
x ms: 995.8196824710096
---------------------------------------------


---------------------------------------------
Before Layer 7 ReLU: x.shape: torch.Size([1, 256, 138, 16]), op: ReLU(inplace=True)
x shape: torch.Size([1, 256, 138, 16])
x: tensor([-34.3560, -42.0583, -27.8299, -23.4590, -22.6265, -21.5393, -21.6146,
        -21.7026, -20.6098, -21.3324])
x ms: 995.8196824710096
After Layer 7: x shape torch.Size([1, 256, 138, 16])
x shape: torch.Size([1, 256, 138, 16])
x: tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
x ms: 76.81944460928672
---------------------------------------------

---------------------------------------------

After pre_encode: audio_signal shape torch.Size([1, 138, 1024])
out shape: torch.Size([1024, 4096])
out: tensor([-0.4425,  0.2388, -0.3707,  0.2516,  0.0822, -0.0622,  0.4018,  0.0587,
         0.0859, -0.3840])
out ms: 0.1556534332707742
out shape: torch.Size([1024])
out: tensor([ 0.6330, -0.7415, -0.0210, -0.3041, -0.5254,  0.2312,  0.2591,  0.3598,
         0.5704,  0.2590])
out ms: 0.3461039746401098
audio_signal shape: torch.Size([1, 138, 1024])
audio_signal: tensor([ 487.9199, -313.3978,  109.8925,  422.0488,   20.6455,  106.9309,
         467.3413, -125.1918,  108.5956,  317.1499])
audio_signal ms: 139228.13879601416
---------------------------------------------

```

Parakeet.cpp:
```console
Tensor 'mel', type: f32
ne = [128 1101 1 1]
Tensor value at [0, 0, 0, 0]: -2.100924
Tensor value at [1, 0, 0, 0]: -3.345516
Tensor value at [2, 0, 0, 0]: -2.774928
Tensor value at [3, 0, 0, 0]: -4.753236
Tensor value at [4, 0, 0, 0]: -4.373011
Tensor value at [5, 0, 0, 0]: -5.602739
Tensor value at [6, 0, 0, 0]: -4.766403
Tensor value at [7, 0, 0, 0]: -4.588367
Tensor value at [8, 0, 0, 0]: -3.590068
Tensor value at [9, 0, 0, 0]: -3.614062
mel mean_sq = 0.9987380058
Tensor 'pre_conv_0', type: f32
ne = [64 551 256 1]
Tensor value at [0, 0, 0, 0]: -3.798953
Tensor value at [1, 0, 0, 0]: -4.508554
Tensor value at [2, 0, 0, 0]: -4.842618
Tensor value at [3, 0, 0, 0]: -3.388401
Tensor value at [4, 0, 0, 0]: -2.732860
Tensor value at [5, 0, 0, 0]: -3.046798
Tensor value at [6, 0, 0, 0]: -3.664583
Tensor value at [7, 0, 0, 0]: -2.804516
Tensor value at [8, 0, 0, 0]: -2.740154
Tensor value at [9, 0, 0, 0]: -2.324420
pre_conv_0 mean_sq = 0.6081864728
Tensor 'pre_conv_0_relu', type: f32
ne = [64 551 256 1]
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
pre_conv_0_relu mean_sq = 0.2496694243
Tensor 'pre_conv_2', type: f32
ne = [32 276 256 1]
Tensor value at [0, 0, 0, 0]: 0.002926
Tensor value at [1, 0, 0, 0]: 0.002926
Tensor value at [2, 0, 0, 0]: 0.002926
Tensor value at [3, 0, 0, 0]: 0.002926
Tensor value at [4, 0, 0, 0]: 0.002926
Tensor value at [5, 0, 0, 0]: 0.002926
Tensor value at [6, 0, 0, 0]: 0.002926
Tensor value at [7, 0, 0, 0]: 0.002926
Tensor value at [8, 0, 0, 0]: 0.002926
Tensor value at [9, 0, 0, 0]: 0.002926
pre_conv_2 mean_sq = 0.5085358742
Tensor 'pre_conv_3', type: f32
ne = [32 276 256 1]
Tensor value at [0, 0, 0, 0]: -8.336175
Tensor value at [1, 0, 0, 0]: -25.244816
Tensor value at [2, 0, 0, 0]: -26.440964
Tensor value at [3, 0, 0, 0]: -23.612862
Tensor value at [4, 0, 0, 0]: -21.322363
Tensor value at [5, 0, 0, 0]: -18.539936
Tensor value at [6, 0, 0, 0]: -17.565189
Tensor value at [7, 0, 0, 0]: -18.338491
Tensor value at [8, 0, 0, 0]: -17.765369
Tensor value at [9, 0, 0, 0]: -17.146439
pre_conv_3 mean_sq = 149.6801475948
Tensor 'pre_conv_3_relu', type: f32
ne = [32 276 256 1]
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
pre_conv_3_relu mean_sq = 12.7164772143
Tensor 'pre_conv_5', type: f32
ne = [16 138 256 1]
Tensor value at [0, 0, 0, 0]: -0.152901
Tensor value at [1, 0, 0, 0]: -0.152901
Tensor value at [2, 0, 0, 0]: -0.152901
Tensor value at [3, 0, 0, 0]: -0.152901
Tensor value at [4, 0, 0, 0]: -0.152901
Tensor value at [5, 0, 0, 0]: -0.152901
Tensor value at [6, 0, 0, 0]: -0.152901
Tensor value at [7, 0, 0, 0]: -0.152901
Tensor value at [8, 0, 0, 0]: -0.152901
Tensor value at [9, 0, 0, 0]: -0.152901
pre_conv_5 mean_sq = 8.0371412658
Tensor 'pre_conv_6', type: f32
ne = [16 138 256 1]
Tensor value at [0, 0, 0, 0]: -34.356064
Tensor value at [1, 0, 0, 0]: -42.058315
Tensor value at [2, 0, 0, 0]: -27.829866
Tensor value at [3, 0, 0, 0]: -23.459055
Tensor value at [4, 0, 0, 0]: -22.626474
Tensor value at [5, 0, 0, 0]: -21.539301
Tensor value at [6, 0, 0, 0]: -21.614588
Tensor value at [7, 0, 0, 0]: -21.702559
Tensor value at [8, 0, 0, 0]: -20.609838
Tensor value at [9, 0, 0, 0]: -21.332411
pre_conv_6 mean_sq = 997.7338405091
Tensor 'pre_conv_6_relu', type: f32
ne = [16 138 256 1]
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
pre_conv_6_relu mean_sq = 76.9038551885
Tensor 'enc_pre_out_w', type: f32
ne = [4096 1024 1 1]
Tensor value at [0, 0, 0, 0]: -0.442476
Tensor value at [1, 0, 0, 0]: 0.238827
Tensor value at [2, 0, 0, 0]: -0.370669
Tensor value at [3, 0, 0, 0]: 0.251574
Tensor value at [4, 0, 0, 0]: 0.082155
Tensor value at [5, 0, 0, 0]: -0.062154
Tensor value at [6, 0, 0, 0]: 0.401806
Tensor value at [7, 0, 0, 0]: 0.058698
Tensor value at [8, 0, 0, 0]: 0.085908
Tensor value at [9, 0, 0, 0]: -0.384031
enc_pre_out_w mean_sq = 0.1556534333
Tensor 'enc_pre_out_b', type: f32
ne = [1024 1 1 1]
Tensor value at [0, 0, 0, 0]: 0.633022
Tensor value at [1, 0, 0, 0]: -0.741507
Tensor value at [2, 0, 0, 0]: -0.021015
Tensor value at [3, 0, 0, 0]: -0.304058
Tensor value at [4, 0, 0, 0]: -0.525447
Tensor value at [5, 0, 0, 0]: 0.231219
Tensor value at [6, 0, 0, 0]: 0.259149
Tensor value at [7, 0, 0, 0]: 0.359799
Tensor value at [8, 0, 0, 0]: 0.570433
Tensor value at [9, 0, 0, 0]: 0.258957
enc_pre_out_b mean_sq = 0.3461039739
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
```
```console
audio_signal shape: torch.Size([1, 138, 1024])
audio_signal: tensor([ 487.9199, -313.3978, 109.8925, 422.0488, 20.6455, 106.9309, 467.3413, -125.1918, 108.5956, 317.1499])
audio_signal ms: 139228.13879601416
```
And this matches quite closely. So the pre-encoder layer now match.


### Misc debugging notes

```console
(Pdb) b venv/lib/python3.13/site-packages/nemo/collections/asr/parts/preprocessing/features.py:414
Breakpoint 8 at /Users/danbev/work/ai/whisper-models/nvidia/parakeet-tdt-0.6b-v3/venv/lib/python3.13/site-packages/nemo/collections/asr/parts/preprocessing/features.py:414
```

### window function
```console
(Pdb) p self.window.shape
torch.Size([400])

(Pdb) (self.window.double()**2).mean().item()
0.374062509671165
```
```console
window func size: 400:
0.000000 0.000062 0.000248 0.000558 0.000992 0.001549 0.002230 0.003035 0.003962 0.005013
Window func sum of squares = 0.374063
```


```console
(Pdb) p x.shape
torch.Size([1, 176000])

(Pdb) (x.double()**2).mean().item()
0.020192833500258116
```
In parakeet.cpp we the input sum of squares is:
```console
Input sum_sq = 0.020193
```

(Pdb) p seq_len
tensor([176000], dtype=torch.int32)

(Pdb) p self.n_fft
512
```
So seq_len (176000) is the length of the original input signal.


```python
    def forward(self, x, seq_len, linear_spec=False):
        seq_len_time = seq_len
        seq_len_unfixed = self.get_seq_len(seq_len)
```

So this is caculating the number of times steps (time frames) that we will have
after the STFT.
```python
    def get_seq_len(self, seq_len):
        # Assuming that center is True is stft_pad_amount = 0
        pad_amount = self.stft_pad_amount * 2 if self.stft_pad_amount is not None else self.n_fft // 2 * 2
        seq_len = torch.floor_divide((seq_len + pad_amount - self.n_fft), self.hop_length)
        return seq_len.to(dtype=torch.long)
```
```console
(Pdb) p self.stft_pad_amount
None
```
So the above wiill take (self.n_fft // 2) * 2. This is calculating the the number
of frames that will result from the STFT.

```console
(Pdb) p pad_amount
512

(Pdb) p self.hop_length
160

(Pdb) p seq_len.item()
1100

(Pdb) p seq_len + pad_amount
tensor([176512], dtype=torch.int32)
```
This the required padding for STFT which recall is because the normal FFT assumes
that the signal is periodic and so we pad the signal to make it appear periodic.

And hop_lenght is the number of samples the window hops/slides forward to compute
the next frame. 

So for 176000 samples we will 1100 hops which gives us 1100 time frames.

```python
        # fix for seq_len = 0 for streaming; if size was 0, it is always padded to 1, and normalizer fails
        seq_len = torch.where(seq_len == 0, torch.zeros_like(seq_len_unfixed), seq_len_unfixed)
```
Torch `where` is like a vectorized if/else statement for the entires tensors
that are involvedr, where(condition, input, output).

And `torch.zeros_like(seq_len_unfixed) will create a new tensor for the same
size and device as `seq_len_unfixed` but filled with zeros. In this case we
only have a scalar:
```console
(Pdb) p seq_len_unfixed
tensor([1100])

(Pdb) p torch.zeros_like(seq_len_unfixed)
tensor([0])
```

```python
        if self.stft_pad_amount is not None:
            x = torch.nn.functional.pad(
                x.unsqueeze(1), (self.stft_pad_amount, self.stft_pad_amount), "constant"
            ).squeeze(1)
```
```console
(Pdb) p self.stft_pad_amount
None
```
So the above will be skipped.

```python

        # dither (only in training mode for eval determinism)
        if self.training and self.dither > 0:
            x += self.dither * torch.randn_like(x)
```
```console
(Pdb) p self.training
False
(Pdb) p self.dither
0.0
```
So the above will be skipped be skipped too as we are not training.

```python
        # do preemphasis
        if self.preemph is not None:
            timemask = torch.arange(x.shape[1], device=x.device).unsqueeze(0) < seq_len_time.unsqueeze(1)
            x = torch.cat((x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1]), dim=1)
            x = x.masked_fill(~timemask, 0.0)
```
```console
(Pdb) p self.preemph
0.97

(Pdb) p x.shape[1]
176000

Before:
(Pdb) p x.view(-1)[0: 10]
tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
(Pdb) (x.double()**2).mean().item()
0.020192833500258116

After:
(Pdb) p x.view(-1)[0: 10]
tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
(Pdb) (x.double()**2).mean().item()
0.003075680178340922
```
And in parakeet.cpp we get:
```console
preemphasis sum_sq = 0.003076
```
So these match up well.

Next we have the STFT and this is where we can't really follow the code one to
one with parakeet.cpp.
```python

        # disable autocast to get full range of stft values
        with torch.amp.autocast(x.device.type, enabled=False):
            x = self.stft(x)
```
amp is Automatic Mixed Precision which is way to optimize operations where they
can be performed in lower precision. The autocase is enables torch to lower the
precision and casts in the block as it sees fit. And notice that this is
actually disabling the autocast feature. So this is where we perform the STFT.

But note that there is implicit padding in the stft function:
```python
    def stft(self, x):
        return torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=False if self.exact_pad else True,
            window=self.window.to(dtype=torch.float, device=x.device),
            return_complex=True,
            pad_mode="constant",
        )
```
Notice that `center=True` is used. This means that the signal is padded with
```console
(Pdb) p self.n_fft
512
(Pdb) p self.hop_length
160
(Pdb) p self.win_length
400
(Pdb) p self.exact_pad
False

Before:
(Pdb) p x.shape
torch.Size([1, 176000])
(Pdb) (x.double()**2).mean().item()
0.003075680178340922

After:
(Pdb) p x.shape
torch.Size([1, 257, 1101])
(Pdb) (x.double()**2).mean().item()
0.22117218886573428
(Pdb) (x.double()**2).mean().item()
0.22117218886573428
```
In parakeet.cpp:
```console

```

```python

        # torch stft returns complex tensor (of shape [B,N,T]); so convert to magnitude
        # guard is needed for sqrt if grads are passed through
        guard = 0 if not self.use_grads else CONSTANT
        x = torch.view_as_real(x)
        x = torch.sqrt(x.pow(2).sum(-1) + guard)
```
```console
(Pdb) p self.use_grads
False
(Pdb) p CONSTANT
1e-05
```
And the above is getting the real values of the complex values that the STFT
produced.
```console
(Pdb) p x.shape
torch.Size([1, 257, 1101])
```
And after the reals:
```console
(Pdb) p x.shape
torch.Size([1, 257, 1101, 2]
```
This row from above is doing an element square of all the elmenent in the x
tensor. And recall that 2 means there are two x-axis values, the real and the
imaginary part. Don't get fooled by the reverse order compared to ggml.
```python
        x = torch.sqrt(x.pow(2).sum(-1) + guard)
```
shape after this:
```console
(Pdb) p x.shape
torch.Size([1, 257, 1101])
            B   N    T
B = batch (1)
N = frequency bins (257)
T = time frames (1101)
```

We are not training so the following will be skipped:
```python
        if self.training and self.nb_augmentation_prob > 0.0:
            for idx in range(x.shape[0]):
                if self._rng.random() < self.nb_augmentation_prob:
                    x[idx, self._nb_max_fft_bin :, :] = 0.0
```
At this point x is the magnitudes of the STFT. This is the amplitude of each
frequency component. In physics the power (energy) of a wave is proportional to
the square of its amplitude. Many aspects of human hearning and sounds analysis
are more directly related to the enery of the sound than its raw amplitude.
So here we are converting the amplitude to power/energy:
```python
        # get power spectrum
        if self.mag_power != 1.0:
            x = x.pow(self.mag_power)
```
```console
(Pdb) p self.mag_power
2.0

DEBUG: Power spectrum before mel filterbank (frames 6-9, first 10 freq bins):
Frame 6: 3.2837e-06 4.2899e-06 7.4460e-06 1.1077e-05 1.4287e-05 1.8596e-05 2.6389e-05 3.6030e-05 4.1719e-05 4.4808e-05
Frame 7: 1.0337e-09 5.5690e-06 8.7599e-05 4.1695e-04 6.7910e-04 4.2742e-04 2.5819e-04 7.3727e-04 2.2563e-03 2.8787e-03
Frame 8: 1.1946e-06 2.0006e-06 2.2658e-05 2.3220e-04 3.3798e-04 4.6885e-04 4.4376e-04 6.5915e-04 1.7280e-03 3.7714e-03
Frame 9: 7.4403e-07 2.1905e-06 3.0375e-05 1.4913e-04 3.3051e-04 6.8874e-04 9.3089e-04 6.0567e-04 6.5043e-04 1.6719e-03
```
And in parakeet.cpp:
```console
Frame 6: 3.0991e-10 8.6247e-08 4.3268e-07 8.0876e-07 1.2299e-06 1.7309e-06 2.3642e-06 2.6058e-06 1.8166e-06 1.7204e-06
Frame 7: 8.4046e-07 1.0176e-05 7.9205e-05 2.2724e-04 2.9626e-04 1.6084e-04 5.8087e-05 4.1034e-04 1.0343e-03 1.1069e-03
Frame 8: 1.6822e-06 6.2448e-07 4.6362e-05 3.3973e-04 7.8540e-04 8.3309e-04 1.9151e-04 3.2344e-04 2.9435e-03 5.2613e-03
Frame 9: 6.9375e-07 4.1823e-06 8.6235e-06 1.4258e-04 2.0025e-04 2.8852e-04 1.0762e-03 7.2638e-04 1.9739e-04 1.8737e-03
```



linear_spec is False so this will be skipped:
```python
        # return plain spectrogram if required
        if linear_spec:
            return x, seq_len
```

Next we apply the filterbanks:
```python
        # disable autocast, otherwise it might be automatically casted to fp16
        # on fp16 compatible GPUs and get NaN values for input value of 65520
        with torch.amp.autocast(x.device.type, enabled=False):
            # dot with filterbank energies
            x = torch.matmul(self.fb.to(x.dtype), x)
```
```console
(Pdb) p self.fb.shape
torch.Size([1, 128, 257])

(Pdb) p self.fb.view(-1)[: 10]
tensor([-0.0000, 0.0284, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000])

(Pdb) p (self.fb.double()**2).mean().item()
2.194062021010566e-06

(before)
(Pdb) p x.shape
torch.Size([1, 257, 1101])
(Pdb) p x.view(-1)[:10]
tensor([0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.4556e-12, 5.9585e-10,
        3.2837e-06, 1.0337e-09, 1.1946e-06, 7.4403e-07])
(Pdb) p (x.double()**2).mean().item()
36.698161585839

(after)
(Pdb) p x.shape
torch.Size([1, 128, 1101])

(Pdb) p x.view(-1)[:10]
tensor([0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.6130e-13, 1.7577e-11,
        1.2174e-07, 1.5803e-07, 5.6772e-08, 6.2160e-08])

(Pdb) p (x.double()**2).mean().item()
0.05799734488294016
```
Next we have log:
```python
        # log features if required
        if self.log:
            if self.log_zero_guard_type == "add":
                x = torch.log(x + self.log_zero_guard_value_fn(x))
            elif self.log_zero_guard_type == "clamp":
                x = torch.log(torch.clamp(x, min=self.log_zero_guard_value_fn(x)))
            else:
                raise ValueError("log_zero_guard_type was not understood")
```

```python
    def log_zero_guard_value_fn(self, x):
        if isinstance(self.log_zero_guard_value, str):
            if self.log_zero_guard_value == "tiny":
                return torch.finfo(x.dtype).tiny
            elif self.log_zero_guard_value == "eps":
                return torch.finfo(x.dtype).eps
            else:
                raise ValueError(
                    f"{self} received {self.log_zero_guard_value} for the "
                    f"log_zero_guard_type parameter. It must be either a "
                    f"number, 'tiny', or 'eps'"
                )
        else:
            return self.log_zero_guard_value
```
So this will just return self.log_zero_guard_value which in our case is:
```console
(Pdb) p self.log_zero_guard_value
5.960464477539063e-08
```

```console
(Pdb) p self.log
True

(Pdb) p self.log_zero_guard_type
'add'

(Pdb) p x.view(-1)[:10]
tensor([-16.6355, -16.6355, -16.6355, -16.6355, -16.6355, -16.6352, -15.5229,
        -15.3404, -15.9664, -15.9212])
(Pdb) p (x.double()**2).mean().item()
102.66635666443821
````

Next we have frame splicing which is about combining multiple frames to
reduce the temporal resolution. This is a common technique in audio processing
but not active in this session.
```python
        # frame splicing if required
        if self.frame_splicing > 1:
            x = splice_frames(x, self.frame_splicing)
```

Next we have normalization:
```python
        # normalize if required
        if self.normalize:
            x, _, _ = normalize_batch(x, seq_len, normalize_type=self.normalize)
```
```console
(Pdb) p self.normalize
'per_feature'

(before)
(Pdb) p x.view(-1)[:10]
tensor([-16.6355, -16.6355, -16.6355, -16.6355, -16.6355, -16.6352, -15.5229, -15.3404, -15.9664, -15.9212])
(Pdb) p (x.double()**2).mean().item()
102.66635666443821

After normalization:
(Pdb) p x.view(-1)[:10]
tensor([-2.1009, -2.1009, -2.1009, -2.1009, -2.1009, -2.1006, -0.9609, -0.7740,
        -1.4154, -1.3690])
(Pdb) p (x.double()**2).mean().item()
0.9987379881895161
```
In parakeet.cpp we have:
```console
DEBUG: Mel spectrogram AFTER normalization:
-2.100924 -2.100924 -2.100924 -2.100924 -2.100922 -2.100621 -0.960900 -0.773955 -1.415364 -1.368993 
Mean of squares (all values) = 0.9987380057
```

```python
        # mask to zero any values beyond seq_len in batch, pad to multiple of `pad_to` (for efficiency)
        max_len = x.size(-1)
```
```console
(Pdb) p max_len
1101

(Pdb) p x.shape
torch.Size([1, 128, 1101])

(Pdb) p x.size(-1)
1101
```
```python
        mask = torch.arange(max_len, device=x.device)
```
```console
(Pdb) p mask
tensor([   0,    1,    2,  ..., 1098, 1099, 1100])
(Pdb) p mask.shape
torch.Size([1101])
```
```python
        mask = mask.repeat(x.size(0), 1) >= seq_len.unsqueeze(1)
        x = x.masked_fill(mask.unsqueeze(1).type(torch.bool).to(device=x.device), self.pad_value)
        del mask
```
```console
(Pdb) p x.view(-1)[:10]
tensor([-2.1009, -2.1009, -2.1009, -2.1009, -2.1009, -2.1006, -0.9609, -0.7740,
        -1.4154, -1.3690])
(Pdb) p (x.double()**2).mean().item()
0.9981744415301794
```
```python
        pad_to = self.pad_to
        if pad_to == "max":
            x = nn.functional.pad(x, (0, self.max_length - x.size(-1)), value=self.pad_value)
        elif pad_to > 0:
            pad_amt = x.size(-1) % pad_to
            if pad_amt != 0:
                x = nn.functional.pad(x, (0, pad_to - pad_amt), value=self.pad_value)
        return x, seq_len
```
Attributes:
```console
(Pdb) c
> /home/danbev/work/ai/whisper-models/nvidia/parkeet-tdt-0.6b-v3/venv/lib/python3.12/site-packages/nemo/collections/asr/parts/preprocessing/features.py(414)forward()

(Pdb) pp self.__dict__
{'_backward_hooks': OrderedDict(),
 '_backward_pre_hooks': OrderedDict(),
 '_buffers': {'fb': tensor([[[-0.0000, 0.0284, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0144, 0.0140,  ..., 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0288,  ..., 0.0000, 0.0000, 0.0000],
         ...,
         [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000,  ..., 0.0017, 0.0009, 0.0000]]]),
              'window': tensor([0.0000e+00, 6.1989e-05, 2.4796e-04, 5.5784e-04, 9.9158e-04, 1.5491e-03,
        2.2301e-03, 3.0347e-03, 3.9624e-03, 5.0132e-03, 6.1867e-03, 7.4826e-03,
        8.9007e-03, 1.0441e-02, 1.2102e-02, 1.3884e-02, 1.5787e-02, 1.7810e-02,
        ...
        5.5784e-04, 2.4796e-04, 6.1989e-05, 0.0000e+00])},
 '_forward_hooks': OrderedDict(),
 '_forward_hooks_always_called': OrderedDict(),
 '_forward_hooks_with_kwargs': OrderedDict(),
 '_forward_pre_hooks': OrderedDict(),
 '_forward_pre_hooks_with_kwargs': OrderedDict(),
 '_is_full_backward_hook': None,
 '_load_state_dict_post_hooks': OrderedDict(),
 '_load_state_dict_pre_hooks': OrderedDict(),
 '_modules': {},
 '_non_persistent_buffers_set': set(),
 '_parameters': {},
 '_rng': <random.Random object at 0x45a26d70>,
 '_state_dict_hooks': OrderedDict(),
 '_state_dict_pre_hooks': OrderedDict(),
 'dither': 0.0,
 'exact_pad': False,
 'forward': <function FilterbankFeatures.forward at 0x78ff32bfbd80>,
 'frame_splicing': 1,
 'hop_length': 160,
 'log': True,
 'log_zero_guard_type': 'add',
 'log_zero_guard_value': 5.960464477539063e-08,
 'mag_power': 2.0,
 'max_length': tensor(1670),
 'n_fft': 512,
 'nb_augmentation_prob': 0.0,
 'nfilt': 128,
 'normalize': 'per_feature',
 'pad_to': 0,
 'pad_value': 0.0,
 'preemph': 0.97,
 'sample_rate': 16000,
 'stft_pad_amount': None,
 'training': False,
 'use_grads': False,
 'win_length': 400}
```

