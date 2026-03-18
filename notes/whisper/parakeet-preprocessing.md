```console
(Pdb) b venv/lib/python3.13/site-packages/nemo/collections/asr/parts/preprocessing/features.py:414
Breakpoint 8 at /Users/danbev/work/ai/whisper-models/nvidia/parakeet-tdt-0.6b-v3/venv/lib/python3.13/site-packages/nemo/collections/asr/parts/preprocessing/features.py:414
```

```console
(Pdb) p x.shape
torch.Size([1, 176000])

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
So the above will be skipped. But note that there is implicit padding in the stft
function:
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
So the above will be skipped be skipped too.

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
```

```python

        # disable autocast to get full range of stft values
        with torch.amp.autocast(x.device.type, enabled=False):
            x = self.stft(x)
```
amp is Automatic Mixed Precision which is way to optimize operations where they
can be performed in lower precision. The autocase is enables torch to lower the
precision and casts in the block as it sees fit. And notice that this is
actually disabling the autocast feature. So this is where we perform the STFT.

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
tensor([-16.6355, -16.6355, -16.6355, -16.6355, -16.6355, -16.6352, -15.5229,
        -15.3404, -15.9664, -15.9212])
(Pdb) p (x.double()**2).mean().item()
102.66635666443821

After normalization:
(Pdb) p x.view(-1)[:10]
tensor([-2.1009, -2.1009, -2.1009, -2.1009, -2.1009, -2.1006, -0.9609, -0.7740,
        -1.4154, -1.3690])
(Pdb) p (x.double()**2).mean().item()
0.9987379881895161
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
