### GPTQ Exploration

So how this works is that we first download calibration data that is passed
through the original model. For each linear layer GPTQ records the inputs (X)
that each layer sees. From them it builds one matrix which is the Hessian:
```console
H = 2 X X^T
```
So X are the input 

This is what GPTQ uses from the calibration data. The loop in gptq then becomes:
```console
q   = quantize(w, scale, zero)
err =  (w - q) / Hinv[i, i]
```

Example:
```console
y = w₁
```

### install

```console
$ pip install -v gptqmodel
```


### model

```console
(venv) spark $ python src/dump-tensor.py models/Qwen2.5-0.5B-Instruct-gptq-4bit/model.safetensors 
metadata: {'format': 'pt'}
model.embed_tokens.weight                                    BF16   [151936, 896]
model.layers.0.input_layernorm.weight                        BF16   [896]
model.layers.0.post_attention_layernorm.weight               BF16   [896]

model.layers.0.mlp.down_proj.g_idx                           I32    [4864]
model.layers.0.mlp.down_proj.qweight                         I32    [608, 896]
model.layers.0.mlp.down_proj.qzeros                          I32    [38, 112]
model.layers.0.mlp.down_proj.scales                          F16    [38, 896]

model.layers.0.mlp.gate_proj.g_idx                           I32    [896]
model.layers.0.mlp.gate_proj.qweight                         I32    [112, 4864]
model.layers.0.mlp.gate_proj.qzeros                          I32    [7, 608]
model.layers.0.mlp.gate_proj.scales                          F16    [7, 4864]

model.layers.0.mlp.up_proj.g_idx                             I32    [896]
model.layers.0.mlp.up_proj.qweight                           I32    [112, 4864]
model.layers.0.mlp.up_proj.qzeros                            I32    [7, 608]
model.layers.0.mlp.up_proj.scales                            F16    [7, 4864]

model.layers.0.self_attn.k_proj.bias                         F16    [128]
model.layers.0.self_attn.k_proj.g_idx                        I32    [896]
model.layers.0.self_attn.k_proj.qweight                      I32    [112, 128]
model.layers.0.self_attn.k_proj.qzeros                       I32    [7, 16]
model.layers.0.self_attn.k_proj.scales                       F16    [7, 128]

model.layers.0.self_attn.o_proj.g_idx                        I32    [896]
model.layers.0.self_attn.o_proj.qweight                      I32    [112, 896]
model.layers.0.self_attn.o_proj.qzeros                       I32    [7, 112]
model.layers.0.self_attn.o_proj.scales                       F16    [7, 896]
model.layers.0.self_attn.q_proj.bias                         F16    [896]
model.layers.0.self_attn.q_proj.g_idx                        I32    [896]
model.layers.0.self_attn.q_proj.qweight                      I32    [112, 896]
model.layers.0.self_attn.q_proj.qzeros                       I32    [7, 112]
model.layers.0.self_attn.q_proj.scales                       F16    [7, 896]

model.layers.0.self_attn.v_proj.bias                         F16    [128]
model.layers.0.self_attn.v_proj.g_idx                        I32    [896]
model.layers.0.self_attn.v_proj.qweight                      I32    [112, 128]
model.layers.0.self_attn.v_proj.qzeros                       I32    [7, 16]
model.layers.0.self_attn.v_proj.scales                       F16    [7, 128]
```
The `.qweight` tensors are packed quantized weights.
The `.i_idx` maps each column to its group?
The `.qzeros` are packed zero-points per group.
The `.scales` are the scales per group, which are like a block and each block/group
has its own scale. I think a group/block is usually 128.

### conversion
In conversion/base.py we have:
```python
    def dequant_model(self):
        ...
        tensors_to_remove: list[str] = []
        new_tensors: dict[str, Callable[[], Tensor]] = {}

        if (quant_config := self.hparams.get("quantization_config")) and isinstance(quant_config, dict):
            quant_method = quant_config.get("quant_method")
            ...

            if quant_method == "bitnet":
                ...
            elif quant_method == "gptq":
                for name in self.model_tensors.keys():
                    if name.endswith(".qweight"):
                        base_name = name.removesuffix(".qweight")
                        g_idx = self.model_tensors[base_name + ".g_idx"]
                        qweight = self.model_tensors[base_name + ".qweight"]
                        qzeros = self.model_tensors[base_name + ".qzeros"]
                        scales = self.model_tensors[base_name + ".scales"]
                        new_tensors[base_name + ".weight"] = (
                            lambda g=g_idx, z=qzeros, w=qweight, s=scales: dequant_gptq(
                                g(), w(), z(), s()
                            )
                        )
                        tensors_to_remove += [
                            base_name + n
                            for n in (
                                ".g_idx",
                                ".qzeros",
                                ".qweight",
                                ".scales",
                            )
                        ]
            ...
```
This function is actually declared inside of the function above but I'm adding
it after to make it more readable.
```python
            def dequant_gptq(g_idx: Tensor, qweight: Tensor, qzeros: Tensor, scales: Tensor) -> Tensor:
                bits = quant_config["bits"]
                assert bits in (2, 3, 4, 8)
                assert qweight.dtype == qzeros.dtype
                maxq = (2 ** bits) - 1
                weight = None
                zeros = None
                pack_dtype_bits = qweight.dtype.itemsize * 8

                if bits in [2, 4, 8]:
                    pack_factor = pack_dtype_bits // bits
                    wf = torch.tensor(list(range(0, pack_dtype_bits, bits)), dtype=torch.int32).unsqueeze(0)
                    if self.lazy:
                        wf = LazyTorchTensor.from_eager(wf)

                    zeros = torch.bitwise_right_shift(
                        qzeros.unsqueeze(2).expand(-1, -1, pack_factor),
                        wf.unsqueeze(0)
                    ).to(torch.int16 if bits == 8 else torch.int8)
                    zeros = torch.bitwise_and(zeros, maxq).reshape(scales.shape)

                    weight = torch.bitwise_and(
                        torch.bitwise_right_shift(
                            qweight.unsqueeze(1).expand(-1, pack_factor, -1),
                            wf.unsqueeze(-1)
                        ).to(torch.int16 if bits == 8 else torch.int8),
                        maxq
                    )
                elif bits == 3:
                    raise NotImplementedError("3-bit gptq dequantization is not yet implemented")

                assert weight is not None
                assert zeros is not None

                weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])

                # gptq_v2 doesn't need to offset zeros
                if quant_config.get("checkpoint_format", "gptq") == "gptq":
                    zeros += 1

                return (scales[g_idx].float() * (weight - zeros[g_idx]).float()).T
```
(wf = weight format?)
```console
(Pdb) p wf
tensor([[ 0,  4,  8, 12, 16, 20, 24, 28]], dtype=torch.int32)
```
This will be passed to from_eager
```console
(Pdb) p wf
<conversion.base.LazyTorchTensor object at 0xfaa3d1a54f80>
(Pdb) p wf._meta
tensor(..., device='meta', size=(1, 8), dtype=torch.int32)
(Pdb) p wf._data
tensor([[ 0,  4,  8, 12, 16, 20, 24, 28]], dtype=torch.int32)
(Pdb) p wf._func
None
```
And qzeros is also a lazy tensor:
```console
(Pdb) p qzeros
<conversion.base.LazyTorchTensor object at 0xfaa3d1a54d10>
(Pdb) p qzeros._meta
tensor(..., device='meta', size=(38, 112), dtype=torch.int32)
(Pdb) p qzeros._data
None
(Pdb) p qzeros._func
<function LazyTorchTensor.from_local_tensor.<locals>.<lambda> at 0xfaa3d1a522a0>

(Pdb) p pack_factor
8
```
Before we dive into further function like torch.bitwise_right_shift I was actually
thinking that LazyTensors was a Torch concept because we seem to be able to
pass them into Torch functions.

Then we have:
```python
                    zeros = torch.bitwise_right_shift(
                        qzeros.unsqueeze(2).expand(-1, -1, pack_factor),
                        wf.unsqueeze(0)
                    ).to(torch.int16 if bits == 8 else torch.int8)
```
This will call the real Torch function but that function checks it arguments and
if is sees one with a `__torch_function__` method it will hand the call over
instead of computing. This will then land in conversion/base.py:
```python
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        del types  # unused

        if kwargs is None:
            kwargs = {}

        if func is torch.Tensor.numpy:
            assert len(args)
            return args[0].numpy()

        return cls._wrap_fn(func)(*args, **kwargs)
```

Now, recall that qzeros contains the zero value for groups/blocks, and these
values are packed:
```console
(Pdb) p qzeros.shape
torch.Size([38, 112])
```
We have a group_size of 128 which means there are 128 weights in each group, 
and we have the layers input colums which is 4864 / 128 = 38 groups. This gives
us 112 packed rows, unpacked this is 128*8=896 ([38, 896].
```console
            group 0        group 1           group 37
          cols 0-127    cols 128-255        cols 4736-4863
row 0   [ 128 weights ][ 128 weights ] ... [ 128 weights ]
row 1   [ 128 weights ][ 128 weights ] ... [ 128 weights ]
...
row 895 [ 128 weights ][ 128 weights ] ... [ 128 weights ]
```
Each "box" is a block of 128 weights sharing one zero point and one scale.

Now, qzeros is this is what we will be shifting right. And the shifts are in
wf.

Lets inspect this in pdb (using the interactive "debugger"):
```console
(Pdb) interact
*interactive*
>>> z = LazyTorchTensor.to_eager(qzeros)
>>> f = LazyTorchTensor.to_eager(wf)

>>> f.shape
torch.Size([1, 8])
>>> f
tensor([[ 0,  4,  8, 12, 16, 20, 24, 28]], dtype=torch.int32)

>>> z.shape
torch.Size([38, 112])

>>> z[0,0]
tensor(2004318071, dtype=torch.int32)
```
Notice that one element of z is 2004318071 which in hex is 0x77777777.
```console
Bits:     31..28   27..24  23..20  19..16  15..12  11..8   7..4    3..0
Nibbles:  [ 0x7 ] [ 0x7 ] [ 0x7 ] [ 0x7 ] [ 0x7 ] [ 0x7 ] [ 0x7 ] [ 0x7 ]
Value:    Val 7    Val 6   Val 5   Val 4   Val 3   Val 2   Val 1   Val 0
```
So, we have a single integer that stores 8 zero-point values all packed into
a single int. The are all the same values 7 which may not be great for understanding
this but those are the real values.

What we want to do is to apply these shifts to all the elements in z, because
we want to unpack all the zeropoints (all 8 values in each integer stored). What
we do is that we duplicate the 
```console
>>> z.shape
torch.Size([38, 112])

>>> a = z.unsqueeze(2)
>>> a.shape
torch.Size([38, 112, 1])
```
When then use expand to expand the last dim to the same as the number of shifts:
-1 means keep the existing dimension sizes:
```console
>>> b = a.expand(-1, -1, 8)
>>> b.shape
torch.Size([38, 112, 8])
```
So now the dimensions add up. And we now have 8 identical copies of the packed
zeros. Remember will want to shift the same value but with 8 different shift
values to extract the 8 different zero point values. And we will be doing this
for the entire tensor.

So we have our shifts in f:
```console
>>> f
tensor([[ 0,  4,  8, 12, 16, 20, 24, 28]], dtype=torch.int32)
>>> f.shape
torch.Size([1, 8])
```
Notice that this does not match b at this point which has the shape [38, 112, 8],
so we need to unsqueeze f to match b:
```console
>>> f.unsqueeze(0).shape
torch.Size([1, 1, 8])
```
And now we can perform the shifts:
```console
>>> c = torch.bitwise_right_shift(b, f.unsqueeze(0))
>>> [hex(x) for x in c[0,0].tolist()]
['0x77777777', '0x7777777', '0x777777', '0x77777', '0x7777', '0x777', '0x77', '0x7']
>>> d = torch.bitwise_and(c, maxq)
>>> d[0,0]
tensor([7, 7, 7, 7, 7, 7, 7, 7], dtype=torch.int32)
>>> d.reshape(scales.shape).shape
torch.Size([38, 896])

>>> f.unsqueeze(0)
tensor([[[ 0,  4,  8, 12, 16, 20, 24, 28]]], dtype=torch.int32)
>>> f.unsqueeze(0).shape
torch.Size([1, 1, 8])
>>> b.shape
torch.Size([38, 112, 8])

```

### bidwise_right_shift
```console
python
>>> import torch

>>> z = torch.tensor([[0x42], [0x71]])
>>> z
tensor([[ 66],
        [113]])
>>> z.shape
torch.Size([2, 1])

wf = torch.tensor([[0, 4]])
>>> wf.shape
torch.Size([1, 2])
>>> wf
tensor([[0, 4]])

b = z.unsqueeze(2).expand(-1, -1, 2)
>>> b.shape
torch.Size([2, 1, 2])

>>> b
tensor([
  [[ 66,  66]],
  [[113, 113]]
])
>>> b.shape
torch.Size([2, 1, 2])

>>> shifts = wf.unsqueeze(0)
>>> shifts
tensor([[[0, 4]]])
>>> shifts.shape
torch.Size([1, 1, 2])
```
When torch evalutates b >> shifts it will line up the shapes from right to left:
```console
b:       [ 2,  1,  2 ]          // torch.Size([2, 1, 2])
shifts:  [ 1,  1,  2 ]          // torch.Size([1, 1, 2])
           │   │   │
           │   │   └─ Dim 2: (2 == 2) Perfect match.
           │   └───── Dim 1: (1 == 1) Perfect match.
           └───────── Dim 0: (2 != 1) Size 1 stretches to match 2!
```
Notice that shifts has 1 along the first dimension. I think of this as torch
creating a new matrix with the same number of rows as b and it just duplicates
the row into this new tensor so that the op can be performed:
```console
Row 0:  [0x42, 0x42]  >>  [0, 4]  -->  [0x42, 0x04]
Row 1:  [0x71, 0x71]  >>  [0, 4]  -->  [0x71, 0x07]
```
Torch does not do any copying of data, that is only for my mental model.

```console
>>> c = torch.bitwise_right_shift(b, shifts)
>>> c
tensor([[[ 66,   4]],

        [[113,   7]]])
```
So the right shifting has now shifted the 4-bit chunks down to the lowest bits,
0-3, but it also moves anything to the left of that as well with it.
So we apply a mask to only keep the lower 4 bits:
```console
>>> d = torch.bitwise_and(c, 0x0F)
>>> d
tensor([[[2, 4]],

        [[1, 7]]])
```
So each packed integer has been split into its two indepentant 4-bit values.
