## PyTorch
PyTorch is a deep learning framework that provides a flexible and efficient
platform for building and training neural networks.

## nn.Module
This is the main class that all other modules inherit from. This class overrides
the special dunder method `__setattr__` to register submodules.
For example, when we write 
```python
    self.layer1 = nn.Linear(10, 5)
```
This will be "intercepted" by `__setattr__` and the layer will be registered as
a submodule something like:
```python
    self._modules['layer1'] = nn.Linear(10, 5)
```
This will end up in `__setattr_` which has a check for modules:
```python
modules = self.__dict__.get("_modules")
if isinstance(value, Module):
    if modules is None:
        raise AttributeError("cannot assign module before Module.__init__() call")
    remove_from(self.__dict__, self._parameters, self._buffers, self._non_persistent_buffers_set)
    for hook in _global_module_registration_hooks.values():
        output = hook(self, name, value)
        if output is not None:
            value = output
    modules[name] = value  # <-- This stores in _modules dict!
```
It does not actually store this value in the python object itself. If we later
access `self.layer1`, this will call `__getattr__`:
```python
    def __getattr__(self, name: str) -> Union[Tensor, "Module"]:
        if "_parameters" in self.__dict__:
            _parameters = self.__dict__["_parameters"]
            if name in _parameters:
                return _parameters[name]
        if "_buffers" in self.__dict__:
            _buffers = self.__dict__["_buffers"]
            if name in _buffers:
                return _buffers[name]
        if "_modules" in self.__dict__:
            modules = self.__dict__["_modules"]
            if name in modules:
                return modules[name]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )
```

When we access an attibute/member of an instance Python will first look in the
`__dict__` of the instance. If it does not find it there, it will call `__getattr__`
and if not found there, it will raise an `AttributeError`.

`nn.Module`'s `__init__` function is what sets up the members that instance of
this class will have:
```
        super().__setattr__("training", True)
        super().__setattr__("_parameters", {})
        super().__setattr__("_buffers", {})
        super().__setattr__("_non_persistent_buffers_set", set())
        super().__setattr__("_backward_pre_hooks", OrderedDict())
        super().__setattr__("_backward_hooks", OrderedDict())
        super().__setattr__("_is_full_backward_hook", None)
        super().__setattr__("_forward_hooks", OrderedDict())
        super().__setattr__("_forward_hooks_with_kwargs", OrderedDict())
        super().__setattr__("_forward_hooks_always_called", OrderedDict())
        super().__setattr__("_forward_pre_hooks", OrderedDict())
        super().__setattr__("_forward_pre_hooks_with_kwargs", OrderedDict())
        super().__setattr__("_state_dict_hooks", OrderedDict())
        super().__setattr__("_state_dict_pre_hooks", OrderedDict())
        super().__setattr__("_load_state_dict_pre_hooks", OrderedDict())
        super().__setattr__("_load_state_dict_post_hooks", OrderedDict())
        super().__setattr__("_modules", {})
```

### nn.Linear
The operation that this layer performs is:
```
y = xW^T + b

where:
x = input matrix
W = weight matrix
b = bias vector
y = output matrix
```
The constructor of Linear take the dimensions of the input and output:
```console
linear = nn.Linear(3, 2)
```
This means the tensor will look like this:
```console
x = [1 2 3]
    [4 5 6]

W = [1 2 3]
    [4 5 6]

b = [1 2]
```
The weight matrix is transposed:
```console
W^T = [1 4]
      [2 5]
      [3 6]
```
And the operation will be a matrix multiplication:
```console
[1, 2, 3] @ [1, 4] 
            [2, 5] = [1*1 + 2*2 + 3*3, 1*4 + 2*5 + 3*6] = [14, 32]
            [3, 6] 
```
And then the bias is added:
```console
[14, 32] + [1, 2] = [15, 34]
```
And the second sample in the input:
```console
[4 5 6] @ [1 4] 
            [2 5] = [4*1 + 5*2 + 6*3, 4*4 + 5*5 + 6*6] = [32, 77]
            [3 6] 
```
And then the bias is added:
```console
[32, 77] + [1, 2] = [33, 79]
```
The resulting output matrix `y` will then be:
```console
y = [15, 34]
    [33, 79]
```

Now, we can use `inspect` to see what the `forward` method of `nn.Linear` does:
```python
    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)
```
So that does not really tell us much. And trying to get the source of `F.linear`
will result in an error:
```console
  File "/usr/lib/python3.11/inspect.py", line 916, in getfile
    raise TypeError('module, class, method, function, traceback, frame, or '
TypeError: module, class, method, function, traceback, frame, or code object was expected, got builtin_function_or_method
```
This is because there is no python source for `F.linear` as it is a C++ function.

So lets look at `F` which is from `torch.nn.functional` which can be found in
https://github.com/pytorch/pytorch/blob/a4fc051c9a91a3170094f933f47fdbd81740de02/torch/nn/functional.py#L2302C1-L2323C2:
```python
linear = _add_docstr(
    torch._C._nn.linear,
    r"""
linear(input, weight, bias=None) -> Tensor

Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

This operation supports 2-D :attr:`weight` with :ref:`sparse layout<sparse-docs>`

{sparse_beta_warning}

This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.

Shape:

    - Input: :math:`(*, in\_features)` where `*` means any number of
      additional dimensions, including none
    - Weight: :math:`(out\_features, in\_features)` or :math:`(in\_features)`
    - Bias: :math:`(out\_features)` or :math:`()`
    - Output: :math:`(*, out\_features)` or :math:`(*)`, based on the shape of the weight
""".format(**sparse_support_notes),
)
```
Now, `_add_docstr` is a function that takes a function and a docstring as input.
And we can see that the function is `torch._C._nn.linear`. This is a C++ function
which we can inspect by importing:
```python
import torch._C

print("torch._C is the C extension module:")
print(f"   Type: {type(torch._C)}")
print(f"   Module: {torch._C}")
```
And printing this:
```console
torch._C is the C extension module:
   Type: <class 'module'>
   Module: <module 'torch._C' from '/home/danbev/work/ai/learning-ai/fundamentals/pytorch/venv/lib/python3.11/site-packages/torch/_C.cpython-311-x86_64-linux-gnu.so'>
```
```python
    if hasattr(torch._C._nn, 'linear'):
        linear_func = torch._C._nn.linear
        print(f"\ntorch._C._nn.linear:")
        print(f"   Type: {type(linear_func)}")
        print(f"   Same as F.linear? {F.linear is linear_func}")
```
```console
torch._C._nn.linear:
   Type: <class 'builtin_function_or_method'>
   Same as F.linear? True
```
The cpp implementation can be found in https://github.com/pytorch/pytorch/blob/c9642048291f427ffd39deb91e8f7c7461b8b75c/aten/src/ATen/native/Linear.cpp#L68C1-L118C2
```c++
Tensor linear(const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias_opt) {
  // _matmul_impl checks this again later, but _flatten_nd_linear does not work on scalars inputs,
  // so let's try to catch this here already
  const auto input_dim = input.dim();
  const auto weight_dim = weight.dim();
  TORCH_CHECK(input_dim != 0 && weight_dim != 0,
              "both arguments to linear need to be at least 1D, but they are ",
              input_dim, "D and ", weight_dim, "D");

  // See [Note: hacky wrapper removal for optional tensor]
  auto bias = bias_opt.has_value()
    ? c10::MaybeOwned<Tensor>::borrowed(*bias_opt)
    : c10::MaybeOwned<Tensor>::owned(std::in_place);
  if (input.is_mkldnn()) {
    return at::mkldnn_linear(input, weight, *bias);
  }
#if defined(C10_MOBILE)
  if (xnnpack::use_linear(input, weight, *bias)) {
    return xnnpack::linear(input, weight, *bias);
  }
#endif
  if (input_dim == 2 && bias->defined()) {
    // Fused op is marginally faster.
    return at::addmm(*bias, input, weight.t());
  }
  if (bias->defined() && !input.is_xla()) {
    // Also hit the fused path for contiguous 3D input, if not using xla
    // backend. Reshaping/flattening has some performance implications on xla.
    bool is_contiguous = input.is_contiguous_or_false();
    if (is_contiguous && input_dim == 3) {
      return _flatten_nd_linear(input, weight, *bias);
    } else if (is_contiguous && input.layout() == c10::kStrided && weight.layout() == c10::kStrided && bias->dim() == 1) {
      return _flatten_nd_linear(input, weight, *bias);
    } else if (parseLinearFlatten3d() && input_dim == 3) {
      // If user forces flattening via env var
      const Tensor input_cont = input.contiguous();
      return _flatten_nd_linear(input_cont, weight, *bias);
    }
  }
  auto output = at::matmul(input, weight.t());
  if (bias->defined()) {
    // for composite compliance use out-of-place version of `add`
    if (isTensorSubclassLike(*bias) ||
        bias->_fw_grad(/*level*/ 0).defined()) {
      output = at::add(output, *bias);
    } else {
      output.add_(*bias);
    }
  }
  return output;
}
```
TODO: Explain how this works with PyBind11.

__wip__

### Debugging
We can use pdb to step through code.
```console
(venv) $ python3 -m pdb convert_hf_to_gguf.py ggml-org/gemma-3-270m --outfile test-bf16.gguf --outtype bf16
```
That will break on the first line of the script. Use `?` for help.
A break point can be set using `b <lineno>` or `b <function>`.
```console
(venv) $ python3 -m pdb convert_hf_to_gguf.py ~/work/ai/models/google/gemma-3-270m-it --outfile test-bf16.gguf --outtype bf16
> /home/danbev/work/ai/llama.cpp/convert_hf_to_gguf.py(4)<module>()
-> from __future__ import annotations
(Pdb) b 11083
Breakpoint 1 at /home/danbev/work/ai/llama.cpp/convert_hf_to_gguf.py:11083
(Pdb) c
INFO:hf-to-gguf:Loading model: gemma-3-270m-it
INFO:hf-to-gguf:Model architecture: Gemma3ForCausalLM
> /home/danbev/work/ai/llama.cpp/convert_hf_to_gguf.py(11083)main()
-> model_instance = model_class(dir_model, output_type, fname_out,
(Pdb)
```
The commands are pretty simlar to gdb so most can be guessed. I'm mainly adding
this as a reference for myself as I forget the command to start pdb and it can
be useful to be able to step through the conversion script and also pytorch model
implementation when converting new models to gguf format.
