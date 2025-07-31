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
