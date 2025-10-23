## convert_hf_to_gguf.py dequantization
Some models are already quantized and to convert them we need to dequantized them in
llama.cpp before converting to gguf format.

So the BaseModel in convert_hf_to_gguf.py has a number of members one is:
```python
class ModelBase:
    ...

    model_tensors: dict[str, Callable[[], Tensor]]
```
This is a dictionary of tensor names to functions that return the tensor.

In the constructor we then have:
```python
        self.model_tensors = self.index_tensors(remote_hf_model_id=remote_hf_model_id)
```
So this allows passing in a remote model id, a hugging face model id, or None in which case
a local model is used (on the local disk):
```python
    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], Tensor]]:
        tensors: dict[str, Callable[[], Tensor]] = {}
```
And we can see the return type is the dictionary of tensor names to functions that return tensors.

If remote tensors are used we have the following code:
```python
        if remote_hf_model_id is not None:
            is_safetensors = True

            logger.info(f"Using remote model with HuggingFace id: {remote_hf_model_id}")
            remote_tensors = gguf.utility.SafetensorRemote.get_list_tensors_hf_model(remote_hf_model_id)
            for name, remote_tensor in remote_tensors.items():
                tensors[name] = lambda r=remote_tensor: LazyTorchTensor.from_remote_tensor(r)

            return tensors
```
So that is how remote models are handled (notice the return statement).

For local models we have:
```python
        part_names: list[str] = ModelBase.get_model_part_names(self.dir_model, "model", ".safetensors")
        is_safetensors: bool = len(part_names) > 0
        if not is_safetensors:
            part_names = ModelBase.get_model_part_names(self.dir_model, "pytorch_model", ".bin")
```
part_names is just the list of weight files, so all the .safetensor files in the  model directory, or
fallback to .bin files.

```python
        tensor_names_from_index: set[str] = set()

        index_name = "model.safetensors" if is_safetensors else "pytorch_model.bin"
        index_name += ".index.json"
        index_file = self.dir_model / index_name
```
That last line is joining the Path in a platform independent way.

If the index file exist the it will be opened and the weight map loaded:
```python
        if index_file.is_file():
            logger.info(f"gguf: loading model weight map from '{index_name}'")
            with open(index_file, "r", encoding="utf-8") as f:
                index: dict[str, Any] = json.load(f)
                weight_map = index.get("weight_map")
                if weight_map is None or not isinstance(weight_map, dict):
                    raise ValueError(f"Can't load 'weight_map' from {index_name!r}")
                tensor_names_from_index.update(weight_map.keys())
        else:
        weight_map = {}
```
Next we will iterate over all the weight files:
```python
        for part_name in part_names:
            logger.info(f"gguf: indexing model part '{part_name}'")
            ctx: ContextManager[Any]
            if is_safetensors:
                from safetensors import safe_open
                ctx = cast(ContextManager[Any], safe_open(self.dir_model / part_name, framework="pt", device="cpu"))
            else:
                ctx = contextlib.nullcontext(torch.load(str(self.dir_model / part_name), map_location="cpu", mmap=True, weights_only=True))
```
ContextManager is a Python type that allows us to use the "with" statement to manage resources. safe_open
returns a ContextManager that opens the safetensor file and allows us iterate over the tensors in it without
loading all the tensors into memory at once.
torch.load does not return a ContextManager but it is wrapped in a dummy context manager using contextlib.nullcontext
so that it can be treated as one.

```python
            with ctx as model_part:
                assert model_part is not None

                # iterate over all the tensors in this part (weight file)
                for name in model_part.keys():
                    if is_safetensors:
                        if self.lazy:
                            data = model_part.get_slice(name)
                            data_gen = lambda data=data: LazyTorchTensor.from_safetensors_slice(data)  # noqa: E731
                        else:
                            data = model_part.get_tensor(name)
                            data_gen = lambda data=data: data  # noqa: E731
                    else:
                        data = model_part[name]
                        if self.lazy:
                            data_gen = lambda data=data: LazyTorchTensor.from_eager(data)  # noqa: E731
                        else:
                            data_gen = lambda data=data: data  # noqa: E731
                    tensors[name] = data_gen
```
lambda data=data: uses a default argument to capture the current value of data. Without this, all
lambdas in the loop would close over the same data variable and end up returning the last one.
And notice that the tensors[name] is set to data_gen which is a function that returns the tensor data.

And finally we have the consistency check:
```python
        if len(tensor_names_from_index) > 0:
            tensor_names_from_parts = set(tensors.keys())
            if len(tensor_names_from_parts.symmetric_difference(tensor_names_from_index)) > 0:
                missing = sorted(tensor_names_from_index.difference(tensor_names_from_parts))
                extra = sorted(tensor_names_from_parts.difference(tensor_names_from_index))
                missing_files = sorted(set(weight_map[n] for n in missing if n in weight_map))
                if len(extra) == 0 and len(missing_files) > 0:
                    raise ValueError(f"Missing or incomplete model files: {missing_files}\n"
                                     f"Missing tensors: {missing}")
                else:
                    raise ValueError("Mismatch between weight map and model parts for tensor names:\n"
                                     f"Missing tensors: {missing}\n"
                                     f"Extra tensors: {extra}")

        return tensors
```

Back in the constructor we later have:
```python
        self.dequant_model()
```
And in dequant_model we first check if the model configuration contains a quantization_config:
```python
    def dequant_model(self):
        tensors_to_remove: list[str] = []
        new_tensors: dict[str, Callable[[], Tensor]] = {}

        if (quant_config := self.hparams.get("quantization_config")) and isinstance(quant_config, dict):
            quant_method = quant_config.get("quant_method")
```
For example a model might contains something like this:
```console
  "quantization_config": {
    "activation_scheme": "dynamic",
    "modules_to_not_convert": null,
    "quant_method": "fp8",
    "weight_block_size": [
      128,
      128
    ]
  },
```
Then we have a few function defined for dequatizing differenct types of quantiztion methods:
```python
            def dequant_bitnet(weight: Tensor, scale: Tensor) -> Tensor:
                ...

            def dequant_gptq(g_idx: Tensor, qweight: Tensor, qzeros: Tensor, scales: Tensor) -> Tensor:
                ...

            def dequant_simple(weight: Tensor, scale: Tensor) -> Tensor:
                scale = scale.float()

                if (weight_block_size := quant_config.get("weight_block_size")):
                    # TODO: make sure it's a list of integers
                    for i, size in enumerate(weight_block_size):
                        scale = scale.repeat_interleave(size, i)
                # unpad the scale (e.g. when the tensor size isn't a multiple of the block size)
                scale = scale[tuple(slice(0, size) for size in weight.shape)]

                return weight.float() * scale
```
I'm showing the simple  as that is the example I'm working with at the moment:
```python
            if quant_method == "bitnet":
                for name in self.model_tensors.keys():
                    if name.endswith(".weight_scale"):
                        weight_name = name.removesuffix("_scale")
                        w = self.model_tensors[weight_name]
                        s = self.model_tensors[name]
                        self.model_tensors[weight_name] = lambda w=w, s=s: dequant_bitnet(w(), s())
                        tensors_to_remove.append(name)
            elif quant_method == "fp8":
                for name in self.model_tensors.keys():
                    if name.endswith(".weight_scale_inv"):
                        weight_name = name.removesuffix("_scale_inv")
                        w = self.model_tensors[weight_name]
                        s = self.model_tensors[name]
                        self.model_tensors[weight_name] = lambda w=w, s=s: dequant_simple(w(), s())
                        tensors_to_remove.append(name)
```
So a model that has quantized weights will also have scale tensors that are used to dequantize the weights.
And these are the inverse scale/delta so to dequantize we just multiply the weights by the scale.
