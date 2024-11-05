## Model conversion notes

### Overview of convert_hf_to_gguf.py
This script starts by
```python
    hparams = Model.load_hparams(dir_model)

    with torch.inference_mode():
        output_type = ftype_map[args.outtype]
        model_architecture = hparams["architectures"][0]

        try:
            model_class = Model.from_model_architecture(model_architecture)
        except NotImplementedError:
            logger.error(f"Model {model_architecture} is not supported")
            sys.exit(1)

        model_instance = model_class(dir_model=dir_model, ftype=output_type, fname_out=fname_out,
                                     is_big_endian=args.bigendian, use_temp_file=args.use_temp_file,
                                     eager=args.no_lazy,
                                     metadata_override=args.metadata, model_name=args.model_name,
                                     split_max_tensors=args.split_max_tensors,
                                     split_max_size=split_str_to_n_bytes(args.split_max_size), dry_run=args.dry_run,
                                     small_first_shard=args.no_tensor_first_split)

        if args.vocab_only:
            ...
        else:
            logger.info("Exporting model...")
            model_instance.write()
            out_path = f"{model_instance.fname_out.parent}{os.sep}" if is_split else model_instance.fname_out
            logger.info(f"Model successfully exported to {out_path}")
```
The line with `torch.inference_mode()` is used to tell torch to not track
gradients (no autograd) and not build up a computation graph. So a new Model
instance is created using the model architecture element from the hparams
dictionary. This is an array in config.json but the first one is used here.
So first we retrieve a Model subclass for the model architecture, then we
call that class's constructor using `model_class`. 

After we have the instance we den call the write method which looks like this:
```python
    def write(self):
        self.prepare_tensors()
        self.prepare_metadata(vocab_only=False)
        self.gguf_writer.write_header_to_file(path=self.fname_out)
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.write_tensors_to_file(progress=True)
        self.gguf_writer.close()
```

And `prepare_tensors` looks like this:
```python
    def prepare_tensors(self):
        max_name_len = max(len(s) for _, s in self.tensor_map.mapping.values()) + len(".weight,")

        for name, data_torch in chain(self.generate_extra_tensors(), self.get_tensors()):
            print(name)
```
Here we can see that how the code is calling the subclass'
`generate_extra_tensors` and `get_tensors` methods if they are overridden and
otherwise it will call the base class' methods. Just wanted to show how
subclasses can override the behavior of the base class.
```
            for new_name, data_torch in (self.modify_tensors(data_torch, name, bid)):
```

### part_names
This are the tensor files for a model., for example:
```console
[
  'model-00001-of-00005.safetensors',
  'model-00002-of-00005.safetensors',
  'model-00003-of-00005.safetensors',
  'model-00004-of-00005.safetensors',
  'model-00005-of-00005.safetensors'
]
```

```python
        self.part_names = Model.get_model_part_names(self.dir_model, "model", ".safetensors")
```
This is passing in a prefix of `model` which we can see that the files have, 
and a suffix of `.safetensors` which we can see that the files have.
```python
    @staticmethod
    def get_model_part_names(dir_model: Path, prefix: str, suffix: str) -> list[str]:
        part_names: list[str] = []
        for filename in os.listdir(dir_model):
            if filename.startswith(prefix) and filename.endswith(suffix):
                part_names.append(filename)

        part_names.sort()

        return part_names
```

### hparams
This property is a dictionary loaded from the model directores config.json file:

The constructor of the base class Model takes a dictionary of hyperparameters
as an argument ,
```python

        self.hparams = Model.load_hparams(self.dir_model) if hparams is None else hparams
        ...


    @staticmethod
    def load_hparams(dir_model: Path):
        with open(dir_model / "config.json", "r", encoding="utf-8") as f:
            return json.load(f)
```

### block_count
```python
    self.block_count = self.find_hparam(["n_layers", "num_hidden_layers", "n_layer", "num_layers"])
    ...

    def find_hparam(self, keys: Iterable[str], optional: bool = False) -> Any:
        key = next((k for k in keys if k in self.hparams), None)
        if key is not None:
            return self.hparams[key]
        if optional:
            return None
        raise KeyError(f"could not find any of: {keys}")
```
The generator expression (k for k in keys if k in self.hparams) is used to
get all the k in keys, but only those that are in self.hparams.


### TensorMap
In `gguf-py/gguf/tensor_mapping.py` we a class TensorNameMap.
There is a dictionary named `mapping_cfg` which is for tensors that are "global"
or not part of a block/layer in the models.
```python
class TensorNameMap:
    mappings_cfg: dict[MODEL_TENSOR, tuple[str, ...]] = {
        ...
    }
    block_mappings_cfg: dict[MODEL_TENSOR, tuple[str, ...]] = {
        ...
    }
```
Both of these dictionaries use the `MODEL_TENSOR` enum as keys which is defined
in constants.py:
```python`
class MODEL_TENSOR(IntEnum):
    TOKEN_EMBD           = auto()
    TOKEN_EMBD_NORM      = auto()
    TOKEN_TYPES          = auto()
    POS_EMBD             = auto()
    OUTPUT               = auto()
    OUTPUT_NORM          = auto()
    ROPE_FREQS           = auto()
    ...
```
This enum is used (but also used in other places) to specify the tensor names
that GGUF uses. For example, in constants.py we have:
```python
TENSOR_NAMES: dict[MODEL_TENSOR, str] = {
    MODEL_TENSOR.TOKEN_EMBD:                "token_embd",
    MODEL_TENSOR.TOKEN_EMBD_NORM:           "token_embd_norm",
    ...
}
```
A specific model can then specify which tensor it uses by adding an entry to
`MODEL_TENSORS`, for example:
```python
MODEL_TENSORS: dict[MODEL_ARCH, list[MODEL_TENSOR]] = {
    MODEL_ARCH.LLAMA: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        ...
}
```
So when a model is read, or more specifically the models tensor files they will
contain tensor names that are specific to that model, and these name can be
different between different models even if they are intended to be used for the
same purpose in the model. The above specifies that the LLAMA model uses the
`TOKEN_EMBD` tensor. So when the tensor named `model.embed_tokens` is read from
a tensor file that will be looked up in the tensor map and this will map to
`TOKEN_EMBD`. This is then what will be used in the gguf file written later.
```
    mappings_cfg: dict[MODEL_TENSOR, tuple[str, ...]] = {
        # Token embeddings
        MODEL_TENSOR.TOKEN_EMBD: (
            "gpt_neox.embed_in",                         # gptneox
            "transformer.wte",                           # gpt2 gpt-j mpt refact qwen dbrx jais exaone
            "transformer.word_embeddings",               # falcon
            "word_embeddings",                           # bloom
            "model.embed_tokens",                        # llama-hf nemotron olmoe
            "tok_embeddings",                            # llama-pth
            "embeddings.word_embeddings",                # bert nomic-bert
            "language_model.embedding.word_embeddings",  # persimmon
            "wte",                                       # gpt2
            "transformer.embd.wte",                      # phi2
            "model.tok_embeddings",                      # internlm2
            "model.embedding",                           # mamba-qbert
            "backbone.embedding",                        # mamba
            "backbone.embeddings",                       # mamba-hf
            "transformer.in_out_embed",                  # Grok
            "embedding.word_embeddings",                 # chatglm
            "transformer.token_embeddings",              # openelm
            "shared",                                    # t5
            "rwkv.embeddings",                           # rwkv
            "language_model.model.embed_tokens",         # mllama
        ),
        ...
    }
```
```python
def get_tensor_name_map(arch: MODEL_ARCH, n_blocks: int) -> TensorNameMap:
    return TensorNameMap(arch, n_blocks)
```
In the constructor for TensorNameMap we have:
```python
    def __init__(self, arch: MODEL_ARCH, n_blocks: int):
        self.mapping = {}
        for tensor, keys in self.mappings_cfg.items():
            if tensor not in MODEL_TENSORS[arch]:
                continue
            tensor_name = TENSOR_NAMES[tensor]
            self.mapping[tensor_name] = (tensor, tensor_name)
            for key in keys:
                self.mapping[key] = (tensor, tensor_name)
```
So this is iterating over all the `mappings_cfg` items, which have the tensor
enum as the key and a tuple of strings as the value of each entry. It will then
ignore any tensor enum that is not included in the dicitionary entry for the
current model architecture.
Notice that this is inserting an entry into the `mapping` dictionar first for
the token name as the key so we can look up the tensor enum by using the
tensor name:
```python
            print(self.mapping['token_embd'])
```
```console
(<MODEL_TENSOR.TOKEN_EMBD: 1>, 'token_embd')
```
And it is also iterating over all the keys and adding them, so we can use
any of them to look up this tensor enum.
```python
            print(self.mapping['model.embed_tokens'])
            print(self.mapping['wte'])
```
```console
(<MODEL_TENSOR.TOKEN_EMBD: 1>, 'token_embd')
(<MODEL_TENSOR.TOKEN_EMBD: 1>, 'token_embd')
```

```python
MODEL_TENSORS: dict[MODEL_ARCH, list[MODEL_TENSOR]] = {
    MODEL_ARCH.LLAMA: [
        MODEL_TENSOR.TOKEN_EMBD,
        ...
}
```
Then it will lookup the name for that tensor using the tensor enum:
```python
TENSOR_NAMES: dict[MODEL_TENSOR, str] = {
    MODEL_TENSOR.TOKEN_EMBD:                "token_embd",
    ...
}
```
This will then be included in the instances `mapping` dictionary.

At some stage later `write` will be called which will call `prepare_tensors`:
Later `prepare_tensors` will call `map_tensor_names(name)`
```python
    def prepare_tensors(self):
        max_name_len = max(len(s) for _, s in self.tensor_map.mapping.values()) + len(".weight,")

        for name, data_torch in chain(self.generate_extra_tensors(), self.get_tensors()):
            ...
```
So this will iterate over the actual tensors in the models tensor files, for
example these might be safe tensors files. So the name of the tensor will be
what ever that model used, for example it might be `model.embed_tokens`.

This will then call `modify_tensors`:
```python
    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        return [(self.map_tensor_name(name), data_torch)]
```
Which in turn will call `map_tensor_name`:
```python
    def map_tensor_name(self, name: str, try_suffixes: Sequence[str] = (".weight", ".bias")) -> str:
        new_name = self.tensor_map.get_name(key=name, try_suffixes=try_suffixes)
        if new_name is None:
            raise ValueError(f"Can not map tensor {name!r}")
        return new_name
```
And this will try to look up the models tensor name by calling `get_name` on
tensor map which will use the models tensor name and lookup the name that should
be used in the gguf file for that tensor:
```python
    def get_name(self, key: str, try_suffixes: Sequence[str] = ()) -> str | None:
        result = self.get_type_and_name(key, try_suffixes = try_suffixes)
        if result is None:
            return None
        return result[1]

    def get_type_and_name(self, key: str, try_suffixes: Sequence[str] = ()) -> tuple[MODEL_TENSOR, str] | None:
        result = self.mapping.get(key)
        if result is not None:
            return result
        for suffix in try_suffixes:
            if key.endswith(suffix):
                result = self.mapping.get(key[:-len(suffix)])
                if result is not None:
                    return result[0], result[1] + suffix
        return None
```
