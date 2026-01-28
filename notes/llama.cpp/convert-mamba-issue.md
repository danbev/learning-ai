### Convert Mamba model issue
This was an issue I ran into when converting a Mamba model which had worked in
the past.

Stepping through convert_hf_to_gguf.py:
```console
(Pdb) l
9049 	        # config, so we need to explicitly override it here.
9050 	        if not self.is_moe:
9051 	            self.gguf_writer.add_add_bos_token(True)
9052
9053 	    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
9054B->	        if self.is_moe and bid is not None:
9055 	            # Skip Multi-Token Prediction (MTP) tensors. These are used for
9056 	            # for speculative decoding but we don't include them in this model
9057 	            # conversion. See https://github.com/ggml-org/llama.cpp/pull/18886
9058 	            if "mtp" in name:
9059 	                print(f"Skipping MTP (Speculative) layer: {name}")
(Pdb) p name
'backbone.layers.0.norm.weight'
```
Stepping through we get to this call:
```console
(Pdb)
> /home/danbev/work/llama.cpp-staging/convert_hf_to_gguf.py(9120)modify_tensors()
-> yield from super().modify_tensors(data_torch, name, bid)
```
```python
    def modify_tensors(
        self, data_torch: Tensor, name: str, bid: int | None
    ) -> Iterable[tuple[str, Tensor]]:
        if (
            name.endswith("block_sparse_moe.input_linear.weight")
            or "shared_mlp" in name
        ):
            return GraniteMoeModel.modify_tensors(self, data_torch, name, bid)

        # Determine whether this is a mamba layer or an attention layer
        if bid in self._ssm_layers:
            return Mamba2Model.modify_tensors(self, data_torch, name, bid)
        elif bid in self._attn_layers:
            return GraniteMoeModel.modify_tensors(self, data_torch, name, bid)
        yield from ModelBase.modify_tensors(self, data_torch, name, bid)
```
```console
(Pdb) p self._ssm_layers
[0, 2, 4, 6, 9, 11, 13, 15, 18, 20, 22, 24, 27, 29, 31, 33, 35, 38, 40, 42, 44, 46, 49, 51, 53, 55, 57, 60, 62, 64, 66, 68, 71, 73, 75, 77, 80, 82, 84, 86]
(Pdb) p bid
0
(Pdb) p name
'backbone.layers.0.norm.weight'
```
So the current tensor is part of the _ssm layers.
```console
(Pdb) s
> /home/danbev/work/llama.cpp-staging/convert_hf_to_gguf.py(8919)modify_tensors()
-> return Mamba2Model.modify_tensors(self, data_torch, name, bid)
```
Lets see if we the tensor name is in the tensor map:
```console
(Pdb)  self.map_tensor_name('backbone.layers.0.norm.weight')
'blk.0.attn_norm.weight'

(Pdb) p self.model_arch
<MODEL_ARCH.NEMOTRON_H_MOE: 77>
```

Looking at the function again:
```python
    def modify_tensors(
        self, data_torch: Tensor, name: str, bid: int | None
    ) -> Iterable[tuple[str, Tensor]]:
        if (
            name.endswith("block_sparse_moe.input_linear.weight")
            or "shared_mlp" in name
        ):
            return GraniteMoeModel.modify_tensors(self, data_torch, name, bid)

        # Determine whether this is a mamba layer or an attention layer
        if bid in self._ssm_layers:
            return Mamba2Model.modify_tensors(self, data_torch, name, bid)
        elif bid in self._attn_layers:
            return GraniteMoeModel.modify_tensors(self, data_torch, name, bid)
        yield from ModelBase.modify_tensors(self, data_torch, name, bid)
```
The last yield makes this a generator function.

We are calling this function like this:
```python
        yield from super().modify_tensors(data_torch, name, bid)
```
So GraniteHybridModel.modify_tensors() is creating a generate object when 
return Mamba2Model.modify_tensors(...), and return will create a stop iteration
but notice that no values has been yielded, so anyone calling this with
`yield from` will get nothing which is what is currently happening. 

We need to yield the values from those calls to enable this to work properly:
```console
(venv) $ git diff convert_hf_to_gguf.py
diff --git a/convert_hf_to_gguf.py b/convert_hf_to_gguf.py
index b85a65143..cf52f25f1 100755
--- a/convert_hf_to_gguf.py
+++ b/convert_hf_to_gguf.py
@@ -8916,9 +8916,11 @@ class GraniteHybridModel(Mamba2Model, GraniteMoeModel):

         # Determine whether this is a mamba layer or an attention layer
         if bid in self._ssm_layers:
-            return Mamba2Model.modify_tensors(self, data_torch, name, bid)
+            yield from Mamba2Model.modify_tensors(self, data_torch, name, bid)
+            return
         elif bid in self._attn_layers:
-            return GraniteMoeModel.modify_tensors(self, data_torch, name, bid)
+            yield from GraniteMoeModel.modify_tensors(self, data_torch, name, bid)
+            return
         yield from ModelBase.modify_tensors(self, data_torch, name, bid)

     def set_gguf_parameters(self):
```

