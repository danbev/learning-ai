## Huggingface Transformers


### 2_Dense
I've come accross a huggingface model, an embedding model, which contains a
directory named `2_Dense`. The directory contains two files:
```console
ls 2_Dense/
config.json  model.safetensors
```
And the content of `config.json` is:    
```console
$ cat 2_Dense/config.json 
{
    "in_features": 768,
    "out_features": 768,
    "bias": false,
    "activation_function": "torch.nn.modules.linear.Identity"
```
So this is a layer that would be applied after the main models output, in this
case the embedding layer.

Notice that the activation function is `torch.nn.modules.linear.Identity`, which
means that the output of the layer is the same as the input.

```console
(venv) $ python inspect-dense.py
Weight matrix shape: torch.Size([768, 768])
Weight dtype: torch.float32
Is identity matrix: False
✅ This layer actually does something - it's NOT just identity!
Weight stats:
  Min: -0.507812
  Max: 0.500000
  Mean: 0.000013
  Std: 0.115306
Diagonal mean: -0.002483
Diagonal std: 0.116433
Top-left 5x5 corner:
tensor([[ 0.0237, -0.0786, -0.0304, -0.0923,  0.0752],
        [ 0.1953,  0.0588,  0.0820, -0.0510, -0.0640],
        [ 0.0811,  0.2041,  0.0287, -0.1211, -0.2236],
        [ 0.0825,  0.3848, -0.0052,  0.0449, -0.0237],
        [-0.0339, -0.0053,  0.1245,  0.0109,  0.0322]])
```
```python
from sentence_transformers import SentenceTransformer

# Load the full model with all components
model = SentenceTransformer('.')

# Get sentence embeddings (this goes through 2_Dense)
texts = ["Hello world", "How are you?"]
embeddings = model.encode(texts)

print(f"Model loaded: {model}")

print(f"Embedding shape: {embeddings.shape}")  # [2, 768] - one embedding per sentence
```
```console
(venv) $ python sentence_transformer.py
Model loaded: SentenceTransformer(
  (0): Transformer({'max_seq_length': 1024, 'do_lower_case': False, 'architecture': 'SomeTextModel'})
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Dense({'in_features': 768, 'out_features': 768, 'bias': False, 'activation_function': 'torch.nn.modules.linear.Identity'})
)
Embedding shape: (2, 768)
```
So this will use the base model as it, then apply the pooling layer and then
the dense layer. The dense layer is the one that is in `2_Dense` and it will
apply the weights to the output of the pooling layer.
SentenceTransformer will scan the model directory for subdirectories that start
with a number, and it will load them in the order of the numbers.

