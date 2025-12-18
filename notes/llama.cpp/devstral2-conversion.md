### Devstral 2 Model conversion
This model can be found in this Hugging Face model repository:
https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512


It is not totally clear to me if this is a new version of Devstral 1 which I think
is https://huggingface.co/mistralai/Devstral-Small-2505, but this model is
multimodal which is a significant different. I was able to run this previous
model and verify the logits (though I did have to make a change in the
convert_hf_to_gguf.py script to get the conversion to work.
```console
 2418         if path_tekken_json.is_file() and not path_tokenizer_json.is_file():
 2419             self._set_vocab_mistral()
```
I needed to add a return to line 2419 so that the script does not continue which
leads to an error.


### Downloading the model files
```console
(venv) $ cd ~/models
(venv) $ export HF_TOKEN=<token>
(venv) $ hf download mistralai/Devstral-Small-2-24B-Instruct-2512 --local-dir Devstral-Small-2-24B-Instruct-2512
```

### Run the original model
Lets run the original model to get some output logits to compare visually against
later.
```console
(venv) $ cd examples/model-conversion
(venv) $ make causual-run-original_model
Input tokens: tensor([[22177,  1044,  2036,  2564,  1395]], device='cuda:0')
Input text: 'Hello, my name is'
Tokenized: ['Hello', ',', 'Ġmy', 'Ġname', 'Ġis']

Logits shape: torch.Size([1, 5, 131072])
Last token logits shape: (131072,)
Vocab size: 131072
First 10 logits: [-7.5    -7.5      2.      -7.5     -7.5     -7.5     -7.5     -7.5     -7.5     -7.09375]
Last 10 logits: [-4.90625 -4.90625 -6.40625 -8.4375  -6.78125 -6.15625 -5.0625  -7.09375 -5.375   -6.3125 ]
Top 5 predictions:
  Token 1278 (' the'): 5.968750
  Token 1032 (' '): 5.468750
  Token 1766 (' ['): 5.125000
  Token 2129 (' “'): 5.062500
  Token 1261 (' a'): 5.000000
Saved bin logits to: data/pytorch-Devstral-Small-2-24B-Instruct-2512.bin
Saved txt logist to: data/pytorch-Devstral-Small-2-24B-Instruct-2512.txt
```

But if I run this using the mistral transformers package I get different logits:
```console
(venv) $ python run-devstral.py
Unrecognized keys in `rope_parameters` for 'rope_type'='yarn': {'max_position_embeddings'}
/home/dbevenius/work/llama.cpp-staging/venv/lib/python3.12/site-packages/torch/cuda/__init__.py:285: UserWarning:
    Found GPU0 NVIDIA GB10 which is of cuda capability 12.1.
    Minimum and Maximum cuda capability supported by this version of PyTorch is
    (8.0) - (12.0)

  warnings.warn(
Loading weights: 100%|█| 1145/1145 [02:51<00:00,  6.68it/s, Materializing param=model.vision_tower.transformer.layers.23.ffn_norm.weig
Input tokens: tensor([[    1, 22177,  1044,  2036,  2564,  1395]], device='cuda:0')
Input text: 'Hello, my name is'
Tokenized: ['<s>', 'Hello', ',', ' my', ' name', ' is']
Processing chunk with tokens 0 to 512
Logits shape: torch.Size([1, 6, 131072])
Last token logits shape: (131072,)
Vocab size: 131072
First 10 logits: [-4.78125 -4.78125 -1.25    -4.78125 -4.78125 -4.78125 -4.78125 -4.78125
 -4.78125 -6.5    ]
Last 10 logits: [-4.03125  -3.46875  -4.5625   -5.625    -4.5      -4.0625   -3.53125
 -5.5      -1.921875 -4.46875 ]
Top 5 predictions:
  Token 1605 (' not'): 9.812500
  Token 2036 (' my'): 8.437500
  Token 1261 (' a'): 8.312500
  Token 1395 (' is'): 7.468750
  Token 1278 (' the'): 7.031250
```
And notice that the model actually does add a BOS token at the start of the input!

What is different between how our python run script runs the model versus
the one above?
This is the python script used above:
```python
import torch
from transformers import (
    Mistral3ForConditionalGeneration,
    MistralCommonBackend,
)
import numpy as np

model_path = "/home/dbevenius/models/Devstral-Small-2-24B-Instruct-2512"

tokenizer = MistralCommonBackend.from_pretrained(model_path, trust_remote_code=True)
model = Mistral3ForConditionalGeneration.from_pretrained(model_path, device_map="auto", trust_remote_code=True)

device = next(model.parameters()).device
prompt = "Hello, my name is"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

print(f"Input tokens: {input_ids}")
print(f"Input text: {repr(prompt)}")
print(f"Tokenized: {tokenizer.convert_ids_to_tokens(input_ids[0])}")

batch_size = 512

with torch.no_grad():
    past = None
    outputs = None
    for i in range(0, input_ids.size(1), batch_size):
        print(f"Processing chunk with tokens {i} to {i + batch_size}")
        chunk = input_ids[:, i:i + batch_size]
        outputs = model(chunk.to(model.device), past_key_values=past, use_cache=True)
        past = outputs.past_key_values

    logits = outputs.logits # type: ignore

    # Extract logits for the last token (next token prediction)
    last_logits = logits[0, -1, :].float().cpu().numpy()

    print(f"Logits shape: {logits.shape}")
    print(f"Last token logits shape: {last_logits.shape}")
    print(f"Vocab size: {len(last_logits)}")

    # Print some sample logits for quick verification
    print(f"First 10 logits: {last_logits[:10]}")
    print(f"Last 10 logits: {last_logits[-10:]}")

    # Show top 5 predicted tokens
    top_indices = np.argsort(last_logits)[-5:][::-1]
    print("Top 5 predictions:")
    for idx in top_indices:
        token = tokenizer.decode([idx])
        print(f"  Token {idx} ({repr(token)}): {last_logits[idx]:.6f}")
```

### Model Conversion setup
The following package versions are required for the conversion:
```console
(venv) $ pip install sentence-transformers==5.2.0
(venv) $ pip install transformers==5.0.0rc0
(venv) $ pip install mistral-common
```

### Model Conversion
```console
(venv) $ export MODEL_PATH=~/models/Devstral-Small-2-24B-Instruct-2512
(venv) $ make-convert-model-bf16
```

```console
(venv) $ export CONVERTED_MODEL=/home/dbevenius/work/llama.cpp-staging/models/Devstral-Small-2-24B-Instruct-2512.gguf
```

### Run the converted model
```console
(venv) $ make causal-run-converted_model
...
Input prompt: "Hello, my name is"
Tokenized prompt (6 tokens): <s> (1)Hello (22177), (1044) my (2036) name (2564) is (1395)
Vocab size: 131072
Saving data to data/llamacpp-Devstral-Small-2-24B-Instruct-2512.bin
First 10 logits: -6.039290 -6.039550 0.213944 -6.042861 -6.039347 -6.043080 -6.039300 -6.039615 -6.039384 -3.323704
Last 10 logits: -4.996169 -3.164495 -3.793222 -6.617929 -4.842769 -5.592521 -3.666847 -4.626977 -5.998837 -5.684978
```

_wip_

### Model verification
