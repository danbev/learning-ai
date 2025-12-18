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
(venv) $ pushd examples/model-conversion
(venv) $ make-convert-model-bf16
Model path: /home/dbevenius/models/Devstral-Small-2-24B-Instruct-2512
Model name: Devstral-Small-2-24B-Instruct-2512
Data  type: bf16
Converted model path:: ../../models/Devstral-Small-2-24B-Instruct-2512.gguf

Metadata override:
INFO:hf-to-gguf:Loading model: Devstral-Small-2-24B-Instruct-2512
Unrecognized keys in `rope_parameters` for 'rope_type'='yarn': {'max_position_embeddings'}
INFO:hf-to-gguf:Model architecture: Mistral3ForConditionalGeneration
Unrecognized keys in `rope_parameters` for 'rope_type'='yarn': {'max_position_embeddings'}
...
INFO:hf-to-gguf:Set model parameters
INFO:hf-to-gguf:gguf: context length = 393216
INFO:hf-to-gguf:gguf: embedding length = 5120
INFO:hf-to-gguf:gguf: feed forward length = 32768
INFO:hf-to-gguf:gguf: head count = 32
INFO:hf-to-gguf:gguf: key-value head count = 8
INFO:hf-to-gguf:gguf: rope scaling type = YARN
INFO:hf-to-gguf:gguf: rope theta = 100000000.0
INFO:hf-to-gguf:gguf: rms norm epsilon = 1e-05
INFO:hf-to-gguf:gguf: file type = 32
...
DEBUG:hf-to-gguf:chkhsh: 0e9433cbbb161f89e264eb32e8e64bfe69e834973ffca5d41d3948a604a3e2a3
DEBUG:hf-to-gguf:tokenizer.ggml.pre: 'pixtral'
DEBUG:hf-to-gguf:chkhsh: 0e9433cbbb161f89e264eb32e8e64bfe69e834973ffca5d41d3948a604a3e2a3
...
Writing:   0%|                                                                                         | 0.00/47.1G [00:00<?, ?byte/s]
/home/dbevenius/work/llama.cpp-staging/examples/model-conversion/../../convert_hf_to_gguf.py:10432:
UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors.
This means writing to this tensor will result in undefined behavior. You may want to copy the array
to protect its data or make it writable before converting it to a tensor. This type of warning will
be suppressed for the rest of this program. (Triggered internally at /pytorch/torch/csrc/utils/tensor_numpy.cpp:213.)
  return torch.from_numpy(byteswap_tensor(tensor.mmap_bytes(), numpy_dtype)).view(dtype).reshape(tensor.shape)
```
I don't think I've see the above warning before.

```console
(venv) $ export CONVERTED_MODEL=/home/dbevenius/work/llama.cpp-staging/models/Devstral-Small-2-24B-Instruct-2512.gguf
```

### Run the converted model
```console
(venv) $ make causual-run-converted_model
Input prompt: "Hello, my name is"
Tokenized prompt (6 tokens): <s> (1)Hello (22177), (1044) my (2036) name (2564) is (1395)
```
Notice that this is using 6 tokens whereas the original model run used 5 tokens.
So the original version is not adding a beginning of sequence token but our
converted model is.

### Tokenization
Like mentioned above the converted model is adding a beginning of sequence token
but the original model is not. 
There is no `add_bos` parameter in the models tokenizer_config.json file but
in the conversion the tokenizer pre type is being set to `pixtral`:
```console
DEBUG:hf-to-gguf:tokenizer.ggml.pre: 'pixtral'
```
And if we look in src/llama-vocab.cpp we can see the following:
```c++
            } else if (
                    tokenizer_pre == "llama3"   ||
                    tokenizer_pre == "llama-v3" ||
                    tokenizer_pre == "llama-bpe"||
                    tokenizer_pre == "falcon3"  ||
                    tokenizer_pre == "falcon-h1" ||
                    tokenizer_pre == "pixtral"  ||
                    tokenizer_pre == "midm-2.0" ||
                    tokenizer_pre == "lfm2") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_LLAMA3;
                ignore_merges = true;
                add_bos = true;
```
So this is why the converted model is adding a beginning of sequence token.
If we add the following line to the conversion script:
```python
self.gguf_writer.add_add_bos_token(False)
```
Then the converted model will not add the beginning of sequence token and this
is actually done for at least one other model conversion:

```console
Input prompt: "Hello, my name is"
Tokenized prompt (5 tokens): Hello (22177), (1044) my (2036) name (2564) is (1395)
Vocab size: 131072
Saving data to data/llamacpp-Devstral-Small-2-24B-Instruct-2512.bin
First 10 logits: -6.455563 -6.455366  0.609470 -6.458275 -6.455173 -6.458473 -6.455353 -6.455137 -6.455266 -4.424087
Last 10 logits:  -4.885547 -3.908254 -4.559022 -6.890335 -5.780299 -5.246257 -4.287138 -5.208242 -5.936634 -5.580862
```
```console
Input text: 'Hello, my name is'
Input tokens: tensor([[22177,  1044,  2036,  2564,  1395]], device='cuda:0')
First 10 logits: [-7.5     -7.5      2.      -7.5     -7.5     -7.5     -7.5     -7.5     -7.5     -7.09375]
Last 10 logits:  [-4.90625 -4.90625 -6.40625 -8.4375  -6.78125 -6.15625 -5.0625  -7.09375 -5.375   -6.3125 ]
```


### Model verification
```console
🔍 GGML Model Validation for model  Devstral-Small-2-24B-Instruct-2512
========================================
PyTorch logits  : data/pytorch-Devstral-Small-2-24B-Instruct-2512.bin
llama.cpp logits: data/llamacpp-Devstral-Small-2-24B-Instruct-2512.bin

Top 10 PyTorch logits: [5.96875 5.46875 5.125   5.0625  5.      4.90625 4.6875  4.59375 4.34375
 4.34375]
Top 10 llama.cpp logits: [7.0536804 6.418002  6.1969543 6.1952853 5.5399556 5.4969754 5.3937917
 4.834832  4.829475  4.7304   ]
Max absolute difference: 5.9784
✅ OK: Lightweight model check successful!
       Ok to proceed with NMSE check...
Model name: Devstral-Small-2-24B-Instruct-2512
PyTorch logits file: data/pytorch-Devstral-Small-2-24B-Instruct-2512.bin
llama.cpp logits file: data/llamacpp-Devstral-Small-2-24B-Instruct-2512.bin
📊 NMSE Check for Model Comparison
==================================================
Reference (ground truth): data/pytorch-Devstral-Small-2-24B-Instruct-2512.bin
Test (to evaluate):       data/llamacpp-Devstral-Small-2-24B-Instruct-2512.bin

Loading reference logits...
  Shape: (131072,), Type: float32
Loading test logits...
  Shape: (131072,), Type: float32

✅ Shapes match: (131072,)

📈 METRICS
==============================
MSE (Mean Squared Error):     1.178915e+00
Reference Variance:           2.508655e+00
NMSE:                         4.699391e-01
Max Absolute Error:           5.978355
Mean Absolute Error:          0.864473
NMSE (dB):                    -3.28 dB

🎯 INTERPRETATION
==============================
❌ Poor match

📋 GUIDANCE
==============================
❌ PROBLEMATIC: Large differences detected.
   Check your conversion process for potential issues.
   Verify you're using the same model weights.

📚 NMSE BENCHMARKS
==============================
< 1e-6:  Essentially identical
< 1e-4:  Excellent (typical for good conversions)
< 1e-3:  Very good
< 1e-2:  Good (acceptable for most use cases)
< 0.1:   Acceptable (may need verification)
> 1.0:   Poor (worse than random)

❌ RESULT: NEEDS REVIEW (NMSE = 4.70e-01)
make: *** [Makefile:63: causal-verify-logits] Error 1
```
_wip_
