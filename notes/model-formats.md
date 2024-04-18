## LLM Model formats
I've been very confused about the different model formats and their
configurations. Different frameworks have different ways of storing and
configuring models and to run these in llama.cpp they have to be converted to
GGUF format. I've not really known what convertion script to run just be looking
at the files in the models directory or the file extensions. This page is an
attempt to clarify this.

### Tokenizer configuration
The tokenization process involves splitting the input text into tokens, which
are the looked up in the vocabulary to get the token IDs.
Models include a vocabulary file, which is a list of all the tokens in the
model. There might be a configuration file in addition to this that specifies
the type of tokenizing that the model uses, like Byte-Pair Encoding (BPE),
WordPiece, SentencePiece, or Unigram, etc.

`tokenizer.json` is a JSON file contains the tokenizer's configuration and
vocabulary, mapping tokens to their respective IDs.

`tokenizer_config.json` is a JSON file that contains the tokenizer's settings
such as whether to lower case the input text, what special tokens to use, etc.
It complements the tokenizer.json file by providing additional configuration
necessary for the tokenizer's operation.

`vocab.txt` is a text file that lists all tokens in the model's vocabulary, each
on a new line.

`merges.txt' is a text file that is used with Byte-Pair Encoding (BPE) and
similar tokenizers. This file contains rules for how to combine subword units
, or tokens, into larger units during tokenization.

### TensorFlow
Uses the SavedModel format, which is a serialized model that can be loaded into
TensorFlow. 
A model is saved using the `tf.saved_model.save` function and you specify the
directory where the model will be saved. This directory will contain the
following files and directories:

* assets: directory which contains file like vocab files, etc.
* variables: directory which contains the model weights and checkpoints
* saved_model.pb: stores the actual TensorFlow program, or model, and a set of
  named signatures, each identifying a function that accepts tensor inputs and
  produces tensor outputs.

tf_model.h5 is a file that contains both the model weights and architecture in a
single file, making it easy to load the entire model with TensorFlow or Keras.

### PyTorch
Uses a `state_dict` which is a Python dictionary object that maps each layer to
its parameter tensor. 
The models are often stored with a `.pth` or `.pt` file extensions and this is
a single file compared to the TensorFlow format which is a directory.

Checkpoints are stored with the `.ckpt` file extension and are used to save the
state of the model and can be used to continue training from a certain point.

This often used pickle to serialize the model and then save it to a file, which
can serialize an object model to file and then load it back to memory later. Is
is not recommended to use pickle to save and load models as it is not secure and
can lead to security vulnerabilities, like arbitrary code execution. Instead
[safetensors] is recommended, which was developed by Hugging Face. Safetensors
an be used to serialize tensors in Python, and the resulting files can be easily
loaded in other languages and platforms, including C++, Java, and JavaScript and
provide a checksum mechanism to ensure that serialized tensors are not corrupted
during storage or transfer.
Savetensors for models might optionally include a model.safetensors.index.json
which provides metadata and an index of names or weight tensors to which 
.safetensors files they correspond to.

Pre-trained models are often stored as `pytorch_model.bin` files which contain
the pre-trained weights.


### ONNX
Models are stored in a framework-independent format using a .onnx file
extension.

Uses protobuf to serialize the model and save it to a file.

### Flax/JAX
Stores models with a .msgpack extension.

[safetensors]: https://github.com/huggingface/safetensors


