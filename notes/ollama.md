## Ollama

###  Models
Ollama stores models in an OCI registry which by default are stored in
~/.ollama/models.

### Using a local model
I wanted to run a gguf model that I have locally and this can be done by running
the following commands:

Make sure the ollama REST server is running
```console
$ cd ~/work/ai/ollama
$ ./ollama serve
```

Create a configuration file that describes the model:
```console
FROM /home/danbev/.ollama/models/llama-2-7b-hf-chat-q4.gguf

LICENSE apache-2.0

# Adjust the parameters based on your needs
PARAMETER stop "Human:"
PARAMETER stop "Assistant:"
PARAMETER temperature 0.7
PARAMETER top_k 40
PARAMETER top_p 0.5
```

Create the model in using the above configuration file (saved as ModelFile):
```console
$ ./ollama create mycustomllama -f Modelfile
```
Run the model:
```console
$ ./ollama run mycustomllama
```

### Inspecting a model
If a gguf model is used we can find the blob for the model by either inspecting
the metadata file:
```console
$ cat ~/.ollama/models/manifests/registry.ollama.ai/library/mycustomllama/latest | jq '.layers[] | select(.mediaType == "application/vnd.ollama.image.model") | .digest'
"sha256:42f685711f23fc73c4558da0d8df22fd18ecabbee9c6c8f8277204902ace10d3"
```
And we can use the digest (with  sha256- as the prefix instead of ':') to find the blob:
```console
$ ./inspect-model.sh ~/.ollama/models/blobs/sha256-42f685711f23fc73c4558da0d8df22fd18ecabbee9c6c8f8277204902ace10d3
INFO:gguf-dump:* Loading: /home/danbev/.ollama/models/blobs/sha256-42f685711f23fc73c4558da0d8df22fd18ecabbee9c6c8f8277204902ace10d3
* File is LITTLE endian, script is running on a LITTLE endian host.
* Dumping 35 key/value pair(s)
      1: UINT32     |        1 | GGUF.version = 3
      2: UINT64     |        1 | GGUF.tensor_count = 291
      3: UINT64     |        1 | GGUF.kv_count = 32
      4: STRING     |        1 | general.architecture = 'llama'
```

The blob is also shows in the servers console:
```console
llama_model_loader: loaded meta data with 32 key-value pairs and 291 tensors from /home/danbev/.ollama/models/blobs/sha256-42f685711f23fc73c4558da0d8df22fd18ecabbee9c6c8f8277204902ace10d3 (version GGUF V3 (latest))
```

### Building from source
```console
$ cd ~/work/ai/ollama
$ make -j 8
$ go build .
```

