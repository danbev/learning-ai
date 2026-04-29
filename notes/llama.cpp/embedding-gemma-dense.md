### EmbeddingGemma dense layers

The model that I converted and published to [ggml-org](https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF/)
does not contain the extra dense layers. These were skipped during the conversion
work which is my fault for not understanding their importance.
Inspecting the existing model we can see that these layers are not present:
```console
(venv) $ ./gguf-py/gguf/scripts/gguf_dump.py ~/Downloads/embeddinggemma-300M-Q8_0.gguf 
INFO:gguf-dump:* Loading: /home/danbev/Downloads/embeddinggemma-300M-Q8_0.gguf
* File is LITTLE endian, script is running on a LITTLE endian host.
   ...
* Dumping 314 tensor(s)
      1:        768 |   768,     1,     1,     1 | F32     | output_norm.weight
      2:  201326592 |   768, 262144,     1,     1 | Q8_0    | token_embd.weight
      3:     196608 |   768,   256,     1,     1 | Q8_0    | blk.0.attn_k.weight
      4:        256 |   256,     1,     1,     1 | F32     | blk.0.attn_k_norm.weight
      5:        768 |   768,     1,     1,     1 | F32     | blk.0.attn_norm.weight
```

Later, support for these layers was added to llama.cpp and the conversion scripts
can now handle them using the command line option `--sentence-transformers-dense-modules`
to convert_hf_to_gguf.py:
```python
    parser.add_argument(
        "--sentence-transformers-dense-modules", action="store_true",
        help=("Whether to include sentence-transformers dense modules. "
              "It can be used for sentence-transformers models, like google/embeddinggemma-300m. "
              "Default these modules are not included.")
    )
```

If we convert the model using this we can then see that the layers are present
in the model file:
```console
(venv) $ ./convert_hf_to_gguf.py ~/work/ai/models/google/embeddinggemma-300m --sentence-transformers-dense-modules --outtype f32
(venv) $ ./build-cpu-release/bin/llama-quantize ~/work/ai/models/google/embeddinggemma-300m/embeddinggemma-300M-F32.gguf Q8_0
(venv) $ ./gguf-py/gguf/scripts/gguf_dump.py /home/danbev/work/ai/models/google/embeddinggemma-300m/ggml-model-Q8_0.gguf
INFO:gguf-dump:* Loading: /home/danbev/work/ai/models/google/embeddinggemma-300m/ggml-model-Q8_0.gguf
* File is LITTLE endian, script is running on a LITTLE endian host.
* Dumping 44 key/value pair(s)
  ...
* Dumping 316 tensor(s)
      1:    2359296 |   768,  3072,     1,     1 | Q8_0    | dense_2.weight
      2:    2359296 |  3072,   768,     1,     1 | Q8_0    | dense_3.weight
      3:        768 |   768,     1,     1,     1 | F32     | output_norm.weight
      4:  201326592 |   768, 262144,     1,     1 | Q8_0    | token_embd.weight
      5:     196608 |   768,   256,     1,     1 | Q8_0    | blk.0.attn_k.weight
      6:        256 |   256,     1,     1,     1 | F32     | blk.0.attn_k_norm.weight
      7:        768 |   768,     1,     1,     1 | F32     | blk.0.attn_norm.weight
```
And the same PR added support in llama.cpp to also use these dense layers.
```c++
ggml_cgraph * llama_model::build_graph(const llm_graph_params & params) const {
    std::unique_ptr<llm_graph_context> llm;
    ...

    // if the gguf model was converted with --sentence-transformers-dense-modules
    // there will be two additional dense projection layers
    // dense linear projections are applied after pooling
    // TODO: move reranking logic here and generalize
    llm->build_dense_out(dense_2_out_layers, dense_2_out_layers_b, dense_3_out_layers);
```
So I think we should update the model in ggml-org.
