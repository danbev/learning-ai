## Accurate Post-Training Quantization for Generative Pre-Trained Transformers
GPTQ is a post-training quantization method for 4-bit quantization that
focuses primarily on GPU inference and performance.

The weights of a model are stored with a specific precision, and saved to some
type of storage, like a file. The precision of each of these weights will
determine the size of the file and also how much memory is required to read
the model into memory. If we can get away with a lower precision we can save
on memory and storage space. This is called quantanisation.

The options we have are to store with 32-bit precision, 16-bit precision, 
8-bit precision, or 4-bit precision. The lower the precision the smaller the
file size and the less memory required to read the model into memory. But won't
the lower precision make the model less accurate?  
Yes, but if we can use compression to get the bit size down and still keep the
precision, we will have the best of both worlds. This is what quantanisation
aims to do.

So we have the weights, bias, and activations that are generated during training
and these can be stored in 32-bit, 16-bit, 8-bit, or 4-bit precision after
training, or  we can quantanise the weights, bias, and activations of a
pre-trained model.

Ranges of values for different precisions:
```
FP32   -34*10^38 to 34*10^38
FP16   -65504 to 65504
INT8   -128 to 127
INT    -8 to 7
```

### Post-training quantanized models
The most popular models are `GGML` and `GPTQ`. `GGML` is a post trained
quantanised model. This means that the model is trained with 32-bit precision
and then quantanised to 8-bit precision sometime afterward.

#### GGML (Georgi Gerganov Machine Learning)
GGML models are optimized for `CPU`s. So the inference is done on CPUs and is
faster on CPUS. The models can be slightly larger than GPTQ models.
"GG" refers to the initials of its originator (Georgi Gerganov) and I think
ML is just for machine learning. It is a [C library](https://github.com/rustformers/llm/blob/main/crates/ggml/README.md).

GPTQ's are optimized `GPU`s. So the inference is done on GPUs and is faster on
GPUs.

So depending on your execution environment you should choose the model that is
optimized for that environment.

For example, on hugging face you might see multiple versions of a model that
have been build/optimized using GGML or GPTQ:
```
https://huggingface.co/TheBloke/Llama-2-7b-Chat-GPTQ
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML
```
This was something that has confused be in the past so it was nice to finally
get an answer to what this suffixes mean.

There is also a new format called
[GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) which is a
replacement for GGML.

### AutoGPTQ
This is a library from Huggingface which enables you to quantanise a pre-trained
transformer model. There are other libraries that are specific to certain
models but AutoGPTQ is a general library that can be used with many models.

I'm just including the pip install command so that I can see the actual name
of the python package.
```console
$ pip install auto-gptq
```
