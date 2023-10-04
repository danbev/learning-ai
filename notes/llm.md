## Large Language Models
Are language models with many parameters, like over 100M, and are pre-trained
on large collections of text (corpra).

This page tries to create an overview of the various large language models and
which companies/orginizations are behind them.

### OpenAI
* GPT-2
* GPT-3
* GPT-4

### Google
* BERT
* T5
* PaLM
* PaLM2

### Facebook
* Llama
* Llama2 (Almost opensource I think)
* RoBERTa
* XLM-R
* XLM-RoBERTa

### EleutherAI
* GPT-Neo
* GPT-NeoX
* GPT-J

### Hugging Face/BigScience
* Bloom

### MosaicML
[MosaicML](https://www.mosaicml.com/) offers models that are completely free
to use even for commercial purposes. 
* MPT-7B (MosaicML Pretrained Transformer) ChatGPT style decoder only.

### Technology Innovation Institute (TII)
* Falcon opensource

## AI Chat applications
* ChatGTP (from OpenAI)
* Bard (from Microsoft)


### Using an OpenSource LLM
The examples that can be found in this repo have been using OpenAI which means
that it has a cost associated with it. So I've been looking into using an
opensource LLM instead. Above we have a couple of alternatives but one thing
to keep in mind is that while these can be run standalone it may not be very
performant to do so. Running one locally would require a GPU or the performance
of it would be very poor. There are cloud providers that offer the ability to
run these models in the cloud and then you can use the API to interact with
like [Replicate](https://replicate.com/) but they also cost money which is what
I'm trying to avoid in this case.

I'm going to investigate using llama.cpp as the inference engine and see if I
can get a chat ui to run against it, all locally.


