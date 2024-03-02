## Groq
Is a company that I've heard migth have been founded by a number of former
Google employees that worked on Googles TPU (Tensor Processing Unit). They have
developed something called a Language Processing Unit (LPU), so a piece of
hardware similar to the TPU specifically for language processing.

They have an inference engine which we can try:
https://groq.com/

And it is indeed super fast. This engine runs models which are currently only
two Mixtral and Llama.

The cards which contains the LPUs don't have much memory, like 200MB of SRAM, on
them so to serve large language models they have to be distributed across
multiple cards (like around 350). So you need many of them which might be an
issue to produce and distribute and buy if they are expensive. Compare this to
GPUs which have a lot of memory on them and can run large models on a single
card or a few cards.
