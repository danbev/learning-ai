## Joint-Embedding Predictive Architecture (JEPA)
The Joint-Embedding part is simliar to what we discussed in [llava](llava.md)
where we have a multipmodal model and an image would be processed by a
[Vision Transfomer](vit.md) which would produce a patch embeddings for an image
which are then projected into the embedding space of the LLM. This was image
and text embeddings can be passed to the LLM at the same time, that is in the
same sequence of token embeddings. And an image of a cat and the word "cat"
would be close in the embedding space. So that is what the joint-embedding part
means which was not clear to me when I started reading about JEPA.

### Energy Based Models (EBM)
Is a trainable system that can take two inputs x and y and tell us how
incompatible they are. The output of this function is called the energy.
For example if x is a video snippet than y would be how good of a continuation
it would be. The energy would be high if the continuation is not good. The
Why is it called energy?  
This is a term from physics. The energy function is a measure of how much the
system wants to change.
```
energy = f(x, y)
```
energy is high. This sounds like a loss function?


