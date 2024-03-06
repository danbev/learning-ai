## Bootstraping Language-Image Pretraining with Frozen Image Encoders and Large Language Models (BLIP-2)
Uses a frozen encoder, like ViT, and a frozen LLM, so none of the models are
fine-tuned (updated during training). Instead they have a lightweight
transformer called Q-Former.

### Q-Former (Querying Transformer)
Q-Former is a lightweight transformer which employs a set, so always the same
number, of learnable query vectors to extract visual features from the frozen
image encoder.

The queries can be view as takes on the image, like segment an "object", count
the number of birds in the image, etc. But these are embeddings so they are
just numbers, but this is something that has been learned during training. The
tasks would be things that come up again an again when trying to relate text to
an image. We then use attention cross attention to relate the queries to the
patch embeddings.

The paper specifies 32 queries of dimension 768, so think of a matrix of 32x768.
The number of parameters of the Q-Former is 188M.
So it looks like this is actually restrict the number of tokens to be limited
to 32, that is 32 patch embeddings each of a dimension of 768. So this is does
not reduce the dimensionality of the patch embeddings, but rather the number of
them.
