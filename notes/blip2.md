## Bootstraping Language-Image Pretraining with Frozen Image Encoders and Large Language Models (BLIP-2)
Uses a frozen encoder, like ViT, and a frozen LLM, so none of the models are
fine-tuned (updated during training). Instead the have a lightweight transformer
called Q-Former.

### Q-Former (Querying Transformer)
Q-Former is a lightweight transformer which employs a set, so always the same
number, of learnable query vectors to extract visual features from the frozen
image encoder.

The queries can be view as takes on the image, like segment an "object", count
the number of birds in the image, etc. But these are embeddings so they are
just number but this is something that has been learned during training. The
tasks would be things that come up again an again when trying to relate text to
an image. We than use attention cross attention to relate the queries to the
patch embeddings.

The paper specifies 32 queries of dimension 768, so think of a matrix of 32x768.
The number of parameters of the Q-Former is 188M.

