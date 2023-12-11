## Attention with Linear Biases Enables Input Length Extrapolation (Alibi)
This is a replacement of position embeddings for transformers. Recall that the
extrapolation issue is when we have trained a model with a certain sequence
length, and then at inference time the input lengths exceed the maximum length.
With absolut positional encoding the model learns the position during training,
remember that the position are added to the vector representation of the input
tokens (the embeddings). Because of this is the inference input exceeds the
training max length the model will often have trouble with the input and come up
with garbage output.
