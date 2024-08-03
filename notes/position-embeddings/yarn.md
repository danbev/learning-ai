## Yet Another RoPE extensioNs method (YARN)
Absolute positioning encoding cannot can't handle context lenghts that are
larger than the context length that they were trained on, or at least the once
they get larger. The absolute positional encoding bakes the position into the
embedding matrix. So the llm will learn/memorize the positioning which means
that it will not handle larger lengths very well.

There are limitation with the other models like absolute sinusoidal position
and the more advanced relative positional encoding schemes like T5 Relative
Bias, RoPE, XPos, and ALiBi. YARN attempts to address their inability to
generalize beyond the context window seen during training.

YARN is designed to extend the context lengths of models trained with RoPE.
RoPE allows the model to learn the relative positions of tokens and can handle
larger context lenghts, but does not handle extrapolation very well.
Now, recall that this limit is set during the training phase and is encoded into
the model's architecture, including its positional embeddings. So the model
will have this information baked into to and the weights will reflect this
information. 

### Position Interpolation
So models are typically trained on a specific context length which is the number
of tokens it can consider at once. This is a constraint in that the model
cannot (accurately) understand of generate larger sequences that that.

Simply increasing the context length, which is called direct extrapolation
usually performs poorly as the model has not learned how to handle longer
sequences. Similary just increasing the number of position indexes does not
work for the same reason.

So, lets say we have a model that has been trained on a context length of 512
and we want to increase the context length to 1024.
What we do is we take the current length that the model was trained on, that is
512 and stretch this out to cover the new length of 1024. So we interpolate
between the current length and the new length. The idea is to map the original
512 positions to the new 1024 positions in a way that maintains the relative
order and spacing.

After this modification the model will be fine-tuned using larger context
lengths.

[paper]: https://arxiv.org/pdf/2309.00071v1.pdf
