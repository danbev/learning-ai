## A Length-Extrapolatable Transformer (XPos RoPE)
This is an extension of RoPE at addresses some of the limitations of RoPE.

### Background
A notable challenge with traditional Transformer models is their inability to
effectively handle input sequences of varying lengths, especially those longer
than the lengths seen during training.

Position Embedding is crucial in sequence modeling for Transformers, enabling
the model to understand the order and relative positioning of elements in a
sequence. Initially, Transformers used absolute position embeddings, but these
were later found to be less efficient than relative position embeddings like
RoPE, which offered better performance. But RoPE has its limitations, especially
when it comes to handling sequences longer than those seen during training.


### XPos
So XPos is similar to RoPE but also handles long-term dependencies better and
sequences of varying lengths.
XPOS enables Transformers to handle input sequences of any length without the
need for re-training, addressing a significant limitation of both absolute and
some relative position embedding methods.

It does so through an innovative approach that combines an attention resolution
metric for measuring position monotonicity and a generalization of the
mathematical formulation of position embedding that includes an exponential
decay factor in the rotation matrix.



Paper: https://arxiv.org/pdf/2212.10554.pdf
