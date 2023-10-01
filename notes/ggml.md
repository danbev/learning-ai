## GGML (Georgi Gerganov Machine Learning)
"GG" refers to the initials of its originator (Georgi Gerganov) and I think
ML is just for machine learning. It is a
[C library](https://github.com/rustformers/llm/blob/main/crates/ggml/README.md).

This is a basic example in [ggml/c-example](fundamentals/ggml/c-example/README.md)
of how to use GGML.

For me it help to draw parallels between GGML and part1 of zero-to-hero, where
we created a Value struct which in addition of holding a value, would also have
a gradient, and an operation and children if the value was created by an
operation. The Value struct also supported autoamtic differentiation. In a
simliar manner GGML has a Tensor struct which holds a value, a gradient, and
an operation and a src array which is simlar to the children in zero-to-hero in
that is would contain the left hand side and right hand side of an operation.
The Tensor struct also supports automatic differentiation and also the ability
do create dot draphs. GGML contains more than that but then tensor is the basic
structure.

