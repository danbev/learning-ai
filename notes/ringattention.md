## Ring Attention
This is an achitecture that enables context lenghts of up to and perhaps more
than 1 million tokens. It can handle long videos and language sequence
(multi-modal). This is done without approximations or additional overhead.

During training the context size are small, like 4K and then are gradually
increased as training progresses. The traing data consists of video, images, and
text.

The modify the positional embeddings parameters to account for the increased
sequence lengths.

Paper: https://arxiv.org/html/2402.08268v1
