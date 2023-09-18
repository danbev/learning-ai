## Hugging face notes
Note about Hugging Face libraries.

### Accellerate
Is a library which provides distributed learning, which is when you train a
model accross multiple GPUs or nodes.

It also provides mixed precision training which is when you use 16-bit
and 32-bit precision arithmetic.

And it has a data parallelism feature which is when you split the data across
multiple GPUs by dividing the data into batches and then each GPU processes
a batch. Note that this is about data and note the same as the distrubuted
training.
