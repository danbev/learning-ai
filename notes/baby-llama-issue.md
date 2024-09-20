## baby-llama issue

I ran into when investigating an issue related to the baby-llama example in
llama.cpp. When stepping through the code to understand the original issue I
notice the following in `init_mode`:
```console
$ gdb --args ./llama-baby-llama
Reading symbols from ./llama-baby-llama...
(gdb) br init_model
Breakpoint 1 at 0x251767: file examples/baby-llama/baby-llama.cpp, line 190.
(gdb) r
...
(gdb)
204	    model->layers.resize(n_layer);
```
Now, if we inspect the size of `model->layers` we see that it is 0:
```console
(gdb) p model->layers.size()
$1 = 0
```
And also `n_layer` is 1:
```console
(gdb) p n_layer
$3 = 1
```
And we can inspect the type of `model->layers`:
```console
(gdb) ptype model->layers
type = std::vector<llama_layer>
```

Now if we step over the resize function we will see something interesting:
```console
(gdb) p model->layers.size()
$2 = 12
```
I also added two print statements to show the size of `n_layer` and
`model->layers.size()`:
```console
n_layer: 1
layers.size(): 2049638230412172414
```
After some digging into this a little and first thinking that this was something
to do with the `resize` function in combination with the `llama_layer` I later
realized that this was actually a symbol conflict. There is a `llama_layer` in
llama.cpp and this object file is compiled into the `llama-baby-llama` binary:
```console
/usr/bin/ccache c++ -std=c++11 -fPIC -O0 -g -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wmissing-declarations -Wmissing-noreturn -pthread -fopenmp  -march=native -mtune=native -Wno-array-bounds -Wno-format-truncation -Wextra-semi -Iggml/include -Iggml/src -Iinclude -Isrc -Icommon -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -D_GLIBCXX_ASSERTIONS -DGGML_USE_OPENMP -DGGML_USE_LLAMAFILE  ggml/src/llamafile/sgemm.o ggml/src/ggml.o ggml/src/ggml-alloc.o ggml/src/ggml-backend.o ggml/src/ggml-quants.o ggml/src/ggml-aarch64.o src/llama.o src/llama-vocab.o src/llama-grammar.o src/llama-sampling.o src/unicode.o src/unicode-data.o common/common.o common/arg.o common/log.o common/console.o common/ngram-cache.o common/sampling.o common/train.o common/build-info.o common/json-schema-to-grammar.o examples/baby-llama/baby-llama.o -o llama-baby-llama -g
```
This could be worked around by renaming the `llama_layer` baby-llama.cpp to
use `baby_llama_layer` instead, use a namespace in baby-llama.cpp, or remove
the object files that are not needed for the baby-llama example. I think the
intent of the baby-llama example is to be a standalone example, not actually
using llama.cpp, so the latter option might be the best. Actually it looks like
the baby-llama example uses `train.h` and might need llama.cpp indirectly, so
perhaps just renaming the `llama_layer` in baby-llama.cpp is the best option.

Renaming the `llama_layer` in baby-llama.cpp to `baby_llama_layer`:
```console
(gdb) p model->layers.size()
$1 = 0
(gdb) ptype model->layers
type = std::vector<baby_llama_layer>
(gdb) p model->layers.size()
$2 = 1
```

