## llama.vim
This document contains notes around the vim plugin `llama.vim`.

### Installation
I'm using pack and this is simply a matter or cloning the
[llama.vim](https://github.com/ggml-org/llama.vim):
```console
$ git clone https://github.com/ggml-org/llama.vim ~/.vim/pack/github/start/llama.vim
```

We need a llama-server running with a model that can handle the format that
is expected by the plugin. A server can be started with the following command:
```console
$ ./build/bin/llama-server --fim-qwen-3b-default --no-warmup
```
With this in place we can start using the plugin by opening a file with vim
and start typing.
