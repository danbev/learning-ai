## LLama.ccp exploration/example project
This project is an exploration of the LLama.cpp library.

#### Configuration
This project uses a git submodule to include the LLama.cpp library. To
add the submodule run (only for documenation purposes):
```console
$ git submodule add https://github.com/ggerganov/llama.cpp llama.cpp
$ git submodule update --init --recursive
```

To update the submodule run:
```console
$ git submodule update --recursive --remote
```

### ctags
```console
$ ctags -R --languages=C++ --c++-kinds=+p --fields=+iaS --extra=+q .
```
