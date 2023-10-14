## LLama.ccp exploration/example project
This project is an exploration of the LLama.cpp library. The goal it to have
small isolated examples that can be run and debugged in isolation to help
understand how the library works.

#### Initial setup
To update the submodule run:
```console
$ git submodule update --recursive --remote
```

### Debugging
The examples in this project can be build with debug symbols enabled allowing
for exploration of the llama.cpp, and ggml.cpp libraries. For example:
```console
(langch) $ !gdb
gdb --args ./simple-prompt 
Reading symbols from ./simple-prompt...

(gdb) br simple-prompt.cpp:7
Breakpoint 1 at 0x410bff: file src/simple-prompt.cpp, line 7.

(gdb) r
Starting program: /home/danielbevenius/work/ai/learning-ai/fundamentals/llama.cpp/simple-prompt 
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib64/libthread_db.so.1".

Breakpoint 1, main (argc=1, argv=0x7fffffffc118) at src/simple-prompt.cpp:7
warning: Source file is more recent than executable.
7	    gpt_params params;
```

### ctags
To generate ctags for the project run:
```console
$ ctags -R --languages=C++ --c++-kinds=+p --fields=+iaS --extra=+q .
```

#### Configuration
This project uses a git submodule to include the LLama.cpp library. To
add the submodule run (only the first time and I'm just adding this for
documenation purposes):
```console
$ git submodule add https://github.com/ggerganov/llama.cpp llama.cpp
$ git submodule update --init --recursive
```
