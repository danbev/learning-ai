## ggml submodule task
This task is about making ggml a git submodule in whisper.cpp.

Things to take into account:
* When we clone we need initialize the submodules (thinking about CI jobs)
 * git clone --recurse-submodules 

* Also documentation needs to be updated.


### Steps
1) Remove ggml source tree
2) Create git submodule:
```console
$ git submodule add https://github.com/ggml-org/ggml
```

3) Update the submodule:
```
$ git submodule update --remote ggml
```
