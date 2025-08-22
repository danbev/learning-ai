## GGML Version handling
This task is about implementing semantic versioning in GGML.

### Background
Currently the version of GGML is generated using git and is a build number (number
of commits):
```console
$ git rev-list --count HEAD
2471
```
And the version might then look like this:
```console
Version: 0.0.2471
```

### Considerations

Currently, the main development of GGML happens in llama.cpp and changes are
synced back to GGML when they are mature enough. This is the current workflow
and something that the version management should take into account.

So we have two main use cases for GGML:
* As a copy of the source code in another project (llama.cpp, whisper.cpp)
* As a submodule in another project


### Semantic Versioning
The idea is to introduce semantic versioning, which consists of three mandatory
parts: `major`, `minor`, and `patch`.

The suggested version format is:
```
major.minor.patch[-dev][-dirty]
```
So for a release build:
```console
0.9.0
```
And for development builds:
```console
0.9.0-dev
```
And for development builds with modifications to ggml sources:
```console
0.9.0-dev-dirty
```

### Release management
To make a new release the following steps would be taken:
* Update the version in ggml/CMakeLists.txt changing the version appropriately, and remove
  the '-dev' suffix.
* Commit the changes to ggml/CMakeLists.txt with a message like "ggml : bump version to 1.0.0"
* Create a new git tag with the version number and push the tag
* Update the version in ggml/CMakeLists.txt increment the minor or patch version
  and set the `-dev` suffix part. Perhaps to `1.1.0-dev` or `1.0.1-dev`


### Checking the version
```console
$ cat << 'EOF' | gcc -Iggml/include -L build/bin -x c -o test_ggml - -lggml-base && LD_LIBRARY_PATH=build/bin ./test_ggml && rm test_ggml
#include "ggml.h"
#include <stdio.h>

int main() {
    printf("ggml version: %s\n", ggml_version());
    return 0;
}
EOF
ggml version: 0.9.0-dev-dirty
```

And using llama-cli to check the version:
```console
$ ./build/bin/llama-cli --version
version: 6242 (4b80a93ed)
built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu
ggml version: 0.9.0-dev-dirty
```
