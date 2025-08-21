## GGML Version handling
This task is about implementing semantic versioning in GGML.

### Background
Currently the version of GGML is generated using git:
```console
$ git rev-list --count HEAD
2471
```
And the version would then look like this:
```console
Version: 0.0.2471
```

### Semantic Versioning
So the idea is to introduce semantic versioning, which consists of three parts:
major, minor, and patch.

The suggested version format is:
```
major.minor.patch[-dev].[commit][-dirty]
```
So for a release build:
```console
1.0.0
```
And for development builds:
```console
1.0.0-dev.029bb39eb
```
And for development builds with modifications to ggml sources:
```console
1.0.0-dev.029bb39eb-dirty
```

```console
$ cat << 'EOF' | gcc -Iggml/include -L build/bin -x c -o test_ggml - -lggml-base && LD_LIBRARY_PATH=build/bin ./test_ggml && rm test_ggml
#include "ggml.h"
#include <stdio.h>

int main() {
    printf("ggml version: %s\n", ggml_version());
    return 0;
}
EOF
ggml version: 1.0.0-dev.22bd5645c
```

### Release management
To make a new release the following steps should be taken:
* Update the version in CMakeLists.txt changing the version appropriately.
* Commit the changes to CMakeLists.txt with a message like "ggml : bump version to 1.0.0".
* Create a new git tag with the version number and push the tag

### Considerations
GGML is often uses in other project by including ggml as either a submodule or
as a copy of the source code. And we need to take this into account.

For users that include GGML as a submodule they would be able to choose a tagged
release.

For users that copy GGML into there source code tree they would need to manually
update to a new version, perhaps also using a tagged release or a commit hash
depending on their needs.
But lets say they copy GGML into their source code tree and use a tagged release,
but then they make modifications to GGML but without updating the version. This
would lead to confusion as the version would not reflect the changes made.

This is a little more difficult than normal projects as the main development
of GGML actually happens in llama.cpp, so this is where the most changes would
be made. So new features are added and tested in llama.cpp, when they changes
are mature enough to be release they will be synced with GGML, and perhaps this
it when the version should be bumped?

### Questions
* When/where should the version management be done?
  This is a little more difficult than normal projects as the main development
  of GGML actually happens in llama.cpp, so this is where the most changes would
  be made. So new features are added and tested in llama.cpp, when the changes
  are mature enough to be release they will be synced with GGML, and perhaps this
  it when the version should be bumped? Or perhaps it does not matter as long as
  when the sync is a release then the tagging needs to be done in ggml.
* And llama.cpp should probably also cut a release when it uses a release of
  GGML, so that the version in llama.cpp matches the version in GGML before
  continuing development.
