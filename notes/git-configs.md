## git config branch
I've found it useful to create empty git branches in projects like llama.cpp
for things like CMakeUserPresets.txt and scripts to run server, or model
conversion scripts.

### Create the initial branch
We do this by usinng the `-orphan` option for git checkout:
```console
$ git checkout --orphan danbev-configs
$ git rm -rf .
```

### git alias
The following git alias can be added to ~/.gitconfig so that we can run
`get-presets` to get the CMakeUserPresets.json into new workspaces. 
```console
[alias]
    get-presets = "!git show origin/danbev-configs:CMakeUserPresets.json > CMakeUserPresets.json"
```
