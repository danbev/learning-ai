### Poetry
This is a package manager for Python. Currently we are using pip and requirements.txt
files. But there is a problem with this as I might run this one of my machines
and install a certain version of numpy, and then later on perform the same install
on another machine and get a different version of numpy possible breaking conversion
scripts.

### Install Poetry
```console
$ pipx install poetry
```


### Using
```console
$ poetry install
```
This will read the pyproject.toml that is a new file in the root of llama.cpp
and will take care of setting up a virtual environment and installing the
dependencies. It will also create a poetry.lock file.
```console
Creating virtualenv llama-cpp-scripts-I8qQi2bX-py3.12 in /Users/danbev/Library/Caches/pypoetry/virtualenvs
Installing dependencies from lock file
```

We can also activate the virtual environment with:
```console
$ poetry shell
```

To run a script in the virtual env we can:
```console
$ poetry run python convert_hf_to_gguf.py --help
```

### install issue
When installing I ran into the following error:
```console
$ poetry run python convert_hf_to_gguf.py --help
The "poetry.dev-dependencies" section is deprecated and will be removed in a future version. Use "poetry.group.dev.dependencies" instead.
PyTorch was not found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
Traceback (most recent call last):
  File "/Users/danbev/work/ai/llama.cpp/convert_hf_to_gguf.py", line 23, in <module>
    import torch
ModuleNotFoundError: No module named 'torch'
```
So pyproject.toml has:
```console
torch = {  source = "pytorch" }
```
This will produce the following in the poetry.lock file:
```
[[package]]
name = "torch"
version = "2.2.1+cpu"
description = "Tensors and Dynamic neural networks in Python with strong GPU acceleration"
optional = false
python-versions = ">=3.8.0"
groups = ["main"]
files = [
    {file = "torch-2.2.1+cpu-cp310-cp310-linux_x86_64.whl", hash = "sha256:5d82422cf04797f1b2a8574b64a916070ec83eef58ad4900615ee0218d7b8b8e"},
    {file = "torch-2.2.1+cpu-cp310-cp310-win_amd64.whl", hash = "sha256:f8914dd0f5f0e5c66fdecd9559403eea9feac82d1ea639b672fde0073c6addbd"},
    {file = "torch-2.2.1+cpu-cp311-cp311-linux_x86_64.whl", hash = "sha256:6bc973d5632374b92b4b293817b4d2ff8c8ce1c784c748b471dba1fffcd9c333"},
    {file = "torch-2.2.1+cpu-cp311-cp311-win_amd64.whl", hash = "sha256:abdec34b0ade8fca0520055e72c3094425ae0ef210718e9c0278121cd3608c32"},
    {file = "torch-2.2.1+cpu-cp312-cp312-linux_x86_64.whl", hash = "sha256:d7339580135da4105c1244a8621faa076990409afeab5a7b642c3c1ee70a5622"},
    {file = "torch-2.2.1+cpu-cp312-cp312-win_amd64.whl", hash = "sha256:039128fcb5548122465b15f679b8831c47d14f0d6c28c1f1b631f8019c104720"},
    {file = "torch-2.2.1+cpu-cp38-cp38-linux_x86_64.whl", hash = "sha256:2b447f7bb50b393b4544b4036d587e39ab524d4353e77c197f6a2727f22b0d47"},
    {file = "torch-2.2.1+cpu-cp38-cp38-win_amd64.whl", hash = "sha256:2ccdf3e5f71e6426ea9e34d21c3cc333b29d4f48299b981d28aeb5112b5495e1"},
    {file = "torch-2.2.1+cpu-cp39-cp39-linux_x86_64.whl", hash = "sha256:2fb340b289760040a16a77a6d70b8a48961abba1822e6f58705c97c80befa03e"},
    {file = "torch-2.2.1+cpu-cp39-cp39-win_amd64.whl", hash = "sha256:e03dc4654ecceeb5b03f0a6f60b342c0e0d267b3ebc61e4f672cace1df8cd930"},
]
```
This only contains windows and linux so this will not work for mac right.
```console
torch = [
    { version = ">=2.2.0", source = "pypi", markers = "sys_platform == 'darwin'" },
    { version = ">=2.2.0+cpu", source = "pytorch", markers = "sys_platform == 'linux'" }
]
```
Changing this and re-generating the poetry.lock file will give us the following:
```console
$ poetry lock --regenerate
```
```console
[[package]]
name = "torch"
version = "2.7.1"
description = "Tensors and Dynamic neural networks in Python with strong GPU acceleration"
optional = false
python-versions = ">=3.9.0"
groups = ["main"]
markers = "sys_platform == \"darwin\""
files = [
    {file = "torch-2.7.1-cp310-cp310-manylinux_2_28_aarch64.whl", hash = "sha256:a103b5d782af5bd119b81dbcc7ffc6fa09904c423ff8db397a1e6ea8fd71508f"},
    {file = "torch-2.7.1-cp310-cp310-manylinux_2_28_x86_64.whl", hash = "sha256:fe955951bdf32d182ee8ead6c3186ad54781492bf03d547d31771a01b3d6fb7d"},
    {file = "torch-2.7.1-cp310-cp310-win_amd64.whl", hash = "sha256:885453d6fba67d9991132143bf7fa06b79b24352f4506fd4d10b309f53454162"},
    {file = "torch-2.7.1-cp310-none-macosx_11_0_arm64.whl", hash = "sha256:d72acfdb86cee2a32c0ce0101606f3758f0d8bb5f8f31e7920dc2809e963aa7c"},
    {file = "torch-2.7.1-cp311-cp311-manylinux_2_28_aarch64.whl", hash = "sha256:236f501f2e383f1cb861337bdf057712182f910f10aeaf509065d54d339e49b2"},
    {file = "torch-2.7.1-cp311-cp311-manylinux_2_28_x86_64.whl", hash = "sha256:06eea61f859436622e78dd0cdd51dbc8f8c6d76917a9cf0555a333f9eac31ec1"},
    {file = "torch-2.7.1-cp311-cp311-win_amd64.whl", hash = "sha256:8273145a2e0a3c6f9fd2ac36762d6ee89c26d430e612b95a99885df083b04e52"},
    {file = "torch-2.7.1-cp311-none-macosx_11_0_arm64.whl", hash = "sha256:aea4fc1bf433d12843eb2c6b2204861f43d8364597697074c8d38ae2507f8730"},
    {file = "torch-2.7.1-cp312-cp312-manylinux_2_28_aarch64.whl", hash = "sha256:27ea1e518df4c9de73af7e8a720770f3628e7f667280bce2be7a16292697e3fa"},
    {file = "torch-2.7.1-cp312-cp312-manylinux_2_28_x86_64.whl", hash = "sha256:c33360cfc2edd976c2633b3b66c769bdcbbf0e0b6550606d188431c81e7dd1fc"},
    {file = "torch-2.7.1-cp312-cp312-win_amd64.whl", hash = "sha256:d8bf6e1856ddd1807e79dc57e54d3335f2b62e6f316ed13ed3ecfe1fc1df3d8b"},
    {file = "torch-2.7.1-cp312-none-macosx_11_0_arm64.whl", hash = "sha256:787687087412c4bd68d315e39bc1223f08aae1d16a9e9771d95eabbb04ae98fb"},
    {file = "torch-2.7.1-cp313-cp313-manylinux_2_28_aarch64.whl", hash = "sha256:03563603d931e70722dce0e11999d53aa80a375a3d78e6b39b9f6805ea0a8d28"},
    {file = "torch-2.7.1-cp313-cp313-manylinux_2_28_x86_64.whl", hash = "sha256:d632f5417b6980f61404a125b999ca6ebd0b8b4bbdbb5fbbba44374ab619a412"},
    {file = "torch-2.7.1-cp313-cp313-win_amd64.whl", hash = "sha256:23660443e13995ee93e3d844786701ea4ca69f337027b05182f5ba053ce43b38"},
    {file = "torch-2.7.1-cp313-cp313t-macosx_14_0_arm64.whl", hash = "sha256:0da4f4dba9f65d0d203794e619fe7ca3247a55ffdcbd17ae8fb83c8b2dc9b585"},
    {file = "torch-2.7.1-cp313-cp313t-manylinux_2_28_aarch64.whl", hash = "sha256:e08d7e6f21a617fe38eeb46dd2213ded43f27c072e9165dc27300c9ef9570934"},
    {file = "torch-2.7.1-cp313-cp313t-manylinux_2_28_x86_64.whl", hash = "sha256:30207f672328a42df4f2174b8f426f354b2baa0b7cca3a0adb3d6ab5daf00dc8"},
    {file = "torch-2.7.1-cp313-cp313t-win_amd64.whl", hash = "sha256:79042feca1c634aaf6603fe6feea8c6b30dfa140a6bbc0b973e2260c7e79a22e"},
    {file = "torch-2.7.1-cp313-none-macosx_11_0_arm64.whl", hash = "sha256:988b0cbc4333618a1056d2ebad9eb10089637b659eb645434d0809d8d937b946"},
    {file = "torch-2.7.1-cp39-cp39-manylinux_2_28_aarch64.whl", hash = "sha256:e0d81e9a12764b6f3879a866607c8ae93113cbcad57ce01ebde63eb48a576369"},
    {file = "torch-2.7.1-cp39-cp39-manylinux_2_28_x86_64.whl", hash = "sha256:8394833c44484547ed4a47162318337b88c97acdb3273d85ea06e03ffff44998"},
    {file = "torch-2.7.1-cp39-cp39-win_amd64.whl", hash = "sha256:df41989d9300e6e3c19ec9f56f856187a6ef060c3662fe54f4b6baf1fc90bd19"},
    {file = "torch-2.7.1-cp39-none-macosx_11_0_arm64.whl", hash = "sha256:a737b5edd1c44a5c1ece2e9f3d00df9d1b3fb9541138bee56d83d38293fb6c9d"},
]
```
Notice that we now have entried for darwin in addition to linux and windows.
So now we should be able to run the script on mac as well.
