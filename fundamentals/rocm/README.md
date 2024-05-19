## ROCm exploration
I currently don't have an AMD GPU so this will only be for learning about the
libraries and compiling until I can get my hands on one.

## Install ROCm

```console
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
    gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/6.1.1/ubuntu jammy main" \
    | sudo tee /etc/apt/sources.list.d/amdgpu.list
sudo apt update
sudo apt install rocm-dev
```
Verify that hipcc is installed
```console
$ hipcc --version
HIP version: 6.1.40092-038397aaa
AMD clang version 17.0.0 (https://github.com/RadeonOpenCompute/llvm-project roc-6.1.1 24154 f53cd7e03908085f4932f7329464cd446426436a)
Target: x86_64-unknown-linux-gnu
Thread model: posix
InstalledDir: /opt/rocm-6.1.1/llvm/bin
Configuration file: /opt/rocm-6.1.1/lib/llvm/bin/clang++.cfg
```

