## vLLM

### Pre-requisites
[cudaNN](https://developer.nvidia.com/cudnn) libary needs to be installed.
```console
$ tar xf cudnn-linux-x86_64-9.0.0.312_cuda12-archive.tar.xz
$ cd cudnn-linux-x86_64-9.0.0.312_cuda12-archive/
$ sudo cp include/cudnn*.h /usr/local/cuda-12.2/include
$ sudo cp lib64/libcudnn* /usr/local/cuda-12.2/lib64
$ sudo chmod a+r /usr/local/cuda-12.2/include/cudnn*.h /usr/local/cuda-12.2/lib64/libcudnn*
```

### Building

```console
$ git clone git@github.com:vllm-project/vllm.git
$ cd vllm
$ python3.11 -m venv venv
$ source venv/bin/activate
$ pip install --editable .
```
