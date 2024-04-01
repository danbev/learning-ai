## TensorRT-LLM  (TRT-LLM)
"NVIDIA TensorRT-LLM is an open-source library that accelerates and optimizes
inference performance of the latest large language models (LLMs) on the NVIDIA
AI platform."

TensorRT is an SDK for inference runtimes (RT).

It does not just take the model wieghts and run them on a GPU but it also
compiles the model and optimizes the kernels to make them as effiecent as
possible on NVIDIA GPUs:
```
  Host  Environment (CPU)       Device Environment (GPU)
  +---------------+
  | Model weights |-----+
  +---------------+     |     +------------+     +----------------+
                        +---->| Compiler   |---->| Compiled Model |
  +---------------+     |     +------------+     +----------------+
  | Optimization  |     |
  |   options     |-----+
  +---------------+
```
Optimization options can things like quantization. Notice that this compilation
is done on the device environment (GPU) and the optimized model is specific to
the GPU in use. So the same type of GPU must be used for compilation and
inference.

Not all models are supported by TensorRT. The model must be supported by the
listed [here](https://github.com/NVIDIA/TensorRT-LLM?tab=readme-ov-file#models).


### Installation
There is a container toolkit that needs to be installed:
```console
$ curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
  sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
$ sudo yum install -y nvidia-container-toolkit
```

```console
$ sudo nvidia-ctk runtime configure --runtime=docker
INFO[0000] Config file does not exist; using empty config 
INFO[0000] Wrote updated config to /etc/docker/daemon.json 
INFO[0000] It is recommended that docker daemon be restarted. 

$ sudo systemctl restart docker

$ sudo docker run --privileged -v --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
Thu Mar 28 11:47:17 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.146.02             Driver Version: 535.146.02   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 4070        Off | 00000000:2F:00.0 Off |                  N/A |
| 35%   29C    P0              23W / 200W |      0MiB / 12282MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
```
So we are using Docker run to run a container which will be based on the
ubuntu image and the command run will be `nvidia-smi`. 
The `--runtime` flag and the `--gpus` flags are new to me.  The `--gpus` flag
allows access to NVIDIA GPU resorces an this needs the
`nvidia-container-toolkit` to be installed. The ``--runtime`` flag is used to
to specify the runtime, which by default is `runc`. This enables containers to
access GPU hardware.
The available runtimes (I'm assuming those apart from the default one) can be
listed with:
```console
$ docker info | grep Runtimes
 Runtimes: io.containerd.runc.v2 nvidia runc
```
Or by looking at the `/etc/docker/daemon.json` file:
```console
$ cat /etc/docker/daemon.json 
{
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    }
}
```

_wip_
