## TensorRT-LLM  (TRT-LLM)
"NVIDIA TensorRT-LLM is an open-source library that accelerates and optimizes
inference performance of the latest large language models (LLMs) on the NVIDIA
AI platform."

TensorRT is an SDK for inference runtimes (RT).


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

_wip_
