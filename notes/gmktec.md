## GMKtec
So I've got an CMKtec with has a AMD Radeon graphics card and an AMD Ryzen AI MAX +, also
called a "Strix Halo".

CPU details:
```console
$ lscpu | grep "Model name"
Model name:                              AMD RYZEN AI MAX+ 395 w/ Radeon 8060S
```
GPU details
```console
$ lspci -nn | grep -E 'VGA|Display'
c5:00.0 Display controller [0380]: Advanced Micro Devices, Inc. [AMD/ATI] Strix Halo [Radeon Graphics / Radeon 8050S Graphics / Radeon 8060S Graphics] [1002:1586] (rev c1)
```

NPU:
```console
$ lsmod | grep amdxdna
amdxdna               151552  0
gpu_sched              65536  2 amdxdna,amdgpu
```

So I'm mainly used to NVIDIA/Apple GPU/TPUs and this is really one of the major points of getting
this machine so I have another environment to test/debug/troubleshoot. For example we will use
ROCm for the Radeon 8060S GPU.

```console
$ clinfo
Number of platforms                               0

ICD loader properties
  ICD loader Name                                 OpenCL ICD Loader
  ICD loader Vendor                               OCL Icd free software
  ICD loader Version                              2.3.3
  ICD loader Profile                              OpenCL 3.0
```
