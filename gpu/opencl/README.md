### Installing OpenCL
```console
$ sudo apt-get install ocl-icd-opencl-dev
```

### Building
```console
$ make main
```

### Running
```console
$ ./main
```

### clinfo
To verify the OpenCL installation, you can use the `clinfo` command. If you
don't have it installed, you can install it with the following command:
```console
$ sudo apt install clinfo
```

```console
$ clinfo
Number of platforms                               2
  Platform Name                                   AMD Accelerated Parallel Processing
  Platform Vendor                                 Advanced Micro Devices, Inc.
  Platform Version                                OpenCL 2.1 AMD-APP (3614.0)
  Platform Profile                                FULL_PROFILE
  Platform Extensions                             cl_khr_icd cl_amd_event_callback 
  Platform Extensions function suffix             AMD
  Platform Host timer resolution                  1ns

  Platform Name                                   NVIDIA CUDA
  Platform Vendor                                 NVIDIA Corporation
  Platform Version                                OpenCL 3.0 CUDA 12.4.89
  Platform Profile                                FULL_PROFILE
```

```console
$ cat /etc/OpenCL/vendors/nvidia.icd 
libnvidia-opencl.so.1


$ whereis  libnvidia-opencl.so.1
libnvidia-opencl.so.1: /usr/lib/x86_64-linux-gnu/libnvidia-opencl.so.1
```

