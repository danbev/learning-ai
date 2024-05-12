
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

###
```console
$ sudo apt install clinfo
$ clinfo
Number of platforms                     0
```

```console
$ cat /etc/OpenCL/vendors/nvidia.icd 
libnvidia-opencl.so.1


$ whereis  libnvidia-opencl.so.1
libnvidia-opencl.so.1: /usr/lib/x86_64-linux-gnu/libnvidia-opencl.so.1
```

