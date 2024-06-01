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

### OpenCL development
When we compile and link an OpenCL program we need to have the headers available
for compilation, these are provided by the `ocl-icd-opencl-dev` package.
On my current system they will be located in `/usr/include/CL/`.
The static linker will also need to add information about the OpenCL
Installable Client Driver (ICD) library, which is also provided by the
`ocl-icd-opencl-dev`.

For example if we take a look at the compiled binary:
```console
$ make dynamic-section 

Dynamic section at offset 0x2d08 contains 28 entries:
  Tag        Type                         Name/Value
 0x0000000000000001 (NEEDED)             Shared library: [libOpenCL.so.1]
 0x0000000000000001 (NEEDED)             Shared library: [libc.so.6]
 0x000000000000000c (INIT)               0x1000
 0x000000000000000d (FINI)               0x1e38
 0x0000000000000019 (INIT_ARRAY)         0x3cf8
 0x000000000000001b (INIT_ARRAYSZ)       8 (bytes)
 0x000000000000001a (FINI_ARRAY)         0x3d00
 0x000000000000001c (FINI_ARRAYSZ)       8 (bytes)
 0x000000006ffffef5 (GNU_HASH)           0x3b0
 0x0000000000000005 (STRTAB)             0x690
 0x0000000000000006 (SYMTAB)             0x3d8
 0x000000000000000a (STRSZ)              567 (bytes)
 0x000000000000000b (SYMENT)             24 (bytes)
 0x0000000000000015 (DEBUG)              0x0
 0x0000000000000003 (PLTGOT)             0x3f08
 0x0000000000000002 (PLTRELSZ)           552 (bytes)
 0x0000000000000014 (PLTREL)             RELA
 0x0000000000000017 (JMPREL)             0xa50
 0x0000000000000007 (RELA)               0x978
 0x0000000000000008 (RELASZ)             216 (bytes)
 0x0000000000000009 (RELAENT)            24 (bytes)
 0x000000000000001e (FLAGS)              BIND_NOW
 0x000000006ffffffb (FLAGS_1)            Flags: NOW PIE
 0x000000006ffffffe (VERNEED)            0x908
 0x000000006fffffff (VERNEEDNUM)         2
 0x000000006ffffff0 (VERSYM)             0x8c8
 0x000000006ffffff9 (RELACOUNT)          4
 0x0000000000000000 (NULL)               0x0
```
Notice that this NEEDS the `libOpenCL.so.1` shared library which we specify
during compilation and linking using `-lOpenCL`.
```
We can take a look at what the dynamic linker will do when we run the program:
```console
$ make ldd
	linux-vdso.so.1 (0x00007ffd4449a000)
	libOpenCL.so.1 => /usr/local/cuda-11.8/lib64/libOpenCL.so.1 (0x00007045ef000000)
	libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007045eec00000)
	libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007045ef2f2000)
	libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007045ef2ed000)
	/lib64/ld-linux-x86-64.so.2 (0x00007045ef317000)
```

### OpenCL ICD (Installable Client Driver)
The OpenCL ICD is a mechanism that allows multiple OpenCL implementations (from
different vendors) to coexist and be used interchangeably by applications. It
achieves this by providing a standard interface for the application to interact
with.

When libOpenCL.so is loaded, it initializes itself and discovers available
vendor-specific ICDs by reading the .icd files in /etc/OpenCL/vendors/. On
windows this is done using HKEY_LOCAL_MACHINE\SOFTWARE\Khronos\OpenCL\Vendors.

```console
$ cat /etc/OpenCL/vendors/nvidia.icd 
libnvidia-opencl.so.1

$ locate libnvidia-opencl.so.1
/usr/lib/i386-linux-gnu/libnvidia-opencl.so.1
/usr/lib/x86_64-linux-gnu/libnvidia-opencl.so.1
```

`libOpenCL.so.1` is the main library that app link against, and is responsible
for loading the actual OpenCL implementation library.

So in main we will have a symbol of `clGetplatformIDs` which is unresolved after
compilation. The dynamic linker will resolve this against the NEEDED library
libOpenCL.so can load the code for that function into the process
memory. This function will in turn delegate/handle the available implementations
available on the system as described above .

I've got a number of `libOpenCL.so` files on my system but there are all there
ICD loaders even if they are in `/usr/local/cuda-11.8/targets/x86_64-linux/lib/`
which is a cuda directory. By including the OpenCL ICD loader in the CUDA
toolkit, NVIDIA ensures that developers have a straightforward way to develop
and run OpenCL applications on NVIDIA hardware without needing to separately
install a different ICD loader.
