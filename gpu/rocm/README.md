## Radeon Open Compute Platform (ROCm)

### Installation
```console
$ sudo usermod -a -G video $LOGNAME
$ sudo dnf -y install rocminfo
$ sudo dnf install rocm-opencl
$ sudo dnf install rocm-clinfo
$ sudo dnf install hipcc
```

Unfortunately, the ROCm OpenCL implementation does not seem to support my
NVIDIA GPU.
```
$ rocminfo 
ROCk module is NOT loaded, possibly no GPU devices
```

```console
$ rocm-clinfo
Number of platforms:				 2
  Platform Profile:				 FULL_PROFILE
  Platform Version:				 OpenCL 3.0 CUDA 12.2.148
  Platform Name:				 NVIDIA CUDA
  Platform Vendor:				 NVIDIA Corporation
  Platform Extensions:				 cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_khr_fp64 cl_khr_3d_image_writes cl_khr_byte_addressable_store cl_khr_icd cl_khr_gl_sharing cl_nv_compiler_options cl_nv_device_attribute_query cl_nv_pragma_unroll cl_nv_copy_opts cl_nv_create_buffer cl_khr_int64_base_atomics cl_khr_int64_extended_atomics cl_khr_device_uuid cl_khr_pci_bus_info cl_khr_external_semaphore cl_khr_external_memory cl_khr_external_semaphore_opaque_fd cl_khr_external_memory_opaque_fd
  Platform Profile:				 FULL_PROFILE
  Platform Version:				 OpenCL 2.1 AMD-APP (3590.0)
  Platform Name:				 AMD Accelerated Parallel Processing
  Platform Vendor:				 Advanced Micro Devices, Inc.
  Platform Extensions:				 cl_khr_icd cl_amd_event_callback 


  Platform Name:				 NVIDIA CUDA
Number of devices:				 1
  Device Type:					 CL_DEVICE_TYPE_GPU
  Vendor ID:					 10deh
  Max compute units:				 46
  Max work items dimensions:			 3
    Max work items[0]:				 1024
    Max work items[1]:				 1024
    Max work items[2]:				 64
  Max work group size:				 1024
  Preferred vector width char:			 1
  Preferred vector width short:			 1
  Preferred vector width int:			 1
  Preferred vector width long:			 1
  Preferred vector width float:			 1
  Preferred vector width double:		 1
  Native vector width char:			 1
  Native vector width short:			 1
  Native vector width int:			 1
  Native vector width long:			 1
  Native vector width float:			 1
  Native vector width double:			 1
  Max clock frequency:				 2505Mhz
  Address bits:					 64
  Max memory allocation:			 3148054528
  Image support:				 Yes
  Max number of images read arguments:		 256
  Max number of images write arguments:		 32
  Max image 2D width:				 32768
  Max image 2D height:				 32768
  Max image 3D width:				 16384
  Max image 3D height:				 16384
  Max image 3D depth:				 16384
  Max samplers within kernel:			 32
  Max size of kernel argument:			 32764
  Alignment (bits) of base address:		 4096
  Minimum alignment (bytes) for any datatype:	 128
  Single precision floating point capability
    Denorms:					 Yes
    Quiet NaNs:					 Yes
    Round to nearest even:			 Yes
    Round to zero:				 Yes
    Round to +ve and infinity:			 Yes
    IEEE754-2008 fused multiply-add:		 Yes
  Cache type:					 Read/Write
  Cache line size:				 128
  Cache size:					 1318912
  Global memory size:				 12592218112
  Constant buffer size:				 65536
  Max number of constant args:			 9
  Local memory type:				 Scratchpad
  Local memory size:				 49152
  Max pipe arguments:				 0
  Max pipe active reservations:			 0
  Max pipe packet size:				 0
  Max global variable size:			 0
  Max global variable preferred total size:	 0
  Max read/write image args:			 0
  Max on device events:				 0
  Queue on device max size:			 0
  Max on device queues:				 0
  Queue on device preferred size:		 0
  SVM capabilities:				 
    Coarse grain buffer:			 Yes
    Fine grain buffer:				 No
    Fine grain system:				 No
    Atomics:					 No
  Preferred platform atomic alignment:		 0
  Preferred global atomic alignment:		 0
  Preferred local atomic alignment:		 0
  Kernel Preferred work group size multiple:	 32
  Error correction support:			 0
  Unified memory for Host and Device:		 0
  Profiling timer resolution:			 1000
  Device endianess:				 Little
  Available:					 Yes
  Compiler available:				 Yes
  Execution capabilities:				 
    Execute OpenCL kernels:			 Yes
    Execute native function:			 No
  Queue on Host properties:				 
    Out-of-Order:				 Yes
    Profiling :					 Yes
  Queue on Device properties:				 
    Out-of-Order:				 No
    Profiling :					 No
  Platform ID:					 0x56027dd902d0
  Name:						 NVIDIA GeForce RTX 4070
  Vendor:					 NVIDIA Corporation
  Device OpenCL C version:			 OpenCL C 1.2 
  Driver version:				 535.146.02
  Profile:					 FULL_PROFILE
  Version:					 OpenCL 3.0 CUDA
  Extensions:					 cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_khr_fp64 cl_khr_3d_image_writes cl_khr_byte_addressable_store cl_khr_icd cl_khr_gl_sharing cl_nv_compiler_options cl_nv_device_attribute_query cl_nv_pragma_unroll cl_nv_copy_opts cl_nv_create_buffer cl_khr_int64_base_atomics cl_khr_int64_extended_atomics cl_khr_device_uuid cl_khr_pci_bus_info cl_khr_external_semaphore cl_khr_external_memory cl_khr_external_semaphore_opaque_fd cl_khr_external_memory_opaque_fd


  Platform Name:				 AMD Accelerated Parallel Processing
Number of devices:				 0

```
