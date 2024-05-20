## Radeon Open Compute Platform (ROCm)
Is an open-source software foundation for GPU computing optimized for AMD GPUs
like Radeon RX Series (Gaming) for example Radeon RX 7900 XTX (pretty much allows
of them start with Radeon.

It supports various programming languages and models, with notable support for
HIP (Heterogeneous-compute Interface for Portability), which allows developers
to write code in a single source file that can run on both AMD and NVIDIA GPUs
with minimal changes. 


### ROCm Ecosystem
```
Dev tools:     ROCm SMI  ROCm Data Center Tool  ROCm Validation Suite

Libraries:     rocBlas     rocFFT  rocSparse  rocSolver  rocAllutation
               rocThrust   rocPrim MIOpen     rocRand    RCCL

Compilers:     hipcc,hipfc rocGDB rocprofiler  hipify/gpufort TENSILE

Programming:   HIP API     OpenMP API  OpenCL
models

Drivers:       Linux device drivers and runtime 
```

`rocAluation` is a library that provides functions that are optimized for
sparse and dense linear algebra operations. Recall that a sparse matrix is one
there the the majority of the elements are zero, and a dense matrix is one where
the majority of the elements are non-zero.

`rocSolver` is a library that provides functions that are optimized for solving
dense linear algebra operations.

`rocThrust` is a port of Thrust library for NVIDIA GPUs to AMD GPUs. It contains
a collection of data parallel algorithms that are optimized for AMD GPUs, like
parallel sorting, reduction, and scan. The main data type that these functions
operate on are `thrust::device_vector`, or `thrust::host_vector`.
Sorting is what it sounds like:
```c++
thrust::device_vector<int> vec = {5, 2, 8, 1, 4};
thrust::sort(d_vec.begin(), vec.end());
// vec is now {1, 2, 4, 5, 8}
```
Scanning is a parallel prefix sum:
```c++
thrust::device_vector<int> vec = {1, 2, 3, 4, 5};
thrust::inclusive_scan(d_vec.begin(), vec.end(), d_vec.begin());
// vec is now {1, 3, 6, 10, 15}
```
Reduction is a parallel sum:
```c++
thrust::device_vector<int> vec = {1, 2, 3, 4, 5};
int sum = thrust::reduce(d_vec.begin(), vec.end());
// sum is 15
```

`rocSparse` is for operations optimized for sparse matrices (which rocAlutation
also does). It focuses only on sparse matrix operations.

`rocPrim` is a library that provides parallel primitives.

`RCCL` (Radeon Collective Communications Library) is a library that enables
high-performance communication between GPUs in a cluster, like broadcasting
operations over a number of GPUs.

`MIOpen` (Machine Intelligence Open "Library") is a library that provides
functions specific to deep learning workloads. It has functions like activation
functions, normalization functions, pooling functions etc.


### Targets
When you see things like gfx1030, gfx1100, gfx1101 these refer to specific
GPU architectures. These identifiers are used to specify the architecture of the
AMD GPUs to ensure that the compiled code is optimized for the specific GPU
model.


### Links
* https://rocmdocs.amd.com/en/latest/what-is-rocm.html


### ROCm Ubuntu 23.04 issue
```console
CMake Error at /usr/share/cmake-3.22/Modules/CMakeTestCXXCompiler.cmake:62 (message):
  The C++ compiler

    "/opt/rocm-6.1.1/llvm/bin/clang++"

  is not able to compile a simple test program.

  It fails with the following output:

    Change Dir: /home/danbev/work/build/ROCm/CMakeFiles/CMakeTmp
    
    Run Build Command(s):/usr/bin/gmake -f Makefile cmTC_fe407/fast && /usr/bin/gmake  -f CMakeFiles/cmTC_fe407.dir/build.make CMakeFiles/cmTC_fe407.dir/build
    gmake[1]: Entering directory 'build/ROCm/CMakeFiles/CMakeTmp'
    Building CXX object CMakeFiles/cmTC_fe407.dir/testCXXCompiler.cxx.o
    /opt/rocm-6.1.1/llvm/bin/clang++    -MD -MT CMakeFiles/cmTC_fe407.dir/testCXXCompiler.cxx.o -MF CMakeFiles/cmTC_fe407.dir/testCXXCompiler.cxx.o.d -o CMakeFiles/cmTC_fe407.dir/testCXXCompiler.cxx.o -c /ROCm/CMakeFiles/CMakeTmp/testCXXCompiler.cxx
    Linking CXX executable cmTC_fe407
    /usr/bin/cmake -E cmake_link_script CMakeFiles/cmTC_fe407.dir/link.txt --verbose=1
    /opt/rocm-6.1.1/llvm/bin/clang++ CMakeFiles/cmTC_fe407.dir/testCXXCompiler.cxx.o -o cmTC_fe407 
    ld.lld: error: unable to find library -lstdc++
    clang++: error: linker command failed with exit code 1 (use -v to see invocation)
    gmake[1]: *** [CMakeFiles/cmTC_fe407.dir/build.make:100: cmTC_fe407] Error 1
    gmake: *** [Makefile:127: cmTC_fe407/fast] Error 2
```
The issue is that the `libstdc++` library 12 is not installed. To fix this, run:
```console
$ sudo apt install libstdc++-12-dev
```

