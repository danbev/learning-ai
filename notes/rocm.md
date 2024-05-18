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

### Links
* https://rocmdocs.amd.com/en/latest/what-is-rocm.html
