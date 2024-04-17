##  SYCL
SYCL abstracts the complexity of managing device memory and executing kernels
(the functions that run on compute devices in parallel) across different
hardware.

Today large high performance computing (HPC) systems are composed of a mix of
CPUs, GPUs, and other accelerators. Each of these devices have their own
low-level programming language/model and memory management and learning each of
these can be time consuming and error prone.


SYCL is a standards-based and vendor-agnostic domain-specific embedded language
for parallel programming, for heterogeneous and homogeneous architectures.
SYCL is a single-source programming model that enables developers to write code
that can run on any of these devices.

So this enables developers to write code, even GPU kernel code, then the SYCL
runtime can decide where to run it, CPU, GPU, or FPGA.

Examples of SYCL implementations include DPC++ from Intel, ComputeCpp from

### Data Parallel C++ (DPC++)
Is a an extension of C++ for data parallel programming, it is based on SYCL.
