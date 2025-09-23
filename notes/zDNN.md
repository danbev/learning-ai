### zDNN (IBM Z Deep Neural Network Library) backend
This is a backend that enables access to IBM's Z integrated Accellerator for AI which is
an on-chip AI accelerator available on IBM z16 system. When I first read this I thought
that they had an integrated GPU on the CPU chip in some way. It is  custom-designed
AI accelerator and not a GPU.

It is accessed/invoked using a new CPU instruction called Neural Network Processing Assist
(NNPA) which was addded to the z16 proessor (and later). It works directly with the CPU rather
than communicating with a separate GPU over PCIe. And this is speciallized for neural network
inference workloads.

It's a synchronous compute device meaning the CPU instruction execution waits for the
accelerator to complete, unlike GPU kernels which are typically asynchronous.


### Execution
* Data preparation
zDNN formats the tensor appropriately on behalf of the caller using an optimized approach,
since the zAIU has very complex data layout requirements that arrange the tensor to
enhance performance characteristics.
* Library call
The backend calls zDNN library functions

* Hardware invocation
zDNN invokes the NNPA instruction on behalf of the caller to drive the accelerator. The zDNN
library issues a Neural Network Processing Assist (NNPA) machine instruction IBMEvolving Solutions

* Synchronous execution
The NNPA machine instruction is executed on the z16 core which calls the AI Accelerator synchronously
