## Graphics Processing Unit (GPU)
Keep in mind that CPU's also have vector hardware like MMX, SIMD, and SSE, but
these are less powerful than GPU's. GPU's are designed to do many calculations
in parallel, which is why they are so good at matrix multiplication. Moving
memory is also an important aspect of GPU's, so they have a lot of memory
bandwidth. GPU's are also designed to be very power efficient, so they are
designed to do a lot of work with a small amount of power.


```
   +--------------------------------------------+
   | System Memory                              |
   +--------------------------------------------+
         |                    |
         |                    | High memory bandwidth
         |                    |
         |                    |
     +--------+           +-------+
     | CPU    |           | GPU   |
     |--------|           |-------|
     |Memory  |           |Memory |
     +--------+           +-------+
```
Is is also possible to have multiple buses between the GPU and system memory
which is called parallelization. This way the GPU can keep transferring data
from system memory into its own memory while it is processing data.

In most systems, the GPU has its own dedicated memory (VRAM - Video RAM) that
is separate from the system's main RAM. VRAM is optimized for the GPU's
high-throughput, parallel processing needs.

CPU to GPU Data Transfer: Initially, data (like a matrix for multiplication or
texture data for rendering) resides in the system's main memory. To be processed
by the GPU, this data needs to be transferred to the GPU's memory. This transfer
typically happens over the PCIe (Peripheral Component Interconnect Express) bus,
a high-speed interface that connects the GPU to the motherboard and through it
to the system memory.

PCIe Bandwidth: The PCIe bus has a certain bandwidth limit, which dictates how
fast data can be transferred between the system memory and the GPU memory. This
transfer can become a bottleneck, especially for large data sets or when
frequent data transfers are required.

Data Fetch by GPU: Once the data is in the GPU's memory, the GPU cores can
access it much faster. The GPU's design allows it to fetch large blocks of data
concurrently, leveraging its parallel processing capabilities. This is
particularly efficient for tasks that can process large chunks of data in
parallel, like rendering graphics or performing computations on matrices.

Data Processing: After fetching the data, the GPU performs the required
processing - this could be rendering frames for a game, computing physics
simulations, or running machine learning algorithms.

Result Transfer Back to CPU: If the results of the GPU's computations need to be
used by the CPU or need to be stored back in the system memory, they are 
ransferred back via the PCIe bus.

Optimizations: To minimize the bottleneck caused by data transfers over PCIe, software and hardware optimizations are often used. These can include minimizing the amount of data that needs to be transferred, using techniques like compression, or organizing data in a way that reduces the need for transfers.
