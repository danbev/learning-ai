### Xilinx
So AMD aquired Xilinx which is why things are named like xrt which stands for
Xilinx Runtime.

These is also XDNA which is Xilinx DNA which is just the name of the NPU hardware
block. And this is also Vitis which is a high-level, also from Xilinx and Vitis
means vine or graphvine in Latin.

```console
+-----------------------------------------------+
|                     Application               | <- whisper.cpp / ONNX Runtime
+-----------------------------------------------+
|                      Vitis AI                 | <- High-Level Framework / Compiler
+-----------------------------------------------+
|                        XRT                    | <- Low-Level Runtime / Driver Shim
+-----------------------------------------------+
|                       XDNA                    | <- Physical NPU Silicon
+-----------------------------------------------+
```

### XDNA
XDNA is the physical NPU silicon that is used in AMD/Xilinx products. This is a
data flow fabric composed of VLIW (Very Long Instruction Word) processing
elements and localized memory blocks (similar to Hailo NPU architecture by the
sounds of this).


### XRT (Xilinx Runtime)
XRT consists of the kernel drivers (amdxdna.ko) and the user-space libraries
(libxrt_coreutil.so which we link against in the Makefile).

XRT doesn't know anything about machine learning, layers, or weights. It can:
* Discover the PCIe device slot.
* Map virtual memory addresses via SVA (Shared Virtual Addressing). 
* Allocate hardware execution rings.
* Push raw compiled binary blocks (.xclbin) directly into the hardware configuration memory.

### Vitis AI
XRT doesn't know what a convolutional layer or a matrix multiplication is.
Vitis AI is the overarching intelligence engine. It contains the quantizer
toolkits and the graph compiler. It takes our high-level neural network, slices
it up, quantizes it to INT8/BF16, and compiles it into an execution stream that
the XDNA tiles can actually run. Once Vitis AI generates that compiled execution
graph, it hands it to XRT to actually load and execute it on the silicon.
