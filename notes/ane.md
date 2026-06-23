### Apple Neural Engine (ANE)
So the hardware target of this is NPUs and not GPUs. It helped me to think of
this as similar to [Hailo](./hailo.md).

So there is a compiler that takes a CoreML model and converts it into an internal
format, model intermediate language (MIL) and then compiles this into a binary
image that can executed by the runtime which is named e5rt. This is similar to
how Hailo also interacts with hailort to actually put the model onto the chip for
execution.

The ANE is baked directly into Apple's Unified Memory system. It shares the
exact same physical pool of system RAM as the CPU and GPU.

### Espresso
This name predates CoreML and any other public machine learning framework that
apple exposed and was named Espresso.

* Espresso v1 to v3: Handled early iOS/macOS internal model deployments.
* Espresso v4: Introduced heavy graph-optimization concepts.
* Espresso v5 (e5): The current, modern iteration. It is completely rewritten to
  MIL graphs and compile them cleanly down to the specific hardware targets of
  modern Apple Silicon (M-series and A-series chips).

This is both a library and a client server architectur for security and isolation.
```console
+-------------------------------------------------------+
|  Application Process (like whisper.cpp)               |
|                                                       |
|   +-----------------------------------------------+   |
|   |  e5rt (Library loaded via dlopen)             |   |
|   +-----------------------+-----------------------+   |
+---------------------------|---------------------------+
                            |
                 IPC (XPC Services)
                            |
+---------------------------v---------------------------+
|  aned (Root System Daemon Process)                    |
|  - Validates and compiles the MIL graph               |
|  - Cryptographically signs the ANE binary image       |
+---------------------------|---------------------------+
                            |
                    IOKit System Calls
                            |
+---------------------------v---------------------------+
|  AppleNeuralEngine Kernel Driver                      |
+---------------------------|---------------------------+
                            |
+---------------------------v---------------------------+
|  Physical ANE Silicon                                 |
+-------------------------------------------------------+
```

### e5rt (Espresso V5 Runtime)
e5rt is a private library that lives inside Apple's private framework directory
which is loaded into our applications process memory space. This is the front
end API.

### aned (Apple Neural Engine Daemon)
aned is the daemon that can talk to the ANE hardware driver. As we can see above
this communicated using IPC from the user process. The ANE hardware will refuse
to execute a binary instruction stream unless it has been vetted and signed by
the operating system. When e5rt passes an unsigned neural network graph over to
aned, the daemon invokes Apple's internal compiler, checks it for invalid or
malicious memory access routines, compiles it, and cryptographically signs the
compiled blob in memory.

If three different apps (e.g., a web browser running a webNN model, a video
editor tracking a face, and whisper.cpp transcribing audio) try to use the ANE at
the exact same time, they would crash the hardware if they wrote directly to the
registers. The aned daemon acts as the centralized traffic cop, queueing requests,
context-switching the hardware state, and keeping app memory sandboxed.

Inspecting the the Espresso framework:
```console
(lldb) br set -f whisper.cpp -l 3477

(lldb) image list Espresso
[  0] 5BBFD717-0C91-3B15-9451-7EF7CFF28DA4 0x00000001ba8d5000 /System/Library/PrivateFrameworks/Espresso.framework/Versions/A/Espresso

(lldb) image lookup -r -s e5rt
73 symbols match the regular expression 'e5rt' in /Users/danbev/work/ai/whisper-aneforge/venv/lib/python3.14/site-packages/aneforge/_lib/libane_e5rt_dispatch.dylib:
        Address: libane_e5rt_dispatch.dylib[0x0000000000001978] (libane_e5rt_dispatch.dylib.__TEXT.__text + 4384)
        Summary: libane_e5rt_dispatch.dylib`prog_destroy(ane_e5rt_program*)
        Address: libane_e5rt_dispatch.dylib[0x0000000000001a7c] (libane_e5rt_dispatch.dylib.__TEXT.__text + 4644)
        Summary: libane_e5rt_dispatch.dylib`compile_and_build_op(char const*, char const*, unsigned long long, ane_e5rt_op_t*)
        Address: libane_e5rt_dispatch.dylib[0x0000000000001de4] (libane_e5rt_dispatch.dylib.__TEXT.__text + 5516)
        Summary: libane_e5rt_dispatch.dylib`port_alloc_and_bind(void*, char const*, unsigned long, int, ane_e5rt_port_t*)
        Address: libane_e5rt_dispatch.dylib[0x0000000000002324] (libane_e5rt_dispatch.dylib.__TEXT.__text + 6860)
```
