## OpenVINO backend
OpenVINO is an open-source toolkit for optimizing and deploying AI inference.
If you search for examples online you will probably see examples where a model
is loaded from a file. But this can all be done programmatrically and there
is an example of this in [simple-inf.cpp](../../fundamentals/openvino-cpp/src/simple-inf.cpp).

The OpenVINO backend, being part of ggml use the gguf file format just like
everything else. What the backend does is that it translates the GGML concepts
to OpenVINO concepts and then uses the OpenVINO runtime to execute the model.

* GGML tensors are translated into ov::Tensor (ov = OpenVINO). 

* OpenVINO has a number of built-in operations and additional one have been created
  in ggml/src/ggml-openvino/openvino/op/*.cpp).

* The computation graph is translated into an ov::Model (which is a directed
  acyclic graph of ov::Node).


I naively thought that a compuation unit like a NPU (Neural Processing Unit)
would be accessable by something like intrinsics or some other low-level
API. But this does not seem to be the case. The NPU is accessed through the
OpenVINO runtime:
```
Application Code
    ↓
Framework (PyTorch/TensorFlow/ONNX/Llama.cpp)
    ↓
OpenVINO Runtime
    ↓
NPU Compiler/Plugin (closed source)
    ↓
NPU Firmware/Driver
    ↓
NPU Hardware (VPU/NPU)
```

Intel NPUs (formerly VPUs) are based on acquired Movidius technology
The actual instruction set and architecture are proprietary

So in the example above when we call:
```c++
        ov::CompiledModel compiled_model = core.compile_model(model, "CPU");
        ov::InferRequest infer_request = compiled_model.create_infer_request();
        infer_request.infer();
```
The following happends behind the scenes:
* Graph Compiler (proprietary) translates the model
* Firmware blob is generated for the specific NPU
* Driver loads this blob to NPU via PCIe/USB
* NPU firmware orchestrates execution

This becomes an issue as in llama.cpp when we perform inference:
```c++
int llama_context::decode(const llama_batch & batch_inp) {
    ...
        const auto * res = process_ubatch(ubatch, LLM_GRAPH_TYPE_DECODER, mctx.get(), status);
```
And process_ubatch will call build_graph:
```c++
llm_graph_result * llama_context::process_ubatch(const llama_ubatch & ubatch, llm_graph_type gtype, llama_memory_context_i * mctx, ggml_status & ret) {
    ...
        gf = model.build_graph(gparams);
```
And this will build the computation graph for the model in question, and this
includes specific sizes for the inputs, like the sequence length, batch size
etc.

In the case of OpenVINO which translate the GGML graph to an OpenVINO model, they
have to perform the above steps of compilation, firmware generation and loading
to the NPU. This is a slow process and it is not feasible to do for every
inference call. Currently they do some form of caching which I'm not exactly sure
how this works and I think this only applied of the NPU case which I can't
really test it as I don't have the hardware.

It sounds like this is not something unique to Intel:
* Apple Neural Engine: Similar black box through Core ML

### NPU issue
To understand the NPU issue better we can think of the NPU as my IoT device. If
I need to update something, like change a configuration in the program, say the
size of a variable, then I actually need to compile and flash the device to see
that change. This is alright if it seldom changes but if it is frequenent it is
time consuming.
The same thing happens with the NPU where if you have operation that contain the
same sizes that are called many times it will bascially be compiled once and
called multiple times. But if the case is that tensor sizes change then it needs
to "flash" the device for the new program. And this is what is happening with
llama.cpp and OpenVINO backend where the sequence length and batch size can
change for every inference call. And this is also the reason why the NPU has
this issue and not the CPU or GPU as they are more flexible and can handle it.
There are solutions like having a fixed sequence lenght and padding the input
to fit but this also wastes resources.

So could the CPU handle the prefill prompt and then the NPU handle the decoding
of tokens with a fixed sequence length of 1?


### Debugging session notes
```console
source ~/work/ai/learning-ai/fundamentals/openvino-cpp/deps/openvino_toolkit_ubuntu24_2025.3.0.19807.44526285f24_x86_64/setupvars.sh

cmake -S . -B build -DLLAMA_CURL=ON \
    -DGGML_OPENVINO=ON \
    -DGGML_NATIVE=ON \
    -DLLAMA_BUILD_TESTS=ON \
    -DCMAKE_BUILD_TYPE=Debug
cmake --build build -- -j 12
````

Running in gdb:
```console
gdb --args \
    ./build/bin/llama-cli \
    -m ../llama.cpp/models/gemma-3-270m-it-qat-q4_0-unquantized-Q4_0.gguf \
    --no-warmup --prompt '"What is capital of France?"' -n 40  -t 4 -no-cnv
```

When `graph_compute` is called this will land in the OpenVINO backend, and the
function `ggml_backend_openvino_graph_compute` will be called:
```console
(gdb) bt
#0  ggml_backend_openvino_graph_compute (backend=0x555555bd2380, cgraph=0x555555bb3e78)
    at /home/danbev/work/ai/llama.cpp-openvino/ggml/src/ggml-openvino/ggml-openvino.cpp:54
#1  0x00007ffff756f0db in ggml_backend_graph_compute_async (backend=0x555555bd2380, cgraph=0x555555bb3e78)
    at /home/danbev/work/ai/llama.cpp-openvino/ggml/src/ggml-backend.cpp:359
#2  0x00007ffff757412a in ggml_backend_sched_compute_splits (sched=0x555555b51d30)
    at /home/danbev/work/ai/llama.cpp-openvino/ggml/src/ggml-backend.cpp:1553
#3  0x00007ffff7574f5e in ggml_backend_sched_graph_compute_async (sched=0x555555b51d30, graph=0x555555fe2930)
    at /home/danbev/work/ai/llama.cpp-openvino/ggml/src/ggml-backend.cpp:1753
#4  0x00007ffff7c2280c in llama_context::graph_compute (this=0x555555b6c1d0, gf=0x555555fe2930, batched=true)
    at /home/danbev/work/ai/llama.cpp-openvino/src/llama-context.cpp:1460
#5  0x00007ffff7c1f4fa in llama_context::process_ubatch (this=0x555555b6c1d0, ubatch=..., gtype=LLM_GRAPH_TYPE_DECODER, 
    mctx=0x555555aecec0, ret=@0x7fffffffaed4: 32767) at /home/danbev/work/ai/llama.cpp-openvino/src/llama-context.cpp:784
#6  0x00007ffff7c20a75 in llama_context::decode (this=0x555555b6c1d0, batch_inp=...)
    at /home/danbev/work/ai/llama.cpp-openvino/src/llama-context.cpp:1088
#7  0x00007ffff7c278c5 in llama_decode (ctx=0x555555b6c1d0, batch=...)
    at /home/danbev/work/ai/llama.cpp-openvino/src/llama-context.cpp:2747
#8  0x00005555555e459d in main (argc=11, argv=0x7fffffffd678) at /home/danbev/work/ai/llama.cpp-openvino/tools/main/main.cpp:671
```
This function is defined in ggml/src/ggml-openvino/ggml-openvino.cpp:
```c++
static enum ggml_status
ggml_backend_openvino_graph_compute(ggml_backend_t backend, struct ggml_cgraph *cgraph) {
    openvino_frontend_compute(backend, cgraph);

    return GGML_STATUS_SUCCESS;
}
```
And openvino_frontend_compute is defined in ggml/src/ggml-openvino/utils.h:
```c++
enum ggml_status openvino_frontend_compute(ggml_backend_t backend, struct ggml_cgraph* cgraph) {
    static ov::Core core;

    static std::string device = getenv("GGML_OPENVINO_DEVICE") ? getenv("GGML_OPENVINO_DEVICE") : "";
    if (device.empty()) {
        const std::vector<std::string> preferred_device = { "GPU", "CPU", "NPU" };
        const auto available_devices = core.get_available_devices();
        for (const auto& dev : preferred_device) {
            if (std::find(available_devices.begin(), available_devices.end(), dev) != available_devices.end()) {
                device = dev;
                break;
            }
        }
    }
```
`ov::Core` is the main entry point to the OpenVINO Runtime. This central object
manages available devices, their properties, model loading, model compilation,
plugin management. Think of this like a session.
SO an environment variable can be set to specify which device to use and if not
set then the preference will be to use a GPU, if not available then CPU, if not
available a NPU.

```c++
    bool is_static = device == "NPU" ? true : false;
    ov::AnyMap config;

    if (getenv("GGML_OPENVINO_DUMP_CGRAPH")) {
        std::string filename = "cgraph.txt";
        GgmlOvDecoder::dump_cgraph(cgraph, filename);
    }

    if (is_naive(cgraph)) {
        return naive_compute(cgraph, core, device, config);
    }
```
is_naive looks like this:
```c++
bool is_naive(struct ggml_cgraph* cgraph) {
    constexpr int naive_graph_size_threshold = 20;
    return cgraph->n_nodes < naive_graph_size_threshold;
}
```
In this debugging session I have the following number of nodes:
```console
(gdb) p cgraph->n_nodes
$10 = 729
```
So this is not a naive graph. 

```c++
            std::shared_ptr<ov::Model> model;
            auto model_weights = GgmlOvDecoder::create_weight_nodes(cgraph, get_types_to_requant(device));
```
Lets take a look at the types that need requantization:
```c++
std::map<ggml_type, ExtraQuantType> get_types_to_requant(const std::string& device) {
    if (device == "NPU") {
        return {
            {GGML_TYPE_Q4_0, ExtraQuantType::Q4_0_128},
            {GGML_TYPE_Q4_1, ExtraQuantType::Q4_0_128},
            {GGML_TYPE_Q4_K, ExtraQuantType::Q4_0_128},
            {GGML_TYPE_Q6_K, ExtraQuantType::F16     },
            {GGML_TYPE_Q5_K, ExtraQuantType::F16     },
        };
    }
    if (device == "GPU") {
        return {
            // gs16 is WIP
            {GGML_TYPE_Q6_K, ExtraQuantType::Q8_0_32},
        };
    }
    return {};
}
```
So for NPUs Q4_0, Q4_1, and Q4_K are requantized to a Q4_0 format but with a
larger block size of 128.
and for GPU Q6_K is requantized to an 8-bit format.


```c++
std::map<std::string, std::shared_ptr<ov::Node>> GgmlOvDecoder::create_weight_nodes(
    struct ggml_cgraph* cgraph, std::map<ggml_type, ExtraQuantType> types_to_requantize) {
    std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
```
`ov::Node` is the the base class for all operations in OpenVINO's computation
graph.

_wip_
