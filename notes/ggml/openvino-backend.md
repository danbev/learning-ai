## OpenVINO backend

### Overview
Basically the overall picture of what is done is that GGML concepts like
ggml_tensor are mapped to OpenVINO concepts like ov::Node. And the ggml_cgraph
is translated into and ov::Model. 

1. Tensor Translation (ggml-decoder.cpp):
    - Converts ggml tensors (ggml_tensor) to OpenVINO tensors (ov::Tensor)
    - Maps ggml data types (F32, F16, BF16, quantized types) to OpenVINO element types (ov::element::Type)
    - Handles shapes and strides conversion

2. Operation Translation (openvino/op/*.cpp):
    - Each ggml operation (like GGML_OP_RMS_NORM, GGML_OP_ROPE, etc.) is converted to OpenVINO operations
    - Custom ops are implemented for operations not natively available in OpenVINO

3. Graph Translation (ggml-decoder.cpp, utils.cpp):
    - The GgmlOvDecoder walks through the ggml compute graph (ggml_cgraph)
    - Identifies inputs, outputs, weights, and operations
    - Creates an OpenVINO model (ov::Model) with equivalent structure
    - Uses OpenVINO's GGML frontend: ov::frontend::ggml::FrontEnd::convert(input_model)

4. Execution Flow (utils.cpp:66):
    - openvino_frontend_compute(backend, cgraph)
    - Compiles the OpenVINO model for the target device (GPU/CPU/NPU)
    - Creates an inference request (ov::InferRequest)
    - Copies input data from ggml tensors to OpenVINO tensors
    - Executes inference via infer_request.infer()
    - Copies results back to ggml tensor memory

### Debugging session notes
Can just to recap, the openvino backend uses openvino c++ api. When
llama_context::graph_compute is called the following function will be called:
```c++
static enum ggml_status
ggml_backend_openvino_graph_compute(ggml_backend_t backend, struct ggml_cgraph *cgraph) {
    openvino_frontend_compute(backend, cgraph);

    return GGML_STATUS_SUCCESS;
}

```
This will land in ggml/src/ggml-openvino/utils.cpp:
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
```c++
std::map<std::string, std::shared_ptr<ov::Node>> GgmlOvDecoder::create_weight_nodes(
    struct ggml_cgraph* cgraph, std::map<ggml_type, ExtraQuantType> types_to_requantize) {
    std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
```
`ov::Node` is the the base class for all operations in OpenVINO's computation
graph.



