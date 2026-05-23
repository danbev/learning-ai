#include "hailo/hailort.hpp"

#include <iostream>
#include <memory>
#include <vector>
#include <numeric>

using namespace hailort;

int main() {
    hailo_vdevice_params_t vdevice_params{};
    if (hailo_init_vdevice_params(&vdevice_params) != HAILO_SUCCESS) {
        std::cerr << "Failed to zero-init parameter layout." << '\n';
        return 1;
    }

    // expected is like std::expected
    auto vdevice_expected = VDevice::create(vdevice_params);
    if (!vdevice_expected) {
        std::cerr << "Failed to mount VDevice context." << '\n';
        return 1;
    }
    // "move" the device, take ownership.
    std::unique_ptr<VDevice> vdevice = vdevice_expected.release();

    // load the model graph file
    auto infer_model_expected = vdevice->create_infer_model("hefs/shortcut_net.hef");
    if (!infer_model_expected) {
        std::cerr << "Failed to parse .hef configuration file map." << '\n';
        return 1;
    }
    auto infer_model = infer_model_expected.release();

    // configure...?
    auto configured_model_expected = infer_model->configure();
    if (!configured_model_expected) {
        std::cerr << "Failed to load mapping structures onto silicon clusters." << '\n';
        return 1;
    }
    auto configured_model = configured_model_expected.release();

    // get primary input/output boundary names
    auto input_names  = infer_model->get_input_names();
    auto output_names = infer_model->get_output_names();

    std::string input_layer_name  = input_names[0];
    std::string output_layer_name = output_names[0];

    size_t input_bytes  = infer_model->input(input_layer_name)->get_frame_size();
    size_t output_bytes = infer_model->output(output_layer_name)->get_frame_size();

    std::cout << "  -> Input Layer Target Name: " << input_layer_name << " (" << input_bytes << " bytes)" << '\n';
    std::cout << "  -> Output Layer Target Name: " << output_layer_name << " (" << output_bytes << " bytes)" << '\n';

    // allocate raw contiguous memory arrays
    std::vector<uint8_t> input_data(input_bytes);
    std::vector<uint8_t> output_data(output_bytes, 0);

    std::iota(input_data.begin(), input_data.end(), 0);

    // initialize IO bindings mapping structures
    auto bindings_expected = configured_model.create_bindings();
    if (!bindings_expected) {
        std::cerr << "Failed to initialize IO descriptor tracking blocks." << '\n';
        return 1;
    }
    auto bindings = bindings_expected.release();

    // attach raw user space memory frames to the target names bound to the device channels
    bindings.input(input_layer_name)->set_buffer(MemoryView(input_data.data(), input_bytes));
    bindings.output(output_layer_name)->set_buffer(MemoryView(output_data.data(), output_bytes));

    // this pushes the input frame via DMA, triggers the NPU fabric, and fills output_data
    auto run_status = configured_model.run(bindings, std::chrono::milliseconds(2000));
    if (run_status != HAILO_SUCCESS) {
        std::cerr << "Hardware execution stalled or threw exception! Error: " << run_status << '\n';
        return 1;
    }

    std::cout << "\nSuccess! Inference completed successfully." << '\n';
    std::cout << "First 5 raw bytes returned out of NPU compute arrays:" << '\n';
    for (size_t i = 0; i < 5 && i < output_bytes; ++i) {
        std::cout << "  -> Index [" << i << "]: " << (int)output_data[i] << '\n';
    }

    return 0;
}
