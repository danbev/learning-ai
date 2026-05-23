#include "hailo/hailort.hpp"
#include <chrono>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace hailort;

int main() {
    const std::string hef_path = "hefs/qwen2.5.hef";
    const std::string network_group_name = "base_model__tbt";

    auto vdevice = VDevice::create().expect("Failed to create VDevice");

    auto infer_model = vdevice->create_infer_model(hef_path, network_group_name).expect("Failed to create infer model");
    infer_model->set_batch_size(1);

    auto configured_model = infer_model->configure().expect("Failed to configure infer model");

    auto bindings = configured_model.create_bindings().expect("Failed to create bindings");

    // Allocate and bind input buffers
    std::map<std::string, std::vector<uint8_t>> input_buffers;
    for (const auto &name : infer_model->get_input_names()) {
        size_t frame_size = infer_model->input(name).expect("Bad input name").get_frame_size();
        std::cout << "  Input  " << name << " size=" << frame_size << '\n';
        input_buffers[name].assign(frame_size, 128);
        auto status = bindings.input(name).expect("Failed to get input binding").set_buffer(MemoryView(input_buffers[name].data(), frame_size));
        if (status != HAILO_SUCCESS) {
            std::cerr << "[-] Failed to set input buffer for " << name << ": " << status << '\n';
            return 1;
        }
    }

    // Allocate and bind output buffers
    std::map<std::string, std::vector<uint8_t>> output_buffers;
    for (const auto &name : infer_model->get_output_names()) {
        size_t frame_size = infer_model->output(name).expect("Bad output name").get_frame_size();
        std::cout << "  Output " << name << " size=" << frame_size << '\n';
        output_buffers[name].resize(frame_size, 0);
        auto status = bindings.output(name).expect("Failed to get output binding").set_buffer(MemoryView(output_buffers[name].data(), frame_size));
        if (status != HAILO_SUCCESS) {
            std::cerr << "[-] Failed to set output buffer for " << name << ": " << status << '\n';
            return 1;
        }
    }

    std::cout << "Running inference..." << '\n';
    auto run_status = configured_model.run(bindings, std::chrono::milliseconds(10000));
    if (run_status != HAILO_SUCCESS) {
        std::cerr << "[-] Inference failed: " << run_status << '\n';
        return 1;
    }

    std::cout << "Inference complete!" << '\n';
    for (auto &[name, buf] : output_buffers) {
        std::cout << "  " << name << " sample logits: ";
        for (size_t i = 0; i < 5 && i < buf.size(); ++i) {
            std::cout << (int)buf[i] << " ";
        }
        std::cout << '\n';
    }

    return 0;
}
