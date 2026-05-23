#include "hailo/hailort.hpp"
#include <chrono>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace hailort;

int main() {
    std::string hef_path = "hefs/qwen2.5.hef";
    std::string network_group_name = "base_model__prefill";

    auto vdevice = VDevice::create().expect("Failed to create VDevice");

    auto infer_model = vdevice->create_infer_model(hef_path, network_group_name).expect("Failed to create infer model");
    infer_model->set_batch_size(1);

    auto configured_model = infer_model->configure().expect("Failed to configure infer model");

    auto bindings = configured_model.create_bindings().expect("Failed to create bindings");

    // Allocate and bind input buffers
    std::map<std::string, std::vector<uint8_t>> input_buffers;
    for (const auto &name : infer_model->get_input_names()) {
        size_t frame_size = infer_model->input(name).expect("Bad input name").get_frame_size();
        std::cout << "  Input  " << name << " -> Required Byte Size: " << frame_size << '\n';
        
        input_buffers[name].assign(frame_size, 128);
        
        auto status = bindings.input(name).expect("Failed to get input binding").set_buffer(MemoryView(input_buffers[name].data(), frame_size));
        if (status != HAILO_SUCCESS) {
            std::cerr << "[-] Failed to bind input " << name << ": " << status << '\n';
            return 1;
        }
    }

    std::map<std::string, std::vector<uint8_t>> output_buffers;
    for (const auto &name : infer_model->get_output_names()) {
        size_t frame_size = infer_model->output(name).expect("Bad output name").get_frame_size();
        std::cout << "  Output " << name << " -> Required Byte Size: " << frame_size << '\n';
        
        output_buffers[name].resize(frame_size, 0);
        
        auto status = bindings.output(name).expect("Failed to get output binding").set_buffer(MemoryView(output_buffers[name].data(), frame_size));
        if (status != HAILO_SUCCESS) {
            std::cerr << "[-] Failed to bind output " << name << ": " << status << '\n';
            return 1;
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    
    auto run_status = configured_model.run(bindings, std::chrono::milliseconds(20000));
    if (run_status != HAILO_SUCCESS) {
        std::cerr << "[-] Prefill inference failed: " << run_status << '\n';
        return 1;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Prefill processing complete! Time taken: " << duration.count() << " ms" << '\n';

    // Print out samples of the processed prompt logits
    for (auto &[name, buf] : output_buffers) {
        std::cout << "  " << name << " sample context logits: ";
        for (size_t i = 0; i < 5 && i < buf.size(); ++i) {
            std::cout << (int)buf[i] << " ";
        }
        std::cout << '\n';
    }

    std::cout << "Releasing Prefill context descriptors to clear NPU lanes..." << '\n';
    return 0;
}
