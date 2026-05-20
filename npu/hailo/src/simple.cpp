#include "hailo/hailort.hpp"
#include <iostream>
#include <memory>

using namespace hailort;

int main() {
    std::cout << "Initializing Hailo VDevice interface..." << '\n';

    hailo_vdevice_params_t vdevice_params{};

    hailo_status status = hailo_init_vdevice_params(&vdevice_params);
    if (status != HAILO_SUCCESS) {
        std::cerr << "Failed to initialize default VDevice parameters. Status: " << status << '\n';
        return 1;
    }

    auto vdevice_expected = VDevice::create(vdevice_params);
    if (!vdevice_expected) {
        std::cerr << "Failed to create Hailo VDevice! Status code: " << vdevice_expected.status() << '\n';
        return 1;
    }

    std::unique_ptr<VDevice> vdevice = vdevice_expected.release();
    std::cout << "Success! VDevice successfully instantiated on the PCIe bus." << '\n';

    auto physical_devices_expected = vdevice->get_physical_devices_ids();
    if (physical_devices_expected) {
        std::cout << "Active Acceleration Nodes found: " << '\n';
        for (const auto &id : physical_devices_expected.value()) {
            std::cout << "  -> Node ID: " << id << '\n';
        }
    }

    std::cout << "Shutting down VDevice interface cleanly." << '\n';
    return 0;
}
