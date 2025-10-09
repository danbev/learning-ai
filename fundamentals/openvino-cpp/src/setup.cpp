#include <openvino/openvino.hpp>
#include <cstdio>
#include <string>

int main(int argc, char* argv[]) {
    try {
        // Create OpenVINO Runtime Core
        ov::Core core;

        printf("OpenVINO Runtime initialized successfully!\n");
        printf("OpenVINO Version: %s\n\n", ov::get_openvino_version().description);

        // Get available devices
        std::vector<std::string> devices = core.get_available_devices();

        printf("Available devices:\n");
        for (const auto& device : devices) {
            printf("%s\n", device.c_str());

            // Get device properties
            printf("  Full device name: %s\n", core.get_property(device, ov::device::full_name).c_str());

            auto device_type = core.get_property(device, ov::device::type);
            const char* type_str = "Unknown";
            switch (device_type) {
                case ov::device::Type::INTEGRATED:
                    type_str = "Integrated";
                    break;
                case ov::device::Type::DISCRETE:
                    type_str = "Discrete";
                    break;
            }
            printf("  Type: %s\n", type_str);
        }

        return 0;
    }

    catch (const std::exception& ex) {
        fprintf(stderr, "Error: %s\n", ex.what());
        return 1;
    }
}
