#include <iostream>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

int main() {
    try {
        auto device = xrt::device(0);
        
        std::cout << "Device Name: " << device.get_info<xrt::info::device::name>() << std::endl;
        std::cout << "BDF (Bus/Device/Function): " << device.get_info<xrt::info::device::bdf>() << std::endl;
        
        std::cout << "\nSUCCESS: XRT is communicating with the NPU!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Could not communicate with NPU. " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
