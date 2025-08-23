#include <iostream>
#include <cstdlib>
#include <webgpu/webgpu_cpp.h>

int main() {
    std::cout << "Attempting to create WebGPU instance..." << std::endl;
    
    wgpu::InstanceDescriptor instanceDescriptor{};

    // This is an array of required features for the instanceDescriptor above.
    wgpu::InstanceFeatureName requiredFeatures[] = {
        wgpu::InstanceFeatureName::TimedWaitAny
    };
    instanceDescriptor.requiredFeatureCount = 1;  // requiring 1 feature
    instanceDescriptor.requiredFeatures = requiredFeatures;

    wgpu::Instance instance = wgpu::CreateInstance(&instanceDescriptor);
    if (instance == nullptr) {
        std::cerr << "Instance creation failed!" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "WebGPU instance created successfully!" << std::endl;
    
    wgpu::RequestAdapterOptions options = {};
    wgpu::Adapter adapter;
    
    auto callback = [&adapter](wgpu::RequestAdapterStatus status, wgpu::Adapter result, const char * message) {
        if (status != wgpu::RequestAdapterStatus::Success) {
            std::cerr << "Failed to get an adapter: " <<
            (message ? message : "Unknown error") << std::endl;
            return;
        }
        adapter = std::move(result);
        std::cout << "Adapter found successfully!" << std::endl;
    };
    
    std::cout << "Requesting adapter..." << std::endl;
    
    // This will request an adapter, and the callback may be called anytime, (AllowSpontaneous)
    wgpu::Future future = instance.RequestAdapter(&options, wgpu::CallbackMode::AllowSpontaneous, callback);

    // Block/await forever for result.
    instance.WaitAny(future, UINT64_MAX);
    
    if (adapter == nullptr) {
        std::cerr << "RequestAdapter failed - no adapter returned!" << std::endl;
        return EXIT_FAILURE;
    }
    
    std::cout << "Adapter request completed successfully!" << std::endl;
    
    // Get adapter limits and info like ggml does
    wgpu::Limits limits{};
    if (!adapter.GetLimits(&limits)) {
        std::cerr << "Failed to get adapter limits!" << std::endl;
        return 1;
    }
    std::cout << "Adapter limits retrieved successfully!" << std::endl;
    
    wgpu::AdapterInfo info{};
    if (!adapter.GetInfo(&info)) {
        std::cerr << "Failed to get adapter info!" << std::endl;
        return 2;
    }
    
    std::cout << "Adapter Information:" << std::endl;
    if (info.vendor.data && info.vendor.length > 0) {
        std::cout << "  Vendor: " << std::string(info.vendor.data, info.vendor.length) << std::endl;
    }
    if (info.architecture.data && info.architecture.length > 0) {
        std::cout << "  Architecture: " << std::string(info.architecture.data, info.architecture.length) << std::endl;
    }
    if (info.device.data && info.device.length > 0) {
        std::cout << "  Device: " << std::string(info.device.data, info.device.length) << std::endl;
    }
    if (info.description.data && info.description.length > 0) {
        std::cout << "  Description: " << std::string(info.description.data, info.description.length) << std::endl;
    }
    
    return EXIT_SUCCESS;
}
