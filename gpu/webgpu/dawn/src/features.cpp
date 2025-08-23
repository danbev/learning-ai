// Simple ShaderF16 Feature Test - Compatible with most Dawn versions
#include <webgpu/webgpu_cpp.h>
#include <iostream>

int main() {
    std::cout << "Testing ShaderF16 support..." << std::endl;
    
    // Create instance with TimedWaitAny
    wgpu::InstanceFeatureName instanceFeatures[] = { wgpu::InstanceFeatureName::TimedWaitAny };
    wgpu::InstanceDescriptor instanceDesc{};
    instanceDesc.requiredFeatureCount = 1;
    instanceDesc.requiredFeatures = instanceFeatures;
    wgpu::Instance instance = wgpu::CreateInstance(&instanceDesc);
    
    if (!instance) {
        std::cerr << "Failed to create instance" << std::endl;
        return 1;
    }
    
    // Get adapter
    wgpu::RequestAdapterOptions adapterOptions{};
    adapterOptions.powerPreference = wgpu::PowerPreference::HighPerformance;
    
    wgpu::Adapter adapter;
    bool adapterReady = false;
    
    auto adapterCallback = [&](wgpu::RequestAdapterStatus status, wgpu::Adapter result, const char* message) {
        if (status == wgpu::RequestAdapterStatus::Success) {
            adapter = std::move(result);
        } else {
            std::cerr << "Adapter request failed: " << (message ? message : "Unknown") << std::endl;
        }
        adapterReady = true;
    };
    
    auto adapterFuture = instance.RequestAdapter(&adapterOptions, wgpu::CallbackMode::AllowSpontaneous, adapterCallback);
    instance.WaitAny(adapterFuture, UINT64_MAX);
    
    if (!adapter) {
        std::cerr << "No adapter available" << std::endl;
        return 1;
    }
    
    // Get adapter info
    wgpu::AdapterInfo info{};
    adapter.GetInfo(&info);
    std::cout << "Testing on: ";
    if (info.device.data && info.device.length > 0) {
        std::cout << std::string(info.device.data, info.device.length);
    }
    std::cout << " (Backend: ";
    switch(info.backendType) {
        case wgpu::BackendType::Vulkan: std::cout << "Vulkan"; break;
        case wgpu::BackendType::D3D12: std::cout << "D3D12"; break;
        case wgpu::BackendType::Metal: std::cout << "Metal"; break;
        default: std::cout << "Other"; break;
    }
    std::cout << ")" << std::endl;
    
    // Test ShaderF16 by trying to create a device with it
    std::cout << "\nTesting ShaderF16 support..." << std::endl;
    
    wgpu::FeatureName requiredFeatures[] = { wgpu::FeatureName::ShaderF16 };
    wgpu::DeviceDescriptor deviceDesc{};
    deviceDesc.requiredFeatureCount = 1;
    deviceDesc.requiredFeatures = requiredFeatures;
    deviceDesc.label = "ShaderF16 Test Device";
    
    bool deviceReady = false;
    bool shaderF16Supported = false;
    wgpu::Device device;
    
    auto deviceCallback = [&](wgpu::RequestDeviceStatus status, wgpu::Device result, const char* message) {
        if (status == wgpu::RequestDeviceStatus::Success) {
            device = std::move(result);
            shaderF16Supported = true;
            std::cout << "✓ ShaderF16 IS supported!" << std::endl;
        } else {
            shaderF16Supported = false;
            std::cout << "✗ ShaderF16 NOT supported" << std::endl;
            if (message) {
                std::cout << "  Error: " << message << std::endl;
            }
        }
        deviceReady = true;
    };
    
    auto deviceFuture = adapter.RequestDevice(&deviceDesc, wgpu::CallbackMode::AllowSpontaneous, deviceCallback);
    instance.WaitAny(deviceFuture, UINT64_MAX);
    
    // Test without ShaderF16 for comparison
    std::cout << "\nTesting device creation without ShaderF16..." << std::endl;
    
    wgpu::DeviceDescriptor basicDeviceDesc{};
    basicDeviceDesc.requiredFeatureCount = 0;
    basicDeviceDesc.requiredFeatures = nullptr;
    basicDeviceDesc.label = "Basic Test Device";
    
    bool basicDeviceReady = false;
    bool basicDeviceSupported = false;
    wgpu::Device basicDevice;
    
    auto basicDeviceCallback = [&](wgpu::RequestDeviceStatus status, wgpu::Device result, const char* message) {
        if (status == wgpu::RequestDeviceStatus::Success) {
            basicDevice = std::move(result);
            basicDeviceSupported = true;
            std::cout << "✓ Basic device creation works" << std::endl;
        } else {
            basicDeviceSupported = false;
            std::cout << "✗ Basic device creation failed" << std::endl;
            if (message) {
                std::cout << "  Error: " << message << std::endl;
            }
        }
        basicDeviceReady = true;
    };
    
    auto basicDeviceFuture = adapter.RequestDevice(&basicDeviceDesc, wgpu::CallbackMode::AllowSpontaneous, basicDeviceCallback);
    instance.WaitAny(basicDeviceFuture, UINT64_MAX);
    
    // Results and recommendations
    std::cout << "\n=== RESULTS ===" << std::endl;
    if (shaderF16Supported) {
        std::cout << "ShaderF16: ✓ SUPPORTED" << std::endl;
        std::cout << "GGML should work with F16 optimizations!" << std::endl;
        std::cout << "If GGML still fails, check GGML's WebGPU backend version compatibility." << std::endl;
    } else {
        std::cout << "ShaderF16: ✗ NOT SUPPORTED" << std::endl;
        std::cout << "This is why GGML fails with your current setup." << std::endl;
    }
    
    if (basicDeviceSupported) {
        std::cout << "Basic WebGPU: ✓ WORKING" << std::endl;
        std::cout << "Your WebGPU setup is functional, just missing F16 support." << std::endl;
    } else {
        std::cout << "Basic WebGPU: ✗ BROKEN" << std::endl;
        std::cout << "There's a deeper issue with your WebGPU setup." << std::endl;
    }
    
    // Cleanup
    if (device) device.Destroy();
    if (basicDevice) basicDevice.Destroy();
    
    return shaderF16Supported ? 0 : 1;
}
