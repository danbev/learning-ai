#include <webgpu/webgpu_cpp.h>

#include <iostream>
#include <vector>

const char* shader_code = R"(
@group(3) @binding(0) var<storage, read> input_a: array<f32>;
@group(3) @binding(1) var<storage, read> input_b: array<f32>;
@group(3) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;

    // Bounds check to ensure we don't access out of bounds
    // Thread 0:   output[0]   = input_a[0] + input_b[0]
    // Thread 1:   output[1]   = input_a[l] + input_b[1]
    // ...
    // Thread 999: output[999] = input_a[999] + input_b[999]
    // Thread 1000-1023: Exit early due to bounds check:
    if (index >= arrayLength(&output)) {
        return;
    }
    
    output[index] = input_a[index] + input_b[index];
}
)";

class vector_addition {
private:
    // WebGPU entry point, creates and manages everything else.
    wgpu::Instance instance;
    // An adapter represents a specifc GPU implementation on the system. Help to identifty
    // the hardware and its capabilities. Can execute anything on a GPU but can get information about it.
    wgpu::Adapter adapter;
    // The actual connection to the GPU device (interface for transitting works)
    wgpu::Device device;
    // Compute pipeline for executing the vector addition shader
    wgpu::ComputePipeline pipeline;

    // WebGPU System
    // ├── Instance (WebGPU runtime )
    //     ├── Adapter (GPU #1 - Intel integrated)
    //     │   └── Device (connection to Intel GPU)
    //     └── Adapter (GPU #2 - NVIDIA discrete)  
    //         └── Device (connection to NVIDIA GPU)
    
    size_t data_size;

    static void webgpuErrorCallback(const wgpu::Device& device, wgpu::ErrorType reason,
                                   wgpu::StringView message, vector_addition* instance) {
        std::cerr << "[WebGPU Error] (" << static_cast<int>(reason) << "): " << std::string(message) << std::endl;
        std::terminate();
    }
    
public:
    vector_addition(size_t size) : data_size(size) {}
    
    bool initialize() {
        wgpu::InstanceDescriptor instance_desc = {};
        wgpu::InstanceFeatureName required_features[] = {
            wgpu::InstanceFeatureName::TimedWaitAny
        };
        instance_desc.requiredFeatureCount = 1;
        instance_desc.requiredFeatures = required_features;
        
        instance = wgpu::CreateInstance(&instance_desc);
        if (!instance) {
            std::cerr << "Failed to create WebGPU instance" << std::endl;
            return false;
        }
        std::cout << "WebGPU instance created with TimedWaitAny" << std::endl;
        
        wgpu::RequestAdapterOptions adapter_opts = {};
        adapter_opts.powerPreference = wgpu::PowerPreference::HighPerformance; // For a descrete GPU
        adapter_opts.forceFallbackAdapter = false; // Don't force fallback adapter
        adapter_opts.featureLevel = wgpu::FeatureLevel::Compatibility;
        //adapter_opts.backendType  = wgpu::BackendType::Metal;

        auto adapterCallback = [&](wgpu::RequestAdapterStatus status, wgpu::Adapter adapter, const char* message) {
            if (status != wgpu::RequestAdapterStatus::Success) {
                std::cerr << "Failed to get adapter: " << (message ? message : "Unknown error") << std::endl;
                return;
            }
            this->adapter = std::move(adapter);
            std::cout << "Adapter acquired" << std::endl;
        };
        
        wgpu::Future future = instance.RequestAdapter(&adapter_opts, wgpu::CallbackMode::AllowSpontaneous, adapterCallback);
        auto wait_status = instance.WaitAny(future, UINT64_MAX);
        if (wait_status != wgpu::WaitStatus::Success) {
            std::cerr << "Failed to wait for adapter: " << static_cast<int>(wait_status) << std::endl;
            return false;
        }
        
        if (!adapter) {
            std::cerr << "No adapter found" << std::endl;
            return false;
        }
        
        wgpu::DeviceDescriptor device_desc = {};
         // Initialize device
        std::vector<wgpu::FeatureName> dev_features = { wgpu::FeatureName::ShaderF16,
                                                        wgpu::FeatureName::ImplicitDeviceSynchronization };
        device_desc.requiredFeatures     = dev_features.data();
        device_desc.requiredFeatureCount = dev_features.size();

        auto deviceCallback = [&](wgpu::RequestDeviceStatus status, wgpu::Device result, const char* message) {
            if (status != wgpu::RequestDeviceStatus::Success) {
                std::cerr << "Failed to get device: " << (message ? message : "Unknown error") << std::endl;
                return;
            }
            device = std::move(result);
            std::cout << "Device acquired" << std::endl;
        };

        device_desc.SetUncapturedErrorCallback(webgpuErrorCallback, this);
        
        // Notice that we use the adapter to get a device.
        wgpu::Future deviceFuture = adapter.RequestDevice(&device_desc, wgpu::CallbackMode::AllowSpontaneous, deviceCallback);
        auto device_wait_status = instance.WaitAny(deviceFuture, UINT64_MAX);
        if (device_wait_status != wgpu::WaitStatus::Success) {
            std::cerr << "Failed to wait for device: " << static_cast<int>(device_wait_status) << std::endl;
            return false;
        }

        if (!device) {
            std::cerr << "No device found" << std::endl;
            return false;
        }
        

        if (!create_pipeline()) {
            return false;
        }
        
        return true;
    }
    
private:
    bool create_pipeline() {

        wgpu::ShaderSourceWGSL shader_source;
        shader_source.code = shader_code;
        
        wgpu::ShaderModuleDescriptor shader_desc;
        shader_desc.nextInChain = &shader_source;
        shader_desc.label = "Vector Addition Shader";
        
        wgpu::ShaderModule shader_module = device.CreateShaderModule(&shader_desc);
        if (!shader_module) {
            std::cerr << "Failed to create shader module - compilation failed" << std::endl;
            return false;
        }
        std::cout << "Shader module created" << std::endl;
        
        // Create compute pipeline
        wgpu::ComputePipelineDescriptor pipeline_desc;
        pipeline_desc.label = "Vector Addition Pipeline";
        pipeline_desc.compute.module = shader_module;
        pipeline_desc.compute.entryPoint = "main";
        pipeline_desc.layout = nullptr; // Auto layout
        
        pipeline = device.CreateComputePipeline(&pipeline_desc);
        if (!pipeline) {
            std::cerr << "Failed to create compute pipeline" << std::endl;
            return false;
        }
        std::cout << "Compute pipeline created successfully" << std::endl;
        
        return true;
    }
    
public:
    std::vector<float> compute(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size() || a.size() != data_size) {
            throw std::runtime_error("Data size mismatch");
        }
        
        size_t buffer_size = data_size * sizeof(float);
        
        // Create buffers
        wgpu::BufferDescriptor buffer_desc_a;
        buffer_desc_a.label = "Input Buffer A";
        buffer_desc_a.size = buffer_size;
        buffer_desc_a.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        wgpu::Buffer buffer_a = device.CreateBuffer(&buffer_desc_a);
        
        wgpu::BufferDescriptor buffer_desc_b;
        buffer_desc_b.label = "Input Buffer B";
        buffer_desc_b.size = buffer_size;
        buffer_desc_b.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        wgpu::Buffer buffer_b = device.CreateBuffer(&buffer_desc_b);
        
        wgpu::BufferDescriptor buffer_desc_out;
        buffer_desc_out.label = "Output Buffer";
        buffer_desc_out.size = buffer_size;
        // Notice that the GPU can write but the CPU cannot directly read this buffer.
        buffer_desc_out.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
        wgpu::Buffer buffer_output = device.CreateBuffer(&buffer_desc_out);
        
        // Staging buffer (for reading results back)
        wgpu::BufferDescriptor staging_desc;
        staging_desc.label = "GPU->CPU staging buffer";
        staging_desc.size = buffer_size;
        // This is used to copy from GPU storage to CPU memory.
        staging_desc.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
        wgpu::Buffer staging_buffer = device.CreateBuffer(&staging_desc);
        
        std::cout << "Buffers created" << std::endl;
        
        // Write a's data to GPU.
        device.GetQueue().WriteBuffer(buffer_a, 0, a.data(), buffer_size);
        // Write b's data to GPU.
        device.GetQueue().WriteBuffer(buffer_b, 0, b.data(), buffer_size);
        std::cout << "Data uploaded to GPU buffers" << std::endl;
        
        // Use the pipeline's auto-generated bind group layout
        wgpu::BindGroupLayout bind_group_layout = pipeline.GetBindGroupLayout(3);
        
        // Create bind group using the pipeline's layout
        std::vector<wgpu::BindGroupEntry> bind_entries(3);
        
        bind_entries[0].binding = 0;
        bind_entries[0].buffer = buffer_a;
        bind_entries[0].offset = 0;
        bind_entries[0].size = buffer_size;
        
        bind_entries[1].binding = 1;
        bind_entries[1].buffer = buffer_b;
        bind_entries[1].offset = 0;
        bind_entries[1].size = buffer_size;
        
        bind_entries[2].binding = 2;
        bind_entries[2].buffer = buffer_output;
        bind_entries[2].offset = 0;
        bind_entries[2].size = buffer_size;
        
        wgpu::BindGroupDescriptor bind_group_desc;
        bind_group_desc.layout = bind_group_layout;  // Use pipeline's layout!
        bind_group_desc.entryCount = bind_entries.size();
        bind_group_desc.entries = bind_entries.data();
        bind_group_desc.label = "Vector Addition Bind Group";
        
        wgpu::BindGroup bind_group = device.CreateBindGroup(&bind_group_desc);
        if (!bind_group) {
            std::cerr << "Failed to create bind group" << std::endl;
            throw std::runtime_error("Bind group creation failed");
        }
        std::cout << "Bind group created with pipeline layout" << std::endl;
        
        // Create command encoder
        wgpu::CommandEncoderDescriptor encoderDesc;
        encoderDesc.label = "Vector Addition Command Encoder";
        wgpu::CommandEncoder encoder = device.CreateCommandEncoder(&encoderDesc);
        
        // Begin compute pass (to record compute operations, nothing is executed yet)
        wgpu::ComputePassDescriptor passDesc;
        passDesc.label = "Vector Addition Compute Pass";
        wgpu::ComputePassEncoder computePass = encoder.BeginComputePass(&passDesc);
        
        // Set pipeline and bind group
        computePass.SetPipeline(pipeline);
        computePass.SetBindGroup(3, bind_group);
        
        // Dispatch compute shader
        uint32_t workgroups = (data_size + 63) / 64;
        std::cout << "Dispatching " << workgroups << " workgroups for " << data_size << " elements" << std::endl;
        computePass.DispatchWorkgroups(workgroups);
        
        computePass.End();
        
        // Copy result to staging buffer
        encoder.CopyBufferToBuffer(buffer_output, 0, staging_buffer, 0, buffer_size);
        
        // This is what actually sends the commands to the GPU for execution.
        wgpu::CommandBufferDescriptor cmd_buffer_desc;
        cmd_buffer_desc.label = "Vector Addition Commands";
        wgpu::CommandBuffer commands = encoder.Finish(&cmd_buffer_desc);
        device.GetQueue().Submit(1, &commands);
        std::cout << "Commands submitted to GPU" << std::endl;
        
        wgpu::QueueWorkDoneStatus work_done_status = wgpu::QueueWorkDoneStatus::Success; // Initialize to Success
        bool work_done = false;
        
        auto work_done_callback = [&](wgpu::QueueWorkDoneStatus status, const char* message) {
            work_done_status = status;
            work_done = true;
            std::cout << "GPU work completed with status: " << static_cast<int>(status);
            if (message) {
                std::cout << ", message: " << message;
            }
            std::cout << std::endl;
        };
        
        using wgpu::CallbackMode;
        wgpu::Future work_done_future = device.GetQueue().OnSubmittedWorkDone(CallbackMode::AllowSpontaneous, work_done_callback);
        auto work_wait_status = instance.WaitAny(work_done_future, UINT64_MAX);
        
        if (work_wait_status != wgpu::WaitStatus::Success) {
            std::cerr << "Failed to wait for GPU work: " << static_cast<int>(work_wait_status) << std::endl;
            return std::vector<float>(data_size, 0.0f);
        }
        
        if (work_done_status != wgpu::QueueWorkDoneStatus::Success) {
            std::cerr << "GPU work failed with status: " << static_cast<int>(work_done_status) << std::endl;
            return std::vector<float>(data_size, 0.0f);
        }
        
        // Read back results
        std::vector<float> results(data_size);
        bool map_complete = false;
        
        // Map staging buffer for reading
        auto map_callback = [&](wgpu::MapAsyncStatus status, const char* message) {
            map_complete = true;
            if (status != wgpu::MapAsyncStatus::Success) {
                std::cerr << "Failed to map buffer: " << (message ? message : "Unknown error") << std::endl;
            } else {
                std::cout << "Buffer mapped successfully" << std::endl;
            }
        };
        
        wgpu::Future map_future = staging_buffer.MapAsync(wgpu::MapMode::Read, 0, buffer_size,
                                                       wgpu::CallbackMode::AllowSpontaneous, map_callback);
        auto map_wait_status = instance.WaitAny(map_future, UINT64_MAX);
        if (map_wait_status != wgpu::WaitStatus::Success) {
            std::cerr << "Failed to wait for buffer mapping: " << static_cast<int>(map_wait_status) << std::endl;
            return results;
        }
        
        // Copy data from mapped buffer
        const float* mapped_data = static_cast<const float*>(staging_buffer.GetConstMappedRange());
        if (mapped_data) {
            std::copy(mapped_data, mapped_data + data_size, results.begin());
            std::cout << "Data copied from GPU buffer" << std::endl;
        } else {
            std::cerr << "Failed to get mapped range" << std::endl;
        }
        
        staging_buffer.Unmap();
        
        return results;
    }
};

int main() {
    std::cout << "WebGPU Vector Addition Example\n" << std::endl;
    
    const size_t data_size = 1000;
    std::vector<float> input_a(data_size), input_b(data_size);
    for (size_t i = 0; i < data_size; i++) {
        input_a[i] = static_cast<float>(i);
        input_b[i] = static_cast<float>(i * 2);
    }
    
    std::cout << "Test data: A[0]=" << input_a[0] << ", B[0]=" << input_b[0] << ", expected=" << (input_a[0] + input_b[0]) << std::endl;
    std::cout << "Test data: A[1]=" << input_a[1] << ", B[1]=" << input_b[1] << ", expected=" << (input_a[1] + input_b[1]) << std::endl;
    
    // Create and initialize vector addition
    vector_addition vector_add(data_size);
    if (!vector_add.initialize()) {
        std::cerr << "Failed to initialize WebGPU" << std::endl;
        return 1;
    }
    
    std::cout << "\nComputing vector addition on GPU..." << std::endl;
    
    std::vector<float> results = vector_add.compute(input_a, input_b);
    
    std::cout << "\nVerifying results..." << std::endl;
    std::cout << "First few results: " << results[0] << ", " << results[1] << ", " << results[2] << std::endl;
    
    bool success = true;
    for (size_t i = 0; i < std::min(data_size, size_t(10)); i++) {
        float expected = input_a[i] + input_b[i];
        if (std::abs(results[i] - expected) > 1e-5f) {
            std::cerr << "Mismatch at index " << i << ": got " << results[i] 
                      << ", expected " << expected << std::endl;
            success = false;
        } else {
            std::cout << "Index " << i << ": " << input_a[i] << " + " << input_b[i] << " = " << results[i] << std::endl;
        }
    }
    
    if (success) {
        std::cout << "\nAll checked results correct!" << std::endl;
    } else {
        std::cout << "\nResults verification failed!" << std::endl;
    }
    
    return success ? 0 : 1;
}
