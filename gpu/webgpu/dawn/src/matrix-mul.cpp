#include <vector>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <dawn/webgpu_cpp.h>
#include <dawn/native/DawnNative.h>

// This shader performs matrix multiplication in parallel on the GPU.
// Each workgroup thread computes one element of the result matrix.
const char* computeShader = R"(
    @group(0) @binding(0) var<storage, read> matrixA: array<f32>;
    @group(0) @binding(1) var<storage, read> matrixB: array<f32>;
    @group(0) @binding(2) var<storage, read_write> resultMatrix: array<f32>;

    @compute @workgroup_size(3, 3)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let row = global_id.x;
        let col = global_id.y;

        let M = 3u;  // Number of rows in A
        let K = 2u;  // Number of columns in A / rows in B
        let N = 3u;  // Number of columns in B

        if (row < M && col < N) {
            var sum = 0.0;
            for (var i = 0u; i < K; i = i + 1u) {
                let a_index = row * K + i;
                let b_index = i * N + col;
                sum = sum + matrixA[a_index] * matrixB[b_index];
            }
            resultMatrix[row * N + col] = sum;
        }
    }
)";

// Helper function to print matrices in a readable format
void printMatrix(const std::vector<float>& matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << std::setw(8) << matrix[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main() {
    // Initialize WebGPU instance
    auto instance = wgpu::CreateInstance();
    //auto instance = std::make_unique<dawn::native::Instance>();
    if (!instance) {
        std::cerr << "Could not initialize WebGPU instance\n";
        return 1;
    }

    wgpu::Adapter adapter;
    wgpu::RequestAdapterOptions adapterOpts = {};
    //adapterOpts.powerPreference = wgpu::PowerPreference::HighPerformance;
    //adapterOpts.backendType = wgpu::BackendType::Vulkan;
    //adapterOpts.adapterType = wgpu::AdapterType::DiscreteGPU;

    instance.RequestAdapter(
        &adapterOpts,
        wgpu::CallbackMode::WaitAnyOnly,
        [&adapter](wgpu::RequestAdapterStatus status, 
                   wgpu::Adapter receivedAdapter, 
                   const char* message) {
            if (status == wgpu::RequestAdapterStatus::Success) {
                adapter = receivedAdapter;
            } else {
                std::cerr << "Adapter request failed: message: " << message << std::endl;
            }
        }
    );

    if (!adapter) {
        std::cerr << "Could not get WebGPU adapter\n";
        return 1;
    }

    // Create a device from the adapter
    wgpu::DeviceDescriptor deviceDesc = {};
    wgpu::Device device = adapter.CreateDevice(&deviceDesc);
    if (!device) {
        std::cerr << "Could not create WebGPU device\n";
        return 1;
    }

    std::vector<float> matrixA = {
        1.0f, 2.0f,
        1.0f, 2.0f,
        1.0f, 2.0f
    };
    std::vector<float> matrixB = {
        1.0f, 2.0f, 3.0f,
        1.0f, 2.0f, 3.0f
    };
    std::vector<float> result(9, 0.0f);  // 3x3 result matrix

    // Create buffer for matrix A
    wgpu::BufferDescriptor bufferADesc;
    bufferADesc.size = sizeof(float) * matrixA.size();
    bufferADesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    bufferADesc.mappedAtCreation = true;
    wgpu::Buffer bufferA = device.CreateBuffer(&bufferADesc);
    memcpy(bufferA.GetMappedRange(), matrixA.data(), sizeof(float) * matrixA.size());
    bufferA.Unmap();

    // Create buffer for matrix B
    wgpu::BufferDescriptor bufferBDesc;
    bufferBDesc.size = sizeof(float) * matrixB.size();
    bufferBDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    bufferBDesc.mappedAtCreation = true;
    wgpu::Buffer bufferB = device.CreateBuffer(&bufferBDesc);
    memcpy(bufferB.GetMappedRange(), matrixB.data(), sizeof(float) * matrixB.size());
    bufferB.Unmap();

    // Create buffer for result matrix
    wgpu::BufferDescriptor resultDesc;
    resultDesc.size = sizeof(float) * result.size();
    resultDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    wgpu::Buffer resultBuffer = device.CreateBuffer(&resultDesc);

    // Create shader module
    wgpu::ShaderModuleWGSLDescriptor wgslDesc;
    wgslDesc.code = computeShader;
    
    wgpu::ShaderModuleDescriptor shaderDesc;
    shaderDesc.nextInChain = &wgslDesc;
    wgpu::ShaderModule computeModule = device.CreateShaderModule(&shaderDesc);

    // Create compute pipeline
    wgpu::ComputePipelineDescriptor pipelineDesc;
    pipelineDesc.compute.module = computeModule;
    pipelineDesc.compute.entryPoint = "main";
    wgpu::ComputePipeline pipeline = device.CreateComputePipeline(&pipelineDesc);

    // Create bind group for passing data to the shader
    std::array<wgpu::BindGroupEntry, 3> entries;
    entries[0].binding = 0;
    entries[0].buffer = bufferA;
    entries[0].size = sizeof(float) * matrixA.size();

    entries[1].binding = 1;
    entries[1].buffer = bufferB;
    entries[1].size = sizeof(float) * matrixB.size();

    entries[2].binding = 2;
    entries[2].buffer = resultBuffer;
    entries[2].size = sizeof(float) * result.size();

    wgpu::BindGroupDescriptor bindGroupDesc;
    bindGroupDesc.layout = pipeline.GetBindGroupLayout(0);
    bindGroupDesc.entryCount = entries.size();
    bindGroupDesc.entries = entries.data();
    wgpu::BindGroup bindGroup = device.CreateBindGroup(&bindGroupDesc);

    // Create and record compute commands
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    wgpu::ComputePassEncoder computePass = encoder.BeginComputePass();

    computePass.SetPipeline(pipeline);
    computePass.SetBindGroup(0, bindGroup);
    computePass.DispatchWorkgroups(1, 1);  // Launch one workgroup of 3x3 threads
    computePass.End();

    // Create staging buffer for reading results back to CPU
    wgpu::BufferDescriptor stagingDesc;
    stagingDesc.size = sizeof(float) * result.size();
    stagingDesc.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
    wgpu::Buffer stagingBuffer = device.CreateBuffer(&stagingDesc);

    // Copy result to staging buffer
    encoder.CopyBufferToBuffer(
        resultBuffer, 0,
        stagingBuffer, 0,
        sizeof(float) * result.size()
    );

    // Submit commands to GPU
    wgpu::CommandBuffer commands = encoder.Finish();
    device.GetQueue().Submit(1, &commands);

    // Map the staging buffer to read results
    bool bufferMapped = false;

    stagingBuffer.MapAsync(
        wgpu::MapMode::Read,
        /* offset = */ (size_t)0,
        (size_t) sizeof(float) * result.size(),
        wgpu::CallbackMode::WaitAnyOnly,
        [](wgpu::MapAsyncStatus status, const char* message, void* userdata) {
            auto bufferMappedPtr = static_cast<bool*>(userdata);
            if (status == wgpu::MapAsyncStatus::Success) {
                *bufferMappedPtr = true;
            } else {
                std::cerr << "Failed to map staging buffer: " << (message ? message : "Unknown error") << "\n";
            }
        },
        // Pass the address of bufferMapped so our lambda can update it
        (void*)&stagingBuffer
    );

    // Wait for the mapping operation to complete
    while (!bufferMapped) {
        device.Tick();
    }

    // Read and print results
    const float* mappedData = static_cast<const float*>(stagingBuffer.GetConstMappedRange());
    std::vector<float> resultData(mappedData, mappedData + result.size());

    std::cout << "Matrix A (3x2):\n";
    printMatrix(matrixA, 3, 2);
    std::cout << "Matrix B (2x3):\n";
    printMatrix(matrixB, 2, 3);
    std::cout << "Result Matrix (3x3):\n";
    printMatrix(resultData, 3, 3);

    // Cleanup
    stagingBuffer.Unmap();
    return 0;
}
