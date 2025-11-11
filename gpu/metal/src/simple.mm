#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

int main(int argc, const char * argv[]) {
    // Autorelease pool for memory management like RAII in C++
    @autoreleasepool {
        // Metal (MLT) Device is protocol (interface).
        // The following is simlar to CUDA's cudaGetDeviceCount.
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();

        // Recall that id is like a void pointer in C/C++
        id<MTLDevice> device = nil;
        for (id<MTLDevice> availableDevice in devices) {
            // Recall that method calls use square brackets [object methodName:parameters]
            // And @"string" is a string literal in Objective-C.
            // Notice that here the object is availableDevice and the method is name.
            if ([availableDevice.name isEqualToString:@"Apple M3"]) {
                device = availableDevice;
                NSLog(@"Using Metal device: %@", device.name);
                break;
            }
        }

        if (!device) {
            NSLog(@"Apple M3 GPU is not available.");
            return -1;
        }

        // Load the precompiled Metal library from a .metallib file
        NSError *error = nil;
        NSString *libraryPath = @"kernel.metallib";
        NSURL *libraryURL = [NSURL fileURLWithPath:libraryPath];
        id<MTLLibrary> defaultLibrary = [device newLibraryWithURL:libraryURL error:&error];
        if (!defaultLibrary) {
            NSLog(@"Failed to load the library. Error: %@", error.localizedDescription);
            return -1;
        }
        NSLog(@"Functions in library:");
        for (NSString *name in defaultLibrary.functionNames) {
            NSLog(@"    %@", name);
        }

        // Create a compute function from the library
        id<MTLFunction> kernelFunction = [defaultLibrary newFunctionWithName:@"simple_multiply"];
        if (!kernelFunction) {
            NSLog(@"Failed to find the kernel function.");
            return -1;
        }
        NSLog(@"Kernel function: %@", kernelFunction.name);

        // Create a compute pipeline state. This is what compiles the AIR to run on the GPU.
        // This is similar to compiling from PTX to SASS in CUDA.
        id<MTLComputePipelineState> computePipelineState = [device newComputePipelineStateWithFunction:kernelFunction error:&error];
        if (!computePipelineState) {
            NSLog(@"Failed to create compute pipeline state. Error: %@", error.localizedDescription);
            return -1;
        }

        // Command queue is used for submitting work to the GPU, similar to CUDA streams.
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        NSUInteger dataSize = 1024; // number of float elements
        float *inputData = (float *)malloc(dataSize * sizeof(float));
        float *outputData = (float *)malloc(dataSize * sizeof(float));

        for (NSUInteger i = 0; i < dataSize; i++) {
            inputData[i] = (float)i;
        }
        int constant_value = 2;

        // Create buffers, and this is pretty nice as Metal has a unified memory model so
        // both the CPU and GPU can access the same memory. So we don't need to allocate explicitly
        // on both sides, nore do we need to copy data between them.
        id<MTLBuffer> inputBuffer = [device newBufferWithBytes:inputData length:dataSize * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> outputBuffer = [device newBufferWithLength:dataSize * sizeof(float) options:MTLResourceStorageModeShared];
        // Debug buffer as we can printf from a kernel in Metal
        id<MTLBuffer> debugBuffer = [device newBufferWithLength:10 * sizeof(int) options:MTLResourceStorageModeShared];
        // Constant
        id<MTLBuffer> constant_buffer = [device newBufferWithLength:1 * sizeof(int) options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        [computeEncoder setComputePipelineState:computePipelineState];
        [computeEncoder setBuffer:inputBuffer offset:0 atIndex:0];
        [computeEncoder setBuffer:outputBuffer offset:0 atIndex:1];
        [computeEncoder setBuffer:debugBuffer offset:0 atIndex:2];
        [computeEncoder setBytes:&constant_value length:sizeof(int) atIndex:3];

        // Calculate threadgroup and grid sizes
        MTLSize gridSize = {dataSize, 1, 1}; // Total threads needed.
        NSUInteger threadGroupSize = computePipelineState.maxTotalThreadsPerThreadgroup; // Threads per group.
        if (threadGroupSize > dataSize) {
            threadGroupSize = dataSize;
        }
        MTLSize threadgroupSize = {threadGroupSize, 1, 1};

        // Dispatch the compute kernel.
        [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [computeEncoder endEncoding];
        // The above is equivalent to the following CUDA kernel launch:
        // dim3 threads(threadGroupSize, 1, 1);
        // dim3 blocks((dataSize + threadGroupSize - 1) / threadGroupSize, 1, 1);
        // myKernel<<<blocks, threads>>>(input, output);

        // Submit to GPU.
        [commandBuffer commit];
        // Block until done.
        [commandBuffer waitUntilCompleted];

        memcpy(outputData, [outputBuffer contents], dataSize * sizeof(float));
        for (NSUInteger i = 0; i < dataSize; i++) {
            //NSLog(@"Output[%lu] = %f", (unsigned long)i, outputData[i]);
        }

        int* debugData = (int*)[debugBuffer contents];
        for (int i = 0; i < 10; i++) {
            NSLog(@"Debug[%d] = %d", i, debugData[i]);
        }

        free(inputData);
        free(outputData);
    }
    return 0;
}

