#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

int main(int argc, const char * argv[]) {
    @autoreleasepool {
	// Metal (MLT) Device
	NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
	id<MTLDevice> device = nil;
	for (id<MTLDevice> availableDevice in devices) {
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

        NSError *error = nil;
        NSString *libraryPath = @"/Users/danbev/work/ai/learning-ai/gpu/metal/kernel.metallib";
        id<MTLLibrary> defaultLibrary = [device newLibraryWithFile:libraryPath error:&error];
        if (!defaultLibrary) {
            NSLog(@"Failed to load the library. Error: %@", error.localizedDescription);
            return -1;
        }
	NSLog(@"Functions in library:");
	for (NSString *name in defaultLibrary.functionNames) {
	    NSLog(@"    %@", name);
	}

        id<MTLFunction> kernelFunction = [defaultLibrary newFunctionWithName:@"simpleMultiply"];
        if (!kernelFunction) {
            NSLog(@"Failed to find the kernel function.");
            return -1;
        }
	NSLog(@"Kernel function: %@", kernelFunction.name);

        id<MTLComputePipelineState> computePipelineState = [device newComputePipelineStateWithFunction:kernelFunction error:&error];
        if (!computePipelineState) {
            NSLog(@"Failed to create compute pipeline state. Error: %@", error.localizedDescription);
            return -1;
        }

        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        NSUInteger dataSize = 1024; // number of float elements
        float *inputData = (float *)malloc(dataSize * sizeof(float));
        float *outputData = (float *)malloc(dataSize * sizeof(float));

        for (NSUInteger i = 0; i < dataSize; i++) {
            inputData[i] = (float)i;
        }

        id<MTLBuffer> inputBuffer = [device newBufferWithBytes:inputData length:dataSize * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> outputBuffer = [device newBufferWithLength:dataSize * sizeof(float) options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        [computeEncoder setComputePipelineState:computePipelineState];
        [computeEncoder setBuffer:inputBuffer offset:0 atIndex:0];
        [computeEncoder setBuffer:outputBuffer offset:0 atIndex:1];

        MTLSize gridSize = {dataSize, 1, 1};
        NSUInteger threadGroupSize = computePipelineState.maxTotalThreadsPerThreadgroup;
        if (threadGroupSize > dataSize) {
            threadGroupSize = dataSize;
        }
        MTLSize threadgroupSize = {threadGroupSize, 1, 1};
        [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [computeEncoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        memcpy(outputData, [outputBuffer contents], dataSize * sizeof(float));
        for (NSUInteger i = 0; i < dataSize; i++) {
            //NSLog(@"Output[%lu] = %f", (unsigned long)i, outputData[i]);
        }

        free(inputData);
        free(outputData);
    }
    return 0;
}

