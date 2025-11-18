#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

int main(int argc, const char * argv[]) {
    @autoreleasepool {
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
        NSString *libraryPath = @"cumsum_simd.metallib";
        NSURL *libraryURL = [NSURL fileURLWithPath:libraryPath];
        id<MTLLibrary> defaultLibrary = [device newLibraryWithURL:libraryURL error:&error];
        if (!defaultLibrary) {
            NSLog(@"Failed to load the library. Error: %@", error.localizedDescription);
            return -1;
        }
        
        NSLog(@"Functions in library:");
        for (NSString *name in defaultLibrary.functionNames) {
            NSLog(@"%@", name);
        }
        
        id<MTLFunction> cumsum = [defaultLibrary newFunctionWithName:@"cumsum_simd"];
        if (!cumsum) {
            NSLog(@"Failed to find the kernel function.");
            return -1;
        }
        NSLog(@"Kernel function: %@", cumsum.name);
        
        id<MTLComputePipelineState> computePipelineState = [device newComputePipelineStateWithFunction:cumsum error:&error];
        if (!computePipelineState) {
            NSLog(@"Failed to create compute pipeline state. Error: %@", error.localizedDescription);
            return -1;
        }
        
        float input_data[] = {1, 2, 3, 4, 5, 6, 7, 8};
        uint32_t count = sizeof(input_data) / sizeof(float);
        printf("Number of elements: %u\n", count);
        
        id<MTLBuffer> input_buffer = [device newBufferWithBytes:input_data 
                                     length:count * sizeof(float) 
                                     options:MTLResourceStorageModeShared];

        id<MTLBuffer> output_buffer = [device newBufferWithLength:count * sizeof(float) 
                                      options:MTLResourceStorageModeShared];

        id<MTLBuffer> count_buffer = [device newBufferWithBytes:&count 
                                     length:sizeof(uint32_t) 
                                     options:MTLResourceStorageModeShared];
        
        id<MTLCommandQueue> command_queue = [device newCommandQueue];
        
        id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        
        [encoder setComputePipelineState:computePipelineState];
        [encoder setBuffer:input_buffer   offset:0 atIndex:0];
        [encoder setBuffer:output_buffer   offset:0 atIndex:1];
        [encoder setBuffer:count_buffer offset:0 atIndex:2];
        
        MTLSize gridSize = MTLSizeMake(count, 1, 1);
        NSUInteger threadGroupSize = MIN(32, count);
        MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        
        float *result = (float *)[output_buffer contents];
        
        printf("Input:  ");
        for (uint32_t i = 0; i < count; i++) {
            printf("%2.0f ", input_data[i]);
        }
        printf("\nCumSum: ");
        for (uint32_t i = 0; i < count; i++) {
            printf("%2.0f ", result[i]);
        }
        printf("\n");
    }
    return 0;
}
