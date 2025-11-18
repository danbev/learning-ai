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
        NSString *libraryPath = @"cumsum.metallib";
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
        
        id<MTLFunction> cumsum = [defaultLibrary newFunctionWithName:@"cumsum_scan"];
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
        
        // First buffer is the input buffer which we create with the input data.
        id<MTLBuffer> input_buffer = [device newBufferWithBytes:input_data 
                                     length:count * sizeof(float) 
                                     options:MTLResourceStorageModeShared];

        // Notice that this buffer does not have any input data, only a length as this is the
        // output buffer.
        id<MTLBuffer> output_buffer = [device newBufferWithLength:count * sizeof(float) 
                                      options:MTLResourceStorageModeShared];

        // And then we have another buffer which is also created with a value, which is the
        // count, or number of elements to process.
        id<MTLBuffer> count_buffer = [device newBufferWithBytes:&count 
                                     length:sizeof(uint32_t) 
                                     options:MTLResourceStorageModeShared];
        
        id<MTLCommandQueue> command_queue = [device newCommandQueue];
        
        // Create pointer to the input and output buffers so that we can swap them later.
        id<MTLBuffer> src_buffer = input_buffer;
        id<MTLBuffer> dst_buffer = output_buffer;
        uint32_t num_passes = 0;
        // This is just left shifting 1 until it is >= count, so we start with
        // s=0, 1u << 0 = 1
        // s=1, 1u << 1 = 2
        // s=2, 1u << 2 = 4
        // s=3, 1u << 3 = 8
        for (uint32_t s = 0; (1u << s) < count; s++) {
            num_passes++;
        }
        // And we also need to include the initial copy of the first element.
        num_passes++;
        printf("Number of passes: %u\n", num_passes);

        
        // So we iterate passses time (currently 4 for 8 elements).
        for (uint32_t pass = 0; pass < num_passes; pass++) {
            uint32_t step = (pass == 0) ? 0 : (1u << (pass - 1));

            // Recall that cumsum is the cumulative sum so the result from one pass will become
            // input to the second pass and so on.
            // pass = 0, just copy the input to the output:
            // [1     2     3     4     5     6     7      8]

            // pass = 1, add each element with the element 1 position back:
            // [1     2     3     4     5     6     7      8]
            // [1    1+2   2+3   3+4   4+5   5+6   6+7   7+8]
            // [1     3     5     7     9    11    13     15]

            // pass = 2, add each element with the element 2 position back:
            // [1     3     5     7     9    11     13     15 ]
            // [1     3    1+5   3+7   5+9   7+11  9+13  11+15]
            // [1     3     6    10    14    18    22      26 ]
            
            // pass = 3, add each element with the element 4 position back:
            // [1     3     6    10    14    18    22      26 ]
            // [1     3     6    10   1+14  3+18  6+22   10+26]
            // [1     3     6    10    15    21    28     36 ]
            
            // Sow we create a buffer with the current step value:
            id<MTLBuffer> step_buffer = [device newBufferWithBytes:&step 
                                        length:sizeof(uint32_t) 
                                        options:MTLResourceStorageModeShared];
            
            id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            
            // Set the pipeline state and the buffers:
            [encoder setComputePipelineState:computePipelineState];
            [encoder setBuffer:src_buffer   offset:0 atIndex:0];
            [encoder setBuffer:dst_buffer   offset:0 atIndex:1];
            [encoder setBuffer:count_buffer offset:0 atIndex:2];
            [encoder setBuffer:step_buffer  offset:0 atIndex:3];
            
            printf("Creating %d threads for pass %u with step %u\n", count, pass, step);
            // Will have 8 threads for each pass.
            MTLSize gridSize = MTLSizeMake(count, 1, 1);
            NSUInteger threadGroupSize = MIN(256, count);
            MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
            
            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            [encoder endEncoding];
            
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
            
            // Swap the src and input buffers so that we keep using the latest result as input.
            id<MTLBuffer> temp = src_buffer;
            // Set src_buffer  to the latest output buffer
            src_buffer = dst_buffer;
            dst_buffer = temp;
        }
        
        // Result is in src_buffer (we swapped after last pass)
        float *result = (float *)[src_buffer contents];
        
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
