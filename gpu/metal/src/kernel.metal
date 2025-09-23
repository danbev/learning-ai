#include <metal_stdlib>
using namespace metal;

kernel void simpleMultiply(const device float* input [[buffer(0)]],
                           device float* output [[buffer(1)]],
                           device int* debug_buffer [[buffer(2)]],
                           uint id [[thread_position_in_grid]]) {
    output[id] = input[id] * 2.0;

    debug_buffer[id] = input[id];
}

