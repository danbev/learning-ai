#include <metal_stdlib>
using namespace metal;

kernel void cumsum_simd(
    device const float * input     [[buffer(0)]],
    device       float * output    [[buffer(1)]],
    constant     uint  & count     [[buffer(2)]],
                 uint    gid       [[thread_position_in_grid]],
                 uint    simd_lane [[thread_index_in_simdgroup]])
{
    float value = 0.0f;
    if (gid < count) {
        value = input[gid];
    }
    
    // So here we are calling this intrinsic function to perform the inclusive scan
    // and we are passing in one element of the input array. This looks like magic to
    // me as I did not understand how this would be possible without some for or parallel.
    // It turns out that the compiler turns this into several low-level operations
    // where the 32 lanes in the SIMD group cooperate. So we can think of this as similar to
    // the loop we have in cumsum.mm but more efficient has the threads can use registers
    // and shared memory to communicate.
    float result = simd_prefix_inclusive_sum(value);
    
    if (gid < count) {
        output[gid] = result;
    }
}
