#include <metal_stdlib>
using namespace metal;

kernel void cumsum_scan(
    device const float * input  [[buffer(0)]],
    device       float * output [[buffer(1)]],
    constant     uint&   count  [[buffer(2)]],
    constant     uint&   step   [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {

    if (gid >= count) return;
    
    if (step == 0) {
        output[gid] = input[gid];
    } else {
        if (gid >= step) {
            output[gid] = input[gid] + input[gid - step];
        } else {
            output[gid] = input[gid];
        }
    }
}
