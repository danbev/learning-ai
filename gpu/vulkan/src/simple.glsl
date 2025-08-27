#version 450 // GLSL version 4.50 or higher required

// work group size of 4x4 so in total 16 threads (invocations).
// So each thread will process on element of the result matrix.
layout(local_size_x = 4, local_size_y = 4, local_size_z = 1) in;
// These local sizes are closely related to dispatch groups in simple.cpp:
// vkCmdDispatch(commandBuffer, 1, 1, 1);
//  Work Groups: 1×1×1 = 1 (work group)
//  Local Size:  4×4×1 = 16 threads per work group
//  Total:       1 × 16 = 16 threads total
//
//                  Column index
//               ↓      ↓      ↓      ↓
//  Thread IDs: (0,0), (1,0), (2,0), (3,0),
//              (0,1), (1,1), (2,1), (3,1),
//              (0,2), (1,2), (2,2), (3,2),
//              (0,3), (1,3), (2,3), (3,3)
//                 ↑      ↑      ↑      ↑	
//                      Row index
//

// The set is is similar to a group in webgpu I think.
layout(set = 0, binding = 0) buffer InputOutputBuffer {
    mat4 matrixA;
    mat4 matrixB;
    mat4 result;
};
// mat4 is a 4x4 matrix of floats, so 16 floats (4 bytes) total (64 bytes).
// The above three matrices will be packed into a single buffer.
// [matrixA: 64 bytes] [matrixB: 64 bytes] [result: 64 bytes]
// Total size: 192 bytes

void main() {
    uint col = gl_GlobalInvocationID.x; // Column index (0, 1, 2, 3)
    uint row = gl_GlobalInvocationID.y; // Row index (0, 1, 2, 3)
    // (col, row) will be unique for each thread.                                    
    
    if(col < 4 && row < 4) { // Bounds check (not strictly necessary here)
        float sum = 0.0;
        for(int i = 0; i < 4; ++i) {
            sum += matrixA[row][i] * matrixB[i][col];
        }
        result[row][col] = sum;
    }
}
