#version 450

layout(local_size_x = 4, local_size_y = 4) in;

layout(set = 0, binding = 0) buffer InputOutputBuffer {
    mat4 matrixA;
    mat4 matrixB;
    mat4 result;
};

void main() {
    uint col = gl_GlobalInvocationID.x;
    uint row = gl_GlobalInvocationID.y;
    
    if(col < 4 && row < 4) {
        float sum = 0.0;
        for(int i = 0; i < 4; ++i) {
            sum += matrixA[row][i] * matrixB[i][col];
        }
        result[row][col] = sum;
    }
}
