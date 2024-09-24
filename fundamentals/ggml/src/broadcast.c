#include <stdio.h>

// Simplified matrix-vector multiplication with broadcasting
void simplified_matrix_vector_mul(
    const float* matrix,  // 3x2 matrix (flattened)
    const float* vector,  // 2x1 vector
    float* result,        // 2x1 result vector
    int matrix_rows,      // 2
    int matrix_cols,      // 3
    int vector_size)      // 2
{
    for (int row = 0; row < matrix_rows; row++) {
        float sum = 0.0f;
        for (int col = 0; col < matrix_cols; col++) {
            // This is where the "broadcasting" happens
            float vector_val = (col < vector_size) ? vector[col] : 1.0f;
            sum += matrix[row * matrix_cols + col] * vector_val;
        }
        result[row] = sum;
    }
}

int main() {
    float matrix[6] = {1, 2, 3, 4, 5, 6};  // 3x2 matrix (ggml "notation"
    float vector[2] = {10, 20};            // 2x1 vector (ggml "notation")
    float result[2];                       // 2x1 result vector (ggml "notation")

    simplified_matrix_vector_mul(matrix, vector, result, 2, 3, 2);

    printf("Result: [%f, %f]\n", result[0], result[1]);
    return 0;
}
