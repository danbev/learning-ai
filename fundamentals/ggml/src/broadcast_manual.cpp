#include <stdio.h>

/*
void ggml_style_matrix_vector_mul(
    const float* matrix,  // 3x2 matrix (flattened)
    const float* vector,  // 2x1 vector
    float* result,        // 2x1 result vector
    int matrix_rows,      // 2
    int matrix_cols,      // 3
    int vector_size)      // 2
{
    for (int row = 0; row < matrix_rows; row++) {
        float sum = 0.0f;
	printf("row: %d\n", row);
        for (int col = 0; col < matrix_cols; col++) {
            // Use modulo for "broadcasting"
            int vector_index = col % vector_size;
	    printf("  col: %d, vector_index: %d, value: %f\n", col, vector_index, vector[vector_index]);
            sum += matrix[row * matrix_cols + col] * vector[vector_index];
        }
        result[row] = sum;
    }
}
*/

// GGML-style matrix multiplication with "extend with 1" broadcasting
void ggml_style_matrix_vector_mul(
    const float* matrix,  // 3x2 matrix (flattened)
    const float* vector,  // 2x1 vector
    float* result,        // 2x1 result vector
    int matrix_rows,      // 2
    int matrix_cols,      // 3
    int vector_size)      // 2
{
    for (int row = 0; row < matrix_rows; row++) {
        float sum = 0.0f;
	printf("row: %d\n", row);
        for (int col = 0; col < matrix_cols; col++) {
            // Use modulo for indexing, but extend with 1
	    //printf("  %d mod %d = %d\n", col, vector_size, (col % vector_size));
	    int vector_idx = col % vector_size;
            float vector_val = (col < vector_size) ? vector[vector_idx] : 1.0f;
	    printf("  col: %d, vector_index: %d, value: %f\n", col, col % vector_size, vector_val);
            sum += matrix[row * matrix_cols + col] * vector_val;
        }
        result[row] = sum;
    }
}

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

    //simplified_matrix_vector_mul(matrix, vector, result, 2, 3, 2);
    ggml_style_matrix_vector_mul(matrix, vector, result, 2, 3, 2);

    printf("Result: [%f, %f]\n", result[0], result[1]);
    return 0;
}
