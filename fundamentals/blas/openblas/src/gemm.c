#include <stdio.h>
#include <cblas.h>

int main() {
    const int M = 2; // Number of rows of matrix A and C
    const int N = 2; // Number of columns of matrix B and C
    const int K = 3; // Number of columns of matrix A and rows of matrix B

    float alpha = 1.0f;
    float beta = 0.0f;

    // Declare matrices A, B, and C
    float A[M*K];
    float B[K*N];
    float C[M*N];

    // Initialize matrices A and B with given values
    A[0] = 1; A[1] = 2; A[2] = 3;
    A[3] = 4; A[4] = 5; A[5] = 6;

    B[0] = 1; B[1] = 2;
    B[2] = 3; B[3] = 4;
    B[4] = 5; B[5] = 6;

    // Initialize matrix C with initial non-zero values
    C[0] = 0; C[1] = 0;
    C[2] = 0; C[3] = 0;

    // Perform matrix multiplication C = alpha * A * B + beta * C
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, C, N);

    // Print the resulting matrix C
    printf("Resulting matrix C:\n");
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%f ", C[i*N + j]);
        }
        printf("\n");
    }

    return 0;
}

