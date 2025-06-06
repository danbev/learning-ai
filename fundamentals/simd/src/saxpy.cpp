#include <cstdio>
#include <omp.h>
#include <chrono>
#include <cstring>

int main() {
    printf("Single (float) Precision a * x + y (SAXPY)\n");
    // Notice that the multiplication operation is only depend on an element
    // of x and y and alpha (which is the same for all) so this could be performed
    // in parallel.

    int n = 10000000;
    float * x = new float[n];
    float * y_naive = new float[n];
    float * y_parallel = new float[n];
    float alpha = 2.5;

    for (int i = 0; i < n; i++) {
        x[i] = i + 1.0f;
        y_naive[i] = n - i;
        y_parallel[i] = n - i;  // Same initial values
    }

    printf("Naive SAXPY...\n");
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++) {
        y_naive[i] = alpha * x[i] + y_naive[i];
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto naive_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("Naive SAXPY done in %ld ms\n", naive_time.count());

    printf("Parallel SAXPY...\n");
    start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        y_parallel[i] = alpha * x[i] + y_parallel[i];
    }
    end = std::chrono::high_resolution_clock::now();
    auto parallel_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("Parallel SAXPY done in %ld ms\n", parallel_time.count());

    printf("Speedup: %.2fx\n", (double)naive_time.count() / parallel_time.count());

    delete[] x;
    delete[] y_naive;
    delete[] y_parallel;

    return 0;
}
