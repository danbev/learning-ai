#include <cblas.h>
#include <stdio.h>

int main() {
    int n = 5;
    float x[] = {1, 2, 3, 4, 5};
    float y[] = {5, 4, 3, 2, 1};
    float alpha = 1.0;

    // y = alpha * x + y
    cblas_saxpy(n, alpha, x, 1, y, 1);

    for (int i = 0; i < n; i++) {
        printf("%f ", y[i]);
    }
    printf("\n");

    return 0;
}

