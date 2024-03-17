#include <CL/sycl.hpp>
#include <iostream>
using namespace sycl;

int main() {
    queue q;
    std::cout << "Running on " << q.get_device().get_info<info::device::name>() << "\n";

    const int size = 1024;
    // malloc_shared is part of the USM (Unified Shared Memory) feature.
    int* a = malloc_shared<int>(size, q);
    int* b = malloc_shared<int>(size, q);
    int* c = malloc_shared<int>(size, q);

    for(int i = 0; i < size; i++) {
        a[i] = i;
        b[i] = i;
    }

    q.parallel_for(range<1>(size), [=](id<1> i) {
        c[i] = a[i] + b[i];
    }).wait();

    for(int i = 0; i < 10; i++) {
        std::cout << c[i] << " ";
    }
    std::cout << "\n";

    free(a, q);
    free(b, q);
    free(c, q);


    return 0;
}

