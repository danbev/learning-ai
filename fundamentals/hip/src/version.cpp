#include <hip/hip_runtime.h>
#include <iostream>

int main() {
    int driverVersion = 0;
    hipDriverGetVersion(&driverVersion);
    std::cout << "HIP Driver Version: " << driverVersion << std::endl;
    return 0;
}

