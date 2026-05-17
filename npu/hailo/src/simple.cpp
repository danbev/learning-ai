#include <iostream>
#include "hailo/hailort.h"

int main() {
    hailo_version_t version;
    hailo_get_library_version(&version);

    std::cout << "HailoRT Version: "
              << (int)version.major << "."
              << (int)version.minor << "."
              << (int)version.revision << std::endl;

    return 0;
}
