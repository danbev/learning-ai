#include <iostream>
#include "openvino/openvino.hpp"

int main(int argc, char* argv[]) {
    ov::Version version = ov::get_openvino_version();
    std::cout << "OpenVINO version:" << version << std::endl;
    return 0;
}
