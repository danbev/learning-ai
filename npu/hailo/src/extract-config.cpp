// Dump the embedded hailo-config.json from the HEF, using the same memory-backed
// Hef::get_external_resources path that works for theta. This file states the real
// LM-head output order and per-shard sizes (and pre/post-process params) so we stop
// guessing the shard->vocab assembly.
#include "hailo/hailort.hpp"
#include "hailo/hef.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>

using namespace hailort;

int main() {
    //const std::string hef_path = "/home/danbev/work/learning-ai/npu/hailo/hefs/Llama3.2-1B-Instruct.hef";
    const std::string hef_path = "/home/danbev/work/learning-ai/npu/hailo/hefs/Qwen2.5-1.5B-Instruct.hef";
    const std::string CONFIG = "hailo-config.json";

    std::ifstream hef_file(hef_path, std::ios::binary | std::ios::ate);
    if (!hef_file.is_open()) { std::cerr << "Failed to open HEF\n"; return 1; }
    std::streamsize hef_size = hef_file.tellg();
    hef_file.seekg(0, std::ios::beg);
    std::vector<uint8_t> hef_bytes(hef_size);
    if (!hef_file.read(reinterpret_cast<char *>(hef_bytes.data()), hef_size)) {
        std::cerr << "Failed to read HEF\n"; return 1;
    }

    auto hef_exp = Hef::create(MemoryView(hef_bytes.data(), hef_bytes.size()));
    if (!hef_exp) { std::cerr << "Hef::create failed: " << hef_exp.status() << "\n"; return 1; }
    auto hef = hef_exp.release();

    auto view_exp = hef.get_external_resources(CONFIG);
    if (!view_exp) {
        std::cerr << "get_external_resources(" << CONFIG << ") failed: " << view_exp.status() << "\n";
        return 1;
    }
    MemoryView v = view_exp.release();
    std::cout << "=== " << CONFIG << " (" << v.size() << " bytes) ===\n";
    std::cout.write(reinterpret_cast<const char *>(v.data()), v.size());
    std::cout << "\n";

    std::ofstream out("/tmp/hailo-config.json", std::ios::binary);
    out.write(reinterpret_cast<const char *>(v.data()), v.size());
    std::cerr << "wrote /tmp/hailo-config.json\n";
    return 0;
}
