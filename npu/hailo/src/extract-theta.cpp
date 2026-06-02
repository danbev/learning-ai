// Extract the REAL RoPE theta the HailoRT reference feeds, using the SAME API the
// reference uses (Hef::get_external_resources), instead of byte-hunting the file.
//   reference: llm_server.cpp:273  TRY(auto theta_view, hef.get_external_resources(THETA));
//              THETA = "rope_theta_data.bin"
// Prints all resource sizes, dumps theta floats, and compares to our [-e,+e] base-1e6
// fallback. Writes the raw theta bytes to /tmp/hef_theta.bin for reuse.
#include "hailo/hailort.hpp"
#include "hailo/hef.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstring>

using namespace hailort;

int main() {
    const std::string hef_path = "/home/danbev/work/learning-ai/npu/hailo/hefs/Qwen2.5-1.5B-Instruct.hef";
    const std::string THETA = "rope_theta_data.bin";
    const int HEAD_DIM = 128;

    // Create the Hef from an IN-MEMORY buffer, not the path. get_external_resources
    // returns NOT_SUPPORTED ("Reading from offset as memview is not supported when
    // reading from file") for a path-created Hef; it only works memory-backed.
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

    // Single-name form (matches reference). If your header only has the vector form,
    // switch to: auto m = hef.get_external_resources({THETA}); ... m->at(THETA)
    auto theta_view_exp = hef.get_external_resources(THETA);
    if (!theta_view_exp) {
        std::cerr << "get_external_resources(THETA) failed: " << theta_view_exp.status() << "\n";
        return 1;
    }
    MemoryView theta_view = theta_view_exp.release();
    const size_t nbytes = theta_view.size();
    const size_t nfloats = nbytes / sizeof(float);
    std::cout << "theta resource: " << nbytes << " bytes = " << nfloats << " float32\n";

    const float * t = reinterpret_cast<const float *>(theta_view.data());

    std::cout << "first 8 :";
    for (int i = 0; i < 8 && i < (int)nfloats; ++i) std::cout << " " << t[i];
    std::cout << "\n[60:68]:";
    for (int i = 60; i < 68 && i < (int)nfloats; ++i) std::cout << " " << t[i];
    std::cout << "\nlast 8  :";
    for (int i = (int)nfloats - 8; i < (int)nfloats; ++i) if (i >= 0) std::cout << " " << t[i];
    std::cout << "\n";

    // our fallback
    std::vector<float> ours(HEAD_DIM);
    for (int i = 0; i < HEAD_DIM / 2; ++i) {
        float e = (float)std::pow(1e6, -(double)i / (double)(HEAD_DIM / 2));
        ours[i] = -e; ours[i + HEAD_DIM / 2] = e;
    }
    if ((int)nfloats == HEAD_DIM) {
        float maxd = 0.f; int arg = 0;
        for (int i = 0; i < HEAD_DIM; ++i) {
            float d = std::fabs(t[i] - ours[i]);
            if (d > maxd) { maxd = d; arg = i; }
        }
        std::cout << "max|diff| vs our [-e,+e] fallback = " << maxd << " @ idx " << arg << "\n";
        std::cout << (maxd < 1e-3 ? "=> MATCHES our fallback.\n"
                                  : "=> DIFFERS -> use the HEF theta.\n");
    } else {
        std::cout << "!! theta length " << nfloats << " != head_dim " << HEAD_DIM
                  << " -> RoPE layout assumption wrong\n";
    }

    std::ofstream out("/tmp/hef_theta.bin", std::ios::binary);
    out.write(reinterpret_cast<const char *>(t), nbytes);
    std::cout << "wrote /tmp/hef_theta.bin (" << nbytes << " bytes)\n";
    return 0;
}
