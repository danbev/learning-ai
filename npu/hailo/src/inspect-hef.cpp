#include "hailo/hailort.hpp"
#include "hailo/hef.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cctype>

using namespace hailort;

static const char * dir_str(hailo_stream_direction_t d) {
    return d == HAILO_H2D_STREAM ? "IN " : "OUT";
}

static const char * order_str(hailo_format_order_t o) {
    switch (o) {
        case HAILO_FORMAT_ORDER_NHWC:        return "NHWC";
        case HAILO_FORMAT_ORDER_NHCW:        return "NHCW";
        case HAILO_FORMAT_ORDER_FCR:         return "FCR";
        case HAILO_FORMAT_ORDER_F8CR:        return "F8CR";
        case HAILO_FORMAT_ORDER_NC:          return "NC";
        case HAILO_FORMAT_ORDER_NHW:         return "NHW";
        case HAILO_FORMAT_ORDER_HAILO_NMS:   return "NMS";
        default:                             return "OTHER";
    }
}

static const char * type_str(hailo_format_type_t t) {
    switch (t) {
        case HAILO_FORMAT_TYPE_AUTO:    return "AUTO";
        case HAILO_FORMAT_TYPE_UINT8:   return "UINT8";
        case HAILO_FORMAT_TYPE_UINT16:  return "UINT16";
        case HAILO_FORMAT_TYPE_FLOAT32: return "FLOAT32";
        default:                        return "?";
    }
}

// Heuristic: does this resource look like printable text (so we can dump it)?
static bool looks_textual(const std::string & name) {
    return name.find(".json") != std::string::npos ||
           name.find("config") != std::string::npos ||
           name.find(".txt")  != std::string::npos;
}

int main(int argc, char ** argv) {
    const std::string hef_path = (argc > 1) ? argv[1] : "hefs/Whisper-Tiny.hef";

    std::ifstream f(hef_path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        std::cerr << "Failed to open " << hef_path << "\n"; return 1;
    }
    std::streamsize sz = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> bytes(sz);
    if (!f.read(reinterpret_cast<char *>(bytes.data()), sz)) {
        std::cerr << "read failed\n"; return 1;
    }

    auto hef_exp = Hef::create(MemoryView(bytes.data(), bytes.size()));
    if (!hef_exp) {
        std::cerr << "Hef::create failed: " << hef_exp.status() << "\n"; return 1;
    }
    Hef hef = hef_exp.release();

    std::cout << "=== HEF: " << hef_path << " (" << sz << " bytes) ===\n\n";

    std::cout << "--- external resources ---\n";
    auto res_names = hef.get_external_resource_names();
    if (res_names.empty()) {
        std::cout << "  (none)\n";
    }
    for (const auto & rn : res_names) {
        auto rv = hef.get_external_resources(rn);
        size_t bytes_sz = rv ? rv->size() : 0;
        std::cout << "  " << rn << "  (" << bytes_sz << " bytes)\n";
        if (rv && looks_textual(rn)) {
            std::cout << "    ---- begin " << rn << " ----\n";
            std::cout.write(reinterpret_cast<const char *>(rv->data()), rv->size());
            std::cout << "\n    ---- end " << rn << " ----\n";
        }
    }
    std::cout << "\n";

    std::cout << "--- network groups ---\n";
    for (const auto & ng : hef.get_network_groups_names()) {
        std::cout << "network group: " << ng << "\n";

        auto vinfos = hef.get_all_vstream_infos(ng);
        if (!vinfos) {
            std::cout << "  get_all_vstream_infos failed: " << vinfos.status() << "\n";
            continue;
        }
        for (const auto & vi : vinfos.value()) {
            // shape is valid for non-NMS orders (all we expect here)
            std::cout << "  " << dir_str(vi.direction) << " " << vi.name
                      << "  HxWxF=" << vi.shape.height << "x" << vi.shape.width << "x" << vi.shape.features
                      << "  " << order_str(vi.format.order) << "/" << type_str(vi.format.type)
                      << "  qp(scale=" << vi.quant_info.qp_scale << " zp=" << vi.quant_info.qp_zp << ")\n";
        }

        auto sorted_out = hef.get_sorted_output_names(ng);
        if (sorted_out) {
            std::cout << "  sorted outputs:";
            for (const auto & o : sorted_out.value()) std::cout << " " << o;
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    return 0;
}
