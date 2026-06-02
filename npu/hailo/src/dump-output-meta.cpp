// Dump full metadata for every LM-head output stream: name, shape (h,w,features),
// format (order, type, flags), frame size, and quant infos. The within-shard lane
// order (NHWC / channel layout) is what we need to map 37984 lanes -> vocab order.
// Compare what we get with HAILO_FORMAT_TYPE_FLOAT32 (transform on) vs leaving the
// native order, to see whether HailoRT is supposed to de-order for us.
#include "hailo/hailort.hpp"
#include <iostream>
#include <fstream>
#include <vector>

using namespace hailort;

static const char* order_str(hailo_format_order_t o) {
    switch (o) {
        case HAILO_FORMAT_ORDER_NHWC: return "NHWC";
        case HAILO_FORMAT_ORDER_NHCW: return "NHCW";
        case HAILO_FORMAT_ORDER_FCR: return "FCR";
        case HAILO_FORMAT_ORDER_F8CR: return "F8CR";
        case HAILO_FORMAT_ORDER_NC: return "NC";
        case HAILO_FORMAT_ORDER_BAYER_RGB: return "BAYER_RGB";
        case HAILO_FORMAT_ORDER_HAILO_NMS: return "HAILO_NMS";
        default: return "OTHER";
    }
}

int main() {
    const std::string hef_path = "/home/danbev/work/learning-ai/npu/hailo/hefs/Qwen2.5-1.5B-Instruct.hef";
    std::ifstream hf(hef_path, std::ios::binary | std::ios::ate);
    std::streamsize sz = hf.tellg(); hf.seekg(0);
    std::vector<uint8_t> bytes(sz); hf.read((char*)bytes.data(), sz);

    auto vdevice = VDevice::create().expect("vdevice");
    auto model = vdevice->create_infer_model(MemoryView(bytes.data(), bytes.size()),
                                             "base_model__prefill").expect("model");
    model->set_enable_kv_cache(true);

    for (int pass = 0; pass < 2; ++pass) {
        const bool as_f32 = (pass == 1);
        std::cout << "\n===== outputs " << (as_f32 ? "WITH set_format_type(FLOAT32)"
                                                   : "NATIVE (no format override)") << " =====\n";
        for (const auto& name : model->get_output_names()) {
            auto os = model->output(name).expect("out");
            if (as_f32) os.set_format_type(HAILO_FORMAT_TYPE_FLOAT32);
            auto shape = os.shape();
            auto fmt = os.format();
            std::cout << name
                      << "  shape=" << shape.height << "x" << shape.width << "x" << shape.features
                      << "  order=" << order_str(fmt.order)
                      << "  type=" << (int)fmt.type
                      << "  flags=" << (int)fmt.flags
                      << "  frame=" << os.get_frame_size() << "\n";
            auto qi = os.get_quant_infos();
            for (size_t i = 0; i < qi.size() && i < 3; ++i) {
                std::cout << "    quant[" << i << "] scale=" << qi[i].qp_scale
                          << " zp=" << qi[i].qp_zp << "\n";
            }
            std::cout << "    (" << qi.size() << " quant infos)\n";
        }
    }
    return 0;
}
