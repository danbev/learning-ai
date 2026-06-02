#include "hailo/hailort.hpp"
#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>

using namespace hailort;

// --- Helpers to extract embeddings from HEF
bool find_hef_resource(const uint8_t * hef, size_t hef_size, const char * res_name, size_t & out_off, size_t & out_size) {
    const size_t nlen = std::strlen(res_name);
    auto rd_varint = [&](size_t & i, uint64_t & v) -> bool {
        v = 0; int sh = 0;
        while (i < hef_size) {
            uint8_t b = hef[i++];
            v |= (uint64_t)(b & 0x7f) << sh;
            if (!(b & 0x80)) return true;
            sh += 7;
            if (sh > 63) return false;
        }
        return false;
    };
    const uint8_t * pat = (const uint8_t *)res_name;
    size_t pos = 0;
    while (pos + nlen <= hef_size) {
        const uint8_t * hit = (const uint8_t *)std::memchr(hef + pos, pat[0], hef_size - nlen - pos + 1);
        if (!hit) break;
        pos = (size_t)(hit - hef);
        if (std::memcmp(hef + pos, pat, nlen) != 0) { ++pos; continue; }
        for (size_t c = (pos > 30 ? pos - 30 : 0); c + 1 < pos; ++c) {
            if (hef[c] != 0x08) continue;
            size_t i = c + 1; uint64_t off = 0, sz = 0;
            if (!rd_varint(i, off)) continue;
            if (i >= hef_size || hef[i] != 0x10) continue;
            i++;
            if (!rd_varint(i, sz)) continue;
            if (i >= hef_size || hef[i] != 0x1a) continue;
            i++;
            if (i >= hef_size || hef[i] != nlen) continue;
            i++;
            if (i + nlen > hef_size || std::memcmp(hef + i, pat, nlen) != 0) continue;
            if (off + sz > hef_size) continue;
            out_off = off; out_size = sz;
            return true;
        }
        ++pos;
    }
    return false;
}

std::vector<float> make_rope_theta(int head_dim, double base) {
    const int half = head_dim / 2;
    std::vector<float> theta(head_dim);
    for (int i = 0; i < half; ++i) {
        float e = (float)std::pow(base, -(double)i / (double)half);
        // CRITICAL: Bake the negative sign in for Hailo's signless RotateHalf
        theta[i]        = -e; 
        theta[i + half] = +e;
    }
    return theta;
}

void fill_attention_mask(uint8_t * buf, int seq_len, int n_tokens, int kv_cache_size, int n_heads, uint8_t mask_value) {
    const int cols = kv_cache_size;
    const int row_stride = cols * n_heads;
    std::memset(buf, 0, (size_t)seq_len * row_stride);

    auto set_head0 = [&](int row, int col0, int count) {
        if (count <= 0) return;
        std::memset(buf + (size_t)row * row_stride + col0, mask_value, (size_t)count);
    };

    const int padding_rows = seq_len - n_tokens;

    for (int r = 0; r < padding_rows; ++r) {
        set_head0(r, 0, cols);
    }

    for (int br = 0; br < n_tokens; ++br) {
        const int row = padding_rows + br;
        const int used_cols = cols - n_tokens;

        set_head0(row, used_cols, br + 1);
    }

    for (int r = 0; r < seq_len; ++r) {
        const uint8_t * head0 = buf + (size_t)r * row_stride;
        for (int h = 1; h < n_heads; ++h) {
            std::memcpy(buf + (size_t)r * row_stride + (size_t)h * cols, head0, cols);
        }
    }
}

int main() {
    const std::string hef_path = "hefs/Qwen2.5-1.5B-Instruct.hef";

    std::ifstream hef_file(hef_path, std::ios::binary | std::ios::ate);
    if (!hef_file) { std::cerr << "Failed to open HEF\n"; return 1; }
    std::streamsize hef_size = hef_file.tellg();
    hef_file.seekg(0, std::ios::beg);
    std::vector<uint8_t> hef_data(hef_size);
    if (!hef_file.read((char*)hef_data.data(), hef_size)) { std::cerr << "Failed to read HEF\n"; return 1; }

    hailort::MemoryView hef_mem(hef_data.data(), hef_data.size());

    auto hef_exp = Hef::create(hef_mem);
    if (!hef_exp) {
        std::cerr << "Failed to create Hef object: " << hef_exp.status() << "\n";
        return 1;
    }
    Hef hef_api = hef_exp.release();

    auto ev = hef_api.get_external_resources("embeddings.bin");
    if (!ev) {
        std::cerr << "embeddings.bin not found via official API\n";
        return 1;
    }
    MemoryView emb = ev.release();
    const uint16_t* hef_emb = reinterpret_cast<const uint16_t*>(emb.data());

    size_t official_offset = (uint8_t*)hef_emb - hef_data.data();
    std::cout << "Official API found embeddings.bin at offset: " << official_offset << "\n";

    auto vdevice = VDevice::create().expect("Failed to create VDevice");
    auto prefill_model = vdevice->create_infer_model(hef_mem, "base_model__prefill").expect("Failed to create infer model");
    prefill_model->set_enable_kv_cache(true);

    for (const auto& name : prefill_model->get_input_names()) {
        if (name.find("input_layer3") != std::string::npos || name.find("input_layer4") != std::string::npos ||
            name.find("input_layer5") != std::string::npos || name.find("input_layer6") != std::string::npos) {
            prefill_model->input(name).expect("").set_format_type(HAILO_FORMAT_TYPE_FLOAT32);
        }
    }
    for (const auto& name : prefill_model->get_output_names()) {
        prefill_model->output(name).expect("").set_format_type(HAILO_FORMAT_TYPE_FLOAT32);
    }

    auto cfg = prefill_model->configure().expect("Failed to configure");
    auto bindings = cfg.create_bindings().expect("Failed to create bindings");

    std::vector<int32_t> tokens = { 
        151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198 };
    int n_tokens = tokens.size();
    
    const int seq_len = 96, n_embd = 1536, n_attn_heads = 12, n_kv_heads = 2, head_dim = 128, kv_cache_size = 2048;
    const uint16_t emb_zp = 16384;
    const uint8_t mask_value = 128;
    const int row_offset = seq_len - n_tokens;

    std::vector<uint16_t> buf_layer1(seq_len * n_embd, emb_zp); // Embeddings
    std::vector<uint8_t>  buf_layer2(seq_len * kv_cache_size * n_attn_heads, 0); // Mask
    std::vector<float>    buf_layer3(seq_len * n_attn_heads * head_dim, 0.0f); // Q cos
    std::vector<float>    buf_layer4(seq_len * n_attn_heads * head_dim, 0.0f); // Q sin
    std::vector<float>    buf_layer5(seq_len * n_kv_heads * head_dim, 0.0f);   // K cos
    std::vector<float>    buf_layer6(seq_len * n_kv_heads * head_dim, 0.0f);   // K sin

    for (int i = 0; i < n_tokens; ++i) {
        const uint16_t* src = hef_emb + (tokens[i] * n_embd);
        uint16_t* dst = buf_layer1.data() + ((row_offset + i) * n_embd);
        std::memcpy(dst, src, n_embd * sizeof(uint16_t));
    }

    fill_attention_mask(buf_layer2.data(), seq_len, n_tokens, kv_cache_size, n_attn_heads, mask_value);

    auto theta = make_rope_theta(head_dim, 1e6);
    for (int i = 0; i < seq_len; ++i) {
        float pos = (i < row_offset) ? 0.0f : (float)(i - row_offset);
        std::vector<float> cos_t(head_dim), sin_t(head_dim);
        for (int l = 0; l < head_dim; ++l) {
            cos_t[l] = std::cos(theta[l] * pos);
            sin_t[l] = std::sin(theta[l] * pos);
        }
        for (int h = 0; h < n_attn_heads; ++h) {
            std::memcpy(buf_layer3.data() + (i * n_attn_heads + h) * head_dim, cos_t.data(), head_dim * sizeof(float));
            std::memcpy(buf_layer4.data() + (i * n_attn_heads + h) * head_dim, sin_t.data(), head_dim * sizeof(float));
        }
        for (int h = 0; h < n_kv_heads; ++h) {
            std::memcpy(buf_layer5.data() + (i * n_kv_heads + h) * head_dim, cos_t.data(), head_dim * sizeof(float));
            std::memcpy(buf_layer6.data() + (i * n_kv_heads + h) * head_dim, sin_t.data(), head_dim * sizeof(float));
        }
    }

    for (const auto& name : prefill_model->get_input_names()) {
        void* data = nullptr;
        if (name.find("input_layer1") != std::string::npos) data = buf_layer1.data();
        else if (name.find("input_layer2") != std::string::npos) data = buf_layer2.data();
        else if (name.find("input_layer3") != std::string::npos) data = buf_layer3.data();
        else if (name.find("input_layer4") != std::string::npos) data = buf_layer4.data();
        else if (name.find("input_layer5") != std::string::npos) data = buf_layer5.data();
        else if (name.find("input_layer6") != std::string::npos) data = buf_layer6.data();
        bindings.input(name).expect("").set_buffer(MemoryView(data, prefill_model->input(name).expect("").get_frame_size()));
    }

    std::map<std::string, std::vector<float>> out_bufs;
    for (const auto& name : prefill_model->get_output_names()) {
        size_t fs = prefill_model->output(name).expect("").get_frame_size();
        out_bufs[name].resize(fs / sizeof(float));
        bindings.output(name).expect("").set_buffer(MemoryView(out_bufs[name].data(), fs));
    }

    std::cout << "\nRunning Real Prefill Inference...\n";
    cfg.init_cache(0);
    cfg.update_cache_offset(n_tokens);
    auto run_status = cfg.run(bindings, std::chrono::milliseconds(10000));
    if (run_status != HAILO_SUCCESS) { std::cerr << "Inference failed: " << run_status << "\n"; return 1; }

    std::vector<float> logits;
    for (const auto& [name, buf] : out_bufs) {
        logits.insert(logits.end(), buf.begin(), buf.end());
    }

    using P = std::pair<float, int>;
    std::vector<P> top;
    for (int i = 0; i < logits.size(); ++i) {
        top.push_back({logits[i], i});
    }
    std::partial_sort(top.begin(), top.begin() + 5, top.end(), std::greater<P>());

    std::cout << "\n--- Top 5 Logits ---\n";
    for (int i = 0; i < 5; ++i) {
        std::cout << "  Token ID " << top[i].second << " -> " << top[i].first << "\n";
    }

    return 0;
}
