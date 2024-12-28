#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <climits>
#include <stdexcept>
#include <vector>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <limits>
#include <array>

#if defined(__clang__)
    #pragma clang diagnostic push
    // Disable *all* warnings from stb.
    // Alternatively, you can selectively ignore some, e.g. "-Wunknown-pragmas"
    #pragma clang diagnostic ignored "-Weverything"
#elif defined(__GNUC__) || defined(__GNUG__)
    #pragma GCC diagnostic push
    // Similarly, you can disable *all* or specific warnings:
    // e.g. #pragma GCC diagnostic ignored "-Wall"
    #pragma GCC diagnostic ignored "-Wswitch-default"
#elif defined(_MSC_VER)
    #pragma warning(push)
    // Disable specific MSVC warnings:
    #pragma warning(disable : 4244 4996)
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

#if defined(__clang__)
    #pragma clang diagnostic pop
#elif defined(__GNUC__) || defined(__GNUG__)
    #pragma GCC diagnostic pop
#elif defined(_MSC_VER)
    #pragma warning(pop)
#endif

struct llama_img {
    unsigned char* data = nullptr; // or float*
    int width;
    int height;
    int channels;
    int aspect_ratio;
};

static llama_img* llama_img_init(int w, int h, int c = 3) {
    llama_img* result = new llama_img();
    result->width = w;
    result->height = h;
    result->channels = c;
    return result;
}

static void llama_img_free(llama_img* img) {
    if (!img) return;
    if (img->data) {
        free(img->data);
        img->data = nullptr;
    }
    delete img;
    img = nullptr;
}

// -----------------------------------------------------------
// 1) EXACT LIST OF SUPPORTED ASPECT RATIOS
//    as in Python's get_all_supported_aspect_ratios
// -----------------------------------------------------------
static std::vector<std::pair<int, int>> get_all_supported_aspect_ratios(int max_image_tiles) {
    std::vector<std::pair<int, int>> aspect_ratios;
    for (int width = 1; width <= max_image_tiles; ++width) {
        for (int height = 1; height <= max_image_tiles; ++height) {
            if (width * height <= max_image_tiles) {
                aspect_ratios.push_back({width, height});
            }
        }
    }
    return aspect_ratios;
}

// -----------------------------------------------------------
// 2) GET THE BEST (canvas_width, canvas_height)
//    as in Python's get_optimal_tiled_canvas
// -----------------------------------------------------------
static std::pair<int,int> get_optimal_tiled_canvas(
    int image_height,
    int image_width,
    int max_image_tiles,
    int tile_size
) {
    // The logic from the Python code:
    //  - Make a list of possible tile arrangements => possible tile_count_x, tile_count_y
    //  - For each arrangement => canvas_w = tile_count_x*tile_size, canvas_h = tile_count_y*tile_size
    //  - Compute scale_w = canvas_w / orig_width, scale_h = canvas_h / orig_height
    //  - scale = min(scale_w, scale_h) for no distortion
    //  - Choose the smallest scale >= 1 if any exist, else largest < 1
    //  - If multiple have the same scale, choose the one with the smallest area

    auto possible_tile_arrangements = get_all_supported_aspect_ratios(max_image_tiles);

    // For each arrangement, store (canvas_width, canvas_height)
    std::vector<std::pair<int, int>> possible_canvas_sizes;
    possible_canvas_sizes.reserve(possible_tile_arrangements.size());

    std::vector<float> scales;
    scales.reserve(possible_tile_arrangements.size());

    // Populate possible canvases and scales
    for (auto& tile_arr : possible_tile_arrangements) {
        int tiles_w = tile_arr.first;
        int tiles_h = tile_arr.second;
        int cw = tiles_w * tile_size;
        int ch = tiles_h * tile_size;

        float scale_w = float(cw) / float(image_width);
        float scale_h = float(ch) / float(image_height);
        float scale   = (scale_w < scale_h ? scale_w : scale_h); // min(scale_w, scale_h)

        possible_canvas_sizes.push_back({ch, cw});  // watch out for (height, width) vs (width, height)!
        // In the Python code, get_optimal_tiled_canvas returns (height, width).
        // So let's match that order: (canvas_height, canvas_width)
        scales.push_back(scale);
    }

    // Partition the “scales” into those >= 1 (upscale possible) and those < 1 (downscale)
    std::vector<float> upscaling_options;
    upscaling_options.reserve(scales.size());
    for (auto sc : scales) {
        if (sc >= 1.0f) {
            upscaling_options.push_back(sc);
        }
    }

    float selected_scale = 1.0f;
    if (!upscaling_options.empty()) {
        // choose the smallest > 1
        selected_scale = *std::min_element(upscaling_options.begin(), upscaling_options.end());
    } else {
        // choose the largest < 1
        selected_scale = *std::max_element(scales.begin(), scales.end());
    }

    // now find all canvases whose scale == selected_scale
    // if multiple => pick one with minimal area
    std::pair<int,int> optimal_canvas{0, 0};
    int min_area = INT_MAX;

    for (size_t i = 0; i < scales.size(); ++i) {
        if (std::fabs(scales[i] - selected_scale) < 1e-7) {
            auto& csize = possible_canvas_sizes[i];
            int area = csize.first * csize.second; // area = height*width
            if (area < min_area) {
                min_area = area;
                optimal_canvas = csize;
            }
        }
    }

    return optimal_canvas;
}

// -----------------------------------------------------------
// 3) GET NEW SIZE to fit into the chosen canvas
//    as in Python's get_image_size_fit_to_canvas
// -----------------------------------------------------------
static std::pair<int,int> get_image_size_fit_to_canvas(
    int image_height,
    int image_width,
    int canvas_height,
    int canvas_width,
    int tile_size
) {
    // Python logic:
    //   target_width  = clip(original_w, tile_size, canvas_width)
    //   target_height = clip(original_h, tile_size, canvas_height)
    //   scale_w = target_width  / original_w
    //   scale_h = target_height / original_h
    //   if (scale_w < scale_h) { new_w=target_width,  new_h=floor(original_h*scale_w) }
    //   else                   { new_h=target_height, new_w=floor(original_w*scale_h) }

    auto target_w = std::max(tile_size, std::min(image_width,  canvas_width));
    auto target_h = std::max(tile_size, std::min(image_height, canvas_height));

    float scale_w = float(target_w) / float(image_width);
    float scale_h = float(target_h) / float(image_height);

    int new_w = 0;
    int new_h = 0;
    if (scale_w < scale_h) {
        new_w = target_w;
        new_h = std::min(int(std::floor(image_height * scale_w)), target_h);
    } else {
        new_h = target_h;
        new_w = std::min(int(std::floor(image_width * scale_h)), target_w);
    }

    return std::make_pair(new_h, new_w); // (height, width) to match Python's return
}

// -----------------------------------------------------------
// 4) PAD the newly resized image to the full canvas
//    i.e. if new_w < canvas_width, pad the difference with zeros
// -----------------------------------------------------------
static void pad_image(
    const unsigned char* src,
    int src_h,
    int src_w,
    int channels,
    int canvas_h,
    int canvas_w,
    unsigned char* dst
) {
    // Fill the entire dst with zeros
    memset(dst, 0, canvas_h * canvas_w * channels);

    // Copy the src image into the top-left corner of dst
    for (int r = 0; r < src_h; r++) {
        memcpy(dst + r * (canvas_w * channels),
               src + r * (src_w * channels),
               size_t(src_w * channels));
    }
}

// -----------------------------------------------------------
// 5) SPLIT TO TILES
//    as in Python's split_to_tiles
//    we assume the image is in CHW or we’ll do it in “channels_last” style
//    but easiest might be channels_last = HxWxC when subdividing
// -----------------------------------------------------------
static std::vector<std::vector<unsigned char>> split_into_tiles(
    const unsigned char* src_hwc,
    int channels,
    int full_h,
    int full_w,
    int num_tiles_h, // e.g. 2
    int num_tiles_w, // e.g. 3
    int tile_h,      // each tile’s height in pixels
    int tile_w       // each tile’s width in pixels
) {
    // final shape = (num_tiles_h * num_tiles_w, tile_h, tile_w, channels)
    // create a vector-of-vectors; each sub-vector is one tile
    std::vector<std::vector<unsigned char>> tiles;
    tiles.reserve(num_tiles_h * num_tiles_w);

    for (int ty = 0; ty < num_tiles_h; ty++) {
        for (int tx = 0; tx < num_tiles_w; tx++) {
            std::vector<unsigned char> tile_data(tile_h * tile_w * channels);

            for (int row = 0; row < tile_h; row++) {
                int src_y = ty * tile_h + row;
                for (int col = 0; col < tile_w; col++) {
                    int src_x = tx * tile_w + col;

                    // if we wanted out-of-bounds checks, we’d do so here
                    int src_index = (src_y * full_w + src_x) * channels;
                    int dst_index = (row * tile_w + col) * channels;

                    for (int c = 0; c < channels; c++) {
                        tile_data[dst_index + c] = src_hwc[src_index + c];
                    }
                }
            }
            tiles.push_back(std::move(tile_data));
        }
    }
    return tiles;
}

// -----------------------------------------------------------
// 6) Normalization
// -----------------------------------------------------------
static void normalize_tile_in_place(
    std::vector<float>& tile_chw,
    int tile_size_h,
    int tile_size_w,
    const float mean[3],
    const float stdv[3]
) {
    // tile_chw is shape [3, tile_size_h, tile_size_w]
    // i.e. index = c*(tile_size_h * tile_size_w) + y*tile_size_w + x
    // do the standard: val = (val - mean[c]) / stdv[c]
    // in Python code, the rescale factor = 1/255 was done first, so do that here as well
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < tile_size_h; y++) {
            for (int x = 0; x < tile_size_w; x++) {
                int idx = c*(tile_size_h*tile_size_w) + y*tile_size_w + x;
                // scale to [0..1]
                float val = tile_chw[idx] / 255.0f;
                // now subtract mean & divide by std
                val = (val - mean[c]) / stdv[c];
                tile_chw[idx] = val;
            }
        }
    }
}

// -----------------------------------------------------------
// 7) Convert tile from HWC -> CHW (float) and apply normalization
//    if you want everything in float CHW for the model
// -----------------------------------------------------------
static std::vector<float> convert_tile_to_float_chw_and_normalize(
    const std::vector<unsigned char>& tile_hwc,
    int tile_size_h,
    int tile_size_w,
    const float mean[3],
    const float stdv[3]
) {
    // allocate float CHW
    std::vector<float> out_chw(3 * tile_size_h * tile_size_w);

    // reorder channels
    // tile_hwc => row-major [ H * W * 3 ]
    // tile_chw => channel-major [ 3 * (H * W) ]
    for (int y = 0; y < tile_size_h; y++) {
        for (int x = 0; x < tile_size_w; x++) {
            for (int c = 0; c < 3; c++) {
                int hwc_idx = (y * tile_size_w + x)*3 + c;
                int chw_idx = c*(tile_size_h*tile_size_w) + (y*tile_size_w) + x;
                out_chw[chw_idx] = float(tile_hwc[hwc_idx]);
            }
        }
    }
    // apply mean/std in-place
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < tile_size_h; y++) {
            for (int x = 0; x < tile_size_w; x++) {
                int idx = c*(tile_size_h*tile_size_w) + y*tile_size_w + x;
                float val = out_chw[idx] / 255.0f;
                val = (val - mean[c]) / stdv[c];
                out_chw[idx] = val;
            }
        }
    }
    return out_chw;
}

// -----------------------------------------------------------
// 8) The final function that replicates the Python pipeline
// -----------------------------------------------------------
llama_img* mllama_load_image_from_file(const char* fname, int max_image_tiles, int tile_size) {
    // 1) Load with stbi, forcing 3 channels (like python convert_to_rgb)
    int orig_w, orig_h, orig_c;
    unsigned char* stbi_data = stbi_load(fname, &orig_w, &orig_h, &orig_c, 3);
    if (!stbi_data) {
        throw std::runtime_error("Could not open or decode: " + std::string(fname));
    }
    printf("Loaded image: w=%d h=%d c=%d\n", orig_w, orig_h, orig_c);

    // 2) Use get_optimal_tiled_canvas() to find best (canvas_h, canvas_w)
    auto [canvas_h, canvas_w] = get_optimal_tiled_canvas(orig_h, orig_w, max_image_tiles, tile_size);
    printf("Chosen canvas: %d x %d\n", canvas_h, canvas_w);

    // 3) Compute the new size that fits within that canvas
    auto [new_h, new_w] = get_image_size_fit_to_canvas(
        orig_h, orig_w,  // original
        canvas_h, canvas_w,
        tile_size
    );
    printf("Resized (no pad) to: %d x %d\n", new_h, new_w);

    // 4) Resize the original to (new_h, new_w)
    //    we’ll keep it in HWC layout for easier tiling
    std::vector<unsigned char> resized_hwc(new_h * new_w * 3);
    // stbir wants you to pass data in W x H for row stride
    // but we have stbi_data as if it were W=orig_w, H=orig_h, 3 channels
    // so input stride = (orig_w * 3)
    // output stride = (new_w * 3)
    stbir_resize_uint8_linear(
        stbi_data, orig_w, orig_h, orig_w * 3,
        resized_hwc.data(), new_w, new_h, new_w * 3,
	STBIR_RGB
    );
    stbi_image_free(stbi_data);
    stbi_data = nullptr;

    // 5) Now pad the resized image to match (canvas_h, canvas_w)
    //    in Python: pad with zeros if new_w < canvas_w or new_h < canvas_h
    std::vector<unsigned char> padded_hwc(canvas_h * canvas_w * 3);
    pad_image(resized_hwc.data(), new_h, new_w, 3, canvas_h, canvas_w, padded_hwc.data());
    // from now on we have a final HWC image of shape = [canvas_h, canvas_w, 3]

    // 6) Figure out how many tiles in each dimension:
    //    num_tiles_height = canvas_h / tile_size
    //    num_tiles_width  = canvas_w / tile_size
    int num_tiles_h = canvas_h / tile_size;
    int num_tiles_w = canvas_w / tile_size;
    printf("Splitting to tiles => %d x %d\n", num_tiles_h, num_tiles_w);

    // 7) Subdivide
    auto tiles_hwc = split_into_tiles(
        padded_hwc.data(), 3, canvas_h, canvas_w,
        num_tiles_h, num_tiles_w, tile_size, tile_size
    );

    // 8) Convert each tile to float CHW + normalization
    //    The Python code used these default means/stdevs:
    //    IMAGENET_STANDARD_MEAN = [0.485, 0.456, 0.406] or maybe [0.48145..]
    //    IMAGENET_STANDARD_STD  = [0.229, 0.224, 0.225] or your own
    //    In your snippet you had:
    //      mean = {0.48145466f, 0.4578275f,  0.40821073f};
    //      std  = {0.26862954f, 0.26130258f, 0.27577711f};
    const float mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    const float stdv[3] = {0.26862954f, 0.26130258f, 0.27577711f};

    std::vector<float> final_float_data;
    final_float_data.reserve(tiles_hwc.size() * 3 * tile_size * tile_size);

    for (auto& tile : tiles_hwc) {
        // HWC -> CHW, then normalize
        auto float_chw = convert_tile_to_float_chw_and_normalize(tile, tile_size, tile_size, mean, stdv);
        final_float_data.insert(final_float_data.end(), float_chw.begin(), float_chw.end());
    }

    // 9) Create a llama_img to hold the final result
    //    We'll treat it as if each tile is stacked “vertically” or “batch dimension,”
    //    but that part is up to how your model code expects it.
    //    You might store them in an NxCxHxW buffer, etc.
    int n_tiles = num_tiles_h * num_tiles_w;
    llama_img* result = llama_img_init(tile_size, tile_size * n_tiles, 3);
    // Or store the real shape differently. We'll keep the Python-ish approach:
    //   pixel_values => shape [batch_size, max_num_images, max_image_tiles, channels, tile_height, tile_width]
    // For a simple example, we just store everything in a single 2D plane:
    //   total W = tile_size
    //   total H = tile_size*n_tiles
    //   data is float, so allocate that
    size_t float_bytes = sizeof(float) * final_float_data.size();
    float* float_buf = (float*)malloc(float_bytes);
    memcpy(float_buf, final_float_data.data(), float_bytes);

    result->data = reinterpret_cast<unsigned char*>(float_buf);
    // Mark aspect ratio ID if you like
    // The Python code uses convert_aspect_ratios_to_ids, etc.
    // Minimally, you might do:
    //   result->aspect_ratio = someIndex;
    // or store (num_tiles_h << 16) + num_tiles_w, etc.

    // Example: find your aspect ratio ID
    // in Python, for j, (num_tiles_h, num_tiles_w) ...
    // we’d do something akin to:
    auto supported = get_all_supported_aspect_ratios(max_image_tiles);
    printf("Supported aspect ratios size: %zu\n", supported.size());
    // we can find the index
    int ar_index = 0;
    for (size_t i = 0; i < supported.size(); i++) {
        if (supported[i].first == num_tiles_w && supported[i].second == num_tiles_h) {
            ar_index = (int)i + 1;
            break;
        }
    }
    result->aspect_ratio = ar_index;

    return result;
}

int main() {
    try {
        int max_tiles  = 4;
        int tile_size  = 560;
	auto supported_ratios = get_all_supported_aspect_ratios(max_tiles);
	for (auto& ar : supported_ratios) {
	    printf("Aspect ratio: %d x %d\n", ar.first, ar.second);
	}

        llama_img* out = mllama_load_image_from_file("apple.jpg", max_tiles, tile_size);

	// Print the first 10 values of the first tile:
	for (int i = 0; i < 10; i++) {
	    printf("%f\n", ((float*)out->data)[i]);
	}

	int aspect_ratio_id = out->aspect_ratio;
	printf("Aspect ratio ID: %d\n", aspect_ratio_id);

        // out->data is float* in CHW for each tile stacked vertically
        // out->width = 224
        // out->height = 224 * (# of tiles)
        // out->aspect_ratio = ???

        // use out->data for inference, etc.
        llama_img_free(out);
    } catch (std::exception &e) {
        fprintf(stderr, "Exception: %s\n", e.what());
        return 1;
    }
    return 0;
}
