#include <cstdint>

extern "C" void simpleCompute(const float* input, float* output, uint64_t size) {
    for (uint64_t i = 0; i < size; ++i) {
        output[i] = input[i] * 2.0f;
    }
}
