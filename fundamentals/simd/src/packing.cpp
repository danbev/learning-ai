#include <immintrin.h>
#include <iostream>
#include <cstdint>

int main() {
    // Simulate one q4_0 block: 16 bytes.
    uint8_t qs[16];

    uint8_t lower_vals[16];
    uint8_t upper_vals[16];
    std::fill(lower_vals, lower_vals + 16, 1);
    std::fill(upper_vals, upper_vals + 16, 2);

    for (int i = 0; i < 16; ++i) {
        qs[i] = (upper_vals[i] << 4) | (lower_vals[i] & 0x0F);
    }

    // Show original packed bytes
    std::cout << "packed qs bytes (low|high):\n";
    for (int i = 0; i < 16; ++i) {
        uint8_t b = qs[i];
        std::cout << "qs[" << i << "] = 0x" << std::hex << ((b >> 4) & 0x0F) << std::dec << "|" << (int)(b & 0x0F) << "\n";
    }

    // Load 16 bytes into a 128-bit SIMD register
    // First we load qs into a 128-bit register, qs contains 16 uint8_t values (16*8=128).
    __m128i v   = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs));
    // Next, this function will create a 128-bit SIMD register with the packed
    // values where every byte is set to 0x0F:
    // m0f = [0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F,
    //        0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F]
    // That is 16 bytes, each byte set to 0x0F (16*8=128 bits).
    // SIMD instruction operate on entire 128-bit registers so we need the mask
    // value for each position in the vector (lane).
    __m128i m0f = _mm_set1_epi8(0x0F);
    // Next _mm_and_si128 will perform a bitwise AND operation on all 16 bytes
    // simultaneously. This gets all the low nibbles (the lower 4 bits) of each byte.
    __m128i lo  = _mm_and_si128(v, m0f);
    // And then we shift each 16-bit element right by 4 bits, which moves the
    // upper nibbles into extractable positions. Then use the same mask m0f to
    // extract what were originally the high nibbles (upper 4 bits) from the
    // shifted data. This is done by _mm_srli_epi16(v, 4).
    __m128i hi  = _mm_and_si128(_mm_srli_epi16(v, 4), m0f);

    uint8_t out_lo[16] = {0};
    uint8_t out_hi[16] = {0};
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out_lo), lo);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out_hi), hi);

    std::cout << "\nlow nibbles (x[0..15]) via v & 0x0F:\n";
    for (int i = 0; i < 16; ++i) {
        std::cout << "out_lo[" << i << "] = " << (int)out_lo[i] << "\n";
    }

    std::cout << "\nhigh nibbles (x[16..31]) via (v>>4) & 0x0F:\n";
    for (int i = 0; i < 16; ++i) {
        std::cout << "out_hi[" << i << "] = " << (int)out_hi[i] << "\n";
    }

    return 0;
}

