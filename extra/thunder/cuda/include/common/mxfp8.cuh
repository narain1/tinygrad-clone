#include "../common/common.cuh"
#include "../pyutils/pyutils.cuh"

#include <cstdint>

// MXFP8 (Microscaling FP8) - A compact floating point format
// Uses a shared exponent for blocks of values for better dynamic range

namespace mxfp8 {

// Block size for microscaling (32 FP8 values share 1 scale)
constexpr int BLOCK_SIZE = 32;

// MXFP8 block structure: 1 byte scale + 16 bytes data (32 x 4-bit values)
struct block_t {
  uint8_t scale;      // shared exponent for block
  uint8_t data[16];   // 32 4-bit values packed
};

// Decode single 4-bit MXFP8 value to float
__device__ __forceinline__ float decode(uint8_t code, uint8_t scale_exp) {
  float sign = (code & 0b1000) ? -1.0f : 1.0f;
  int exp = (code >> 1) & 0b11;
  int mant = code & 0b1;
  
  float val = (exp != 0) ? (1.0f + 0.5f * mant) * exp2f(float(exp - 1)) : 0.5f * mant;
  
  float scale;
  if (scale_exp >= 2) {
    scale = exp2f(float(scale_exp - 128));
  } else {
    scale = exp2f(float((scale_exp == 1) ? -127 : -128));
  }
  
  return sign * val * scale;
}

// Encode float to 4-bit MXFP8 value
__device__ __forceinline__ uint8_t encode(float val, float scale) {
  float scaled = val / scale;
  float abs_scaled = fabsf(scaled);
  
  if (abs_scaled > 7.5f) abs_scaled = 7.5f;
  
  uint8_t sign = (scaled < 0.0f) ? 0b1000 : 0;
  int exp = 0, mant = 0;
  
  if (abs_scaled >= 1.0f) {
    if (abs_scaled >= 4.0f) exp = 3;
    else if (abs_scaled >= 2.0f) exp = 2;
    else exp = 1;
  }
  
  if (exp > 0) {
    float base = exp2f(float(exp - 1));
    float residual = abs_scaled / base - 1.0f;
    mant = (residual >= 0.5f) ? 1 : 0;
  } else {
    mant = (abs_scaled >= 0.5f) ? 1 : 0;
  }
  
  return sign | (exp << 1) | mant;
}

// Compute scale for a block of values
__device__ __forceinline__ uint8_t compute_scale(const float* values, int count) {
  float max_val = 0.0f;
  for (int i = 0; i < count; i++) {
    max_val = fmaxf(max_val, fabsf(values[i]));
  }
  
  if (max_val == 0.0f) return 0;
  
  int exp = int(floorf(log2f(max_val))) + 128;
  return uint8_t(max(0, min(255, exp)));
}

} // namespace mxfp8
