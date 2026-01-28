#include "kittens.cuh"

using namespace kittens;

// MXFP8 (Microscaling FP8) format
// Each block has 32 FP8 values sharing a single 8-bit exponent (scale)
// This provides better dynamic range than standard FP8
constexpr int MXFP8_BLOCK_SIZE = 32;
constexpr int MXFP8_SCALE_BITS = 8;

// MXFP8 E4M3 format: 1 sign bit, 4 exponent bits, 3 mantissa bits
// Values range from -448 to 448
struct mxfp8_block {
  uint8_t scale;           // shared exponent for the block
  uint8_t data[16];        // 32 4-bit values packed into 16 bytes
};

// Decode a single mxfp8 value to float
__device__ __forceinline__ float decode_mxfp8_value(uint8_t code, uint8_t scale_exp) {
  // Extract sign, exponent, and mantissa from 4-bit code
  float sign = (code & 0b1000) ? -1.0f : 1.0f;
  int exp = (code >> 1) & 0b11;
  int mant = code & 0b1;
  
  // Compute value: (1.0 + 0.5 * mant) * 2^(exp-1) if exp != 0, else 0.5 * mant
  float val = (exp != 0) ? (1.0f + 0.5f * mant) * __expf((exp - 1) * 0.69314718f) : 0.5f * mant;
  
  // Apply shared scale
  float scale;
  if (scale_exp >= 2) {
    scale = __expf((scale_exp - 128) * 0.69314718f);
  } else {
    scale = __expf(((scale_exp == 1) ? -127 : -128) * 0.69314718f);
  }
  
  return sign * val * scale;
}

// Encode a float value to mxfp8 (4-bit code)
__device__ __forceinline__ uint8_t encode_mxfp8_value(float val, float scale) {
  float scaled = val / scale;
  float abs_scaled = fabsf(scaled);
  
  // Clamp to representable range
  if (abs_scaled > 7.5f) abs_scaled = 7.5f;
  
  uint8_t sign = (scaled < 0.0f) ? 0b1000 : 0;
  
  // Find best exponent and mantissa
  int exp = 0;
  int mant = 0;
  
  if (abs_scaled >= 1.0f) {
    if (abs_scaled >= 4.0f) exp = 3;
    else if (abs_scaled >= 2.0f) exp = 2;
    else exp = 1;
  }
  
  if (exp > 0) {
    float base = __expf((exp - 1) * 0.69314718f);
    float residual = abs_scaled / base - 1.0f;
    mant = (residual >= 0.5f) ? 1 : 0;
  } else {
    mant = (abs_scaled >= 0.5f) ? 1 : 0;
  }
  
  return sign | (exp << 1) | mant;
}

// Kernel to convert FP32/BF16 tensor to MXFP8 format
template<typename T>
__global__ void encode_mxfp8_kernel(mxfp8_block *out, const T *in, int n) {
  int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_blocks = (n + MXFP8_BLOCK_SIZE - 1) / MXFP8_BLOCK_SIZE;
  
  if (block_idx >= total_blocks) return;
  
  int start_idx = block_idx * MXFP8_BLOCK_SIZE;
  int end_idx = min(start_idx + MXFP8_BLOCK_SIZE, n);
  
  // Find max absolute value in this block
  float max_val = 0.0f;
  for (int i = start_idx; i < end_idx; i++) {
    float val = static_cast<float>(in[i]);
    max_val = fmaxf(max_val, fabsf(val));
  }
  
  // Compute scale exponent
  uint8_t scale_exp = 0;
  if (max_val > 0.0f) {
    float log2_max = log2f(max_val);
    int exp = static_cast<int>(floorf(log2_max)) + 128;
    scale_exp = static_cast<uint8_t>(max(0, min(255, exp)));
  }
  
  // Compute scale from exponent
  float scale;
  if (scale_exp >= 2) {
    scale = __expf((scale_exp - 128) * 0.69314718f);
  } else {
    scale = __expf(((scale_exp == 1) ? -127 : -128) * 0.69314718f);
  }
  
  // Encode values
  out[block_idx].scale = scale_exp;
  for (int i = 0; i < 16; i++) {
    int idx_low = start_idx + i;
    int idx_high = start_idx + i + 16;
    
    uint8_t low = (idx_low < end_idx) ? encode_mxfp8_value(static_cast<float>(in[idx_low]), scale) : 0;
    uint8_t high = (idx_high < end_idx) ? encode_mxfp8_value(static_cast<float>(in[idx_high]), scale) : 0;
    
    out[block_idx].data[i] = (low & 0xF) | ((high & 0xF) << 4);
  }
}

// Kernel to convert MXFP8 format back to FP32
__global__ void decode_mxfp8_kernel(float *out, const mxfp8_block *in, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  
  int block_idx = idx / MXFP8_BLOCK_SIZE;
  int in_block_idx = idx % MXFP8_BLOCK_SIZE;
  
  uint8_t scale_exp = in[block_idx].scale;
  
  // Extract the 4-bit code from packed data
  int byte_idx = in_block_idx / 2;
  uint8_t packed = in[block_idx].data[byte_idx];
  uint8_t code = (in_block_idx % 2 == 0) ? (packed & 0xF) : ((packed >> 4) & 0xF);
  
  out[idx] = decode_mxfp8_value(code, scale_exp);
}

// Matrix multiplication with MXFP8 inputs
template<int BLOCK_SIZE>
__global__ void mxfp8_matmul_kernel(float *C, const mxfp8_block *A_blocks, const mxfp8_block *B_blocks, 
                                    int M, int N, int K) {
  int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
  int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  
  if (row >= M || col >= N) return;
  
  float sum = 0.0f;
  
  for (int k = 0; k < K; k++) {
    // Decode A[row, k] and B[k, col] from MXFP8
    int a_block_idx = (row * K + k) / MXFP8_BLOCK_SIZE;
    int a_in_block = (row * K + k) % MXFP8_BLOCK_SIZE;
    
    int b_block_idx = (k * N + col) / MXFP8_BLOCK_SIZE;
    int b_in_block = (k * N + col) % MXFP8_BLOCK_SIZE;
    
    // Decode A value
    uint8_t a_scale = A_blocks[a_block_idx].scale;
    int a_byte = a_in_block / 2;
    uint8_t a_packed = A_blocks[a_block_idx].data[a_byte];
    uint8_t a_code = (a_in_block % 2 == 0) ? (a_packed & 0xF) : ((a_packed >> 4) & 0xF);
    float a_val = decode_mxfp8_value(a_code, a_scale);
    
    // Decode B value
    uint8_t b_scale = B_blocks[b_block_idx].scale;
    int b_byte = b_in_block / 2;
    uint8_t b_packed = B_blocks[b_block_idx].data[b_byte];
    uint8_t b_code = (b_in_block % 2 == 0) ? (b_packed & 0xF) : ((b_packed >> 4) & 0xF);
    float b_val = decode_mxfp8_value(b_code, b_scale);
    
    sum += a_val * b_val;
  }
  
  C[row * N + col] = sum;
}
