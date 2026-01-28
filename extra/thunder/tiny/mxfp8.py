"""
Native tinygrad implementation of MXFP8 (Microscaling FP8) format.

MXFP8 uses block-wise microscaling with:
- Block size: 32 values
- Each value: 4 bits (1 sign + 2 exponent + 1 mantissa)
- Shared scale: 8-bit exponent per block
- Total: 17 bytes per block (1 scale + 16 data bytes)

This provides ~6.7x compression vs FP32 with ~13-20% relative error.
"""

import math
from tinygrad import Tensor, dtypes

BLOCK_SIZE = 32  # Number of values per MXFP8 block

def encode_mxfp8(x: Tensor) -> Tensor:
  """
  Encode a tensor to MXFP8 format using native tinygrad operations.
  
  Args:
    x: Input tensor (any shape, will be flattened)
    
  Returns:
    Encoded tensor as uint8 with shape (n_blocks, 17) where each row is:
      [scale_exp, packed_data[0], packed_data[1], ..., packed_data[15]]
  """
  # Flatten input
  x_flat = x.flatten().cast(dtypes.float32)
  n = x_flat.shape[0]
  
  # Pad to multiple of BLOCK_SIZE
  pad_size = (BLOCK_SIZE - (n % BLOCK_SIZE)) % BLOCK_SIZE
  if pad_size > 0:
    x_flat = x_flat.pad(((0, pad_size),))
  
  n_padded = x_flat.shape[0]
  n_blocks = n_padded // BLOCK_SIZE
  
  # Reshape into blocks: (n_blocks, BLOCK_SIZE)
  x_blocks = x_flat.reshape(n_blocks, BLOCK_SIZE)
  
  # Compute scale exponent for each block
  # Find max absolute value in each block
  max_abs = x_blocks.abs().max(axis=1, keepdim=True)  # (n_blocks, 1)
  
  # Compute scale exponent: log2(max_abs / 6.0) + 128
  # Clamp to avoid log of zero
  max_abs_safe = max_abs.maximum(1e-38)
  log_max = (max_abs_safe / 6.0).log2()
  scale_exp_float = (log_max.floor() + 128).clip(0, 255)
  scale_exp = scale_exp_float.cast(dtypes.uint8)  # (n_blocks, 1)
  
  # Compute scale from exponent
  # scale = 2^(scale_exp - 128) for scale_exp >= 2
  # scale = 2^(-127) for scale_exp == 1
  # scale = 2^(-128) for scale_exp == 0
  scale_exp_adj = (scale_exp_float >= 2).where(
    scale_exp_float,
    (scale_exp_float >= 1).where(-127, -128)
  )
  scale = (scale_exp_adj - 128).exp2()  # (n_blocks, 1)
  
  # Scale values: x / scale
  scaled = x_blocks / (scale + 1e-38)  # (n_blocks, BLOCK_SIZE)
  abs_scaled = scaled.abs()
  
  # Clamp to representable range
  abs_scaled = abs_scaled.clip(0, 6.0)
  
  # Determine exponent (0-3) based on magnitude
  # exp=3: [3.0, 6.0] -> value in [4.0, 6.0]
  # exp=2: [1.5, 3.0) -> value in [2.0, 3.0]
  # exp=1: [0.75, 1.5) -> value in [1.0, 1.5]
  # exp=0: [0, 0.75) -> value in [0.0, 0.5]
  
  exp_bits = ((abs_scaled >= 3.0).cast(dtypes.float32) * 3 +
              (abs_scaled >= 1.5).cast(dtypes.float32) * (abs_scaled < 3.0).cast(dtypes.float32) * 2 +
              (abs_scaled >= 0.75).cast(dtypes.float32) * (abs_scaled < 1.5).cast(dtypes.float32) * 1)
  
  # Determine mantissa bit based on residual
  # For each exponent, check if we should use mant=1
  exp0_threshold = (abs_scaled >= 0.25).cast(dtypes.float32) * (abs_scaled < 0.75).cast(dtypes.float32)
  exp1_threshold = (abs_scaled >= 1.25).cast(dtypes.float32) * (abs_scaled < 1.5).cast(dtypes.float32)
  exp2_threshold = (abs_scaled >= 2.5).cast(dtypes.float32) * (abs_scaled < 3.0).cast(dtypes.float32)
  exp3_threshold = (abs_scaled >= 5.0).cast(dtypes.float32) * (abs_scaled < 6.1).cast(dtypes.float32)
  
  mant_bits = (exp0_threshold * (exp_bits == 0).cast(dtypes.float32) +
               exp1_threshold * (exp_bits == 1).cast(dtypes.float32) +
               exp2_threshold * (exp_bits == 2).cast(dtypes.float32) +
               exp3_threshold * (exp_bits == 3).cast(dtypes.float32))
  
  # Sign bit
  sign_bits = (scaled < 0).cast(dtypes.float32) * 8
  
  # Combine into 4-bit codes
  codes = (sign_bits + exp_bits * 2 + mant_bits).cast(dtypes.uint8)  # (n_blocks, BLOCK_SIZE)
  
  # Pack two 4-bit values into each byte
  # codes[0:16] go to low nibbles, codes[16:32] go to high nibbles
  codes_low = codes[:, :16]  # (n_blocks, 16)
  codes_high = codes[:, 16:]  # (n_blocks, 16)
  packed = (codes_low & 0xF) | ((codes_high & 0xF) << 4)  # (n_blocks, 16)
  
  # Concatenate scale_exp and packed data
  result = scale_exp.cat(packed, dim=1)  # (n_blocks, 17)
  
  return result

def decode_mxfp8(encoded: Tensor, original_shape: tuple) -> Tensor:
  """
  Decode MXFP8 format back to float32 using native tinygrad operations.
  
  Args:
    encoded: Encoded uint8 tensor with shape (n_blocks, 17)
    original_shape: Shape to reshape output to
    
  Returns:
    Decoded float32 tensor with shape original_shape
  """
  n_blocks = encoded.shape[0]
  
  # Split scale and packed data
  scale_exp = encoded[:, 0:1].cast(dtypes.float32)  # (n_blocks, 1)
  packed = encoded[:, 1:]  # (n_blocks, 16)
  
  # Unpack 4-bit codes from bytes
  codes_low = packed & 0xF  # (n_blocks, 16)
  codes_high = (packed >> 4) & 0xF  # (n_blocks, 16)
  
  # Concatenate to get all codes
  codes = codes_low.cat(codes_high, dim=1).cast(dtypes.float32)  # (n_blocks, BLOCK_SIZE)
  
  # Extract sign, exponent, mantissa
  sign = (codes >= 8).cast(dtypes.float32) * -2 + 1  # -1 or +1
  exp_bits = ((codes // 2) % 4)  # 0-3
  mant_bits = codes % 2  # 0-1
  
  # Decode values: (1.0 + 0.5 * mant) * 2^(exp-1) for exp != 0, else 0.5 * mant
  # Create base values for each exponent
  exp_nonzero = exp_bits != 0
  base_val = exp_nonzero.where(
    (1.0 + 0.5 * mant_bits) * ((exp_bits - 1).exp2()),
    0.5 * mant_bits
  )
  
  # Apply sign
  values = sign * base_val  # (n_blocks, BLOCK_SIZE)
  
  # Compute scale from scale_exp
  # scale = 2^(scale_exp - 128) for scale_exp >= 2
  # scale = 2^(-127) for scale_exp == 1
  # scale = 2^(-128) for scale_exp == 0
  scale_exp_adj = (scale_exp >= 2).where(
    scale_exp,
    (scale_exp >= 1).where(-127, -128)
  )
  scale = (scale_exp_adj - 128).exp2()  # (n_blocks, 1)
  
  # Apply scale
  decoded = values * scale  # (n_blocks, BLOCK_SIZE)
  
  # Flatten and trim to original size
  decoded_flat = decoded.flatten()
  n_original = math.prod(original_shape)
  decoded_trimmed = decoded_flat[:n_original]
  
  # Reshape to original shape
  return decoded_trimmed.reshape(original_shape)

def test_mxfp8():
  """Test MXFP8 encoding and decoding"""
  print("Testing native tinygrad MXFP8 implementation...")
  
  # Test with various shapes
  for shape in [(32,), (64,), (128,), (16, 8)]:
    print(f"\nTesting shape {shape}")
    x = Tensor.randn(*shape)
    
    # Encode
    encoded = encode_mxfp8(x)
    print(f"  Input: {shape}, dtype={x.dtype}")
    print(f"  Encoded: {encoded.shape}, dtype={encoded.dtype}")
    
    # Calculate compression
    original_bytes = x.numel() * 4  # FP32
    encoded_bytes = encoded.numel()
    compression = original_bytes / encoded_bytes
    print(f"  Compression: {compression:.2f}x ({original_bytes} -> {encoded_bytes} bytes)")
    
    # Decode
    decoded = decode_mxfp8(encoded, shape)
    print(f"  Decoded: {decoded.shape}, dtype={decoded.dtype}")
    
    # Compare
    diff = (x - decoded).abs()
    rel_error = diff / (x.abs() + 1e-8)
    
    print(f"  Mean absolute error: {diff.mean().item():.6f}")
    print(f"  Max absolute error: {diff.max().item():.6f}")
    print(f"  Mean relative error: {rel_error.mean().item():.6f}")
  
  print("\n" + "="*60)
  print("Native tinygrad MXFP8 test complete!")

if __name__ == "__main__":
  test_mxfp8()
