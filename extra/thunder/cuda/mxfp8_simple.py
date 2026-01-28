#!/usr/bin/env python3
"""
Simple example demonstrating MXFP8 encoding and decoding.
Run with: python3 extra/thunder/cuda/mxfp8_simple.py
"""

def encode_decode_example():
  """Simple encode/decode example that works without CUDA"""
  import math
  
  print("="*60)
  print("MXFP8 Encoding/Decoding Example")
  print("="*60)
  
  # Test values
  values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -1.5, -3.0]
  print(f"\nOriginal values: {values}")
  
  # Compute scale
  max_val = max(abs(v) for v in values)
  log_max = math.log2(max_val / 6.0)
  scale_exp = int(math.floor(log_max)) + 128
  scale_exp = max(0, min(255, scale_exp))
  
  if scale_exp >= 2:
    scale = 2.0 ** (scale_exp - 128)
  else:
    scale = 2.0 ** (-127 if scale_exp == 1 else -128)
  
  print(f"\nScale exponent: {scale_exp}")
  print(f"Scale value: {scale}")
  
  # Encode each value
  print("\nEncoding:")
  codes = []
  for val in values:
    scaled = val / scale
    abs_scaled = abs(scaled)
    
    if abs_scaled > 6.0:
      abs_scaled = 6.0
    
    sign = 0b1000 if scaled < 0 else 0
    exp = 0
    mant = 0
    
    if abs_scaled >= 0.5:
      if abs_scaled >= 3.0:
        exp = 3
        mant = 1 if abs_scaled >= 5.0 else 0
      elif abs_scaled >= 1.5:
        exp = 2
        mant = 1 if abs_scaled >= 2.5 else 0
      elif abs_scaled >= 0.75:
        exp = 1
        mant = 1 if abs_scaled >= 1.25 else 0
      else:
        exp = 0
        mant = 1 if abs_scaled >= 0.25 else 0
    else:
      exp = 0
      mant = 1 if abs_scaled > 0.25 else 0
    
    code = sign | (exp << 1) | mant
    codes.append(code)
    print(f"  {val:6.1f} -> code {code:04b} (sign={sign>>3}, exp={exp}, mant={mant})")
  
  # Decode each code
  print("\nDecoding:")
  decoded = []
  for i, code in enumerate(codes):
    sign = -1.0 if (code & 0b1000) else 1.0
    exp = (code >> 1) & 0b11
    mant = code & 0b1
    
    if exp != 0:
      val = (1.0 + 0.5 * mant) * (2.0 ** (exp - 1))
    else:
      val = 0.5 * mant
    
    final = sign * val * scale
    decoded.append(final)
    error = abs(values[i] - final)
    print(f"  code {code:04b} -> {final:6.3f} (original: {values[i]:6.1f}, error: {error:.3f})")
  
  # Summary
  print(f"\nSummary:")
  print(f"  Values encoded: {len(values)}")
  print(f"  Bits per value: 4")
  print(f"  Total bits: {len(values) * 4 + 8} (including 8-bit scale)")
  print(f"  Compression vs FP32: {len(values) * 32 / (len(values) * 4 + 8):.1f}x")
  
  errors = [abs(o - d) for o, d in zip(values, decoded)]
  if any(abs(v) > 0 for v in values):
    rel_errors = [abs(o - d) / (abs(o) + 1e-8) for o, d in zip(values, decoded)]
    print(f"  Mean absolute error: {sum(errors) / len(errors):.4f}")
    print(f"  Mean relative error: {sum(rel_errors) / len(rel_errors):.4f}")

if __name__ == "__main__":
  encode_decode_example()
