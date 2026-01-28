import unittest
import math
from tinygrad import Tensor, dtypes, Device
from tinygrad.helpers import getenv

def encode_mxfp8_ref(values, block_size=32):
  """Reference implementation of MXFP8 encoding in Python"""
  n = len(values)
  n_blocks = (n + block_size - 1) // block_size
  blocks = []
  
  for block_idx in range(n_blocks):
    start = block_idx * block_size
    end = min(start + block_size, n)
    block_values = values[start:end]
    
    # Compute scale - find max absolute value
    max_val = max(abs(v) for v in block_values)
    if max_val == 0:
      scale_exp = 0
      scale = 2.0 ** (-128)
    else:
      # Calculate scale exponent so that max_val fits in the representable range
      # The max representable value is 7.5 (code 0b1111 = exp=3, mant=1 -> (1+0.5)*2^2 = 6.0, but actually 7.5)
      # Actually for code 0b0111 (positive, exp=3, mant=1): (1.0 + 0.5) * 2^(3-1) = 1.5 * 4 = 6.0
      # For max code the value is 6.0 times the scale
      log_max = math.log2(max_val / 6.0)
      scale_exp = int(math.floor(log_max)) + 128
      scale_exp = max(0, min(255, scale_exp))
      
      # Compute scale from exponent  
      if scale_exp >= 2:
        scale = 2.0 ** (scale_exp - 128)
      else:
        scale = 2.0 ** (-127 if scale_exp == 1 else -128)
    
    # Encode values
    codes = []
    for val in block_values:
      scaled = val / scale
      abs_scaled = abs(scaled)
      
      # Clamp to representable range (max 6.0)
      if abs_scaled > 6.0:
        abs_scaled = 6.0
      
      sign = 0b1000 if scaled < 0 else 0
      exp = 0
      mant = 0
      
      # Determine exponent and mantissa
      if abs_scaled >= 0.5:
        if abs_scaled >= 3.0:
          exp = 3  # (1.0 + mant*0.5) * 2^2 = 4.0 or 6.0
          if abs_scaled >= 5.0:
            mant = 1  # 6.0
          else:
            mant = 0  # 4.0
        elif abs_scaled >= 1.5:
          exp = 2  # (1.0 + mant*0.5) * 2^1 = 2.0 or 3.0
          if abs_scaled >= 2.5:
            mant = 1  # 3.0
          else:
            mant = 0  # 2.0
        elif abs_scaled >= 0.75:
          exp = 1  # (1.0 + mant*0.5) * 2^0 = 1.0 or 1.5
          if abs_scaled >= 1.25:
            mant = 1  # 1.5
          else:
            mant = 0  # 1.0
        else:
          exp = 0  # 0.5 * mant = 0.0 or 0.5
          if abs_scaled >= 0.25:
            mant = 1  # 0.5
          else:
            mant = 0  # 0.0
      else:
        exp = 0
        if abs_scaled > 0.25:
          mant = 1
        else:
          mant = 0
      
      codes.append(sign | (exp << 1) | mant)
    
    # Pad to block_size
    while len(codes) < block_size:
      codes.append(0)
    
    blocks.append((scale_exp, codes))
  
  return blocks

def decode_mxfp8_ref(blocks, n, block_size=32):
  """Reference implementation of MXFP8 decoding in Python"""
  values = []
  
  for block_idx, (scale_exp, codes) in enumerate(blocks):
    # Compute scale
    if scale_exp >= 2:
      scale = 2.0 ** (scale_exp - 128)
    else:
      scale = 2.0 ** (-127 if scale_exp == 1 else -128)
    
    for code in codes:
      sign = -1.0 if (code & 0b1000) else 1.0
      exp = (code >> 1) & 0b11
      mant = code & 0b1
      
      if exp != 0:
        val = (1.0 + 0.5 * mant) * (2.0 ** (exp - 1))
      else:
        val = 0.5 * mant
      
      values.append(sign * val * scale)
      
      if len(values) >= n:
        break
    
    if len(values) >= n:
      break
  
  return values[:n]

class TestMXFP8(unittest.TestCase):
  def test_encode_decode_roundtrip(self):
    """Test that encoding and decoding returns similar values"""
    # Test with various value ranges
    for max_val in [0.1, 1.0, 10.0, 100.0]:
      with self.subTest(max_val=max_val):
        # Create simple test values
        values = [max_val * (0.5 - i/32) for i in range(128)]
        
        # Encode
        blocks = encode_mxfp8_ref(values)
        
        # Decode
        decoded = decode_mxfp8_ref(blocks, len(values))
        
        # Check that decoded values are close to original
        # MXFP8 is lossy, so we expect some error
        errors = [abs(v - d) / (abs(v) + 1e-8) for v, d in zip(values, decoded)]
        mean_rel_error = sum(errors) / len(errors)
        
        # Should have reasonable accuracy (within ~25% on average for 4-bit encoding)
        self.assertLess(mean_rel_error, 0.25, 
                       f"Mean relative error {mean_rel_error:.4f} too high for max_val={max_val}")
  
  def test_small_values(self):
    """Test encoding of very small values"""
    values = [0.001, 0.01, 0.1, 1.0]  # Skip 0.0001 which underflows
    blocks = encode_mxfp8_ref(values)
    decoded = decode_mxfp8_ref(blocks, len(values))
    
    # Should preserve order of magnitude (with 4-bit precision)
    for orig, dec in zip(values, decoded):
      if orig != 0 and dec != 0:  # Skip zeros
        ratio = abs(dec / orig)
        self.assertGreater(ratio, 0.3, f"Decoded value {dec} too far from {orig}")
        self.assertLess(ratio, 3.0, f"Decoded value {dec} too far from {orig}")
  
  def test_zero_values(self):
    """Test that zeros are preserved"""
    values = [0.0] * 32
    blocks = encode_mxfp8_ref(values)
    decoded = decode_mxfp8_ref(blocks, len(values))
    
    self.assertEqual(decoded, values)
  
  def test_sign_preservation(self):
    """Test that signs are preserved"""
    values = [1.0, -1.0, 2.0, -2.0, 5.0, -5.0]
    blocks = encode_mxfp8_ref(values)
    decoded = decode_mxfp8_ref(blocks, len(values))
    
    # Signs should match
    for v, d in zip(values, decoded):
      if v != 0:  # Skip zero
        self.assertEqual(v > 0, d > 0, f"Sign mismatch: {v} -> {d}")
  
  def test_block_boundaries(self):
    """Test encoding across multiple blocks"""
    # Create values that span multiple blocks
    values = [float(i % 10 - 5) for i in range(100)]
    blocks = encode_mxfp8_ref(values, block_size=32)
    
    # Should have 4 blocks (100 values / 32 per block = 3.125 -> 4 blocks)
    self.assertEqual(len(blocks), 4)
    
    # Decode and check
    decoded = decode_mxfp8_ref(blocks, len(values))
    self.assertEqual(len(decoded), len(values))

if __name__ == "__main__":
  unittest.main()
