"""
Test to verify native tinygrad MXFP8 implementation matches expected behavior.
This is a unit test that can run without CUDA.
"""

import unittest
from tinygrad import Tensor, dtypes

class TestNativeMXFP8(unittest.TestCase):
  def test_encode_decode_roundtrip(self):
    """Test that encode/decode preserves approximate values"""
    from extra.thunder.tiny.mxfp8 import encode_mxfp8, decode_mxfp8
    
    # Test with a simple tensor
    x = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, -2.0,
                0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 0.0, 0.25,
                10.0, 20.0, 30.0, 40.0, 50.0, 60.0, -10.0, -20.0,
                5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 0.0, 0.125])
    
    # Encode and decode
    encoded = encode_mxfp8(x)
    decoded = decode_mxfp8(encoded, x.shape)
    
    # Check shape preserved
    self.assertEqual(x.shape, decoded.shape)
    
    # Check dtype
    self.assertEqual(encoded.dtype, dtypes.uint8)
    self.assertEqual(decoded.dtype, dtypes.float32)
    
    # MXFP8 is lossy, so we expect some error
    # But it should be reasonable (< 50% on average for this data)
    rel_error = ((x - decoded).abs() / (x.abs() + 1e-8)).mean().item()
    self.assertLess(rel_error, 0.5, f"Relative error {rel_error} too high")
  
  def test_compression_ratio(self):
    """Test that compression is approximately 7.5x"""
    from extra.thunder.tiny.mxfp8 import encode_mxfp8
    
    x = Tensor.randn(128)
    encoded = encode_mxfp8(x)
    
    original_bytes = x.numel() * 4  # FP32
    encoded_bytes = encoded.numel()
    compression = original_bytes / encoded_bytes
    
    # Should be around 7.5x compression
    self.assertGreater(compression, 7.0)
    self.assertLess(compression, 8.0)
  
  def test_different_shapes(self):
    """Test encoding/decoding with various shapes"""
    from extra.thunder.tiny.mxfp8 import encode_mxfp8, decode_mxfp8
    
    shapes = [(32,), (64,), (128,), (16, 8), (8, 8, 2)]
    
    for shape in shapes:
      with self.subTest(shape=shape):
        x = Tensor.randn(*shape)
        encoded = encode_mxfp8(x)
        decoded = decode_mxfp8(encoded, shape)
        
        # Check shape preserved
        self.assertEqual(x.shape, decoded.shape)
        
        # Check reasonable error
        rel_error = ((x - decoded).abs() / (x.abs() + 1e-8)).mean().item()
        self.assertLess(rel_error, 0.5)
  
  def test_zeros(self):
    """Test that zeros are handled correctly"""
    from extra.thunder.tiny.mxfp8 import encode_mxfp8, decode_mxfp8
    
    x = Tensor.zeros(32)
    encoded = encode_mxfp8(x)
    decoded = decode_mxfp8(encoded, x.shape)
    
    # Zeros should decode to zeros (or very close)
    self.assertLess(decoded.abs().max().item(), 0.01)
  
  def test_sign_preservation(self):
    """Test that signs are generally preserved"""
    from extra.thunder.tiny.mxfp8 import encode_mxfp8, decode_mxfp8
    
    x = Tensor([1.0, -1.0, 2.0, -2.0, 5.0, -5.0, 10.0, -10.0,
                3.0, -3.0, 4.0, -4.0, 6.0, -6.0, 7.0, -7.0,
                1.5, -1.5, 2.5, -2.5, 3.5, -3.5, 4.5, -4.5,
                5.5, -5.5, 6.5, -6.5, 7.5, -7.5, 8.5, -8.5])
    
    encoded = encode_mxfp8(x)
    decoded = decode_mxfp8(encoded, x.shape)
    
    # Check that most signs match (allowing for some quantization error)
    # Count how many have the same sign
    same_sign = ((x > 0) == (decoded > 0)).cast(dtypes.float32).sum().item()
    total = x.numel()
    
    # At least 90% should have correct sign
    self.assertGreater(same_sign / total, 0.9)
  
  def test_block_boundaries(self):
    """Test encoding across block boundaries"""
    from extra.thunder.tiny.mxfp8 import encode_mxfp8, decode_mxfp8
    
    # Create tensor that spans multiple blocks (32 values per block)
    x = Tensor.randn(100)  # Will create 4 blocks (128 values padded)
    
    encoded = encode_mxfp8(x)
    decoded = decode_mxfp8(encoded, (100,))
    
    # Should have 4 blocks (100 / 32 = 3.125, rounds up to 4)
    n_blocks = encoded.shape[0]
    self.assertEqual(n_blocks, 4)
    
    # Each block is 17 bytes
    self.assertEqual(encoded.shape[1], 17)
    
    # Decoded should match original shape
    self.assertEqual(decoded.shape, (100,))
  
  def test_device_agnostic(self):
    """Test that implementation works on default device"""
    from extra.thunder.tiny.mxfp8 import encode_mxfp8, decode_mxfp8
    
    # Create tensor on default device
    x = Tensor.randn(64)
    
    # Should work without specifying device
    encoded = encode_mxfp8(x)
    decoded = decode_mxfp8(encoded, x.shape)
    
    # Check it produces reasonable results
    rel_error = ((x - decoded).abs() / (x.abs() + 1e-8)).mean().item()
    self.assertLess(rel_error, 0.5)

if __name__ == "__main__":
  unittest.main()
