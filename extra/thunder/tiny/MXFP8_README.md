# Native Tinygrad MXFP8 Implementation

This is a native tinygrad implementation of MXFP8 (Microscaling FP8) format that uses only tinygrad tensor operations, making it device-agnostic and more maintainable than raw CUDA kernels.

## Overview

MXFP8 (Microscaling FP8) is a compact floating-point format where:
- Values are grouped into blocks of 32 elements
- Each block shares a single 8-bit scale exponent
- Each value is encoded as 4 bits (1 sign, 2 exponent, 1 mantissa)
- Total: 17 bytes per block (1 byte scale + 16 bytes data)

This provides ~7.5x compression compared to FP32, with ~13-25% relative error.

## Advantages of Native Tinygrad Implementation

Unlike the CUDA implementation in `extra/thunder/cuda/mxfp8.cu`, this implementation:

✅ **Device-agnostic**: Works on any backend (CPU, CUDA, Metal, WebGPU, etc.)
✅ **Maintainable**: Uses high-level tinygrad operations instead of low-level CUDA
✅ **Automatic differentiation**: Can integrate with tinygrad's autograd if needed
✅ **Composable**: Easy to integrate with other tinygrad operations

## Usage

```python
from extra.thunder.tiny.mxfp8 import encode_mxfp8, decode_mxfp8
from tinygrad import Tensor

# Create a tensor
x = Tensor.randn(128, 128)

# Encode to MXFP8 format
encoded = encode_mxfp8(x)
print(f"Original: {x.numel() * 4} bytes")
print(f"Encoded: {encoded.numel()} bytes")
print(f"Compression: {(x.numel() * 4) / encoded.numel():.2f}x")

# Decode back to float32
decoded = decode_mxfp8(encoded, x.shape)

# Check accuracy
error = (x - decoded).abs().mean()
print(f"Mean absolute error: {error.item():.6f}")
```

## Implementation Details

### Encoding Process

1. **Flatten and pad** input to multiple of 32
2. **Reshape** into blocks of 32 values
3. **Compute scale** for each block (based on max absolute value)
4. **Quantize** each value to 4 bits:
   - 1 bit for sign
   - 2 bits for exponent (0-3)
   - 1 bit for mantissa (0-1)
5. **Pack** two 4-bit values into each byte
6. **Concatenate** scale byte and packed data

### Decoding Process

1. **Split** scale exponent and packed data
2. **Unpack** 4-bit values from bytes
3. **Extract** sign, exponent, mantissa bits
4. **Compute** base value: `(1.0 + 0.5*mant) * 2^(exp-1)` (or `0.5*mant` if exp=0)
5. **Apply** sign and shared scale
6. **Reshape** to original shape

## Test Results

Example output from `python extra/thunder/tiny/mxfp8.py`:

```
Testing shape (128,)
  Input: (128,), dtype=dtypes.float
  Encoded: (4, 17), dtype=dtypes.uchar
  Compression: 7.53x (512 -> 68 bytes)
  Decoded: (128,), dtype=dtypes.float
  Mean absolute error: 0.120858
  Max absolute error: 1.468596
  Mean relative error: 0.199657
```

## Comparison with CUDA Implementation

| Aspect | Native Tinygrad | CUDA Kernel |
|--------|----------------|-------------|
| Device Support | All backends | CUDA only |
| Code Complexity | ~200 lines Python | ~170 lines CUDA + wrapper |
| Performance | Good (optimized by tinygrad) | Excellent (hand-tuned) |
| Maintainability | High (tinygrad ops) | Medium (raw CUDA) |
| Use Case | General purpose | Performance-critical |

## Files

- **`mxfp8.py`** - Native tinygrad implementation with `encode_mxfp8()` and `decode_mxfp8()`

## See Also

- CUDA implementation: `extra/thunder/cuda/mxfp8.cu` and `mxfp8.py`
- Flash Attention example: `extra/thunder/tiny/fa.py`
- Thunder Kittens toolkit: `extra/thunder/tiny/tk/`
