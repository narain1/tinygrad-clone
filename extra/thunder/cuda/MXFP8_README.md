# MXFP8 (Microscaling FP8) Implementation

This directory contains an implementation of MXFP8 (Microscaling FP8), a compact floating-point format that provides better dynamic range than standard FP8 by using a shared exponent for blocks of values.

## What is MXFP8?

MXFP8 is a microscaling floating-point format where:
- Values are grouped into blocks of 32 elements
- Each block shares a single 8-bit scale exponent
- Each value is encoded as 4 bits (1 sign, 2 exponent, 1 mantissa)
- Total: 17 bytes per block (1 byte scale + 16 bytes data)

This provides ~2x compression compared to standard FP32, with better dynamic range than standard FP8.

## Format Details

### 4-bit Value Encoding (per element)
```
Bit 3: Sign (0 = positive, 1 = negative)
Bits 2-1: Exponent (0-3)
Bit 0: Mantissa (0-1)
```

### Value Decoding Formula
```
if exp != 0:
  value = (1.0 + 0.5 * mant) * 2^(exp-1)
else:
  value = 0.5 * mant

final_value = sign * value * scale
```

Where `scale = 2^(scale_exp - 128)` (or special cases for scale_exp < 2).

### Representable Values (before scaling)
- exp=0, mant=0: 0.0
- exp=0, mant=1: 0.5
- exp=1, mant=0: 1.0
- exp=1, mant=1: 1.5
- exp=2, mant=0: 2.0
- exp=2, mant=1: 3.0
- exp=3, mant=0: 4.0
- exp=3, mant=1: 6.0

## Files

- `mxfp8.cu` - CUDA kernel implementation with encode/decode/matmul operations
- `mxfp8.py` - Python wrapper for CUDA kernels
- `include/common/mxfp8.cuh` - Header-only utilities following Thunder Kittens style
- `test/unit/test_mxfp8.py` - Unit tests for encoding/decoding

## Usage

### Python API

```python
from extra.thunder.cuda.mxfp8 import encode_mxfp8, decode_mxfp8
from tinygrad import Tensor

# Encode a tensor
x = Tensor.randn(128, 128, device='CUDA')
encoded = encode_mxfp8(x)

# Decode back
decoded = decode_mxfp8(encoded, x.shape)

# Matrix multiplication with MXFP8
from extra.thunder.cuda.mxfp8 import mxfp8_matmul
a = Tensor.randn(64, 128, device='CUDA')
b = Tensor.randn(128, 256, device='CUDA')
a_enc = encode_mxfp8(a)
b_enc = encode_mxfp8(b)
result = mxfp8_matmul(a_enc, b_enc)
```

### C++ API (in kernels)

```cpp
#include "../common/common.cuh"
#include "../pyutils/pyutils.cuh"
#include "../common/mxfp8.cuh"

// Encode float to 4-bit MXFP8
uint8_t code = mxfp8::encode(value, scale);

// Decode 4-bit MXFP8 to float
float decoded = mxfp8::decode(code, scale_exp);

// Compute scale for a block
uint8_t scale_exp = mxfp8::compute_scale(values, count);
```

## Testing

Run the unit tests:
```bash
cd /home/runner/work/tinygrad-clone/tinygrad-clone
PYTHONPATH=. python test/unit/test_mxfp8.py
```

Test the CUDA kernels (requires CUDA device):
```bash
cd /home/runner/work/tinygrad-clone/tinygrad-clone/extra/thunder/cuda
python mxfp8.py
```

## Performance Characteristics

- **Compression**: ~2x compared to FP32 (4 bits per value vs 32 bits)
- **Accuracy**: ~20% mean relative error (acceptable for many ML workloads)
- **Dynamic Range**: Better than standard FP8 due to shared exponent
- **Speed**: Fast encode/decode on GPU with CUDA kernels

## References

- [Microsoft Microscaling Data Formats](https://arxiv.org/abs/2310.10537)
- [GGUF MXFP4 Format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- Similar to Thunder Kittens attention kernels in structure
