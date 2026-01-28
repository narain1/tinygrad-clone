import pathlib
import numpy as np
from tinygrad import Device, Tensor, dtypes
from tinygrad.helpers import Context, getenv
from tinygrad.runtime.support.compiler_cuda import pretty_ptx, NVCCCompiler

def compile_mxfp8_kernel():
  """Compile the MXFP8 CUDA kernel"""
  code = (pathlib.Path(__file__).parent / "mxfp8.cu").read_text()
  device = Device["CUDA"]
  kitten_args = [
    f"-I{(pathlib.Path(__file__).parent / 'include').as_posix()}", 
    "-std=c++20", 
    "--expt-relaxed-constexpr"
  ]
  lib = NVCCCompiler(device.compiler.arch, kitten_args).compile(code)
  return lib, device

def encode_mxfp8(tensor: Tensor) -> Tensor:
  """
  Encode a tensor to MXFP8 format.
  
  MXFP8 uses microscaling: each block of 32 values shares a single 8-bit exponent.
  Each value is encoded as 4 bits (1 sign, 2 exponent, 1 mantissa).
  
  Args:
    tensor: Input tensor (float32 or bfloat16)
    
  Returns:
    Encoded tensor as uint8 array (17 bytes per 32 values: 1 scale + 16 data bytes)
  """
  if tensor.device != "CUDA":
    raise ValueError("MXFP8 encoding only supports CUDA tensors")
  
  lib, device = compile_mxfp8_kernel()
  
  # Get kernel name for encoding
  kernel_name = None
  for line in lib.decode().split("\n"):
    if ".globl" in line and "encode_mxfp8_kernel" in line:
      kernel_name = line.split("\t")[1]
      break
  
  if kernel_name is None:
    raise RuntimeError("Could not find encode_mxfp8_kernel in compiled code")
  
  prg = device.runtime(kernel_name, lib)
  
  # Calculate output size: each 32 elements -> 17 bytes (1 scale + 16 data)
  n = tensor.numel()
  n_blocks = (n + 31) // 32
  out = Tensor.empty(n_blocks * 17, device='CUDA', dtype=dtypes.uint8)
  
  tensor_flat = tensor.flatten()
  Tensor.realize(tensor_flat, out)
  
  # Launch kernel
  threads_per_block = 256
  num_blocks = (n_blocks + threads_per_block - 1) // threads_per_block
  
  prg(out.uop.buffer.ensure_allocated()._buf, 
      tensor_flat.uop.buffer._buf, 
      n,
      global_size=(num_blocks,1,1), 
      local_size=(threads_per_block,1,1))
  
  return out

def decode_mxfp8(encoded: Tensor, shape: tuple) -> Tensor:
  """
  Decode MXFP8 format back to float32.
  
  Args:
    encoded: Encoded uint8 tensor
    shape: Original tensor shape to restore
    
  Returns:
    Decoded float32 tensor
  """
  if encoded.device != "CUDA":
    raise ValueError("MXFP8 decoding only supports CUDA tensors")
  
  lib, device = compile_mxfp8_kernel()
  
  # Get kernel name for decoding
  kernel_name = None
  for line in lib.decode().split("\n"):
    if ".globl" in line and "decode_mxfp8_kernel" in line:
      kernel_name = line.split("\t")[1]
      break
  
  if kernel_name is None:
    raise RuntimeError("Could not find decode_mxfp8_kernel in compiled code")
  
  prg = device.runtime(kernel_name, lib)
  
  # Calculate output size
  n = np.prod(shape)
  out = Tensor.empty(n, device='CUDA', dtype=dtypes.float32)
  
  Tensor.realize(encoded, out)
  
  # Launch kernel
  threads_per_block = 256
  num_blocks = (n + threads_per_block - 1) // threads_per_block
  
  prg(out.uop.buffer.ensure_allocated()._buf,
      encoded.uop.buffer._buf,
      n,
      global_size=(num_blocks,1,1),
      local_size=(threads_per_block,1,1))
  
  return out.reshape(shape)

def mxfp8_matmul(a: Tensor, b: Tensor) -> Tensor:
  """
  Matrix multiplication with MXFP8 encoded inputs.
  
  Args:
    a: Left matrix (M x K), MXFP8 encoded
    b: Right matrix (K x N), MXFP8 encoded
    
  Returns:
    Result matrix (M x N) in float32
  """
  if a.device != "CUDA" or b.device != "CUDA":
    raise ValueError("MXFP8 matmul only supports CUDA tensors")
  
  lib, device = compile_mxfp8_kernel()
  
  # Get kernel name for matmul
  kernel_name = None
  for line in lib.decode().split("\n"):
    if ".globl" in line and "mxfp8_matmul_kernel" in line:
      kernel_name = line.split("\t")[1]
      break
  
  if kernel_name is None:
    raise RuntimeError("Could not find mxfp8_matmul_kernel in compiled code")
  
  prg = device.runtime(kernel_name, lib)
  
  # For now, assume a and b are 2D
  M, K = a.shape
  K2, N = b.shape
  assert K == K2, "Matrix dimensions must match for multiplication"
  
  out = Tensor.empty(M, N, device='CUDA', dtype=dtypes.float32)
  Tensor.realize(a, b, out)
  
  # Launch kernel with 16x16 thread blocks
  BLOCK_SIZE = 16
  grid_x = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
  grid_y = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
  
  prg(out.uop.buffer.ensure_allocated()._buf,
      a.uop.buffer._buf,
      b.uop.buffer._buf,
      M, N, K,
      global_size=(grid_x, grid_y, 1),
      local_size=(BLOCK_SIZE, BLOCK_SIZE, 1))
  
  return out

if __name__ == "__main__":
  # Test MXFP8 encoding and decoding
  print("Testing MXFP8 encoding/decoding...")
  
  # Create test tensor
  x = Tensor.randn(128, 128, device='CUDA', dtype=dtypes.float32)
  print(f"Input shape: {x.shape}, dtype: {x.dtype}")
  print(f"Input stats: min={x.min().item():.4f}, max={x.max().item():.4f}, mean={x.mean().item():.4f}")
  
  # Encode
  print("\nEncoding to MXFP8...")
  encoded = encode_mxfp8(x)
  print(f"Encoded shape: {encoded.shape}")
  compression_ratio = (x.numel() * 4) / encoded.numel()
  print(f"Compression ratio: {compression_ratio:.2f}x")
  
  # Decode
  print("\nDecoding from MXFP8...")
  decoded = decode_mxfp8(encoded, x.shape)
  print(f"Decoded shape: {decoded.shape}, dtype: {decoded.dtype}")
  print(f"Decoded stats: min={decoded.min().item():.4f}, max={decoded.max().item():.4f}, mean={decoded.mean().item():.4f}")
  
  # Compare
  print("\nComparison:")
  diff = (x - decoded).abs()
  print(f"Mean absolute error: {diff.mean().item():.6f}")
  print(f"Max absolute error: {diff.max().item():.6f}")
  relative_error = (diff / (x.abs() + 1e-8)).mean()
  print(f"Mean relative error: {relative_error.item():.6f}")
  
  print("\n" + "="*50)
  print("MXFP8 test completed!")
