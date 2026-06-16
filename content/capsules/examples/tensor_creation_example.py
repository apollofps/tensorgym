"""Interactive example for the 'Creating Tensors' capsule.

Run this as-is to see how different factory functions create tensors with
specific shapes and dtypes. Verified in PyTorch 2.12.0 (CPU).
"""

import torch

# torch.zeros: all zeros
z = torch.zeros(2, 3)
print("zeros(2, 3) shape:", tuple(z.shape))    # (2, 3)
print("zeros(2, 3) dtype:", z.dtype)           # torch.float32
print("zeros(2, 3):\n", z)

# torch.ones: all ones
o = torch.ones(3, 4, dtype=torch.float64)
print("\nones(3, 4) shape:", tuple(o.shape))    # (3, 4)
print("ones(3, 4) dtype:", o.dtype)            # torch.float64

# torch.rand: uniform random in [0, 1)
r = torch.rand(2, 2)
print("\nrand(2, 2) shape:", tuple(r.shape))    # (2, 2)
print("rand(2, 2):\n", r)

# torch.arange: like Python range but returns a tensor
a = torch.arange(0, 10, 2)
print("\narange(0, 10, 2):", a.tolist())        # [0, 2, 4, 6, 8]
print("arange dtype:     ", a.dtype)           # torch.int64

# torch.linspace: evenly spaced between start and end (inclusive)
l = torch.linspace(0, 1, 5)
print("\nlinspace(0, 1, 5):", l.tolist())       # [0.0, 0.25, 0.5, 0.75, 1.0]
print("linspace dtype:   ", l.dtype)           # torch.float32

# torch.tensor: from a Python list
t = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
print("\ntensor from list shape:", tuple(t.shape))  # (2, 2)
print("tensor from list:\n", t)
