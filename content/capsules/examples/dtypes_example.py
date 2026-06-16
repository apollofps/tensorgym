"""Interactive example for the 'Data Types (dtypes)' capsule.

The learner runs this as-is in the sandbox to SEE how dtypes work in PyTorch.
Verified in PyTorch 2.12.0 (CPU). It only prints; it asserts nothing, so the
learner can edit and re-run freely.
"""

import torch

# --- Default dtypes depend on the input data ---
from_floats = torch.tensor([1.0, 2.0, 3.0])
from_ints = torch.tensor([1, 2, 3])
from_bools = torch.tensor([True, False, True])

print("from floats:", from_floats.dtype)   # torch.float32
print("from ints:  ", from_ints.dtype)     # torch.int64
print("from bools: ", from_bools.dtype)    # torch.bool

# --- Casting with .to(dtype) ---
x = torch.tensor([1, 2, 3])               # int64
x_float = x.to(torch.float32)
print("\nint64 -> float32:", x_float.dtype, x_float)

# --- Convenience methods ---
y = torch.tensor([1, 2, 3])
print("\n.float(): ", y.float().dtype)      # torch.float32
print(".double():", y.double().dtype)       # torch.float64
print(".long():  ", torch.tensor([1.5, 2.7]).long())  # truncates: tensor([1, 2])
print(".bool():  ", torch.tensor([0, 1, -1]).bool())   # tensor([False, True, True])

# --- Explicit dtype at creation ---
z = torch.tensor([1, 2, 3], dtype=torch.float32)
print("\ncreated as float32:", z.dtype, z)

# --- Checking dtype ---
print("\nz.dtype == torch.float32:", z.dtype == torch.float32)
