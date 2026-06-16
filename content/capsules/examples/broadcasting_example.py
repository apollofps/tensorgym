"""Interactive example for the 'Broadcasting Rules' capsule.

Run this to see how PyTorch broadcasts tensors of different shapes during
element-wise operations. Verified in PyTorch 2.12.0 (CPU).
"""

import torch

# (2, 3) matrix + (3,) vector — vector broadcasts across rows
matrix = torch.tensor([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]])
bias = torch.tensor([10.0, 20.0, 30.0])
result = matrix + bias
print("matrix (2,3) + bias (3,):")
print("  matrix:", matrix.tolist())
print("  bias:  ", bias.tolist())
print("  result:", result.tolist())         # [[11, 22, 33], [14, 25, 36]]
print("  result shape:", tuple(result.shape))  # (2, 3)

# (3, 1) column + (1, 4) row — both dimensions stretch
col = torch.tensor([[1.0], [2.0], [3.0]])   # shape (3, 1)
row = torch.tensor([[10.0, 20.0, 30.0, 40.0]])  # shape (1, 4)
outer = col + row
print("\ncol (3,1) + row (1,4):")
print("  result shape:", tuple(outer.shape))    # (3, 4)
print("  result:\n", outer)

# Scalar broadcasts with everything
s = torch.tensor(100.0)
print("\nscalar + matrix:", (s + matrix).tolist())

# Incompatible shapes — this would raise an error
# Uncomment to see: torch.rand(2, 3) + torch.rand(4)
print("\nNote: (2, 3) + (4,) would raise a RuntimeError — 3 != 4")
