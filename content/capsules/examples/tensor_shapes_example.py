"""Interactive example for the 'Understanding Tensor Shapes' capsule.

Run this to see how .shape, .ndim, and .numel() describe tensors of different
ranks. Verified in PyTorch 2.12.0 (CPU).
"""

import torch

# Scalar (0-D tensor)
scalar = torch.tensor(42.0)
print("scalar value:", scalar.item())
print("scalar shape:", tuple(scalar.shape))    # ()
print("scalar ndim: ", scalar.ndim)            # 0
print("scalar numel:", scalar.numel())         # 1

# Vector (1-D tensor)
vector = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
print("\nvector shape:", tuple(vector.shape))   # (5,)
print("vector ndim: ", vector.ndim)            # 1
print("vector numel:", vector.numel())         # 5

# Matrix (2-D tensor)
matrix = torch.zeros(3, 4)
print("\nmatrix shape:", tuple(matrix.shape))   # (3, 4)
print("matrix ndim: ", matrix.ndim)            # 2
print("matrix numel:", matrix.numel())         # 12

# 3-D tensor (e.g. a batch of images without channels)
batch = torch.rand(2, 3, 4)
print("\n3-D shape:   ", tuple(batch.shape))    # (2, 3, 4)
print("3-D ndim:    ", batch.ndim)             # 3
print("3-D numel:   ", batch.numel())          # 24

# .shape and .size() are equivalent
print("\n.shape == .size():", matrix.shape == matrix.size())  # True
