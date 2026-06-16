"""Interactive example for the 'Tensor Indexing and Slicing' capsule.

Run this to see basic indexing, slicing, boolean indexing, fancy indexing, and
diagonal extraction. Verified in PyTorch 2.12.0 (CPU).
"""

import torch

x = torch.tensor([[10.0, 20.0, 30.0],
                   [40.0, 50.0, 60.0],
                   [70.0, 80.0, 90.0]])

# Basic indexing — single element
print("x[0, 1]:       ", x[0, 1].item())        # 20.0

# Row selection
print("x[0] (row 0):  ", x[0].tolist())         # [10.0, 20.0, 30.0]

# Column selection
print("x[:, 2] (col 2):", x[:, 2].tolist())     # [30.0, 60.0, 90.0]

# Slicing — rows 0-1, columns 1-2
print("x[0:2, 1:3]:\n", x[0:2, 1:3])           # [[20, 30], [50, 60]]

# Boolean indexing — select elements > 50
mask = x > 50
print("\nmask (x > 50):\n", mask)
print("x[x > 50]:     ", x[mask].tolist())      # [60.0, 70.0, 80.0, 90.0]

# Fancy indexing — select rows 0 and 2
idx = torch.tensor([0, 2])
print("\nfancy x[[0,2]]:\n", x[idx])             # rows 0 and 2

# Diagonal
diag = torch.diagonal(x)
print("\ndiagonal:      ", diag.tolist())         # [10.0, 50.0, 90.0]
