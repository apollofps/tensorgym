"""Interactive example for the 'Matrix Multiplication' capsule.

The learner runs this as-is in the sandbox to SEE how matmul shape rules work.
Verified in PyTorch 2.12.0 (CPU). It only prints; it asserts nothing, so the
learner can edit and re-run freely.
"""

import torch

# --- 2-D matrix multiplication: (N, K) @ (K, M) -> (N, M) ---
a = torch.tensor([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0]])          # shape (2, 3)
b = torch.tensor([[1.0, 0.0, 1.0, 2.0],
                   [0.0, 1.0, 0.0, 1.0],
                   [1.0, 1.0, 1.0, 0.0]])     # shape (3, 4)

result = a @ b                                 # same as torch.matmul(a, b)
print("a shape:          ", tuple(a.shape))            # (2, 3)
print("b shape:          ", tuple(b.shape))            # (3, 4)
print("a @ b shape:      ", tuple(result.shape))       # (2, 4)
print("a @ b:\n", result)

# --- torch.mm: strictly 2-D ---
mm_result = torch.mm(a, b)
print("\ntorch.mm result:  ", tuple(mm_result.shape))  # (2, 4)

# --- Batched matmul: (B, N, K) @ (B, K, M) -> (B, N, M) ---
batch_a = torch.randn(5, 2, 3)   # 5 matrices of shape (2, 3)
batch_b = torch.randn(5, 3, 4)   # 5 matrices of shape (3, 4)

batch_result = torch.bmm(batch_a, batch_b)
print("\nbatch_a shape:    ", tuple(batch_a.shape))         # (5, 2, 3)
print("batch_b shape:    ", tuple(batch_b.shape))           # (5, 3, 4)
print("bmm result shape: ", tuple(batch_result.shape))      # (5, 2, 4)

# torch.matmul also handles batched inputs (and broadcasts batch dims)
matmul_result = torch.matmul(batch_a, batch_b)
print("matmul batch shape:", tuple(matmul_result.shape))    # (5, 2, 4)
