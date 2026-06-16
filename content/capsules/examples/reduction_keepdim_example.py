"""Interactive example for the 'Reduction Dimensions and keepdim' capsule.

The learner runs this as-is in the sandbox to SEE how keepdim changes the output
shape. Verified in PyTorch 2.12.0 (CPU). It only prints; it asserts nothing, so
the learner can edit and re-run freely.
"""

import torch

x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

print("x shape:               ", tuple(x.shape))            # (2, 3)
print("sum(dim=1):            ", tuple(x.sum(dim=1).shape))  # (2,)  -> row dim removed
print("sum(dim=1, keepdim):   ", tuple(x.sum(dim=1, keepdim=True).shape))  # (2, 1)

row_sums = x.sum(dim=1, keepdim=True)   # (2, 1)
print("row_sums:              ", row_sums.squeeze(1).tolist())
print("x / row_sums (rows->1):", (x / row_sums).tolist())
