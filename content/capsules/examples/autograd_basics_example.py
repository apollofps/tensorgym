"""Interactive example for the 'Autograd Basics' capsule.

The learner runs this as-is in the sandbox to SEE how autograd computes
gradients through a computation graph. Verified in PyTorch 2.12.0 (CPU).
It only prints; it asserts nothing, so the learner can edit and re-run freely.
"""

import torch

x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x ** 2           # y = [4.0, 9.0]
z = y.sum()           # z = 13.0 (scalar)
z.backward()
print(f"x = {x}")
print(f"y = x^2 = {y}")
print(f"z = sum(y) = {z.item()}")
print(f"dz/dx = 2*x = {x.grad}")  # [4.0, 6.0]
print(f"x.is_leaf = {x.is_leaf}")  # True
print(f"y.is_leaf = {y.is_leaf}")  # False
