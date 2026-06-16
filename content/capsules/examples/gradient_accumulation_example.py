"""Interactive example for the 'Gradient Accumulation' capsule.

Shows how gradients accumulate across backward() calls, and how to zero them.
Verified in PyTorch 2.12.0 (CPU). It only prints; it asserts nothing, so the
learner can edit and re-run freely.
"""

import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# First backward: gradient of x^2 is 2x
loss1 = (x ** 2).sum()
loss1.backward()
print(f"After first backward (x^2):  x.grad = {x.grad}")  # [2.0, 4.0, 6.0]

# Second backward WITHOUT zeroing: gradients accumulate
loss2 = (x * 3).sum()
loss2.backward()
print(f"After second backward (x*3): x.grad = {x.grad}")  # [5.0, 7.0, 9.0]
print("  (accumulated: 2x + 3)")

# Zero the gradient
x.grad.zero_()
print(f"After zeroing:               x.grad = {x.grad}")  # [0.0, 0.0, 0.0]

# Now a clean backward
loss3 = (x * 5).sum()
loss3.backward()
print(f"After clean backward (x*5):  x.grad = {x.grad}")  # [5.0, 5.0, 5.0]
