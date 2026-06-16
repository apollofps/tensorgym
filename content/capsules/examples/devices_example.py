"""Interactive example for the 'Devices (CPU vs CUDA)' capsule.

The learner runs this as-is in the sandbox to SEE how device attributes work.
Verified in PyTorch 2.12.0 (CPU). It only prints; it asserts nothing, so
the learner can edit and re-run freely.
"""

import torch

x = torch.tensor([1.0, 2.0, 3.0])

print("x.device:          ", x.device)
print("x.device.type:     ", x.device.type)

y = torch.tensor([4.0, 5.0, 6.0], device="cpu")
print("y.device:          ", y.device)

z = x.to("cpu")
print("z.device (after .to('cpu')):", z.device)

print("CUDA available?    ", torch.cuda.is_available())

w = torch.zeros(2, 3, device=torch.device("cpu"))
print("w.device:          ", w.device)
print("w.shape:           ", tuple(w.shape))
