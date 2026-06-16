"""Interactive example for the 'nn.Module Basics' capsule.

The learner runs this as-is in the sandbox to SEE how nn.Module, nn.Linear,
and nn.ReLU fit together. Verified in PyTorch 2.12.0 (CPU). It only prints;
it asserts nothing, so the learner can edit and re-run freely.
"""

import torch
import torch.nn as nn

torch.manual_seed(42)


class TwoLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 4)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(4, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


model = TwoLayerNet()

x = torch.randn(5, 3)
output = model(x)
print("Input shape: ", tuple(x.shape))       # (5, 3)
print("Output shape:", tuple(output.shape))   # (5, 2)

print("\nModel parameters:")
for name, param in model.named_parameters():
    print(f"  {name:20s} shape={tuple(param.shape)}")

total = sum(p.numel() for p in model.parameters())
print(f"\nTotal learnable parameters: {total}")
