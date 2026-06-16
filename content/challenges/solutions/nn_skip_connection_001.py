"""Reference solution for nn-skip-connection-001.

Verified in PyTorch 2.12.0, CPU.
"""

import torch
import torch.nn as nn


def residual_forward(x: torch.Tensor) -> torch.Tensor:
    torch.manual_seed(42)
    linear = nn.Linear(4, 4)
    return torch.relu(linear(x)) + x
