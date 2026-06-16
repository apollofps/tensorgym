"""Reference solution for nn-linear-module-001.

Verified in PyTorch 2.12.0, CPU.
"""

import torch
import torch.nn as nn


def two_layer_forward(x: torch.Tensor) -> torch.Tensor:
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    return model(x)
