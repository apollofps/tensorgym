"""Reference solution for nn-train-eval-toggle-001.

Verified in PyTorch 2.12.0, CPU.
"""

import torch
import torch.nn as nn


def eval_forward(x: torch.Tensor) -> torch.Tensor:
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(4, 4), nn.Dropout(p=0.5))
    model.eval()
    return model(x)
