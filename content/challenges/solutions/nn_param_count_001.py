"""Reference solution for nn-param-count-001.

Verified in PyTorch 2.12.0, CPU.
"""

import torch.nn as nn


def count_params(in_size: int, hidden: int, out_size: int) -> int:
    model = nn.Sequential(
        nn.Linear(in_size, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, out_size),
    )
    return sum(p.numel() for p in model.parameters())
