"""Reference solution for debug-wrong-reduction-001. Verified in PyTorch 2.12.0, CPU."""

import torch


def fix_row_mean(x: torch.Tensor) -> torch.Tensor:
    return x.mean(dim=1, keepdim=True)
