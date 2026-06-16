"""Reference solution for debug-shape-mismatch-001. Verified in PyTorch 2.12.0, CPU."""

import torch


def fix_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b.T
