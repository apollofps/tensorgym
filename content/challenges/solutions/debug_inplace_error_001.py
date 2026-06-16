"""Reference solution for debug-inplace-error-001. Verified in PyTorch 2.12.0, CPU."""

import torch


def fix_inplace(x: torch.Tensor) -> torch.Tensor:
    x = x.clone().requires_grad_(True)
    z = x + 1
    y = (z ** 2).sum()
    y.backward()
    return x.grad
