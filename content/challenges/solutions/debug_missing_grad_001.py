"""Reference solution for debug-missing-grad-001. Verified in PyTorch 2.12.0, CPU."""

import torch


def fix_gradient(x: torch.Tensor) -> torch.Tensor:
    x = x.clone().requires_grad_(True)
    y = (x ** 2).sum()
    y.backward()
    return x.grad
