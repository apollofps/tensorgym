"""Reference solution for autograd-accumulate-001. Verified in PyTorch 2.12.0, CPU."""

import torch


def accumulated_grad(x: torch.Tensor) -> torch.Tensor:
    x = x.clone().requires_grad_(True)
    y1 = (x**2).sum()
    y1.backward()
    y2 = (x * 3).sum()
    y2.backward()
    return x.grad
