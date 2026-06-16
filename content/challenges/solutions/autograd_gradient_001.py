"""Reference solution for autograd-gradient-001. Verified in PyTorch 2.12.0, CPU."""

import torch


def poly_grad(x: torch.Tensor) -> torch.Tensor:
    x = x.clone().requires_grad_(True)
    y = (x**3 + 2 * x).sum()
    y.backward()
    return x.grad
