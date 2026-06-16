"""Reference solution for autograd-chain-001. Verified in PyTorch 2.12.0, CPU."""

import torch


def sin_x2_grad(x: torch.Tensor) -> torch.Tensor:
    x = x.clone().requires_grad_(True)
    y = torch.sin(x**2).sum()
    y.backward()
    return x.grad
