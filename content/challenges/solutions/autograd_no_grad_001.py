"""Reference solution for autograd-no-grad-001. Verified in PyTorch 2.12.0, CPU."""

import torch


def no_grad_add(x: torch.Tensor) -> torch.Tensor:
    x = x.clone().requires_grad_(True)
    with torch.no_grad():
        result = x + x**2
    return result
