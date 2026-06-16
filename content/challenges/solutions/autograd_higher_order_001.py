"""Reference solution for autograd-higher-order-001. Verified in PyTorch 2.12.0, CPU."""

import torch


def second_derivative(x: torch.Tensor) -> torch.Tensor:
    x = x.clone().requires_grad_(True)
    y = (x**4).sum()
    grad1 = torch.autograd.grad(y, x, create_graph=True)[0]
    grad2 = torch.autograd.grad(grad1.sum(), x)[0]
    return grad2
