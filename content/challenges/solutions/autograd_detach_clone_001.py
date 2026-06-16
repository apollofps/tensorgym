"""Reference solution for autograd-detach-clone-001. Verified in PyTorch 2.12.0, CPU."""

import torch


def safe_copy(x: torch.Tensor) -> torch.Tensor:
    x = x.clone().requires_grad_(True)
    y = x * 2
    return y.detach().clone()
