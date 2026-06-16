import torch


def inplace_relu(x: torch.Tensor) -> torch.Tensor:
    return x.clamp_(min=0)
