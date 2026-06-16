import torch


def flatten_view(x: torch.Tensor) -> torch.Tensor:
    return x.view(-1)
