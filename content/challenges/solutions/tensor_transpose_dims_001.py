import torch


def swap_last_two(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(-2, -1)
