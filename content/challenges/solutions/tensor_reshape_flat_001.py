import torch


def flatten_batch(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(1)
