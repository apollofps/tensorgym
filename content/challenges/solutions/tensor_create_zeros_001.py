import torch


def make_zeros(shape, dtype_name: str) -> torch.Tensor:
    if isinstance(shape, torch.Tensor):
        shape = tuple(shape.long().tolist())
    return torch.zeros(shape, dtype=getattr(torch, dtype_name))
