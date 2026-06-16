import torch


def batch_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.bmm(a, b)
