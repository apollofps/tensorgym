import torch


def add_bias(matrix: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return matrix + bias
