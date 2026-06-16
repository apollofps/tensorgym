"""Reference solution for nn-custom-loss-001.

Verified in PyTorch 2.12.0, CPU.
"""

import torch


def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float) -> torch.Tensor:
    error = pred - target
    abs_error = error.abs()
    quadratic = torch.clamp(abs_error, max=delta)
    linear = abs_error - quadratic
    loss = 0.5 * quadratic ** 2 + delta * linear
    return loss.mean()
