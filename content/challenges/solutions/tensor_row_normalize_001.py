"""Reference solution for tensor-row-normalize-001.

Verified in the pinned environment (PyTorch 2.12.0, CPU) by
tests/test_content_verification.py. Do not reveal this to the learner before
hint level 4.
"""

import torch


def normalize_rows(x: torch.Tensor) -> torch.Tensor:
    # Sum along dim=1 (across each row) and keep the dimension so the result has
    # shape (N, 1) and broadcasts back against the original (N, M) tensor.
    return x / x.sum(dim=1, keepdim=True)
