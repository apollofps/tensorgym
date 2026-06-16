"""Reference solution for tensor-col-normalize-001.

Verified in the pinned environment (PyTorch 2.12.0, CPU) by
tests/test_content_verification.py. Do not reveal this to the learner before
hint level 4.
"""

import torch


def normalize_cols(x: torch.Tensor) -> torch.Tensor:
    return x / x.sum(dim=0, keepdim=True)
