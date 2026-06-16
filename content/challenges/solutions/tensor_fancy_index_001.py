"""Reference solution for tensor-fancy-index-001.

Verified in the pinned environment (PyTorch 2.12.0, CPU) by
tests/test_content_verification.py. Do not reveal this to the learner before
hint level 4.
"""

import torch


def select_rows(data: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return data[indices.long()]
