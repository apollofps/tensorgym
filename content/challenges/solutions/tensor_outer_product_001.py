"""Reference solution for tensor-outer-product-001.

Verified in the pinned environment (PyTorch 2.12.0, CPU) by
tests/test_content_verification.py. Do not reveal this to the learner before
hint level 4.
"""

import torch


def outer_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a.unsqueeze(1) * b
