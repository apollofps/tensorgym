"""Reference solution for nn-training-step-001.

Verified in PyTorch 2.12.0, CPU.
"""

import torch
import torch.nn as nn


def train_one_step(x: torch.Tensor, y: torch.Tensor) -> float:
    torch.manual_seed(42)
    model = nn.Linear(3, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
    return loss.item()
