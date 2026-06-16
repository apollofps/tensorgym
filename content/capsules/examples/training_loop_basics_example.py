"""Interactive example for the 'Training Loop Basics' capsule. Verified in PyTorch 2.12.0 (CPU)."""

import torch
import torch.nn as nn

torch.manual_seed(42)

model = nn.Linear(2, 1)
x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
y = torch.tensor([[1.0], [2.0], [3.0], [4.0]])

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

print("=== Training Loop: zero_grad → forward → loss → backward → step ===\n")

for step in range(1, 4):
    # 1. Clear old gradients
    optimizer.zero_grad()

    # 2. Forward pass: feed input through the model
    predictions = model(x)

    # 3. Compute loss: how far are predictions from targets?
    loss = criterion(predictions, y)

    # 4. Backward pass: compute gradients for every parameter
    loss.backward()

    # 5. Optimizer step: update parameters using gradients
    optimizer.step()

    print(f"Step {step} | Loss: {loss.item():.6f}")

print("\n=== Parameters after training ===")
for name, param in model.named_parameters():
    print(f"  {name}: {param.data}")
