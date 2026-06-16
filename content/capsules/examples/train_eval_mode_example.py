"""Interactive example for the 'Train Mode vs Eval Mode' capsule.

The learner runs this as-is in the sandbox to SEE how Dropout behaves
differently in train vs eval mode. Verified in PyTorch 2.12.0 (CPU).
It only prints; it asserts nothing, so the learner can edit and re-run freely.
"""

import torch
import torch.nn as nn

torch.manual_seed(42)

x = torch.ones(1, 10)

model = nn.Sequential(
    nn.Linear(10, 10),
    nn.Dropout(p=0.5),
)

# Train mode (default): Dropout randomly zeroes ~50% of values
model.train()
out_train1 = model(x)
out_train2 = model(x)
print("Train mode, run 1:", out_train1.detach().tolist())
print("Train mode, run 2:", out_train2.detach().tolist())
print("Same output?", torch.equal(out_train1, out_train2))

# Eval mode: Dropout is a no-op, output is deterministic
model.eval()
out_eval1 = model(x)
out_eval2 = model(x)
print("\nEval mode, run 1: ", out_eval1.detach().tolist())
print("Eval mode, run 2: ", out_eval2.detach().tolist())
print("Same output?", torch.equal(out_eval1, out_eval2))

print("\nmodel.training:", model.training)
