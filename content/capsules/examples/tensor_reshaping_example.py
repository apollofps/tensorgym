"""Interactive example for the 'Reshaping Tensors' capsule.

Run this to see reshape, view, transpose, permute, and flatten in action.
Verified in PyTorch 2.12.0 (CPU).
"""

import torch

# Start with a 1-D tensor of 12 elements
x = torch.arange(12, dtype=torch.float32)
print("original shape:", tuple(x.shape))              # (12,)

# reshape to 3x4
r = x.reshape(3, 4)
print("reshape(3, 4): ", tuple(r.shape))              # (3, 4)
print(r)

# reshape with -1 (infer one dimension)
r2 = x.reshape(2, -1)
print("\nreshape(2, -1):", tuple(r2.shape))            # (2, 6)

# view — same as reshape but requires contiguous memory
v = x.view(4, 3)
print("view(4, 3):    ", tuple(v.shape))              # (4, 3)

# transpose — swap two dimensions
m = torch.rand(3, 4)
t = m.transpose(0, 1)
print("\noriginal:     ", tuple(m.shape))              # (3, 4)
print("transpose(0,1):", tuple(t.shape))              # (4, 3)

# permute — rearrange all dimensions
batch = torch.rand(2, 3, 4)
p = batch.permute(0, 2, 1)
print("\n3-D original: ", tuple(batch.shape))          # (2, 3, 4)
print("permute(0,2,1):", tuple(p.shape))              # (2, 4, 3)

# flatten — collapse dimensions
images = torch.rand(2, 3, 4, 4)
print("\nimages shape:  ", tuple(images.shape))        # (2, 3, 4, 4)
flat = images.flatten(1)
print("flatten(1):    ", tuple(flat.shape))           # (2, 48)
flat_all = images.flatten()
print("flatten():     ", tuple(flat_all.shape))       # (96,)
