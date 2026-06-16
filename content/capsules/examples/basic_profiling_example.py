"""Interactive example for the 'Basic Profiling and Timing' capsule.

The learner runs this as-is in the sandbox to SEE how to time operations
correctly. Verified in PyTorch 2.12.0 (CPU). It only prints; it asserts
nothing, so the learner can edit and re-run freely.
"""

import time
import torch

# --- Method 1: Manual timing with perf_counter ---
x = torch.randn(1000, 1000)
y = torch.randn(1000, 1000)

# Warm-up (discard first run)
_ = torch.matmul(x, y)

times = []
for _ in range(10):
    start = time.perf_counter()
    _ = torch.matmul(x, y)
    end = time.perf_counter()
    times.append(end - start)

times.sort()
print(f"Manual timing (10 runs):")
print(f"  Median: {times[len(times)//2]*1000:.2f} ms")
print(f"  Min:    {min(times)*1000:.2f} ms")
print(f"  Max:    {max(times)*1000:.2f} ms")

# --- Method 2: torch.utils.benchmark ---
timer = torch.utils.benchmark.Timer(
    stmt="torch.matmul(x, y)",
    globals={"x": x, "y": y, "torch": torch},
)

result = timer.blocked_autorange(min_run_time=0.5)
print(f"\ntorch.utils.benchmark.Timer:")
print(f"  Median: {result.median*1000:.2f} ms")
print(f"  Mean:   {result.mean*1000:.2f} ms")
print(f"  Runs:   {result.number_per_run} x {result.times.__len__()} blocks")
