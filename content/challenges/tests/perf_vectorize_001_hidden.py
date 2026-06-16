"""Hidden checks for perf-vectorize-001.

These are NOT shown to the learner. The executor runs them on Submit to verify
the solution handles different inputs and avoids Python loops.
"""

CHECKS = [
    {
        "name": "hidden-negative-input",
        "visibility": "hidden",
        "kind": "exact_allclose",
        "input": {"args": [[[0.0, -1.0]], 5.0]},
        "params": {"expected": [[5.0, 4.0]]},
    },
    {
        "name": "hidden-no-loops",
        "visibility": "hidden",
        "kind": "no_loops",
    },
    {
        "name": "hidden-same-shape",
        "visibility": "hidden",
        "kind": "same_shape",
        "input": {"args": [[[0.0, -1.0]], 5.0]},
    },
]
