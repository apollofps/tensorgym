"""Hidden checks for perf-view-vs-copy-001.

Verifies correctness on a different shape and confirms the output shape is 1-D.
"""

CHECKS = [
    {
        "name": "hidden-3x2-values",
        "visibility": "hidden",
        "kind": "exact_allclose",
        "input": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        "params": {"expected": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]},
    },
    {
        "name": "hidden-3x2-shape",
        "visibility": "hidden",
        "kind": "expect_shape",
        "input": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        "params": {"shape": [6]},
    },
]
