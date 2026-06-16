"""Hidden checks for perf-batch-matmul-001.

Verifies identity-batch multiplication, output shape, and no-loops constraint.
"""

CHECKS = [
    {
        "name": "hidden-identity-batch",
        "visibility": "hidden",
        "kind": "exact_allclose",
        "input": {
            "args": [
                [[[1.0, 0.0], [0.0, 1.0]]],
                [[[3.0, 4.0], [5.0, 6.0]]],
            ]
        },
        "params": {"expected": [[[3.0, 4.0], [5.0, 6.0]]]},
    },
    {
        "name": "hidden-identity-shape",
        "visibility": "hidden",
        "kind": "expect_shape",
        "input": {
            "args": [
                [[[1.0, 0.0], [0.0, 1.0]]],
                [[[3.0, 4.0], [5.0, 6.0]]],
            ]
        },
        "params": {"shape": [1, 2, 2]},
    },
    {
        "name": "hidden-no-loops",
        "visibility": "hidden",
        "kind": "no_loops",
    },
]
