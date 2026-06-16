"""Hidden checks for tensor-outer-product-001.

These are NOT shown to the learner. The zero-element vector tests that
broadcasting works correctly when one input contains zero, and the shape check
catches solutions that accidentally transpose the result.
"""

CHECKS = [
    {
        "name": "hidden-with-zero",
        "visibility": "hidden",
        "kind": "exact_allclose",
        "input": {"args": [[1.0, 0.0, 2.0], [3.0, 4.0]]},
        "params": {"expected": [[3.0, 4.0], [0.0, 0.0], [6.0, 8.0]]},
    },
    {
        "name": "hidden-shape",
        "visibility": "hidden",
        "kind": "expect_shape",
        "input": {"args": [[1.0, 0.0, 2.0], [3.0, 4.0]]},
        "params": {"shape": [3, 2]},
    },
    {
        "name": "hidden-no-loops",
        "visibility": "hidden",
        "kind": "no_loops",
    },
]
