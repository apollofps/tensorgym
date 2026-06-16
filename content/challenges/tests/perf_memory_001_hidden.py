"""Hidden checks for perf-memory-001.

Verifies in-place ReLU with a different input, checks shape is preserved,
and confirms the output is a tensor.
"""

CHECKS = [
    {
        "name": "hidden-mixed-values",
        "visibility": "hidden",
        "kind": "exact_allclose",
        "input": [[-5.0, -2.0, 0.0, 3.0, 7.0]],
        "params": {"expected": [[0.0, 0.0, 0.0, 3.0, 7.0]]},
    },
    {
        "name": "hidden-same-shape",
        "visibility": "hidden",
        "kind": "same_shape",
        "input": [[-5.0, -2.0, 0.0, 3.0, 7.0]],
    },
    {
        "name": "hidden-output-is-tensor",
        "visibility": "hidden",
        "kind": "output_is_tensor",
        "input": [[-5.0, -2.0, 0.0, 3.0, 7.0]],
    },
]
