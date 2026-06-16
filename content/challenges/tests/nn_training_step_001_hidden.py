"""Hidden checks for nn-training-step-001.

These are NOT shown to the learner. They verify the training step runs without
error on different input shapes, confirming the full zero_grad → forward → loss
→ backward → step pipeline works correctly.
"""

CHECKS = [
    {
        "name": "hidden-different-batch",
        "visibility": "hidden",
        "kind": "runs_without_error",
        "input": {
            "args": [
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
                [[0.5], [1.0], [1.5]],
            ]
        },
    },
    {
        "name": "hidden-single-sample",
        "visibility": "hidden",
        "kind": "runs_without_error",
        "input": {
            "args": [
                [[1.0, 2.0, 3.0]],
                [[4.0]],
            ]
        },
    },
]
