"""Hidden checks for nn-train-eval-toggle-001.

These are NOT shown to the learner. They verify that the model produces correct
output shapes for different batch sizes and that eval mode is active (Dropout
disabled means deterministic output).
"""

CHECKS = [
    {
        "name": "hidden-batch-shape",
        "visibility": "hidden",
        "kind": "expect_shape",
        "input": [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        "params": {"shape": [2, 4]},
    },
    {
        "name": "hidden-output-is-tensor",
        "visibility": "hidden",
        "kind": "output_is_tensor",
        "input": [[0.0, 0.0, 0.0, 0.0]],
    },
]
