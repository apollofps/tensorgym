"""Hidden checks for nn-skip-connection-001.

These are NOT shown to the learner. They verify that the residual connection
preserves the input shape and that the output is a proper tensor.
"""

CHECKS = [
    {
        "name": "hidden-single-sample-shape",
        "visibility": "hidden",
        "kind": "expect_shape",
        "input": [[1.0, 2.0, 3.0, 4.0]],
        "params": {"shape": [1, 4]},
    },
    {
        "name": "hidden-output-is-tensor",
        "visibility": "hidden",
        "kind": "output_is_tensor",
        "input": [[0.0, 0.0, 0.0, 0.0]],
    },
    {
        "name": "hidden-batch-three-shape",
        "visibility": "hidden",
        "kind": "same_shape",
        "input": [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
    },
]
