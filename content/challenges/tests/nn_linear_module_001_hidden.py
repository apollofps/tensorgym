"""Hidden checks for nn-linear-module-001.

These are NOT shown to the learner. They verify the network architecture produces
the correct output shape for various batch sizes and that the output is a tensor.
"""

CHECKS = [
    {
        "name": "hidden-single-sample-shape",
        "visibility": "hidden",
        "kind": "expect_shape",
        "input": [[1.0, 2.0, 3.0, 4.0]],
        "params": {"shape": [1, 2]},
    },
    {
        "name": "hidden-output-is-tensor",
        "visibility": "hidden",
        "kind": "output_is_tensor",
        "input": [[1.0, 0.0, -1.0, 2.0]],
    },
]
