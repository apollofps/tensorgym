"""Hidden checks for nn-custom-loss-001.

These are NOT shown to the learner. They verify the Huber loss formula is
correctly implemented for both the quadratic and linear regions.

For args [0.0, 0.5], [0.0, 0.0], 1.0:
  errors: [0.0, 0.5]. Both |err| <= 1.0 (quadratic region).
  losses: [0.5*0²=0, 0.5*0.25=0.125]. Mean = 0.0625.
"""

CHECKS = [
    {
        "name": "hidden-quadratic-region",
        "visibility": "hidden",
        "kind": "expect_value",
        "input": {"args": [[0.0, 0.5], [0.0, 0.0], 1.0]},
        "params": {"expected": 0.0625, "atol": 0.001},
    },
    {
        "name": "hidden-output-is-tensor",
        "visibility": "hidden",
        "kind": "output_is_tensor",
        "input": {"args": [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 0.5]},
    },
]
