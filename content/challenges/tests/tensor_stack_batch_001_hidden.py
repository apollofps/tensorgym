"""Hidden checks for tensor-stack-batch-001.

These are NOT shown to the learner. The 3x3 matrices test a non-trivial size,
and the value check ensures the batch ordering is correct (a first, then b,
then c). The no_loops check catches manual construction.
"""

CHECKS = [
    {
        "name": "hidden-3x3-shape",
        "visibility": "hidden",
        "kind": "expect_shape",
        "input": {
            "args": [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
                [[19.0, 20.0, 21.0], [22.0, 23.0, 24.0], [25.0, 26.0, 27.0]],
            ],
        },
        "params": {"shape": [3, 3, 3]},
    },
    {
        "name": "hidden-3x3-values",
        "visibility": "hidden",
        "kind": "exact_allclose",
        "input": {
            "args": [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
                [[19.0, 20.0, 21.0], [22.0, 23.0, 24.0], [25.0, 26.0, 27.0]],
            ],
        },
        "params": {
            "expected": [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
                [[19.0, 20.0, 21.0], [22.0, 23.0, 24.0], [25.0, 26.0, 27.0]],
            ],
        },
    },
    {
        "name": "hidden-no-loops",
        "visibility": "hidden",
        "kind": "no_loops",
    },
]
