"""Hidden checks for tensor-fancy-index-001.

These are NOT shown to the learner. The tests cover repeated indices (selecting
the same row twice), a wider matrix, and the no-loops constraint. The
input_dtype param ensures the runner converts list args to float32, so the
solution must call .long() on the indices tensor internally.
"""

CHECKS = [
    {
        "name": "hidden-repeated-indices",
        "visibility": "hidden",
        "kind": "exact_allclose",
        "input": {
            "args": [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                [0, 2, 2, 3],
            ],
        },
        "params": {
            "expected": [[1.0, 2.0, 3.0], [7.0, 8.0, 9.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            "input_dtype": "float32",
        },
    },
    {
        "name": "hidden-repeated-shape",
        "visibility": "hidden",
        "kind": "expect_shape",
        "input": {
            "args": [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                [0, 2, 2, 3],
            ],
        },
        "params": {
            "shape": [4, 3],
            "input_dtype": "float32",
        },
    },
    {
        "name": "hidden-no-loops",
        "visibility": "hidden",
        "kind": "no_loops",
    },
]
