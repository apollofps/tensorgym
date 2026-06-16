"""Hidden checks for tensor-transpose-dims-001."""

CHECKS = [
    {"name": "hidden-shape-3x5x2", "visibility": "hidden", "kind": "expect_shape",
     "input": [
         [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]],
         [[11.0, 12.0, 13.0, 14.0, 15.0], [16.0, 17.0, 18.0, 19.0, 20.0]],
         [[21.0, 22.0, 23.0, 24.0, 25.0], [26.0, 27.0, 28.0, 29.0, 30.0]],
     ],
     "params": {"shape": [3, 5, 2]}},
    {"name": "hidden-2d", "visibility": "hidden", "kind": "expect_shape",
     "input": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
     "params": {"shape": [3, 2]}},
    {"name": "hidden-no-loops", "visibility": "hidden", "kind": "no_loops"},
]
