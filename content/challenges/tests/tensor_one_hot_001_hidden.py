CHECKS = [
    {"name": "hidden-4-classes", "visibility": "hidden", "kind": "exact_allclose",
     "input": {"args": [[3, 0, 1, 2], 4]},
     "params": {"expected": [[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                "input_dtype": "int64"}},
    {"name": "hidden-shape", "visibility": "hidden", "kind": "expect_shape",
     "input": {"args": [[3, 0, 1, 2], 4]},
     "params": {"shape": [4, 4], "input_dtype": "int64"}},
    {"name": "hidden-dtype", "visibility": "hidden", "kind": "expect_dtype",
     "input": {"args": [[0, 1], 5]},
     "params": {"dtype": "float32", "input_dtype": "int64"}},
    {"name": "hidden-no-loops", "visibility": "hidden", "kind": "no_loops"},
]
