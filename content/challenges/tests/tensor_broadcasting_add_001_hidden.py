CHECKS = [
    {"name": "hidden-larger", "visibility": "hidden", "kind": "exact_allclose",
     "input": {"args": [[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], [100.0, 200.0]]},
     "params": {"expected": [[101.0, 201.0], [102.0, 202.0], [103.0, 203.0]]}},
    {"name": "hidden-shape", "visibility": "hidden", "kind": "expect_shape",
     "input": {"args": [[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], [100.0, 200.0]]},
     "params": {"shape": [3, 2]}},
    {"name": "hidden-no-loops", "visibility": "hidden", "kind": "no_loops"},
]
