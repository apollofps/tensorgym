CHECKS = [
    {"name": "hidden-shape-2x5", "visibility": "hidden", "kind": "expect_shape",
     "input": {"args": [[2, 5], "int32"]}, "params": {"shape": [2, 5]}},
    {"name": "hidden-dtype-int32", "visibility": "hidden", "kind": "expect_dtype",
     "input": {"args": [[2, 5], "int32"]}, "params": {"dtype": "int32"}},
    {"name": "hidden-1d", "visibility": "hidden", "kind": "expect_shape",
     "input": {"args": [[10], "float32"]}, "params": {"shape": [10]}},
    {"name": "hidden-no-loops", "visibility": "hidden", "kind": "no_loops"},
]
