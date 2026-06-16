CHECKS = [
    {"name": "hidden-different-values", "visibility": "hidden", "kind": "exact_allclose",
     "input": {"args": [[1.0, 0.0, 5.0]]},
     "params": {"expected": [2.0, 0.0, 10.0]}},
    {"name": "hidden-shape-3", "visibility": "hidden", "kind": "expect_shape",
     "input": {"args": [[1.0, 0.0, 5.0]]},
     "params": {"shape": [3]}},
    {"name": "hidden-output-is-tensor", "visibility": "hidden", "kind": "output_is_tensor",
     "input": {"args": [[2.0, 3.0]]}},
]
