CHECKS = [
    {"name": "hidden-different-values", "visibility": "hidden", "kind": "exact_allclose",
     "input": {"args": [[0.0, 3.0]]},
     "params": {"expected": [2.0, 8.0]}},
    {"name": "hidden-shape-2", "visibility": "hidden", "kind": "expect_shape",
     "input": {"args": [[0.0, 3.0]]},
     "params": {"shape": [2]}},
    {"name": "hidden-output-is-tensor", "visibility": "hidden", "kind": "output_is_tensor",
     "input": {"args": [[1.0, 2.0]]}},
]
