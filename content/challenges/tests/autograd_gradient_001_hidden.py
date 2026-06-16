CHECKS = [
    {"name": "hidden-zeros-and-threes", "visibility": "hidden", "kind": "exact_allclose",
     "input": {"args": [[0.0, 3.0]]}, "params": {"expected": [2.0, 29.0]}},
    {"name": "hidden-shape", "visibility": "hidden", "kind": "expect_shape",
     "input": {"args": [[0.0, 3.0]]}, "params": {"shape": [2]}},
    {"name": "hidden-negative", "visibility": "hidden", "kind": "exact_allclose",
     "input": {"args": [[-1.0, -2.0]]}, "params": {"expected": [5.0, 14.0]}},
    {"name": "hidden-output-is-tensor", "visibility": "hidden", "kind": "output_is_tensor",
     "input": {"args": [[1.0]]}},
]
