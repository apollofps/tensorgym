CHECKS = [
    {"name": "hidden-zero-and-five", "visibility": "hidden", "kind": "exact_allclose",
     "input": {"args": [[0.0, 5.0]]}, "params": {"expected": [3.0, 13.0]}},
    {"name": "hidden-shape", "visibility": "hidden", "kind": "expect_shape",
     "input": {"args": [[0.0, 5.0]]}, "params": {"shape": [2]}},
    {"name": "hidden-negative", "visibility": "hidden", "kind": "exact_allclose",
     "input": {"args": [[-1.0, -3.0]]}, "params": {"expected": [1.0, -3.0]}},
    {"name": "hidden-output-is-tensor", "visibility": "hidden", "kind": "output_is_tensor",
     "input": {"args": [[2.0]]}},
]
