CHECKS = [
    {"name": "hidden-zero-and-three", "visibility": "hidden", "kind": "exact_allclose",
     "input": {"args": [[0.0, 3.0]]}, "params": {"expected": [0.0, 108.0]}},
    {"name": "hidden-shape", "visibility": "hidden", "kind": "expect_shape",
     "input": {"args": [[0.0, 3.0]]}, "params": {"shape": [2]}},
    {"name": "hidden-negative", "visibility": "hidden", "kind": "exact_allclose",
     "input": {"args": [[-2.0]]}, "params": {"expected": [48.0]}},
    {"name": "hidden-output-is-tensor", "visibility": "hidden", "kind": "output_is_tensor",
     "input": {"args": [[1.0]]}},
]
