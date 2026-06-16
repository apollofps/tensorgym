CHECKS = [
    {"name": "hidden-non-square", "visibility": "hidden", "kind": "exact_allclose",
     "input": {"args": [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]},
     "params": {"expected": [[1.5], [3.5], [5.5]]}},
    {"name": "hidden-shape-3x1", "visibility": "hidden", "kind": "expect_shape",
     "input": {"args": [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]},
     "params": {"shape": [3, 1]}},
    {"name": "hidden-output-is-tensor", "visibility": "hidden", "kind": "output_is_tensor",
     "input": {"args": [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]}},
]
