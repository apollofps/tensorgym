CHECKS = [
    {"name": "hidden-half", "visibility": "hidden", "kind": "exact_allclose",
     "input": {"args": [[0.5]]}, "params": {"expected": [0.9689]}},
    {"name": "hidden-shape", "visibility": "hidden", "kind": "expect_shape",
     "input": {"args": [[0.5]]}, "params": {"shape": [1]}},
    {"name": "hidden-pi", "visibility": "hidden", "kind": "exact_allclose",
     "input": {"args": [[1.7725]]}, "params": {"expected": [-3.5450]}},
    {"name": "hidden-output-is-tensor", "visibility": "hidden", "kind": "output_is_tensor",
     "input": {"args": [[1.0]]}},
]
