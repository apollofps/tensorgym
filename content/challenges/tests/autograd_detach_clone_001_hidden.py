CHECKS = [
    {"name": "hidden-values", "visibility": "hidden", "kind": "exact_allclose",
     "input": {"args": [[3.0, 4.0]]}, "params": {"expected": [6.0, 8.0]}},
    {"name": "hidden-no-grad", "visibility": "hidden", "kind": "expect_requires_grad",
     "input": {"args": [[3.0, 4.0]]}, "params": {"requires_grad": False}},
    {"name": "hidden-negative", "visibility": "hidden", "kind": "exact_allclose",
     "input": {"args": [[-1.0, 0.0, 5.0]]}, "params": {"expected": [-2.0, 0.0, 10.0]}},
    {"name": "hidden-output-is-tensor", "visibility": "hidden", "kind": "output_is_tensor",
     "input": {"args": [[1.0]]}},
]
