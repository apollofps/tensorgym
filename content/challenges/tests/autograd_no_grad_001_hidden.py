CHECKS = [
    {"name": "hidden-zeros-ones", "visibility": "hidden", "kind": "exact_allclose",
     "input": {"args": [[0.0, 1.0]]}, "params": {"expected": [0.0, 2.0]}},
    {"name": "hidden-no-grad", "visibility": "hidden", "kind": "expect_requires_grad",
     "input": {"args": [[0.0, 1.0]]}, "params": {"requires_grad": False}},
    {"name": "hidden-negative", "visibility": "hidden", "kind": "exact_allclose",
     "input": {"args": [[-1.0, -2.0]]}, "params": {"expected": [0.0, 2.0]}},
    {"name": "hidden-output-is-tensor", "visibility": "hidden", "kind": "output_is_tensor",
     "input": {"args": [[5.0]]}},
]
