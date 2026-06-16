"""Hidden checks for nn-param-count-001.

These are NOT shown to the learner. They verify parameter counting with different
network dimensions.

For args [10, 5, 3]:
  Linear(10, 5): 10*5 + 5 = 55
  Linear(5, 5):  5*5 + 5  = 30
  Linear(5, 3):  5*3 + 3  = 18
  Total: 55 + 30 + 18 = 103
"""

CHECKS = [
    {
        "name": "hidden-different-sizes",
        "visibility": "hidden",
        "kind": "expect_value",
        "input": {"args": [10, 5, 3]},
        "params": {"expected": 103},
    },
    {
        "name": "hidden-minimal-network",
        "visibility": "hidden",
        "kind": "expect_value",
        "input": {"args": [1, 1, 1]},
        "params": {"expected": 6},
    },
]
