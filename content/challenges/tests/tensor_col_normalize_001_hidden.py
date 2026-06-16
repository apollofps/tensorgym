"""Hidden checks for tensor-col-normalize-001.

These are NOT shown to the learner. The API loads the ``CHECKS`` list and sends
it to the sandboxed executor on Submit. Each check uses the same declarative
``kind`` protocol implemented in services/executor/runner.py.

The non-square matrix catches solutions that confuse dim=0 and dim=1. The
single-column edge case ensures the solution generalises beyond wide matrices.
"""

CHECKS = [
    {
        "name": "hidden-non-square",
        "visibility": "hidden",
        "kind": "cols_sum_to_one",
        "input": [[1.0, 1.0, 1.0, 1.0], [2.0, 0.0, 1.0, 1.0], [0.5, 0.5, 0.5, 0.5]],
    },
    {
        "name": "hidden-non-square-shape",
        "visibility": "hidden",
        "kind": "same_shape",
        "input": [[1.0, 1.0, 1.0, 1.0], [2.0, 0.0, 1.0, 1.0], [0.5, 0.5, 0.5, 0.5]],
    },
    {
        "name": "hidden-single-col",
        "visibility": "hidden",
        "kind": "cols_sum_to_one",
        "input": [[3.0], [1.0]],
    },
    {
        "name": "hidden-no-loops",
        "visibility": "hidden",
        "kind": "no_loops",
    },
]
