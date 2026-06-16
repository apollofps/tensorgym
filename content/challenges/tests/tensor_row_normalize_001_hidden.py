"""Hidden checks for tensor-row-normalize-001.

These are NOT shown to the learner. The API loads the ``CHECKS`` list and sends
it to the sandboxed executor on Submit. Each check uses the same declarative
``kind`` protocol implemented in services/executor/runner.py.

The non-square matrix is the important one: a solution that forgets
``keepdim=True`` produces a row-sum of shape (N,) which cannot broadcast against
an (N, M) tensor when M != N, surfacing the classic reduction-dimension mistake.
"""

CHECKS = [
    {
        "name": "hidden-non-square",
        "visibility": "hidden",
        "kind": "rows_sum_to_one",
        "input": [[1.0, 1.0, 1.0, 1.0], [2.0, 0.0, 1.0, 1.0], [0.5, 0.5, 0.5, 0.5]],
    },
    {
        "name": "hidden-non-square-shape",
        "visibility": "hidden",
        "kind": "same_shape",
        "input": [[1.0, 1.0, 1.0, 1.0], [2.0, 0.0, 1.0, 1.0], [0.5, 0.5, 0.5, 0.5]],
    },
    {
        "name": "hidden-single-row",
        "visibility": "hidden",
        "kind": "rows_sum_to_one",
        "input": [[3.0, 1.0]],
    },
    {
        "name": "hidden-no-loops",
        "visibility": "hidden",
        "kind": "no_loops",
    },
]
