# TensorGym Research Manifest

**Research date:** 2026-06-15
**Researcher tooling:** live web access (WebSearch / WebFetch), GitHub REST API, PyPI
JSON API, and execution inside the pinned Docker image. Every claim below was
fetched or executed this session — none is from model memory.

> Per `TENSORGYM_SPEC.md` and `agents.md`: do not fabricate sources, versions,
> tests, browsing claims, or benchmarks. This manifest records only verified facts.

---

## 1. Supported PyTorch version

| Field | Value | How verified |
|---|---|---|
| Latest stable PyTorch | **2.12.0** | GitHub Releases API: tag `v2.12.0`, `prerelease=false`, `published_at=2026-05-13T17:38:06Z` |
| Cross-check | docs `stable` → `2.12` | `https://docs.pytorch.org/docs/stable/...` redirects to `/docs/2.12/...` |
| Known discrepancy | `docs/versions.html` still labels **2.9.0** "stable" | Documentation-site lag; resolved in favour of 2.12.0 via the two sources above |

A first web search returned an inconsistent auto-summary (claimed both "2.12.0" and
"2.10.0"). It was resolved with the **machine-readable** GitHub Releases API, which
is not subject to summarizer error.

## 2. Pinned MVP environment

```yaml
content_environment:
  pytorch_version: "2.12.0"     # latest stable
  python_version: "3.12"        # torch 2.12.0 requires_python >=3.10 (PyPI); 3.12 chosen as conservative-stable
  cuda_version: null            # CPU-only MVP, labeled CPU-verified
  install_source: "https://download.pytorch.org/whl/cpu"
  reviewed_at: "2026-06-15"
```

- **Install source verified:** `torch-2.12.0+cpu-cp312-cp312-manylinux_2_28_aarch64.whl`
  (and `x86_64`) are present on the CPU wheel index.
- **Built & executed:** image `tensorgym-executor:py312-torch2.12.0-cpu` runs
  `python 3.12.13`, `torch 2.12.0+cpu` (confirmed inside the container with
  `--network none`).

## 3. Official installation source

PyTorch CPU wheel index: `https://download.pytorch.org/whl/cpu` (used in
`services/executor/Dockerfile`). `pip install torch==2.12.0` resolves to the
`2.12.0+cpu` local version from this index.

## 4. Core documentation pages read (this session)

| id | URL | Verified content |
|---|---|---|
| `pytorch-broadcasting-semantics` | https://docs.pytorch.org/docs/2.12/notes/broadcasting.html | "Starting at the trailing dimension, sizes must be equal, one of them is 1, or one does not exist." Sections: General / In-place / Backwards compatibility. |
| `torch-sum` | https://docs.pytorch.org/docs/2.12/generated/torch.sum.html | Signature `torch.sum(input, dim, keepdim=False, *, dtype=None) -> Tensor`; `keepdim=True` retains the reduced dim with size 1, else it is removed. |
| `pytorch-release-2-12` | https://github.com/pytorch/pytorch/releases/tag/v2.12.0 | 2.12.0 is the latest stable release. |

These are recorded machine-readably in `content/sources/source-registry.yaml`.

## 5. APIs used in the MVP slice

`torch.tensor`, `Tensor.sum(dim, keepdim=...)`, broadcasting division (`/`),
plus tensor-metadata introspection (`shape`, `dim`, `dtype`, `device`,
`requires_grad`, `grad_fn`, `numel`, `element_size`, `is_contiguous`).

## 6. Deprecated APIs to avoid

None relevant to this slice. (The slice uses only stable, non-deprecated tensor
ops. The content-audit command in a later milestone will scan for deprecations.)

## 7. Verification environment

All content is verified by `tests/test_content_verification.py`, which runs the
reference solution, the capsule example, and the warm-up solution **inside the
pinned Docker image**. CI-equivalent command: `make test`.

## 8. Behavior verified by execution (pinned env)

```
x.sum(dim=1, keepdim=True).shape == (N, 1)
x.sum(dim=1).shape            == (N,)
x / x.sum(dim=1, keepdim=True) -> each row sums to 1.0
x / x.sum(dim=1)  on (2,3)     -> RuntimeError: The size of tensor a (3) must
                                  match the size of tensor b (2) at non-singleton dimension 1
```

The last line is the exact failure the targeted-feedback path classifies as a
**Broadcasting error**.

## 9. Open questions / uncertain areas

- **Docs-site version lag:** `versions.html` lags the actual release; treat the
  GitHub release + `stable` redirect as authoritative.
- **GPU / MPS / CUDA behavior:** not verified. MVP is CPU-only and labeled so. Any
  device-specific content must be re-verified before publishing.
- **Compiler behavior (`torch.compile`, graph breaks):** out of scope for this
  slice; requires separate verification in the pinned env before any capsule ships.

## 10. Scope honesty

This manifest covers only the **one** beginner challenge built in Milestone 1
(`tensor-row-normalize-001`) and its capsule/warm-up. The full library (30
challenges, 15 capsules) is **not** yet researched or generated, per instructions.
