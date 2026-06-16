# TensorGym Agent Instructions

## Source of truth

Read `TENSORGYM_SPEC.md` before planning or implementing work. Treat it as the product, research, security, curriculum, and acceptance specification.

When a chat instruction conflicts with `TENSORGYM_SPEC.md`, stop and identify the conflict rather than silently changing the scope.

## Mandatory behavior

- Use current official documentation for version-sensitive technical content.
- Do not fabricate browsing, citations, tests, benchmark results, or verification.
- Pin and test a stable PyTorch environment before publishing curriculum.
- Only publish educational content marked `verified`.
- Never execute learner code inside the product API process.
- Keep the code executor isolated from the application server.
- Build and test one complete vertical slice before generating the full content library.
- Do not begin with authentication, social features, a large design system, or unrelated infrastructure.
- Do not claim completion unless acceptance criteria have been tested.

## Current execution order

1. Inspect the repository.
2. Confirm web browsing and Docker access.
3. Read the full specification.
4. Produce the requested architecture, threat model, schemas, research plan, milestones, and risks.
5. Implement the first vertical slice.
6. Run tests and report exact results.
7. Stop for review before expanding the challenge library.

## Change discipline

- Keep changes small and reviewable.
- Record important architectural decisions.
- Update documentation when behavior changes.
- Preserve learner code and state through prerequisite-learning flows.
- Keep curriculum content separate from application code.
