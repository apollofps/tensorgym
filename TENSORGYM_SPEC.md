# Master Build Prompt: TensorGym

## Role

You are a senior full-stack engineer, ML systems engineer, security-conscious platform engineer, technical researcher, and instructional designer.

Build a local-first web application called **TensorGym** that teaches PyTorch through active problem solving rather than passive reading.

The product should feel like a combination of:

- LeetCode for coding challenges
- Brilliant for interactive explanations
- Exercism for progressive feedback
- A lightweight tensor and profiler laboratory
- A practical preparation environment for ML systems and LLM inference engineering

Do not build a conventional tutorial website filled with long chapters.

The platform must use a:

> Challenge-first, just-in-time, documentation-grounded learning model.

The learner should spend most of their time writing, running, debugging, predicting, profiling, and improving real PyTorch code.

---

# Critical Research Requirement

Before creating curriculum content, use live web research to identify the current stable PyTorch version and read the relevant official documentation.

Treat model memory only as a way to formulate search questions. Do not treat model memory as the final source of technical truth.

Do not generate PyTorch theory, API behavior, examples, performance claims, compiler behavior, or version-specific guidance solely from prior model knowledge.

Before publishing educational content:

1. Search the web.
2. Read the current official documentation.
3. verify relevant API signatures and behavior.
4. Test examples in the pinned environment.
5. Record sources and verification metadata.
6. Publish only content marked as verified.

If web access is unavailable:

- Do not invent current versions.
- Do not claim content has been researched.
- Do not fabricate citations.
- Do not publish version-sensitive content.
- Continue only with infrastructure, schemas, and clearly marked unverified content scaffolding.
- Produce a list of blocked research tasks.

---

# Learner Profile

Assume the primary learner:

- Has general Python programming experience
- May understand machine learning at a surface level
- May have briefly used PyTorch
- Does not have dependable PyTorch knowledge
- May not understand tensor semantics, autograd, modules, GPU execution, profiling, transformers, or compilation
- Does not want to spend most of their time reading
- Learns best by solving problems and receiving targeted support
- Ultimately wants to move toward:
  - ML systems
  - LLM inference
  - model serving
  - inference platforms
  - compiler-aware PyTorch
- Does not currently need deep accelerator compiler, LLVM, MLIR, or CUDA backend development

The platform must work for both:

1. A beginner who needs prerequisite teaching
2. A learner who already knows the concept and wants to begin the challenge immediately

Do not assume the learner knows an API merely because it appears in starter code.

---

# Core Learning Philosophy

Use this learning loop:

```text
Diagnose knowledge
        ↓
Present challenge
        ↓
Learner attempts or requests prerequisite help
        ↓
Provide targeted concept material
        ↓
Learner runs an interactive example
        ↓
Learner completes a tiny warm-up
        ↓
Return to original challenge
        ↓
Run tests and provide feedback
        ↓
Recommend one targeted next action
```

The learner should never be forced to read a long lesson before coding.

Use progressive disclosure:

- Show the challenge first when appropriate
- Display required prerequisites
- Let the learner select “I don’t know this yet”
- Provide concise just-in-time learning material
- Preserve the learner’s code
- Return the learner to the exact challenge after preparation

---

# Product Goals

The platform should teach the learner to:

1. Read and manipulate tensors confidently
2. Reason about shapes, dimensions, strides, dtypes, and devices
3. Understand autograd and gradient flow
4. Build and debug `nn.Module` code
5. Write correct training and evaluation loops
6. Understand transformer inference fundamentals
7. Understand prefill, decode, attention, masking, and KV caching
8. Profile PyTorch programs
9. Detect performance anti-patterns
10. Understand eager versus compiled execution
11. Use `torch.compile` practically
12. Recognize graph breaks, guards, and recompilation
13. Prepare for later work with vLLM, SGLang, and inference-serving systems

---

# Non-Goals for the Initial MVP

Do not initially build:

- Social networking
- Public leaderboards
- Payments
- Multi-tenancy
- Complex authentication
- A marketplace
- Full Kubernetes execution
- Full GPU fleet scheduling
- vLLM integration
- SGLang integration
- LLVM lessons
- MLIR lessons
- Accelerator backend development
- AI-generated unreviewed curriculum
- A large design system
- Decorative gamification

Design extension points for future capabilities, but prioritize a working learning experience.

---

# Main Platform Modes

## 1. Coding Challenges

Create LeetCode-style exercises where the learner implements real PyTorch code.

Examples:

- Create a tensor with a required shape and dtype
- Reshape a tensor correctly
- Normalize each row without loops
- Implement broadcasting correctly
- Implement linear regression using tensor operations
- Write a custom loss function
- Implement a custom `nn.Module`
- Write a manual training loop
- Implement masked softmax
- Implement scaled dot-product attention
- Build a basic KV cache
- Fix GPU synchronization inside a loop
- Make a function compatible with `torch.compile`

Each challenge must include:

- Title
- Difficulty
- Challenge type
- Concepts taught
- Concepts tested
- Required prerequisites
- Optional prerequisites
- Estimated completion time
- Concise problem statement
- Starter code
- Function signature
- Constraints
- Visible examples
- Visible tests
- Hidden tests
- Progressive hints
- Linked Concept Capsules
- Reference solution
- Solution explanation
- Common mistakes
- Follow-up challenge recommendations
- Source metadata
- Verification metadata

## 2. Debugging Challenges

Show broken PyTorch code and ask the learner to repair it.

Examples:

- Shape mismatch
- Incorrect broadcasting
- Wrong reduction dimension
- Detached computation graph
- Missing gradient reset
- Incorrect train/eval mode
- Device mismatch
- In-place autograd failure
- Retained graph or memory leak
- Incorrect attention mask
- Excessive CPU–GPU synchronization
- Repeated recompilation
- `torch.compile` graph break

The learner must edit code rather than select a multiple-choice answer.

## 3. Prediction Challenges

Ask the learner to predict:

- Output shape
- Output values
- Dtype
- Device
- Stride
- Contiguity
- Gradient
- Broadcasting behavior
- Whether an operation is differentiable
- Whether a graph break may occur
- Whether recompilation may occur
- Approximate memory behavior

After submission, visualize what actually happened.

## 4. Guided Drills

Provide small one-to-three-minute exercises for learners who are not ready for a full challenge.

Examples:

- Choose the correct reduction dimension
- Repair one line
- Predict one output shape
- Replace one Python loop
- Move one tensor to the correct device
- Preserve a dimension using `keepdim`
- Identify why a gradient is missing

## 5. Tensor Laboratory

Create an interactive environment where the learner can:

- Execute arbitrary PyTorch snippets
- Inspect tensor shape
- Inspect rank
- Inspect dtype
- Inspect device
- Inspect strides
- Inspect contiguity
- Inspect `requires_grad`
- Inspect `grad_fn`
- Estimate tensor memory
- Visualize broadcasting
- View simplified autograd relationships
- Measure execution time
- Compare CPU and GPU when GPU support exists
- Inspect CUDA allocated and reserved memory when available
- Compare eager execution with `torch.compile`
- Save experiments

## 6. Mini-Projects

Create a progressive project ladder:

1. Linear regression using tensor operations
2. Small neural network with a manual training loop
3. Classifier with validation and checkpointing
4. Attention from basic tensor operations
5. Minimal autoregressive generation
6. Generation with KV caching
7. Eager versus `torch.compile` benchmark
8. Batched model inference API
9. Concurrent load testing
10. Future vLLM integration

Each project should contain milestones and automated checks.

---

# Beginner Onboarding

Provide an optional diagnostic with approximately 8–12 short tasks covering:

- Python functions
- Python indexing
- Basic NumPy-like operations
- Tensor shapes
- Matrix multiplication
- Basic neural-network vocabulary
- Gradients
- Device awareness
- PyTorch familiarity

The diagnostic must not block access to the platform.

Based on the diagnostic, generate an initial profile such as:

```text
Python foundations: sufficient
Tensor shapes: beginner
Broadcasting: not demonstrated
Autograd: not demonstrated
nn.Module: surface familiarity
GPU execution: not started
```

Allow the learner to:

- Skip the diagnostic
- Override the result
- Start anywhere
- Re-run the diagnostic later

---

# Challenge Readiness System

Every challenge must declare:

- Concepts taught
- Concepts tested
- Required prerequisites
- Helpful optional prerequisites

Before starting a challenge, show:

```text
Required knowledge

✓ Tensor creation
✓ Basic indexing
? Broadcasting
? Reduction dimensions
```

Provide these actions:

- Start challenge
- Quick refresher
- I don’t know this yet

Do not require the learner to pass a prerequisite quiz.

---

# Concept Capsules

Create concise just-in-time study units called **Concept Capsules**.

A Concept Capsule should take approximately 2–5 minutes and teach one narrow topic.

Examples:

- What tensor shape means
- How dimensions work
- Broadcasting rules
- Reduction dimensions
- Difference between `view` and `reshape`
- Why gradients accumulate
- Train mode versus evaluation mode
- What a KV cache stores
- What `torch.compile` does
- What a graph break is

Each capsule must contain:

1. A concise explanation
2. One visual or shape example
3. One executable PyTorch example
4. One common mistake
5. One tiny prediction or coding check
6. Official source links
7. Supported version information
8. Verification status
9. A button to return to the original challenge

Prefer 100–300 words of explanation.

Do not generate long generic essays.

Do not reveal the solution to the parent challenge.

---

# Prerequisite Recovery Flow

When the learner does not know a required concept, use:

```text
Original challenge
    ↓
Missing concept identified
    ↓
Concept Capsule
    ↓
Interactive example
    ↓
Tiny warm-up
    ↓
Return to original challenge
```

Preserve:

- Editor contents
- Current challenge
- Test history
- Scroll position when practical
- Hint level
- Unsubmitted work

Example:

```text
Challenge: Normalize every row independently

Missing concepts:
- reduction dimensions
- keepdim
- broadcasting

Preparation:
1. Capsule: Reduction dimensions
2. Broadcasting visualizer
3. Warm-up: Calculate row sums
4. Return to the original challenge
```

---

# Challenge Support Panel

Include a **Learn** panel in the challenge workspace.

Actions:

- Explain the prerequisite
- Show concise API reference
- Show a smaller example
- Visualize tensor shapes
- Give me a warm-up problem
- Explain this error
- Give me a hint
- Show the full solution

These must be separate actions.

Progressive help levels:

## Level 0: Concept reminder

Explain the underlying concept without applying it directly to the answer.

## Level 1: Directional hint

Point toward the relevant operation or dimension.

## Level 2: Structured hint

Describe the steps without exact final code.

## Level 3: Partial code

Provide pseudocode or a limited fragment.

## Level 4: Full walkthrough

Provide the complete solution with explanation.

Track which help levels were used.

---

# Adaptive Learning Engine

Track concept mastery independently.

Example:

```text
tensor shapes: 85%
broadcasting: 45%
autograd: 20%
device management: 65%
attention masks: not started
```

Use evidence such as:

- Attempts
- Tests passed
- Time taken
- Hints used
- Capsules opened
- Warm-ups completed
- Mistake categories
- Performance on later related problems
- Whether the learner solved a transfer problem

Opening a capsule is not proof of mastery.

After each completed activity, recommend exactly one next action:

- Advance to a harder challenge
- Complete a prerequisite capsule
- Solve a targeted drill
- Revisit a weak concept
- Start a mini-project milestone

Use deterministic rules in the MVP.

Design an interface for future AI-assisted adaptation, but do not make AI-generated decisions mandatory for the first release.

---

# Mistake Taxonomy

Classify errors using a structured taxonomy:

- Shape error
- Dimension-selection error
- Broadcasting error
- Dtype error
- Device error
- Numerical error
- Gradient error
- State-management error
- Data-loader error
- Performance anti-pattern
- Forbidden API usage
- Timeout
- Runtime exception
- Compilation error
- Graph-break issue
- Recompilation issue

Map each mistake to:

- A concise explanation
- A relevant Concept Capsule
- A tiny remediation drill
- A future review challenge

Example:

```text
Detected issue:
Your reduction removed dimension 1, producing shape [8] instead of [8, 1].

Recommended preparation:
- Capsule: Reduction dimensions and keepdim
- Drill: Predict three reduction output shapes
```

---

# Feedback Principles

Good feedback:

> Your numerical values are correct, but the output has shape `[8]` instead of `[8, 1]`. The reduced dimension was removed, so the result cannot broadcast against the original tensor.

Poor feedback:

> Wrong answer.

Feedback should first explain:

1. What was observed
2. What concept is involved
3. What the learner should inspect
4. The smallest useful next step

Do not reveal final code immediately.

---

# Visualizations

Create technically useful visualizations.

## Tensor Inspector

Show:

- Shape
- Rank
- Dtype
- Device
- Stride
- Contiguity
- `requires_grad`
- `grad_fn`
- Approximate memory size

## Shape-Flow Viewer

Example:

```text
[batch, sequence, hidden]
        ×
[hidden, vocabulary]
        ↓
[batch, sequence, vocabulary]
```

## Broadcasting Viewer

Show:

- Shapes aligned from the trailing dimension
- Compatible dimensions
- Expanded dimensions
- Incompatible dimensions

## Autograd Viewer

Show a simplified backward graph for small examples.

## Performance Viewer

Show:

- Median latency
- P95 latency
- Throughput
- Memory use
- Warm versus cold execution
- Eager versus compiled execution

Do not add visualizations that do not help understanding.

---

# Curriculum Structure

Represent the curriculum as a dependency graph rather than only a linear list.

## Level 0: PyTorch Orientation

Topics:

- What PyTorch is
- Tensor versus Python list
- Tensor versus NumPy array
- Installing and importing PyTorch
- CPU versus GPU
- Reading tensor output
- Finding API documentation
- Running a basic operation

Exit task:

Create, inspect, and transform several tensors.

## Level 1: Tensor Fundamentals

Topics:

- Tensor creation
- Shapes and dimensions
- Indexing and slicing
- Reshaping
- `view`
- `reshape`
- `squeeze`
- `unsqueeze`
- Concatenation
- Stacking
- Dtypes
- Devices
- Broadcasting
- Reduction operations
- Matrix multiplication
- Strides
- Contiguous tensors

Exit project:

Implement common data transformations using tensor operations without Python loops.

## Level 2: Autograd

Topics:

- `requires_grad`
- Computation graphs
- Leaf tensors
- `backward`
- Gradient accumulation
- `detach`
- `no_grad`
- In-place operations
- Custom loss functions

Exit project:

Implement and train linear regression manually.

## Level 3: Neural-Network Construction

Topics:

- `nn.Module`
- Parameters
- Layers
- Activations
- Loss functions
- Optimizers
- Initialization
- Train mode
- Evaluation mode
- Datasets
- Data loaders
- State dictionaries
- Saving and loading

Exit project:

Build and train a small classifier with a manual training and validation loop.

## Level 4: Transformer Inference Foundations

Topics:

- Embeddings
- Matrix dimensions in attention
- Softmax
- Attention masks
- Multi-head attention
- Positional information
- Autoregressive generation
- Prefill
- Decode
- KV caching
- Batch and sequence dimensions

Exit project:

Implement a small attention layer and a minimal cached generation loop.

## Level 5: Performance-Aware PyTorch

Topics:

- Vectorization
- GPU synchronization
- CPU-to-GPU transfers
- Pinned memory
- Mixed precision
- Batch size
- Memory allocation
- PyTorch profiler
- Benchmarking correctly
- Warm-up iterations
- CUDA graphs conceptually

Exit project:

Find and fix performance problems in an intentionally inefficient inference program.

## Level 6: Compilation-Aware PyTorch

Topics:

- Eager versus compiled execution
- `torch.compile`
- TorchDynamo
- FX graphs
- Guards
- Graph breaks
- Recompilation
- Dynamic shapes
- TorchInductor
- Operator fusion
- Triton-generated kernels conceptually

Exit project:

Take a model containing graph breaks, identify them, fix them, and compare eager with compiled performance.

---

# Documentation-Grounded Content Requirements

## Source Priority

Use sources in this order:

1. Official PyTorch documentation
2. Official PyTorch tutorials
3. Official PyTorch recipes
4. Official PyTorch blog posts and release notes
5. Official PyTorch source code and tests
6. Official Hugging Face documentation
7. Official NVIDIA documentation
8. Official vLLM documentation and repository
9. Official SGLang documentation and repository
10. Original research papers
11. High-quality secondary sources only when official sources are insufficient

Do not use random blogs, SEO tutorial sites, copied documentation, or unverified posts as the primary source of truth.

## Preferred Official Sources

Prefer authoritative domains and repositories such as:

```text
pytorch.org
docs.pytorch.org
github.com/pytorch
huggingface.co/docs
github.com/huggingface
docs.nvidia.com
developer.nvidia.com
github.com/NVIDIA
docs.vllm.ai
github.com/vllm-project
docs.sglang.ai
github.com/sgl-project
arxiv.org
```

Still verify that the page is current and relevant to the pinned version.

## Version Pinning

Before generating curriculum:

1. Identify the latest stable PyTorch release from official sources.
2. Record the release date.
3. Select one stable PyTorch version for the MVP.
4. Pin a compatible Python version.
5. Pin relevant dependencies.
6. Store versions in machine-readable configuration.
7. Display the supported PyTorch version in the application.
8. Build and test all examples using the exact environment.

Do not target nightly builds unless explicitly requested.

Clearly label experimental APIs.

Example:

```yaml
content_environment:
  pytorch_version: "actual-verified-version"
  python_version: "actual-compatible-version"
  transformers_version: "actual-verified-version"
  cuda_version: null
  reviewed_at: "YYYY-MM-DD"
```

Do not leave placeholders in the implemented product.

## No Memory-Only Claims

Do not state version-sensitive technical claims without verification.

Examples of claims that require research or testing:

- API signatures
- Default argument values
- Supported dtypes
- Device-specific support
- Compiler behavior
- Graph-break behavior
- Deprecation status
- Introduction version
- Performance behavior
- GPU compatibility

When uncertain, explicitly label the behavior:

- version-dependent
- backend-dependent
- hardware-dependent
- experimental
- not yet verified

## Research Workflow for Every Concept Capsule

Before creating a capsule:

1. Identify the narrow concept.
2. Search official documentation.
3. Read the relevant pages.
4. Confirm the documentation version.
5. Check linked API reference pages.
6. Check release notes if behavior may have changed.
7. Run a minimal verification example.
8. Record source metadata.
9. Write an original concise explanation.
10. Validate the example and warm-up.

## Research Workflow for Every API Entry

For each API entry:

1. Confirm signature.
2. Confirm parameter names.
3. Confirm defaults.
4. Confirm input requirements.
5. Confirm output-shape behavior.
6. Confirm dtype and device notes.
7. Check warnings and deprecations.
8. Run the example.
9. Link to the exact official page.

Do not reproduce full documentation pages.

## Research Workflow for Every Challenge

Before publishing a challenge:

1. Identify involved concepts and APIs.
2. Research current official behavior.
3. Confirm supported starter code.
4. Confirm current best practice.
5. Run the reference solution.
6. Run visible tests.
7. Run hidden tests.
8. Run inside the actual sandbox image.
9. Verify hints are current.
10. Verify the explanation matches runtime behavior.
11. Attach sources.
12. Record verification metadata.

Do not publish a challenge unless its reference solution executes successfully.

---

# Source and Verification Metadata

Every Concept Capsule, API reference entry, and advanced challenge must include source metadata.

Example:

```yaml
sources:
  - id: pytorch-broadcasting-semantics
    title: Broadcasting semantics
    organization: PyTorch
    url: "exact-official-url"
    source_type: official_documentation
    documentation_version: "verified-version"
    accessed_at: "YYYY-MM-DD"
    relevant_sections:
      - "General semantics"
    supports:
      - "Dimensions are compared from the trailing dimension"
      - "Compatible dimensions are equal, one, or missing"

verification:
  environment:
    python: "verified-version"
    pytorch: "verified-version"
  test_file: "verification/broadcasting_basics.py"
  verified_at: "YYYY-MM-DD"
  status: passed
```

Do not invent metadata.

## User-Visible Sources

Every capsule should include a compact source section:

```text
Sources
- PyTorch documentation: Broadcasting semantics
- PyTorch API reference: torch.mean
- Verified with PyTorch X.Y.Z on YYYY-MM-DD
```

Use exact links.

Put detailed citations in a collapsible panel.

## Claim-Level Traceability

Allow important claims to map to source IDs and verification tests.

Example:

```yaml
claims:
  - text: "Using keepdim=True preserves the reduced dimension with size one."
    source_ids:
      - torch-mean-api
    verification_test: tests/claims/test_keepdim.py
```

The UI does not need a citation after every sentence, but the underlying content must be traceable.

---

# Executable Verification

Documentation is necessary but not sufficient.

Build a verification suite that:

- Imports every API used
- Runs every Concept Capsule example
- Runs every guided drill solution
- Runs every challenge reference solution
- Checks expected shapes
- Checks expected dtypes
- Checks gradient behavior
- Detects warnings
- Detects deprecated APIs
- Runs against the pinned environment

Fail CI when:

- An example does not execute
- An import is invalid
- A reference solution fails
- An expected output changes
- A deprecated API is used without an explicit teaching purpose
- Source metadata is missing
- Verification metadata is missing
- A published content item is not marked verified

---

# Documentation Freshness

Every content item must include:

```yaml
review_status:
  last_reviewed_at: "YYYY-MM-DD"
  review_due_at: "YYYY-MM-DD"
  supported_pytorch_version: "verified-version"
  stale: false
```

Mark content for review when:

- The supported PyTorch version changes
- A linked page changes substantially
- An API is deprecated
- Verification tests fail
- A source link breaks
- The review interval expires

Do not silently present stale material as current.

## Content Statuses

Support these statuses:

```text
draft
researched
verified
published
stale
```

Only verified content may be published.

---

# Content Audit Command

Create a command such as:

```bash
make audit-content
```

It should:

1. Check the latest stable PyTorch release
2. Compare it with the pinned version
3. Check official documentation URLs
4. Report broken or redirected links
5. Run content verification tests
6. Identify deprecated APIs
7. Identify stale content
8. Generate a review report
9. Avoid automatically rewriting educational content without review

Report:

- Current pinned version
- Latest stable version
- Potentially stale content
- Failed tests
- Broken sources
- Deprecated APIs
- Items requiring manual review

---

# Handling Conflicting Sources

When sources disagree:

1. Prefer documentation for the pinned stable version.
2. Prefer current source code and tests over old blog posts.
3. Check release notes.
4. Test the behavior in the pinned environment.
5. Label backend-specific behavior.
6. Do not merge incompatible claims into vague explanations.

Use explicit language such as:

> In the platform’s pinned PyTorch version, the observed behavior is X. Older material may show Y.

Only state version history when verified.

---

# Backend-Specific Behavior

Clearly identify behavior that depends on:

- CPU
- CUDA
- CUDA version
- GPU architecture
- macOS MPS
- ROCm
- Operating system
- Eager execution
- Compiled execution
- TorchInductor backend
- Dynamic shapes
- Static shapes
- Mixed precision

If the MVP verifies only CPU behavior, label it CPU-verified.

Do not teach backend-specific behavior as universal.

---

# Compiler Content Requirements

For:

- `torch.compile`
- TorchDynamo
- TorchInductor
- FX
- Graph breaks
- Guards
- Dynamic shapes
- Recompilation

Use current official PyTorch compiler documentation.

Requirements:

- Verify examples in the pinned environment
- Prefer runnable examples
- Capture diagnostics where practical
- Clearly label experimental behavior
- Do not infer graph breaks solely from intuition when they can be tested
- Store relevant verification logs or tests

---

# Performance Content Requirements

Performance claims must be measured.

For claims involving latency, throughput, or memory:

1. Cite the official recommendation when available.
2. Benchmark in a controlled environment.
3. Record hardware and software.
4. Use warm-up iterations.
5. Synchronize correctly for accelerator timing.
6. Use multiple runs.
7. Report distributions or summary statistics.
8. Avoid universal claims.

Use language such as:

> In the tested environment, configuration A reduced median latency by 18%. Results may differ by hardware, model shape, and backend.

Do not state “X is faster” without context.

---

# Research Artifacts

Store research artifacts separately.

Suggested structure:

```text
content/
  capsules/
  challenges/
  api-reference/
  sources/
    source-registry.yaml
  research/
    tensor-shapes.md
    broadcasting.md
    autograd.md
    torch-compile.md
  verification/
    examples/
    claims/
    benchmarks/
```

Research notes should contain:

- Search queries
- Official pages reviewed
- Conflicting information
- Source-of-truth decisions
- APIs tested
- Verification results
- Open questions

Do not expose unreviewed research notes as lessons.

---

# Initial Research Manifest

Before generating the content library, produce a research manifest with:

1. Supported PyTorch version
2. Compatible Python version
3. Official installation source
4. Official documentation version
5. Core documentation pages
6. Relevant tutorials and recipes
7. APIs planned for the MVP
8. Deprecated APIs to avoid
9. Verification environment
10. Date of research
11. Open questions
12. Uncertain areas

Do not generate the full challenge library until the manifest is internally consistent.

---

# Code Execution and Security

Learners will execute arbitrary Python code.

Do not execute learner code in the web-server process.

For the local MVP:

- Use isolated Docker containers
- Disable network access inside execution containers
- Apply CPU limits
- Apply memory limits
- Apply process limits
- Apply execution timeouts
- Use ephemeral workspaces
- Use read-only filesystems where practical
- Terminate runaway processes
- Return structured stdout and stderr
- Return visible and hidden test results
- Return tensor metadata
- Do not expose Docker control sockets to learner code
- Do not use unrestricted `eval` or `exec` in the application server

Support CPU execution first.

Create an executor abstraction for future:

- Local NVIDIA GPU
- Remote GPU worker
- Kubernetes job
- Sandboxed cloud execution

---

# Suggested Technology Stack

Use current stable versions verified from official sources.

## Frontend

- Next.js
- TypeScript
- React
- Monaco Editor
- Tailwind CSS
- A restrained component library
- Recharts or another lightweight charting library

## Backend

- FastAPI
- Python
- Pydantic
- SQLAlchemy
- SQLite
- Alembic

## Execution

- Docker sandbox
- PyTorch CPU image for MVP
- Pytest
- Structured JSON result protocol

## Repository Layout

```text
apps/
  web/
  api/

services/
  executor/

packages/
  challenge-schema/

content/
  challenges/
  capsules/
  drills/
  api-reference/
  sources/
  research/
  verification/

infra/
```

Use strong typing.

Keep dependencies minimal.

---

# Content Schemas

## Challenge Schema

```yaml
id: tensor-broadcasting-001
title: Normalize Each Row
type: implementation
difficulty: easy

concepts_taught:
  - row-wise normalization

concepts_tested:
  - broadcasting
  - reductions

required_prerequisites:
  - tensor-shapes
  - reduction-dimensions

optional_prerequisites:
  - keepdim

concept_capsules:
  - reduction-dimensions
  - broadcasting-basics

prompt: |
  Implement a function that normalizes every row independently.

starter_code: |
  import torch

  def normalize_rows(x: torch.Tensor) -> torch.Tensor:
      pass

function_name: normalize_rows

constraints:
  - Do not use Python loops.

visible_tests:
  - name: basic-matrix
    input: "..."

hidden_test_file: tests/tensor_broadcasting_001.py

hints:
  - level: 1
    text: Consider which dimension contains the values belonging to one row.
  - level: 2
    text: Preserve the reduced dimension so the result can broadcast.

solution_file: solutions/tensor_broadcasting_001.py

estimated_minutes: 10

sources:
  - pytorch-broadcasting-semantics
  - torch-mean-api

verification:
  status: verified
  verified_at: "YYYY-MM-DD"
  environment_id: pytorch-mvp
```

## Concept Capsule Schema

```yaml
id: broadcasting-basics
title: Broadcasting Tensor Dimensions
estimated_minutes: 4

supported_pytorch_version: "verified-version"

summary: |
  Original concise explanation based on verified official sources.

learning_objectives:
  - Determine whether two shapes can broadcast
  - Identify expanded dimensions

interactive_example: examples/broadcasting_basics.py

warmup_challenge_id: broadcasting-warmup-001

common_mistakes:
  - Comparing dimensions from the left
  - Removing a dimension required for broadcasting

sources:
  - pytorch-broadcasting-semantics

verification:
  status: verified
  verified_at: "YYYY-MM-DD"
```

## API Reference Schema

```yaml
id: torch-mean
name: torch.mean
signature: "verified-signature"
purpose: "Concise task-focused description"
input_requirements:
  - "verified requirement"
shape_behavior:
  - "verified behavior"
common_errors:
  - "common error"
example_file: examples/torch_mean.py
official_url: "exact-official-url"
supported_pytorch_version: "verified-version"
verification:
  status: verified
  verified_at: "YYYY-MM-DD"
```

---

# Initial Content Requirements

Create at least:

- 30 full challenges
- 20 guided drills
- 15 Concept Capsules
- 1 optional beginner diagnostic
- 1 mini-project

Challenge distribution:

- 10 tensor and shape challenges
- 6 autograd challenges
- 6 `nn.Module` and training-loop challenges
- 4 debugging challenges
- 4 performance-oriented challenges

Capsules should cover:

- Tensor creation
- Shapes
- Indexing
- Reshaping
- Broadcasting
- Reduction dimensions
- Matrix multiplication
- Dtypes
- Devices
- Autograd
- Gradient accumulation
- `nn.Module`
- Training loops
- Train versus eval mode
- Basic profiling

Every content item must include sources and verification.

All reference solutions must run in CI.

---

# User Interface

Main screens:

1. Dashboard
2. Diagnostic
3. Learning path
4. Challenge browser
5. Challenge workspace
6. Concept Capsule viewer
7. Tensor laboratory
8. Mini-projects
9. Progress and mastery
10. Saved experiments

## Challenge Workspace Layout

- Left: prompt, constraints, examples, prerequisites
- Center: Monaco editor
- Bottom: output and test results
- Right: Learn panel, tensor inspector, visualizations

Controls:

- Run
- Submit
- Quick refresher
- I don’t know this yet
- Explain error
- Request hint
- Reset
- Return to challenge

Keyboard shortcuts:

- Run
- Submit
- Reset
- Request hint

The interface should be optimized for long study sessions.

Avoid excessive animation and visual clutter.

---

# Gamification

Use restrained gamification.

Allow:

- Mastery progress
- Optional streaks
- Meaningful badges

Example badges:

- Broadcasting fluency
- Autograd debugger
- No-loop tensor programmer
- Memory detective
- Compile-friendly programmer

Do not award points merely for opening lessons or repeatedly running code.

---

# MVP Scope

The first usable release must include:

- Single-user local experience
- Optional diagnostic
- Adaptive starting path
- Challenge browser
- Challenge workspace
- Monaco editor
- Docker execution
- Visible tests
- Hidden tests
- Concept Capsules
- Guided warm-ups
- Progressive hints
- Tensor metadata
- Structured feedback
- Progress persistence
- Mastery tracking
- 30 challenges
- 15 capsules
- One mini-project
- Setup documentation
- Content-authoring documentation
- Research manifest
- Verification suite
- Content audit command

---

# Development Process

Before implementation:

1. Inspect the repository.
2. Confirm browser and web-research access.
3. Confirm Docker availability.
4. Write a concise product plan.
5. Write an architecture plan.
6. Define learner flows.
7. Define challenge, capsule, API, and source schemas.
8. Threat-model code execution.
9. Define MVP acceptance criteria.
10. Create an implementation checklist.
11. Produce the initial research manifest.

Then build in vertical slices.

Recommended order:

1. One challenge with code execution
2. One linked Concept Capsule
3. One warm-up
4. Return-to-challenge flow
5. Visible and hidden tests
6. Structured feedback
7. Progress persistence
8. Diagnostic
9. Mastery system
10. Content library
11. Tensor visualizations
12. Mini-project support
13. Content audit tooling
14. Documentation and testing

Do not begin with an elaborate design system.

---

# Engineering Quality Requirements

- Use strict typing in Python and TypeScript
- Add useful error handling
- Keep the executor separate from the product API
- Add unit tests
- Add integration tests
- Add end-to-end tests for the main learner flow
- Add linting
- Add formatting
- Provide reproducible local setup
- Use Docker Compose
- Include seed data
- Document architectural decisions
- Avoid placeholder implementations for core functionality
- Do not claim a feature works unless it has been tested
- Do not fake research
- Do not fabricate citations
- Do not fabricate benchmark results

---

# Definition of Done for Educational Content

A content item is complete only when:

- Claims are based on authoritative current sources
- Source metadata exists
- Code executes in the pinned environment
- Expected output is tested
- Prerequisites are linked
- Hints do not prematurely reveal the answer
- Explanations match observed behavior
- Version is recorded
- Review date is recorded
- Verification suite passes
- Status is marked verified

Only verified content may be published.

---

# MVP Acceptance Criteria

The MVP is complete when a new learner can:

1. Open the application locally
2. Take or skip the diagnostic
3. Receive an initial learning path
4. Open a challenge
5. See prerequisite concepts
6. Attempt it immediately or request preparation
7. Read a concise Concept Capsule
8. Run an interactive example
9. Complete a tiny warm-up
10. Return to the original challenge with code preserved
11. Execute real PyTorch code safely
12. Inspect output and tensor metadata
13. Submit against visible and hidden tests
14. Receive targeted feedback
15. Request progressively stronger help
16. Complete the challenge
17. See concept mastery update
18. Receive one appropriate next recommendation
19. Resume progress after restarting

Also provide:

- README
- Architecture documentation
- Learner-flow documentation
- Code-execution threat model
- Challenge-authoring guide
- Concept-Capsule authoring guide
- Source and citation guide
- Curriculum dependency graph
- Research manifest
- Verification environment documentation
- Test commands
- Seed content
- Content-audit command
- Roadmap for:
  - GPU execution
  - transformers
  - `torch.compile`
  - vLLM
  - distributed inference

---

# Required First Response

Do not start implementing immediately.

First produce:

1. Repository assessment
2. Browser and web-access confirmation
3. Product architecture
4. Security threat model
5. Learner-flow diagrams
6. Content schemas
7. Initial research plan
8. MVP checklist
9. Milestone plan
10. Risks and assumptions

Then implement the first complete vertical slice:

```text
Challenge
    ↓
Missing prerequisite
    ↓
Concept Capsule
    ↓
Warm-up
    ↓
Return to challenge
    ↓
Run and submit
    ↓
Targeted feedback
    ↓
Progress update
```

Do not generate the entire challenge library until this vertical slice is working and tested.
