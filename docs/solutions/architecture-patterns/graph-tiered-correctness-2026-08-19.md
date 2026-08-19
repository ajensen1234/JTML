---
title: "Graph-tiered correctness for CUDA cost evaluation"
date: 2026-08-19
category: docs/solutions/architecture-patterns
module: JTML CUDA cost evaluation
problem_type: architecture_pattern
component: testing
severity: medium
related_components:
  - GPUMetrics
  - GPUModel
  - RenderEngine
  - EvaluationExecutor
  - EvaluationContext
tags:
  - cuda
  - graph
  - correctness
  - oracle
  - tiered-gate
---

# Graph-tiered correctness for CUDA cost evaluation

## Context

The CUDA-Graph greedy executor must not weaken the existing two-tier oracle
(Tier-1 analytic bit-exact via `jtml.direct_optimizer`, Tier-2 silhouette
IoU ≥ 0.85 via `jtml.oracle` / `jtml.oracle_multistage`) while still admitting
a final floating-point tolerance for composition. The prior bit-identity harness
(`test/oracle/bit_identity_test.cpp` U10) enforced a single `HASHLINE_SEQ`
over final `double` scores, which would hide a Layer-B raw-int divergence behind
a Layer-C tolerance.

## Decision

Adopt a **layered gate** for any graph-admitted `DIRECT_DILATION` path:

- **Layer A — rendered image** (`unsigned char` after `FillTriangleKernel`): byte-identical
  over `test/golden/fem_golden.jts` / `baseline.json` frames (`diff 0` bytes).
- **Layer B — raw int metric reductions** (`pixel_score`, `distance_map_score` /
  `edge_pixels_count`, `intersection` / `union`, `white_count` after
  `cudaMemcpyAsync`): bit-exact per reduction. This is the code gate for graph
  admission; a failure is never absorbed into tolerance.
- **Layer C — final double composition** (`white_sum + (-pixel_score) + distance/edge+0.1`):
  bounded by the **frozen** tolerance in `test/golden/graph_pre_registration.json`
  (`abs 1e-12`, `rel 1e-9` with rationale). Tolerance is pre-registered in U2,
  not derived after U7.

`test/oracle/layered_correctness_test.cpp` is the code gate (oracle;gpu,
`TIMEOUT 3600`, `WORKING_DIRECTORY repo root`, repeats `>=3x` to catch
interleaving races from shared reduction targets). `test/oracle/bit_identity_test.cpp`
retains its original `HASHLINE_SEQ` plus a retained-with-coverage U7 case, and
`test/oracle/multistage_oracle_test.cpp` retains its 3-frame caps/IoU harness plus
a U7 admission smoke. The Tier-2 oracle (`oracle_test.cpp` / `multistage`)
remains the human-visible gate (`IoU ≥ 0.85`, vertically flipped).

## Consequences

- An `int` metric divergence can no longer be hidden by widening `double` tolerance.
- The frozen `graph_pre_registration.json` artifact is the single source for
  `abs`/`rel` and for `N=1,2,4` workloads — any change requires re-approval as
  scope change (coherence R10).
- Repeats `≥3x` make a shared `dev_pixel_score` / `dev_distance_map_score`
  race nondeterministically visible, per `TEST_IMPACT_MATRIX.md` U7 row.

## When to apply

Apply this tiering whenever a new cost-function graph recipe is admitted.
Biplane or additional recipes must each pass Layer A/B exact before Layer C is
checked; do not add a second recipe until the `direct_dilation_monoplane`
recipe passes.

## References

- `docs/TEST_IMPACT_MATRIX.md` (U2 frozen matrix, U7 retained-with-coverage rows)
- `test/golden/graph_pre_registration.json` (frozen `layer_c_tolerance` and workloads)
- `docs/solutions/conventions/jtml-testability-and-cmake-conventions-2026-08-07.md`
  (two-tier oracle, `LABELS headless` vs `oracle;gpu`, `WORKING_DIRECTORY`)
- `docs/plans/2026-08-19-011-feat-cuda-graph-greedy-evaluation-executor-plan.md` U7
