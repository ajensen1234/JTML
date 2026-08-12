# Task for worker

Implement U1 of docs/plans/2026-08-12-008-refactor-optimizer-graph-container-plan.md in /home/ajj/repo/uf/JTML.

## Unit U1 — Tier-0 metric-semantics pins + CPU references

Goal: Characterize the cost-path behaviors at the pure surface before any fix — the run's R13 characterization pass that creates the surfaces the fixes plug into. Requirements: R8 (Phase 0 pins). Dependencies: none.

Files:
- Create: test/unit/test_metric_semantics.cpp, test/unit/test_metric_semantics_properties.cpp
- Modify: test/CMakeLists.txt (two new targets, headless label, TIMEOUT)

Approach:
- Per-metric deterministic cases + hegel PBT invariants per the run's Tier-0 inventory (read the inventory in .panoptes/optimizer-deep-dive/angles/02-cost-path-foundation.org — grep 'Tier-0' and 'CropIndexToGlobal'): chamfer stage functions; distance-map CropIndexToGlobal full-coverage pin (CORRECTED formula passes bijectivity, BUGGY formula as written at src/compute/distance_map_metric.cu:27 fails — this is the RED characterization, assert that the buggy variant fails, documented as the spec U4's fix must satisfy); Mahfouz float-vs-truncated ratio pair (kernel-as-spec on 2.55/x255/-2.67/-1); IoU/L1 with the IOU(empty,empty) reference decision (1.0 as spec-with-flagged-deviation, kernel gives 0/0 NaN at src/compute/iou.cu:82-83); dilation registry/constants pins (settings_constants.h IS a verbatim Flood tibia transcription — pin it as such; engine runtime dilation 6/4/1; baseline.json's dilation_px {6,3,1} is the known-stale docs-claim); the stage-guard accessor pin (a minimal getStage() accessor in include/compute/CostFunctionManager.h is a SEPARATE later unit U2 — here write the pin against the corrected guard semantics and note the accessor dependency); sym_trap tibia-transform pin; DD PolePenalty extraction + init-0/Y-axis pins.
- Direct-compile pattern: pure sources compiled into the test targets (no Qt/CUDA surface); props target follows the hegel rpath/dl recipe.

Execution note: characterization-first — the index pin is RED against today's kernel by design; do not 'fix' the test to match the bug.

Patterns to follow (READ THESE FIRST):
- test/unit/test_direct_optimizer.cpp + test/unit/test_direct_optimizer_properties.cpp (direct-compile + hegel twin target wiring)
- test/CMakeLists.txt (existing target patterns, LABELS headless, TIMEOUT)
- docs/solutions/conventions/jtml-testability-and-cmake-conventions-2026-08-07.md (direct-compile pattern, AUTOMOC gotcha, cumulative-budget doctrine)
- docs/solutions/tooling-decisions/qt6-hegel-oracle-tooling-recipes-2026-08-08.md (hegel rpath/dl recipe for props targets)
- docs/solutions/logic-errors/cost-function-parameter-double-truncation-2026-08-08.md (typed cost params: PBT round-trip invariants catch type-class bugs)

Test scenarios (from the plan):
- Happy path: each metric's CPU reference returns the documented value for a hand-computed input (e.g., 2x2 crop with known EDGE/DILATED pattern)
- Edge: IOU(empty,empty) returns spec'd 1.0 (flagged deviation)
- Edge: CropIndexToGlobal covers every pixel exactly once for the oracle's actual 64x64-crop / 16x16-block geometry (PBT invariant: bijective mapping)
- Edge: Mahfouz truncated-vs-float ratio pair differs measurably
- Error path: the buggy index formula demonstrably fails the full-coverage invariant (RED, run and record)
- Integration: registry/constants pins match baseline.json's recorded budget shape; dilation values pin the ENGINE runtime (6/4/1)

Verification: headless suite green via pixi; the index pin recorded RED (expected-fail surfaced, not silently removed); the PBT twins pass their invariants.

## Execution rules
- This is a jj repo: DO NOT describe/commit/new. Leave the working copy dirty; the orchestrator commits.
- Build/test ONLY via pixi: `pixi run build` then `pixi run ctest -R 'metric_semantics' --output-on-failure` (or `pixi run test` if the targeted run is unavailable). Never bare cmake/ninja.
- Read the referenced files before writing. Match existing conventions exactly.
- The buggy index formula at src/compute/distance_map_metric.cu:27 is: `int i = (blockIdx.y + gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;` — the corrected form is `blockIdx.y * gridDim.x`. Extract the real geometry constants from the kernel/launch site when writing the bijectivity test.
- Report: files created/modified, test run results (names + pass/fail), how the RED pin is encoded, any deviations from the plan or discoveries.

## Acceptance Contract
Acceptance level: checked
Completion is not accepted from prose alone. End with a structured acceptance report.

Criteria:
- criterion-1: Implement the requested change without widening scope

Required evidence: changed-files, tests-added, commands-run, residual-risks, no-staged-files

Finish with a fenced JSON block tagged `acceptance-report` in this shape:
Use empty arrays when no items apply; array fields contain strings unless object entries are shown.
`criteriaSatisfied[].status` must be exactly one of: satisfied, not-satisfied, not-applicable.
`commandsRun[].result` must be exactly one of: passed, failed, not-run.
`manualNotes` and `notes` are optional strings; an empty string means no note and does not satisfy `manual-notes` evidence.
```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "specific proof"
    }
  ],
  "changedFiles": [
    "src/file.ts"
  ],
  "testsAddedOrUpdated": [
    "test/file.test.ts"
  ],
  "commandsRun": [
    {
      "command": "command",
      "result": "passed",
      "summary": "short result"
    }
  ],
  "validationOutput": [
    "validation output or concise summary"
  ],
  "residualRisks": [
    "none"
  ],
  "noStagedFiles": true,
  "diffSummary": "short description of the diff",
  "reviewFindings": [
    "blocker: file.ts:12 - issue found, or no blockers"
  ],
  "manualNotes": "anything else the parent should know"
}
```