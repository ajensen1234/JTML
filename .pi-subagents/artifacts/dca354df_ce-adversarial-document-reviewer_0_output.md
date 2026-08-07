Based on my full read of both the plan and the origin requirements document (plus verification of the golden_oracle.org and CMake/test tree claims in the repo), here is my adversarial review.

```json
{
  "reviewer": "ce-adversarial-document-reviewer",
  "findings": [
    {
      "title": "U4 worker loop depends on U5, not declared",
      "severity": "P2",
      "section": "Implementation Units → U4 (Dependencies / Approach)",
      "why_it_matters": "U4 specs the coordinator's worker as running 'the (eventual) DirectOptimizer loop' with the real GPU eval 'in production', but DirectOptimizer is not created until U5, and U4's dependency list is only 'U1, U3' — not U5. At the time an implementer writes U4, no DirectOptimizer exists, so the worker's production-code path is undefined: they must either invent whether it calls the still-embedded DIRECT inside optimizer_manager (GPU-bound, contradicting headless) or a placeholder loop. This is the load-bearing seam where the hangs live, and the ordering ambiguity is exactly what the plan elsewhere promises to nail down.",
      "finding_type": "error",
      "autofix_class": "manual",
      "confidence": 75,
      "evidence": [
        "**Dependencies:** U1, U3.  (U4)",
        "Worker runs the (eventual) `DirectOptimizer` loop with an injected `std::function<double(const Point6D&)>` (stub in tests, real GPU eval in production). (U4 Approach)",
        "**Dependencies:** U1, U3 (CUDA-free `data_structures_6D`).  (U5)"
      ],
      "suggested_fix": "Either add U5 to U4's dependencies and reorder so DirectOptimizer extraction precedes the coordinator worker binding, or explicitly state that the U4 worker runs a placeholder/trivial loop and is rewired to DirectOptimizer in U5/U6. Remove the 'eventual' ambiguity so an implementer knows exactly what the U4 worker runs headlessly."
    },
    {
      "title": "R3 DRR compare reference deferred to implementation",
      "severity": "P3",
      "section": "Open Questions → Deferred to Implementation / U2 Approach",
      "why_it_matters": "The origin marks the DRR/cost comparison reference (CPU vs GPU) a 'Resolve Before Planning' precondition of the oracle phase (R3), but the plan defers 'Whether a CPU DRR render reference exists or must be built for Tier-2' to implementation while U2 simultaneously claims to 'Resolve the R3 compare reference'. An implementer starting U2 must make a baseline-defining decision (does Tier-2 compare against a CPU render, or is it GPU-only?) that the plan promised pre-resolved, which risks re-opening the Tier-2 gate's correctness basis mid-flight.",
      "finding_type": "omission",
      "autofix_class": "manual",
      "confidence": 75,
      "evidence": [
        "Whether a CPU DRR render reference exists or must be built for Tier-2 (depends on whether the cost can run CPU-side; if not, Tier-2 is GPU-only and labeled accordingly). (Deferred to Implementation)",
        "Resolve the R3 compare reference: **Tier-1** analytic CPU golden runs in CI; **Tier-2** GPU tolerance-based gate runs only under the `oracle` label on a GPU machine... (U2 Approach)"
      ],
      "suggested_fix": "Decide and record in the plan whether Tier-2's silhouette comparison reference is the known-good Labels (B) directly or requires a CPU-rendered DRR reference, so U2 carries no open correctness-defining decision into implementation."
    },
    {
      "title": "R7b real GPU-init-failure test has no owner",
      "severity": "P3",
      "section": "Implementation Units → U4 / U6 Test scenarios",
      "why_it_matters": "R7(b) requires the real GPU-init-failure path to be exercised in the explicitly-flagged GPU target, and the plan's Requirements Trace binds R7 to U4. But U4's error-path scenario covers only the headless stub-init failure, and U6's scenarios (integration + cumulative-budget edge) add no real GPU-init-failure test, and no implementation unit creates one. The requirement is asserted but never owns a test scenario in any unit, so an implementer has no defined place to land the R7b GPU-target test.",
      "finding_type": "omission",
      "autofix_class": "manual",
      "confidence": 75,
      "evidence": [
        "Error path: injected cost-init failure → `ERROR` state surfaced then back to idle, re-launchable, no deadlock (headless, stub — real GPU-init-failure handled in the GPU target, per R7b). (U4 Test scenarios)",
        "**Test scenarios:** Integration: on a GPU machine, running the pipeline with the extracted optimizer yields a pose within tolerance of the pre-extraction result / the Tier-2 oracle. Edge case: the cumulative-budget behavior matches the pre-extraction run.  (U6)"
      ],
      "suggested_fix": "Add an explicit GPU-target test scenario (in U6 or a dedicated GPU unit) asserting real GPU-init failure surfaces a clean ERROR→idle without deadlock, so R7b has an owning test rather than being referenced with no landing site."
    },
    {
      "title": "R13 default-run ctest timeout not specified",
      "severity": "P3",
      "section": "Implementation Units → U1 Approach / Test scenarios",
      "why_it_matters": "R13 and AE5 require every headless lifecycle test AND the default ctest run to be bound by a timeout so a hang fails the suite rather than hanging CI. U1 (which Requirements Trace maps to R13) specifies only `add_test`/`catch_discover_tests` and ctest labels; the only timeout mechanism named is per-test `QSignalSpy::wait` in U4, which turns a hang into an assertion failure only while a test is inside `wait`. A deadlock not reached by a logical `wait` call would hang ctest indefinitely, defeating AE5's 'hang fails rather than hangs' CI guarantee.",
      "finding_type": "omission",
      "autofix_class": "safe_auto",
      "confidence": 75,
      "evidence": [
        "Modify: ... test/CMakeLists.txt (register targets + `add_test`/`catch_discover_tests`, add `headless` default label and `gpu`/`oracle` labels) (U1 Files)",
        "Integration: a deliberately-stuck worker timeout fails the test (AE5 behavior). (U4 Test scenarios — relies on QSignalSpy::wait, not a ctest timeout)",
        "Set every headless lifecycle test (and the default ctest run) with a timeout so a hang becomes a failing test rather than a hung CI job. (R13)"
      ],
      "suggested_fix": "In U1, specify `set_tests_properties(... TIMEOUT n)` (or `ctest --timeout`) on the headless/default suite so the ctest run itself is time-bounded, not just individual `QSignalSpy::wait` assertions."
    }
  ],
  "residual_risks": [
    "U2's pre-flight gate (Qt5 GPU baseline, D3) blocks U8 (Qt6 migration) and Tier-2 AE2 if the user cannot capture it; the plan flags the fallback (defer migration) but a stall here cascades to R16.",
    "The plan defers 'Qt6 before or after heavy MVVM' re-evaluation to the Phase-3 gate; if migrated after deep MVVM, the same MainScreen lines get double-touched, potentially re-running U7's manual-visual gauntlet.",
    "Tier-2 fractional-silhouette tolerance values are recorded only at baseline capture (Phase-1), so the oracle gate's numeric stringency is not fixed at plan time.",
    "A minor plan-internal path inconsistency: U2 creates `test/oracle/oracle_test.cpp` but the Output Structure tree shows only `test/golden/` with no `test/oracle/` — the GPU oracle test's canonical location is ambiguous."
  ],
  "deferred_questions": [
    "Whether Tier-2's comparison reference is the known-good Labels (B) or a separately built CPU DRR render — left open per the deferred compare-reference item.",
    "Whether real GPU-init-failure (R7b) is actually testable given a GPU is required, and under which labeled target it lands.",
    "Qt6-before-or-after-MVVM sequencing to avoid double-touching MainScreen — deferred to the Phase-3 completion gate."
  ]
}
```