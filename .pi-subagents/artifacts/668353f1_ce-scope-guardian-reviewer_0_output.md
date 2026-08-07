I have verified the plan's repo claims (line counts, `Point6D(Pose)` ctor at `.h:36`/`.cpp:34`, `render_engine.cuh` include, cumulative budget semantics `budget_ += branch/leaf_budget` with `cost_function_calls_` reset only before trunk, Qt5 find_package, `qt = "5.*"`, `test/vtk/test_case` = left knee vs Kneel_1 = right knee). The plan is accurately grounded and honors the R1–R16/AE1–AE5 contract with no dropped origin content and no meaningful scope creep. Remaining findings are plan-internal dependency/clarity gaps.

```json
{
  "reviewer": "ce-scope-guardian-reviewer",
  "findings": [
    {
      "title": "U4 worker body forward-refs DirectOptimizer not yet extracted",
      "severity": "P2",
      "section": "U4 (Headless optimize-coordinator + worker)",
      "why_it_matters": "U4's stated goal and primary test scenario (R4/R5) require a coordinator/worker testable 'with an injected stub cost', but the injected `std::function` cost seam described there is an abstraction that U5 (extract DirectOptimizer) actually creates. Since U4 lists Dependencies U1, U3 (not U5), an implementer reaching U4 either stalls on 'what does the worker run?' or must hand-build the cost-injection plumbing and the worker loop, then rebuild it again inside DirectOptimizer in U5 — i.e. the ordering forces either duplication or an implied U4->U5 dependency that the plan never states.",
      "finding_type": "omission",
      "autofix_class": "manual",
      "confidence": 75,
      "evidence": [
        "**Dependencies:** U1, U3.  **Approach:** `OptimizeCoordinator` is a `QObject` ... Worker runs the (eventual) `DirectOptimizer` loop with an injected `std::function<double(const Point6D&)>` (stub in tests, real GPU eval in production).",
        "U5 **Dependencies:** U1, U3 (CUDA-free `data_structures_6D`).  **Approach:** `DirectOptimizer` ... exposes `Point6D optimize(std::function<double(const Point6D&)>)`"
      ],
      "suggested_fix": "State explicitly in U4 that the worker runs a temporary inline DIRECT loop over the injected cost, replaced verbatim by DirectOptimizer in U5 (no duplication of the seam), OR fold the DirectOptimizer/media extraction (U5) ahead of U4 and make the coordinator's worker consume it. At minimum, add an explicit 'U4 worker-body is replaced at U5' note so the implementer knows the injected-cost seam is provisional."
    },
    {
      "title": "R13 default-ctest-run timeout not wired in U1/CI",
      "severity": "P3",
      "section": "U1 (Test harness, frameworks, and CI)",
      "why_it_matters": "Origin R13 explicitly requires both per-test lifecycle timeouts AND 'the default ctest run' bound by a timeout so a hang fails the job. The plan wires per-test `QSignalSpy::wait` timeouts (AE5, U4) but never adds a suite/run-level ctest timeout or `CTEST_TEST_TIMEOUT`; a hang in a pure Catch2/DIRECT test (no event loop) would still hang CI, defeating the stated R13 intent.",
      "finding_type": "omission",
      "autofix_class": "safe_auto",
      "confidence": 75,
      "evidence": [
        "U1 Test scenarios: 'Integration: `pixi run test` configures, builds, and runs the headless suite and reports pass/fail' and 'Error path: a deliberately failing assertion yields a non-zero ctest exit (CI fails)'",
        "R13 (origin): 'Bind every headless lifecycle test (and the default ctest run) with a timeout so a hang becomes a failing test rather than a hung CI job.'"
      ],
      "suggested_fix": "In U1/CI, run `ctest -L headless --timeout <N>` (or set `CTEST_TEST_TIMEOUT`) so the entire default headless run is timeout-bounded, and record this in the U1 Verification line alongside 'a GPU-labeled test is not run by default'."
    },
    {
      "title": "Pre-migration units specify Qt6:: while build is still Qt5",
      "severity": "P3",
      "section": "U1 harness / Key Technical Decisions",
      "why_it_matters": "The harness and coordinator phases (U1–U4) are built and run before the Qt6 migration (U8), yet the plan's normative language uses `Qt6::Test` / `find_package(Qt6)`. If followed literally an implementer would attempt `find_package(Qt6)` at U1 against a `qt = \"5.*\"` pixi and fail the build; the inline caveat in U1 self-corrects this, but the contradictory phrasing risks a wasted build/debug cycle on exactly the phase that is meant to de-risk everything else.",
      "finding_type": "error",
      "autofix_class": "safe_auto",
      "confidence": 50,
      "evidence": [
        "Key Technical Decisions: 'QtTest (`Qt6::Test`) for the QObject/QThread/QSignalSpy coordinator seam'",
        "U1 Approach: 'Wire `find_package(Qt6 COMPONENTS ... Test)` — but note: at U1 the project is still on Qt5, so use whatever Qt is current'"
      ],
      "suggested_fix": "Normalize the pre-migration units (U1–U4) to the current Qt: `Qt5::Test` / `find_package(Qt5 COMPONENTS Core Gui Widgets Test)` with a note that this flips to `Qt6::` at U8, so only U8 carries the Qt6 rename."
    },
    {
      "title": "Output Structure omits oracle_test.cpp and harness smoke test",
      "severity": "P3",
      "section": "Output Structure",
      "why_it_matters": "The Implementation Units create `test/oracle/oracle_test.cpp` (U2) and `test/unit/test_harness_smoke.cpp` (U1), but the top-level Output Structure never lists them (only `test/golden/`, `test/unit/`, `test/lifecycle/`), and the oracle test's home directory (`test/oracle/`) appears nowhere in the tree. An implementer mapping from the Output Structure to the units will not know where the GPU-labeled oracle test lives or that it is a distinct target.",
      "finding_type": "omission",
      "autofix_class": "safe_auto",
      "confidence": 50,
      "evidence": [
        "U2 Files: 'Create: `test/golden/pose/fem.jts` ..., `test/golden/baseline.json` (settings + tolerance values), `test/oracle/oracle_test.cpp` (Catch2, GPU-labeled)'",
        "U1 Files: 'Test: `test/unit/test_harness_smoke.cpp`'",
        "Output Structure lists only `test/golden/` (pose/, images/, baseline.json), `test/unit/` (test_data_structures.cpp, test_direct_optimizer.cpp, test_pose_file_io.cpp), `test/lifecycle/coordinator_test.cpp`"
      ],
      "suggested_fix": "Add `test/oracle/oracle_test.cpp` and `test/unit/test_harness_smoke.cpp` to the Output Structure (or move the oracle test into `test/golden/`) so the top-level file map matches the Implementation Units' file lists."
    }
  ],
  "residual_risks": [
    "Tier-2 GPU Kneel_1 oracle and the whole Qt6 migration hinge on the user capturing the Qt5 GPU baseline (D3/pre-flight gate); the plan correctly defers U2/U8 if it is unobtainable, but this is an external A1 dependency that can push the effort's schedule or truncate oracle coverage.",
    "GPU/CUDA reductions are not bit-reproducible across hardware, so Tier-2 correctness rests on recorded per-axis tolerances captured on the same machine; any hardware/toolchain drift in the tolerance capture is not itself oracle-checked.",
    "U4/U5 worker-body ambiguity (finding 1) can produce temporary duplicate DIRECT-loop plumbing if the implementer misreads the 'eventual DirectOptimizer' sequencing.",
    "Oracle regression safety spans a single case (Kneel_1); refactors touching other case behavior are implicitly unprotected until the follow-on 'expand oracle' work.",
    "Qt6 OpenCV Qt-binding coordination and the conda-forge vtk vs vtk_installer.sh choice are only resolved at Phase 5, so U8 carries late-detected build risk."
  ],
  "deferred_questions": [
    "Exact Tier-2 tolerance values (translations, rotations, silhouette pixel threshold) and the chosen CPU-vs-GPU compare reference — recorded at Qt5 baseline capture (U2/D3) per R3; an implementer cannot finalize the oracle gate before this exists.",
    "Whether a CPU DRR render reference exists or must be built for Tier-2, or whether Tier-2 is GPU-only (the plan labels this 'depends on whether the cost can run CPU-side').",
    "Qt6-before-MVVM vs MVVM-before-Qt6 sequencing (U8 lists U7 'ideally'): the plan deliberately defers this to the Phase-3 completion gate to avoid double-touching the same MainScreen lines."
  ]
}
```