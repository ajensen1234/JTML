```json
{
  "reviewer": "ce-scope-guardian-reviewer",
  "findings": [
    {
      "title": "MVVM decomposition and CI have no acceptance example",
      "severity": "P2",
      "section": "Acceptance Examples",
      "why_it_matters": "The MVVM decomposition (R8-R10) is half of the document's own stated deliverable (it is in the title and problem frame), yet no acceptance example demonstrates it, and neither R13 (CI) nor R8-R10 appear in any of AE1-AE3. A planner/implementer reading the acceptance examples will find no verifiable gate proving the seams moved out of MainScreen or that CI actually gates every change, so the 'handoff quality' safety net is silent on exactly the structural-change goal the effort exists to deliver. Adding acceptance examples for R8-R10 and R13 closes this gap so the headline outcomes are testably attested.",
      "finding_type": "omission",
      "autofix_class": "gated_auto",
      "confidence": 75,
      "evidence": [
        "AE1. **Covers R4, R5, R7.** Given a headless `QCoreApplication` and a stub cost function, when the coordinator runs an optimize cycle to completion (and again after a stop), it returns to the idle state and accepts another launch; no test hangs or deadlocks.",
        "AE2. **Covers R1, R2, R16.** Given the `golden_oracle.org` Kneel_1 case...",
        "AE3. **Covers R11, R12.** The default test target runs and passes with no GPU and no display; real-VTK/GPU cases require the explicit flag/target.",
        "R8. Decompose `MainScreen` so presentation (widgets, slots, VTK render binding) is separated from app-state/command orchestration (frames, models, selection, save/load, optimize intent) and from services (optimization, persistence, cost, compute).",
        "R13. Provide a working CI on the pixi build that configures, builds, and runs the headless test suite so every change is gated."
      ],
      "suggested_fix": "Add an AE4 targeting R8-R10 (e.g., assert the coordinator/services no longer live in MainScreen and each extraction gate went green) and an AE for R13 (the CI job configures/builds/runs the headless suite and fails on a regression)."
    },
    {
      "title": "R7 GPU init failure contradicts headless requirement",
      "severity": "P2",
      "section": "Requirements (R7) / Acceptance AE1",
      "why_it_matters": "R7 requires covering 'a failed cost/GPU init,' but the only headless vehicle the document provides (R4/R5 stub cost function, no GPU) cannot exercise a real GPU-init failure, and R11/AE3 explicitly exclude GPU from the default run. An implementer therefore cannot satisfy R7 headlessly as written and will either write a non-headless test (violating R11/AE3) or silently drop the GPU-init half, leaving the requirement unverifiable. Rewriting R7 to split the headless-stub-covered sub-cases from the GPU target makes the coverage achievable within the document's own constraints.",
      "finding_type": "error",
      "autofix_class": "gated_auto",
      "confidence": 75,
      "evidence": [
        "R7. Cover init-failure and stop paths: a failed cost/GPU init surfaces an error into the UI state and returns to idle without deadlock; a stop request returns to idle promptly.",
        "R5. A lifecycle test drives launch → running → finished → ready-to-re-launch using an injected stub cost function (no GPU) and asserts the coordinator returns to a safe idle state and accepts a second launch...",
        "R11. Enable testing in CMake; the default test run is headless (no interactive QApplication/GUI, no VTK render window, no GPU)."
      ],
      "suggested_fix": "Split R7 into (a) headless stub-covered paths: cost-init failure and stop, no GPU, and (b) a GPU-init-failure path that runs only under the explicit-flag GPU target; update AE1 to state which sub-cases run in the default headless run."
    },
    {
      "title": "Handoff-quality references nonexistent seam/test-gate list",
      "severity": "P2",
      "section": "Success Criteria (Handoff quality)",
      "why_it_matters": "The success criterion promises to hand a planner 'the seam/test-gate order below,' but no such ordered list exists anywhere in the document; the only reference to seam ordering defers to a future /ce-plan step. A planner reading the criterion will search for the promised list, find nothing, and must re-derive the ordering themselves, directly undermining the stated 'handoff quality' success criterion. Either include the ordered seam+gate list or reword the criterion to state that ordering is produced by the planning step it references.",
      "finding_type": "omission",
      "autofix_class": "gated_auto",
      "confidence": 75,
      "evidence": [
        "**Handoff quality:** This doc, plus `golden_oracle.org` and the seam/test-gate order below, gives a planner an unambiguous list of which seams to cut, in what order, each with the test that must be green before the next cut.",
        "## Next Steps\n-> /ce-plan for structured implementation planning (may be split into phases: (1) oracle + test-runner + CI foundation; (2) lifecycle seam; (3) DIRECT/cost seams; (4) MVVM decomposition; (5) Qt6 migration gated by the oracle)."
      ],
      "suggested_fix": "Add an explicit ordered seam/test-gate list (coordinator/lifecycle first → DIRECT/cost seams → MVVM decomposition → Qt6 migration, each with its green-before-next gate), or rewrite the sentence to state the order is produced during /ce-plan rather than present 'below.'"
    }
  ],
  "residual_risks": [
    "R16 (Qt5→Qt6 migration) is the largest scope expansion and execution risk; success depends on the pre-migration oracle capture (D3), which itself depends on a GPU-capable baseline run that is not yet demonstrated.",
    "Headless Qt (offscreen) + QSignalSpy lifecycle determinism with a real worker thread is not yet proven in this environment; timeline risk for R4/R5/R7.",
    "The numeric tolerance for the DRR/cost golden comparison is unresolved and deferred; if a CPU render reference must be built, scope grows materially beyond the current goal.",
    "The seam/test-gate ordering is not yet recorded (see Finding 3), so the extraction sequence is not actually pinned by this document."
  ],
  "deferred_questions": [
    "Which test framework (Catch2 vs QtTest) and how it is declared in pixi (R14).",
    "Exact coordinator design: where the headless state machine lives and how the QCoreApplication + QSignalSpy lifecycle test is wired (R4/R5).",
    "Headless Qt platform choice and whether the DRR/cost comparison uses a CPU or GPU reference renderer (R3).",
    "The concrete seam/test-gate ordering the document promises but does not contain (Finding 3)."
  ]
}
```