```json
{
  "reviewer": "ce-design-lens-reviewer",
  "findings": [
    {
      "title": "Failure/cancel collapse into idle; no error state for view",
      "severity": "P1",
      "section": "R7 / F1 / R4",
      "why_it_matters": "The coordinator state machine is defined only as idle → running → finished → idle, yet R7 makes failure and stop return to idle while simultaneously 'surfacing an error into the UI state'. That requires the error to live somewhere the view can render, but the document never specifies the error state, how it is displayed, how it is cleared, or whether re-launch dismisses it. On the exact hang-prone seam this plan exists to fix, an implementer will guess at the error/cancel interaction model, producing a UI that either never shows why an optimize died or shows a stale error on the next launch. Defining the error surface (distinct state vs. transient error payload consumed by the thin view, plus its lifecycle) removes the guess.",
      "finding_type": "omission",
      "autofix_class": "gated_auto",
      "confidence": 75,
      "evidence": [
        "a failed cost/GPU init surfaces an error into the UI state and returns to idle without deadlock; a stop request returns to idle promptly.",
        "the coordinator moves idle → running → finished → idle",
        "a coordinator/orchestration object ... owns the state machine (idle → running → finished → idle) and the worker thread"
      ],
      "suggested_fix": "State explicitly whether failure/stop produce a distinct terminal state (e.g. error/cancelled) or a transient error/cancel signal consumed by the thin view, and specify how the view renders and clears it and whether re-launch resets it."
    },
    {
      "title": "Stop/cancel propagation unspecified; 'promptly' undefined on hang seam",
      "severity": "P1",
      "section": "R7 / F1",
      "why_it_matters": "The stop path is written as 'a stop request returns to idle promptly', but nothing defines how the stop request reaches a possibly CPU/GPU-bound worker, whether cancellation is cooperative (a flag the DIRECT loop polls) or preemptive, or how long 'promptly' is. The entire motivation is the 'hangs after optimizer finishes' class, and a stop issued mid-kernel is precisely where a return-to-idle can deadlock. Without the cancellation model and timing, AE1's 'and again after a stop' test cannot even be authored with a pass/fail criterion, and an implementer will guess at a mechanism that may not actually unblock the worker thread.",
      "finding_type": "omission",
      "autofix_class": "gated_auto",
      "confidence": 75,
      "evidence": [
        "a stop request returns to idle promptly",
        "a failure or stop path returns to idle without hanging",
        "Given a headless QCoreApplication and a stub cost function, when the coordinator runs an optimize cycle to completion (and again after a stop), it returns to the idle state and accepts another launch"
      ],
      "suggested_fix": "Specify the cancellation mechanism (cooperative flag the worker checks between/inside iterations vs. thread interruption), where it is set, and a concrete acceptance bound for 'promptly' (e.g. the stub-return-to-idle must occur within N ms or must not block the calling thread)."
    },
    {
      "title": "Headless oracle gate depends on deferred CPU-vs-GPU render decision",
      "severity": "P2",
      "section": "R1–R3 / AE2 / Deferred to Planning",
      "why_it_matters": "The central promise is a headless numeric-regression gate (F2, AE2), and R11 restricts the default run to no GPU and no VTK render window — yet producing the reference projections/golden comparison requires a render reference, and whether a CPU render/DRR path exists is explicitly deferred 'needs research'. Until that single decision is made, AE2's acceptance ('yields a pose within documented tolerance') is unverifiable and the handoff-quality goal ('tells a planner an unambiguous list of which seams to cut') cannot be met for any seam that touches numerics. A planner beginning the 'oracle foundation' phase will block on this instead of on the work itself.",
      "finding_type": "omission",
      "autofix_class": "gated_auto",
      "confidence": 75,
      "evidence": [
        "[Affects R3][Needs research] Whether the DRR/cost golden comparison uses a CPU render reference or the GPU path, and the numeric tolerance to document.",
        "R11. ... the default test run is headless (no interactive QApplication/GUI, no VTK render window, no GPU)",
        "AE2. ... yields a pose within documented tolerance of fem.jts (C) and projections matching the known-good Labels (B)"
      ],
      "suggested_fix": "Resolve before planning (not merely in 'Deferred to Planning'): decide the CPU render reference vs. GPU path used to reproduce the baseline headlessly, confirm a CPU render path exists, and record the numeric tolerance — since this decision gates AE2 and every numerical seam."
    },
    {
      "title": "R9 green-gate conflicts with no-characterization rule for UI cuts",
      "severity": "P2",
      "section": "R8 / R9 / R11 / R12",
      "why_it_matters": "R8 splits MainScreen into presentation (widgets, slots, VTK binding) plus state/services, and R9 requires 'each extraction lands with its test gate already green ... no extraction ships untested'. But presentation-only extractions cannot be headless-tested (R11, no GUI), and R12 forbids both Qt-mocking and god-object characterization — the two mechanisms that could verify a pure-presentation cut. An implementer of a UI-layer extraction therefore has no defined gate and no permitted test technique, creating a direct contradiction that will either be ignored (violating R9) or resolved by an ad-hoc, inconsistent per-cut test approach.",
      "finding_type": "error",
      "autofix_class": "gated_auto",
      "confidence": 75,
      "evidence": [
        "R8. Decompose MainScreen so presentation (widgets, slots, VTK render binding) is separated from app-state/command orchestration ... and from services",
        "R9. Each extraction lands with its test gate already green before the next cut; no extraction ships untested.",
        "R12. no Qt-mocking infrastructure ... and no \"instantiate the real MainScreen god object\" characterization tests.",
        "R11. the default test run is headless (no interactive QApplication/GUI, no VTK render window, no GPU)"
      ],
      "suggested_fix": "Define the per-layer gate: state that logic/service/coordinator extractions get headless unit gates while presentation-only extractions are gated by compile plus an explicitly-scheduled integration/manual-visual check, and record which cuts fall in each category so R9 is satisfiable."
    }
  ],
  "residual_risks": [
    "Five load-bearing decisions (test framework R14, coordinator design R4/R5, headless platform R11, Qt6 mechanics R16, and the CPU-vs-GPU oracle reference R3) sit in 'Deferred to Planning'; if the planning phase is skipped or under-run, any one can block a phase.",
    "The golden-oracle baseline (D3) is a one-time Qt5/GPU capture that must precede any refactor or migration (R1/R16); if the capture is lost or the GPU environment is unavailable mid-effort, R1 and AE2 cannot be re-established.",
    "The stop/cancellation decision (finding 2) interacts with the GPU kernel; if DIRECT cannot cooperatively cancel mid-kernel, the 'prompt' stop promise may be unachievable and needs a scoped fallback."
  ],
  "deferred_questions": [
    "Which test framework (Catch2 vs QtTest) is chosen and added to pixi — explicitly deferred by R14.",
    "Exact coordinator design — where the state machine lives and how the headless QCoreApplication + QSignalSpy test is wired (deferred, affects R4/R5).",
    "Numeric tolerance to document for the golden comparison, and confirmation a CPU render reference exists for headless oracle reproduction (deferred, affects R3).",
    "Headless Qt platform (offscreen) selection for the default test target (deferred, affects R11)."
  ]
}
```