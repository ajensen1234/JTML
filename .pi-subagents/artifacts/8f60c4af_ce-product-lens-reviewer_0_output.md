{
  "reviewer": "ce-product-lens-reviewer",
  "findings": [
    {
      "title": "Full MVVM decomposition may exceed the stated outcome",
      "severity": "P2",
      "section": "Requirements (R8-R10) / Success Criteria",
      "why_it_matters": "A planner following this doc will commit to a full MVVM decomposition of a 5,806-line god object and a Qt6 migration to satisfy a Success Criterion that is purely about testability. The stated human outcome — 'change an algorithm or a button path and get a fast headless pass/fail' — is already delivered by the lifecycle seam (R4-R7), the golden-oracle gate, the headless suite, and CI; full presentation-layer decomposition is a separable, much larger bet. The doc never examines the 80/20 path (ship phases 1-3, verify the outcome, then decide if MVVM is still warranted), so the reader cannot tell whether R8-R10 is load-bearing for the goal or a parallel ambition that roughly doubles effort and risk.",
      "finding_type": "omission",
      "autofix_class": "gated_auto",
      "confidence": 75,
      "evidence": [
        "**Human outcome:** A change to an algorithm or a button path is verifiable by a fast, headless test — never requiring launching the GUI and clicking to find a hang or regression.",
        "R6. The GUI binds to the coordinator as a thin caller, so the thread/completion logic lives in headless-testable code rather than inside a widget slot.",
        "R8. Decompose `MainScreen` so presentation (widgets, slots, VTK render binding) is separated from app-state/command orchestration (frames, models, selection, save/load, optimize intent) and from services (optimization, persistence, cost, compute)."
      ],
      "suggested_fix": "Explicitly state whether R8-R10 is required to meet the stated Success Criterion or is a separate maintainability goal; if the latter, re-sequence so phases 1-3 (oracle + runner + CI, lifecycle seam, DIRECT/cost seams) land and are validated against the human outcome before committing the budget to full MVVM decomposition."
    },
    {
      "title": "Oracle determinism is the load-bearing bet, deferred as routine",
      "severity": "P1",
      "section": "Requirements (R2, R3) / Outstanding Questions",
      "why_it_matters": "Every anti-circularity and regression guarantee in this plan collapses to a single mechanism: R3's 'deterministic' numeric comparison within 'documented tolerances,' which requires an independent ground truth (R2) for the DRR/cost seam. The doc explicitly defers 'whether the DRR/cost golden comparison uses a CPU render reference or the GPU path, and the numeric tolerance to document' to planning while simultaneously asserting 'Resolve Before Planning: None.' Inversion: if the GPU path is nondeterministic or no tolerance can be pinned that both passes the baseline and still catches regressions, the golden-oracle gate the whole effort rests on cannot be built, and the plan has no fallback — the single biggest risk to the premise is treated as a routine technical detail.",
      "finding_type": "omission",
      "autofix_class": "gated_auto",
      "confidence": 75,
      "evidence": [
        "R3. Compute/cost regression comparisons must be deterministic and use the documented tolerances where numeric noise requires it.",
        "[Affects R3][Needs research] Whether the DRR/cost golden comparison uses a CPU render reference or the GPU path, and the numeric tolerance to document.",
        "### Resolve Before Planning\nNone — the direction (Approach A, golden-first, lifecycle-first), the Qt6 migration, and the oracle (`golden_oracle.org`) are decided."
      ],
      "suggested_fix": "Promote the determinism/tolerance question from 'Deferred to Planning' to a gating decision with a decided fallback (e.g., commit to a CPU render reference or an analytic cost ground truth if the GPU path proves nondeterministic), and state what the plan does if a usable tolerance cannot be established."
    },
    {
      "title": "Single-fixture oracle bounds the regression guarantee",
      "severity": "P2",
      "section": "Key Flows (F2) / Dependencies (D2) / Acceptance Examples (AE2)",
      "why_it_matters": "The golden oracle and therefore the entire anti-regression claim are anchored to a single fixture, Kneel_1. A refactor that passes Kneel_1 within tolerance can still silently regress every other case, so the 'fearless editing' guarantee the doc promises is only as strong as one data point, yet the doc asserts fixtures are 'sufficient' and never plans for expanding coverage. The reader inherits a regression net that protects exactly one scenario unless they notice and add breadth themselves.",
      "finding_type": "omission",
      "autofix_class": "manual",
      "confidence": 75,
      "evidence": [
        "D2. `example_studies/Kneel_1/` and `test/vtk/test_case/` contain sufficient fixtures for oracle capture and are usable as-is.",
        "AE2. **Covers R1, R2, R16.** Given the `golden_oracle.org` Kneel_1 case ...",
        "The ML-integration pathway (running a trained model over the original image) is deferred; the binary-silhouette path is the primary oracle."
      ],
      "suggested_fix": "State explicitly that regression safety currently spans one oracle case, and add a low-cost plan (or at least a stated trigger) for extending the oracle to additional fixtures, so the fearlessness claim is honest about its coverage breadth."
    },
    {
      "title": "Manual Qt5 baseline is the unconfirmed critical-path gate",
      "severity": "P2",
      "section": "Dependencies (D3) / Requirements (R1) / Outstanding Questions",
      "why_it_matters": "The entire golden-first strategy (and Qt6 migration, R16) starts at a non-automated, user-performed baseline on a GPU-capable environment, listed as dependency D3 with no fallback and no confirmation. The doc simultaneously declares 'Resolve Before Planning: None,' yet if the user cannot or does not produce that baseline, R1 and R16 block and nothing in the plan can ship. Treating the plan's starting gate as a mere assumption hides the single point of failure that gates all downstream value.",
      "finding_type": "omission",
      "autofix_class": "gated_auto",
      "confidence": 75,
      "evidence": [
        "D3. The user can/will produce a one-time Qt5 baseline before migration (per D1), including on a GPU-capable environment.",
        "R1. Establish a ground-truth baseline from the current, working Qt5 app (per `golden_oracle.org` — Kneel_1 test case) before any behavioral/structural change or the Qt6 migration.",
        "### Resolve Before Planning\nNone — ... the oracle (`golden_oracle.org`) are decided."
      ],
      "suggested_fix": "Convert D3 from an assumption into an explicit pre-flight check (confirm the user's GPU-capable baseline exists or will be produced before phase 1 is considered startable), and note a fallback if the baseline is not obtainable."
    },
    {
      "title": "Qt6 migration bundled with god-object refactor not risk-examined",
      "severity": "P3",
      "section": "Key Decisions / Scope Boundaries (R16)",
      "why_it_matters": "Two large transformations — decomposing the 5,806-line god object and migrating Qt5→Qt6 — are committed to the same gated effort, which widens the change surface and makes it harder to attribute a regression to either cause even with the oracle gate. It is a deliberate, gated choice (arguably the right time pre-future-research), but the doc does not examine whether combining them dilutes the central 'protected behavior' goal or whether deferring the migration to a follow-on lowers risk without sacrificing the outcome. This is an advisory strategy note rather than a defect.",
      "finding_type": "omission",
      "autofix_class": "manual",
      "confidence": 50,
      "evidence": [
        "R16. Migrate Qt5 → Qt6 as part of this effort, gated by the golden oracle and the headless suite (both captured pre-migration)",
        "Migrate to Qt6, gated by the oracle: the Qt6 migration is real scope here, but behavior is protected by the pre-migration baseline + headless suite.",
        "### Resolve Before Planning\nNone — ... the Qt6 migration ... are decided."
      ],
      "suggested_fix": "Add an explicit note weighing whether Qt6 migration should be sequenced after the refactor is green (to isolate causes), or a stated rationale for keeping it in-band, so the combined-risk decision is deliberate rather than implicit."
    }
  ],
  "residual_risks": [
    "Numeric determinism of the DRR/cost oracle (GPU vs CPU reference) is unresolved; if no usable tolerance can be pinned, the central gate cannot be built.",
    "Regression safety currently spans a single oracle case (Kneel_1); other fixtures exist in the repo but are not part of the documented gate.",
    "The plan blocks on a manual, user-performed Qt5 baseline on a GPU-capable environment; if unobtainable, R1/R16 cannot proceed.",
    "Headless Qt offscreen platform in the pixi environment is assumed viable (D4) but unverified.",
    "Bundling a full Qt6 migration with the god-object decomposition may complicate regression attribution and widen the change surface."
  ],
  "deferred_questions": [
    "Is full MVVM decomposition (R8-R10) in-scope for the stated 'fearless editing' outcome, or a separable follow-on whose cost should be gated separately?",
    "What tolerance and which render reference (CPU vs GPU) will the golden DRR/cost comparison use, and what is the fallback if the chosen path proves nondeterministic?",
    "Who confirms the user can/will produce the Qt5 GPU baseline, and what happens to R1/R16 if it is not obtainable?",
    "Should the oracle be expanded to additional fixtures before the 'fearless editing' claim is made, and what triggers that expansion?"
  ]
}