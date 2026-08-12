I've completed a full read of the plan, verified the owner goal against the origin brainstorm, and spot-checked the code claims I lean on (kernel bug at `distance_map_metric.cu:27` confirmed; stage-guard tautology at `CostFunctionManager.cpp:44–48` confirmed; `Optimize()` landmarks at :1096/:1135/:1151/:1234 confirmed; oracle flat-3000 monoplane twin confirmed at `oracle_test.cpp:244/:287–292`; stale caps comment actually at `direct_optimizer.h:58–59`, not :19–20 as U8 cites).

Premise challenge: holds — the run shape is genuinely hard-coded (verified), the baseline is contaminated by a live verified kernel bug, and the plan's PoC framing (stages-as-data, bit-identical, one graph) matches the owner's stated goal in the origin (R6–R7, AE1, success criteria). Scope boundaries do protect the graph-vs-CUDA separation consistently: the plan's gates are oracle-IoU + bit-identity, never evals/s, and every deferred perf item is named with its owning plan. The path-dependent commitments (StageSpec schema, Options field set, adapter signature) are additive by design (R13) and forward-compatible (Calibration-by-value for biplane, stub kinds for polish/ML initializer). Inversion passes: the re-baseline magnitude, Cut B drift, and wizard-region risks are each pre-mitigated in the plan's own risk table. One advisory survives filtering; the rest were deliberate, acknowledged, or other-persona items.

{
  "reviewer": "ce-product-lens-reviewer",
  "findings": [
    {
      "title": "Meter-fix deferral is bucketed under the CUDA-separation rationale, but the 1000x unit error is a user-visible correctness fix — the deferral's stated rationale does not cover it, and no in-plan owner appears if the perf plan slips",
      "severity": "P3",
      "section": "Scope Boundaries / Requirements Trace (R10)",
      "why_it_matters": "The plan justifies deferring the meter's 1000x unit-error fix with the session decision 'to keep the graph work independent of CUDA work', but a units bug is not CUDA work — the origin this plan cites (origin R10) motivated early absorption precisely because it is user-visible (IPS/ETA display wrong ~1000x on Linux). A reader of the deferral sees a technical-separation rationale that does not actually cover the meter fix, and the user-visible wrongness persists for the entire duration of this plan plus the follow-up perf plan's lead time (the perf plan is itself gated on this plan's re-baseline). Naming the user-facing cost in the deferral note — or moving the fix into Phase 0 as origin R10 had it — closes the gap without disturbing the graph/CUDA separation.",
      "finding_type": "omission",
      "autofix_class": "manual",
      "confidence": 50,
      "evidence": [
        "No CUDA/perf work: no Cut 0 instrumentation, no Cut 4 meter fix, no async copies, no N-way batching, no batch seam (origin R10–R14 → the follow-up perf plan, which also owns the meter's 1000× unit-error fix).",
        "R10 (Cut 0 + Cut 4 early): **deferred to the follow-up perf plan** (see Scope Boundaries — session decision to keep CUDA work separate; this plan's gates are oracle-IoU + bit-identity, not measured evals/s)",
        "The algorithm battery, the CUDA/perf workstream (incl. Cut 0 instrumentation + Cut 4 meter fix, origin R10), and the polish horizon are **separate follow-up plans** — per the session decision to keep the graph work independent of CUDA work."
      ]
    }
  ],
  "residual_risks": [
    "C++-typed registry (R2/U7) means 'registering a graph' remains a code edit + rebuild; the problem frame's 'configurations, not code edits' is realized as typed data in a dedicated TU. Suppressed: deliberate origin decision (A1 is 'a code change, by design'; Key Decision 'C++-typed graphs + named registry, not JSON') and the plan mirrors it in R2/U7.",
    "U4 re-baselines the 6/4/1 dilation shape before U5's probe arbitrates 6/3/1-vs-6/4/1; if the probe favors 6/3/1, the pinned baseline churns in a follow-up event. Suppressed: the plan explicitly acknowledges the reconciliation is 'deliberately NOT resolved in this plan' and scopes the one-re-baseline rule to this plan's single-variable event.",
    "'Bit-identical' / 'zero production-visible change' (U9) are scoped to the re-baselined engine, while U2 (sym_trap tibia x→z fix) and U4 (index fix) deliberately change production recovery. Suppressed: scoping is stated at each site ('bit-identical against the re-baselined oracle', U2 'zero oracle impact'); misreading risk judged pedantic.",
    "Stale landmark: U8 cites the stale caps comment at direct_optimizer.h:19–20; verified current location is :58–59. Suppressed: cross-reference accuracy is coherence-lens territory, and the plan's own risk table instructs 're-grep before editing'.",
    "PoC verification (bit-identity diff, four lineage invariants, re-baseline) lives on the GPU machine, not headless CI. Suppressed: existing convention, acknowledged in the plan's risk table ('no oracle assertion in headless CI')."
  ],
  "deferred_questions": [
    "Should the meter's user-visible error (IPS/ETA ~1000x) be tracked as a known-visible defect during this plan's duration, since the fix now lives in a follow-up plan gated on this plan's re-baseline?",
    "If U5's probe arbitrates the dilation lineage as 6/3/1 rather than 6/4/1, does the follow-up reconciliation change code or docs only? The plan assumes doc-level; a code change would create a second baseline event."
  ]
}