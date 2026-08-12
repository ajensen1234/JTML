All verifications complete. I have what I need — the code confirms most landmarks, and I've identified the internal-consistency issues with airtight evidence from the plan itself plus its cited normative sources.

```json
{
  "reviewer": "ce-coherence-reviewer",
  "findings": [
    {
      "title": "Phase labels on U2/U3/U4 contradict the plan's own Overview grouping and the origin R8 phase spine the plan claims to execute",
      "severity": "P2",
      "section": "Implementation Units (U2/U3/U4/U5/U6) vs Overview vs Requirements Trace R8",
      "why_it_matters": "An implementer mapping units onto the plan's headline claim 'The plan executes Phases 0–2 of the confirmed research-run path (origin R8)' gets three of six labeled units wrong: U2 (behavior-neutral fixes) and U3 (finite-check) are labeled Phase 1 and U4 (re-baseline) Phase 2, while origin R8 — the traced requirement — puts fixes + finite-check + re-baseline all in Phase 0, the probe/oracle in Phase 1, and Cuts A/B/C/F in Phase 2. The plan's own Overview groups the re-baseline with the foundation ('the foundation that makes the baseline trustworthy (bug pins, behavior-neutral fixes, the live distance-map index fix, exactly one re-baseline, the CUDA-free finite-check)'), and the mislabels invert the phase sequence against the dependency graph: U5/U6 (labeled Phase 1) both declare 'Dependencies: U4' (labeled Phase 2), so the 'critical path of the whole program' unit sits inside a phase that follows its consumers. Relabeling U2/U3/U4 to Phase 0 (and U7–U10 to Phase 2, or leaving them unlabeled as now) makes the unit map, the overview, and origin R8 agree.",
      "finding_type": "error",
      "autofix_class": "gated_auto",
      "confidence": 100,
      "evidence": [
        "[plan U4] **Requirements:** R7, R8 (Phase 2); the critical path of the whole program",
        "[plan U2] **Requirements:** R8 (Phase 1)",
        "[plan U3] **Requirements:** R8 (Phase-1 addition); supports R9",
        "[plan U5] **Requirements:** R8 (Phase 1) ... **Dependencies:** U4",
        "[plan U6] **Requirements:** R7, R8 (Phase 1); pins U9 ... **Dependencies:** U4 (U5 recommended — ...)",
        "[plan Overview] The plan executes Phases 0–2 of the confirmed research-run path (origin R8): the foundation that makes the baseline trustworthy (bug pins, behavior-neutral fixes, the live distance-map index fix, exactly one re-baseline, the CUDA-free finite-check), the two measurement instruments that pin the container (z-profile probe, multi-stage oracle), and the container itself (Cuts A/B/C/F → the `jtml-production` graph)",
        "[origin, docs/brainstorms/2026-08-12-optimizer-path-requirements.md R8] Phase 0 foundation (bug pins → behavior-neutral fixes → the live distance-map index fix → exactly one single-variable re-baseline, Canny pinned 3/0/150, recovered-pose delta recorded; plus the CUDA-free finite-check ...) → Phase 1 measurement (z-profile probe → multi-stage oracle on the driver seam → ...) → Phase 2 graph container (Cut A pure builders in parallel with Phase 1; Cut B script-driven loop + adapter after the oracle exists to pin it; Cut C `Options` with bit-identical defaults; Cut F torch include/link hygiene)"
      ]
    },
    {
      "title": "R13 is cited as a binding requirement in Key Technical Decisions, U1, and U8 but is absent from the Requirements Trace, and origin R13 (the trace's numbering source) is the parallel-eval gates — not the defaults-proof strategy the plan attributes to it",
      "severity": "P3",
      "section": "Requirements Trace vs Key Technical Decisions / U1 / U8",
      "why_it_matters": "The Requirements Trace is the plan's only in-document requirement map, and it deliberately includes even non-executed requirements (R9 'honored', R10 'deferred') — yet R13, cited three times as a normative strategy, is not in it. A reader resolving R13 per the trace's numbering finds the origin's R13 is 'Gates for the workstream' of the parallel-eval effort (deferred to the follow-up perf plan), which has nothing to do with the guarded-divergence/bit-identical-defaults role U1 and U8 assign it; the research run's own R13 (bit-identical pin surface, per the synthesis) is a different numbering system the plan never labels. Either the citation misattributes the requirement or the plan silently mixes two numbering schemes; the guarded-divergence text itself is self-contained, so implementation survives, but the trace cannot be trusted as the requirements map without this resolved.",
      "finding_type": "omission",
      "autofix_class": "manual",
      "confidence": 100,
      "evidence": [
        "[plan Key Technical Decisions] every new path guards on \"different from default\" so defaults reproduce today's search bit-identically (R13)",
        "[plan U1] **Goal:** Characterize the seven cost-path behaviors at the pure surface before any fix — the R13 characterization pass that creates the surfaces the fixes plug into.",
        "[plan U8] Guarded divergence: every path guards on \"different from default\" before diverging (R13 bit-identical-defaults proof strategy).",
        "[plan Requirements Trace] full list: R1 ... R10 only; R9 is 'honored — no variant work in this plan' and R10 is '**deferred** to the follow-up perf plan' — both non-executed requirements are traced, R13 is not",
        "[origin, docs/brainstorms/2026-08-12-optimizer-path-requirements.md] R13. **Gates** for the workstream: headless green, oracle green, bit-identity diff empty, measured improvement within the pre-registered bands (async copies → 60–300 evals/s; N-way → 1,000–5,000; Flood's 3,000 evals/s on a 2018 GTX 970 as the cross-era parity line)"
      ]
    },
    {
      "title": "Cut D is never mentioned: the plan executes 'Cuts A/B/C/F', defers Cut E (R4), and is silent on Cut D, which the normative base defines as the adapter factory",
      "severity": "P3",
      "section": "Overview / Requirements Trace R4 / Scope Boundaries",
      "why_it_matters": "The plan's grounding ('.panoptes/optimizer-deep-dive/synthesis.org') enumerates architecture cuts A–F, with Cut D = 'adapter factory ... whenever the oracle is green' / 'Cut D moves only the twin's body; the factory must stay stage-shape-agnostic'. The plan names A/B/C/F (executed) and E (deferred via R4) but never states Cut D's disposition — merged into U9 (which does absorb the twin body into BuildGpuCostAdapter), deferred, or dropped. A reader reconciling the plan against its own normative grounding cannot tell whether the adapter-factory work is planned, and if the author intended a merge, the 'Cut B' scope in U9 silently grew beyond the run's Cut B definition. A one-line disposition note resolves it.",
      "finding_type": "omission",
      "autofix_class": "manual",
      "confidence": 75,
      "evidence": [
        "[plan Overview] the container itself (Cuts A/B/C/F → the `jtml-production` graph)",
        "[plan R4] execute through existing seams; no new layer; Cut E deferred",
        "[plan U9 files/approach] the :286–291 twin body moves into the adapter; BuildGpuCostAdapter(principal_model, calibration, stage_manager) = the RunDirectStage lambda body + the oracle twin's body",
        "[synthesis.org line 402] Cut D (adapter factory) whenever the oracle is green",
        "[synthesis.org line 124] Cut D moves only the twin's body; the factory must stay stage-shape-agnostic (no hard-coded range/budget/dilation inside)"
      ]
    },
    {
      "title": "Two cited line landmarks do not match the current tree: the stale direct_optimizer.h comment is at :58–59, not :19–20, and optimizer_manager.cpp:1294–1300 contains no early return (the SymTrap zero-pose guard return is at :1304–1308)",
      "severity": "P3",
      "section": "Context & Research (direct_optimizer.h note) / U6 (sym-trap pins)",
      "why_it_matters": "U8's 'fix the stale illustrative comment at direct_optimizer.h:19–20' sends the implementer to lines that contain the extraction header comment — the 'trunk 10k → branch 20k → leaf 30k' comment is at :58–59 in the current tree. U6's relay-count pin 'assert == 60 to catch the silent early return at optimizer_manager.cpp:1294–1300' cites a range that is the tail of RunDirectStage (error return at :1290–1294, writeback through :1300) with no early return in it; the only plausible referent — CalculateSymTrap's INVALID-STARTING-POSE guard ('if (current_optimum_location_.xa == 0 ...) { ... return; }') — sits at :1304–1308, and the plan's own risk table claims 'Research pass re-verified all landmarks at current lines'. An implementer grepping the cited ranges will land on the wrong constructs; re-grep and re-cite both landmarks before U6/U8 land.",
      "finding_type": "error",
      "autofix_class": "gated_auto",
      "confidence": 75,
      "evidence": [
        "[plan Context & Research] Note the stale illustrative comment at direct_optimizer.h:19–20 (\"trunk 10k → branch 20k → leaf 30k\") — fix opportunistically in U8.",
        "[plan U6] `orientationSymTrapUpdated` relay count ≥ 1 (assert == 60 to catch the silent early return at optimizer_manager.cpp:1294–1300)",
        "[current tree, include/domain/direct_optimizer.h:58-59] // keeping a single running counter across stages (trunk 10k -> branch 20k / // -> leaf 30k). Setting the offset makes GetCostFunctionCalls() and the",
        "[current tree, src/coordinator/optimizer_manager.cpp:1294-1300] }\n\n    /*Write the stage result back into the running members.*/\n    cost_function_calls_ = opt.GetCostFunctionCalls();\n    current_optimum_location_ = opt.GetOptimumLocation();\n    current_optimum_value_ = opt.GetOptimumValue();\n}",
        "[current tree, src/coordinator/optimizer_manager.cpp:1303-1308] void OptimizerManager::CalculateSymTrap() {\n    if (current_optimum_location_.xa == 0 &&\n        current_optimum_location_.ya == 0 &&\n        current_optimum_location_.za == 0) {\n        cout << \"ERROR: INVALID STARTING POSE FOR SYMMETRY TRAP\" << endl;\n        return;"
      ]
    },
    {
      "title": "U1 goal claims 'seven cost-path behaviors' but the Approach enumerates eight items",
      "severity": "P3",
      "section": "Implementation Units (U1)",
      "why_it_matters": "The count in the Goal ('the seven cost-path behaviors') does not match the eight semicolon-separated items enumerated in the same unit's Approach (chamfer stage functions; distance-map CropIndexToGlobal; Mahfouz ratio pair; IoU/L1; dilation registry/constants; stage-guard accessor pin; sym_trap tibia-transform pin; DD PolePenalty extraction + init-0/Y-axis pins). The count is only reconcilable if the stage-guard accessor pin is silently excluded from 'cost-path behaviors' — a reading the sentence never states. The enumeration itself is the operative spec, so there is no implementation hazard, but the count should be corrected to 8 or the number dropped so the goal and the inventory agree.",
      "finding_type": "error",
      "autofix_class": "gated_auto",
      "confidence": 50,
      "evidence": [
        "[plan U1 Goal] Characterize the seven cost-path behaviors at the pure surface before any fix — the R13 characterization pass that creates the surfaces the fixes plug into.",
        "[plan U1 Approach] chamfer stage functions; distance-map CropIndexToGlobal full-coverage pin (**corrected formula passes, buggy formula fails** — RED today, this is the spec for U4's fix); Mahfouz float-vs-truncated ratio pair; IoU/L1 with the `IOU(∅,∅)` reference decision (1.0 as spec-with-flagged-deviation); dilation registry/constants pins extended to the lineage values (settings_constants.h IS a Flood tibia transcription — pin it as such); the stage-guard accessor pin; the sym_trap tibia-transform pin; DD PolePenalty extraction + init-0/Y-axis pins."
      ]
    },
    {
      "title": "HLD 'Today's Optimize()' leaf block shows CalculateSymTrap followed by an unqualified RunDirectStage; the code gates the leaf search on !sym_trap_call, and the plan's own costCalls-20000 pins depend on that gating",
      "severity": "P3",
      "section": "High-Level Technical Design (Today's Optimize() column)",
      "why_it_matters": "The 'Today' column presents the leaf block as 'CalculateSymTrap() (directive SymTrap) / RunDirectStage(...)' with no conditional, while the current code runs the leaf search only when 'enable_leaf_ && !error_occurrred_ && !sym_trap_call' — which is exactly why U6/U9 pin 'costCalls() lands at 20000 (leaf skipped — NOT 35000)' for the SymTrap directive. U9 instructs the loop be 'transcribed verbatim'; an implementer transcribing from the Today column alone would add a leaf search under SymTrap and break the 20000 pin (the plan's own gates would catch it, but the diagram should not misstate the current shape). The 'After Cut B' column resolves the case ('Sym_Trap: repeat=0 -> init+dilate+emit, CalculateSymTrap, no search'), so annotating the Today column's RunDirectStage as gated on !sym_trap_call closes the gap.",
      "finding_type": "error",
      "autofix_class": "gated_auto",
      "confidence": 50,
      "evidence": [
        "[plan HLD Today column] leaf block (:1111-1163)\n  init CFM leaf (cfm 2)\n  dilate + emit\n  CalculateSymTrap()  (directive SymTrap)\n  RunDirectStage(...)",
        "[plan U6] costCalls() lands at 20000 (leaf skipped — NOT 35000)",
        "[plan U9] Sym_Trap directive → repeat=0 leaf → no search, costCalls at 20000, relay == 60 (pins from U6 stay green through the relocation)",
        "[current tree, src/coordinator/optimizer_manager.cpp:1138-1151] if (optimizer_settings_.enable_leaf_ && !error_occurrred_ &&\n            !sym_trap_call) { ... budget_ += optimizer_settings_.leaf_budget; ... RunDirectStage(optimizer_settings_.leaf_range, leaf_manager_); }"
      ]
    }
  ],
  "residual_risks": [
    "Line-landmark findings (direct_optimizer.h:19–20, optimizer_manager.cpp:1294–1300) were verified against the current tree; the tree may have shifted since the plan's 2026-08-12 research pass, which the plan's own risk table mitigates by 're-grep before editing'.",
    "The phase-label and R13 findings rely on origin R8 and synthesis.org being the intended reference frame — both are cited by the plan itself as its normative grounding.",
    "The 'seven cost-path behaviors' and HLD leaf-block findings are advisory-level (confidence 50): each has a plausible charitable reconciliation, so route to FYI rather than forcing an edit."
  ],
  "deferred_questions": [
    "Cut D disposition: was the run's 'adapter factory' cut intentionally merged into U9 (Cut B), deferred, or dropped? The plan is silent and the normative base defines it.",
    "Which R13 numbering does the plan intend — the origin's R13 (parallel-eval workstream gates) or the research run's R13 (bit-identical pin surface / characterization pass)? The trace uses origin numbering but the citations match the run's meaning."
  ]
}
```