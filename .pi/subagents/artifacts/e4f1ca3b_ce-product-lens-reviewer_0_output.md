I have the full picture now. The repo context is decisive: the handoff this plan itself cites records "NEXT: ... synthesis items 1 → 3 → 2 → 5 → 8", the panoptes synthesis (2026-08-11, one day before this plan) scores QML "Low now" with a "QML: NO-GO now; record the decision" verdict, and the codebase decision record (`src/view/CMakeLists.txt:6`) says to revisit QML only "on a concrete UI need QWidgets cannot deliver, or Qt 6.8+/VTK 9.4+ maturity" — none of which the plan engages. The plan itself is disciplined and well-sequenced (U3 data-integrity before U5 virtualization before U6 tests); the gaps are strategic, not executional.

{"reviewer": "ce-product-lens-reviewer", "findings": [
  {
    "title": "Plan ignores recorded algorithm roadmap it displaces",
    "severity": "P1",
    "section": "Problem Frame / Sources & References / U8",
    "why_it_matters": "The repo's recorded roadmap — written 2026-08-11, one day before this plan — says NEXT is 'synthesis items 1 → 3 → 2 → 5 → 8' (seven live cost-path bugs, metric-ablation harness, the multi-stage oracle scored 'HIGHEST-LEVERAGE SINGLE DIRECTION', DIRECT variants, compute perf), and scores the QML direction 'Low now' with 'QML: NO-GO now; record the decision'. This plan commits 8 units to the experimental QML app without acknowledging, sequencing against, or arguing against that roadmap — even though it cites the very handoff that names those items as next — so an owner reading the plan cannot see the trade-off being made, and the recorded next phase is silently deprioritized. U8 then defers naming 'the next likely phase' to the handoff document itself, so the sequencing decision never lands on record; if the front-end direction is ever abandoned per the decision record's 'revisit on a concrete UI need' trigger, the polish/profiling units (U4/U7) are sunk cost. The plan should state why this pass wins the slot — e.g., the data-integrity fixes and test suite (U3/U6) pin the shared VM-layer bridge contracts that the algorithm roadmap depends on, which the plan never says.",
    "finding_type": "omission",
    "autofix_class": "gated_auto",
    "suggested_fix": "Add a short 'Strategic sequencing' note to Problem Frame: acknowledge the handoff's recorded NEXT (synthesis items 1→3→2→5→8) and the panoptes QML verdict (item 12: 'Low now', decision record at src/view/CMakeLists.txt:6), state that the owner's 2026-08-12 request supersedes/defer them (or state the pass's VM-layer case), and name the next likely phase explicitly in U8 instead of deferring it to the handoff.",
    "confidence": 75,
    "evidence": [
      "Origin: direct owner request (2026-08-12) — \"take a look at the experimental qml stuff ... and have the myriad of qt qml skills while you make a plan on improving it\"",
      "The handoff names the deferred follow-ups (qt_add_qml_module conversion, torch worker thread, shell dirty indicator, widgets-app QML islands) and the next likely phase.",
      "Handoff: `docs/handoff-2026-08-11-vm-layer-extraction.md`",
      "Everything else (widgets app, `src/view`, backend seams, oracle, golden fixtures, packaging) is historical and untouched."
    ]
  },
  {
    "title": "In-scope unsaved-changes guard deferred for lower-value polish",
    "severity": "P3",
    "section": "Key Technical Decisions (D12) / Scope Boundaries — Deferred to Follow-Up Work",
    "why_it_matters": "The shell-level dirty indicator (M6) is the only deferred item that is both in-scope (pure view-layer, `src/app/experimental/**`) and user-facing, yet the plan commits to typography, contrast, hit-target, and keyboard polish (U4) while leaving the one guard that signals unsaved state off the pass. The flow analysis rated it minor and the rationale is scope control, so nothing breaks — but the dirty-state plumbing already exists in SettingsBridge, so folding a shell-level indicator into U4 is a small delta, and absent a concrete trigger for the follow-up, the deferred list becomes where the most user-visible gap in the flow analysis quietly lives.",
    "finding_type": "omission",
    "autofix_class": "gated_auto",
    "suggested_fix": "Either fold the shell-level dirty indicator into U4 (badge + dirty aggregation; SettingsBridge already tracks settings dirty state) or record a concrete trigger for the follow-up (e.g., shipped-vehicle decision), so the deferral is a decision rather than a default.",
    "confidence": 50,
    "evidence": [
      "No shell-level unsaved-changes indicator (M6): deferred.",
      "Scope control; both flagged as acceptable v1 by the flow analysis. Answers Q8."
    ]
  }
], "residual_risks": [
  "Torch GUI-thread freeze (M4) deferral is defensible — the fix needs backend-seam threading changes outside the sandbox — and measuring it in U7 is the right call, but U7's measurement should land in the U8 handoff with an explicit decision trigger; otherwise the app's most visible performance wart can sit quantified-but-unaddressed indefinitely.",
  "The no-qt_add_qml_module decision (D9) is sound from a product standpoint: it only delays IDE/language-server tooling, not user-visible behavior, and the render-smoke/oracle risk argument holds.",
  "Value contingency: if the experimental front-end is later abandoned per the decision record's revisit trigger, the U6 test suite and U8 conventions still retain value for the shared VM layer, but the U4 polish and U7 profiling effort would be sunk; the plan does not state which of its units carry cross-front-end value.",
  "The I1/I2 stale-pose-table fix rests on an unverified premise (Dialog content persistence across open/close); the plan already handles this by verifying empirically in U3 before implementing — accepted risk, correctly managed."
], "deferred_questions": [
  "Did the owner intend the 2026-08-12 request to supersede the handoff's recorded NEXT phase (synthesis items 1→3→2→5→8), and should the U8 handoff name those items explicitly as the follow-on phase?",
  "Is there evidence the owner actually runs jtml_experimental for real studies (vs the widgets app)? This determines whether U4 polish and U7 profiling justify their owner manual-visual time over the algorithm roadmap."
]}