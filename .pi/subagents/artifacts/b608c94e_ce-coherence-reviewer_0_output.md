All verification complete. I've read the full plan, cross-checked every internal reference, and verified the plan's file/line claims against the working tree (`main.qml`, `SettingsPanel.qml`, `PoseCell.qml`, `Theme.qml`, `pixi.toml`, `test/CMakeLists.txt`, plans 005/006). Key verified facts: the document cites "R11" and "U9" which don't exist in this plan; the dependency graph omits U2→U5 and U2→U6 edges that both units' Dependencies fields and the graph rationale require; the "5 glue blocks (main.qml:417-461)" / "4 glue blocks + ML strip" inventory is stale (actual: 6 `Connections` blocks at lines 419-525, all referencing the viewport; the ML strip at 721-860 never references it); and U7 references a "U5's report" and a "pre-U5 baseline" that cannot exist (U5 produces no report; U7 runs after U5).

```json
{
  "reviewer": "ce-coherence-reviewer",
  "findings": [
    {
      "title": "U2 cites undefined requirement R11",
      "severity": "P2",
      "section": "Implementation Units — U2 (Requirements)",
      "why_it_matters": "The Requirements Trace defines R1-R9 only, so 'R11' resolves to nothing; a reader tracing U2's requirement coverage must guess the intent. The parenthetical '(tokens before extraction)' exactly matches decision D11 ('Theme tokens first, extraction second'), so the intended reference is D11 — and plan-005's R11 (the QmlVtkRenderer seam) is unrelated, ruling out a cross-plan reading.",
      "finding_type": "error",
      "autofix_class": "safe_auto",
      "suggested_fix": "Change U2's Requirements line to 'Requirements: R2, D11 (tokens before extraction), R9'.",
      "confidence": 100,
      "evidence": [
        "**Requirements:** R2, R11 (tokens before extraction), R9",
        "| D11 | **Theme tokens first, extraction second:** extend `Theme.qml` (dirty/badge tokens, TypeScale roles), then re-point every hardcoded color/`font.pixelSize` site, *then* extract components. |",
        "R9. Scope guard: only `src/app/experimental/**` + new test files + additive `test/CMakeLists.txt` registrations + new docs."
      ]
    },
    {
      "title": "Risks table cites nonexistent unit U9",
      "severity": "P2",
      "section": "Risks & Dependencies",
      "why_it_matters": "The plan defines units U1-U8 only, so 'U9 boundary' is unresolvable; a governance check looking for unit U9 finds nothing. The constraint 'repeated in every unit's Files lists' is requirement R9 (the Scope guard), which is indeed restated in every unit — so the intended reference is R9, and the stale ID breaks the risk-mitigation mapping.",
      "finding_type": "error",
      "autofix_class": "safe_auto",
      "suggested_fix": "Change 'U9 boundary' to 'R9 boundary' in the Scope-creep risk row.",
      "confidence": 100,
      "evidence": [
        "| Scope creep into historical files | U9 boundary is absolute and repeated in every unit's Files lists; `jj diff` reviewed per unit |",
        "R9. Scope guard: only `src/app/experimental/**` + new test files + additive `test/CMakeLists.txt` registrations + new docs."
      ]
    },
    {
      "title": "Dependency graph missing U2→U5, U2→U6 edges",
      "severity": "P2",
      "section": "High-Level Technical Design — Unit dependency graph",
      "why_it_matters": "U5's Dependencies field explicitly names 'U2 (PosesTable component)' and U6's names 'U2 (components)', but the mermaid graph draws only U3→U5 and U3/U4/U5→U6. The graph's own rationale says 'U2 is the structural foundation (components + tokens) every later unit edits', and the graph elsewhere draws transitive chains in full (U2→U7 plus U3→U7 plus U5→U7), so the omissions are not a minimal-edge convention. A scheduler reading only the diagram sees U5/U6 as independent of U2 even though both edit files U2 creates.",
      "finding_type": "error",
      "autofix_class": "safe_auto",
      "suggested_fix": "Add 'U2 --> U5' and 'U2 --> U6' edges to the mermaid graph, matching the units' Dependencies fields.",
      "confidence": 100,
      "evidence": [
        "**Dependencies:** U3 (D2 commit contract must exist first — C1/C2 are the data-integrity prerequisite), U2 (PosesTable component)",
        "**Dependencies:** U2 (components), U3 (contracts to pin), U4 (keyboard + contrast to pin), U5 (table to pin)",
        "U3 --> U5[U5 Pose-table virtualization]",
        "Rationale: U1 is the read-only baseline (feeds the triage). U2 is the structural foundation (components + tokens) every later unit edits."
      ]
    },
    {
      "title": "U7 cites nonexistent 'U5's report' and pre-U5 baseline",
      "severity": "P2",
      "section": "Implementation Units — U7 (Approach)",
      "why_it_matters": "U5 produces no report (its Files/Verification list none; profiler reports are U7 deliverables), and U7 runs after U5 (U5→U7 edge; U7's Dependencies: 'U5 (virtualized table)'), so a 'pre-U5 baseline' of delegate creation cannot be captured during U7. D8 compounds the confusion by calling virtualization 'the profiling-driven fix' when the profiling unit lands after it. The implementer writing U7's reports must reconstruct what the parenthetical meant; the actual two-standalone-reports convention belongs to U7's own pre/post hotspot-fix loop.",
      "finding_type": "error",
      "autofix_class": "manual",
      "suggested_fix": "Reword the parenthetical to reference U7's own before/after reports, e.g. 'the O(n) delegate-creation cost that motivated U5 (documented expectation, not a profile); U7 captures before/after as two standalone reports per the skill's no-delta rule'.",
      "confidence": 75,
      "evidence": [
        "Expected hotspot candidates to verify: pose-table delegate creation at dataset load (pre-U5 baseline — U5's report notes the before/after as two standalone reports)",
        "**Dependencies:** U2, U3 (stable, fixed shell), U5 (virtualized table)",
        "The dialog currently instantiates 6 TextFields × every frame (O(n) at load); virtualization is the profiling-driven fix."
      ]
    },
    {
      "title": "Bridge 'no new logic' constraint contradicts D3/D4 behavior",
      "severity": "P2",
      "section": "Implementation Units — U3 (Files/Test) + Scope Boundaries",
      "why_it_matters": "U3's Test line says 'never new logic in the bridges' and the Scope Boundaries say 'bridges stay pass-throughs; all behavior stays in the seams', yet D3 makes PoseBridge the single pose-table refresh owner (hooks runStateChanged/viewerPoseApplied and refreshes the model) and U3's approach adds seed-dropping and enablement-state logic to MlBridge. The constraint and the approach give an implementer opposite instructions about where refresh/seed logic may live; the 'pass-through' wording needs reconciling with the additive-relay design described in System-Wide Impact.",
      "finding_type": "error",
      "autofix_class": "manual",
      "suggested_fix": "Clarify the constraint to 'bridge changes are additive relays only — no behavior changes to existing paths or APIs', or explicitly scope the D3/D4 refresh/seed logic out of the 'no new logic' statement.",
      "confidence": 75,
      "evidence": [
        "extend the existing headless suites, never new logic in the bridges",
        "No changes to the backend seams' public APIs (bridges stay pass-throughs; all behavior stays in the seams).",
        "**Pose-table freshness (I1/I2):** PoseBridge connects `optimizerBridge.runStateChanged` (completed/error) and its own `viewerPoseApplied`/`scenePoseChanged` relay → single table refresh."
      ]
    },
    {
      "title": "Stale glue-block inventory: wrong line range and counts",
      "severity": "P2",
      "section": "High-Level Technical Design — Component wiring contract (+ Context)",
      "why_it_matters": "The extraction contract is built on 'the 5 glue blocks in main.qml (417-461)' and 'the viewport id is referenced from the toolbar, 4 glue blocks, and the ML strip', but the current main.qml has six Connections blocks spanning lines 419-525 — the cited range 417-461 covers only three of them, leaving the optimizer/ml/pose glue at 472-525 out of the extraction inventory. Verified against the file: all six glue blocks reference the viewport id (lines 430-516), and the ML strip (lines ~721-860) never references it, contradicting the '4 glue blocks + ML strip' claim; the two plan passages also disagree with each other (5 vs 4 glue blocks). An implementer planning glue moves will miss blocks or plan re-points for references that don't exist.",
      "finding_type": "error",
      "autofix_class": "manual",
      "suggested_fix": "Re-inventory main.qml's Connections blocks (currently six, at lines 419/447/461/472/493/510: studyBridge ×2, viewport, optimizerBridge, mlBridge, poseBridge) and correct the line range and the '4 glue blocks + ML strip' claim.",
      "confidence": 100,
      "evidence": [
        "5 `Connections` blocks to the bridges (main.qml:417-461)",
        "**Single-owner `Connections` per bridge:** the 5 glue blocks in main.qml (417-461) move to the component that owns the surface they feed (viewport glue stays in the center-panel composition; dialog glue in the dialog components).",
        "**The viewport id is referenced from the toolbar, 4 glue blocks, and the ML strip** — the center region is a component exposing `property alias viewport`, never `findChild`."
      ]
    },
    {
      "title": "Center-region component missing from target file structure",
      "severity": "P2",
      "section": "High-Level Technical Design — Target file structure vs Component wiring contract",
      "why_it_matters": "The wiring contract mandates 'the center region is a component exposing property alias viewport' and 'viewport glue stays in the center-panel composition', but the Target file structure lists no center-region component (main.qml keeps 'region composition') and U2's Create list contains only StudyPanel/MlStrip/RunBar/PosesTable/PosesDialog. The implementer of U2 has no file in which to place the viewport glue, and the 'never findChild' rule has no target; either the structure is missing a center-panel component or the contract's mandate is wrong.",
      "finding_type": "omission",
      "autofix_class": "manual",
      "suggested_fix": "Either add the center-region component (e.g., ViewportPanel.qml) to the Target file structure and U2's Create list, or reword the contract to keep the viewport and its glue inside main.qml's region composition.",
      "confidence": 75,
      "evidence": [
        "**The viewport id is referenced from the toolbar, 4 glue blocks, and the ML strip** — the center region is a component exposing `property alias viewport`, never `findChild`.",
        "main.qml → bootstrap only: Window + toolbar + region composition + FileDialogs (one owner per dialog) + message/replace dialogs.",
        "Create: `src/app/experimental/StudyPanel.qml`, `MlStrip.qml`, `RunBar.qml`, `PosesTable.qml`, `PosesDialog.qml`"
      ]
    },
    {
      "title": "U6 'deliberately last' contradicts U7's code changes",
      "severity": "P3",
      "section": "High-Level Technical Design — Unit dependency graph (Rationale)",
      "why_it_matters": "The rationale says U6 'is deliberately last among the code units so tests exercise final behavior', but U7 is also a code unit that runs after/parallel to U6 (no U6→U7 edge) and modifies components ('Modify: the component(s) the hotspot analysis identifies (expected: PoseBridge notifyCellChanged whole-row dataChanged if confirmed)'). If U7's fixes land after U6's tests, the tests do not exercise the final behavior the rationale claims; the sentence misleads schedulers about U6/U7 ordering.",
      "finding_type": "error",
      "autofix_class": "manual",
      "suggested_fix": "Reword to 'last among the contract-changing code units' or add an explicit U6→U7 note explaining that U7's perf-only changes may follow the tests.",
      "confidence": 50,
      "evidence": [
        "U6 is the aggregation point for the pinned contracts (U3/U4/U5) and is deliberately last among the code units so tests exercise final behavior.",
        "Modify: the component(s) the hotspot analysis identifies (expected: PoseBridge `notifyCellChanged` whole-row `dataChanged` if confirmed — per-role/per-cell notify; any hot bindings the trace shows)"
      ]
    },
    {
      "title": "D3 refresh trigger set differs from U3",
      "severity": "P3",
      "section": "Key Technical Decisions (D3) vs Implementation Units — U3",
      "why_it_matters": "D3 states PoseBridge hooks 'runStateChanged (completed)' while U3 implements 'runStateChanged (completed/error)'. A reader of the decision table alone would refresh only on completed and miss the error-state refresh that U3 specifies; the two trigger sets should match, with U3's fuller set authoritative.",
      "finding_type": "error",
      "autofix_class": "safe_auto",
      "suggested_fix": "Change D3's 'runStateChanged (completed)' to 'runStateChanged (completed/error)' to match U3.",
      "confidence": 50,
      "evidence": [
        "| D3 | **Single pose-table refresh owner:** `PoseBridge` hooks `runStateChanged` (completed) and `viewerPoseApplied`/`scenePoseChanged` and refreshes the table model. |",
        "**Pose-table freshness (I1/I2):** PoseBridge connects `optimizerBridge.runStateChanged` (completed/error) and its own `viewerPoseApplied`/`scenePoseChanged` relay → single table refresh."
      ]
    },
    {
      "title": "U7 profiler artifacts fall outside R9 scope guard",
      "severity": "P2",
      "section": "Implementation Units — U7 (Files) vs R9 / Scope Boundaries",
      "why_it_matters": "R9 and the Scope Boundaries allow only src/app/experimental/** changes, new test files, additive test/CMakeLists.txt registrations, and new docs, but U7's Files list creates a brand-new top-level profiler/ directory (traces + skill-default reports). A .qtd trace file is not a document, so U7's own deliverables violate the letter of the 'absolute' owner scope constraint; a per-unit jj-diff gate enforcing R9 would flag them. profiler/ does not exist in the repo and is not git-ignored, so this is a real new directory.",
      "finding_type": "omission",
      "autofix_class": "manual",
      "suggested_fix": "Amend R9/the Scope Boundaries to explicitly allow profiler/ output artifacts (traces + skill-default reports), or relocate them under docs/.",
      "confidence": 75,
      "evidence": [
        "Create: `profiler/traces/qmlprofiler-trace-jtml_experimental-*.qtd`, `profiler/reports/profile-report-jtml_experimental-*.md` (skill defaults), `docs/reviews/2026-08-12-qml-experimental-profile.md` (repo-visible summary)",
        "R9. Scope guard: only `src/app/experimental/**` + new test files + additive `test/CMakeLists.txt` registrations + new docs.",
        "- No changes outside `src/app/experimental/**` (QML + bridges + main.cpp + CMakeLists + qrc + qmldir), new test files, additive `test/CMakeLists.txt` registrations, and new docs."
      ]
    }
  ],
  "residual_risks": [
    "Line-number claims (Connections at 419-525, ML strip ~721-860, dirty badge ~218-227, Run-closes-dialogs ~975-983, Qt.callLater at 422) were verified against the working tree at review time and will drift if main.qml changes during implementation.",
    "No P0/P1 blockers found; all findings are consistency defects with mechanical or small-rewording fixes.",
    "The 'pre-U5 baseline' issue (F4) leaves a measurement expectation that cannot be satisfied as written; the author should confirm intent before U7 execution.",
    "The center-region component question (F7) changes U2's deliverable list; unresolved it will force an on-the-fly decision during extraction."
  ],
  "deferred_questions": [
    "F7: should the plan add a center-region component (e.g., ViewportPanel.qml) to the target structure, or keep viewport glue inside main.qml's region composition?",
    "F4: is the O(n)-delegate-creation 'baseline' meant as a documented pre-virtualization expectation, or as a trace that must be captured before U5 lands (which would require reordering the graph)?"
  ]
}
```