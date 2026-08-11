```json
{
  "reviewer": "ce-scope-guardian-reviewer",
  "findings": [
    {
      "title": "Dangling trace IDs R11 and U9 in U2 and risks table",
      "severity": "P2",
      "section": "U2 (Requirements line) + Risks & Dependencies table",
      "why_it_matters": "The requirement trace is the plan's accountability spine: U2 cites 'R11' but requirements only run R1–R9, and the Risks table cites 'U9' but units only run U1–U8. An implementer or reviewer running the stated trace (every unit maps to a requirement) will fail to resolve these IDs and may misread U2's mandate — the parenthetical '(tokens before extraction)' matches D11, not any requirement. Both are one-token typos, but they break the plan's 1:1 trace claim in exactly the places the plan advertises it.",
      "finding_type": "error",
      "autofix_class": "safe_auto",
      "suggested_fix": "Change 'R11 (tokens before extraction)' to 'D11 (tokens before extraction)' in U2's Requirements line (line 397), and change 'U9 boundary' to 'R9 boundary' in the Risks table (line 847).",
      "confidence": 100,
      "evidence": [
        "**Requirements:** R2, R11 (tokens before extraction), R9",
        "| Scope creep into historical files | U9 boundary is absolute and repeated in every unit's Files lists; `jj diff` reviewed per unit |",
        "Requirements Trace lists exactly R1 through R9 (\"R1. Structured review pass...\" ... \"R9. Scope guard: only `src/app/experimental/**` + new test files + additive `test/CMakeLists.txt` registrations + new docs.\") and Implementation Units lists exactly U1–U8."
      ]
    },
    {
      "title": "Scope guard excludes U3 test-file edits, U7 profiler outputs",
      "severity": "P2",
      "section": "Scope Boundaries + R9 vs U3/U7 Files lists",
      "why_it_matters": "The scope guard — labeled 'absolute' — permits only `src/app/experimental/**`, new test files, additive `test/CMakeLists.txt` registrations, and new docs. U3's own Files list modifies two existing `test/unit/*.cpp` files ('additive cases' is not a new test file), and U7 creates a new top-level `profiler/` directory whose .qtd traces and reports are neither tests nor docs. An implementer or reviewer applying R9 literally will hit an unanswerable guard violation during U3 and U7 — either stalling or silently breaching the boundary the plan repeats as absolute.",
      "finding_type": "error",
      "autofix_class": "gated_auto",
      "suggested_fix": "Amend R9 and the Scope Boundaries enumeration to explicitly include: additive extensions to existing `test/unit/` bridge suites, and skill-mandated artifact outputs under a `profiler/` directory (U7). Requires author sign-off since it widens the stated change set.",
      "confidence": 75,
      "evidence": [
        "No changes outside `src/app/experimental/**` (QML + bridges + main.cpp + CMakeLists + qrc + qmldir), new test files, additive `test/CMakeLists.txt` registrations, and new docs. The owner constraint is absolute...",
        "- Test: `test/unit/experimental_pose_bridge_test.cpp` (additive cases), `test/unit/experimental_ml_bridge_test.cpp` (additive cases) — extend the existing headless suites",
        "Create: `profiler/traces/qmlprofiler-trace-jtml_experimental-*.qtd`, `profiler/reports/profile-report-jtml_experimental-*.md` (skill defaults)"
      ]
    },
    {
      "title": "Pre-U5 profile baseline uncapturable in stated unit order",
      "severity": "P2",
      "section": "U7 (Dependencies + expected hotspot candidates) vs U5 (deliverables)",
      "why_it_matters": "U7 depends on U5 ('U5 (virtualized table)'), yet its expected hotspot analysis calls for a 'pre-U5 baseline' trace of pose-table delegate creation at dataset load — once U5 lands, the table is already a ListView and the before-state is gone. The text further attributes the before/after note to 'U5's report', but U5's verification lists only table tests and manual-visual, with no report deliverable. The U7 implementer must improvise (profile a pre-U5 revision or reorder work) to deliver the promised two standalone reports, and the dangling attribution will mislead.",
      "finding_type": "error",
      "autofix_class": "manual",
      "suggested_fix": "State explicitly where the baseline trace is captured: either profile the pre-change table during U5 before the ListView swap lands (add a U5 profiling step), or have U7 profile a pre-U5 revision via jj. Drop or correct the 'U5's report notes the before/after' attribution.",
      "confidence": 75,
      "evidence": [
        "**Dependencies:** U2, U3 (stable, fixed shell), U5 (virtualized table)",
        "Expected hotspot candidates to verify: pose-table delegate creation at dataset load (pre-U5 baseline — U5's report notes the before/after as two standalone reports)",
        "**Verification:** - Table tests green; manual-visual with a large study (100+ frames): smooth scrolling, correct commits, no flicker/stale cells."
      ]
    },
    {
      "title": "PoseBridge refresh owner vs thin pass-through invariant",
      "severity": "P3",
      "section": "Scope Boundaries + D3 + System-Wide Impact (unchanged invariants)",
      "why_it_matters": "The plan's boundary states bridges stay pass-throughs with 'all behavior' in the seams, and lists 'the thinness rule' among unchanged invariants — yet D3/U3 assign PoseBridge a cross-bridge coordination role (hooking `runStateChanged` and refreshing the table model), which is the single largest behavior added to any bridge. A reviewer applying the stated invariant will push back mid-U3, and the implementer gets no guidance on where the refresh decision is allowed to live. The tension is resolvable but the plan should not claim the invariant is unchanged while amending it.",
      "finding_type": "error",
      "autofix_class": "manual",
      "suggested_fix": "Say explicitly that the refresh owner is plumbing (relay signals + model notify only, no policy) and remains consistent with the thinness rule, or move the refresh trigger view-side with PoseBridge exposing a refresh slot — and update the 'unchanged invariants' wording accordingly.",
      "confidence": 50,
      "evidence": [
        "No changes to the backend seams' public APIs (bridges stay pass-throughs; all behavior stays in the seams).",
        "**Single pose-table refresh owner:** `PoseBridge` hooks `runStateChanged` (completed) and `viewerPoseApplied`/`scenePoseChanged` and refreshes the table model.",
        "**Unchanged invariants:** the render-thread contract, the delegate selection contract (no `QItemSelectionModel`), the thinness rule, the Material-Dark style pin..."
      ]
    },
    {
      "title": "D10 merge documentation promise missing from U6 scenarios",
      "severity": "P3",
      "section": "D10 vs U6 tst_StudyFlows scenario list",
      "why_it_matters": "D10 commits to documenting the models-first merge behavior 'in the flow tests', but U6's tst_StudyFlows load-ordering list covers calibration-first messaging, one-use calibration disable, replace-confirm Yes/No/Esc, and partial-load messaging — no models-first-then-images merge case. The promised pin can silently not ship, leaving the Q7 behavior (merge on `frameCount > 0`) unpinned by the very suite the plan presents as its coverage guarantee. A one-scenario addition closes the trace.",
      "finding_type": "omission",
      "autofix_class": "gated_auto",
      "suggested_fix": "Add an explicit scenario to U6's (a) load-ordering list: images loaded after models-only dataset merge (rule keys preserved) rather than replace — pinning D10's documented behavior.",
      "confidence": 50,
      "evidence": [
        "**Models-without-frames image load stays a merge** (replace rule keys on `frameCount > 0`), documented in the flow tests.",
        "(a) load ordering — calibration-first messages for Images and Models, one-use calibration disable, replace-confirm Yes/No/Esc, partial-load message"
      ]
    }
  ],
  "residual_risks": [
    "Flow-analysis findings M1/M3 never receive a documented disposition in the plan (only M2/M4/M5/M6 are addressed); U1's triage is the catch-all — verify at U1 that they are triaged rather than silently dropped.",
    "The stale-pose-table premise (Dialog content persistence across open/close) is unverified; the plan gates the I1/I2 fixes on an empirical check, so U3/U6 scope shrinks if the premise is false.",
    "D4 seed-invalidation behavior depends on plan-006's stale-seed guard, which the plan itself flags for verification at implementation; if the controller already drops stale seeds, the bridge change and its U6 pin shrink to a view-layer signal."
  ],
  "deferred_questions": []
}
```