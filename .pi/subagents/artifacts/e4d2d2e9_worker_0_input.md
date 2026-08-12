# Task for worker

You are a delegated subagent running from a fork of the parent session. Treat the inherited conversation as reference-only context, not a live thread to continue. Do not continue or answer prior messages as if they are waiting for a reply. Your sole job is to execute the task below and return a focused result for that task using your tools.

Task:
You are the CONTEXT ANALYZER for a compound-learning capture.

Extract the problem context from the description. Classify the TRACK: this is a knowledge-track learning (convention/best-practice about Qt 6.7 ListView delegate patterns — the 'problems' are framework behaviors that were confirmed, not just a single defect).

Return: (a) the track (bug|knowledge) + problem_type enum value; (b) the category directory (per the mapping); (c) the component enum value; (d) module; (e) severity; (f) a suggested filename [sanitized-slug]-2026-08-12.md; (g) the YAML frontmatter skeleton with track-appropriate fields (knowledge track: module, date, problem_type, component, severity, applies_when, tags; bug-track fields only if the track is bug); (h) a 1-2 sentence doc title.

RETURN TEXT ONLY — do not write any files.

CANONICAL FRONTMATTER CONTRACT (docs/solutions/):
Tracks: bug (problem_types: build_error, test_failure, runtime_error, performance_issue, database_issue, security_issue, ui_bug, integration_issue, logic_error) vs knowledge (developer_experience, workflow_issue, best_practice, documentation_gap, architecture_pattern, design_pattern, tooling_decision, convention).
Required both tracks: module (string), date (YYYY-MM-DD), problem_type (enum above), component (enum: rails_model, rails_controller, rails_view, service_object, background_job, database, frontend_stimulus, hotwire_turbo, email_processing, brief_system, assistant, authentication, payments, development_workflow, testing_framework, documentation, tooling), severity (critical|high|medium|low).
Bug track additionally REQUIRES: symptoms (array 1-5), root_cause (enum: missing_association, missing_include, missing_index, wrong_api, scope_issue, thread_violation, async_timing, memory_leak, config_error, logic_error, test_isolation, missing_validation, missing_permission, missing_workflow_step, inadequate_documentation, missing_tooling, incomplete_setup), resolution_type (enum: code_fix, migration, config_change, test_fix, dependency_update, environment_setup, workflow_improvement, documentation_update, tooling_addition, seed_data_update).
Knowledge track: no extra required fields; optional applies_when (array 1-5).
Optional both: related_components (array), tags (array <=8, lowercase hyphen-separated).
Category mapping: build_error->build-errors/, test_failure->test-failures/, runtime_error->runtime-errors/, performance_issue->performance-issues/, database_issue->database-issues/, security_issue->security-issues/, ui_bug->ui-bugs/, integration_issue->integration-issues/, logic_error->logic-errors/, developer_experience->developer-experience/, workflow_issue->workflow-issues/, best_practice->best-practices/, documentation_gap->documentation-gaps/, architecture_pattern->architecture-patterns/, design_pattern->design-patterns/, tooling_decision->tooling-decisions/, convention->conventions/.
YAML safety: array items starting with ` [ * & ! | > % @ ? or containing ': ' must be double-quoted.
Doc templates: bug track = Problem / Symptoms / What Didn't Work / Solution / Why This Works / Prevention / Related Issues. Knowledge track = Context / Guidance / Why This Matters / When to Apply / Examples / Related.

PRIMARY PROBLEM TO DOCUMENT (repo: /home/ajj/repo/uf/JTML, Qt 6.7.2 QML app):
During plan 007's QML improvement pass + the follow-up code-review round, four Qt 6.7 ListView-delegate behaviors were confirmed the hard way (each cost a debugging session; two were only caught by a new headless Qt Quick Test harness):
1. ListView.view (the attached property) is NULL inside NESTED delegate items (only the delegate ROOT receives it) — the frame-picker regression: onClicked: ListView.view.currentIndex = index threw a silent TypeError per click; fix: use the list's id.
2. Qt 6.7 ListView pooling does NOT drop focus (hidden/reparented items keep activeFocus) — the 'focus loss fires editingFinished before pooling' commit-on-pool assumption never holds; fix: an edited flag + explicit commitIfEditing() flushed from the ATTACHED ListView.onPooled handler (delegate-root `signal pooled()`/onPooled: never fire — the pool notifications are ListView-attached signals that the delegate must DECLARE).
3. When a delegate declares required properties (e.g. required property int frameIndex), the IMPLICIT model context object goes STALE on pool reuse — frameIndex re-bound to 3 while model.x stayed 0 (test-proven diagnostic); fix: also declare `required property var model` so the model re-binds through the required-property mechanism. Related: required property double x/y/z on a delegate root collides with Item's FINAL geometry properties ('Cannot override FINAL property' — app unloadable); name roles anything but geometry names.
4. Escape/arrow keys on a QQC2 TextField: Keys.onPressed on the control root captures them before the internal editor (needed for the D7 cell-navigation contract: Left/Right between cells, Up/Down rows, Esc reverts).
Evidence files: src/app/experimental/PosesTable.qml (delegate with required frameIndex + required model + attached onPooled/onReused + onFrameIndexChanged flush + focusCell/moveToRow), PoseCell.qml (edited flag, commitIfEditing, resetDisplay with focus-retained tuple re-capture, Component.onDestruction commit, Keys.onPressed), StudyPanel.qml (the ListView.view-nested fix), test/qml/tst_PosesTable.qml + tst_PoseCell.qml (the pins).
Context: the harness (test/qml/, quick_test_main, fakes injected via required properties) is what proved #2 and #3; the trace-driven debugging method (qmlprofiler event census) found #1.
EXISTING DOC (likely high overlap): docs/solutions/conventions/jtml-qml-view-testing-2026-08-12.md — a conventions entry covering the harness recipe, qmllint gate, profiling build, and a 'QML gotchas confirmed on this stack' section that already lists #1 (ListView.view nested), #2 (pooling focus + attached signals), and the FINAL-collision — but NOT #3 (the stale implicit model with required properties, discovered later in the review round) and NOT #4. Read it.

## Acceptance Contract
Acceptance level: attested
Completion is not accepted from prose alone. End with a structured acceptance report.

Criteria:
- criterion-1: Return concrete findings with file paths and severity when applicable

Required evidence: review-findings, residual-risks

Finish with a fenced JSON block tagged `acceptance-report` in this shape:
Use empty arrays when no items apply; array fields contain strings unless object entries are shown.
`criteriaSatisfied[].status` must be exactly one of: satisfied, not-satisfied, not-applicable.
`commandsRun[].result` must be exactly one of: passed, failed, not-run.
`manualNotes` and `notes` are optional strings; an empty string means no note and does not satisfy `manual-notes` evidence.
```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "specific proof"
    }
  ],
  "changedFiles": [
    "src/file.ts"
  ],
  "testsAddedOrUpdated": [
    "test/file.test.ts"
  ],
  "commandsRun": [
    {
      "command": "command",
      "result": "passed",
      "summary": "short result"
    }
  ],
  "validationOutput": [
    "validation output or concise summary"
  ],
  "residualRisks": [
    "none"
  ],
  "noStagedFiles": true,
  "diffSummary": "short description of the diff",
  "reviewFindings": [
    "blocker: file.ts:12 - issue found, or no blockers"
  ],
  "manualNotes": "anything else the parent should know"
}
```