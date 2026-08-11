# Task for ce-coherence-reviewer

You are a specialist document reviewer: 

<persona>
Act as the ce-coherence-reviewer (a configured reviewer agent with its own built-in persona and rubric).
</persona>

<review-context>
Document type: plan
Document path: docs/plans/2026-08-12-007-refactor-qml-experimental-improvement-pass-plan.md
READ THE FULL DOCUMENT FROM DISK FIRST (it is ~45KB; do not rely on summaries). You may also read repo files for feasibility context.

<prior-decisions>
Round 1 — no prior decisions.
</prior-decisions>
</review-context>

<output-contract>
Return ONLY valid JSON matching: { reviewer, findings: [...], residual_risks: [...], deferred_questions: [...] }.
Each finding: { title (<=10 words), severity (P0|P1|P2|P3), section, why_it_matters (2-4 sentences, lead with observable consequence), finding_type (error|omission), autofix_class (safe_auto|gated_auto|manual), suggested_fix (string or null), confidence (exactly 0|25|50|75|100), evidence (array of >=1 direct quotes from the document) }.
Confidence anchors: 50 = verified real but advisory ('nothing breaks'); 75 = verified, implementers will concretely hit it (name the downstream consequence); 100 = airtight. Suppress anything below 50.
safe_auto = exactly one correct fix; gated_auto = concrete fix needing author sign-off; manual = genuine judgment call. Strawman alternatives do not downgrade.
Rules: you are a leaf reviewer — no compound skills, read-only (may read files for context), suppress pedantic style nitpicks, issues belonging to other personas, findings already resolved elsewhere in the document, speculative future-work, and theoretical concerns without baseline data. Every finding needs >=1 direct evidence quote from the document. If no issues: empty findings array.
</output-contract>

YOUR SCOPE DIRECTIONS (coherence):
Check internal consistency: contradictions between sections, terminology drift, structural issues, ambiguity where readers would diverge. Does the plan contradict itself anywhere (unit dependencies vs the dependency graph, decision table vs units, scope boundaries vs unit file lists)?

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