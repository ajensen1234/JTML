# Task for panoptes-librarian-generic

ROUND 3 (final deepening, wave 1) of the optimizer-deep-dive panoptes run (runId i-want-to-deepen-the-optimization-angle-here-both-in-terms-o-mspdhzo444m4, workspace /home/ajj/repo/uf/JTML/.panoptes/optimizer-deep-dive).
Read your EXISTING angle file + the workspace synthesis.org FIRST, then APPEND a '## Round 3' section via the panoptes_write_artifact tool: call it with runId=i-want-to-deepen-the-optimization-angle-here-both-in-terms-o-mspdhzo444m4, path=angles/NN-<slug>.org, content=<your Round 3 section, with its own '## Round 3' header>, append=true. The write is the deliverable; if the write is rejected, RETURN the full Round 3 content in your output instead.
Your round-1/2 content stays untouched. NEVER spawn subagents. You are read-only except for that sanctioned artifact append.
KEY NEW CAPABILITY: the Zotero corpus is searchable — use the zotero_rag_query tool (semantic over /home/ajj/zotero-paper-text) and read corpus files directly (e.g. via the read tool on the paths given in your brief) to cite PRIMARY TEXTS. Corpus files are at /home/ajj/zotero-paper-text/<initial>/<citekey>.txt.
ANGLE: backend-architecture (append to angles/04-backend-architecture.org)
PERSONA: senior C++/CUDA systems architect (unchanged). Validate the architecture against the lineage's actual staging, with primary evidence from the corpus: (1) STAGESCRIPT FIDELITY - Flood 2018 (corpus /home/ajj/zotero-paper-text/f/floodAutomatedRegistration3D2018.txt) documents the real staging: trunk, THREE branch restarts at the same dilation, z-leaf with asymmetric ranges (20 mm z vs 5 mm in-plane), dilation decrease by 4, and the restart hyper-rectangle sized to escape symmetric traps. Verify the StageScript-as-data schema from round 2 can express this config (branch count 3, per-stage dilation derivation, asymmetric ranges) AND JTML's current 2-branch shape - 'stages as data' must cover the lineage, not just today. (2) CONVERGENCE TRADEOFF - the paper explicitly states the restart 'sacrifices a notion of global convergence for improved asymptotic performance' (~line 445): the executor/StageKind design (Cut E) should carry this tradeoff as a documented design note, and the DirectOptimizer::Options work should state where the cover property is intentionally dropped. (3) Cut B pin strategy vs the lineage: oracle_test's hand-rolled twin mirrors the trunk-only flat shape; with Flood's staging as the target shape, which assertions must the script-driven loop preserve? (4) Re-check the torch boundary + the payload evolution against the corpus's differentiable-forward-projector work (kimDifferentiableForwardProjector2023) - does a differentiable X-ray forward projector change the 'no torch in the cost path' boundary?

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