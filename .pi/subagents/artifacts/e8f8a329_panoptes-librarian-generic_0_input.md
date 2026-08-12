# Task for panoptes-librarian-generic

ROUND 3 (final deepening, wave 1) of the optimizer-deep-dive panoptes run (runId i-want-to-deepen-the-optimization-angle-here-both-in-terms-o-mspdhzo444m4, workspace /home/ajj/repo/uf/JTML/.panoptes/optimizer-deep-dive).
Read your EXISTING angle file + the workspace synthesis.org FIRST, then APPEND a '## Round 3' section via the panoptes_write_artifact tool: call it with runId=i-want-to-deepen-the-optimization-angle-here-both-in-terms-o-mspdhzo444m4, path=angles/NN-<slug>.org, content=<your Round 3 section, with its own '## Round 3' header>, append=true. The write is the deliverable; if the write is rejected, RETURN the full Round 3 content in your output instead.
Your round-1/2 content stays untouched. NEVER spawn subagents. You are read-only except for that sanctioned artifact append.
KEY NEW CAPABILITY: the Zotero corpus is searchable — use the zotero_rag_query tool (semantic over /home/ajj/zotero-paper-text) and read corpus files directly (e.g. via the read tool on the paths given in your brief) to cite PRIMARY TEXTS. Corpus files are at /home/ajj/zotero-paper-text/<initial>/<citekey>.txt.
ANGLE: measurement-apparatus (append to angles/03-measurement-apparatus.org)
PERSONA: GPU test-infrastructure engineer (unchanged). Deepen with the lineage + field evidence now in the Zotero corpus: (1) Z-LEAF BUDGET - Flood 2018 (corpus file /home/ajj/zotero-paper-text/f/floodAutomatedRegistration3D2018.txt) gave the out-of-plane leaf stage 50,000 evals (equal to trunk) with range 20 mm z vs 5 mm in-plane, explicitly because out-of-plane translation is 'the least accurate degree of freedom for single-plane fluoroscopy'. Update the multi-stage oracle spec: the z-stage budget should be a first-class parameter, and the ablation shapes should include Flood's 50k/3x15k/50k shape vs JTML's 20k/2x5k/5k. (2) Z-PROFILE PROBE vs the field - Yamazaki 2004 (corpus /home/ajj/zotero-paper-text/y/yamazakiImprovementDepthPosition2004.txt) is a dedicated depth-position improvement paper: what remedies does it document (and what accuracy does the field report for depth position in single-plane fluoro)? Fold its findings into the z-profile probe spec + the z-remedies question. (3) STATE-OF-THE-ART BENCHMARKS - burtonFullyAutomaticTracking2023/2024 + jensenAutonomousMethodExtracting2021 + arulampalamAccuracyNovelAutomated2025 report modern automatic 2D-3D registration accuracy: extract comparable numbers (translation/rotation errors) as external benchmarks for the ablation runner's reports. (4) The perturbation suite: does any corpus paper report gate-like sensitivity analysis (Flood's dilation sensitivity table is one - femur robust 8-12, tibia 4-8) that the suite should mirror?

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