# Task for panoptes-librarian-source

ROUND 3 (final deepening, wave 1) of the optimizer-deep-dive panoptes run (runId i-want-to-deepen-the-optimization-angle-here-both-in-terms-o-mspdhzo444m4, workspace /home/ajj/repo/uf/JTML/.panoptes/optimizer-deep-dive).
Read your EXISTING angle file + the workspace synthesis.org FIRST, then APPEND a '## Round 3' section via the panoptes_write_artifact tool: call it with runId=i-want-to-deepen-the-optimization-angle-here-both-in-terms-o-mspdhzo444m4, path=angles/NN-<slug>.org, content=<your Round 3 section, with its own '## Round 3' header>, append=true. The write is the deliverable; if the write is rejected, RETURN the full Round 3 content in your output instead.
Your round-1/2 content stays untouched. NEVER spawn subagents. You are read-only except for that sanctioned artifact append.
KEY NEW CAPABILITY: the Zotero corpus is searchable — use the zotero_rag_query tool (semantic over /home/ajj/zotero-paper-text) and read corpus files directly (e.g. via the read tool on the paths given in your brief) to cite PRIMARY TEXTS. Corpus files are at /home/ajj/zotero-paper-text/<initial>/<citekey>.txt.
ANGLE: cost-path-foundation (append to angles/02-cost-path-foundation.org)
RESOLVE the two open questions with primary texts now in the Zotero corpus: (1) MAHFOUZ NORMALIZATION - the full text (corpus file /home/ajj/zotero-paper-text/m/mahfouzRobustMethodRegistration2003.txt) documents a weighted combination of an intensity product metric (deliberately NOT normalized) + a contour/edge overlap weighted MORE heavily than area; the original optimizer is Nelder-Mead simplex with negative weights. Execute the kernel-as-spec decision rule from round 2: do the kernel's 2.55 scale + host x255 + the intensity/contour combination match the paper's design (read src/compute/implant_mahfouz_metric.cu), and what should the Tier-0 reference tests pin? (2) DILATION CONTRADICTION - the Flood 2018 full text (corpus file /home/ajj/zotero-paper-text/f/floodAutomatedRegistration3D2018.txt) documents femur trunk dilation 10 / tibia 6, branch = trunk - 4, leaf = 1; budgets trunk 50k / 3 branches x 15k / z-leaf 50k; branch ranges 15/15/25/25/25/25; leaf 5/5/20/5/5/5; Canny aperture 3, thresholds 40/120 (tibia) or 30/80 (femur A). Resolve the three-way contradiction (code 6/4/1 vs docs 6/3/1 vs Flood femur 10/6/1 + tibia 6/2/1) with a recommendation, and update the dilation-schedule ablation (synthesis item 7) to test the lineage configs. Also: (3) the Canny divergence (oracle 3/0/150 vs Flood 40/120) - flag for the re-baseline decision. (4) Jensen 2024 (jensenNovelPostprocessingTechnique2024) on symmetric-implant ambiguity - what does the field's current sym-trap remedy suggest for the sym_trap cost function's future?

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