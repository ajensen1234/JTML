# Task for panoptes-librarian-generic

ROUND 3 (final deepening, wave 2) of the optimizer-deep-dive panoptes run (runId i-want-to-deepen-the-optimization-angle-here-both-in-terms-o-mspdhzo444m4, workspace /home/ajj/repo/uf/JTML/.panoptes/optimizer-deep-dive).
Read your EXISTING angle file + the workspace synthesis.org FIRST, then APPEND a '## Round 3' section via the panoptes_write_artifact tool: call it with runId=i-want-to-deepen-the-optimization-angle-here-both-in-terms-o-mspdhzo444m4, path=angles/NN-<slug>.org, content=<your Round 3 section, with its own '## Round 3' header>, append=true. The write is the deliverable; if the write is rejected, RETURN the full Round 3 content in your output instead.
Your round-1/2 content stays untouched. NEVER spawn subagents. You are read-only except for that sanctioned artifact append.
KEY CAPABILITY: the Zotero corpus is searchable — use zotero_rag_query (semantic over /home/ajj/zotero-paper-text) and read corpus files directly via the read tool (paths in your brief) to cite PRIMARY TEXTS.
ANGLE: differentiable-polish (append to angles/06-differentiable-polish.org)
PERSONA: differentiable-rendering researcher (unchanged). Final deepening with the full differentiable-rendering corpus now searchable: (1) THE CORPUS SURVEY - verify and extend the round-1/2 MVP with the library's set: liuSoftRasterizerDifferentiable2019 (SoftRas), laineModularPrimitivesHighPerformance2020, zhangPathspaceDifferentiableRendering2020, mehtaTheoryTopologicalDerivatives2023, nicoletLargeStepsInverse2021, wangSimpleApproachDifferentiable2024 + bangaruDifferentiableRenderingNeural2022 (SDF-based), gopalakrishnanFastAutoDifferentiableDigitally2022 (DiffDRR-family), kimDifferentiableForwardProjector2023: which gradient strategy (soft-rasterizer coverage, edge-sampling, SDF reparameterization, forward-projection) is the cheapest to implement against JTML's hand-written CUDA silhouette rasterizer - and does the DiffDRR family change the 'no torch in the cost path' boundary? (2) THE SIMPLEX PRECEDENT - Mahfouz 2003 (corpus /home/ajj/zotero-paper-text/m/mahfouzRobustMethodRegistration2003.txt) originally used Nelder-Mead simplex with negative weights for this exact registration: re-evaluate the polish MVP's gradient choice against a cheap simplex-polish alternative (the ablation arm should include a simplex arm if the surrogate valley is shallow). (3) YAMAZAKI 2004 (corpus /home/ajj/zotero-paper-text/y/yamazakiImprovementDepthPosition2004.txt) is a dedicated depth-position paper: does it report gradient-type z remedies or geometry-based ones - how does that interact with the z-depth-channel NO-GO from round 1? (4) Zotero pulls from round 2 (NVDiffrast etc.) - re-rank now that half the list is in-library: which pulls still matter and which are superseded?

## Acceptance Contract
Acceptance level: checked
Completion is not accepted from prose alone. End with a structured acceptance report.

Criteria:
- criterion-1: Implement the requested change without widening scope

Required evidence: changed-files, tests-added, commands-run, residual-risks, no-staged-files

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