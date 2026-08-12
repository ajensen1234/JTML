# Task for panoptes-librarian-docs

ROUND 3 (final deepening, wave 1) of the optimizer-deep-dive panoptes run (runId i-want-to-deepen-the-optimization-angle-here-both-in-terms-o-mspdhzo444m4, workspace /home/ajj/repo/uf/JTML/.panoptes/optimizer-deep-dive).
Read your EXISTING angle file + the workspace synthesis.org FIRST, then APPEND a '## Round 3' section via the panoptes_write_artifact tool: call it with runId=i-want-to-deepen-the-optimization-angle-here-both-in-terms-o-mspdhzo444m4, path=angles/NN-<slug>.org, content=<your Round 3 section, with its own '## Round 3' header>, append=true. The write is the deliverable; if the write is rejected, RETURN the full Round 3 content in your output instead.
Your round-1/2 content stays untouched. NEVER spawn subagents. You are read-only except for that sanctioned artifact append.
KEY NEW CAPABILITY: the Zotero corpus is searchable — use the zotero_rag_query tool (semantic over /home/ajj/zotero-paper-text) and read corpus files directly (e.g. via the read tool on the paths given in your brief) to cite PRIMARY TEXTS. Corpus files are at /home/ajj/zotero-paper-text/<initial>/<citekey>.txt.
ANGLE: direct-variants (append to angles/01-direct-variants.org)
RE-VERIFY round 1/2 conclusions against the primary texts now available in the Zotero corpus: stripinisNewDIRECTGLhAlgorithm2021 (FULL TEXT - round 1 used the thesis approximation), jonesLipschitzianOptimizationLipschitz1993 (the original DIRECT), gablonskyLocallyBiasedFormDIRECT2001 + gablonskyModificationsDirectAlgorithm2001 (the locally-biased/-gb source), stripinisImprovedSchemeSelection2018 (POH selection), stripinisDGONewDIRECTtype2021 (reference implementation), floodAutomatedRegistration3D2018 (DIRECT-JTA - the tool's lineage). Concrete tasks: (1) verify the GLh surrogate formula + feasibility detection against the paper itself, not the thesis; (2) verify the gb phase-switch numbers against Gablonsky 2001 + the 2022 study's citations; (3) check whether Jones 1993's original epsilon/convergence discussion changes the 'epsilon=0 is pathological' claim; (4) validate the 1-DTC-GL-gb landing variant against the DGO toolbox paper's implementation semantics; (5) the DIRECT-JTA restart explicitly 'sacrifices a notion of global convergence for improved asymptotic performance' (Flood corpus file ~line 445) - how does that tradeoff argument bear on the variant switch for JTML's multistage shape? (6) the analytic battery: what GKLS/DIRECTGOLib specifics does the DGO paper document that round 2's battery spec should adopt?

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