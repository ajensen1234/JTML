# Task for panoptes-librarian-source

ROUND 3 (final deepening, wave 2) of the optimizer-deep-dive panoptes run (runId i-want-to-deepen-the-optimization-angle-here-both-in-terms-o-mspdhzo444m4, workspace /home/ajj/repo/uf/JTML/.panoptes/optimizer-deep-dive).
Read your EXISTING angle file + the workspace synthesis.org FIRST, then APPEND a '## Round 3' section via the panoptes_write_artifact tool: call it with runId=i-want-to-deepen-the-optimization-angle-here-both-in-terms-o-mspdhzo444m4, path=angles/NN-<slug>.org, content=<your Round 3 section, with its own '## Round 3' header>, append=true. The write is the deliverable; if the write is rejected, RETURN the full Round 3 content in your output instead.
Your round-1/2 content stays untouched. NEVER spawn subagents. You are read-only except for that sanctioned artifact append.
KEY CAPABILITY: the Zotero corpus is searchable — use zotero_rag_query (semantic over /home/ajj/zotero-paper-text) and read corpus files directly via the read tool (paths in your brief) to cite PRIMARY TEXTS.
ANGLE: compute-perf (append to angles/05-compute-perf.org)
Final deepening with the corpus's rendering/DRR evidence: (1) SANITY BASELINE - Flood 2018 (corpus /home/ajj/zotero-paper-text/f/floodAutomatedRegistration3D2018.txt) reports ~3,000 iterations/sec on a GTX 970 (2018) with 145k evals/frame (~50 s/frame): use it as the cross-era sanity baseline for the instrumentation protocol - the first nvtx/cudaEvent record should report where JTML's per-eval time sits relative to that era's number, and the ms/call meter fix (Cut 4, the 1000x unit error) should be validated against a wall-clock cross-check. (2) RENDER-PATH ALTERNATIVES - the library has the fast-DRR family: dorghamGPUAcceleratedGeneration2012, tornaiFastDRRGeneration2012, abdellahGPUAccelerationDigitally2015 (bindless textures + CUDA/OpenGL interop), gopalakrishnanFastAutoDifferentiableDigitally2022 (auto-diff DRR), kimDifferentiableForwardProjector2023: what do they license for Cut 3 (fragment-fill) and beyond - do any techniques (bindless textures, block projection, splatting) apply to JTML's silhouette rasterizer (src/compute/render_engine.cu), and what would each cost in this hand-written CUDA stack? (3) ITERATION-SPEED CONTEXT - the field's reported registration throughput (Flood's 3k/s on GTX 970, the DRR papers' fps numbers) gives the perf target a concrete reference: what evals/s should Cut 0-2 plausibly reach on the RTX 3090, and what gates should the cuts use?

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