# Task for panoptes-librarian-source

ROUND 2 (deepening, wave 2) of the optimizer-deep-dive panoptes run (runId i-want-to-deepen-the-optimization-angle-here-both-in-terms-o-mspdhzo444m4, workspace /home/ajj/repo/uf/JTML/.panoptes/optimizer-deep-dive).
Read your EXISTING angle file + the workspace synthesis.org FIRST, then APPEND a '## Round 2' section via the panoptes_write_artifact tool: call it with runId=i-want-to-deepen-the-optimization-angle-here-both-in-terms-o-mspdhzo444m4, path=angles/NN-<slug>.org, content=<your Round 2 section, with its own '## Round 2' header>, append=true. The write is the deliverable; if the write is rejected, RETURN the full Round 2 content in your output instead.
Your round-1 file stays untouched. NEVER spawn subagents. You are read-only except for that sanctioned artifact append.
ANGLE: compute-perf (append to angles/05-compute-perf.org)
Convert round 1's structural claims into measured, ordered cuts: (1) the instrumentation protocol as the first deliverable: nvtx3 ranges + cudaEvent timing placements (per-eval, per-stage, per-frame), the nsys capture recipe, and the exact first-run record (kernel times, D2H stalls, gaps) that replaces the derived 20-60 evals/s estimate; (2) Cut 1's full spec: which 3 of the 5 D2H syncs become async terminal copies + one event wait (render_engine.cu bounding-box/fragment, fast_implant_dilation pixel, distance_map x2), the bit-identity argument per cut (int-atomics order-independence), and the stray cudaGetLastError at render_engine.cu:713 deletion; (3) Cut 2 iteration batching: the double-buffered eval-state design (which buffers, who allocates, the enqueue-before-wait seam with the backend-architecture angle's contract), with a headless-verifiable determinism argument; (4) Cut 3's decision procedure: the fixed FillTriangle grid bound tractability (fragment count pose-dependence) - what measurement triggers the cut, the device-side clamp design, and the safe fallback (keep the sync); (5) Cut 4: the event-based ms/call meter spec (replaces clock() at optimizer_manager.cpp:1274-1281, includes GPU time) and where the UI reads it; (6) the measurement protocol each cut must pass: headless green, oracle green, bit-identity diff empty, measured event-time improvement >= prediction (with revert rule). LOCAL CONTEXT: read src/compute/render_engine.cu (Render L711-836, cudaGetLastError at :713), src/compute/fast_implant_dilation_metric.cu (L159-296), src/compute/distance_map_metric.cu, src/compute/gpu_metrics.cu (int atomicAdd reductions), pixi.toml (nvtx dep :17, -DLIBNVTOOLSEXT :76), src/coordinator/optimizer_manager.cpp (ms/call meter :1274-1281, UpdateDisplay ~30fps). Conventions: measure-first, oracle gates stay green (IoU >= 0.85), headless default. NEVER spawn subagents.

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