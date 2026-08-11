# Task for panoptes-librarian-generic

ROUND 2 (deepening, wave 2) of the optimizer-deep-dive panoptes run (runId i-want-to-deepen-the-optimization-angle-here-both-in-terms-o-mspdhzo444m4, workspace /home/ajj/repo/uf/JTML/.panoptes/optimizer-deep-dive).
Read your EXISTING angle file + the workspace synthesis.org FIRST, then APPEND a '## Round 2' section via the panoptes_write_artifact tool: call it with runId=i-want-to-deepen-the-optimization-angle-here-both-in-terms-o-mspdhzo444m4, path=angles/NN-<slug>.org, content=<your Round 2 section, with its own '## Round 2' header>, append=true. The write is the deliverable; if the write is rejected, RETURN the full Round 2 content in your output instead.
Your round-1 file stays untouched. NEVER spawn subagents. You are read-only except for that sanctioned artifact append.
ANGLE: differentiable-polish (append to angles/06-differentiable-polish.org)
PERSONA: differentiable-rendering researcher with deep CUDA/scientific-computing expertise - knows Modular Primitives, DIB-R, NVDiffrast; brutally honest about implementation cost in a hand-written CUDA stack. Move from verdicts to an executable MVP spec: (1) the precondition gate made measurable: what exactly the z-profile probe must show about the chamfer surrogate's valley depth vs the dilated-overlap valley (threshold metric, which z_trans range, how it gates GO); (2) the forward-difference MVP concretely: the 6-renders-per-step gradient (which DOF perturbations, step size, how the existing RenderEngine renders 6 poses), the cost-surrogate choice (chamfer distance-map - confirm which existing metric reads it), and what 'staging the barycentrics FillTriangleKernel already computes' means as a concrete kernel refactor; (3) the analytic projection Jacobian + gradient-of-distance-map kernel design (1 render/step): the math (Jacobian of silhouette projection wrt pose, dD/dx from the distance map), kernel structure, and float-accumulation + production backface/leaf-dilation matching rules; (4) the ablation arm that arbitrates contradiction 6 (12x thesis vs 1.5-3x leaf-phase estimate): DIRECT-only@B vs DIRECT@(B-B_p)+polish@B_p with equal cumulative budget, IoU >= 0.85 gate, per-axis gap recording, and what sample size of frames makes the arm conclusive; (5) the Zotero pull list with per-paper expected value: NVDiffrast, arXiv:1906.07870, Gablonsky thesis, Veriserum, SoftRas/DIB-R, DiffDRR/DiffPose - what each would change about the MVP or the NO-GO list. LOCAL CONTEXT: read src/compute/render_engine.cu (the silhouette rasterizer, ~8 kernels, FillTriangleKernel, 2 D2H syncs), include/compute/frame.h, the papers/ PDFs (Laine 2011 HPG, Modular Primitives TOG 39(6)); the cost path has no torch (torch only in ML services); synthesis says depth-channel machinery 80% present but NOT a z fix. Conventions: measurement-first, R13 pin-first, domain Qt-free. NEVER spawn subagents.

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