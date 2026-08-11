# Task for panoptes-librarian-generic

ROUND 2 (deepening) of the optimizer-deep-dive panoptes run (runId i-want-to-deepen-the-optimization-angle-here-both-in-terms-o-mspdhzo444m4, workspace /home/ajj/repo/uf/JTML/.panoptes/optimizer-deep-dive).
Read your EXISTING angle file + the workspace synthesis.org FIRST, then APPEND a '## Round 2' section via the panoptes_write_artifact tool: call it with runId=i-want-to-deepen-the-optimization-angle-here-both-in-terms-o-mspdhzo444m4, path=angles/NN-<slug>.org, content=<your Round 2 section, with its own '## Round 2' header>, append=true. The write is the deliverable; if the write is rejected, RETURN the full Round 2 content in your output instead.
Your round-1 file stays untouched. NEVER spawn subagents. You are read-only except for that sanctioned artifact append.
ANGLE: measurement-apparatus (append to angles/03-measurement-apparatus.org)
PERSONA: GPU test-infrastructure engineer for numerical/scientific C++ - harnesses that make optimizer behavior measurable; headless-first, GPU where semantics demand. Concretize what round 1 sketched: (1) the perturbation suite that calibrates the IoU 0.85 gate's checked coverage: what regressions trip it, how big a perturbation is visible, suite shape (per-variant perturbation grid vs targeted injections) - this answers 'what size of regression trips the gate'; (2) the z-profile probe as an executable spec: the 31-renders-per-variant sweep (z_trans grid, which variants, the valley-depth metric, the per-variant z-sensitivity ranking hypothesis chamfer-heavy >= Mahfouz >= pure overlap); (3) the multi-stage oracle's seam addition: typed Directive::SymTrap spec (enum value, driver payload, what breaks if omitted) + how the tibia-after-femur pass drives OptimizerRunController past the QML bridge's SingleModelOnly pre-check; (4) the ablation.json schema: full field list (variant x budget shape x dilation schedule x backface x frame x metric -> IoU/L1/z-gap + determinism seeds), the reduced-shape-per-PR vs full-shape-nightly split, the gpu ctest label design; (5) the cross-GPU noise floor plan: what the banded L1 gate needs, what IoU retune looks like, what can be measured today on the single RTX 3090 (seed determinism, run-to-run variance). LOCAL CONTEXT: oracle today = test/oracle/oracle_test.cpp (3 frames, trunk-only, budget 3000, backface OFF); baseline.json authoritative budgets 20k/25k/30k/35k + gates (IoU 0.85 hard, L1 0.0207, z 6.31mm informational); driver seam include/coordinator/optimizer_run_driver.h + OptimizerRunController (plan 006, all green); RunDirectStage injected-cost lambda at src/coordinator/optimizer_manager.cpp:1225+ is the z-probe entry; ctest labels oracle/render with TIMEOUT 3600.

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