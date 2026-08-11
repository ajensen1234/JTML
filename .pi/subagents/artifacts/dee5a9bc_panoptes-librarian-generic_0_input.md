# Task for panoptes-librarian-generic

ROUND 2 (deepening) of the optimizer-deep-dive panoptes run (runId i-want-to-deepen-the-optimization-angle-here-both-in-terms-o-mspdhzo444m4, workspace /home/ajj/repo/uf/JTML/.panoptes/optimizer-deep-dive).
Read your EXISTING angle file + the workspace synthesis.org FIRST, then APPEND a '## Round 2' section via the panoptes_write_artifact tool: call it with runId=i-want-to-deepen-the-optimization-angle-here-both-in-terms-o-mspdhzo444m4, path=angles/NN-<slug>.org, content=<your Round 2 section, with its own '## Round 2' header>, append=true. The write is the deliverable; if the write is rejected, RETURN the full Round 2 content in your output instead.
Your round-1 file stays untouched. NEVER spawn subagents. You are read-only except for that sanctioned artifact append.
ANGLE: backend-architecture (append to angles/04-backend-architecture.org)
PERSONA: senior C++/CUDA systems architect for scientific computing - seams that make numerical backends swappable without behavior drift; hates speculative abstraction. Tighten the open ends: (1) the Cut E deferral gate made falsifiable: what specific evidence of 'the driver seam insufficient' would trigger StageKind dispatch + headless executor - name the concrete failing scenario the multi-stage oracle would expose; (2) the StageScript-as-data schema: exact fields (stage kind, range, budget, dilation derivation, repeat semantics for the branch group), how the trunk->branch x2->leaf emit sequence preserves today's dilation/emit ordering bit-identically, where DeriveStageCostParams lives; (3) Cut B's pin strategy: how oracle_test's hand-rolled twin gets absorbed by the script-driven Optimize loop without touching the golden (which assertions move, which stay); (4) the seam contract with the compute-perf double-buffered eval state ('eval state is double-buffered and the cost layer can enqueue before it waits') - who owns the buffers, how the DirectOptimizer injection boundary stays std::function; (5) the torch boundary (contradiction 5): where the linked-but-banned line sits in the target architecture and what would move it; (6) DirectOptimizer::Options cut: the bit-identical-defaults constraint (R13) - how the Options defaults reproduce today's epsilon=0/Jarvis/trisection behavior and the launch-field plumbing through OptimizerRunLaunch. LOCAL CONTEXT: read include/coordinator/optimizer_run_driver.h, src/coordinator/optimizer_run_driver.cpp, include/coordinator/optimizer_manager.h (Initialize :61-77), src/coordinator/optimizer_manager.cpp (Optimize L940-1215, RunDirectStage L1225-1296), include/domain/direct_optimizer.h, docs/solutions/conventions/jtml-shared-vm-layer-2026-08-11.md. Conventions: services never reference coordinator; domain Qt-free; explicit .cpp lists; R13 pin-first.

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