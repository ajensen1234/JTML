# Task for ce-best-practices-researcher

Research current engineering best practices for: (a) slowly refactoring a large untested legacy C++ GUI god object toward MVVM/testability (strangler pattern, characterization testing, dependency injection, seams); (b) choosing between Catch2 and QtTest for a mixed pure-logic + Qt application test suite; (c) golden-master / golden-file regression testing of deterministic compute (numerics tolerance, CPU-vs-GPU determinism concerns for a CUDA/DRR renderer). Planning a refactor of JTML, a Qt5+VTK+CUDA+OpenCV 2D-3D knee-implant registration GUI app (repo root /home/ajj/repo/uf/JTML, C++20, CMake, pixi). Goal: enable fearless editing via (1) a golden-oracle regression gate captured from the known-good Qt5 app (Kneel_1 fixture: silhouettes in example_studies/Kneel_1, femoral STL KR_right_7_fem.stl, pose fem.jts, known-good Labels projections; optimizer settings: trunk/2-branch/1-leaf stages, DIRECT_DILATION, trunk 6px/branch 3px/leaf 1px dilation, trunk range +-30, branch +-20, leaf +-3 with z opened ~100, ~10000 iteration budget); (2) a headless-testable optimize-lifecycle seam: a coordinator owning an idle->running->finished->error(state machine + worker thread, tested with an injected stub cost function under QCoreApplication + QSignalSpy with zero GPU; (3) extracting the pure DIRECT optimizer (ConvexHull/TrisectPotentiallyOptimal/Denormalize/SetSearchRange are currently PRIVATE methods of src/core/optimizer_manager.cpp, hard-coupled to CUDA types GPUMetrics/GPUModel/GPUFrames) behind an injected cost-function boundary (double eval(Point6D)); (4) decomposing MainScreen (src/gui/mainscreen.cpp, 5806 lines) toward MVVM (view/state/services); (5) migrating Qt5->Qt6 (pixi.toml pins qt=5.*; CMake find_package(Qt5); vtk_installer.sh forces -DVTK_QT_VERSION=5 -DVTK_USE_QT6=OFF). There are NO tests (test/ subdir commented out in CMake), NO test framework declared in pixi, and CI (.github/workflows/cmake.yml) is inert boilerplate. test/vtk/test_case is a separate LEFT-knee case. The app entry is src/gui/main.cpp. A prior abandoned attempt added 18 test files + wrapper abstractions (qt_wrappers.h function-pointer wrappers around QFileDialog/QMessageBox, characterize-the-real-god-object tests) rejected as low-value/bunk. AVOID: Qt-mocking wrappers, run-the-real-MainScreen characterization tests, doc sprawl.

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