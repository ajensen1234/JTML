---
description: Testing strategy, quality gates, and verification approach for JTML.
status: active
---
# Testing & Verification

## Current state
**No automated test suite.** Two hand-crafted smoke tests exist in `test/`:
- `vtk/` — interactive VTK window (hard-codes a Windows path, no-op on Linux)
- `nfd/` — "Hello World" stub

No ctest, no gtest, no CI (GitHub Actions config exists but is non-functional — targets dormant `actions-test` branch, installs no real deps).

## Quality gates (current)
1. `pixi run build` — must compile with 0 errors
2. `pixi run tidy` — clang-tidy, check for new warnings
3. Manual smoke test: `pixi run run` — verify app launches and basic flow works

## Anti-cheat rules
- Do NOT weaken assertions to make tests pass
- If a test reveals a bug: log in `bugs/known-bugs.md` and fix with explicit scope
- Do NOT ship known BUG-001/002/003 (high severity GPU bugs) without a fix

## Recommended testing approach (no framework yet)
For now, test by building and running. When verifying optimizer changes:
1. Load a known study from `example_studies/`
2. Run optimizer to convergence
3. Compare output pose to expected `.jts` kinematics file

## Future testing gap
GPU kernel correctness has zero automated coverage. The three high-severity GPU bugs (BUG-001, BUG-002, BUG-003) were only found by code inspection. GPU unit tests with known inputs/outputs should be added.

## Artifacts
- Scan reports: `.tasks/TASK-MB-MAP/`
- Bug registry: `.memory-bank/bugs/known-bugs.md`
