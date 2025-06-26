---
description: Task backlog and execution plan. Empty until PRD is provided.
status: draft
---
# Backlog

> PRD-less rule: backlog skeleton is valid, but waves/TASK-IDs are NOT created until a PRD exists (or explicit human instruction).
> Run `/prd` after providing `prd.md` to decompose future work.

## Conventions
Each task should include:
- goal
- touched files (expected)
- verification steps
- docs-first update

## Task state model
- `Status: planned|ready|in_progress|blocked|done|failed`
- `Wave: W1|W2|W3|...`
- `Depends on: TASK-... | none`

## Known work items (pre-PRD, from codebase scan)

These are candidates for a future PRD. Not formal tasks yet.

### Smart pointer conversion (~30 classes remaining)
- See: [guides/smart-pointers.md](../guides/smart-pointers.md)
- Conversion tool: `class_conversion_prompt.md` in repo root

### Bug fixes (from codebase scan)
- See: [bugs/known-bugs.md](../bugs/known-bugs.md)
- High priority: BUG-001, BUG-002, BUG-003 (GPU correctness)
- High priority: BUG-004, BUG-005, BUG-006 (core correctness)

### Test infrastructure
- Add GPU kernel unit tests (currently zero coverage)
- Fix vtk_tester Windows path
- Wire up ctest

### Build/CI
- Fix GitHub Actions CI (currently non-functional)
- Consider portable CUDA arch flags
- Wire in USE_CUDNN via find_package

---

## Waves (fill after PRD)

_No waves defined yet._
