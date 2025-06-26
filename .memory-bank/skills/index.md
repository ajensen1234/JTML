---
description: Registry of available skills and when to use them in this repository.
status: active
---
# Skills

## Installed
- cold-start — bootstrap Memory Bank (done)
- mb-execute — execute a task with protocol + quality gates
- mb-verify — verify task against acceptance criteria
- mb-map-codebase — re-scan codebase and refresh MB
- mb-review — multi-expert MB review
- mb-garden — lint/refactor MB
- superpowers:systematic-debugging — use when encountering bugs
- superpowers:test-driven-development — use before implementing fixes

## When to use
- Bootstrap / memory: `/cold-start`, `/mb-init`
- PRD decomposition: `/prd` (provide prd.md first)
- Task execution: `/execute TASK-ID`
- Verification (UAT): `/verify TASK-ID`
- Review: `/review`, `/mb-review`
- Maintenance: `/mb-garden`, `/mb-sync`
- Harness: `/mb-harness`
