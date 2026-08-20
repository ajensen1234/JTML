---
title: "A deepened remaining unit is not automatically implementation-ready"
date: 2026-08-20
category: docs/solutions/workflow-issues
module: JTML CUDA cost evaluation planning
problem_type: workflow_issue
component: development_workflow
severity: high
applies_when:
  - "A later plan unit is marked deepened or reviewed and the next prompt says implement it"
  - "CUDA, GPU, or other lifecycle work still has open admission, ownership, capture, or teardown design"
  - "A CUDA-aware or adversarial review listed unresolved blockers on a unit that already has an approach section"
tags:
  - planning
  - cuda-graph
  - admission
  - lifecycle
  - plan-012
  - plan-011
related_components:
  - documentation
  - tooling
---

# A deepened remaining unit is not automatically implementation-ready

## Context

Plan 011 U1–U5 landed a real monoplane `DIRECT_DILATION` graph recipe, private evaluation contexts, and a frozen rev-2 measurement contract. The leftover work was written as deepened U6–U8. Handoffs and `next-prompt.md` then told the next agent to implement U6.

A later CUDA-aware review showed that U6 was still design work: empty `{}` vectors could not represent R8 after `SetBatchCost` was installed, `createGraph` returned a `GraphExecWrapper*` while pool `Shutdown` treated `ctx.graph_exec` as a raw `cudaGraphExec_t`, Global capture had no app-wide quiescent boundary, recipe capture required `in_flight` while `Recycle(false)` did not un-check, hang teardown was unspecified, post-launch failures escaped as `std::invalid_argument`, and production default-deny / honest U7–U8 gates were not closed.

The correct next step was a **fresh Plan 012**, not coding 011 U6. The existing stub-failure learning (`jtml-cuda-graph-stub-failure-2026-08-19.md`) already forbids marking *code* done from circular tests. This learning is one stage earlier: do not start `ce-work` on a unit whose admission and lifecycle are still unresolved.

## Guidance

Treat “deepened” as “the unit has more words,” not “the unit is implementable.”

Before starting implementation on remaining GPU / executor / lifecycle work, the unit must answer all of these. If any answer is “invent it while coding,” stop and write a replacement plan.

1. **Admission transaction.** Every fallible key, provider, park, capture, and instantiate step has a named moment *before* the production `SetBatchCost` (or equivalent) is replaced. Failure at that moment leaves the previous supported path installed.
2. **Failure vocabulary.** Empty vectors, wrong-size vectors, and exceptions already used for contract bugs cannot also mean “not admitted.” Typed outcomes exist, or the unit is not ready.
3. **Ownership.** Every resource has one destroyer. If `createGraph` returns a wrapper and a pool destructor assumes a raw CUDA handle, the unit must say which map owns the wrapper and that the other field stays null.
4. **Capture / concurrent submission.** If capture mode is Global, the unit names the thread that begins and ends capture and how competing CUDA/UI work is parked or refused. “Capture on the feeder thread” is not a coordinator.
5. **Lease / `in_flight` protocol.** If capture requires a checked-out context, prepare vs launch leases are distinct, and abort has an explicit un-check (`ForceRelease`) vs poison (`LeavePoisoned`). `Recycle(false)` that returns without un-checking is a leak, not a protocol.
6. **Hang vs ordinary error.** Ordinary CUDA errors drain and release. A watchdog or device hang must not `Synchronize`/`Free` in-flight buffers. Recovery is named (usually process restart), not left as “maybe reset the device.”
7. **Error conversion.** Post-launch failures reach the existing user-visible error path (`OptimizerError` here). They do not escape a worker thread as `std::invalid_argument`.
8. **Default-deny retain gate.** Production stays off until a real correctness oracle and a machine-qualified measurement exist. A serial-passthrough comparison or a missing `nsys` timeline is not retain.
9. **Headless vs GPU TUs.** If headless tests link only the `.cpp`, the unit cannot direct-call a `.cu` symbol. Hooks or an equivalent CUDA-free seam are specified.

If a review already listed those as open blockers, **do not deepen the old unit in place and then implement it.** Open a new plan that supersedes the remaining units. Keep landed units checked. Point `AGENTS.md` and the handoff at the new plan.

Product requirements (origin R-IDs) can stay stable while lifecycle constraints (C-IDs) are still being designed. Mixing those two layers in one “just wire U6” unit is how the plan went wrong.

## Why This Matters

Coding an unresolved lifecycle unit produces one of two failures JTML has already seen:

- **Stubs that pass circular tests** — the anti-pattern in `jtml-cuda-graph-stub-failure-2026-08-19.md`.
- **A real-looking wiring attempt that ships the wrong admission semantics** — U12 overwritten by executor serial passthrough, `{}` becoming `invalid_argument`, wrappers double-freed at shutdown, or capture racing the UI.

Both waste a GPU session and create false confidence. Re-planning is cheaper than either.

## When to Apply

- A handoff says “U6 is deepened, implement it” after a CUDA or lifecycle review listed open blockers.
- Remaining work sits on `SetBatchCost`, graph capture, stream teardown, or production feature flags.
- Two documents disagree: the plan checkbox is open, but `next-prompt.md` treats the approach as source of truth.
- You are tempted to “just add a coordinator in the same unit” while writing code.

Do not apply this to small, already-landed units with a closed ownership story (Plan 011 U1–U5 stay implementable history). Do not use it to delay a unit that already has typed outcomes, a destroyer, a capture boundary, and a deny-by-default gate.

## Examples

**Before (wrong):** Plan 011 U6 is long, five personas reviewed it, `next-prompt.md` says implement the hook seam. The next agent starts `ce-work` on U6. Capture coordination, R8-after-install, and hang teardown are invented mid-diff — or stubbed.

**After (right):** The same review is read as “U6 is not a coding unit.” Write Plan 012 with C1–C11 resolved in Key Technical Decisions, new U-IDs, and `AGENTS.md` pointing at 012. Do not reimplement 011 U1–U5. Do not check 011 U6.

**Related split of duties:**

| Learning | Stage it guards |
|---|---|
| This doc | Planning completeness *before* `ce-work` |
| `jtml-cuda-graph-stub-failure-2026-08-19.md` | Verification completeness *before* `[x]` |
| `jtml-cuda-evaluation-context-executor-2026-08-17.md` | Target architecture once the unit is actually implementable |
| `graph-tiered-correctness-2026-08-19.md` | Aspirational Layer A/B/C gate until Plan 012 U6 proves it |

## Related

- `docs/plans/2026-08-20-012-feat-cuda-graph-executor-admission-plan.md` (replacement plan)
- `docs/plans/2026-08-19-011-feat-cuda-graph-greedy-evaluation-executor-plan.md` (U1–U5 landed; U6–U8 superseded)
- `docs/handoff-2026-08-19-cuda-graph-greedy-evaluation-executor.md`
- `docs/solutions/logic-errors/jtml-cuda-graph-stub-failure-2026-08-19.md`
- `docs/solutions/architecture-patterns/jtml-cuda-evaluation-context-executor-2026-08-17.md`
- `docs/solutions/architecture-patterns/graph-tiered-correctness-2026-08-19.md` (aspirational)
- `docs/brainstorms/2026-08-19-cuda-graph-greedy-evaluation-executor-requirements.org`
