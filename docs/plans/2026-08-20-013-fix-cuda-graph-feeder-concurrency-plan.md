---
title: Fix the CUDA-graph greedy feeder hot-spin poll (U7 re-qualification)
type: fix
status: active
date: 2026-08-20
origin: docs/plans/2026-08-20-012-feat-cuda-graph-executor-admission-plan.md
---

# Fix the CUDA-graph greedy feeder hot-spin poll (U7 re-qualification)

## Overview

Plan 012 U7 measured the greedy CUDA-graph `EvaluationExecutor` at **0.114× serial** and recorded a `reverted` verdict in `test/golden/graph_performance_baseline.json`. nsys diagnostics proved this verdict is a **host-bound measurement artifact, not a graph-capability limit**: the feeder's poll loop hot-spins `cudaEventQuery` (1,526,320 calls = 81.8% of all CUDA API time, ~0.5 µs each) while the GPU idles at **1.6% busy** and the launch-to-launch host gap sits at **p50 = 712 µs** on a ~2.36 µs kernel (12412-tri Kneel_1 fixture).

This plan replaces the zero-delay busy poll with a **bounded, event-driven wait discipline** at the greedy-loop / `pollHook_` seam, re-qualifies U7 with the corrected feeder, and makes the retain/revert decision meaningful (GPU-busy as the constrained resource). The U7 gate numbers (1.20× N=2, stage-wall ≤5%, p99 ≤10%, nsys ≥30% concurrent) are **already pre-registered** in `test/golden/graph_pre_registration.json` and are treated as frozen; this plan does not re-negotiate them.

The measured "reverted" verdict stands as the honest frozen baseline — it is the *starting point* this plan fixes, not a conclusion. `test/golden/graph_performance_baseline.json` will be regenerated only after the corrected feeder passes the oracle gates.

---

## Problem Frame

- **User/business problem:** the CUDA-graph batch cost path (the proposed throughput mechanism for POH evaluation in DIRECT registration) cannot be admitted because its first honest measurement is 8.8× slower than serial. The verdict is a host-poll artifact, but until the feeder waits properly, retain is impossible and the 011/012 machinery stays default-deny.
- **Technical root cause (nsys-verified):** `src/compute/evaluation_executor.cpp:299-300` polls every in-flight lease with `pollHook_` every loop iteration, and `:338` backs off with only `std::this_thread::yield()` — no sleep, no backoff. `src/compute/evaluation_executor.cu:34-42` installs the busy `cudaEventQuery` tri-state as `pollHook_`. The host burns a driver round-trip per poll while the GPU is idle.
- **Scope:** the wait/pacing logic of the greedy feeder only. No change to DIRECT semantics, the graph recipe, capture, admission policy shape, or lease bookkeeping. The admitted-N ceiling is exercised by the harness (n_max=4 at `test/oracle/graph_throughput_oracle_test.cu:411`) and the frozen `N_values [1,2,4]` in `test/golden/graph_pre_registration.json`; raising it is **explicitly deferred** (see Scope Boundaries).
- **Success criteria (frozen, not re-negotiated):** benefit = gN2_eps / serial_eps ≥ **1.20×** at batch 16; stage wall regress ≤ **5%**; p99 latency regress ≤ **10%**; nsys shows **≥30% concurrent kernel time at N=2** and **max host-to-device gap < 50 µs**; **zero production-path `cudaStreamSynchronize` / `cudaEventSynchronize` / blocking `cudaMemcpy`** in the admitted path; `nsys` unavailable ⇒ `blocked`, not retained.

---

## Requirements Trace

- R1. The greedy feeder must stop hot-spinning `cudaEventQuery`; the host wait must be bounded so GPU busy% is the constraint, not the poll loop. (Compound finding, fix direction 2/3.)
- R2. The corrected feeder must preserve the executor's existing semantics: out-of-order completion → input-ordered scores, pending leases stay in flight, completed contexts never re-polled, error → PostLaunchAbort + drain + ForceRelease, watchdog → LeavePoisoned + poisoned latch. (Pinned by `test/unit/hook_feeder_test.cpp`.)
- R3. The admitted (graph) path must stay **sync-free**: no `cudaStreamSynchronize` / `cudaEventSynchronize` / blocking `cudaMemcpy` in the concurrent path (R13 of plan 012). Blocking wait is allowed only for the sole-remaining-context / dedicated serialized step, where the capture-invalidator rule permits it.
- R4. Pacing must be **injectable/configurable** so headless fake-hook tests (which use a 1 ms watchdog and exact per-lease poll-count pins) are not broken by fixed sleeps.
- R5. The U7 harness must re-qualify with the corrected feeder and regenerate `test/golden/graph_performance_baseline.json` — with GPU busy% and nsys timeline as the constraint evidence (anti-stub: real `cudaGraphLaunch`, real kernels, non-zero distinct scores).
- R6. The retain verdict, if achieved, must be **machine-qualified** (hostname/GPU/driver/commit recorded) and must not flip production `GraphAdmissionPolicy` (which stays default-deny; runtime opt-in + layered artifact ≥1 + retained verdict is the production contract, out of scope here).

---

## Scope Boundaries

- **In scope:** poll-loop pacing (bounded backoff / event sweep / sole-context blocking wait) in `src/compute/evaluation_executor.cpp` + the CUDA `pollHook_` in `src/compute/evaluation_executor.cu`; a paced, injectable wait primitive; the U7 harness re-qualification run; the regenerated baseline JSON; tests for the new pacing.
- **Explicit non-goals:**
  - No change to DIRECT semantics, the graph recipe, `graph_key_assembler`, `CaptureCoordinator`, or `GraphAdmissionPolicy` shape.
  - No change to lease bookkeeping (`Checkout`/`Recycle`/`ForceRelease`/`LeavePoisoned` semantics).
  - No production wiring (production executor pool stays uninitialized; `optimizer_manager.cpp` default-deny stands).
  - No change to the frozen `N_values [1,2,4]` or the 1.20×/0.90× thresholds in `test/golden/graph_pre_registration.json`.
  - No in-process `cudaDeviceReset`; hang recovery stays terminal/restart.

### Deferred to Follow-Up Work

- **Raising the admitted-N ceiling** (beyond N=4): on the 12412-tri fixture the binding cap is the harness n_max (`graph_throughput_oracle_test.cu:411`) and the frozen `N_values`; the half-memory formula (`bank_state.cuh:281`) is a secondary ceiling. Demonstrating N=8–16 overlap needs either a larger scratch fixture (more triangles → bigger kernels → longer per-eval wall so overlap matters) or a framework admission change — a separate plan (016) after this one proves the wait fix.
- **Production pool initialization + admission evidence wiring** (`optimizer_manager.cpp:806, 1425-1448`): currently unreachable (default-deny); only meaningful after a retained machine-qualified verdict exists.

---

## Context & Research

### Relevant Code and Patterns

- **Hot-spin sites (verified, current tree):**
  - `src/compute/evaluation_executor.cpp:242-345` — hook-driven greedy loop; `:299-300` polls all in-flight leases; `:338` `std::this_thread::yield()` only; `:285-291` same no-backoff in empty-in-flight branch.
  - `src/compute/evaluation_executor.cu:34-42` — `InstallCudaFeederHooks` pollHook: `cudaEventQuery(ev)` tri-state (`cudaSuccess→Done`, `cudaErrorNotReady→Pending`, else `Error`).
  - `src/compute/evaluation_executor.cpp:352-398` — legacy headless stub loop (same shape; gated off when hooks installed; leave alone).
- **Existing blocking-wait precedent:** `src/compute/cost_capacity_service.cu:384,408` — the enqueue/complete callback path already uses `cudaEventSynchronize(event)` after `cudaEventRecord` — documented in-repo distinction between blocking wait (serial completion) and event polling (greedy feeder).
- **The executor seam:** `include/compute/evaluation_executor.h:79-92` (six hook types + installers), `:28` `PollResult` tri-state, `:114` watchdog (5 s default), `:68` `setWatchdogTimeout`. Completion events are `cudaEventDisableTiming` (`evaluation_context.cpp:188-201`).
- **U7 harness:** `test/oracle/graph_throughput_oracle_test.cu` four arms (`:364-423`), steady_clock wall timing, warmup 3 / trials 10, gate at `:433-463`, writes baseline JSON `:466-478`; `makeGraphExec` at `:329-348`.
- **Test seams:** `test/unit/hook_feeder_test.cpp` (fake hooks, poll-count pins), `test/unit/evaluation_context_lease_test.cpp` (pool semantics), `test/oracle/evaluation_executor_graph_test.cu` (anti-stub real-GPU), `test/oracle/layered_correctness_test.cpp` (Layer A/B/C).

### Institutional Learnings

- `docs/solutions/performance-issues/jtml-graph-feeder-cudaeventquery-hotspin-2026-08-20.md` — THE compound finding: root cause, fix directions, prevention rules ("measure GPU busy% before trusting a verdict").
- `docs/solutions/architecture-patterns/jtml-cuda-evaluation-context-executor-2026-08-17.md` (refreshed 2026-08-20) — corrected poll discipline: bounded backoff, blocking wait only for serialized/last context (capture-invalidator rule), sweep-Done, "NEVER busy-poll the entire lease set at zero delay".
- `docs/solutions/logic-errors/jtml-cuda-graph-stub-failure-2026-08-19.md` — anti-stub protocol (no `[x]` on trust; `ctest -L oracle`; nsys kernel census; non-circular tests; `complete() != 0.0`).
- `docs/solutions/workflow-issues/jtml-deepened-unit-not-implementation-ready-2026-08-20.md` — nine readiness questions; admission transaction and failure vocabulary.
- `docs/solutions/conventions/jtml-testability-and-cmake-conventions-2026-08-07.md` — headless vs oracle labels; CMake GLOB trap (new `.cpp`/`.cu` must be in the explicit list at `src/compute/CMakeLists.txt:39-48`).

### External References

- CUDA Programming Guide §2.5.1 (blocking vs non-blocking vs callback), §2.5.7 (`cudaStreamWaitEvent`), §2.5.8 ("synchronization of any kind should be delayed as long as possible"), §4.2 CUDA Graphs (instantiate once, launch ~2.5 µs flat on Ampere; `cudaGraphExec_t` cannot run concurrently with itself ⇒ N distinct instances for N in-flight). https://docs.nvidia.com/cuda/cuda-programming-guide/
- NVIDIA Technical Blog: "Constant Time Launch for Straight-Line CUDA Graphs" (Ampere ~2.5 µs + ~1 ns/node), "Getting Started with CUDA Graphs" (first launch ~33% slower; ~400 µs instantiate once).
- **Key refinement over the compound/blueprint numbers:** the 50–200 µs sleep band is **wrong for 2.36 µs kernels** — 50 µs ≈ 20 kernel executions wide; it would idle the GPU longer than the work lasts. The corrected bound is a **few-ten-µs sleep** (e.g. 10–25 µs) or, better, graph-launch constant-time + multi-event sweep with sub-kernel-window backoff.

---

## Key Technical Decisions

- **Decision 1 — Pacing primitive: injectable bounded-backoff + multi-event sweep, NOT fixed 50–200 µs sleep, NOT `cudaEventSynchronize` in the admitted path.**
  Rationale: CUDA-docs research showed fixed 50–200 µs backoff idles the GPU ~20 kernels wide on this fixture; the guide recommends delaying sync and issuing all independent work first. The admitted path must stay sync-free (R13), so `cudaEventSynchronize` is restricted to the sole-remaining-context / dedicated serialized step (capture-invalidator rule), where the existing `cost_capacity_service.cu:384` precedent lives. The poll remains `cudaEventQuery` tri-state (correct API usage), but paced by an injectable wait strategy.
- **Decision 2 — Wait strategy selection is a hook-level decision, injectable from tests.**
  Rationale: the headless tests (`hook_feeder_test.cpp`) use fake hooks with a 1 ms watchdog and exact poll-count pins (`:213-216`); a fixed sleep inside `RunBatchWithCost` would break them. The pacing lives behind a small injectable seam (e.g. `setPollPacing` or an installed wait-strategy hook with a `sleep_for` override), so headless tests keep tight-loop semantics while the CUDA path paces. Test-only override, no production behavior change.
- **Decision 3 — The single allowed blocking wait is the "last lease" case.** When `inFlight.size()==1 && nextPos==poses.size()` (nothing left to enqueue, one context in flight), the host may block on that context's completion event (`cudaEventSynchronize`) instead of polling — this is the sole case where blocking is safe (no other live work shares the stream) and matches the blueprint's "when only one context remains". The multi-context path stays query+backoff/sweep.
- **Decision 4 — Keep the U7 gate numbers frozen; regenerate the baseline only after the oracle gates pass.** The 1.20×/0.90×/5%/10%/30%/50 µs thresholds are already pre-registered (`graph_pre_registration.json:100-108`); the plan-012 text claiming they are absent (plan `:157`) is stale — this plan does not re-litigate them. The current `reverted` baseline JSON is the honest starting artifact and is regenerated by the harness only when the corrected feeder qualifies.
- **Decision 5 — No new `.cu`/`.cpp` TU unless needed; if a wait-strategy helper is extracted, add it to the explicit source list (`src/compute/CMakeLists.txt:39-48`).** Avoids the CMake GLOB trap.

---

## Open Questions

### Resolved During Planning

- *Is `cudaEventSynchronize` allowed on the graph path?* — Only for the sole-remaining-context / dedicated serialized step (Decision 3), consistent with the capture-invalidator rule and the `cost_capacity_service.cu:384` precedent. The admitted multi-context path stays sync-free.
- *What backoff bound?* — Few-ten-µs (10–25 µs) for the multi-event sweep, not the 50–200 µs band (wrong for µs-scale kernels). Adaptive option deferred.
- *How do tests keep their poll-count pins?* — Pacing is injectable; headless tests use the no-sleep override.

### Deferred to Implementation

- *Exact sleep bound (10 vs 25 µs) and whether adaptive* — tune empirically under nsys; must stay below the per-eval wall target.
- *Whether the sweep services events in index order or oldest-first* — preserve OOO completion (input-ordered result store); decide against the pinned tests.
- *Exact hook/seam shape* — `setPollPacing` vs a wait-strategy hook; implementer chooses the smallest seam that keeps headless tests green.

---

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not implementation specification. The implementing agent should treat it as context, not code to reproduce.*

```text
Greedy loop (src/compute/evaluation_executor.cpp, hook path) with injected pacing:

for pose in input order:
    acquire free context
    enqueue pose work on context.stream
    record completion event (cudaEventDisableTiming)

while in-flight or pending poses:
    # Submit-all-first, then wait (CUDA guide 2.5.8: delay sync)
    poll all in-flight leases once -> {Done, Pending, Error}
    for each Done:
        completeFromPins -> store result at original input index
        recycle context
    if any Done:
        reset watchdog
        continue (immediate re-sweep: completions just happened)
    if inFlight.size()==1 && nextPos==poses.size():
        # Sole-remaining-context: blocking wait allowed (capture-invalidator safe)
        cudaEventSynchronize(completion_event)   # ONLY this case
        completeFromPins -> store -> recycle
    else:
        # Multi-context pending: bounded backoff, then re-poll
        pacingHook_.wait()    # injectable: sleep_for(10-25us) or yield or no-op (tests)
        if watchdog exceeded: LeavePoisoned all in-flight; return WatchdogPoisoned
```

Key points:
- The **poll hook stays `cudaEventQuery` tri-state** (correct API); only the pacing around it changes.
- **All independent work is issued before any waiting** — the current code already enqueues up to `pool_.size()` before polling; keep that ordering.
- The **sweep services only Done events**; pending leases are re-polled after backoff — never hot-spun at zero delay.
- The **sole-context blocking wait** is the only `cudaEventSynchronize` on the hook path, and it is safe because no other live work shares that stream.
- Watchdog stays a host-side steady-clock timer; pacing must keep poll intervals far below the 5 s watchdog floor (and the 1 ms test floor uses the injectable override).

---

## Implementation Units

- [ ] U1. **[Injectable poll pacing in the greedy loop]**

**Goal:** Replace the zero-delay `yield()`-only backoff in the hook-driven greedy loop with an injectable, bounded pacing strategy — while preserving OOO completion, input-ordered scores, watchdog poison, and all pinned poll-count semantics.

**Requirements:** R1, R2, R4

**Dependencies:** None

**Files:**
- Modify: `src/compute/evaluation_executor.cpp` (greedy loop `:242-345`; add pacing seam)
- Modify: `include/compute/evaluation_executor.h` (pacing seam declaration, e.g. `setPollPacing` or wait-strategy hook type + installer)
- Test: `test/unit/hook_feeder_test.cpp` (extend: pacing seam override keeps existing pins green; add a pacing-invocation test)

**Approach:**
- Add a small injectable wait strategy (e.g. `PollPacing` with a `wait()` step: `sleep_for` / `yield` / no-op). Default for the CUDA path: bounded backoff (few-ten-µs) between sweeps; default for headless tests: no-op/yield so existing 1 ms-watchdog and exact poll-count pins hold.
- In the greedy loop: after a sweep with no completions and >1 lease in flight, invoke the pacing wait before the next sweep; on the sole-remaining-context case, use the blocking wait (U2). Preserve "reset watchdog on any Done; continue immediately on Done" so completions are not delayed by backoff.
- Do not touch lease bookkeeping, `Checkout`/`Recycle`, or the `useHooks` gating.

**Execution note:** Test-first. Extend `hook_feeder_test.cpp` first (pacing seam override + a test that the pacing hook is invoked between sweeps), then implement the loop change.

**Patterns to follow:**
- Existing hook installers (`InstallPollHook` etc.) at `include/compute/evaluation_executor.h:87-92` for the seam shape.
- `cost_capacity_service.cu:384` blocking-wait precedent for the sole-context case (U2).

**Test scenarios:**
- Happy path: fake hooks, 3 poses, pool 2 — scores input-ordered despite OOO completion; poll counts unchanged from today's pins (each pending lease polled exactly twice: Pending then Done; doneSet never re-polled).
- Happy path: pacing seam with a `sleep_for` override is invoked between sweeps when nothing completes (spy on the pacing hook; assert ≥1 call with zero completions).
- Happy path: completion immediately after a sweep resets the watchdog and re-sweeps without backoff (completions are not delayed by pacing).
- Edge case: pacing override = no-op — behavior identical to today (existing 1 ms watchdog tests still pass unchanged).
- Error path: pollHook returns `Error` mid-batch — PostLaunchAbort + drain + ForceRelease; pacing must not mask the error (unchanged from today's pins `:264-293`).
- Integration: `test/unit/hook_feeder_test.cpp` whole suite green (poll-count, OOO, watchdog, poisoned-refusal pins unchanged).

**Verification:** All `jtml.hook_feeder` tests pass with the pacing seam; a spy test proves the pacing hook is invoked between zero-completion sweeps; the sole-context blocking path is exercised (U2) with the sync restricted to that case.

---

- [ ] U2. **[Sole-remaining-context blocking wait + CUDA pacing hooks]**

**Goal:** Implement the only allowed `cudaEventSynchronize` (sole-remaining-context) and wire the CUDA-side pacing (bounded backoff) into `InstallCudaFeederHooks`, so the real GPU path paces correctly and stays sync-free in the multi-context admitted path.

**Requirements:** R1, R3

**Dependencies:** U1

**Files:**
- Modify: `src/compute/evaluation_executor.cu` (`InstallCudaFeederHooks`: install the bounded-backoff pacing for the multi-context path; keep `cudaEventQuery` tri-state pollHook)
- Modify: `src/compute/evaluation_executor.cpp` (sole-context case invokes a blocking wait hook — installed by the CUDA feeder as `cudaEventSynchronize`, headless tests leave it unset/no-op)
- Test: `test/unit/hook_feeder_test.cpp` (sole-context path: fake hook returns Done after the blocking-wait hook is invoked)
- Test: `test/oracle/evaluation_executor_graph_test.cu` (extend: nsys/anti-stub — real launches, distinct scores, no per-eval sync in the multi-context admitted path; sole-context case may sync)

**Approach:**
- Add a `WaitSoleContextHookFn`-style hook (or extend the pacing seam) that the CUDA installer sets to `cudaEventSynchronize(ctx->completion_event)`; headless tests install a no-op or fake. The greedy loop calls it only when `inFlight.size()==1 && nextPos==poses.size()`.
- CUDA installer (`evaluation_executor.cu`): install pacing = `std::this_thread::sleep_for(10-25 µs)` (bounded, tunable constant) between sweeps; the 1 ms-test floor is not hit because headless tests never install this installer.
- Keep completion events `cudaEventDisableTiming`; do not add `cudaEventBlockingSync` (the blocking-sync flag is a follow-up decision, not needed for the sole-context case).
- Anti-stub: the oracle must show real `cudaGraphLaunch` + real kernels + non-zero distinct scores, and the nsys timeline must show the multi-context path free of sync gaps (R13).

**Execution note:** GPU-oracle verification is mandatory before marking done (`ctest -L oracle`, nsys kernel census).

**Patterns to follow:**
- `cost_capacity_service.cu:384` (`cudaEventSynchronize(event)` after `cudaEventRecord`) for the blocking-wait precedent.
- `evaluation_executor.cu:34-42` tri-state pollHook shape.

**Test scenarios:**
- Happy path (unit, fake hooks): 1 pose, pool 1 — sole-context blocking-wait hook invoked exactly once; result correct; poll hook not re-polled after Done.
- Happy path (unit): 2 poses, pool 2 — multi-context path never invokes the blocking-wait hook (only the pacing backoff).
- Edge case: sole-context blocking-wait hook returns (i.e. event completes) with an error status — mapped to PostLaunchAbort + drain, not a hang.
- Integration (oracle, real GPU): `evaluation_executor_graph_test.cu` — real graph launches, distinct non-zero scores, input-ordered, no per-eval sync in multi-context path; sole-context case may sync (assert allowed sites only).
- Integration (oracle): nsys on the fixed harness shows GPU busy% up from 1.6% and host-to-device gap well below the 50 µs R13 bound in the multi-context arms.

**Verification:** `jtml.hook_feeder` + `jtml.evaluation_executor_graph` green; nsys on the oracle shows real kernels and a paced (not hot-spun) poll loop; multi-context path sync-free per R13.

---

- [ ] U3. **[U7 re-qualification run + baseline regeneration]**

**Goal:** Re-run the U7 paired harness with the corrected feeder, capture the machine-qualified measurement, and regenerate `test/golden/graph_performance_baseline.json` only if the oracle gates pass.

**Requirements:** R5, R6

**Dependencies:** U2 (and U6 of plan 012 — layered oracle must pass before retain)

**Files:**
- Modify: `test/golden/graph_performance_baseline.json` (regenerated by the harness; only after gates pass)
- Read-only: `test/oracle/graph_throughput_oracle_test.cu`, `test/golden/graph_pre_registration.json`

**Approach:**
- Run `pixi run build`; then the U7 harness under nsys: `nsys profile --stats=true -o /tmp/u7_profile_req .build/bin/jtml_test_graph_throughput_oracle` (per the harness's documented method string).
- Record GPU busy%, concurrent kernel time at N=2 (≥30%), max host-to-device gap (<50 µs), benefit (≥1.20× at batch 16), stage wall (≤5%), p99 (≤10%).
- Verdict logic is in the harness (`graph_throughput_oracle_test.cu:441-463`): `nsys` missing ⇒ `blocked`; N<2 ⇒ N=1 criterion only; else retained iff benefitOk && wallOk && p99Ok. **Do not hand-edit the verdict.**
- Anti-stub: nsys must show real `FillTriangle` / metric kernels (not 4-byte memsets); `complete() != 0.0`; non-circular.
- If the gates fail: document the numbers honestly in the baseline (verdict `reverted` again with reason), keep default-deny, and treat this plan's retain outcome as a no-go with the honest numbers — the compound finding's prevention rule ("measure GPU busy% before accepting a verdict") makes a second host-bound artifact impossible to hide.

**Execution note:** GPU-machine run; `ctest -L oracle` first, then the nsys profile. No `[x]` on trust.

**Patterns to follow:**
- Anti-stub protocol (`jtml-cuda-graph-stub-failure-2026-08-19.md`).
- Frozen pre-registration as the single source of thresholds (`graph_pre_registration.json:100-108`).

**Test scenarios:**
- Integration: four-arm harness runs end-to-end under nsys; verdict artifact written with machine/driver/commit; `admitted_N` recorded.
- Integration: nsys kernel census shows real kernels; GPU busy% reported (target: meaningfully >1.6%, device-constrained).
- Error path: if `nsys` unavailable on the run machine — verdict `blocked`, baseline unchanged, plan outcome recorded as blocked (not retained).
- Regression: `jtml.layered_correctness` still green (Layer A/B/C within frozen tolerance) after the pacing change.

**Verification:** `graph_performance_baseline.json` regenerated with the corrected numbers and machine-qualified fields; the nsys reference stored; verdict reflects the honest gate outcome. If `retained`, the machine-qualified artifact is the input to the (separate, deferred) production admission wiring.

---

- [ ] U4. **[Docs + knowledge store sync]**

**Goal:** Update the handoff, plan-012 residual judgments, and any stale claims so the knowledge store reflects the corrected feeder and the re-qualification outcome.

**Requirements:** (supporting) R6

**Dependencies:** U3 (outcome known)

**Files:**
- Modify: `docs/handoff-2026-08-20-cuda-graph-executor-admission.md` (U7 status: poll fixed; re-qualification outcome; residual judgments updated)
- Modify: `docs/plans/2026-08-20-012-feat-cuda-graph-executor-admission-plan.md` (U7 section: note the corrected feeder + this plan's outcome; fix the stale "no minimum-benefit multiplier" claim at `:157` if still present)
- Modify: `docs/solutions/performance-issues/jtml-graph-feeder-cudaeventquery-hotspin-2026-08-20.md` (append the resolution: the fix landed / re-qualification result)
- Modify: `docs/solutions/architecture-patterns/jtml-cuda-evaluation-context-executor-2026-08-17.md` (note the implemented pacing, correct the 50–200 µs band to the few-ten-µs bound per the CUDA-docs finding)

**Approach:**
- Only after U3 outcome: if `retained` — update the handoff residual judgment "a no-go keeps default-deny" to reflect the machine-qualified retain and note the separate production-wiring follow-up (016). If `reverted` again or `blocked` — record the honest numbers, keep default-deny, and note the compound finding's prevention rule was honored (GPU busy% measured).
- Fix the stale 50–200 µs backoff band in the refreshed blueprint to the corrected few-ten-µs bound (CUDA-docs finding: 50 µs ≈ 20 kernel widths on this fixture).

**Execution note:** Doc-only; no code.

**Patterns to follow:** `docs/solutions/` frontmatter + `jj describe`/`jj new` per logical change (AGENTS.md).

**Test scenarios:**
- Test expectation: none — docs only. (Rationale: no behavioral change; the re-qualification outcome is the artifact.)

**Verification:** handoff/plan/compound/blueprint reflect the corrected feeder + actual U3 outcome; no stale "reverted as graph-capability limit" or "50–200 µs backoff" claims remain.

---

## System-Wide Impact

- **Interaction graph:** the greedy loop (`evaluation_executor.cpp`) → `pollHook_`/pacing seam → `InstallCudaFeederHooks` (`evaluation_executor.cu`) → `EvaluationContextPool` (untouched). The U7 harness and both GPU oracles consume the executor directly.
- **Error propagation:** unchanged — pollHook `Error` → PostLaunchAbort + drain + ForceRelease; watchdog → LeavePoisoned + `WatchdogPoisoned`; pacing must not convert either into a hang (bounded intervals below the watchdog floor).
- **State lifecycle risks:** pacing sleeps must not starve the 1 ms-watchdog headless tests (injectable no-op override) nor the 5 s production watchdog (bounded 10–25 µs intervals are 3–4 orders of magnitude below the floor).
- **API surface parity:** no public API change; the pacing seam is a new installer only. `PollResult` tri-state, hook types, and pool semantics unchanged.
- **Integration coverage:** `jtml.hook_feeder` (fake hooks), `jtml.evaluation_executor_graph` (real GPU, anti-stub), `jtml.layered_correctness` (Layer A/B/C), `jtml.graph_throughput_oracle` (U7 gate).
- **Unchanged invariants:** DIRECT semantics, graph recipe, capture, admission policy shape, lease bookkeeping, completion events stay `cudaEventDisableTiming`.

---

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Backoff bound still idles the GPU (10–25 µs vs ~2.36 µs kernel) | Keep the bound tunable and measure under nsys; the multi-event sweep + graph constant-time launch (~2.5 µs) narrow the host gap; GPU busy% is the gate, not the sleep constant |
| Sole-context `cudaEventSynchronize` violates R13 as written | Restricted to the sole-remaining-context case (capture-invalidator rule); nsys evidence shows the multi-context admitted path sync-free; document the single allowed sync site in U2/U3 |
| Headless tests break on pacing (1 ms watchdog, exact poll counts) | Pacing is injectable; headless tests keep the no-op override; a spy test proves the seam is invoked without changing pins |
| `cudaEventQuery`-based sweep degenerates to hot-spin | The pacing seam guarantees a bounded interval between sweeps; a unit test asserts ≥1 pacing invocation per zero-completion sweep |
| N=4 ceiling still caps overlap; re-qualification may fail the 1.20× gate even with a perfect poll | Honest outcome: the gate numbers stay frozen; a second `reverted` records the true device-constrained number (with GPU busy% evidence) and N-raise moves to the deferred follow-up (016); no fake retain |
| nsys unavailable on the run machine | Verdict `blocked` (harness logic); baseline unchanged; plan outcome recorded as blocked |
| Regenerated baseline is misread as "graph proven" | Baseline stays machine-qualified; production `GraphAdmissionPolicy` remains default-deny; retain only wires admission in the separate deferred follow-up |

---

## Documentation / Operational Notes

- The U7 harness regenerates `test/golden/graph_performance_baseline.json` in place; the commit field is the jj-tracked git ref (`graph_throughput_oracle_test.cu:468`).
- nsys command (harness method string): `nsys profile --stats=true -o /tmp/u7_profile_req .build/bin/jtml_test_graph_throughput_oracle`.
- No production rollout: the executor pool stays uninitialized in production; this plan changes harness/oracle-visible behavior only.

---

## Sources & References

- **Origin document:** `docs/plans/2026-08-20-012-feat-cuda-graph-executor-admission-plan.md` (U7 section `:569-605`, C6 `:63,199`, R13 `:588`)
- Compound finding: `docs/solutions/performance-issues/jtml-graph-feeder-cudaeventquery-hotspin-2026-08-20.md`
- Refreshed blueprint: `docs/solutions/architecture-patterns/jtml-cuda-evaluation-context-executor-2026-08-17.md`
- Anti-stub: `docs/solutions/logic-errors/jtml-cuda-graph-stub-failure-2026-08-19.md`
- Frozen gates: `test/golden/graph_pre_registration.json` (`minimum_benefit` `:100-108`, `N_values [1,2,4]`)
- Current artifact: `test/golden/graph_performance_baseline.json` (reverted verdict)
- External: CUDA Programming Guide §2.5/§4.2, NVIDIA blogs (Constant Time Launch, Getting Started with CUDA Graphs)
