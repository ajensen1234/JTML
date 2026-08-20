---
title: Fix the CUDA-graph greedy feeder hot-spin poll (U7 re-qualification)
type: fix
status: active
date: 2026-08-20
deepened: 2026-08-20
origin: docs/plans/2026-08-20-012-feat-cuda-graph-executor-admission-plan.md
---

# Fix the CUDA-graph greedy feeder hot-spin poll (U7 re-qualification)

## Overview

Plan 012 U7 measured the greedy CUDA-graph `EvaluationExecutor` at **0.114× serial** and recorded a `reverted` verdict in `test/golden/graph_performance_baseline.json`. nsys diagnostics showed the verdict is dominated by a **host-bound hot-spin artifact**: the feeder's poll loop calls `cudaEventQuery` 1,526,320 times (81.8% of all CUDA API time, ~0.5 µs each) while the GPU idles. The launch-to-launch host gap (p50 = 712 µs) dwarfs the work being fed.

This plan replaces the zero-delay busy poll with a **bounded, event-driven wait discipline** at the greedy-loop / `pollHook_` seam, re-qualifies U7, and makes the retain/revert decision meaningful. Critically, this revision corrects the **work-quantum framing**: the meaningful GPU work quantum is ~97 µs per-eval residency (pre-registered cut0 GPU-event 97.2 µs), not the 2.36 µs mean kernel — so the honest device ceiling is ~4.6× (N=4 perfect CPU packing), not 60×. A second `reverted` verdict is a **real possible outcome** of this plan, not just a deferred-N artifact; the plan's U0 probe decides reachability *before* spending the gate run.

## Problem Frame

- **User/business problem:** the CUDA-graph batch cost path cannot be admitted because its first honest measurement was 8.8× slower than serial, and the verdict was a host-poll artifact. Until the feeder waits properly, retain is impossible and the machinery stays default-deny.
- **Technical root cause (nsys-verified):** `src/compute/evaluation_executor.cpp:299-300` polls every in-flight lease with `pollHook_` every loop iteration; `:338` backs off with only `std::this_thread::yield()` — no sleep, no backoff. `src/compute/evaluation_executor.cu:34-42` installs the busy `cudaEventQuery` tri-state as `pollHook_`. The host burns ~0.5 µs/query while the GPU idles.
- **Scope:** wait/pacing logic of the greedy feeder only. No change to DIRECT, the graph recipe, capture, admission policy shape, or lease bookkeeping. Raising admitted-N (beyond n_max=4) and production wiring stay **explicitly deferred** (Scope Boundaries).
- **Success criteria:** the frozen gates (benefit = gN2/serial ≥ **1.20×** at batch 16; wall ≤5%; p99 ≤10%; nsys ≥30% concurrent at N=2; <50 µs gap; zero production-path sync) are **pre-registered** in `test/golden/graph_pre_registration.json` and are the gate **only after** the U0 probe (below) confirms they are reachable at N≤4. The U7 gate itself is **contested by this plan's honest band** (1.1–1.30× at N=2): the plan keeps the 1.20× threshold but does **not** treat a second `reverted` as failure *if the probe bounds show the gate is not within device reachability*. This distinction is the anti-false-confidence posture.

The frozen "reverted" verdict stands as the honest baseline — the starting point this plan fixes. `graph_performance_baseline.json` is regenerated **only** when the corrected feeder qualifies through the oracle + nsys gates, and the regeneration is itself verified by read-back assertion (M9).

---

## Problem Statement

- **Root cause (nsys-verified, ¥F1):** host-bound hot-spin of `cudaEventQuery`.
- **Measured magnitude (baseline.json:10, compound finding):** graph N=1 1207 µs/pose vs serial 111 µs/pose (batch 16) — the spin accounts for most of the 10.9×.
- **Amdahl reality (performance-audited):** serial per-pose **111 µs** wall, of which **~97 µs is GPU residency** (cut0 GPU-event 97.2 µs). Serial is already ~87% device-saturated. Overlapping N device-saturated evals caps N=2 at ~2.06× perfect, N=4 at ~4.6× perfect. **60× is a misreading of GPU-idle; the true ceiling is ~4.6×.** The 1.20× gate at N=2 needs overlap fraction **f ≥ ~0.28** with per-pose host ≤ ~12 µs — thin, and measurable only by an nsys SM census (U0), not inferable from the spin numbers.

## Requirements Trace
- R1. Stop the hot-spin; bounded, event-driven wait so GPU busy% is the constraint. (Compound dir. 2/3; perf review.)
- R2. Preserve executor semantics: OOO → input-ordered scores; pending leases stay in flight; no re-poll of completed; error → PostLaunchAbort + drain + ForceRelease; watchdog → LeavePoisoned + poisoned latch. (Pinned by `test/unit/hook_feeder_test.cpp`.)
- R3. The admitted (graph) path must stay **sync-free** — no `cudaStreamSynchronize` / `cudaEventSynchronize` / blocking `cudaMemcpy` — **including the sole-remaining-context case** (this revision: the `zero_sync` frozen gate admits no carve-out). The sole-context tail uses a **bounded timed wait** (query+pacing loop), never a blocking sync. (Testing F1, adversarial F-C; frozen `graph_pre_registration.json:97`.)
- R4. Pacing is **injectable/configurable** (headless fake-hook tests use a 1 ms watchdog + exact poll-count pins; a fixed sleep breaks them).
- R5. U7 harness re-qualifies with the corrected feeder and regenerates the baseline **only with machine-qualified + nsys-constraint + layered-verdict assertion before `retained`** (testing F3/F6) and a read-back assertion (M9).
- R6. Retain is machine-qualified (hostname/GPU/driver/commit) and does **not** flip production `GraphAdmissionPolicy` (default-deny unchanged; runtime opt-in + layered artifact ≥1 + retained verdict is a separate follow-up).

## Scope Boundaries

- **In scope:** poll-loop pacing (event sweep + adaptive wait) in `src/compute/evaluation_executor.cpp` + CUDA `pollHook` in `src/compute/evaluation_executor.cu`; injectable wait primitive; U0 probe; U7 harness extension (nsys + layered + readback assertions); baseline regeneration; tests.
- **Explicit non-goals:**
  - No DIRECT/graph-recipe/capture/admission-policy change.
  - No lease bookkeeping change (`Checkout`/`Recycle`/`ForceRelease`/`LeavePoisoned`).
  - No production wiring.
  - No change to frozen `N_values [1,2,4]` or thresholds (`graph_pre_registration.json`).
  - No in-process `cudaDeviceReset`.

### Deferred to Follow-Up Work

- **N-raise (>4) to plan 016** — the harness n_max (`graph_throughput_oracle_test.cu:411`), the frozen `N_values`, and the half-memory admit formula (`bank_state.cuh:281`) all cap at 4. Demonstrating N=8–16 needs a bigger scratch fixture or an admission change — after this plan proves the wait + U0-probe result.
- **Production pool init + admission evidence wiring** (`optimizer_manager.cpp:806,1425-1448`): unreachable today; meaningful only after retain.

## Context & Research

- Hot-spin sites (`evaluation_executor.cpp:242-345` greedy loop, `:299-300` poll-all, `:338` yield-only; `evaluation_executor.cu:34-42` tri-state pollHook; `cost_capacity_service.cu:384` blocking-wait precedent, non-graph path).
- Executor seam (`evaluation_executor.h:79-92` hooks, `:28` PollResult, `:114` watchdog 5 s, `:68` setWatchdogTimeout; completion events disable-timing `evaluation_context.cpp:188-201`).
- U7 harness four arms (`graph_throughput_oracle_test.cu:364-423`), steady_clock wall, warmup 3 / trials 10 (***raised to ≥50, perf review***), gate `:433-463`, baseline write `:466-478`.
- Tests: `hook_feeder_test.cpp`, `evaluation_context_lease_test.cpp`, `evaluation_executor_graph_test.cu`, `layered_correctness_test.cpp`.

### Institutional Learnings
- Compound finding (`jtml-graph-feeder-cudaeventquery-hotspin-2026-08-20.md`): fix directions, "measure busy% before verdict".
- Refreshed blueprint (`jtml-cuda-evaluation-context-executor-2026-08-17.md`): corrected poll discipline; bounded backoff; "NEVER busy-poll the whole lease set at zero delay"; capture-invalidator rule.
- Anti-stub protocol (`jtml-cuda-graph-stub-failure-2026-08-19.md`).
- Readiness questions (`jtml-deepened-unit-not-implementation-ready-2026-08-20.md`).
- CMake GLOB trap (`jtml-testability-and-cmake-conventions-2026-08-07.md`).

### External References
- CUDA Programming Guide §2.5 (blocking vs non-blocking vs callback; "delay synchronization", §2.5.8), §4.2 CUDA Graphs (instantiate once, constant-time ~2.5 µs on Ampere; launch cannot run concurrently with itself).
- NVIDIA Tech Blog "Constant Time Launch" (Ampere ~2.5 µs + ~1 ns/node), "Getting Started with CUDA Graphs" (first launch ~33% slower).

---

## Key Decisions (revised by review)

- **D1 — Ordering.** Preserve the existing "**submit all independent work first, then wait**" shape (already present at `:282` pool-size gate). This is not a new change; it's the baseline the review confirmed.
- **D2 — Wait primitive: injectable adaptive query-sweep, no blocking sync.** The multi-context path sweeps `cudaEventQuery`, handles `Done`, then **bounded adaptive wait** `sleep = clamp(T_on_completion / (2N), 3 µs, 25 µs)`, resampled per sweep. Drops the fixed 50–200 µs band (review: 50 µs ≈ 4–10 evals wide; and the correct anchor is the ~97 µs per-eval residency, not the 2.36 µs kernel). No `cudaEventSynchronize` anywhere on the graph path (R13 + frozen `zero_sync`).
- **D3 — Sole-context: bounded timed wait, NOT blocking.** When `inFlight.size()==1 && nextPos==poses.size()` (last lease), still poll with pacing, BUT observe the **watchdog boundary**: a hung GPU on the last lease must still poison. Binding `cudaEventSynchronize` would ignore the watchdog (R2 violation). Use the same confined query+pacing loop with a watchdog check every iteration (D2).
- **D4 — GPU-busy is a pre-gate, and the harness must assert it.** A `retained` verdict requires the harness to assert the nsys gates (≥30% concurrent, <50 µs gap) and the layered `graph_layer_verdict.json` PASS **before** writing. GPU busy% is measured, not assumed. A run with benefit ≥1.20 but busy ~1.6% is `blocked`/`reverted`, never `retained` (anti-artifact).
- **D5 — Trial count raised to ≥50** (or bootstrapped p90 interval) so the p99 slicing is not max-of-10 noise; report a real p5–p95 band. (Perf review.)
- **D6 — No new `.cu`/`.cpp` extracted unless needed; if the wait helper is extracted, add to `src/compute/CMakeLists.txt:39-48`. (CMake GLOB.)

---

## High-Level Technical Design

> *Directional guidance for review, not implementation specification.*

```text
Greedy hook loop (executor.cpp):
  # submit-all-first (existing :282 gate)
  for pose: acquire free ctx; enqueue on ctx.stream; record completion event
  while in-flight or pending:
      poll all in-flight leases -> {Done, Pending, Error}
      for each Done: completeFromPins -> store at original index; recycle
      if any Done: reset watchdog; continue       # no sleep on Done
      if inFlight.size()==1 && nextPos==poses.size():
          # sole remaining: still query-paced, but watchdog-check each iter
          sweep again (poll the 1 lease) with D2 bounded wait; 
          if watchdog timed out: LeavePoisoned; return WatchdogPoisoned
      else:
          pacing.wait()   # adaptive sleep clamp(T/2N,3us,25us); no-op in unit tests
          if watchdog exceeded: LeavePoisoned; return WatchdogPoisoned
  # --- gate readout ---
  # after batch: assert nsys >=30% concurrent, <50us gap, layer_vpass==PASS
  # ONLY then record retained/reverted with machine-qualified fields
```

## Implementation Units

- [ ] U0. **[Pre-flight probe: host floor + SM-overlap, decide gate reachability]**

**Goal:** Measure the spin-free per-pose host floor and the actual SM-overlap fraction **before** committing the 1.20× gate spend. This is the anti-artifact fix from the perf/adversarial review — stop it from running a scripted second `reverted`.

- Files: add `test/oracle/throughput_probe_test.cu` (or an in-harness probe arm), `test/golden/probe_measurement.md` (read-only unless JTML_UPDATE_GOLDEN=1)
- Approach:
  - Run ONLY the graph N=1 arm (no overlap, no admitted-N effect) with the corrected feeder under nsys. Measure per-pose host floor `h_N1` and median launch-to-completion residency `T`.
  - if `h_N1 >= ~111 µs` → the 1.20× gate is unreachable at N=2 (Amdahl); **stop, record `probe=unreachable`, do not spend the time gate.** This is the honest decision point.
  - if `h_N1 <= ~40 µs` → the gate is reachable; proceed to U1–U3.
  - Record nsys SM-overlap fraction `f` (at N=2, non-jump start) for later armed-plan projection.
- **Execution note:** Test-first light; this is a measurement probe, not a behavior gate. Must run on the GPU.
- **Patterns to follow:** anti-stub (real kernels, nsys census, complete()!=0.0).
- **Verification:** `probe_measurement.md` written with h_N1, T, f; the gate-reachability decision recorded.

---

- [ ] U1. **[Injectable adaptive poll pacing in the greedy loop]**

**Goal:** Replace the zero-delay yield-only backoff with an injectable adaptive pacing seam — while preserving OOO input-order completion, pending-lease-pending, no-re-poll, watchdog poison, and the exact poll-count pins.

**Requirements:** R1, R2, R4

**Files:**
- Modify: `src/compute/evaluation_executor.cpp` (greedy loop `:242-345`, pacing seam)
- Modify: `include/compute/evaluation_executor.h` (pacing seam type + installer)
- Test: `test/unit/hook_feeder_test.cpp` (extend)

**Approach:**
- Add `PollPacing` injectable wait (adaptive-sleep / yield / no-op). CUDA default = adaptive (D2/D5); headless default = no-op.
- Greedy loop: sweep, handle Dones; on any Done, continue immediately; else bounded many-context adaptive sleep; sole-context uses the D2 timed loop (U2).
- Do not re-poll completed; do not reorder result index.

**Patterns to follow:** hook installers at `evaluation_executor.h:87-92`; `cost_capacity_service.cu` blocking-wait precedent only as the *serial* helper, not the admitted path.

**Test scenarios:**
- Happy: fake hooks, 3 poses, pool 2, OOO done → input-ordered scores; poll counts == today's (each pending polled twice; done never re-polled).
- Happy: **pacing-exactly-once** — zero completions, pool 2 → the pacing hook called exactly once per sweep (spy count == 1), not ≥1.
- Happy: **no-pacing-on-Done** — pool 2 both Done in the same sweep → pacing count == 0 (pacing only when no completion).
- Edge: pacing = no-op → identical to today (1 ms watchdog tests stay green).
- Error: pollHook Error mid-batch → PostLaunchAbort + drain + ForceRelease (unchanged).
- Integration (oracle): real graph, distinct scores, no per-eval sync.

**Verification:** `jtml.hook_feeder` green; spy proves **exactly-once** pacing per zero-completion sweep; no-pacing-on-Done.

---

- [ ] U2. **[Sole-context bounded timed wait + CUDA adaptive pacing hooks]**

**Goal:** Implement the sole-remaining-context **bounded, watchdog-porous wait** (NOT blocking `cudaEventSynchronize`), and the CUDA adaptive accelerator hooks, keeping the accepted path sync-free (R3 / frozen `zero_sync`).

**Requirements:** R1, R3

**Files:**
- Modify: `src/compute/evaluation_executor.cu` (`InstallCudaFeederHooks`): adaptive watchdog-aware pacing; **no `cudaEventSynchronize`** in any path.
- Modify: `src/compute/evaluation_executor.cpp` (sole-context uses the timed query+wait loop, watchdog-checked; no blocking primitive).
- Test: `test/hook_feeder_test.cpp` (sole-context: bounded loop, watchdog); `test/oracle/evaluation_executor_graph_test.cu` (assert no blocking sync at any node, incl. tail).

**Approach:**
- Sole-context branch: `while (inFlight==1 && no extra poses)`: query the gaze-only; if Done → complete; if watchdog elapsed → LeavePoisoned + WatchdogPoisoned; else bounded-adaptive wait and re-probe. **No call to `cudaEventSynchronize` anywhere.**
- CUDA installer: pollHook stays `cudaEventQuery` tri-state; pacing = adaptive sleep.
- Anti-stub: real kernels, distinct scores, input-ordered; oracle asserts zero sync in tail.

**Test scenarios:**
- Happy (unit): 1 pose pool 1 — sole-context bounded wait invoked, result correct, no blocking sync call.
- Edge: **sole-context hang still poisons** — fake hook never Done, watchdog 1 ms → `WatchdogPoisoned` + `isPoisoned()` + `IsPoisoned`. **(This was impossible under the pre-review blocking design; the goal.)**
- Happy (unit): 2 poses pool 2, both Done same sweep — sole branch not entered.
- Edge: sole-context query error → PostLaunchAbort + drain.
- Integration (oracle): 2-context batch input-ordered scores; **zero `cudaEventSynchronize` observed** on the oracle's sync-census (assert at code, not comment).
- Integration (perf/probe): N=2 SMP overlap f measured; nsys shows gap < 50 µs.

**Verification:** `jtml.hook_feeder` (incl. hang-poisons), `jtml.evaluation_executor_graph`; oracle sync-census == 0; nsys gap < 50 µs.

---

- [ ] U3. **[U7 re-qualification, nsys+layered+readback-gated]**

**Goal:** Run the four-arm harness with the corrected profile, now with the harness **asserting the nsys + layered gates BEFORE `retained`, and read-back asserting the written JSON.**

**Files:**
- Modify: `test/oracle/graph_throughput_oracle_test.cu` — (a) read `test/golden/graph_layer_verdict.json`, require `verdict==PASS` before `retained`; (b) parse the nsys stats/ SM-census → assert ≥30% concurrent at N=2, <50 µs gap before `retained`; (c) raise trials ≥50 + bootstrapped p90 benefit interval; (d) read-back assert the written JSON (verdict — computed, commit/hostname/driver/admitted_N present).
- Reference: `test/golden/graph_pre_registration.json` thresholds.

**Approach:** after U2. The verdict logic (`:441-463`) must be extended, not just the prose G (`:454-462`), so `retained` cannot be written without nsys/layered evidence.

**Test scenarios:**
- Integration: nsys kernel census real; deliver `retained` only when gate + layered + nsys all pass.
- Integration: gap unluckily >50 µs but benefit 1.25× — **assert `reverted`, not retained** (proves the new gate bites).
- Integration: layered verdict != PASS → `blocked`, baseline not `retained`.
- Readback: after write, re-read the JSON; assert verdict/reason/commit/hostname/admitted_N present; mismatch → fail.

---

- [ ] U4. **[Docs + knowledge store sync]**

**Files:**
- `docs/handoff-2026-08-20-cuda-graph-executor-admission.md`, `docs/plans/2026-08-20-012-feat-...-plan.md` (fix stale "no minimum-benefit"/"blocking wait allowed" if present), `docs/solutions/...feature-tagged-solution` compound (append the resolution), `refreshed blueprint` (correct the 50–200 µs band to adaptive).

**Approach:** after U3 outcome; per `jj describe`+`jj new`.

---

## System-Wide Impact

- **Interaction:** greedy loop → pacing seam → `pollHook_`/CUDA-feeder → `EvaluationContextPool` (lease bookkeeping untouched). Harness + both oracles consume it.
- **Error propagation:** Error → PostLaunchAbort + drain + ForceRelease; watchdog → LeavePoisoned; both respect the bounded pacing (never blocked to hang).
- **State lifecycle:** the sole-context hang now poisons (U2); pacing ≤25 µs vs 1 ms test floor is 40× margin → no flake; 5 s production watchdog orders far below.
- **API surface:** no public change; a new pacing installer only.
- **Integration coverage:** hook_feeder, evaluation_executor_graph, layered_correct, graph_throughput_oracle (with nsys+layered gates).
- **Unchanged invariants:** DIRECT, recipe, capture, admission, lease, `disable-timing` events, and the frozen `zero_sync`.

## Risks & Dependencies

| Risk | Mitigation |
|---|---|
| Backoff bound 10–25 µs on 2.36 µs kernel still 4–10 kernel-wide | Correct anchor is ~97 µs per-eval residency; bound is 3–25 µs = small fraction of one eval; resample adaptive; GPU busy is the pre-gate, harness-asserted (D5) |
| 1.20 × may be unreachable at N=2 (f<0.28, h>12 µs) | U0 probe decides reachability before U3; if `probehind`, record `reverted` as honest device outcome — not a fake retain |
| A second `reverted` reads as "graph-capability limit" | The probe + nsys-gate + layered + readback make it device-evidence-based; the compound's prevention rule is honored by measuring busy |
| `zero_sync` frozen gate vs any sync | This plan keeps **zero** `cudaEventSynchronize` on the accepted path (D2/D3); no carve-out to renegotiate |
| Headless 1 ms watchdog / poll pins break on pacing | Pacing is injectable (no-op in unit tests); throttled |
| nsys unavailable | Harness verdict = `blocked`; baseline unchanged |
| GPU busy still ~1.6% after fix | Sequence => `reverted`/`blocked`, not `retained` (anti-artifact) |

## Documentation / Operational Notes

- nsys cmd: `nsys profile --stats=true -o /tmp/u7_probe_req ...` and `/tmp/u7_req`.
- Baseline regenerated only on qualified harness; `commit` = jj-tracked git ref.
- No production rollout.

## Sources & References
- Origin: plan-012 U7 (`:569-605`), C6/R13
- Compound: `jtml-graph-feeder-cudaeventquery-hotspin-2026-08-20.md`
- Refreshed blueprint (corrected band deferred to U4)
- Anti-stub, layer-gate, readiness
- Frozen `graph_pre_registration.json` (`N_values [1,2,4]`; 1.20×; 30% N2; 50 µs; zero_sync) and current `graph_performance_baseline.json` (reverted)
- External: CUDA guide §2.5/§4.2; NVIDIA blogs