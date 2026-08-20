# CUDA Gate Review — Plan 013 fix-cuda-graph-feeder-concurrency

**Reviewed:** 2026-08-20 · **Base:** `docs/plans/2026-08-20-013-fix-cuda-graph-feeder-concurrency-plan.md` (deepened 2026-08-20, + CUDA persona gate)

**Verdict:** **REAL** — the plan is technically sound; the P0/P1 findings were genuinely introduced by the deepened refactor and each has a concrete, code-anchored fix. Two measurement-level issues (P1) were corrected in this revision.

## Findings

| # | Lens | Severity | Conf | Verdict | Plan/line | Source checked | Corrected understanding |
|---|---------|-----|------|---------|------------|------------------|
| 1 | cuda-concurrency | P0 | 75 | **REAL** | plan HLTD `sweep again` (plan:101-104) | evaluation_executor.cpp:104 | Outer sweep already polled `sole_lease` once (Pending); D3 then polls it a 2nd time in the same iteration → violates `exactly twice / never a third` oracle, + poll-count drift in the sole tail (2/iter). Fix: single-poll-per-iteration pin (identical to correctness F4). |
| 2 | cuda-correctness | P1 | 100 | **REAL** | plan :101-107, :173 | evaluation_executor.cpp:293-339 | No poll-Error arm during the sweep → ordinary CUDA errors become watchdog poison (sole) or leak (multi-context). Restore explicit `Error→abort+ForceRelease` arm before any pacing/watchdog (mirror plan-011 321-328). |
| 3 | cuda-concurrency | P0 | 100 | **REAL** | plan HLTD :101-104 | exec.cpp :338 + sole-CU: :172 | The `sweep again (poll the 1 lease)` double-polls the sole lease within one iteration → breaks the K-vs-exactly-2 oracle (hook_feeder_test.cpp:213-216); real-GPU poll-count drifts (2/iter vs 1). Fix: sole-tail enters BEFORE the outer sweep; each iter EXACTLY one poll. |
| 4 | cuda-measurement | P1 | 100 | **REAL** | plan: 194-213 | oracle/harness | `nsys` timeline was the right tool for the <50µs gap (P1) — do NOT use `nsys stats` for SM-occupancy (bogus). Updated U3 to parse ncu SM-occupancy/concurrent-KERNEL census. |
| 5 | cuda-measurement | P1 | 75 | REAL | U3 :189-202, 213 | oracle harness | nsys timeline for gap; ncu for SM/concurrent; both correct per tool. |
| 6 | cuda-orchestration | P2 | 50 | ALREADY-ADDRESSED | plan :U0 circular | exec | Corrected U0: reachability GATE is N=2 SM-f overlap f (NCU SM-occupancy/concurrent-KERNEL), sole-lease done in U2; binary on (f,host) vs 1.20 budget — no middle band ambiguity. |

## Required plan edits (all applied above)

1. **U0**: after U1/U2, host-floor gate = N=2 overlap `f` (NCU) — the `h_N1` host floor measured with pacing isolated + T isolated. If `f < ~0.28` OR host floor > ~12 µs → gate unreachable at N=2, record `probe=unreachable` + stop (honest device outcome). Binary, no middle band. (`plan:126-134`)
2. **HLTD**: single-poll per iteration in the sole tail; **remove `sweep again`** (`plan:101-104`) — the sole lease is polled once (pending→done), never twice per iteration, watchdog-checked each iteration (`plan:104,117`).
3. **Error arm restored**: `Error → abort + ForceRelease` (result.clear+teardown+ForceRelease, incl. errored) **before** any Done/pacing (`plan:100-101`) — ordinary CUDA errors become revert/leak fixes (P0).
4. **Watchdog arm**: all in-flight poisoned + `poisoned_.store(true)` latch → `WatchdogPoisoned` (plan:111-112,117-118) matching today's executor.cpp:330-337.

## Persona stats
- Persona gate: each CUDA persona loads its own NVIDIA refs from `~/.pi/agent/skills/cuda-skill/references/` (MANIFEST.md = CUDA 13.3.1 / Nsight 2026.2.1).
- Verified same root fact: N admitted == N distinct `cudaGraphExec` handles == pool of handles; graph exec cannot run concurrently with itself (§4.2.7).

## Findings → Required edits (all landed in the plan above)

- cuda-concurrency `P0` → **applied** (sole-tail loop before general sweep; exactly-one-poll per iter)
- cuda-correctness `P1` → **applied** (explicit `Error abort` arm before any Done)
- cuda-measurement `P1` → **applied** (U3 uses NCU for SM/concurrent; nsys for the gap)
- cuda-graph-lifecycle `P1` → **applied** (N==handles invariant recorded for 016; complete() scoped to serial)
- cuda-orchestration `P1/P2` → **applied** (U0 reordered after U1/U2; reachability gate = N=2 f)