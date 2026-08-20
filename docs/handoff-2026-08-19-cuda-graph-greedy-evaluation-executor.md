# Handoff: CUDA-Graph Greedy Evaluation Executor (Plan 011) — SUPERSEDED for remaining work

**Prepared:** 2026-08-19 — post-U5 implementation, full-docs reconciliation, and U6–U8 design review

- **Active plan is now 012:** `docs/plans/2026-08-20-012-feat-cuda-graph-executor-admission-plan.md`
- **012 handoff:** `docs/handoff-2026-08-20-cuda-graph-executor-admission.md`
- **This file:** historical 011 U1–U5 status only. Do **not** implement 011 U6 from the sections below.
- **Requirements:** `docs/brainstorms/2026-08-19-cuda-graph-greedy-evaluation-executor-requirements.org`
- **Architecture blueprint:** `docs/solutions/architecture-patterns/jtml-cuda-evaluation-context-executor-2026-08-17.md`
- **Anti-stub learning:** `docs/solutions/logic-errors/jtml-cuda-graph-stub-failure-2026-08-19.md`
- **Layered-correctness draft:** `docs/solutions/architecture-patterns/graph-tiered-correctness-2026-08-19.md` — **aspirational until U7 proves it**.

---

## First turn for the next agent

1. Read root `AGENTS.md` — it now points at current work and explains where information lives.
2. Run `ctx_index` on `docs/` (or reuse source `jtml-docs` if already indexed), then query focused facts with `ctx_search` rather than broad greps. Useful queries:
   - `"U6 graph executor wrapper lifetime completeFromPins"`
   - `"U7 layered correctness circular serial passthrough"`
   - `"U8 rev-2 throughput 12412 Nsight retain threshold"`
3. Read plan 011 **U6–U8** before editing code. They were deepened after a five-persona doc review and are the implementation source of truth.
4. Run `jj st` and `jj log --no-graph -r '@-::@-'`. The docs-preparation change should be committed before starting implementation; do not mix it with U6 code.
5. Use `pixi run build`; use `ctest` target filters first. Never use raw git, cmake, make, or nvcc.

---

## Honest implementation status

### Landed and verified

| Unit | State | Checkpoint / evidence |
|---|---|---|
| U1 | Real `EvaluationContextPool`: non-blocking stream, disable-timing event, private device counters, pinned overflow, idempotent shutdown | `llxwvusv` lineage; `src/compute/evaluation_context.cpp` |
| U2 | Test-impact matrix + frozen rev-2 workload contract | `ktpnwxll`; `test/golden/graph_pre_registration.json` has real Kneel_1 12412-triangle / 1024x1024 values |
| U3 | Real `ProbeCapturableOpSet` over a caller-supplied production op set | `srlpqvvp`; `src/compute/graph_preflight.cu` |
| U4 | Device-driven persistent workers + graph-capturable metric chain: stream-ordered counter clears, persistent prefix/fill/overflow kernels, async overflow tail, no host bbox packet on context path | `ytvowmkl`; U4 production oracle exercises real `RenderPhase(EvaluationContext&)` |
| U5 | Real monoplane `DIRECT_DILATION` graph recipe: captures production `EnqueueRenderPhase` + FID + distance-map chain, instantiates per-context graph, patches WorldToPixel params, computes real white-sum baseline, and has real serial-parity/non-blank fixture oracle | `ntqxnwox` (`62855b11`) |
| Test hygiene | Cut-0 measurement no longer rewrites tracked golden on every oracle run; normal run writes ignored scratch output; `JTML_UPDATE_GOLDEN=1` is explicit rebaseline opt-in | `lmrxuoss` (`243976c8`) |

### Still open — do not mistake scaffolding for implementation

| Unit | Current reality | Next implementation target |
|---|---|---|
| U6 | `EvaluationExecutor::RunBatch` is serial-cost simulation; `.cu` is dummy; explicit-context CostFunctionManager methods are stubs; production does not register recipe; null-recipe U6 branch currently risks replacing the real U12 batch with serial passthrough | Implement the reviewed hook-based graph executor from plan U6: executor-owned wrapper lifetime, populated key/provider, event-query poll, no-sync `completeFromPins`, frame invalidation, R7/R8 split, OptimizerError conversion |
| U7 | Existing layered tests are circular because graph `RunBatch` calls serial cost. No real Layer A image or Layer B raw-int compare. Fixture has old fabricated 300k values. | Real graph vs serial Layer A/B/C oracle using rev-2 Kneel_1 fixture; recipe-owned reduction set only; >=3x concurrency stress |
| U8 | `graph_throughput_oracle_test.cu` is a rev-1/synthetic stub. It asserts `triangle_count=300000` against rev-2 golden `12412`, so `jtml.graph_throughput_oracle` is an expected oracle red. Baseline JSON has placeholders. | Four-arm real paired harness (serial N=1, graph N=1, N=2, N=max), mandatory Nsight Systems timeline, pre-registered minimum-benefit + retain/revert decision |

---

## U6 non-negotiable design constraints

These were extracted from full docs recon plus five-persona review; plan 011 now contains the full version.

1. **Graph wrapper lifetime:** `createGraph` returns `GraphExecWrapper*`; only `recipe->destroyGraph` frees graph template + exec + wrapper. `EvaluationContextPool::Shutdown()` treats `ctx.graph_exec` as raw `cudaGraphExec_t`, so never put the wrapper there. Executor owns `graphExecs_[contextIndex]`, clears them idempotently on Shutdown.
2. **Headless/real split:** headless tests link `evaluation_executor.cpp`, not `.cu`. Keep one ordered loop in `.cpp`, driven by injected enqueue/poll/completeFromPins/teardown hooks declared in the CUDA-free header and installed from `.cu`. Do not direct-call a `.cu` symbol from `.cpp`.
3. **Key/provider:** `CostFunctionManager` must assemble `GraphRecipeCaptureInputs`; `GPUModel` needs a render-engine accessor. Build one complete `GraphRecipeKey` used by OptimizerManager admission and executor capture. Current production key has zero dimensions/triangles and therefore cannot preflight successfully.
4. **Frame/stage invalidation:** captured graph bakes comparison buffers, dilation, and cached white sum. Key + frame/parameter generation must invalidate/recreate wrappers when any of these changes.
5. **Completion:** executor records `ctx.completion_event` *after* `cudaGraphLaunch`, outside capture. Poll `cudaEventQuery`; `cudaErrorNotReady` is pending, any other non-success is an error. After event success call `completeFromPins`, never syncing `recipe.complete()`.
6. **R7/R8:** before first successful launch, provider/key/create/update/launch failure is `NOT_SUBMITTED` and leaves serial/U12 intact. After first launch, any error/overflow/watchdog clears results and must reach `OptimizerError`, not uncaught `std::invalid_argument`.
7. **U12 coexistence:** one `SetBatchCost` only. Keep U12 as graph-unavailable fallback if retained, but never let a null recipe replace it with executor serial passthrough.
8. **Rollout:** graph admission stays experimental/off until U7 passes and U8 records a retained verdict.

---

## CUDA facts the next agent must use

CUDA reference root: `~/.pi/agent/skills/cuda-skill/references/`.

- `cuda-runtime-docs/modules/group__cudart__event.md` — `cudaEventQuery`: `cudaSuccess` complete, `cudaErrorNotReady` pending, other errors real; disable-timing events suit query polling.
- `cuda-runtime-docs/modules/group__cudart__stream.md` — capture modes, `cudaStreamQuery`, and the caveat that querying an event last recorded inside a capture is prohibited.
- `cuda-runtime-docs/modules/group__cudart__graph.md` and `cuda-guide/04-special-topics/cuda-graphs.md` — graph instantiate/launch/node-param lifecycle.
- `best-practices-guide/` — streams express concurrency but do not guarantee it; Systems first for overlap, Compute second for selected-kernel detail.

---

## Validation state at handoff

- `pixi run build` passed after U5 work.
- Focused U4/U5 chain was green: `jtml.evaluation_context`, `jtml.graph_recipe_preflight`, `jtml.graph_capture_probe`, `jtml.u4_production_integration`, `jtml.graph_recipe_direct_dilation`, `jtml.layered_correctness`, `jtml.bit_identity`.
- Headless suite was `58/59`; the only known pre-existing red is `jtml.qml_lint`.
- `jtml.graph_throughput_oracle` fails as the expected U8 rev-1 `300000` stub red; do not "fix" it outside U8.
- `jtml.cut0_measurement` now passes without mutating `test/golden/cut0_measurement.md`.

---

## jj discipline

- Always `jj describe -m "<scope>: <message>"` then `jj new` for each logical change.
- Never run `jj restore` with verified uncommitted work. Prior accident restored real U1 work/stale stubs. Commit/checkpoint first.
- Golden measurements are now protected: ordinary Cut-0 runs are non-mutating; only `JTML_UPDATE_GOLDEN=1` deliberately rewrites `test/golden/cut0_measurement.md`.

---

## Recommended next change

Start a single **U6 executor wiring** change only after reading the deepened plan. Keep it atomic:

1. Add hook seam + wrapper/key ownership to `EvaluationExecutor`.
2. Add provider/key assembler via CostFunctionManager/GPUModel.
3. Wire real `.cu` poll hooks and `completeFromPins`.
4. Fix OptimizerManager recipe registration/null-recipe branch/error conversion.
5. Add headless injection tests + GPU executor oracle.
6. Run focused build/tests, checkpoint, then independent correctness review before U7.
