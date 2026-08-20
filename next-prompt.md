# Next Agent Prompt — JTML Plan 011 Resume

## Mission

Continue **Plan 011 — CUDA-Graph Greedy Evaluation Executor**. U1–U5 are real, reviewed, checkpointed; do **not** reimplement them. The next implementation unit is **U6: real greedy CUDA graph executor wiring**. U7/U8 are fully deepened designs, not implementation work yet.

## First turn: orient without broad exploration

1. Read `AGENTS.md`.
2. Run `ctx_index` over `docs/` if `jtml-docs` is not already indexed; use `ctx_search` for focused facts:
   - `U6 graph executor wrapper lifetime completeFromPins`
   - `U6 GraphRecipeKey provider null recipe U12 fallback`
   - `U7 layered correctness circular serial passthrough`
   - `U8 rev-2 throughput 12412 Nsight retain threshold`
3. Read these in order:
   - `docs/plans/2026-08-19-011-feat-cuda-graph-greedy-evaluation-executor-plan.md` — **U6 section first**
   - `docs/handoff-2026-08-19-cuda-graph-greedy-evaluation-executor.md`
   - `docs/solutions/architecture-patterns/jtml-cuda-evaluation-context-executor-2026-08-17.md`
   - `docs/solutions/logic-errors/jtml-cuda-graph-stub-failure-2026-08-19.md`
4. `jj st`, `jj log --no-graph -r '@-::@-'`, then `pixi run build`.

## Current checkpointed state

- **U1 real pool:** `llxwvusv` — private context stream/event/counters/pinned overflow.
- **U2 rev-2 frozen contract:** `ktpnwxll` — real Kneel_1 12412-triangle / 1024x1024 fixture.
- **U3 real capture probe:** `srlpqvvp`.
- **U4 persistent workers:** `ytvowmkl` — production `RenderPhase(EvaluationContext&)`, device crop + graph-capturable metrics.
- **U5 real graph recipe:** `ntqxnwox` / `62855b11` — captures production enqueue chain, `cudaGraphExecKernelNodeSetParams`, real white-sum baseline, non-blank serial-parity oracle.
- **Test hygiene:** `lmrxuoss` / `243976c8` — Cut-0 measurement no longer rewrites tracked golden unless `JTML_UPDATE_GOLDEN=1`.

The working copy should contain only this documentation-preparation change. Commit it separately before U6 code.

## U6: non-negotiable implementation constraints

1. **Wrapper lifetime:** `DirectDilationMonoplaneRecipe::createGraph()` returns a `GraphExecWrapper*`. The executor owns it and destroys it through `recipe->destroyGraph()`. Never put that wrapper into `ctx.graph_exec`: pool Shutdown treats that field as a raw `cudaGraphExec_t`.
2. **Headless / CUDA TU split:** headless tests link `evaluation_executor.cpp`, not the `.cu` TU. Keep one ordered greedy loop in `.cpp`, driven by injected CUDA-free hooks (`enqueue`, `poll`, `completeFromPins`, `teardown`) declared in the header and installed by `.cu`. Do not direct-call a `.cu`-only symbol from `.cpp`.
3. **Provider + real key:** CostFunctionManager must supply `GraphRecipeCaptureInputs`; GPUModel needs a primary RenderEngine accessor. Build one real `GraphRecipeKey` (frame dims, model triangles, stride, CUB storage, dilation, calibration hash, etc.) for BOTH admission and capture. Current production key is zero-filled and would always preflight false.
4. **Frame/stage invalidation:** graph capture bakes comparison-frame pointers, dilation, and white sum. Cache wrapper + full key + frame/parameter generation; destroy/recreate when any change occurs.
5. **Polling:** executor calls `cudaEventRecord(ctx.completion_event, ctx.stream)` only AFTER `cudaGraphLaunch`, outside capture. `cudaEventQuery == cudaErrorNotReady` means pending; any other non-success is a real error. After success, call a new no-sync `completeFromPins()`, never syncing `recipe.complete()` in the greedy path.
6. **R7/R8:** before first successful launch, provider/key/create/update/launch failure is `NOT_SUBMITTED` and preserves U12/serial. After first launch, error/overflow/watchdog clears results and reaches `OptimizerError` — never an uncaught `std::invalid_argument` or silent serial fallback.
7. **Production rollout:** fix OptimizerManager's null-recipe branch that currently overwrites U12 with executor serial passthrough. Graph admission remains experimental/off until U7 + U8 prove it.

## Tests and expected reds

- Run focused tests before broad suites: `jtml.evaluation_context`, `jtml.graph_recipe_preflight`, `jtml.graph_capture_probe`, `jtml.u4_production_integration`, `jtml.graph_recipe_direct_dilation`, then new U6 targets.
- `ctest -L headless` has one known pre-existing red: `jtml.qml_lint`.
- `jtml.graph_throughput_oracle` is an **expected U8 oracle red** until U8 replaces the rev-1 `triangle_count=300000` stub. Do not change it in U6.
- Normal Cut-0 runs now leave `test/golden/cut0_measurement.md` clean. Never use `jj restore` to clean test side effects; identify and fix the producer.

## Tooling discipline

- **jj only:** `jj describe -m "<scope>: <message>"`, then `jj new`, one logical change at a time. Never raw git.
- **pixi only:** `pixi run build`, `pixi run test`; no raw cmake/make/nvcc.
- **ReadSeek:** digest → edit → digest → edit. Use `language: "cpp"` for `.h`, `.cu`, `.cuh` reads/searches.
- **CUDA references:** `~/.pi/agent/skills/cuda-skill/references/` — event query (`cuda-runtime-docs/modules/group__cudart__event.md`), capture/query modes (`group__cudart__stream.md`), graph lifecycle (`group__cudart__graph.md`, `cuda-guide/04-special-topics/cuda-graphs.md`).

## Definition of a good U6 checkpoint

A single atomic U6 change with real hook-installed graph launch/event polling, real key/provider/wrapper lifetime, R7/R8 error semantics, headless injection tests + GPU executor oracle, focused tests green, independent correctness review, then `jj describe` + `jj new`. Do not start U7 until that checkpoint is reviewed.
