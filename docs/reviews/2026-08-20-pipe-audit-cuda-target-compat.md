# P13 Pipe — CUDA Target Compatibility Review (cuda-target-compat)

**Repo:** `/home/ajj/repo/uf/JTML` (task named `/repo/self.JTML`; the CWD is the live tree — pixi.toml, CMakeLists.txt, src/compute, include/compute all resolve here)
**Lens:** compute-capability targeting, toolkit/runtime version skew, binary compat.
**Mode:** READ-ONLY. No files edited; evidence is file paths + quoted lines.

---

## A. CUDA toolkit vs. arch list — CONFIRMED

| Fact | Evidence | Verdict |
|---|---|---|
| pixi toolkit targets CUDA 12.9 | `pixi.toml:7` `platforms = [{ platform = "linux-64", cuda = "12.9" }]`; `pixi.toml:13` `cuda-version = "12.9.*"`; `:40` `cuda-toolkit = ">=12.9.0,<13"` | confirmed |
| Installed nvcc is 12.9.41 | `nvcc --version` → `Cuda compilation tools, release 12.9, V12.9.41` (`Build cuda_12.9.r12.9/compiler.35813241_0`) | **matches task claim** |
| Installed runtime header is 12.9 | `<env>/targets/x86_64-linux/include/cuda_runtime_api.h` → `#define CUDART_VERSION  12090` (12.9) | consistent |
| CMake default arch | `CMakeLists.txt:58-60` `if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES) set(CMAKE_CUDA_ARCHITECTURES native)` | confirmed |
| Configure arch list | `pixi.toml:94` `-DTORCH_CUDA_ARCH_LIST=8.6 8.9 12.0`; `:95` `-DCMAKE_CUDA_ARCHITECTURES=native` | confirmed |
| **Actual per-TU nvcc arch set** | `.build/compile_commands.json` nvcc invocations carry `-gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120` on every `.cu` TU | **compute 8.6/8.9/12.0 → sm_86/89/120 confirmed at the compiler level** |
| Probe device | `test/golden/probe_measurement.md:4` — RTX 3090 Ti (sm_86), driver 610.57.04, CUDA 12.9 | sm_86 ⊂ arch list |

**Verdict A (confirmed):** toolkit is CUDA 12.9 (nvcc 12.9.41, runtime 12090). Build target compiles every TU for **compute_86/89/120 → sm_86/89/120**. The probe device (sm_86) is at the *lowest* arch in the list — all three fatbins are linked, but the device is only ever serviced by the sm_86 fatbin at runtime.

---

## B. Async/concurrency API set available on 12.9 for a graph-free multi-stream worker

All confirmed against the **installed 12.9 header** (`<repo>/.pixi/envs/default/targets/x86_64-linux/include/cuda_runtime_api.h`) and the repo call sites. (Installed-toolkit header is authoritative here; skill refs snapshot CUDA 13.3 — newer — but every API below is a long-standing pre-12 API also present in the 12.9 header directly.)

| API | Installed 12.9 header | In-repo use | status |
|---|---|---|---|
| `cudaStreamCreateWithFlags(cudaStreamNonBlocking)` | present | `cost_capacity_service.cu:182`, `evaluation_context.cpp:188`, `graph_preflight.cu:58,198` | **used** |
| `cudaEventCreateWithFlags(cudaEventDisableTiming)` | present | `cost_capacity_service.cu:183`, `evaluation_context.cpp:196` | **used** (no `cudaEventBlockingSync`) |
| `cudaEventRecord` / `cudaEventQuery` | present | executor + probes | **used** |
| `cudaEventSynchronize` | present (host-blocking) | `cost_capacity_service.cu:384` (calibration helper), tests/teardown | used but **NOT on the admitted executor path** |
| **`cudaStreamWaitEvent(stream,event,flags)`** | **present** (`cuda_runtime_api.h:2847`) | **0 references anywhere in repo** | **available, unused** |
| `cudaMemcpyAsync` (pinned D2H/D2D) | present | `render_engine.cu:1383,1390,1701` | **used** |
| `cudaMemcpy*Async` 2D/3D variants | present | none | available |
| **`cudaMallocAsync` (stream-ordered alloc)** | **present** (`cuda_runtime_api.h:8805`) | none | **available, unused** |
| Persistent-worker kernels | n/a (device code) | `StridePrefixPersistentKernel` + `FillTrianglePersistentKernel` defined (`render_engine.cu:923,962`), occupancy-queried at init (`:374-382`), **launched** in `EnqueueRenderPhase(EvaluationContext&)` (`:1648,1671`) | **present and linking** (graph-free path) |
| `cudaGraphLaunch` / `cudaGraphInstantiate` / `cudaGraphExecKernelNodeSetParams` | present | `graph_recipe_direct_dilation.cu`, `evaluation_executor.cu` | **used** on graph path |

**Driver-version note:** CUDA 12.x is a new major; per the skill's `16.4` (semantic-versioning) doc, the runtime does not bump minimum driver on *minor* releases — only major. Installed probe driver **610.57.04** (`probe_measurement.md:4`) is far above the 12.x floor. The compiled sm_120 arch requires a Blackwell-capable driver **at runtime** for that fatbin; on the sm_86 probe device it is inert today, but any deployment on Ada/Hopper-class hardware would need a driver exposing compute 12.0.

**Persistent-worker verdict:** `StridePrefixPersistentKernel` and `FillTrianglePersistentKernel` are present, compiled into `jtml_compute`, sized by real occupancy queries, and launched from the **non-blocking, graph-free** `EnqueueRenderPhase(EvaluationContext&)`. Not disabled anywhere (no `#if` gate; the serial `RenderPhase(BankState&)` path still uses `StridePrefixKernel`/`FillTriangleKernel` separately).

---

## C. Gotchas for the sync-free rule (R13 / zero_sync)

1. **`cudaEventSynchronize` is host-blocking and is *not* on the admitted path — correctly.** The executor asserts `evaluation_executor.cu:50` — "no `cudaEventSynchronize` anywhere on the admitted path (R13 / zero_sync)". The only production `cudaEventSynchronize` is in `cost_capacity_service.cu:384`, inside the serial bank-capacity **calibration helper** (its own `cudaEventRecord` + `cudaEventSynchronize` finish loop). A graph-free multi-stream worker must NOT adopt the calibration helper's pattern into the enqueue/host/Recycle hot path. This boundary is currently honoured.

2. **`cudaStreamWaitEvent` is the valid device-side alternative — but it is an ordering primitive, not a completion primitive.** Per CUDA guide `02-basics/asynchronous-execution` (refs): "`cudaStreamWaitEvent()` … makes all the commands added to the given stream after the call delay their execution until the given event has completed." It does **not block the host** and does **not report completion to the host**. So:
   - It is fully compat with the sync-free boundary (it enqueues device-side dependencies; it never host-syncs).
   - It **cannot** replace `cudaEventQuery` for the "has this context's work finished → Recycle" decision. A multi-stream graph-free worker still needs a per-context completion event + `cudaEventQuery` on each stream — which is exactly what the current `EvaluationContext` (`evaluation_context.cpp:188,196`) already installs.
   - Where `cudaStreamWaitEvent` *would* add value: **cross-stream ordering** (a worker stream that must not start until another context's event completed). That is not needed on the current per-context independent-stream design, so its absence (0 uses) is not a defect — it is the correct append-only option.
   - **Gotcha if adopted later:** per skill `03-advanced/advanced-host-programming` refs, an event that has never been `cudaEventRecord`'d returns success from `cudaStreamWaitEvent`/`cudaEventQuery` — an **engineered bind order** bug. The emitting stream's work must be fully queued *before* `cudaEventRecord`, and the waiting stream's `cudaStreamWaitEvent` must reference an event that *was* recorded. The current code records the event once, after enqueue, "outside capture" (`evaluation_executor.cu:31,`); if you add wait-streams, that ordering must be preserved and the event never be re-recorded while a waiter references it.
   - Guide event-config recommendation (skill `cuda-runtime-docs/.../group__cudart__event.md:52`): `cudaEventDisableTiming` without `cudaEventBlockingSync` gives best performance with `cudaStreamWaitEvent()`/`cudaEventQuery()`. **The existing `EvaluationContext` and capacity-service events are already created exactly this way** (`cudaEventCreateWithFlags(&event, cudaEventDisableTiming)`), so a future `cudaStreamWaitEvent` integration has zero config changes.

3. **`cudaMemcpyAsync` targets must be pinned.** The graph-free persistent-worker path already uses pinned host buffers (`/DeviceAlloc` via `cudaHostAlloc`) and async D2H of the overflow flag (`render_engine.cu:1701`). Do not feed it pageable host memory — async copies to unpinned host memory fall back to a synchronous notification (serialise the stream). Not a current defect, but a name-check gotcha for anything that mirrors the pattern.

---

## Findings JSON

```json
{
  "findings": [
    {
      "lens": "cuda-target-compat",
      "title": "Toolkit 12.9 + arch list compute_86/89/120 confirmed; nvcc 12.9.41",
      "severity": "P2",
      "confidence": 100,
      "evidence": [
        "pixi.toml:7 platforms [{ linux-64, cuda=12.9 }], :13 cuda-version=12.9.*, :40 cuda-toolkit >=12.9,<13",
        "nvcc --version => 'Cuda compilation tools, release 12.9, V12.9.41'",
        ".build/compile_commands.json: nvcc ... -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120"
      ],
      "owner": "release",
      "suggestedFix": "None (confirmation). Target arch set is compute_86/89/120 == launcher arch (sm_86/89/120); device is sm_86."
    },
    {
      "lens": "cuda-target-compat",
      "title": "cudaStreamWaitEvent available on 12.9 but unused across the repo (0 refs)",
      "severity": "P2",
      "confidence": 100,
      "evidence": [
        "Installed header cuda_runtime_api.h:2526 'extern __host__ ... cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned int flags __dv(0))'",
        "repo-wide grep 'cudaStreamWaitEvent' => 0 matches in src/include/test"
      ],
      "owner": "maintainer",
      "suggestedFix": "For a graph-free multi-stream worker: adopt it as the device-side ordering primitive (cross-stream deps). It does NOT report completion to host; each context still needs its own cudaEventQuery on the bounded wait."
    },
    {
      "lens": "cuda-target-compat",
      "title": "cudaMallocAsync (stream-ordered allocation) available on 12.9 but unused",
      "severity": "P2",
      "confidence": 100,
      "evidence": [
        "Installed header cuda_runtime_api.h:8805 'extern ... cudaMallocAsync(void **devPtr, size_t size, cudaStream_t hStream)'",
        "repo-wide grep 'cudaMallocAsync' => 0 matches under src/include"
      ],
      "owner": "maintainer",
      "suggestedFix": "Optional: use cudaMallocAsync per-context to stream-order workspace allocation in the multi-per-context design; not required today since the existing pools preallocate."
    },
    {
      "lens": "cuda-target-compat",
      "title": "Persistent workers present, sized by occupancy, launched on graph-free path — not disabled",
      "severity": "P2",
      "confidence": 100,
      "evidence": [
        "render_engine.cu:923/962 'StridePrefixPersistentKernel' / 'FillTrianglePersistentKernel' defs",
        "render_engine.cu:346-382 persistent_fill_blocks_/persistent_stride_blocks_ sized via cudaOccupancyMaxActiveBlocksPerMultiprocessor",
        "render_engine.cu:1648,1671 launched in EnqueueRenderPhase(EvaluationContext&) on the non-blocking stream"
      ],
      "owner": "maintainer",
      "suggestedFix": "None — confirm enabling is correct; they are part of the U4 graph-free persistent-worker chain."
    },
    {
      "lens": "cuda-target-compat",
      "title": "Runtime for compute_120 fatbin is driver-arch-dependent even though device is sm_86",
      "severity": "P3",
      "confidence": 50,
      "evidence": [
        "compile_commands.json:gencode includes compute_120,code=sm_120",
        "probe_measurement.md:13 device is RTX 3090 Ti (sm_86) driver 610.57.04"
      ],
      "owner": "release",
      "suggestedFix": "Note in deployment: if the worker is ever run on Blackwell-class hardware, the driver must support compute 12.0, not just the sm_86 machine. On this machine the sm_120 fatbin is inert."
    },
    {
      "lens": "cuda-target-compat",
      "title": "cudaStreamWaitEvent silent-success pitfall and required record-order — key if adopted",
      "severity": "P3",
      "confidence": 75,
      "evidence": [
        "skill ref advanced-host-programming: a non-recorded event always returns success from cudaEventQuery/cudaStreamWaitEvent",
        "evaluation_executor.cu:31 cudaEventRecord is AFTER the enqueue, outside the queue tail"
      ],
      "owner": "maintainer",
      "suggestedFix": "If cudaStreamWaitEvent is introduced, never wait on an un-recorded event; keep the emit stream's work queued before cudaEventRecord and do not cudaEventRecord over a live referenced wait."
    }
  ]
}
```

---

## Acceptance report