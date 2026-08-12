All research complete. Here is the distilled report.

---

## Institutional Learnings Search Results

### Search Context
- **Feature/Task**: JTML 'graph container' — data-driven optimizer stage sequences (StageScript) over the existing `Optimize()` loop in `src/coordinator/optimizer_manager.cpp`, preserving behavior bit-identically (R13 pin-first); Tier-0 characterization pins for 7 cost-path bugs; one oracle re-baseline; z-profile probe; multi-stage oracle test instruments.
- **Keywords Used**: stage-loop, stage math, optimizer_manager, Optimize, budget, cumulative caps, cost-path/cost-function, GPU/CUDA, direct optimizer, oracle/golden, baseline.json, re-baseline, pin-first, characterization, AUTOMOC, CMake GLOB, headless, QtTest seam, layer purity, R13/R15, epoch, seed.
- **Files Scanned**: 15 total (`docs/solutions/` has exactly 6 subdirs: build-errors, conventions, logic-errors, test-failures, tooling-decisions, ui-bugs)
- **Relevant Matches**: 13 (11 strong/moderate + 2 adjacent-weak)

### Critical Patterns
`docs/solutions/patterns/critical-patterns.md` does **not exist** in this repo — no global critical-patterns file to honor; the per-file conventions below carry the weight.

### Relevant Learnings

#### 1. Headless Testing and Testability-Refactor Conventions for a Qt/GPU Desktop App (JTML)
- **File**: `docs/solutions/conventions/jtml-testability-and-cmake-conventions-2026-08-07.md`
- **Module**: JTML (testing_framework)
- **Problem Type**: `convention` (severity: medium)
- **Relevance**: The master playbook for exactly this work — it documents the previous `DirectOptimizer` extraction out of `OptimizerManager::Optimize`, the headless seam pattern, the oracle/baseline.json contract, and the CMake traps you'll hit adding StageScript sources.
- **Key Insight**: **Cumulative budget is load-bearing** — the stage sequence is trunk 20,000 → 2 branches 5,000 each → leaf/z-search 5,000, accumulated (`cost_function_calls_` zeroed only before trunk, then `+=` per stage), with the seed consuming one unit; verified against `optimizer_settings.cpp` + `OptimizerManager::Optimize`, not prose. "Do NOT 'fix' it to per-stage caps." Extracting the loop means replacing the GPU touchpoint with an injected `std::function<double(const Point6D&)>` cost callback in denormalized physical space; Tier-1 golden = analytic cost, Tier-2 oracle = silhouette/IoU (never raw pose — CUDA reductions/atomics are not bit-reproducible); baselines live in `test/golden/baseline.json` (captured from the current app → a behavior-preservation gate, **not** a correctness check).
- **Severity**: medium (but load-bearing for bit-identical preservation)

#### 2. Shared VM layer (plan 006): the conventions that make the two front-ends one
- **File**: `docs/solutions/conventions/jtml-shared-vm-layer-2026-08-11.md`
- **Module**: jtml_coordinator
- **Problem Type**: `convention` (severity: medium)
- **Relevance**: The stage-loop/manager patterns doc. `OptimizerRunControllerCore` is the Qt/GPU-free core (gate, 5-state machine, **epoch, stage math, seed**) precedent for a StageScript core; and the doc explicitly reserves the seam you need: "This is also the **multi-stage oracle's future entry point** (synthesis item 3)".
- **Key Insight**: `OptimizerRunDriver` (interface + production adapter) is the swappable-run seam that wraps `OptimizerManager` UNTOUCHED — drive the graph container through it, don't fake the manager (`Initialize` is non-virtual; test subclasses can't intercept it). R13 per cut type: parameterized extractions are "verbatim-behavior extraction with signature adaptation — gated by the **characterization test + behavior diff**, not a literal `jj diff`". **Measurement-first doctrine**: "no algorithm deltas before the harness exists (**the silhouette-IoU oracle cannot see the z-weak-axis problem**)" — the documented gap your z-profile probe is presumably filling. Also: "pin, don't unify" (4 divergent save-last-pose behaviors pinned by call-site table tests), epoch-tagged runs + `threadActive` Start gate, destructor contract = stop → `quit()` + bounded `wait()` → delete, and the grep-gate that coordinator code contains no `QWidget`/`QQuick`/`vtkRenderWindow`.
- **Severity**: medium

#### 3. MainScreen decomposition playbook: QListView swap, settings interleave, per-frame controller API
- **File**: `docs/solutions/conventions/jtml-mainscreen-decomposition-patterns-2026-08-10.md`
- **Module**: jtml_view
- **Problem Type**: `convention` (severity: medium)
- **Relevance**: Direct precedent for refactoring a GPU/interleave-heavy loop while preserving behavior byte-identically — the same problem as StageScript-over-`Optimize()`.
- **Key Insight**: For the segment/estimate block, "a controller-owned loop forces the interleave to change; instead **the view owns the loop and the controller exposes per-frame ops**… Byte-identical interleave is then verifiable by **counting the calls before/after**." Apply: keep `Optimize()` as the call-owner while StageScript drives it, and pin the per-stage call counts (cost-function calls, progress/processEvents/render emissions) before/after as the Tier-0 verification. Also: torch's `ATen/core/ivalue_inl.h` does `#undef slots` — torch-bearing includes must sit after Qt-object headers (`optimizer_manager.h`…); GPU-gated tests label `oracle` only (there is no `gpu` label).
- **Severity**: medium

#### 4. JTML Qt5->Qt6 migration, hegel PBT, and oracle tooling recipes
- **File**: `docs/solutions/tooling-decisions/qt6-hegel-oracle-tooling-recipes-2026-08-08.md`
- **Module**: JTML (build_test_tooling)
- **Problem Type**: `tooling` (severity: medium)
- **Relevance**: The oracle re-baseline recipe book — how the Tier-2 appearance oracle was built and how baselines must be re-captured.
- **Key Insight**: Load-bearing gate is **silhouette/IoU, never raw recovered pose** (DIRECT numeric convergence is noisy). Pin frame↔label correspondence **empirically** (start-pose render IoU == 1.0), never by filename; binary label TIFFs are bottom-left y-origin vs the renderer's top-left → `cv::flip(label, 0)`; feed the oracle processed Frame outputs (edge/dilation/intensity/distance-map), not raw x-ray; ctest entries need `WORKING_DIRECTORY` = repo root (ctest defaults to `.build`); label `oracle` only, never headless. Gate migrations with the **pre-migration oracle + headless suite first** — the same sequencing applies to your re-baseline (capture the new baseline from the current build before the container lands). Also: hegel PBT is runner-agnostic (call inside a Catch2 `TEST_CASE`), conda GCC ignores `BUILD_RPATH` (append `-Wl,-rpath` via `target_link_options`).
- **Severity**: medium

#### 5. Cost-function parameter registry silently truncated doubles to int
- **File**: `docs/solutions/logic-errors/cost-function-parameter-double-truncation-2026-08-08.md`
- **Module**: jta_cost_function
- **Problem Type**: `logic_error` (severity: high)
- **Relevance**: One of the cost-path bug class the Tier-0 pins must cover — silent narrowing in the cost parameter registry (`Parameter<double>` stored `int`), invisible to example-based tests because integer-valued doubles round-trip.
- **Key Insight**: Pin with a **hegel PBT asserting bit-exact typed round-trip** (negatives, ±0.0, fractional draws) — and "when a property test whose invariant is clearly right fails against existing code, treat the code as the bug; do not weaken the invariant to 'pass'". Also: the stale-build trap — a reused pre-split `.build` tree hid the dependency ("ninja reported no work to do") until a forced recompile; a clean build dir/CI doesn't hit it.
- **Severity**: high

#### 6. Guarding an allocation path: null-init the freed members and validate the guard's precondition (GPUHeatmap 0-keypoint)
- **File**: `docs/solutions/logic-errors/jtml-heatmap-guard-allocator-preconditions-2026-08-12.md`
- **Module**: jtml_compute
- **Problem Type**: `logic_error` (severity: high)
- **Relevance**: Second documented cost-path bug inside the `OptimizerManager::Initialize` flow (`Error uploading heatmap to GPU!` on 0-keypoint studies) — a candidate for your 7-bug characterization set, and the model for how to pin GPU paths that can't run headless.
- **Key Insight**: The guard triad — (1) constructor-level no-upload state so `cudaFree(0)` is a legal no-op, (2) null-init every member the destructor frees, (3) validate the guard's precondition (keypoint count was never initialized in the `Frame` ctor → garbage reads bypassed the guard). "The 0-keypoint GPU path is oracle/manual-visual territory (no headless GPU) — **the pins are the constructor/guard invariants** plus the owner's heatmap-less run; run compute-sanitizer on teardown." Keep the frame-aligned vector contract (skipped upload still pushes a valid object so `at(i)` consumers stay aligned).
- **Severity**: high

#### Additional matches (compact)
- `docs/solutions/conventions/jtml-layered-lib-split-2026-08-08.md` (`convention`, medium) — every new StageScript `.cpp`/`Q_OBJECT` header must be added to the **explicit source list** (GLOB headers + `CONFIGURE_DEPENDS` only catches headers); `jtml_coordinator` STATIC owns `optimizer_manager`; graph is downward-only `domain ← services ← coordinator ← view ← app`; zero-behavior-change relocations are attested via scripted exact-string rewrites + `jj diff`.
- `docs/solutions/build-errors/jtml-moc-signals-section-placement-duplicate-definition-2026-08-11.md` (`build_error`, medium) — sibling AUTOMOC trap: a `signals:` section mid-class mocs every following accessor → `multiple definition of <Class>::<accessor>` between `.cpp.o` and `mocs_compilation.cpp.o` (duplicate = mis-scoped signals; undefined = header never in source list); keep `signals:` last — relevant when the graph container adds signals to Q_OBJECT stages.
- `docs/solutions/test-failures/jtml-test-isolation-qsettings-cache-2026-08-12.md` (`test_failure`, medium) — headless-seam trap: QSettings/QStandardPaths cache the config path per process, so per-case `XDG_CONFIG_HOME` redirects silently leak state (one isolated `TEST_CASE` per config, redirect before first QSettings construction); `tryCompare` does not resolve dotted paths — compare the terminal object's own property.
- `docs/solutions/tooling-decisions/jtml-rendering-runtime-xcb-qvtk-2026-08-10.md` (`tooling_decision`, high) — headless render verification must go through the QVTK path (`widget->grab()` PNGs, non-blank check), never a standalone `vtkRenderWindow` (VTK built `VTK_USE_X=OFF`/no EGL/OSMESA → factory returns base class, `vtkWindowToImageFilter` segfaults); run under `QT_QPA_PLATFORM=xcb`; when suspecting a refactor broke rendering, "diff the display path against the pre-refactor baseline first".
- `docs/solutions/conventions/jtml-qml-experimental-frontend-2026-08-11.md` (`convention`, medium) — R13-forced duplication residue precedent: `LaunchOptimizer` drive sequence mirrored in `OptimizerBridge` and `BuildCostFunctionRegistryEntries` replicated + golden-fixture-pinned until plan 006 extracted them; also confirms the app drives the REAL `OptimizerManager`.
- `docs/solutions/conventions/jtml-qml-view-testing-2026-08-12.md` (`convention`, medium) — QML harness aliases the REAL shipped `.qml` sources (no drift) — the no-drift fixture pattern to copy for test instruments.
- `docs/solutions/ui-bugs/jtml-qml-model-pose-sync-queued-functor-never-delivered-2026-08-11.md` (`ui_bug`, high) — `QMetaObject::invokeMethod(this, functor, Qt::QueuedConnection)` returns true but the functor never executes — prefer by-value signal emit; smoke leg 3.5 drives the real pipeline headlessly. (Adjacent — test-instrument pattern only.)
- `docs/solutions/conventions/qml-listview-delegate-patterns-2026-08-12.md` and `docs/solutions/ui-bugs/jtml-qml-trace-driven-debugging-2026-08-12.md` — QML-delegate/profiler content, weak relevance to stage-loop refactoring (noted for completeness).

### Recommendations
- **StageScript entry point**: consume the `OptimizerRunDriver` seam (`src/coordinator/`) — it was explicitly reserved as "the multi-stage oracle's future entry point" — and model the stage-sequence core on `OptimizerRunControllerCore`'s Qt/GPU-free purity (grep-gate: no `QWidget`/`QQuick`/`vtkRenderWindow` in coordinator).
- **Bit-identical preservation anchors**: (a) keep the cumulative budget semantics verbatim (trunk 20k + 2×5k branches + 5k leaf, seed consumes one unit — "do NOT fix to per-stage caps"); (b) verify byte-identical interleave by **counting calls before/after** (per-frame controller pattern); (c) gate the cut with the characterization test + behavior diff, not a literal `jj diff` (R13 per cut type).
- **Tier-0 pins for the 7 cost-path bugs**: only 2 of the 7 are documented in solutions (double-truncation registry, heatmap 0-keypoint guard) — use hegel bit-exact PBT for registry values, constructor/guard invariants for GPU paths, and treat a failing clear invariant as a real defect, never weaken it. Consider capturing the other 5 with `/ce-compound` after they're pinned.
- **Oracle re-baseline**: record in `test/golden/baseline.json`; keep the IoU/silhouette gate (never raw pose); pin frame↔label correspondence empirically; capture the pre-container baseline from the current build before the graph container lands; run oracle-labeled tests from repo root only.
- **Z-profile probe**: the docs explicitly state "the silhouette-IoU oracle cannot see the z-weak-axis problem" — your probe is the sanctioned measurement-first prerequisite; no algorithm deltas before it exists.
- **CMake/build**: any new StageScript `.cpp` and any new `Q_OBJECT`/signals-bearing header must be added to the explicit source list of its layer (GLOB + `CONFIGURE_DEPENDS` won't save you); keep `signals:` sections last; beware stale `.build` trees hiding new dependencies (forced recompile or clean dir).

---