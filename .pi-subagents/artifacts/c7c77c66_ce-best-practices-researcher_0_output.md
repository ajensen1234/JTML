I now have comprehensive grounding in the repo and authoritative research across all three topics. Let me compose the final research findings.

---

# JTML Refactor — Best-Practices Research Findings

Researched against the actual repo at `/home/ajj/repo/uf/JTML` (Qt5 + VTK 9.3 + CUDA 12.4 + OpenCV + PyTorch, C++20, CMake, pixi). This report is organized by (a) legacy-refactor strategy, (b) test-framework choice, (c) golden-file/regression testing, then mapped to your five concrete goals. Findings include concrete file paths and severities.

---

## Part A — Refactoring the god object (strangler, characterization, seams, DI)

### A1. Characterization tests — the correct first gate (your goal 1)
Authority: **Michael Feathers, *Working Effectively with Legacy Code*** (ch. 13, "Characterization Tests"); Wikipedia/cloudamite/understandlegacycode confirm **characterization test ≡ golden-master ≡ approval test**.

Key principle (Feathers): *legacy code = code without tests*, and the way to make it safe to edit is to **pin down existing behavior before changing it**. Two important corollaries for you:
1. **Test the seam/boundary, not the 5806-line `MainScreen`.** The prior abandoned attempt correctly rejected "run-the-real-MainScreen characterization tests" — characterization tests should be written at the *highest level you care to change*, which here is the **DIRECT optimizer / cost-function boundary**, not the GUI.
2. **A characterization test is worth only as much as the captured "known-good" is.** Your `golden_oracle.org` design is exactly the right pattern: cap the **final pose** (fem.jts) and the **projected Labels** from a trusted run.

**Recommendation (goal 1):** Build the golden oracle as an **optimizer-level characterization test**: load `example_studies/Kneel_1` (silhouettes + `KR_right_7_fem.stl`), run the DIRECT pipeline (trunk/2-branch/1-leaf, `DIRECT_DILATION`, dilations 6/3/1 px, ranges ±30/±20/±3 with z ~±100, ~10k iter budget), and a) assert the resulting pose converges to within tolerance of `fem.jts`, and b) assert projected silhouettes match `Labels/` within pixel tolerance. Do **not** drive `MainScreen` UI to capture these.

### A2. Seams — where the testability hooks go (your goals 2 & 3)
Authority: Feathers ch. 4 ("The Seam Model") + Martin Fowler's `LegacySeam` bliki. Definitions you should internalize:
- **Seam** = a place where you can alter behavior *without editing in that place*.
- **Enabling point** = the mechanism to change that behavior (here: a **constructor-injectable `std::function`**).

**Concrete seams found in the repo (with severity):**

| Seam | File:line | Current coupling | Severity |
|---|---|---|---|
| **DIRECT core** (`ConvexHull`, `TrisectPotentiallyOptimal`, `DenormalizeRange`, `DenormalizeFromCenter`, `SetSearchRange`, `SetStartingPoint`) | `src/core/optimizer_manager.cpp:869,879,1495,1553,1623,1633` | Private methods of `OptimizerManager`, but they reference **no CUDA state**; their only real dependency is `EvaluateCostFunction(Point6D)` (`optimizer_manager.cpp:1437`) | **HIGH — this is the cheapest, highest-value seam** |
| `DirectDataStorage`, `HyperBox6D`, `Point6D` | `src/core/direct_data_storage.cpp`, `src/core/data_structures_6D.cpp` | Already **pure** classes. **Only** coupling to GPU: `Point6D(gpu_cost_function::Pose)` ctor at `include/core/data_structures_6D.h:36` + `.cpp:34` (via `#include "gpu/render_engine.cuh"` at `data_structures_6D.h:24`), and `Point6D` is used in `sym_trap_functions` | **HIGH — a 1-line header surgery frees these for a pure test target** |
| **Optimizer header pollution** | `include/core/optimizer_manager.h:6-20` | `#include`s six CUDA `.cuh` headers | **MED** — any TU that includes `optimizer_manager.h` drags in CUDA; extraction must produce a CUDA-free header |
| Cost-function boundary | `CostFunctionManager::callActiveCostFunction()` | `optimizer_manager.h` members `trunk_manager_/branch_manager_/leaf_manager_` | **MED** — this is what gets replaced by the injected `double(Point6D)` |

**Recommended seam design (goal 3):** extract a new pure class, e.g. `DirectOptimizer`, whose shape is:
```cpp
class DirectOptimizer {
public:
  using CostFn = std::function<double(const Point6D&)>; // denormalized cost
  DirectOptimizer(CostFn cost, Point6D range, Point6D start, unsigned budget);
  Point6D run(); // ConvexHull/TrisectPotentiallyOptimal loop
};
```
This is the **"object seam"** Feathers prefers (explicit, maintainable — as opposed to preprocessing/link seams you'd need if you kept it inside `.cpp`). The extraction is **pure behavior-preserving** because `ConvexHull`/`Trisect`/`Denormalize*` are already deterministic functions of the data storage + cost. Test it with an **analytic cost function (e.g. a shifted quadratic/sphere)** → gives you a *bit-exact deterministic* golden for the algorithm itself, independent of GPU.

**To break the `Point6D`→`Pose` coupling (blocker for clean extraction):**
- Severity **HIGH, `include/core/data_structures_6D.h:36`**: move the `Point6D(gpu_cost_function::Pose)` constructor into the GPU-facing code (or delete and convert at call sites — only 2 real call sites: `optimizer_manager.cpp:1353`, `mainscreen.cpp:1376`). Then drop `#include "gpu/render_engine.cuh"` and `data_structures_6D` links with zero CUDA.

### A3. Strangler-fig for `MainScreen` and Qt5→Qt6 (your goals 4 & 5)
Authority: Fowler's StranglerFig + Azure Strangler Fig pattern + Shopify engineering writeup.

Principle: **incrementally replace functionality slice-by-slice; keep old and new coexisting until the old is decommissioned.** Apply it *inside* the process: as you decompose `MainScreen` into view/state/services, each extracted service replaces a call site in place, and the old call path is suppressed.

For `src/gui/mainscreen.cpp` (5806 lines, 79 `MainScreen::` methods, header `include/gui/mainscreen.h` → 400+ line private section):
- **Don't attempt big-bang MVVM.** Strangle outward from the one seam you already control: the **optimize-lifecycle coordinator** (goal 2). `OptimizerManager` is already `QObject` + worker-thread-based (`optimizer_manager.h` uses `QThread`); the state machine idle→running→finished→error is a *contained, GUI-free* target that can be tested under `QCoreApplication` + `QSignalSpy` with **zero GPU** by injecting a stub cost function. This is the ideal first strangled slice because it has boundaries (signals/slots) that MVVM-style `View/ViewModel/Service` can reuse.
- Then decompose view state out of `MainScreen` incrementally (model list → a service, pose storage → a service, render → a service), each behind a seam.

### A4. Dependency injection is "constructor injection of a `std::function`", not a DI container
Authorities: Dodd/Platis "Break the coupling in C++", Reverse Society / Holisticon on callback DI. For C++, the idiomatic testable seam is a **`std::function`** (or pointer-to-function) injected in the constructor — no magic container needed. Your coordinator and `DirectOptimizer` seams both follow this.

**Avoidance recommendation (honor the prior abandoned attempt):** do **not** recreate `qt_wrappers.h` function-pointer wrappers around `QFileDialog`/`QMessageBox` — you already correctly rejected that as low-value. Qt's supported seam for dialogs is `QDialog::exec` abstraction via injecting a factory, or testing at the model/command level instead. Keep the GUI thin enough that it needs no mocking.

---

## Part B — Catch2 vs QtTest for a mixed pure-logic + Qt suite

**Finding: use a hybrid, with a clear split.** Both are battle-tested; the decision hinges on *which layer* you test.

**QtTest (`QtTest`, `QSignalSpy`)**
- Strength: **native Qt object-model awareness** — testing signals/slots, widgets, and models is purpose-built (`QSignalSpy`, `QTest::qWait`, `QAbstractItemModelTester`). 
- No extra dependency — it ships with Qt, so it **migrates automatically to Qt6** (huge for your goal 5).
- Correct for: **the optimize-lifecycle coordinator** (idle→running→finished→error + worker thread) and any future widget/model tests, because they genuinely need the Qt event loop + `QSignalSpy`.
- Headless: works with `QCoreApplication` + `-platform offscreen` (confirmed: QtTest needs no X server for console/event-loop tests; `QSignalSpy::wait()` pumps the event loop).

**Catch2 (v3, compiled library)**
- Strength: **far richer test authoring** — BDD-style `GIVEN/WHEN/THEN`, `SECTION`s, generators, matchers, `Approx`/`WithinAbs` floating-point matchers, parameterized `TEMPLATE_TEST_CASE`. These make deterministic-numerics and DIRECT assertions much clearer than QtTest's `QCOMPARE`.
- First-class CMake integration: `Catch2::Catch2WithMain`, `catch_discover_tests()` → each `TEST_CASE` auto-registers with CTest, runs in parallel (see Catch2 docs).
- Correct for: **the extracted pure DIRECT core**, `Point6D`/`HyperBox6D`/`DirectDataStorage`, `StlReader`, `direct_data_storage`, and the golden-file comparisons — i.e., everything that does **not** need the Qt event loop.
- Caveat (confirmed via Catch2 GitHub #1217 + StackOverflow): Qt signals/slots need the event loop, so if you use Catch2 for Qt code you must **hijack Catch2's `main`** to construct `QCoreApplication`/`QApplication` (the established `#define CATCH_CONFIG_RUNNER` pattern). This is a real integration cost you avoid by using QtTest for the Qt layer.

**Bottom line / preference for this project:**
- **Pure logic (DIRECT, data structures, STL IO, golden comparisons) → Catch2 v3.** Third-party precedent: KWinFT ported their Qt project *to* Catch2 for exactly this reason (rich assertions) while keeping QtTest for integration.
- **QObject/threading/coordinator + widget/model tests → QtTest** (`QSignalSpy`), since these already need `QCoreApplication` and QtTest gives them for free with zero event-loop gluing.
- Both register into the same CTest via `catch_discover_tests()` (Catch2) and QtTest executables as CTest tests.

**Concrete setup for the goal-2 coordinator test** (this is the highest-leverage first suite):
```cpp
// coordinator_statemachine_test.cpp  (QtTest)
#include <QtTest>
#include <QCoreApplication>
#include <QSignalSpy>
class CoordinatorTest : public QObject {
  Q_OBJECT
private slots:
  void transitionsIdleToRunningToFinished() {
    Coordinator c(std::function<double(const Point6D&)>{
        [](const Point6D&){ return 0.0; }}); // stub cost, ZERO GPU
    QSignalSpy finished(&c, &Coordinator::finished);
    c.start();
    QVERIFY(finished.wait(2000));       // pumps event loop
    QCOMPARE(c.state(), Coordinator::Finished);
  }
};
QTEST_MAIN(CoordinatorTest)  // generates QCoreApplication main
#include "coordinator_statemachine_test.moc"
```
Run with `QT_QPA_PLATFORM=offscreen ctest`.

---

## Part C — Golden-master / golden-file regression of deterministic compute

### C1. Determinism caution (CPU-vs-GPU for the DRR renderer)
Authorities: NVIDIA CUDA Programming Guide ("Floating Point and IEEE 754"), NVIDIA Warp "Deterministic Execution", NVIDIA Tao/`framework-reproducibility`. Core facts your plan must respect:
- **Floating-point is *not* associative.** Reduction order changes results. 
- **CUDA is non-deterministic by default wherever there are floating-point atomics, auto-tuned algorithm selection (e.g. cuBLAS/cuDNN), or unordered reductions.** Warp is explicit: *"the most common source of non-determinism is a floating-point atomic."*
- **CPU and GPU are almost never bit-identical.** IEEE-754 standardizes *approximations*, not exact results, and FMA contraction differs.

**Implication for the golden oracle:** a golden pose captured from the CUDA renderer will **not** be bit-reproducible across GPUs/drivers, and will differ from a CPU render. **Do not demand bit-exact equality for anything touching the GPU renderer.**

### C2. Two-tier golden strategy (recommended)
1. **Tier 1 — pure CPU golden (bit-exact, deterministic):** golden the *extracted DIRECT algorithm* against an **analytic cost function** (no GPU at all). This validates the optimizer's logic exactly and is fully portable across CI. This is the defensible regression gate for the refactor.
2. **Tier 2 — GPU integration/oracle (tolerance-based):** compare the renderer's projected silhouettes/Labels using **relative/absolute tolerance and thresholded pixel diff**, *not* bit-exact equality. Capture golden images on the same hardware/platform it will run on (this is exactly why Chromium's screenshot-testing guidance says "run in the same environment where the baseline was captured" and why they added `-platform offscreen`).

### C3. Numerics-tolerance specifics
- **Pose regression:** compare `fem.jts` output against gold with explicit tolerances (e.g. `abs(x)<0.5 mm`, `abs(angle)<0.5°`, or `WithinAbs`/`Approx` matchers in Catch2). The DIRECT optimum for the *known-good* pose should land within optimizer resolution of the truth pose — document per-axis (z is opened to ±100, so z tolerance should be looser or you accept a different z).
- **Image/pixel regression:** use **thresholded pixel-difference** (count differing pixels beyond an intensity threshold, assert it's below an absolute pixel count or percent). This is the approach used by pbrt golden-image testing, Skia Gold, and echoSVG's `IMAGE_COMPARISONS.md`. For a binary silhouette (your "known-good" Labels are projected binary silhouettes), a small pixel-count threshold is appropriate — silhouette edges are where CUDA sampling/reduction nondeterminism shows first.
- **Do not** use a raw byte/golden sniffing of the floating cost history of DIRECT — DIRECT is deterministic *given a deterministic cost*, but the **cost history itself** is only meaningful at the same precision; assert on the final pose + the projected images, not on the full cost trace.

### C4. Where the golden files live
Mirror `golden_oracle.org`'s structure (`example_studies/Kneel_1`, `Labels/`, `fem.jts`, `KR_right_7_fem.stl`) into a `test/golden/` fixture dir; treat golden images/poses as committed artifacts with a documented regeneration command (rebase-on-approval, not silent overwrite).

---

## Concrete mapping to your five goals (severity-tagged)

1. **Golden-oracle regression gate** — HIGH value, do first. Optimizer-level characterization from `Kneel_1`; two-tier (Tier-1 CPU bit-exact DIRECT + analytic cost; Tier-2 tolerance-based Labels/silhouette). Run headless via `QT_QPA_PLATFORM=offscreen`.
2. **Optimize-lifecycle coordinator seam** — HIGH value, do next. Style: `QObject` state machine + worker thread, `QCoreApplication` + `QSignalSpy` + **injected stub cost** → zero GPU, fully testable in CI. Use QtTest here.
3. **Extract pure DIRECT** — HIGH value. New CUDA-free `DirectOptimizer` behind `std::function<double(const Point6D&)>`. Unblock by removing `Point6D(Pose)` ctor (`data_structures_6D.h:36`, `.cpp:34`) and dropping the `render_engine.cuh` include. This is pure `ConvexHull`/`TrisectPotentiallyOptimal`/`Denormalize*` extraction.
4. **Decompose `MainScreen` toward MVVM** — MED value / long-horizon. Strangle outward from services; start with the coordinator service, then models/pose/render services. Avoid widget-mocking wrappers; avoid "run the real MainScreen" tests. (Note: the available `mvvm` skill is generic/TypeScript — not directly applicable to Qt C++; rely on Qt's signals/slots/`Q_PROPERTY` as the binding surface.)
5. **Qt5→Qt6 migrate** — MED value, sequence LAST. Qt5 is EOL for OSS (only commercial extended-security exists); Qt 6.5 OSS support also ends ~April 2026, so target a current Qt6 LTS. Key blocker: **VTK is currently force-built Qt5** via `vtk_installer.sh` (`-DVTK_USE_QT6=OFF -DVTK_QT_VERSION=5 -DQt6_DIR=""`) and `pixi.toml` pins `qt = "5.*"`; `CMakeLists.txt:find_package(Qt5 ...)`. VTK supports Qt6 (`VTK_QT_VERSION=6`, needs the `OpenGLWidgets` component), but this is a coordinated pixi + VTK-installer + CMake change. **Sequence why it's last:** you want the golden oracle (goal 1) capturing the *known-good Qt5 behavior* first, so a Qt6 migration that perturbs rendering is caught by the gate instead of silently changing results.

---

## Residual risks
- **GPU/cross-platform nondeterminism** of the CUDA DRR renderer means the Tier-2 golden must be tolerance-based and re-captured per platform/GPU — a Tier-1-only CI gate will not catch renderer regressions on other hardware. 
- CUDA is a hard prerequisite; **CI without a GPU** cannot run Tier-2 (silhouette/Labels GPU tests) — plan for GPU-labeled CI jobs or skip-if-no-CUDA guards.
- `sym_trap_functions` uses `Point6D` from the same header; removing the `Pose` ctor must be verified against `sym_trap_functions` and the two call sites (`optimizer_manager.cpp:1353`, `mainscreen.cpp:1376`).
- The abandoned-attempt history (18 test files, wrapper abstractions) signals a risk of **over-engineering the test scaffolding**; keep goal-2/3 tests thin and behavior-focused.
- DIRECT optimizer quality isn't mathematically validated in this repo — the golden gate captures *current* behavior, not *correct* optimization; a known-good vs. perfect-pose discrepancy (especially opened-up z) may surface as a real (pre-existing) defect, not a regression.

## Review findings (severity-tagged)
- **high:** Direct-to-GPU coupling of the DIRECT core lives only in private methods + `EvaluateCostFunction` (`optimizer_manager.cpp:1437,1495,1553,1623,1633`); extraction is a low-risk behavior-preserving move once the `Point6D(Pose)` ctor (`data_structures_6D.h:36`/`.cpp:34`) is removed.
- **high:** `test/` is dead: root `CMakeLists.txt:122` has `#add_subdirectory(test)` commented; `test/CMakeLists.txt` only adds `nfd` + `vtk` (separate left-knee case); CI is inert boilerplate (`cmake.yml` triggers on a `actions-test` branch only, no install/setup, no deps). No test framework declared in `pixi.toml`.
- **med:** `OptimizerManager` header (`optimizer_manager.h:6-20`) drags six CUDA `.cuh` includes into any including TU — extraction must produce a CUDA-free header.
- **med:** Qt5 is EOL for OSS; migration is coupled to `vtk_installer.sh` Qt5 pin and `pixi.toml qt=5.*`; sequence after the golden oracle.
- **low:** `golden_oracle.org` is solid; recommend encoding its settings (trunk/branch/leaf budgets, dilations 6/3/1, ranges) as a committed fixture so the gate is reproducible.

(No source/test build was performed — this was read-only research; all findings are from static analysis + authoritative documentation.)

---