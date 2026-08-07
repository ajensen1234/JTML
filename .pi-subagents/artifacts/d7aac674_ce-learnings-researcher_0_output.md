I have completed my research. There is no `docs/solutions/` directory nor initialized memory KB in this repo — the institutional knowledge lives in the brainstorm decision record and `golden_oracle.org`. Here are the distilled learnings.

---

## Institutional Learnings Search Results

### Search Context
- **Feature/Task**: Placing a golden-oracle regression gate, a headless-testable optimize-lifecycle seam, a pure-DIRECT extraction behind an injected cost boundary, and eventual MainScreen MVVM decomposition + Qt6 migration for JTML (Qt5+VTK+CUDA+OpenCV 2D-3D knee registration).
- **Keywords Used**: golden-master / behavior-preservation, false-confidence, circular-tests, lifecycle orchestration, testability seam, MVVM/god-object, Qt6/VTK migration, test-framework, CI, anti-circularity, nocache
- **Files Scanned**: ~30 knowledge/artifact files (full `docs/`, `.memory-bank/`, `.claude/`, `.pi-subagents/artifacts/`, root docs, CMake/pixi configs); no dedicated KB directory exists.
- **Relevant Matches**: 5 (2 high-severity decision-record items + oracle-spec + 2 data/infrastructure facts)

### Critical Patterns
`docs/solutions/patterns/critical-patterns.md` does **not** exist in this repo. There is no `docs/solutions/` tree at all. The institutional knowledge for this exact effort is concentrated in two docs: `docs/brainstorms/2026-08-07-testability-mvvm-refactor-requirements.md` and `golden_oracle.org`.

### Relevant Learnings

#### 1. The abandoned-attempt failure modes are the single most load-bearing learning (severity: HIGH)
- **File**: `docs/brainstorms/2026-08-07-testability-mvvm-refactor-requirements.md` (Problem Frame; R2; R12; Key Decisions)
- **Module**: testability / test-harness design for the whole refactor
- **Problem Type**: `workflow_issue` (inferred — captured as the driving retrospective lesson)
- **Relevance**: This is the authoritative account of why the prior 18-file test attempt was rejected, and it defines the hard constraints the new plan must respect. The attempt failed on two independent axes: (a) **false confidence** — tests covered the pure math and a GPU-bound "run the real MainScreen" test, so they were green while never touching the thread/orchestration seams where the real hangs/regressions lived; (b) **circularity** — several test helpers re-derived the code-under-test's own math (pose-file parsing and denormalization tests reimplemented production logic), so green tests proved nothing.
- **Key Insight**: The plan's sequencing is *directly* derived from this: **lifecycle-seam-first** (seal the optimize threading before touching DIRECT math, on the hypothesis that hangs live in orchestration), **no Qt-mocking wrappers** (`qt_wrappers.h` function-pointer wrappers around `QFileDialog`/`QMessageBox` are explicitly banned), **no instantiate-the-real-`MainScreen` god-object characterization tests**, and **anti-circularity (R2)** — no test may assert a value re-derived from the code under test; extracted logic must be exercised against the oracle or an independent ground truth (analytic cost functions, known-good `Labels` projections B).
- **Severity**: High

#### 2. The R1–R16 decision record is the anchor convention to plan against (severity: HIGH)
- **File**: `docs/brainstorms/2026-08-07-testability-mvvm-refactor-requirements.md` (Requirements R1–R16; Acceptance Examples AE1–AE5)
- **Module**: whole refactor (oracle, lifecycle, DIRECT/cost seams, MVVM, Qt6, CI)
- **Problem Type**: `convention` (inferred — it is the settled decision/requirements contract)
- **Relevance**: Any plan that diverges from these requirements risks repeating the abandoned attempt (R12) or shipping un-gated extractions (R9). The contract fixes the key semantic decisions: the oracle is a **behavior-preservation/golden-master gate, not a correctness check** (a green oracle proves the refactor preserved current behavior; genuine correctness comes from independent sources and the known-good Labels B); the **optimize lifecycle (idle→running→finished→error, worker thread, `QSignalSpy` under `QCoreApplication`, stub cost fn, bounded/re-triggerable)** is R4–R7; **DIRECT is preserved and gated, not re-derived** (R15); **Qt6 migration is real scope**, gated by the pre-migration baseline + headless suite with a bounded manual GUI smoke step (R16); **testing enabled in CMake with default headless run and GPU/real-VTK in a separate flagged target** (R11, R13).
- **Key Insight**: Treat R1–R16 and AE1–AE5 as normative. Notably R7 splits testable-vs-not by seam: cost-init-failure and stop paths are covered headlessly with a stub cost (no GPU), while the *real* GPU-init-failure path lives only in an explicitly-flagged GPU target — so the headless default must not try to touch CUDA init failure.
- **Severity**: High

#### 3. Golden-oracle settings were flagged inconsistent and remain a pre-flight precondition (severity: MEDIUM)
- **File**: `golden_oracle.org`; inconsistency flagged in `docs/brainstorms/...` → Outstanding Questions → "Resolve Before Planning"
- **Module**: oracle foundation (R3, AE2, D1)
- **Problem Type**: `documentation_gap` (inferred)
- **Relevance**: R3/AE2 require agreed numeric tolerances and a decided DRR/cost reference (CPU render vs GPU path) recorded in `golden_oracle.org` **before planning the oracle phase**. The brainstorm flags that the optimizer-settings block in `golden_oracle.org` was inconsistent (dilation list duplication/omission). Note: the current `golden_oracle.org` I read now reads "trunk 6px / branch 3px / leaf 1px dilated" — consistent with the task's expected values (trunk 6, branch 3, leaf 1; ranges ±30/±20/±3 with z opened ~100; ~10000 budget; DIRECT_DILATION) — so it may have been corrected since the brainstorm flagged it. **Verify**, don't assume.
- **Key Insight**: The tolerance/reference decision is the load-bearing precondition of the golden gate; it must be resolved and recorded before phase 1 can be considered startable (D3 pre-flight gate also requires the user to capture the Qt5 GPU baseline first).
- **Severity**: Medium

#### 4. The only real test data in the repo is the wrong (left) knee; the oracle must source `example_studies/Kneel_1` (severity: MEDIUM)
- **File**: `test/vtk/test_case/` (contains `KR_left_8_fem.stl`, `KR_left_8_tib.stl`, `fem.jts`), vs `example_studies/Kneel_1/` (`KR_right_7_fem.stl`, `KR_right_6_tib.stl`, `fem.jts`, `Labels/`)
- **Module**: test fixtures / oracle data (D2)
- **Problem Type**: `convention` / data-reality (inferred)
- **Relevance**: Confirmed: `test/vtk/test_case` is a **LEFT-knee** case (`KR_left_8_*`) and is *not* usable to reproduce the Kneel_1 oracle. The oracle must draw silhouettes (A), known-good Labels (B), `fem.jts` (C), and femoral STL (D) from `example_studies/Kneel_1/`. The only existing real test-side C++ is `test/nfd/nfd_speed_test.cpp` (a speed benchmark, not a regression test) — the `test/` subdir is otherwise commented out (`#add_subdirectory(test)` at `CMakeLists.txt:122`, `#enable_testing()` at `:129`).
- **Key Insight**: Do not reuse `test/vtk/test_case` as oracle input; point the oracle gate at `example_studies/Kneel_1/` and treat it as the authoritative single-case regression fixture until additional fixtures are added as a follow-on.
- **Severity**: Medium

#### 5. Infrastructure conventions: no test framework, no CI, no memory KB — R14 and R13 are net-new (severity: LOW/MEDIUM)
- **File**: `pixi.toml` (no `[tasks]`, no Catch2/QtTest declared), `CMakeLists.txt:121-129` (test subdir + `enable_testing()` commented), `.github/workflows/cmake.yml` (inert; triggers only on branch literally named `actions-test`, runs plain `cmake -B build`), `.build/` present
- **Module**: test harness / CI / onboarding conventions (R13, R14)
- **Problem Type**: `tooling_decision` (inferred)
- **Relevance**: Verified: no test framework is declared in pixi (R14 must add one — Catch2 vs QtTest is a deferred-planner decision), CI is inert boilerplate (ASSUMPTION marked verified in the brainstorm), and there is **no initialized project/global memory** (`/home/ajj/.pi/memory-md/JTML` not found). There is no `AGENTS.md`; `justfile` only has format/tidy aliases via `format.sh`/`tidy.sh` (clang-format/clang-tidy). The Qt6/VTK work is already partly underway (the checked-in `big-main-diff.diff` shows RPATH additions for a locally-built VTK install and changed `REQUIRED_LIBS` in `CMakeLists.txt`), consistent with R16's VTK rebuild against Qt6.
- **Key Insight**: The headless suite, timeout-bounded lifecycle tests, and CI must all be built from scratch; don't assume a harness exists. Capture this refactor's outcome with `/ce-compound` afterward — with no memory KB, the only durable record today is the brainstorm doc + `golden_oracle.org`.
- **Severity**: Low–Medium

### Recommendations
- **Plan against the R1–R16 + AE1–AE5 contract verbatim.** Treat the brainstorm doc as the normative decision record; any seam/sequencing divergence (especially lifecycle-first and per-layer gating) must be explicit.
- **Sequence exactly as the doc phases it:** (1) oracle + test-runner + CI foundation → (2) lifecycle seam → (3) DIRECT/cost seams → (4) MVVM (only after re-validation against human outcome) → (5) Qt6 gated by the oracle. This ordering is itself an institutional lesson from the abandoned attempt.
- **Enforce anti-circularity (R2) and the two AVOIDs structurally**, not by good intentions: reject any plan item that re-implements production math or wraps Qt (`QFileDialog`/`QMessageBox`) behind function pointers, and reject any "instantiate real `MainScreen`" characterization test.
- **Resolve the pre-flight gates before planning the oracle phase:** the DRR/cost reference (CPU vs GPU) + tolerance values, the Qt5 GPU baseline capture feasibility (D3 fallback = defer Qt6), and re-confirm the (now-consistent-looking) `golden_oracle.org` settings block.
- **Source all oracle fixtures from `example_studies/Kneel_1/`** (right knee); never from `test/vtk/test_case` (left knee).
- **Add a test framework to pixi (R14), enable testing + headless default in CMake, and build real CI (R13)** — all net-new, none exists today.

---

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Returned 5 concrete findings, each with repo-relative file paths and severity: (1) HIGH abandoned-attempt failure modes (false confidence + circular tests) in docs/brainstorms/2026-08-07-testability-mvvm-refactor-requirements.md; (2) HIGH R1-R16 decision record/AE1-AE5 contract in same doc; (3) MEDIUM golden_oracle.org settings inconsistency + pre-flight tolerance/reference precondition; (4) MEDIUM wrong-knee data reality (test/vtk/test_case is KR_left_8_* vs example_studies/Kneel_1 right knee) + #add_subdirectory(test)/#enable_testing() commented at CMakeLists.txt:122/129; (5) LOW-MED infra facts (no test framework in pixi.toml, inert .github/workflows/cmake.yml, no AGENTS.md, no initialized memory, RPATH/VTK Qt6 work already in big-main-diff.diff)."
    }
  ],
  "changedFiles": [],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "find docs/solutions/ && ls docs/{brainstorms,plans,superpowers}",
      "result": "passed",
      "summary": "Confirmed no docs/solutions/ KB exists; docs/brainstorms has exactly one file (the decision record)."
    },
    {
      "command": "cat docs/brainstorms/2026-08-07-testability-mvvm-refactor-requirements.md",
      "result": "passed",
      "summary": "Read the full R1-R16 decision record + abandoned-attempt lessons + pre-flight preconditions."
    },
    {
      "command": "cat golden_oracle.org && grep -n test CMakeLists.txt && ls test/vtk/test_case/*.stl",
      "result": "passed",
      "summary": "Verified golden_oracle settings (trunk6/branch3/leaf1, DIRECT_DILATION, ranges), test subdir + enable_testing commented out, and left-knee KR_left_8_* fixtures."
    },
    {
      "command": "memory_check + find .memory-bank .claude -type f",
      "result": "passed",
      "summary": "Confirmed empty .memory-bank/.claude and no initialized /home/ajj/.pi/memory-md/JTML memory."
    }
  ],
  "validationOutput": "Verification: test subdir disabled (CMakeLists.txt:122,129); no test framework in pixi.toml; .github/workflows/cmake.yml inert; AGENTS.md absent; only real test C++ is test/nfd/nfd_speed_test.cpp; test/vtk/test_case is left-knee (KR_left_8_*). golden_oracle.org currently reads trunk 6px/branch 3px/leaf 1px (consistent with the task), though the brainstorm flagged it as previously inconsistent - flag for verification before planning.",
  "residualRisks": [
    "No docs/solutions/ tree or initialized memory exists, so the 'institutional learnings' here are the brainstorm decision record + golden_oracle.org rather than a curated learnings KB; the absence is itself signal that the refactor outcome should be captured via /ce-compound.",
    "golden_oracle.org settings block was flagged inconsistent in the brainstorm; current file content appears corrected but should be re-verified before planning (pre-flight precondition R3/D1).",
    "DRR/cost reference (CPU vs GPU render) and numeric tolerance values are unresolved pre-conditions for the oracle gate and are not yet recorded in golden_oracle.org.",
    "Qt5 GPU baseline capture feasibility (D3 pre-flight) is unconfirmed; if unobtainable, R1/R16 needs a fallback (defer Qt6).",
    "The three 'Resolve Before Planning' decisions are upstream of phase 1 and out of this researcher's scope to resolve."
  ],
  "noStagedFiles": true,
  "diffSummary": "Research-only task: no source files changed; findings are read-only distillations with no diff.",
  "reviewFindings": [
    "info: docs/solutions/ does not exist in this repo; institutional knowledge is concentrated in docs/brainstorms/2026-08-07-testability-mvvm-refactor-requirements.md and golden_oracle.org. No blockers."
  ],
  "manualNotes": "Highest-value takeaways for the planner: (1) lifecycle-seam-first sequencing, no-Qt-mocking wrappers, and no real-MainScreen characterization tests are direct lessons from the abandoned 18-file attempt and are codified as R2/R12; (2) the golden oracle is a behavior-preservation gate, not a correctness check - independent ground truth (analytic cost fns + known-good Labels B) is required to avoid circularity; (3) source oracle fixtures exclusively from example_studies/Kneel_1 (right knee), never test/vtk/test_case (left knee); (4) test harness, test framework, and CI are all net-new (R11/R13/R14)."
}
```

**Summary:** No `docs/solutions/` KB or initialized memory exists in this repo, so I surfaced the actual institutional knowledge that governs this refactor: the **brainstorm decision record (R1–R16 + AE1–AE5)**, which embeds the **abandoned-attempt lessons** (false-confidence + circular tests) that drive the mandatory lifecycle-seam-first sequencing, the anti-circularity rule, and the explicit bans on Qt-mocking wrappers and god-object characterization tests. I also flagged the golden-oracle pre-flight preconditions (settings/DRR-reference/tolerance), the wrong-knee test-data reality (`test/vtk/test_case` is left knee; use `example_studies/Kneel_1`), and that the test framework, headless harness, and CI are all net-new.