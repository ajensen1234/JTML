All validation complete. Final state:

- **Build**: `pixi run build` green (both new targets compile; the sym_trap pin's compile-time CUDA/OpenCV header chain parses with only OpenCV deprecation warnings — same relaxation as the existing `jtml.experimental_settings` target).
- **Targeted run**: `pixi run ctest --test-dir .build -R 'metric_semantics' --output-on-failure` → 2/2 passed.
- **Full headless suite**: `pixi run test` → 51/51 passed (49 pre-existing + 2 new; zero regressions).
- **RED pins surfaced**: the two sym_trap `[!mayfail]` pins genuinely fail as expected (`x2tib[0][3]` = 60.0 vs spec 50.0; `fem2tib[0][3]` = 0.0 vs spec 10.0), reported as "2 failed as expected" with the failure details in the console — recorded, not removed. The buggy-index-formula `REQUIRE_FALSE` characterization passes (documents that `distance_map_metric.cu:27` fails full coverage; crop pixels 0–1023 never visited, blocks (0,1)/(1,0) collide).
- **PBT twin**: 4,831 assertions across 6 hegel invariants, all green.
- **jj**: working copy left dirty, no describe/commit/new (per execution rules); diff is exactly the 3 intended files.

Implemented U1 of plan 008 — Tier-0 metric-semantics pins + CPU references.
Changed files: test/unit/test_metric_semantics.cpp (new, 22 deterministic pins), test/unit/test_metric_semantics_properties.cpp (new, 6 hegel PBT invariants), test/CMakeLists.txt (two headless targets, TIMEOUT 120/300).
Validation: `pixi run build` passed; `pixi run ctest --test-dir .build -R 'metric_semantics' --output-on-failure` 2/2 passed; `pixi run test` 51/51 headless green; RED sym_trap pins fail-as-expected (surfaced in output); PBT ran 4,831 assertions green.
Open risks/questions: see residualRisks below.
Recommended next step: U2 (behavior-neutral fixes — stage-guard `||`→`&&` + `getStage()` accessor, DD `min_dist=0.0` + Y-axis, sym_trap x-slot, dead-code deletion) — flips the RED sym_trap pins green (update transcribed call + drop `[!mayfail]`), lands the accessor-dependent stage-guard pin.