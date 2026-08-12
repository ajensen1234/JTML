All verification complete. I've traced every load-bearing claim in the plan against the code. Summary of what I verified before writing findings:

**Verified correct (extensive):** stage-loop landmarks (:995–996 trunk reset, branch group init/dilate/emit before the repeat loop, per-branch `budget_ +=`, `CalculateSymTrap` call, leaf error-gated destruct vs trunk unconditional, epilogue trunk-restore); driver seam (by-value launch, verbatim forwarding, sole `new OptimizerManager`); cumulative guard `(calls + offset) < budget_`; `Directive::SymTrap = 5`; the CFM stage-guard tautology (`||` chain at CostFunctionManager.cpp:46–48, and `stage_` is never read anywhere — blast radius nil); the `distance_map_metric.cu:27` grid-index `+`-vs-`*` bug; oracle twin + kBudget=3000 + IoU 0.85 gate; baseline.json caps [20/25/30/35k], Canny 3/0/150, 6.31 mm z-gap, 15 mm z band; v1 graph ranges/budgets match settings_constants.h 1:1; production dilation 6/4/1 (branch→4, leaf→1 overrides); ATen include at CostFunctionManager.h:28 + torch link feasible as PRIVATE (all consumers link torch themselves); DD uninitialized `min_dist` + Y_dist≡X_dist; sym_trap tibia z-z duplication (:106); commented-out dead DistanceMapMetric; Mahfouz pointer-guards (:324/:446); int-atomic determinism; int-atomic NaN-never-stores already true; tibia fixtures present; test conventions.

**Found 5 findings** — 2×P2, 3×P3, all anchored ≥50.

```json
{
  "reviewer": "ce-feasibility-reviewer",
  "findings": [
    {
      "title": "U6 sym-trap costCalls pin '20000' contradicts the production shape: code runs trunk + 2×branch = 30000 before the sym-trap (only the leaf search is skipped)",
      "severity": "P2",
      "section": "U6 (Multi-stage oracle on the driver seam) — 'Three sym-trap pins'",
      "why_it_matters": "An implementer asserting costCalls()==20000 against the oracle's production-shape tibia SymTrap run will observe 30000 (trunk 20000 + 2 branches × 5000; ENABLE_BRANCH=true and NUMBER_BRANCHES=2 are the defaults, and the plan says the oracle drives 'the production shape — the same shape the jtml-production graph will express in U9', which includes the Branch spec). The pin fails on the happy path, and the dangerous resolution is to 'fix' the manager to skip branches for Sym_Trap so the pin passes — a behavior change that silently narrows the very sym-trap coverage the multistage oracle was built to provide. The derivable correct value for the plan's own shape is 30000 (leaf skipped, NOT 35000); the alternative (branches-off tibia launch) must be stated explicitly and contradicts 'the production shape'.",
      "finding_type": "error",
      "autofix_class": "gated_auto",
      "confidence": 75,
      "evidence": [
        "plan: 'costCalls() lands at 20000 (leaf skipped — NOT 35000)'",
        "repo src/coordinator/optimizer_manager.cpp:1040 (trunk, unconditional): 'if (!error_occurrred_) { RunDirectStage(optimizer_settings_.trunk_range, trunk_manager_); }'",
        "repo src/coordinator/optimizer_manager.cpp:1080-1103 (branch loop, no sym_trap guard): 'for (int branch_index = 0; branch_index < optimizer_settings_.enable_branch_ * optimizer_settings_.number_branches; branch_index++) { ... budget_ += optimizer_settings_.branch_budget; ... RunDirectStage(optimizer_settings_.branch_range, branch_manager_); }'",
        "repo src/coordinator/optimizer_manager.cpp:1140 (only the leaf search is sym-trap-guarded): 'if (optimizer_settings_.enable_leaf_ && !error_occurrred_ && !sym_trap_call) {'",
        "repo include/domain/settings_constants.h: 'const bool ENABLE_BRANCH = true;' / 'const int NUMBER_BRANCHES = 2;'",
        "plan: 'Drive OptimizerRunController directly ... with the production shape — the same shape the jtml-production graph will express in U9'"
      ]
    },
    {
      "title": "U6 orientationSymTrapUpdated pin 'assert == 60' is off by one: the happy path relays 61 (60 in-loop emits + 1 final reset emit); the cited early-return line ref is also ~10 lines stale",
      "severity": "P3",
      "section": "U6 — 'Three sym-trap pins'",
      "why_it_matters": "CalculateSymTrap emits onUpdateOrientationSymTrap 60 times in the loop (iter_val=60) and once more after the loop to restore the initial pose, and the controller relay (optimizer_run_controller.cpp:332) forwards 1:1 with no filtering — so a happy-path relay count is 61, and an implementer asserting == 60 gets a failing pin on the very run the pin is meant to validate. The mechanically correct pins are == 61 or >= 60 (the silent early return yields 0, which '>= 1' discriminates). The early-return guard is actually at optimizer_manager.cpp:1304-1309 (CalculateSymTrap begins :1303), not :1294-1300.",
      "finding_type": "error",
      "autofix_class": "safe_auto",
      "confidence": 75,
      "evidence": [
        "plan: 'orientationSymTrapUpdated relay count ≥ 1 (assert == 60 to catch the silent early return at optimizer_manager.cpp:1294–1300)'",
        "repo src/coordinator/optimizer_manager.cpp:1313-1326: 'int iter_val = 60;' ... 'for (int i = 0; i < iter_val; i++) { emit onUpdateOrientationSymTrap(pose_list.at(i).x, ...'",
        "repo src/coordinator/optimizer_manager.cpp:1347: 'emit onUpdateOrientationSymTrap(pose_6D.x, pose_6D.y, ...);' (final reset, after the loop)",
        "repo src/coordinator/optimizer_run_controller.cpp:332: 'emit orientationSymTrapUpdated(x, y, z, xa, ya, za);' (1:1 relay, no filtering)",
        "repo src/coordinator/optimizer_manager.cpp:1303-1309: 'void OptimizerManager::CalculateSymTrap() { if (current_optimum_location_.xa == 0 && ... za == 0) { cout << \"ERROR: INVALID STARTING POSE FOR SYMMETRY TRAP\" << endl; return; }'"
      ]
    },
    {
      "title": "TIMEOUT 600 on jtml.oracle_multistage is too short for the production shape: ~135k GPU evals ≈ 30–75+ min at the flat oracle's own documented rate (~13–33 ms/eval)",
      "severity": "P2",
      "section": "U6 — Files (new target 'jtml.oracle_multistage', ... TIMEOUT 600)",
      "why_it_matters": "The multistage oracle executes the full production shape through the real manager: femur run across 3 frames (3 × 35000 evals, per 'frame-to-frame chaining across the 3 Kneel_1 frames') plus the tibia SymTrap run (~30000 evals) ≈ 135k cost evals. The existing flat oracle documents its rate: 9000 evals (3 × kBudget 3000) take 'a few minutes' (oracle_test.cpp:244), and the synthesis anchors JTML's per-eval cost at 50–150× Flood's ~3000 evals/s (≈17–50 ms/eval). At even the optimistic 13 ms/eval the run needs ~29 min; CTest will kill it at 600 s, so the plan's own U6 gate ('jtml.oracle_multistage green on the GPU machine') cannot pass as specced — the implementer burns run 1 discovering the kill, and the tempting 'fix' (shrinking the budget shape to fit 600 s) would gut the caps gate that is the oracle's entire point. The TIMEOUT must be re-derived from a measured run (or the femur/tibia runs split into separate targets), and the run is nightly-grade, not ad-hoc.",
      "finding_type": "error",
      "autofix_class": "gated_auto",
      "confidence": 75,
      "evidence": [
        "plan: 'Create: test/oracle/multistage_oracle_test.cpp (new jtml.oracle_multistage target, oracle/gpu label, TIMEOUT 600)'",
        "plan: 'Integration: frame-to-frame chaining across the 3 Kneel_1 frames' and 'with the production shape'",
        "repo test/oracle/oracle_test.cpp:244: 'const unsigned int kBudget = 3000;  // few minutes; production uses 20k/25k/30k'",
        "repo test/golden/baseline.json budget_note: 'The Tier-2 oracle run was reduced to a flat 3000 for run time and is NOT the production budget shape.'",
        "repo test/CMakeLists.txt: 'add_test(NAME jtml.oracle COMMAND jtml_test_oracle)' (existing oracle target has no TIMEOUT; the plan's new targets get TIMEOUT 600)"
      ]
    },
    {
      "title": "U9's 'qml parity unchanged (the QML bridge never executes runs)' is factually wrong — OptimizerBridge::run() does execute runs — and U9's verification omits the existing qml_parity_check instrument that exercises the bridge path",
      "severity": "P3",
      "section": "U9 (Cut B) — Verification / System-Wide Impact 'API surface parity'",
      "why_it_matters": "The QML bridge does execute runs: OptimizerBridge::run() calls controller_->start(req) (OptimizerBridge.cpp:200), and the repo's own parity gate (test/oracle/qml_parity_check.cpp) is documented as driving 'the app's OWN optimizer run path (OptimizerBridge ...) ... calls OptimizerBridge::run()' with the degenerate 3000/0/0 shape. After Cut B the bridge path therefore runs the script-driven loop, so 'qml parity' needs verification through that instrument — which exists in the tree and is not listed anywhere in U9's verification. The correct derivation: the bridge executes through the controller; parity follows from the bit-identity diff + running qml_parity_check post-Cut-B (it would catch exactly the stage-transcription class, e.g., budget-0 seed evals at branch/leaf), not from the bridge being inert.",
      "finding_type": "error",
      "autofix_class": "gated_auto",
      "confidence": 75,
      "evidence": [
        "plan: 'qml parity unchanged (the QML bridge never executes runs)'",
        "repo src/app/experimental/OptimizerBridge.cpp:200: 'controller_->start(req);' inside 'void OptimizerBridge::run()'",
        "repo test/oracle/qml_parity_check.cpp: 'the app's OWN optimizer run path (OptimizerBridge — the real wiring the QML app drives: OptimizeIntent Controller gate -> OptimizerManager + QThread -> Initialize ...)' and 'calls OptimizerBridge::run()'",
        "plan (U9 Verification): 'headless + oracle green; golden assertions verbatim; bit-identity diff empty; jtml-production is the running default with zero production-visible change (qml parity unchanged)' — no qml_parity_check listed"
      ]
    },
    {
      "title": "Sym-Trap runs write Results.csv / Results.xyz / Results2D.xy into the test working directory and sleep ~5 s — the multistage oracle's tibia run triggers these side effects unmentioned in the plan",
      "severity": "P3",
      "section": "U6 — tibia-after-femur drive sequence",
      "why_it_matters": "Every tibia SymTrap run through the real manager calls CalculateSymTrap, which opens Results.csv, Results.xyz and Results2D.xy in the process CWD and sleeps 60 × (5000/60) ms ≈ 5 s. The multistage oracle (TIMEOUT 600) will produce three stray files in whatever directory ctest runs from, and adds ~5 s of wall clock plus the 60 extra full GPU cost evals (~1–3 min at the documented rate) on top of the search. The implementer debugging the tibia pass will be surprised by these files and the sleep; running from a build-dir CWD (ctest default) contains them, but the plan should document the side effects so they are not mistaken for test output or oracle artifacts.",
      "finding_type": "omission",
      "autofix_class": "manual",
      "confidence": 50,
      "evidence": [
        "repo src/coordinator/optimizer_manager.cpp:1341-1359: 'std::ofstream myfile; myfile.open(\"Results.csv\"); ... myfile2.open(\"Results.xyz\"); ... myfile3.open(\"Results2D.xy\");'",
        "repo src/coordinator/optimizer_manager.cpp:1330: 'std::this_thread::sleep_for(std::chrono::milliseconds(5000 / iter_val));'",
        "plan: 'Run 2 tibia (SymTrap, femur row = Run-1 recovery)'"
      ]
    }
  ],
  "residual_risks": [
    "Multistage oracle wall-clock runtime (~30–75+ min for the production shape, see P2 finding on TIMEOUT 600): the target is nightly-grade and needs a measured TIMEOUT and CI scheduling decision after U6 run 1.",
    "Dilation reconciliation (baseline.json records 6/3/1; the production code's overrides yield 6/4/1 — verified: DIRECT_DILATION default 6, branch→4, leaf→1 in mainscreen.cpp:4898-4900 / SettingsBridge) stays deliberately unresolved until U5's probe data; U4's re-baseline edits baseline.json but does not list updating the stale dilation_px block.",
    "Sym-trap early-return condition (femur recovery with exactly zero rotation) silently degrades the tibia run to relay count 0 and no files; the relay pin catches it, but the cost is a wasted ~30k-eval GPU run.",
    "The plan's own acknowledged line-number drift is real but small: CalculateSymTrap :1303 (early return :1304–1309), EvaluateCostFunctionAtPoint :1378, RunDirectStage :1234 — all within a few lines of the cited landmarks; re-grep-before-edit doctrine in the risk table is adequate.",
    "Re-baseline magnitude is GPU-machine-dependent; the plan correctly pre-registers the direction (z toward fem.jts, 6.31 mm gap shrinks) and records rather than asserts magnitude."
  ],
  "deferred_questions": []
}
```