I've read the full plan and verified its load-bearing claims against the repo (Optimize() stage loop, sym_trap path, CalculateSymTrap, the stage-guard tautology, the distance-map kernel launch geometry, baseline.json, the driver seam, StageLabel, CMake test conventions). Key verifications: the distance-map `blockIdx.y + gridDim.x` bug is real and live (2-D grid); the stage-guard tautology is real but `stage_` is never read anywhere; `stage_` is private with no accessor; CalculateSymTrap emits 61 relays (60 loop + 1 restore); the SymTrap path runs trunk + branches unguarded; baseline.json records branch dilation 3 while the plan's graph asserts 4; the existing flat-3000 oracle uses TIMEOUT 3600 while the plan gives the multi-stage oracle 600.

```json
{
  "reviewer": "ce-adversarial-document-reviewer",
  "findings": [
    {
      "title": "Sym-Trap costCalls pin (20000) contradicts the current engine, which runs trunk + branch stages in SymTrap mode (30000 with the production shape)",
      "severity": "P1",
      "section": "U6 (Test scenarios) / U9 (Test scenarios)",
      "why_it_matters": "U6 is the pre-Cut-B pinning instrument: its sym-trap pins are supposed to lock the CURRENT engine's behavior so U9 can prove bit-identity. The 20000 pin is wrong against today's engine — the trunk section of Optimize() runs unconditionally and the branch section is not guarded by sym_trap_call, so with the production shape (2 branches) a SymTrap run executes 20000 + 2×5000 = 30000 cost calls (only the leaf SEARCH is skipped; the leaf init + CalculateSymTrap still run). The implementer therefore gets a red gate for the wrong reason and must silently choose between disabling branches for the tibia run (contradicting 'the same shape the jtml-production graph will express in U9' and U7's 'reproduces the current loop's shape verbatim for every directive') or correcting the pin to 30000 — an undocumented scope decision either way, and the plan's own requirements contradict each other on this point.",
      "finding_type": "error",
      "autofix_class": "gated_auto",
      "confidence": 75,
      "evidence": [
        "U6: 'Edge case: sym_trap directive — costCalls lands at 20000 (not 35000); relay count == 60.'",
        "U9: 'Edge case: Sym_Trap directive → repeat=0 leaf → no search, costCalls at 20000, relay == 60 (pins from U6 stay green through the relocation).'",
        "U6: 'Drive OptimizerRunController directly ... with the production shape — the same shape the jtml-production graph will express in U9.'",
        "src/coordinator/optimizer_manager.cpp (verified): trunk RunDirectStage at :1029 is unguarded by sym_trap_call; branch loop 'for (int branch_index = 0; branch_index < optimizer_settings_.enable_branch_ * optimizer_settings_.number_branches; branch_index++)' with 'budget_ += optimizer_settings_.branch_budget;' (:1068–1097) is also unguarded; only the leaf search is gated on '!sym_trap_call' (:1139–1140)"
      ]
    },
    {
      "title": "orientationSymTrapUpdated relay-count pin (assert == 60) is wrong: the engine emits 61 (60 sweep poses + one unconditional restore emit)",
      "severity": "P2",
      "section": "U6 (Approach — three sym-trap pins)",
      "why_it_matters": "The pin is meant to catch the zero-rotation early return, but asserting exactly 60 fails against a healthy engine: CalculateSymTrap emits onUpdateOrientationSymTrap once per loop iteration (iter_val=60) and then unconditionally once more to 'set model back to initial pose' before writing Results.csv — the controller relays both, so the count is 61. The implementer burns a GPU debug cycle discovering the off-by-one and must guess whether the restore emit was meant to be excluded, while the plan presents the pin as a precise trap-catcher.",
      "finding_type": "error",
      "autofix_class": "gated_auto",
      "confidence": 75,
      "evidence": [
        "U6: 'orientationSymTrapUpdated relay count ≥ 1 (assert == 60 to catch the silent early return at optimizer_manager.cpp:1294–1300)'",
        "src/coordinator/optimizer_manager.cpp (verified): 'for (int i = 0; i < iter_val; i++) { emit onUpdateOrientationSymTrap(...' (:1325–1326) plus, after the loop, '// set model back to intial pose\\n emit onUpdateOrientationSymTrap(\\n pose_6D.x, pose_6D.y, ...' (:1346–1347)"
      ]
    },
    {
      "title": "jtml.oracle_multistage TIMEOUT 600 cannot fit the production shape: 3 frames × 35k evals + tibia sym-trap run is ~40–120 min (the plan's own grounding: 10–30 min/frame; the flat-3000 oracle already uses TIMEOUT 3600)",
      "severity": "P2",
      "section": "U6 (Files) / U5 (Files)",
      "why_it_matters": "The multi-stage oracle is the load-bearing pinning instrument for Cut B, and with TIMEOUT 600 the ctest target is killed on every execution: the production shape is 35k evals/frame across all three Kneel_1 frames plus a tibia SymTrap run (another ~30k evals + 60 evals + ~5 s of sleeps), and the plan's own normative grounding puts the 35k shape at 10–30 min/frame at 20–60 evals/s on the RTX 3090 — 40–120+ minutes total. The repo's existing convention confirms the scale: the flat-3000 oracle (a few minutes) is registered with TIMEOUT 3600. The implementer will hit a guaranteed timeout on the instrument that everything else is pinned against.",
      "finding_type": "error",
      "autofix_class": "gated_auto",
      "confidence": 75,
      "evidence": [
        "U6: 'Create: test/oracle/multistage_oracle_test.cpp (new jtml.oracle_multistage target, oracle/gpu label, TIMEOUT 600)'",
        "U6: 'Drive OptimizerRunController directly ... with the production shape' and 'Integration: frame-to-frame chaining across the 3 Kneel_1 frames.'",
        "test/CMakeLists.txt:1277 (verified): 'set_tests_properties(jtml.oracle PROPERTIES LABELS \"oracle\" TIMEOUT 3600' (the flat-3000 oracle)",
        ".panoptes/optimizer-deep-dive/synthesis.org:199 (plan's normative grounding): 'full 35k production shape ~10–30 min/frame' at '~20–60 evals/s'"
      ]
    },
    {
      "title": "v1 graph's branch dilation 4 is asserted as 'verified against ... test/golden/baseline.json', but baseline.json records branch dilation 3 — and U1's 'pins match baseline.json's dilation values' has no decision rule for the conflict",
      "severity": "P2",
      "section": "Overview (Problem Frame) / Key Technical Decisions / U1 (Test scenarios) / U5",
      "why_it_matters": "The plan's headline graph claims the production shape (branch dil 4) is verified against baseline.json, but baseline.json records 'dilation_px: trunk 6, branch 3, leaf 1' — the plan's own canonical source contradicts its graph, and the code-side session default is 4 (SettingsBridge kBranchDilationDefault). U1's integration pin ('registry/constants pins match baseline.json's recorded budget shape and dilation values') is therefore unsatisfiable as written (3 vs 4), the multi-stage oracle is given no stated branch dilation for the production shape, and the instrument that would arbitrate (U5's probe) runs after the pins that depend on the answer and records rather than gates. The implementer must pick a side with no authority; a wrong pick silently shifts the cost/IoU surface the whole PoC is judged against.",
      "finding_type": "error",
      "autofix_class": "gated_auto",
      "confidence": 75,
      "evidence": [
        "Overview: 'The current shape, expressed as the v1 graph (verified against the code and test/golden/baseline.json):' and '-> [Branch x2: classic DIRECT, budget 5000 each, range (15,15,25,25,25,25), dil 4]'",
        "test/golden/baseline.json (verified): '\"dilation_px\": { \"trunk\": 6, \"branch\": 3, \"leaf\": 1 }'",
        "U1: 'Integration: registry/constants pins match baseline.json's recorded budget shape and dilation values.'",
        "U5: 'dilations {6,3,1,4} (the probe arbitrates 6/3/1-vs-6/4/1 by data)' — the arbitration runs after U1/U6 pins are set"
      ]
    },
    {
      "title": "'stageText reports Trunk → Branch → Leaf' misstates the observation channel: the leaf stage renders as 'Extra Z-Translation' and branches as numbered 'Branch N' labels",
      "severity": "P3",
      "section": "U6 (Test scenarios) / U9 (Test scenarios)",
      "why_it_matters": "The oracle asserts stageText from the counter-derived channel, and StageLabel in optimizer_run_controller_core.cpp returns 'Trunk', 'Branch 1/2', 'Extra Z-Translation' (the leaf/z stage), 'Finished' — never 'Leaf'. An assertion transcribed from the plan's sequence fails on the first otherwise-green run, forcing the implementer to rediscover the channel's real labels and second-guess whether the relocation is allowed to change them.",
      "finding_type": "error",
      "autofix_class": "safe_auto",
      "confidence": 75,
      "evidence": [
        "U6: 'Happy path: Covers AE1. — femur run costCalls lands on 20/25/30/35k; per-frame IoU ≥ 0.85 vs the re-baselined values; stageText reports Trunk → Branch → Leaf.'",
        "U9: 'stageText sequence Trunk → Branch → Leaf.'",
        "src/coordinator/optimizer_run_controller_core.cpp (verified): StageLabel returns 'Branch ' + number and, for the leaf band, 'return \"Extra Z-Translation\";'"
      ]
    },
    {
      "title": "U1's 'stage-guard accessor pin' presupposes an accessor that does not exist — adding one is a second edit inside the wizard-owned 'DO NOT EDIT' region, contradicting the stated one-exception invariant",
      "severity": "P3",
      "section": "U1 (Approach) / U2 (Test scenarios) / System-Wide Impact",
      "why_it_matters": "stage_ is a private member (CostFunctionManager.h:147) with no getter anywhere, and grep shows it is never read — so the U2 tautology fix has zero functional effect and the U1 pin is the only way to make it observable. Implementing the pin as described requires a public accessor in the wizard-owned header, a second exception the plan's invariant ('the wizard-owned regions of CostFunctionManager.* (one in-place one-line exception, U2)') forbids; the implementer must either silently deviate, drop the pin, or invent a friend/test seam. The plan should state which is intended.",
      "finding_type": "omission",
      "autofix_class": "manual",
      "confidence": 75,
      "evidence": [
        "U1: 'the stage-guard accessor pin' (in the per-metric pin list)",
        "U2: 'Happy path: after the stage-guard fix, the accessor pin reports the stage set by the caller (previously always Trunk).'",
        "System-Wide Impact: 'the wizard-owned regions of CostFunctionManager.* (one in-place one-line exception, U2)'",
        "include/compute/CostFunctionManager.h:147 (verified): 'Stage stage_;' in the private section under a '(DO NOT EDIT)' banner; no getStage() exists anywhere"
      ]
    }
  ],
  "residual_risks": [
    "Script-loop nesting is ambiguous: the U9 diagram builds stage_script_ once, but the per-frame budget_/cost_function_calls_ reset and seed chaining must stay INSIDE the per-frame loop; if the stage iteration is hoisted outside the frame loop, the caps gate catches it only after a ~1h GPU run.",
    "U8's Options cut is the largest bit-identity surface: today's DirectOptimizer has no explicit selection/epsilon/split/tie knobs, so 'each mapped line-by-line onto today's code' is aspirational; bit-identity rests entirely on guarded-divergence discipline plus the golden/PBT/oracle parity evidence, and there is a fourth 4-arg DirectOptimizer call site (optimize_coordinator.cpp:36, OptimizeWorker — currently unused) the call-site sweep must not miss.",
    "jtml.z_profile TIMEOUT 600 is also tight-too-small for the full sweep set (dilations {6,3,1,4} + lineage S1–S4 + semantics axis × 3 frames × 31 renders + repeats), depending on the achieved evals/s.",
    "CalculateSymTrap writes Results.csv and Results2D.xy into the process CWD; oracle runs at repo root will pollute the working tree on every GPU run (the plan does not mention this side effect).",
    "The 6.31 mm frame-1 gap used as the re-baseline direction anchor comes from a Qt5-era capture (captured_baseline in baseline.json); the plan hedges correctly ('recorded, not asserted'), but the delta's interpretability as pure index-fix effect is weaker than the 'single-variable' framing suggests.",
    "The stage-guard fix (U2) changes no behavior today because stage_ is dead state; its only purpose is future-proofing for cfm_index, so the U1 pin is a pure-hygiene observable — cheap, but easy to over-invest in."
  ],
  "deferred_questions": [
    "Which branch dilation is authoritative for the U1/U6 pins: 3 (baseline.json) or 4 (widgets/SettingsBridge first-run value and the plan's own graph)? The plan needs an explicit decision rule (e.g., pin the engine value at U1, reconcile baseline.json at the U4 re-baseline event).",
    "Is the tibia SymTrap run intended to execute branch stages (engine behavior: costCalls = 30000) or to be trunk-only (the plan's 20000 pin)? The answer changes both the U6 pin and the U9 SymTrap StageScript, and 'verbatim transcription' (U7) conflicts with 20000.",
    "Should the relay-count assertion count the restore emit (61 total) or only the 60 sweep emissions?",
    "Is adding a public stage_ accessor to the wizard-owned CostFunctionManager.h approved as a second documented exception for U1's pin?"
  ]
}
```