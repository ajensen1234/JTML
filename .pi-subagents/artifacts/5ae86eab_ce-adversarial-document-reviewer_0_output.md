All context gathered. The document claims largely verify against the repo (line counts 5806/1722 are accurate; test/CMakeLists disabled; no Catch2 in pixi; `.github/workflows/cmake.yml` is indeed inert boilerplate triggering only on `actions-test`). My findings therefore target the epistemological weaknesses in the plan itself.

```json
{
  "reviewer": "ce-adversarial-document-reviewer",
  "findings": [
    {
      "title": "Golden oracle is not independent ground truth; anti-circularity overclaim",
      "severity": "P1",
      "section": "Problem Frame / R2 / Key Decisions (Golden-oracle-first)",
      "why_it_matters": "The document asserts that both prior failure modes — false confidence and circularity — are 'structurally prevented' by the golden oracle. But R1 mandates the baseline be 'established from the current, working Qt5 app,' i.e. captured FROM the code-under-test itself. A baseline derived from the code-under-test's own output tests stability, not correctness: any bug present in the current Qt5 app is locked in and will 'pass,' silently reintroducing the false-confidence mode the plan claims to eliminate. The oracle only genuinely prevents the narrower helper-re-derivation circularity; the doc accidentally overstates the guarantee, so an implementer may believe a green oracle disproves a numeric regression when it only proves behavior is unchanged. The doc should explicitly label the oracle as a behavior-preservation (golden-master) gate, and only treat genuinely independent sources (analytic cost functions, known-good projections) as correctness checks.",
      "finding_type": "omission",
      "autofix_class": "manual",
      "confidence": 75,
      "evidence": [
        "R2. No test may assert an expected value derived by re-deriving the code-under-test's own math. Extracted logic must be exercised against the golden oracle or another independent ground truth (analytic cost functions, real fixture round-trips, known-good projections).",
        "R1. Establish a ground-truth baseline from the current, working Qt5 app ... The baseline must come from validated behavior, not from re-implementing code.",
        "Both failure modes must be structurally prevented.",
        "Golden-oracle-first: capture/define the baseline from the known-good Qt5 app before any migration or refactor; this prevents both false confidence and circular tests."
      ]
    },
    {
      "title": "Oracle gate's numeric tolerance undefined and deferred despite R3/AE2 hinging on it",
      "severity": "P1",
      "section": "R3 / AE2 / Dependencies (D1) / Deferred-to-Planning",
      "why_it_matters": "The central regression gate is unusable as written: R3 and AE2 require comparison 'within documented tolerance,' yet the tolerance value is an open, deferred research item ('Whether the DRR/cost golden comparison uses a CPU render reference or the GPU path, and the numeric tolerance to document'). Worse, the authoritative source D1 designates golden_oracle.org, whose Optimizer Settings block is internally inconsistent — dilation is listed as 'branch is 6px dilated, branch is 3px dilated' with the trunk level missing. An implementer following 'golden-first' ordering hits the gate immediately without a reproducible reference to enforce, so sequencing cannot actually gate anything until a planner resolves tolerance. The doc must commit to defining tolerances and a CPU-vs-GPU comparison reference as a precondition of the oracle phase, not leave it deferred.",
      "finding_type": "omission",
      "autofix_class": "gated_auto",
      "confidence": 75,
      "evidence": [
        "R3. Compute/cost regression comparisons must be deterministic and use the documented tolerances where numeric noise requires it.",
        "AE2 ... yields a pose within documented tolerance of `fem.jts` (C) and projections matching the known-good Labels (B).",
        "[Affects R3][Needs research] Whether the DRR/cost golden comparison uses a CPU render reference or the GPU path, and the numeric tolerance to document.",
        "D1. `golden_oracle.org` documents the oracle ... and the optimizer settings; it is the authoritative reference for R1–R3."
      ],
      "suggested_fix": "Move 'define documented numeric tolerances and choose CPU-vs-GPU comparison reference' from Deferred up to a Resolve-Before-Planning / precondition item for the oracle phase, and have the planner validate/repair the optimizer-settings block in golden_oracle.org before treating it as authoritative."
    },
    {
      "title": "'No hangs/deadlock' acceptance lacks timeout mechanism to fail a hang",
      "severity": "P2",
      "section": "R5 / AE1 / R7",
      "why_it_matters": "The entire point of the lifecycle seam is to catch the 'hangs after optimizer finishes' class headlessly, yet the acceptance criteria state only that 'no test hangs or deadlocks.' A headless lifecycle test that does deadlock or spin will hang the CI job indefinitely instead of failing, converting the exact failure mode the plan targets into an uncaught, stuck gate that blocks every future change with no diagnostic. The document never specifies per-test timeouts or a ctest timeout policy, so an implementer following R5/AE1 as written has no mechanism by which a hang becomes a red test. Test-harness infra (R11-R14) should mandate timeouts so a stuck test fails loudly rather than silently hanging.",
      "finding_type": "omission",
      "autofix_class": "gated_auto",
      "confidence": 75,
      "evidence": [
        "AE1 ... when the coordinator runs an optimize cycle to completion (and again after a stop), it returns to the idle state and accepts another launch; no test hangs or deadlocks.",
        "R5. A lifecycle test drives launch → running → finished → ready-to-re-launch using an injected stub cost function (no GPU) and asserts ... catching the 'hangs after optimizer finishes' class.",
        "R7. ... a stop request returns to idle promptly."
      ],
      "suggested_fix": "Add an explicit requirement under the test-harness/infrastructure section that every headless lifecycle test (and the default ctest run) be bounded by a timeout, so a hang is converted into a failed/failing test rather than a hung CI."
    },
    {
      "title": "Hang diagnosis asserted without root-cause evidence; seam may miss real hang site",
      "severity": "P3",
      "section": "Key Decisions (Lifecycle-seam-first) / Problem Frame",
      "why_it_matters": "The ordering decision 'lifecycle-seam-first' rests on the asserted premise that 'the hangs live in orchestration/threading, not in the math.' No root-cause analysis or evidence is cited to support this diagnosis, yet the whole seam plan is built on it. If the actual hang/deadlock or the historical regression site lives instead in a widget slot, the event loop, or a VTK render binding, the coordinator+worker seam will not reproduce or cover it, and the 'fast headless pass/fail' promise silently misses the real failure class again — the same trap the abandoned prior agent fell into. The doc should either cite the evidence behind the hang-site claim or make Lifecycle-seam-first conditional on confirming the hang reproduces through the coordinator seam before committing to it as the priority.",
      "finding_type": "omission",
      "autofix_class": "manual",
      "confidence": 50,
      "evidence": [
        "Lifecycle-seam-first: seal the optimize-lifecycle (headless) before the pure DIRECT math, because the hangs live in orchestration/threading, not in the math.",
        "...it produced passing tests that ... never touching the thread/orchestration seams where the regressions actually lived"
      ]
    },
    {
      "title": "Qt6 migration 'caught headlessly' overstates gate; UI regressions not headless-catchable",
      "severity": "P3",
      "section": "R16 / Scope Boundaries",
      "why_it_matters": "R16 claims migration 'so migration regressions are caught headlessly rather than by running the GUI,' but a Qt6 port most plausibly breaks exactly the GUI/render/UI-binding things that are explicitly excluded from the default headless run ('GUI and GPU/real-VTK tests are not part of the default headless run'). A migration can pass every numerics and lifecycle test while silently breaking rendering, layout, or widget behavior, giving false confidence that the migration is safe. The success criteria/documented outcome probably still demands someone click the GUI for the Qt6 migration, so the plan should state that explicitly rather than imply the headless suite subsumes migration verification, or add a bounded explicit GUI smoke step for R16.",
      "finding_type": "omission",
      "autofix_class": "manual",
      "confidence": 50,
      "evidence": [
        "R16. Migrate Qt5 → Qt6 as part of this effort, gated by the golden oracle and the headless suite (both captured pre-migration), so migration regressions are caught headlessly rather than by running the GUI.",
        "GUI and GPU/real-VTK tests are not part of the default headless run."
      ]
    }
  ],
  "residual_risks": [
    "Golden-master lock-in: oracle captured from the code-under-test preserves current values, so a latent numerics bug in the Qt5 app will be certified as correct by the gate; only genuinely independent analytic fixtures detect it.",
    "Tolerance/GPU-vs-CPU determinism unresolved: if the baseline is captured on a GPU machine (D3) but the default headless run executes CPU code paths (R11), float divergence between the two may exceed any chosen tolerance, so the gate may be flaky or unusable until this is decided.",
    "Fixture 'usable as-is' (D2) is unverified; Kneel_1 and test/vtk/test_case existence is confirmed but their completeness for a deterministic CPU/GPU oracle capture (calibration consistency, silhouette extraction parity) is not established.",
    "Dilation settings in the authoritative oracle source are internally inconsistent (branch listed twice, trunk mobilisation missing), so numeric assertions derived from them may be wrong even where tolerance is defined."
  ],
  "deferred_questions": [
    "What exact numeric tolerance and comparison reference (CPU reference render vs GPU path) will R3/AE2 enforce, and does the tolerance diverge between GPU-captured baseline and headless CPU run?",
    "What per-test timeout policy/mechanism (ctest timeout) converts a headless lifecycle hang into a failed test rather than a hung CI job?",
    "What, if any, explicit (bounded) GUI smoke/visual verification step accompanies the Qt6 migration that cannot be covered by the headless suite?",
    "What is the evidence (root-cause analysis) that the historical hang/regression site is the optimize orchestration seam rather than a widget/event-loop/VTK binding?",
    "Which test framework (Catch2 vs QtTest) and Qt offscreen platform will be added to pixi for R14/R11, and does the chosen framework support headless QSignalSpy/QCoreApplication lifecycle tests?"
  ]
}
```