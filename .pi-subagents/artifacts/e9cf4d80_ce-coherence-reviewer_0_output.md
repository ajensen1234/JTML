I now have enough verified evidence. Let me compose the review.

**Analysis summary:**

The plan honors the R1–R16/AE1–AE5 contract — no origin requirement or acceptance example is dropped, all U-ID references resolve, and `lifecycle-seam-first` ordering (U4 before U5) matches the origin. The D3 pre-flight gate consistently blocks U2 and U8 with a consistent fallback. The one real plan-internal contradiction is the MVVM gate (U7) vs. its declared dependency list.```json
{
  "reviewer": "ce-coherence-reviewer",
  "findings": [
    {
      "title": "MVVM gate requires oracle; U7 deps omit it",
      "severity": "P2",
      "section": "Implementation Units / U7 (MVVM decomposition)",
      "why_it_matters": "The plan's stated gate and the unit's formal dependency list disagree. The Overview, Key Decisions, and the U7 header all say MVVM may only proceed after 'phases 1–3 are green,' and phase 1 includes the golden-oracle foundation (U2), which is explicitly blocked until the user captures the Qt5 GPU baseline (D3). But U7's Dependencies list is only 'U4, U6 (coordinator + DIRECT in place)' — it omits U2. An implementer following the dependency list can start (and ship) the MVVM decomposition before the oracle baseline exists, even when D3 is unobtainable — directly contradicting R8 ('committed to only after phases 1–3 land') and the plan's own Headless/oracle-gated narrative, which otherwise defers work when the baseline is missing.",
      "finding_type": "omission",
      "autofix_class": "gated_auto",
      "confidence": 75,
      "evidence": [
        "**MainScreen MVVM decomposition (view vs state vs services)** *(sequenced after phases 1–3 are green)*",
        "**Dependencies:** U4, U6 (coordinator + DIRECT in place).",
        "(4) MVVM decomposition of `MainScreen` (only after the first three are green)",
        "**MVVM is sequenced, not committed upfront (R8):** decompose `MainScreen` only after phases 1–3 are green and re-validated against the human outcome.",
        "**Notes (pre-flight gate):** This unit cannot truly pass until the user captures the Qt5 GPU baseline (D3)."
      ],
      "suggested_fix": "Add U2 to U7's Dependencies (and to the gate description) so the dependency graph matches the stated 'after phases 1–3 are green' gate, or explicitly document that MVVM is intentionally decoupled from the oracle baseline (in which case amend the Overview/Key-Decision gate text to say MVVM is gated only on coordinator+DIRECT, not the baseline capture)."
    },
    {
      "title": "R16 MainScreen API revalidation unanchored in U8",
      "severity": "P3",
      "section": "Implementation Units / U8 (Qt5→Qt6 migration)",
      "why_it_matters": "R16 explicitly requires revalidating Qt5-API calls 'across MainScreen,' and U8's Goal repeats 'API revalidation across MainScreen,' but the unit's Files list enumerates only pixi.toml, vtk_installer.sh, CMakeLists.txt, src/gui/main.cpp, and src/Study2Grid/main.cpp — no MainScreen source file is listed as a modification target. The revalidation touchpoint is only implied (by the bounded manual GUI smoke step and the goal text), so an implementer has no concrete MainScreen file to audit and could treat the Qt6 port as complete without the required MainScreen API pass.",
      "finding_type": "omission",
      "autofix_class": "gated_auto",
      "confidence": 50,
      "evidence": [
        "- R16. Qt5→Qt6: pixi pin, VTK rebuild, `find_package(Qt6)`, API revalidation, gated by pre-migration oracle + bounded manual GUI smoke.",
        "**Goal:** Move the whole build to Qt6: pixi pin, VTK rebuild against Qt6, `find_package(Qt6)`, API revalidation across `MainScreen`, plus the `QSurfaceFormat::setDefaultFormat` fix and a bounded manual GUI smoke step.",
        "**Files:** Modify: `pixi.toml` (`qt = \"5.*\"` → `qt >= 6` ...), `vtk_installer.sh` (...), `CMakeLists.txt` (`find_package(Qt6)`, `Qt5::`→`Qt6::`), `src/gui/main.cpp` (...), `src/Study2Grid/main.cpp` (...)"
      ],
      "suggested_fix": "Add `src/gui/mainscreen.cpp` / `include/gui/mainscreen.h` to U8's Modify list as the explicit target for the R16 Qt5-API revalidation pass, so the requirement has a concrete home rather than being folded into the manual smoke step."
    }
  ],
  "residual_risks": [
    "MVVM (U7) could be started and shipped before the golden-oracle baseline (U2) is captured, because U7's dependency list omits U2 even though the narrative gates MVVM on 'phases 1–3 green' — an implementer could reasonably conclude either way.",
    "Exact Tier-2 numeric tolerances (translations, rotations, silhouette pixel threshold) are deferred to baseline capture (U2), so any work performed before U2 completes has undefined numeric acceptance thresholds for the GPU oracle.",
    "The high-level phase narrative (Overview) does not surface U3 (data-structure decoupling) as a distinct step, so its placement between the oracle foundation and the lifecycle seam is only inferable from the unit list, not the phase description."
  ],
  "deferred_questions": [
    "Whether Qt6 migration (U8) should precede heavy MVVM decomposition (U7): the Key Decision defers this to the Phase-3 completion gate ('re-evaluate at the Phase-3 completion gate') while U8 marks U7 as 'ideally' a dependency, leaving the final ordering unresolved until that gate."
  ]
}
```