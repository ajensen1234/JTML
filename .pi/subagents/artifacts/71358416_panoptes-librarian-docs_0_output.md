Round 3 is appended and verified intact (file now ends at the Round 3 Sources section; rounds 1–2 untouched). All six concrete tasks were executed against the corpus primary texts with corpus file paths cited throughout, and the write is the deliverable.

## Task completion summary

**Task 1 (GLh formula vs thesis):** verified from the paper itself (`s/stripinisNewDIRECTGLhAlgorithm2021.txt`) — Eqs. 3–6 give raw surrogate f_min + d, selection-time value d̃ = d/√n (f̃_min ≡ 0 drops out), flat-case f̃ = 1, the 1e-6 recompute threshold (Algorithm 1 L18–22), Definition 1 feasibility detection, and the paper's own numbers (5/67 DIRECTLib, 100/100 Emmental). Round-2's formula confirmed with a two-level sharpening.

**Task 2 (gb numbers vs Gablonsky + 2022 citations):** Gablonsky 2001 (`j/j.m.gablonskyLocallyBiasedFormDIRECT2001.txt`) is the single-phase locally-biased DIRECT-l (one-POH-per-group), **not** the two-phase source — the phase-switch numbers stay with Gb-DISIMPL [28]; the DIRECTGO paper's prose (`s/stripinisDGONewDIRECTtype2021.txt` §2.2) is now a **second primary text** confirming canonical phase semantics; I_DTC_GL family version-pinned to DIRECTGO v1.1.0.

**Task 3 (Jones 1993 ε/convergence):** Definition 4.1's ε|f_min| filter is inert at f_min=0 (mathematically, not just critique); ε=0 is never tested in 1993 (smallest 1e-7) and the convergence proof (Eq. 10) holds at ε=0 — "ε=0 is pathological" is an empirical, budget-regime-dependent performance claim; SHU is the paper's own named exception; Gablonsky's ε = max(10⁻⁴|f_min|, 10⁻⁸) floor is a documented f_min≈0 mitigation.

**Task 4 (1-DTC-GL-gb vs DGO semantics):** GL = two-step Pareto with unique union S = G ∪ L (the DGO paper's version, "on average more effective"); gb = canonical; 1-DTC = least-split verbatim; and the repo config matches **no** studied variant exactly.

**Task 5 (JTA tradeoff):** the exact quote verified (L447–458); JTML's stage geometry is a ~2.5× downscale transcription of JTA's (branch 15/25/25 identical; budgets 50k/15k×3/50k); global convergence is already sacrificed by design → the battery needs a staged **basin-capture** metric; ε ≈ 10⁻⁷ is the lineage's deliberate choice; z excluded from JTA success.

**Task 6 (battery):** DGO paper documents v1.0 (119 problems, 81 box at n∈{2,5,10,15}), two-tier ε_pe, Mmax 2×10⁶, data profiles, and the 80-problem hidden-constraint protocol with its 13/80 caveat; **zero GKLS content in the DGO paper**; Emmental-GKLS documented in the GLh paper as a ready-made constrained battery.