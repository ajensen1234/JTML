I've read the full document, cross-referenced its internal references, and verified the `(A)/(B)/(C)/(D)` oracle lettering against `golden_oracle.org` (it is consistent — no finding there). Two internal-consistency issues surfaced.

```json
{
  "reviewer": "ce-coherence-reviewer",
  "findings": [
    {
      "title": "\"Approach A\" referenced but never defined",
      "severity": "P3",
      "section": "Outstanding Questions / Resolve Before Planning",
      "why_it_matters": "A reader reaching the 'Resolve Before Planning' section finds the decided direction named as 'Approach A', but no section of the document defines, labels, or otherwise introduces 'Approach A'. The reader must either guess that it refers to the 'Strangler extraction' key decision or wonder whether a prior approach-comparison artifact is being referenced. Since the two co-listed items ('golden-first', 'lifecycle-first') map one-to-one onto the labeled Key Decisions 'Golden-oracle-first' and 'Lifecycle-seam-first', the third item shadows 'Strangler extraction, not rewrite' — but the doc never says so, leaving jargon that a planner cannot resolve from the text alone.",
      "finding_type": "error",
      "autofix_class": "safe_auto",
      "confidence": 75,
      "evidence": [
        "None — the direction (Approach A, golden-first, lifecycle-first), the Qt6 migration, and the oracle (`golden_oracle.org`) are decided."
      ],
      "suggested_fix": "Replace 'Approach A' with the actual, in-document name of the approach, e.g. 'the direction (strangler extraction, golden-first, lifecycle-first)'. If 'Approach A' refers to a prior planning artifact not included in this document, define it inline instead of emitting a bare code."
    },
    {
      "title": "Forward ref 'seam/test-gate order below' has no target",
      "severity": "P2",
      "section": "Success Criteria / Handoff quality",
      "why_it_matters": "The Handoff-quality criterion asserts the doc '\u2026plus `golden_oracle.org` and the seam/test-gate order below' delivers 'an unambiguous list of which seams to cut, in what order, each with the test that must be green before the next cut.' No such per-seam, per-test-gate ordering exists anywhere in the document: the only related content is the coarse high-level phase list in Next Steps (foundation; lifecycle seam; DIRECT/cost seams; MVVM decomposition; Qt6 migration), which does not enumerate individual seams or the test that gates each. A planner reading the Success Criteria will look for a named, ordered seam-with-gate list and not find it, creating a mismatch between what the doc claims to hand off and what it actually contains.",
      "finding_type": "omission",
      "autofix_class": "gated_auto",
      "confidence": 75,
      "evidence": [
        "This doc, plus `golden_oracle.org` and the seam/test-gate order below, gives a planner an unambiguous list of which seams to cut, in what order, each with the test that must be green before the next cut.",
        "-> /ce-plan for structured implementation planning (may be split into phases: (1) oracle + test-runner + CI foundation; (2) lifecycle seam; (3) DIRECT/cost seams; (4) MVVM decomposition; (5) Qt6 migration gated by the oracle)."
      ],
      "suggested_fix": "Either (a) add the promised ordered seam/test-gate list to the document, or (b) reword the Handoff-quality bullet to point at what actually exists (the Next Steps phases) and drop the promise of a per-seam 'each with the test that must be green' breakdown. Choosing between these changes the document's content/scope, so it needs planner sign-off."
    }
  ],
  "residual_risks": [
    "R3 requires documented numeric tolerances for the oracle comparison, but neither this document nor golden_oracle.org states them; the doc explicitly defers this to 'Needs research'. A planner must still derive concrete tolerances before R3 can be implemented as written.",
    "golden_oracle.org is written as loose informal prose; the requirements doc treats it as the authoritative tolerance/settings source (D1, F2, AE2). Any discrepancy in how the planner interprets the numeric settings could silently shift the gate that R1-R3 depend on.",
    "R14 leaves test-framework choice (Catch2 vs QtTest) open to planning; none of the requirements or acceptance examples pin down the framework, so the headless-suite requirements (R4/R5/R7, AE1) are framework-agnostic and the actual wiring is unresolved until planning."
  ],
  "deferred_questions": [
    "Confirm whether 'Approach A' is shorthand for the 'Strangler extraction' decision or refers to a prior approach-comparison artifact outside this document.",
    "Confirm whether the per-seam 'test that must be green before the next cut' ordering is expected to be authored in this requirements doc or is a /ce-plan deliverable."
  ]
}
```