---
title: "QML debugging with qmlprofiler traces: count the handler firings (the frame-picker case)"
date: 2026-08-12
category: ui-bugs
module: jtml_view
problem_type: ui_bug
component: frontend_stimulus
symptoms:
  - "frame list clicks did nothing: currentIndex never moved, the viewport background never switched"
  - "no error dialog, no crash — a silent TypeError per click (ListView.view null in a nested MouseArea)"
  - "the harness's sync pin passed because it skipped the mouse path as 'ListView-standard'"
tags: [qml, debugging, qmlprofiler, trace, listview]
---

# QML debugging with qmlprofiler traces: count the handler firings

The frame-picker regression (plan 007 owner-feedback round 2) was pinned
down in minutes using the **qmlprofiler trace of the failing session** —
no debugger, no re-run, no added logging. This entry is the method, so
future QML interactivity bugs start from the trace instead of guessing.

## The method: a trace is an event census

qmlprofiler records a **range event per JS execution**: every signal
handler, every binding evaluation, every function call, with
file:line + start time + duration. A trace of a failing session answers
the first debugging question — *did the code even run?* — for every
hypothesis at once:

1. **Capture during the failing session** (the normal profiling flow:
   `$CONDA_PREFIX/lib/qt6/bin/qmlprofiler -o trace.qtd -- <app>`, close
   the app to save). No instrumentation needed — the trace is the
   instrumentation.
2. **Count firings per handler.** The `.qtd` is XML: `<eventData>` maps
   `event index → (displayname "File.qml:line", type)`, `<profilerDataModel>`
   holds the `<range startTime=... eventIndex=...>` records. Count ranges
   per eventIndex, join to the displayname:
   - handler fired 13×, signal downstream fired 1× → the click path runs
     but the state change fails (this case: `onClicked` 13×,
     `onCurrentIndexChanged` 1× — currentIndex never moved)
   - handler fired 0× → the event never reached the control (disabled,
     swallowed, wrong geometry)
   - handler fired with huge durations → the handler itself is the cost
3. **Check the timeline gaps.** The ranges' `startTime` ordering shows
   whether events cluster around a load, a run, or a specific click — the
   single `onCurrentIndexChanged` here was the post-load deferral, long
   before the 13 clicks.
4. **Cross-check with the harness.** A Qt Quick Test that clicks the real
   delegate reproduces app behavior headlessly — and the failure's exact
   TypeError printed in the test output. The pin that catches the class
   stays as a permanent test.

## The case (2026-08-12, frame picker dead)

- Symptom: clicking frames did nothing (no highlight move, no viewport
  change). No error surfaced in the app.
- Trace census: `StudyPanel.qml:104` (delegate MouseArea `onClicked`)
  fired **13 times**; `StudyPanel.qml:68` (`onCurrentIndexChanged`)
  fired **once** — and that once was the dataset-load deferral, before
  the clicks began. Conclusion: clicks reached the handler, but
  `currentIndex` never changed.
- Harness reproduction: clicking `list.itemAtIndex(2)` printed the
  app's silent error: `StudyPanel.qml:104: TypeError: Value is null and
  could not be converted to an object`.
- Root cause: `ListView.view` (the attached property) is **null in
  nested delegate items** — it is only provided on the delegate root. The
  extraction's `ListView.view` conversion was fine at root level
  (`width: ListView.view.width`); the click rewiring
  (`onClicked: ListView.view.currentIndex = index`) was the first nested
  usage. Fix: use the list's id (`frameList.currentIndex = index`).
- Why it survived review + tests: qmllint does not flag it; the harness
  pin explicitly skipped the mouse path as "ListView-standard". **The
  lesson: pin the mouse path — `mouseClick(delegate, x, y)` on
  `itemAtIndex(row)` — for every delegate with a click contract.**

## When to reach for this

- "Click/scroll/key does nothing" regressions in QML (handler-firing
  census beats guessing)
- Signal-chain bugs (which links of the chain actually ran?)
- Interaction regressions after refactors that touch delegates, attached
  properties, or scoping
- Any session where a trace already exists — re-analyze before re-running

## Caveats

- The trace only saves on clean app exit (kill = no trace).
- The 2D profiler does not capture the VTK viewport (Quick 3D territory).
- Short sessions (<5 s) under-sample; the census is about counts and
  ordering, not durations, for interactivity bugs.
- MemoryAllocation events dominate raw counts; filter by event type.

## Related

- Conventions: `docs/solutions/conventions/jtml-qml-view-testing-2026-08-12.md`
  (harness + lint + profiling recipes; the `ListView.view` nested-scope
  gotcha)
- Profile: `profiler/reports/profile-report-jtml_experimental-2026-08-12.md`
- Prior debug trail: `docs/solutions/ui-bugs/jtml-qml-model-pose-sync-queued-functor-never-delivered-2026-08-11.md`
- Tool: the qt-qml-profiler skill (parse script + report format)
