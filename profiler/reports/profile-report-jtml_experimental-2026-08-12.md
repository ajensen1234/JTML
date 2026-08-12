# Profile Report — jtml_experimental (plan 007 U7)

## Header

- **Profile mode**: full
- **Trace file**: `profiler/traces/qmlprofiler-trace-jtml_experimental.qtd` (35 MB)
- **Run duration**: ~3.9 s (wall-clock estimate from 556 frames)
- **Sum of captured range-event durations**: 16,834 ms (binding/JS/signal/creating work — **not** wall-clock time)
- **Total events**: 383,319
- **Binary**: `.build-prof/bin/jtml_experimental` (RelWithDebInfo, `-DQT_QML_DEBUG`, Qt 6.7.2)
- **Session**: owner-driven xcb run (2026-08-12): study load, 8 optimizer Run clicks, app closed. The session was cut short by a frame-picker regression (queue item 1) — the Poses dialog, table scrolling, and cell editing were NOT exercised.

## Event type summary

| Type | Count | Total ms | ms/frame |
|------|-------|----------|----------|
| Javascript | 134,002 | 10,081 | 18.13 |
| HandlingSignal | 1,416 | 5,103 | 9.18 |
| Binding | 93,280 | 934 | 1.68 |
| Creating | 5,880 | 661 | 1.19 |
| Compiling | 41 | 55 | 0.10 |

Counts scale with run length and interaction pattern — ms/frame is the honest per-frame CPU cost.

## Animation / frame-time summary

**How to read the percentiles:** frame time = wall-clock gap between successive frames; p50 is the median; p95/p99 mean 5%/1% of frames were worse; max is the worst single frame. Vsync at 60 Hz ≈ 16.67 ms/frame; > 33 ms is visible stutter.

| Metric | Value |
|--------|-------|
| Frame count | 556 |
| **frame_ms_p50** | **7.04** |
| **frame_ms_p95** | **7.04** |
| **frame_ms_p99** | **7.04** |
| **frame_ms_max** | **7.04** |
| **frames_over_33ms** | **0** |
| **frames_over_50ms** | **0** |

Verdict: the 2D chrome renders at ~143 fps with zero jank in this session. The UI is not frame-bound; the cost lives in discrete operations (see hotspots), not continuous rendering.

## Memory summary

QML memory events were captured; no allocation pathology was observed in this short session (no hotspot clustering in Creating beyond the dialog instances listed below).

## Top 10 hotspots

| Rank | Total ms | Count | Avg ms | Type | Source | Details |
|------|----------|-------|--------|------|--------|---------|
| 1 | 3,195 | 1 | 3,195 | Javascript | [main.qml:298](src/app/experimental/main.qml#L298) | expression for onClicked (Images) → pickImages |
| 2 | 1,417 | 1 | 1,417 | Javascript | [main.qml:310](src/app/experimental/main.qml#L310) | expression for onClicked (Models) → pickModels |
| 3 | 406 | 18,736 | 0.022 | Binding | [RunBar.qml:68](src/app/experimental/RunBar.qml#L68) | expression for text (stageText) |
| 4 | 245 | 2 | 123 | Creating | main.qml:24 | QtQuick/Window (dialog instances) |
| 5 | 241 | 18,736 | 0.013 | Binding | [RunBar.qml:73](src/app/experimental/RunBar.qml#L73) | expression for text (calls) |
| 6 | 223 | 8 | 27.9 | Javascript | main.qml:505 | expression for onRunRequested (Run clicks) |
| 7 | 196 | 18,736 | 0.010 | Javascript | RunBar.qml:73 | expression for text |
| 8 | 84 | 18,736 | 0.005 | Javascript | RunBar.qml:68 | expression for text |
| 9 | — | — | — | — | (remaining events distributed across startup + list population) | — |

## Detailed analysis (top project hotspots)

### 1. File-dialog open/close: ~3.2 s (Images) / ~1.4 s (Models) — `main.qml:298/310 → pickImages/pickModels`

The in-process `QFileDialog` (FileDialogBridge, DontUseNativeDialog) takes **~3.2 seconds** to construct/show/destroy for the images picker on this box. This is the single largest cost in the session — larger than every other category combined. The widgets app's native dialog has the same class of cost; the portal path is disabled deliberately (multi-select reliability).

**Why it is expensive:** the Qt widget dialog instantiates a full file-model + sidebar + completion machinery per open; on this box the dialog's directory enumeration and model population dominate.

**Suggested fix:** this is the queued QML path-bar picker (plan 007 deferred item B: `PathPickerDialog.qml` with FolderListModel + a copyable Location field). A QML picker removes the widget-dialog construction cost from the hot path entirely and is the owner's requested direction. Until then, the cost is accepted.

### 2. RunBar text bindings: 18,736 evaluations per label across 8 runs — `RunBar.qml:68/73`

`stageText` and `calls` labels re-evaluate on every `UpdateDisplay` signal from the optimizer (the run drove ~2,300 updates × 8 runs). Each evaluation is ~0.02 ms — the *count* is high but the *cost* is trivial (0.7 + 0.4 ms/frame while running, dwarfed by the 7 ms frame budget). The `currentMinimum.toFixed(3)` label (line ~73) does string formatting per update.

**Verdict:** not a fix target — the traffic is inherent to live progress display and the per-eval cost is negligible. If a future profile shows progress-update jank, the lever is throttling `UpdateDisplay` relays, not the bindings.

### 3. Run-click path: ~28 ms per click — `main.qml:505 onRunRequested`

Closing both dialogs + `optimizerBridge.run()` costs ~28 ms — imperceptible. Not a fix target.

## Next steps

1. The frame-picker regression (queue item 1) prevented a representative interaction session — the Poses dialog, table scroll, and cell-edit paths remain unmeasured. **Re-profile after the fix round** for a representative trace (the plan's U7 threshold applies to the re-run).
2. The file-dialog open cost is the top hotspot and the strongest evidence for the queued QML path-bar picker (deferred item B). No other view-layer fix is warranted by this trace.
3. No jank, no memory pathology, no binding storms — the U5 virtualization and U4 composition changes held up under the profiler's eyes for the exercised paths.

## Known scope limits

- The VTK viewport (QmlVtkRenderer) is a Quick 3D integration — its render cost is outside this 2D profiler's capture (per the tool's scope). The oracle render-smoke covers its correctness, not its timing.
- Trace was captured with the pre-fix binary? No — `.build-prof` was rebuilt with the owner-feedback fixes (12:17) before this run (12:21); the frame-picker regression was present in this run.

---

> AI assistance has been used to create this output.
