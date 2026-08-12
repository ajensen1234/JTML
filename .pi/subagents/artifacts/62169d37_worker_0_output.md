Everything validated — build green, 47/47 headless, no unaccepted qmllint warnings, scope clean. U3 complete.

## U3 Summary

**Work items → implementations:**

1. **D2 PoseCell commit contract (C1/C2):** capture `(commitFrame, commitModel, commitAxis)` at focus-in; commit against captured values on `editingFinished`; failed commits re-arm the display via `Qt.binding(() => storedValue.toFixed(3))` — the binding survives, no one-shot assignment, recycle-safe.
2. **D3 refresh owner (I1/I2):** hub wiring — `optimizerBridge.runStateChanged` → refresh on Completed/Error (lambda), `studyBridge.viewerPoseApplied` → `PoseBridge::refreshTable()`. Code comment records the confirmed stale-on-reopen premise (QQC2 Dialog keeps contentItem; U1 review D-05).
3. **D4 seed invalidation (I3):** hub wiring — `viewerPoseApplied` + `poseBridge.poseTableChanged` (fires on edits/copy/loads) → `OptimizerBridge::clearSeedPose()`. A save does **not** clear (persistence ≠ pose write). **Deviation:** clear is unconditional, not "on the seeded frame+model" — the seed's frame/model is captured inside the controller core with no readable accessor, and touching the coordinator seam is forbidden; unconditional clear is the minimal correct choice (any manual write overrides the estimate anyway).
4. **D5 run-lock (I4 + D-07):** per-component `runLocked`; Black-sil., Fem/Tib, **and the Camera/Model toggles** (D-07 escapees) joined; viewport `enabled: false` + a visible "Running — interaction locked" dim/pill overlay (new `Theme.overlayDim` token).
5. **D6 Estimate (I5):** `mlBridge.hasSegmentModel` in the binding + a hint label mirroring the bridge's guard message.
6. **D-08 ButtonGroup:** all three groups now use re-asserting `Binding` objects (bridge value = single source; group keeps exclusivity). Applied to all three in U3 because `MlBridge.cpp:555` writes `backgroundMode` programmatically after a segment — the killed-binding bug is currently observable there (deviation from "U4 applies", documented in code).
7. **D7 keyboard (I6):** frame list `onCurrentIndexChanged → setCurrentFrame` with a `suppressFrameSync` guard around the existing `Qt.callLater` deferral (transient -1/0 never reaches the bridge); model rows toggle on Space/Enter; both lists get `activeFocusOnTab` + a visible focus indicator.
8. **Tests:** pose suite +3 cases (133 assertions), ml suite +1 integration case (91 assertions) — all headless, pinning D3 wiring via `modelReset` and D4 via the new additive `OptimizerBridge::hasSeedPose()` read.