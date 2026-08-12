// Plan 007 U6 — view-flow pins at panel level with injected fakes:
//  - StudyPanel: frame-list click syncs the bridge; model rows toggle on
//    Space; the run lock disables both lists; a dataset swap never
//    writes a transient -1 to the bridge;
//  - MlStrip: the ML degradation enabled-binding matrix (Segment/
//    Estimate incl. estimate-requires-segment, D6) and the run lock on
//    the black-silhouette + implant-kind controls;
//  - RunBar: Run emits runRequested, Stop reaches the bridge, the
//    run-state enablement flips;
//  - PosesDialog: the dirty-close guard fires discardRequested on a
//    dirty close and stays silent for a confirmed (Run) close.
import QtQuick
import QtTest
import "qrc:/components"

Item {
    id: root
    width: 660
    height: 700

    FakeAppBridge { id: fakeApp }
    FakeStudyBridge { id: fakeStudy }
    FakeOptimizerBridge { id: fakeOpt }
    FakeMlBridge { id: fakeMl }
    FakePoseBridge { id: fakePose }

    // Fake native picker (the FileDialogBridge shape the toolbar calls:
    // getOpenFileNames(title, filter, startDir, purpose) -> path list).
    QtObject {
        id: fakePicker
        property var returnPaths: []
        function getOpenFileNames(title, filter, startDir, purpose) {
            return returnPaths
        }
    }
    // Fake renderer surface for the interaction-mode toggles.
    QtObject {
        id: fakeViewport
        property int interactionMode: 0
        property var modeLog: []
        function setInteractionMode(m) { modeLog = modeLog.concat([m]) }
    }

    Component {
        id: panelComp
        StudyPanel {}
    }
    Component {
        id: stripComp
        MlStrip {}
    }
    Component {
        id: barComp
        RunBar {}
    }
    Component {
        id: dialogComp
        PosesDialog {}
    }
    Component {
        id: toolbarComp
        Toolbar {}
    }

    SignalSpy {
        id: runRequestedSpy
        signalName: "runRequested"
    }
    SignalSpy {
        id: discardSpy
        signalName: "discardRequested"
    }
    SignalSpy {
        id: msgSpy
        signalName: "showMessageRequested"
    }
    SignalSpy {
        id: replaceSpy
        signalName: "replaceRequested"
    }

    TestCase {
        id: testCase
        name: "StudyFlows"
        when: windowShown

        function init() {
            fakeStudy.setCurrentFrameLog = []
            fakeStudy.toggleLog = []
            fakeStudy.loadImagesLog = []
            fakeStudy.loadModelsLog = []
            fakePicker.returnPaths = []
            fakeViewport.modeLog = []
            fakeViewport.interactionMode = 0
            fakeStudy.frameListModel.clear()
            fakeStudy.modelListModel.clear()
            for (let i = 0; i < 5; ++i)
                fakeStudy.frameListModel.append({ display: "Frame " + i })
            for (let i = 0; i < 3; ++i)
                fakeStudy.modelListModel.append({ display: "Model " + i })
            fakeStudy.frameCount = 5
            fakeStudy.currentFrame = 0
            fakeStudy.primaryModelIndex = -1
            fakeStudy.selectedModels = []
            fakeStudy.selectedModelCount = 0
            fakeStudy.hasDataset = false
            fakeApp.frameCount = 5
            fakeOpt.running = false
            fakeOpt.canRun = true
            fakeOpt.stopCalls = 0
            fakeOpt.runCalls = 0
            fakeMl.hasSegmentModel = false
            fakeMl.hasEstimateModel = false
            fakeMl.blackSilhouette = false
            fakePose.dirty = false
            runRequestedSpy.clear()
            discardSpy.clear()
        }

        function test_frameSelectionSyncsBridge() {
            const panel = createTemporaryObject(panelComp, root, {
                appBridge: fakeApp, studyBridge: fakeStudy,
                optimizerBridge: fakeOpt, width: 240, height: 400 })
            verify(!!panel, "Component exists")
            const list = findChild(panel, "studyFrameList")
            verify(!!list, "Object exists")
            wait(50)
            // D7 (I6): ANY currentIndex change (click, keyboard Up/Down,
            // programmatic) syncs the bridge — the highlight and the
            // bridge state cannot diverge. The mouse->currentIndex path is
            // ListView-standard (owner manual-visual covers it); this pins
            // the sync contract itself.
            list.currentIndex = 1
            wait(50)
            compare(fakeStudy.currentFrame, 1)
            verify(fakeStudy.setCurrentFrameLog.indexOf(1) !== -1)
            list.currentIndex = 3
            wait(50)
            compare(fakeStudy.currentFrame, 3)
        }

        function test_frameClickMovesIndex() {
            // Owner feedback 2026-08-12: the frame picker did not move
            // between frames (13 clicks, currentIndex never changed). The
            // old pin skipped the mouse path as 'ListView-standard' —
            // this pins it. Click row 2 of the frame list; the bridge
            // must receive frame 2.
            const panel = createTemporaryObject(panelComp, root, {
                appBridge: fakeApp, studyBridge: fakeStudy,
                optimizerBridge: fakeOpt, width: 240, height: 400 })
            verify(!!panel, "Component exists")
            const list = findChild(panel, "studyFrameList")
            verify(!!list, "Object exists")
            fakeStudy.frameListModel.append({ display: "frame 0" })
            fakeStudy.frameListModel.append({ display: "frame 1" })
            fakeStudy.frameListModel.append({ display: "frame 2" })
            wait(50)
            const row2 = list.itemAtIndex(2)
            verify(!!row2, "Row 2 exists")
            mouseClick(row2, 12, 12)
            wait(50)
            compare(list.currentIndex, 2)
            compare(fakeStudy.currentFrame, 2)
        }

        function test_modelClickTogglesRow() {
            // Same delegate mechanics as the frame list (required index +
            // nested MouseArea): clicking row 1 must toggle row 1, not 0.
            const panel = createTemporaryObject(panelComp, root, {
                appBridge: fakeApp, studyBridge: fakeStudy,
                optimizerBridge: fakeOpt, width: 240, height: 400 })
            verify(!!panel, "Component exists")
            const list = findChild(panel, "studyModelList")
            verify(!!list, "Object exists")
            fakeStudy.modelListModel.append({ display: "model 0" })
            fakeStudy.modelListModel.append({ display: "model 1" })
            fakeStudy.modelListModel.append({ display: "model 2" })
            wait(50)
            const row1 = list.itemAtIndex(1)
            verify(!!row1, "Row 1 exists")
            mouseClick(row1, 12, 12)
            wait(50)
            verify(fakeStudy.selectedModels.indexOf(1) !== -1,
                   "Row 1 selected, got: "
                   + JSON.stringify(fakeStudy.selectedModels))
        }

        function test_spaceTogglesModelRow() {
            const panel = createTemporaryObject(panelComp, root, {
                appBridge: fakeApp, studyBridge: fakeStudy,
                optimizerBridge: fakeOpt, width: 240, height: 400 })
            verify(!!panel, "Component exists")
            const list = findChild(panel, "studyModelList")
            verify(!!list, "Object exists")
            wait(50)
            const row2 = list.itemAtIndex(2)
            verify(!!row2, "Object exists")
            row2.forceActiveFocus()
            keyClick(Qt.Key_Space)
            wait(50)
            compare(fakeStudy.toggleLog.length, 1)
            compare(fakeStudy.toggleLog[0], 2)
        }

        function test_runLockDisablesLists() {
            const panel = createTemporaryObject(panelComp, root, {
                appBridge: fakeApp, studyBridge: fakeStudy,
                optimizerBridge: fakeOpt, width: 240, height: 400 })
            verify(!!panel, "Component exists")
            const frameList = findChild(panel, "studyFrameList")
            const modelList = findChild(panel, "studyModelList")
            verify(!!frameList, "Object exists")
            verify(!!modelList, "Object exists")
            fakeOpt.running = true
            compare(frameList.enabled, false)
            compare(modelList.enabled, false)
            fakeOpt.running = false
            compare(frameList.enabled, true)
        }

        function test_datasetSwapNeverWritesNegativeOne() {
            const panel = createTemporaryObject(panelComp, root, {
                appBridge: fakeApp, studyBridge: fakeStudy,
                optimizerBridge: fakeOpt, width: 240, height: 400 })
            verify(!!panel, "Component exists")
            const list = findChild(panel, "studyFrameList")
            verify(!!list, "Object exists")
            // Review fix (2026-08-12): simulate the REAL swap — the
            // model instance is replaced, which resets currentIndex to a
            // transient -1 while suppressFrameSync is raised. The old
            // test never created the transient, so removing the D7 guard
            // left it green.
            fakeStudy.frameListModel.append({ display: "f0" })
            fakeStudy.frameListModel.append({ display: "f1" })
            fakeStudy.frameListModel.append({ display: "f2" })
            list.currentIndex = 2
            wait(50)
            fakeStudy.currentFrame = 2
            fakeStudy.frameListModel.clear()
            fakeStudy.frameListModel.append({ display: "f0" })
            fakeStudy.frameListModel.append({ display: "f1" })
            fakeStudy.frameListModel.append({ display: "f2" })
            fakeStudy.datasetChanged()
            wait(150)   // let the Qt.callLater deferral run
            // The view re-synced to the bridge's frame...
            compare(list.currentIndex, 2)
            // ...and the transient reset never reached the bridge.
            compare(fakeStudy.setCurrentFrameLog.indexOf(-1), -1)
        }

        function test_mlDegradationMatrix() {
            const strip = createTemporaryObject(stripComp, root, {
                mlBridge: fakeMl, studyBridge: fakeStudy,
                optimizerBridge: fakeOpt, width: 240, height: 300 })
            verify(!!strip, "Component exists")
            const seg = findChild(strip, "mlSegmentButton")
            const est = findChild(strip, "mlEstimateButton")
            verify(!!seg, "Object exists")
            verify(!!est, "Object exists")
            // Nothing loaded: both disabled.
            compare(seg.enabled, false)
            compare(est.enabled, false)
            fakeStudy.hasDataset = true
            fakeStudy.currentFrame = 0
            fakeMl.hasSegmentModel = true
            compare(seg.enabled, true)
            compare(est.enabled, false)   // still needs primary + estimate
            fakeStudy.primaryModelIndex = 0
            fakeMl.hasEstimateModel = true
            compare(est.enabled, true)
            // D6: estimate REQUIRES the segment model.
            fakeMl.hasSegmentModel = false
            compare(est.enabled, false)
        }

        function test_mlRunLock() {
            const strip = createTemporaryObject(stripComp, root, {
                mlBridge: fakeMl, studyBridge: fakeStudy,
                optimizerBridge: fakeOpt, width: 240, height: 300 })
            verify(!!strip, "Component exists")
            fakeStudy.hasDataset = true
            fakeStudy.currentFrame = 0
            fakeStudy.primaryModelIndex = 0
            fakeMl.hasSegmentModel = true
            fakeMl.hasEstimateModel = true
            const seg = findChild(strip, "mlSegmentButton")
            const blackSil = findChild(strip, "mlBlackSilButton")
            const fem = findChild(strip, "mlFemButton")
            const tib = findChild(strip, "mlTibButton")
            verify(!!seg, "Object exists")
            verify(!!blackSil, "Object exists")
            verify(!!fem, "Object exists")
            verify(!!tib, "Object exists")
            fakeOpt.running = true
            compare(seg.enabled, false)
            compare(blackSil.enabled, false)
            compare(fem.enabled, false)
            compare(tib.enabled, false)
            fakeOpt.running = false
            compare(seg.enabled, true)
        }

        function test_blackSilLabelContrast() {
            // Owner feedback 2026-08-12: the Black sil. label rendered
            // BLACK on the dark panel (Material style color did not reach
            // the control). Pin: the label color must contrast >= 4.5:1
            // against the panel background (WCAG AA at caption size).
            const strip = createTemporaryObject(stripComp, root, {
                mlBridge: fakeMl, studyBridge: fakeStudy,
                optimizerBridge: fakeOpt, width: 240, height: 300 })
            verify(!!strip, "Component exists")
            const blackSil = findChild(strip, "mlBlackSilButton")
            verify(!!blackSil, "Object exists")
            const label = blackSil.contentItem
            verify(!!label, "Object exists")
            const fg = Qt.rgba(label.color.r, label.color.g,
                              label.color.b, 1)
            const bg = Qt.rgba(0.105882, 0.117647, 0.141176, 1)  // Theme.panel
            const lum = function(c) {
                const ch = function(v) {
                    return v <= 0.03928 ? v / 12.92
                                        : Math.pow((v + 0.055) / 1.055, 2.4)
                }
                return 0.2126 * ch(c.r) + 0.7152 * ch(c.g)
                       + 0.0722 * ch(c.b)
            }
            const l1 = lum(fg), l2 = lum(bg)
            const ratio = (Math.max(l1, l2) + 0.05)
                          / (Math.min(l1, l2) + 0.05)
            verify(ratio >= 4.5,
                   "Black sil. label contrast vs panel: " + ratio.toFixed(2))
        }

        function test_runBarRunRequested() {
            const bar = createTemporaryObject(barComp, root, {
                optimizerBridge: fakeOpt, width: 600, height: 44 })
            verify(!!bar, "Component exists")
            runRequestedSpy.target = bar
            const run = findChild(bar, "runBarRunButton")
            const stop = findChild(bar, "runBarStopButton")
            verify(!!run, "Object exists")
            verify(!!stop, "Object exists")
            compare(run.enabled, true)
            compare(stop.enabled, false)
            mouseClick(run)
            tryCompare(runRequestedSpy, "count", 1)
            fakeOpt.running = true
            fakeOpt.canRun = false
            compare(run.enabled, false)
            compare(stop.enabled, true)
            mouseClick(stop)
            tryCompare(fakeOpt, "stopCalls", 1)
            runRequestedSpy.target = null
        }

        function test_dirtyCloseGuard() {
            const dialog = createTemporaryObject(dialogComp, root, {
                poseBridge: fakePose, studyBridge: fakeStudy,
                optimizerBridge: fakeOpt })
            verify(!!dialog, "Component exists")
            discardSpy.target = dialog
            fakePose.dirty = true
            dialog.open()
            wait(100)
            dialog.close()
            wait(100)
            tryCompare(discardSpy, "count", 1)   // dirty close asks
            // Confirmed close (the Run-button path sets this first) stays
            // silent.
            dialog.open()
            wait(100)
            dialog.discardConfirmed = true
            dialog.close()
            wait(100)
            compare(discardSpy.count, 1)         // unchanged
            dialog.discardConfirmed = false
            discardSpy.target = null
        }

        // ---- Toolbar load flows (007 R3) ------------------------------
        function makeToolbar() {
            return createTemporaryObject(toolbarComp, root, {
                studyBridge: fakeStudy, optimizerBridge: fakeOpt,
                fileDialogBridge: fakePicker, viewport: fakeViewport,
                shellDirty: false, width: 900, height: 44 })
        }

        function test_toolbarCalibrationFirstMessage() {
            const bar = makeToolbar()
            verify(!!bar, "Component exists")
            msgSpy.target = bar
            fakeStudy.hasCalibration = false
            const images = findChild(bar, "toolbarImagesButton")
            verify(!!images, "Object exists")
            mouseClick(images)
            tryCompare(msgSpy, "count", 1)
            compare(msgSpy.signalArguments[0][1], "Load Calibration First!")
            msgSpy.target = null
        }

        function test_toolbarReplaceRequested() {
            const bar = makeToolbar()
            verify(!!bar, "Component exists")
            replaceSpy.target = bar
            fakeStudy.hasCalibration = true
            fakeStudy.frameCount = 5   // a second image set -> replace
            fakePicker.returnPaths = ["/s/0001.tif", "/s/0002.tif"]
            const images = findChild(bar, "toolbarImagesButton")
            verify(!!images, "Object exists")
            mouseClick(images)
            tryCompare(replaceSpy, "count", 1)
            const paths = replaceSpy.signalArguments[0][0]
            verify(!!paths && paths.length === 2,
                   "replaceRequested carries the paths")
            compare(paths[0], "/s/0001.tif")
            compare(fakeStudy.loadImagesLog.length, 0)  // not loaded directly
            replaceSpy.target = null
        }

        function test_toolbarModelsAfterCalibration() {
            const bar = makeToolbar()
            verify(!!bar, "Component exists")
            fakeStudy.hasCalibration = true
            fakePicker.returnPaths = ["/s/femur.stl", "/s/tibia.stl"]
            const models = findChild(bar, "toolbarModelsButton")
            verify(!!models, "Object exists")
            mouseClick(models)
            // tryCompare does not resolve dotted paths — compare the
            // array's own length property.
            tryCompare(fakeStudy.loadModelsLog, "length", 2)
            compare(fakeStudy.loadModelsLog[0], "/s/femur.stl")
            compare(fakeStudy.loadModelsLog[1], "/s/tibia.stl")
        }

        function test_toolbarRunLock() {
            const bar = makeToolbar()
            verify(!!bar, "Component exists")
            const camera = findChild(bar, "toolbarCameraModeButton")
            const model = findChild(bar, "toolbarModelModeButton")
            const images = findChild(bar, "toolbarImagesButton")
            verify(!!camera && !!model && !!images, "Object exists")
            // Review I4/D-07: the mode toggles escaped the original lock
            // inventory — pinned now.
            fakeOpt.running = true
            compare(camera.enabled, false)
            compare(model.enabled, false)
            compare(images.enabled, false)
            fakeOpt.running = false
            compare(camera.enabled, true)
            compare(model.enabled, true)
            compare(images.enabled, true)
        }

        function test_shellDirtyPill() {
            const bar = makeToolbar()
            verify(!!bar, "Component exists")
            const pill = findChild(bar, "toolbarDirtyPillLabel")
            verify(!!pill, "Object exists")
            bar.shellDirty = true
            wait(50)
            compare(pill.text, "● unsaved")
            bar.shellDirty = false
            wait(50)
            compare(pill.text, "saved")
        }
    }
}
