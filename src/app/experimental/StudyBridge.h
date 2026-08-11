// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// 005 U4: StudyBridge — the thin study-load adapter (R3, R17). Pass-through
// orchestration only: QML FileDialogs pick paths → the shared StudyLoadController
// (plan 006 U7 / R11 — the ONE load path both front-ends call: calibration
// one-use + dataset-replace policy, parse → populate → dedup → counts) over
// the app-owned dataset (ExperimentalSession) + the direct-compiled
// FrameListModel/ModelListModel + the app-owned ExperimentalScene. No behavior
// lives here beyond the orchestration order; every semantic (partial loads,
// dedup, calibration formats, camera decisions) comes from the seams it
// delegates to.
//
// Ownership (plan 005 bridge decomposition): AppBridge (the hub) owns the
// ExperimentalSession and creates this adapter; the list models are created
// here (direct-compiled — jtml_view is NOT linked, R1) and swapped for fresh
// instances when a dataset replace wipes the lists (the models are write-once
// with no reset API). Retired instances are released with deleteLater() — by
// the time the deletion event runs, the QML bindings have re-pointed at the
// fresh instances (datasetChanged was emitted and the event loop spun).
//
// Selection contract (delegate-based, no QItemSelectionModel anywhere):
//  - frame list: a single current index (QML ListView.currentIndex);
//  - model list: a multi-select row set owned here (DelegateSelection), with
//    primary = first selected row (the widgets selected[0].row() rule),
//    exposed to QML as selectedModels / primaryModelIndex /
//    selectedModelCount;
//  - every change mirrors into ExperimentalSession::session_state
//    (SetModelCount / SetFrameCount / SetSelectedModels / SetCurrentFrame —
//    the widgets SyncSessionState() tail) for U6's optimizer wiring.
//
// Scene → renderer chain: the bridge mutates the app-owned ExperimentalScene
// and emits sceneBackgroundChanged / sceneModelsChanged / sceneCameraChanged;
// main.qml glue calls the QmlVtkRenderer GUI-thread slots
// (updateBackground / updateModels / updateCamera). The bridge never touches
// VTK and never sees the renderer.
//
// Error mapping (widgets QMessageBox precedents): FileOpenFailed is silent;
// PixelSizeZero / InvalidCode / size-mismatch / calibration-required surface
// as messageRequested → one QML Dialog.
//
// Dataset replace (review fix): loading images while frames exist is a
// second study. QML confirms, then calls clearDataset() before loadImages().
// clearDataset() wipes frames/models/locations + fresh list models; the
// calibration is deliberately KEPT (one-use per session, widgets parity).

#pragma once

#include <QObject>
#include <QStringList>
#include <QVariantList>

class AppBridge;
class DelegateSelection;
class ExperimentalScene;
class FrameListModel;
class ModelListModel;
class SessionStateController;
namespace jta {
class SessionController;
}
struct ExperimentalSession;

/*The shared study-load controller (plan 006 U7): wraps the bridge's
 * SessionController (the parse seam + the shared active-camera/count
 * mirrors) and consults the session-state controller's M7 run-in-flight
 * probe at each load (L17). QtCore-only plain class — value member.*/
#include "services/study_load_controller.h"

class StudyBridge : public QObject {
    Q_OBJECT

    // Study surface: load-action enablement + pre-load shell states (R17).
    Q_PROPERTY(bool hasCalibration READ hasCalibration NOTIFY datasetChanged)
    Q_PROPERTY(bool calibratedForMonoplane READ calibratedForMonoplane NOTIFY datasetChanged)
    Q_PROPERTY(bool calibratedForBiplane READ calibratedForBiplane NOTIFY datasetChanged)
    Q_PROPERTY(bool hasDataset READ hasDataset NOTIFY datasetChanged)
    Q_PROPERTY(int frameCount READ frameCount NOTIFY datasetChanged)
    Q_PROPERTY(int modelCount READ modelCount NOTIFY datasetChanged)

    // Direct-compiled list models (fresh instances on dataset replace).
    Q_PROPERTY(QObject* frameListModel READ frameListModel NOTIFY datasetChanged)
    Q_PROPERTY(QObject* modelListModel READ modelListModel NOTIFY datasetChanged)

    // Delegate selection contract (no QItemSelectionModel).
    Q_PROPERTY(int currentFrame READ currentFrame NOTIFY selectionChanged)
    Q_PROPERTY(int primaryModelIndex READ primaryModelIndex NOTIFY selectionChanged)
    Q_PROPERTY(int selectedModelCount READ selectedModelCount NOTIFY selectionChanged)
    Q_PROPERTY(QVariantList selectedModels READ selectedModels NOTIFY selectionChanged)

public:
    explicit StudyBridge(AppBridge* hub, ExperimentalSession* session,
                         ExperimentalScene* scene,
                         SessionStateController* session_state_controller,
                         QObject* parent = nullptr);
    ~StudyBridge() override;

    // ---- Load actions (paths come from the QML FileDialogs) -------------
    Q_INVOKABLE void loadCalibration(const QString& file_path);
    Q_INVOKABLE void loadImages(const QStringList& paths);
    Q_INVOKABLE void loadModels(const QStringList& paths);
    // Dataset replace tail: wipes frames/models/locations + fresh list
    // models + scene; keeps the calibration (one-use per session). QML
    // calls this after the replace-confirm, before re-loading.
    Q_INVOKABLE void clearDataset();

    // ---- Delegate selection contract ------------------------------------
    Q_INVOKABLE void setCurrentFrame(int index);  // -1 = none
    Q_INVOKABLE void toggleModelSelected(int row);  // out-of-range rows ignored
    Q_INVOKABLE void clearModelSelection();
    Q_INVOKABLE bool isModelSelected(int row) const;

    // Model-centric pose sync (plan-005 feedback #2): called from QML when
    // the renderer reports an EndInteraction on the model style. Writes the
    // visually arranged pose into LocationStorage (so the optimizer starts
    // from it) + the scene, then emits viewerPoseApplied for the viewport
    // readout refresh. The scene model index is name-matched to loaded_models
    // (fallback: index).
    Q_INVOKABLE void applyViewerPose(int sceneModelIndex, double x, double y,
                                     double z, double xa, double ya, double za);

    // ---- Reads ----------------------------------------------------------
    bool hasCalibration() const;
    bool calibratedForMonoplane() const;
    bool calibratedForBiplane() const;
    bool hasDataset() const;
    int frameCount() const;
    int modelCount() const;
    int currentFrame() const;
    int primaryModelIndex() const;
    int selectedModelCount() const;
    QVariantList selectedModels() const;
    FrameListModel* frameListModel();
    ModelListModel* modelListModel();

signals:
    void datasetChanged();
    void selectionChanged();
    // The viewer finished a model-centric drag; the arranged pose is written
    // to the storage + scene. QML forwards to viewport.updatePose(index) for
    // the readout refresh (idempotent re-apply).
    void viewerPoseApplied(int sceneModelIndex);
    // Error/notice mapping (widgets QMessageBox precedents); QML shows one
    // Dialog for these.
    void messageRequested(const QString& title, const QString& message);
    // Scene → renderer chain: QML glue calls the renderer's GUI-thread slots.
    void sceneBackgroundChanged();
    void sceneModelsChanged();
    void sceneCameraChanged();

private:
    void syncSessionState();
    void syncHubCounts();
    void updateSceneBackground();
    void updateSceneModels();
    void updateSceneCamera();

    AppBridge* hub_;
    ExperimentalSession* session_;
    ExperimentalScene* scene_;
    jta::SessionController* controller_;
    /*Plan 006 U7: the shared study-load controller — the ONE load path both
     * front-ends call. Wraps controller_ (declared before it) and probes
     * session_state_controller_->runInFlight() (M7 → L17) at each load; the
     * probe is invoked only at load time, never during construction.*/
    jta::StudyLoadController study_load_controller_;
    /*Plan 006 U6: the shared session-state controller (owned by AppBridge,
     * the composition root — the hub wires the run-in-flight probe + the
     * seed-clear there). syncSessionState writes through it (the widgets
     * SyncSessionState tail relocated); the wrapped SessionState IS
     * ExperimentalSession::session_state.*/
    SessionStateController* session_state_controller_;
    DelegateSelection* selection_;
    FrameListModel* frame_list_model_;
    ModelListModel* model_list_model_;
};
