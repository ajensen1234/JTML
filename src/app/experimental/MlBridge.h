// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// 005 U7 / 006 U8: MlBridge — the thin ML adapter (R8, R16). Per-implant .pt
// pickers (segment femur / segment tibia / ONE estimate model — the plan's
// review fix), the loaded .pt path/state shown in the UI, per-frame segment +
// estimate ops on the CURRENT frame only (v1 loop scope: the all-frames
// deferral keeps the progress/cancel surface simple), the estimate seeds
// OptimizerBridge, and graceful degradation without .pt models (AE4:
// buttons disabled with a hint label; a clear message if invoked anyway;
// the plain-optimize path is untouched).
//
// Per-frame controller API (the U8 lesson, plan 004): the bridge owns the
// per-frame calls + the status surface; SegmentationController /
// ImplantEstimator stay per-frame/stateless. There is NO loop here — v1
// scope is the current frame.
//
// Shared orchestration (plan 006 U8 / R12 part, R13): the per-frame
// segment -> estimate -> SavePose -> seed CHAIN now lives once in the
// jta::MlOrchestrator (services); this bridge keeps the view-side surface
// (.pt pickers, the availability/status flags, the guards, the seed
// wiring) and injects the torch/CUDA ops. The chain (the old widgets
// segmentHelperFunction / estimate slots + this bridge's mirror):
//  - segment: torch::jit::load(seg .pt, kCUDA) -> SegmentFrame(original,
//    black_sil_used, model, 1024, 1024) -> copyTo(inverted) + the Frame
//    post-processing (edge/dilated/distance/curvature — segmentHelperFunction
//    parity) -> scene background mode = Inverted (the segmented view);
//  - estimate: torch::jit::load(est .pt, kCUDA) + GPUModel from the primary
//    model's triangle buffers + ImplantEstimateContext (scratch buffers,
//    calibration — the estimate slots' construction) ->
//    EstimateImplantPose(clamp per implant kind) -> SavePose into the
//    session storage + scene pose + OptimizerBridge seed (the next run()
//    applies it; the widgets equivalent is the estimate slots' SavePose
//    feeding LaunchOptimizer's by-value pose matrix).
//
// Degradation state machine (AE4, headless-testable — plan 005 U7 test
// scenario b): every torch/GPU entry is guarded behind the availability
// flags (hasSegmentModel / hasEstimateModel), so the missing-model paths
// never invoke torch and the state machine is testable headless. The
// torch/GPU paths themselves are manual-visual under xcb (the U9 parity
// run arbitrates numerically).
//
// torch #undef slots rule: this header is Qt-object + plain-data only (no
// torch includes; ml_orchestrator.h is torch-free too); the torch-bearing
// includes live in the .cpp AFTER the Qt-object headers (the repo's
// documented rule).

#pragma once

#include <QObject>
#include <QString>

/*Shared segment/estimate orchestrator (plan 006 U8 / R12 part): torch-free
 * header (OpenCV + domain only) — safe before any Qt-object include; the
 * torch-bearing includes stay in the .cpp AFTER the Qt-object headers.*/
#include "services/ml_orchestrator.h"

class AppBridge;
class ExperimentalScene;
class ExperimentalSession;
class OptimizerBridge;
class SettingsBridge;
class StudyBridge;
namespace jta {
class SegmentationController;
}

class MlBridge : public QObject {
    Q_OBJECT

    // ---- .pt model paths (QML FileDialog pickers / env fallback) --------
    // segmentFemPt: femur segmentation model (env fallback JTML_SEG_PT).
    // segmentTibPt: tibia segmentation model (no env fallback — the oracle
    // defines JTML_SEG_PT as the femur model; tibia comes from the picker).
    // estimatePt: the ONE pose-regression model (env fallback
    // JTML_FEM_ESTIMATE_PT). file:// URLs are normalized to local paths.
    Q_PROPERTY(QString segmentFemPt READ segmentFemPt WRITE setSegmentFemPt
                   NOTIFY mlModelsChanged)
    Q_PROPERTY(QString segmentTibPt READ segmentTibPt WRITE setSegmentTibPt
                   NOTIFY mlModelsChanged)
    Q_PROPERTY(QString estimatePt READ estimatePt WRITE setEstimatePt
                   NOTIFY mlModelsChanged)

    // ---- Availability flags (the AE4 degradation surface) ---------------
    // At least one segment .pt is set (Segment button enablement + the
    // bridge's guard before any torch work).
    Q_PROPERTY(bool hasSegmentModel READ hasSegmentModel NOTIFY mlModelsChanged)
    // The estimate .pt is set (Estimate button enablement + guard).
    Q_PROPERTY(bool hasEstimateModel READ hasEstimateModel NOTIFY mlModelsChanged)

    // ---- Segment/estimate knobs -----------------------------------------
    // Implant kind: picks the segment .pt (kind-preferred) + the estimate's
    // clamp_z_to_principal (Femur = true — the femoral slot's
    // principal-distance z clamp; Tibia = false — the tibial slot's direct
    // z). The two widgets estimate slots differ only in this clamp.
    Q_PROPERTY(int implantKind READ implantKind WRITE setImplantKind NOTIFY
                   mlModelsChanged)
    // Black silhouettes in the original image (widgets
    // actionBlack_Implant_Silhouettes... checkbox parity).
    Q_PROPERTY(bool blackSilhouette READ blackSilhouette WRITE
                   setBlackSilhouette NOTIFY mlModelsChanged)
    // Viewport background display mode (0 = Original, 1 = Inverted): the
    // ML strip's view toggle — the segmented result lives in the frame's
    // inverted image and the scene mode switches to Inverted on segment.
    Q_PROPERTY(int backgroundMode READ backgroundMode WRITE setBackgroundMode
                   NOTIFY backgroundModeChanged)

    // ---- Estimate result surface (QML display) --------------------------
    Q_PROPERTY(bool hasEstimate READ hasEstimate NOTIFY estimateChanged)
    Q_PROPERTY(QString estimateText READ estimateText NOTIFY estimateChanged)

    // ---- Status/hint surface (the AE4 hint label) -----------------------
    Q_PROPERTY(QString statusText READ statusText NOTIFY statusChanged)

public:
    explicit MlBridge(
        AppBridge* hub,
        ExperimentalSession* session,
        ExperimentalScene* scene,
        StudyBridge* study_bridge,
        SettingsBridge* settings_bridge,
        OptimizerBridge* optimizer_bridge,
        QObject* parent = nullptr);
    ~MlBridge() override;

    // ---- .pt model paths ------------------------------------------------
    QString segmentFemPt() const;
    void setSegmentFemPt(const QString& path);
    QString segmentTibPt() const;
    void setSegmentTibPt(const QString& path);
    QString estimatePt() const;
    void setEstimatePt(const QString& path);

    // ---- Availability flags ---------------------------------------------
    bool hasSegmentModel() const;
    bool hasEstimateModel() const;

    // ---- Segment/estimate knobs -----------------------------------------
    int implantKind() const;
    void setImplantKind(int kind);
    bool blackSilhouette() const;
    void setBlackSilhouette(bool used);
    int backgroundMode() const;
    void setBackgroundMode(int mode);

    // ---- Estimate result + status ---------------------------------------
    bool hasEstimate() const;
    QString estimateText() const;
    QString statusText() const;

    // ---- ML actions (QML buttons; v1 loop scope = the current frame) ----
    // Segment the CURRENT frame with the kind-preferred segment .pt:
    // torch load -> SegmentFrame -> the frame's inverted image + the
    // post-processing (edge/dilated/distance/curvature, widgets parity) ->
    // viewport to the segmented view. Degradation: no study / no current
    // frame / no segment model -> typed message, no torch work.
    Q_INVOKABLE void segmentCurrentFrame();
    // Estimate the primary model's pose on the CURRENT frame (the widgets
    // estimate-slot flow): segment first (a segment model is REQUIRED, as
    // in the widgets estimate actions), then GPUModel + torch pose model +
    // ImplantEstimateContext -> EstimateImplantPose -> SavePose + scene
    // pose + OptimizerBridge seed (the estimate seeds the optimizer) +
    // estimate display. Degradation: guards before any torch work.
    Q_INVOKABLE void estimateCurrentFrame();
    // Clear the estimate display + the pending optimizer seed (stale on a
    // selection change). Safe to call at any time; the saved pose stays.
    Q_INVOKABLE void clearEstimate();

    // The kind-preferred segment .pt (Femur -> segmentFemPt, Tibia ->
    // segmentTibPt), falling back to the other one when the preferred is
    // unset (one Segment button mirroring the two widgets actions). Public
    // so the preference rule is observable (headless-testable state
    // machine).
    QString activeSegmentModelPath() const;

signals:
    // .pt paths / availability / knobs changed (button + label rebinding).
    void mlModelsChanged();
    // The estimate result display changed.
    void estimateChanged();
    // The status/hint label changed.
    void statusChanged();
    // The viewport background display mode changed.
    void backgroundModeChanged();
    // The single QML Dialog mechanism (same channel as StudyBridge/
    // OptimizerBridge).
    void messageRequested(const QString& title, const QString& message);
    // QML glue relays: viewport.updateBackground() / viewport.updatePose().
    void sceneBackgroundChanged();
    void poseEstimated(int modelIndex);

private:
    // The primary selected model's row (-1 when none) — the v1
    // single-model rule (pose ops are primary-model-only).
    int primaryModelRow() const;
    // Guards shared by both actions: no run in flight, a current frame, a
    // loaded dataset. Returns false + a typed message when blocked.
    bool guardStudyReady();
    // The torch/GPU segment core (only called with a valid .pt path).
    void runSegmentOnCurrentFrame();
    void setStatus(const QString& text);

    AppBridge* hub_;
    ExperimentalSession* session_;
    ExperimentalScene* scene_;
    StudyBridge* study_bridge_;
    SettingsBridge* settings_bridge_;
    OptimizerBridge* optimizer_bridge_;
    /*Owned per-frame controller (stateless; the header stays torch-free —
     * the complete type + torch headers live in the .cpp only).*/
    jta::SegmentationController* segmentation_controller_ = nullptr;

    /*Shared ML orchestrator (plan 006 U8 / R12 part): the per-frame
     * segment -> estimate -> SavePose -> seed chain; the bridge injects
     * the torch/CUDA ops (wrapping segmentation_controller_ above) and
     * keeps the .pt pickers, the guards/status surface and the seed
     * wiring. Torch-free header (the torch includes stay in the .cpp
     * after the Qt-object headers).*/
    jta::MlOrchestrator ml_orchestrator_;

    QString segment_fem_pt_;
    QString segment_tib_pt_;
    QString estimate_pt_;
    int implant_kind_ = 0;  // ImplantKind::Femur
    bool black_sil_used_ = false;
    int background_mode_ = 0;  // BackgroundMode::Original
    bool has_estimate_ = false;
    QString estimate_text_;
    QString status_text_;
};
