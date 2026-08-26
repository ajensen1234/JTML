// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// 005 U7 / 006 U8: MlBridge implementation — see the header for the
// contract. The segment/estimate chain now lives once in the shared
// jta::MlOrchestrator (services); this bridge keeps the view-side surface
// (.pt pickers, availability/status flags, guards, seed wiring) and injects
// the torch/CUDA ops wrapping the seams (SegmentationController /
// ImplantEstimator / GPUModel / machine_learning_tools — untouched).

#include "MlBridge.h"

// Qt-object headers FIRST (the repo's documented torch #undef slots rule:
// torch-bearing includes come after any Qt-object header in a mixed TU).
#include <QUrl>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "AppBridge.h"
#include "ExperimentalScene.h"
#include "ExperimentalSession.h"
#include "OptimizerBridge.h"
#include "SettingsBridge.h"
#include "StudyBridge.h"

// Torch/CUDA-bearing headers (after the Qt-object headers above).
#include <torch/torch.h>

#include "compute/CostFunctionManager.h"  // dilation param (widgets parity)
#include "compute/gpu_model.cuh"          // GPUModel (estimate context)
#include "domain/data_structures_6D.h"
#include "services/ml_orchestrator.h"
#include "services/segmentation_controller.h"

namespace {

/*The segmentation/estimate input dims (the widgets hardcodes 1024 x 1024 —
 * the traced models' batch contract).*/
constexpr unsigned int kInputWidth = 1024;
constexpr unsigned int kInputHeight = 1024;

/*QML FileDialog yields file:// URLs; the torch loader wants local paths
 * (the widgets QFileDialog returned plain paths). Plain paths pass through
 * untouched (the headless tests call with plain paths).*/
QString LocalPath(const QString& path) {
    if (path.startsWith(QStringLiteral("file://"))) {
        return QUrl(path).toLocalFile();
    }
    return path;
}

/*The estimate display row: mm translations + degree rotations, rounded to
 * 2 decimals (widgets pose display parity).*/
QString FormatPose(const Point6D& pose) {
    return QStringLiteral("x %1 · y %2 · z %3 · xa %4 · ya %5 · za %6")
        .arg(pose.x, 0, 'f', 2)
        .arg(pose.y, 0, 'f', 2)
        .arg(pose.z, 0, 'f', 2)
        .arg(pose.xa, 0, 'f', 2)
        .arg(pose.ya, 0, 'f', 2)
        .arg(pose.za, 0, 'f', 2);
}

}  // namespace

MlBridge::MlBridge(
    AppBridge* hub,
    ExperimentalSession* session,
    ExperimentalScene* scene,
    StudyBridge* study_bridge,
    SettingsBridge* settings_bridge,
    OptimizerBridge* optimizer_bridge,
    QObject* parent) :
    QObject(parent),
    hub_(hub),
    session_(session),
    scene_(scene),
    study_bridge_(study_bridge),
    settings_bridge_(settings_bridge),
    optimizer_bridge_(optimizer_bridge),
    segmentation_controller_(new jta::SegmentationController) {
    /*Env fallback (plan 005 U7): JTML_SEG_PT (femur segmentation model) +
     * JTML_FEM_ESTIMATE_PT (pose regression) — the oracle's user-provided
     * fixture vars; a picker selection overrides. The tibia segment model
     * has no env var (the oracle defines JTML_SEG_PT as the femur model).*/
    const char* seg_pt = std::getenv("JTML_SEG_PT");
    if (seg_pt != nullptr && seg_pt[0] != '\0') {
        segment_fem_pt_ = QString::fromLocal8Bit(seg_pt);
    }
    const char* est_pt = std::getenv("JTML_FEM_ESTIMATE_PT");
    if (est_pt != nullptr && est_pt[0] != '\0') {
        estimate_pt_ = QString::fromLocal8Bit(est_pt);
    }
    /*A stale estimate display/seed would mislead after a frame or model
     * selection change: clear both (the saved pose stays in the storage —
     * only the display + the one-shot run seed are stale).*/
    connect(
        study_bridge_,
        &StudyBridge::selectionChanged,
        this,
        &MlBridge::clearEstimate);
}

MlBridge::~MlBridge() {
    delete segmentation_controller_;
}

/*---- .pt model paths ----*/

QString MlBridge::segmentFemPt() const {
    return segment_fem_pt_;
}

void MlBridge::setSegmentFemPt(const QString& path) {
    const QString local = LocalPath(path);
    if (segment_fem_pt_ == local) {
        return;
    }
    segment_fem_pt_ = local;
    emit mlModelsChanged();
}

QString MlBridge::segmentTibPt() const {
    return segment_tib_pt_;
}

void MlBridge::setSegmentTibPt(const QString& path) {
    const QString local = LocalPath(path);
    if (segment_tib_pt_ == local) {
        return;
    }
    segment_tib_pt_ = local;
    emit mlModelsChanged();
}

QString MlBridge::estimatePt() const {
    return estimate_pt_;
}

void MlBridge::setEstimatePt(const QString& path) {
    const QString local = LocalPath(path);
    if (estimate_pt_ == local) {
        return;
    }
    estimate_pt_ = local;
    emit mlModelsChanged();
}

/*---- Availability flags (the AE4 degradation surface) ----*/

bool MlBridge::hasSegmentModel() const {
    return !segment_fem_pt_.isEmpty() || !segment_tib_pt_.isEmpty();
}

bool MlBridge::hasEstimateModel() const {
    return !estimate_pt_.isEmpty();
}

/*---- Segment/estimate knobs ----*/

int MlBridge::implantKind() const {
    return implant_kind_;
}

void MlBridge::setImplantKind(int kind) {
    if (kind != 0 && kind != 1) {
        return;
    }
    if (implant_kind_ == kind) {
        return;
    }
    implant_kind_ = kind;
    emit mlModelsChanged();
}

bool MlBridge::blackSilhouette() const {
    return black_sil_used_;
}

void MlBridge::setBlackSilhouette(bool used) {
    if (black_sil_used_ == used) {
        return;
    }
    black_sil_used_ = used;
    emit mlModelsChanged();
}

int MlBridge::backgroundMode() const {
    return background_mode_;
}

void MlBridge::setBackgroundMode(int mode) {
    if (mode != 0 && mode != 1) {
        return;
    }
    if (background_mode_ == mode) {
        return;
    }
    background_mode_ = mode;
    scene_->setBackgroundMode(static_cast<BackgroundMode>(mode));
    emit backgroundModeChanged();
    emit sceneBackgroundChanged();
}

/*---- Estimate result + status ----*/

bool MlBridge::hasEstimate() const {
    return has_estimate_;
}

QString MlBridge::estimateText() const {
    return estimate_text_;
}

QString MlBridge::statusText() const {
    return status_text_;
}

/*---- ML actions (v1 loop scope = the current frame) ----*/

void MlBridge::segmentCurrentFrame() {
    if (!guardStudyReady()) {
        return;
    }
    /*Degradation (AE4): no segment .pt -> clear message, no torch work,
     * nothing changes (the buttons are disabled in the shell; the guard
     * covers direct calls).*/
    if (!hasSegmentModel()) {
        emit messageRequested(
            QStringLiteral("Error!"),
            QStringLiteral("No segmentation model loaded — pick a femur or "
                           "tibia .pt model first."));
        setStatus(QStringLiteral("No segmentation model (.pt) loaded."));
        return;
    }
    runSegmentOnCurrentFrame();
}

void MlBridge::estimateCurrentFrame() {
    if (!guardStudyReady()) {
        return;
    }
    if (!hasEstimateModel()) {
        emit messageRequested(
            QStringLiteral("Error!"),
            QStringLiteral("No pose-estimation model loaded — pick an "
                           "estimate .pt model first."));
        setStatus(QStringLiteral("No estimate model (.pt) loaded."));
        return;
    }
    const int primary = primaryModelRow();
    if (primary < 0) {
        emit messageRequested(
            QStringLiteral("Error!"),
            QStringLiteral("Select one model to estimate."));
        setStatus(QStringLiteral("No model selected."));
        return;
    }
    /*The widgets estimate actions segment FIRST and REQUIRE the segment
     * model (they prompt through the nested segment action); degradation
     * with a clear message, no torch work.*/
    if (!hasSegmentModel()) {
        emit messageRequested(
            QStringLiteral("Error!"),
            QStringLiteral("Estimate needs a segmentation model too — pick "
                           "a femur or tibia .pt first."));
        setStatus(QStringLiteral("No segmentation model (.pt) loaded."));
        return;
    }

    /*Segment the current frame first (the widgets estimate-slot flow:
     * on_actionSegment_*_triggered() before the pose regression). A failed
     * segment ABORTS the estimate — the regression would otherwise run on
     * the stale inverted image and overwrite the segment-failure status
     * (review fix P2-3).*/
    if (!runSegmentOnCurrentFrame()) {
        return;
    }

    const int frame = study_bridge_->currentFrame();
    const QString est_path = estimate_pt_;

    /*torch::jit::load (widgets estimate-slot parity — the view keeps the
     * model loading; a failure surfaces the widgets' typed message).*/
    torch::jit::Module module;
    try {
        module = torch::jit::load(est_path.toStdString(), torch::kCUDA);
    } catch (const c10::Error&) {
        emit messageRequested(
            QStringLiteral("Error!"),
            QStringLiteral("Cannot load PyTorch Torch Script model at: ") +
                est_path);
        setStatus(QStringLiteral("Estimate model failed to load."));
        return;
    }
    torch::jit::Module* model = &module;

    Frame& frame_data = session_->loaded_frames[static_cast<size_t>(frame)];
    const unsigned int orig_height =
        static_cast<unsigned int>(frame_data.GetInvertedImage().rows);
    const unsigned int orig_width =
        static_cast<unsigned int>(frame_data.GetInvertedImage().cols);
    /*Scratch buffers (the estimate slots malloc/free these around the
     * loop — the estimator never owns them).*/
    auto host_image = static_cast<unsigned char*>(
        malloc(kInputWidth * kInputHeight * sizeof(unsigned char)));
    auto orientation = new float[3];

    /*GPUModel from the primary model's triangle buffers (the oracle test's
     * construction — the Model already carries the STL floats; the widgets
     * re-reads the STL with stl_reader_BIG, same data). Backface culling
     * OFF (the widgets comment: "BACKFACE CULLING APPEARS TO BE GIVING
     * ERRORS" — the oracle pins the same config).*/
    Model& primary_model =
        session_->loaded_models[static_cast<size_t>(primary)];
    if (!primary_model.initialized_correctly_ ||
        primary_model.triangle_vertices_.size() < 9) {
        emit messageRequested(
            QStringLiteral("Error!"),
            QStringLiteral("The selected model's STL did not load "
                           "correctly."));
        setStatus(QStringLiteral("Model STL missing."));
        free(host_image);
        delete[] orientation;
        return;
    }
    auto gpu_mod = new gpu_cost_function::GPUModel(
        "model",
        /*principal_model=*/true,
        static_cast<int>(orig_height),
        static_cast<int>(orig_width),
        /*device_primary_cam=*/0,
        /*use_backface_culling_primary_cam=*/false,
        &primary_model.triangle_vertices_[0],
        &primary_model.triangle_normals_[0],
        static_cast<int>(primary_model.triangle_vertices_.size() / 9),
        session_->calibration_file.camera_A_principal_);
    if (!gpu_mod->IsInitializedCorrectly()) {
        emit messageRequested(
            QStringLiteral("Error!"),
            QStringLiteral("Could not initialize the GPU model for the "
                           "pose estimate."));
        setStatus(QStringLiteral("GPU model failed to initialize."));
        delete gpu_mod;
        free(host_image);
        delete[] orientation;
        return;
    }

    /*Per-frame estimate context (the estimate slots' construction verbatim:
     * GPU model, torch pose model, scratch buffers, calibration).*/
    torch::Tensor gpu_byte_placeholder(torch::zeros(
        {1, 1, kInputHeight, kInputWidth},
        device(torch::kCUDA).dtype(torch::kByte)));
    jta::ImplantEstimateContext ctx;
    ctx.gpu_mod = gpu_mod;
    ctx.model = model;
    ctx.host_image = host_image;
    ctx.orientation = orientation;
    ctx.gpu_byte_placeholder = gpu_byte_placeholder;
    ctx.calibration = session_->calibration_file;
    ctx.input_width = kInputWidth;
    ctx.input_height = kInputHeight;
    ctx.orig_width = orig_width;
    ctx.orig_height = orig_height;

    /*clamp_z_to_principal per implant kind (the two widgets estimate slots
     * differ only in this): Femur = true — the femoral slot's
     * principal-distance z clamp; Tibia = false — the tibial slot's direct
     * z.*/
    const bool clamp_z_to_principal = (implant_kind_ == 0);
    /*The shared orchestrator (plan 006 U8 / R12 part): the per-frame
     * estimate op -> SavePose -> seed chain. The op wraps the estimate
     * math (this controller's per-frame op with the built context); the
     * save writes the session storage (the by-value matrix Initialize
     * copies — the widgets equivalent is the estimate slots' SavePose
     * feeding LaunchOptimizer); the RETURNED seed is wired below (R8).*/
    const auto estimate_op =
        [this, &ctx, clamp_z_to_principal](const cv::Mat& inverted) {
            return segmentation_controller_->EstimateImplantPose(
                ctx, inverted, clamp_z_to_principal);
        };
    const auto save_pose = [this](int f, int m, const Point6D& pose) {
        session_->model_locations.SavePose(f, m, pose);
    };
    jta::MlEstimateOutcome outcome = ml_orchestrator_.EstimateFrame(
        frame, primary, frame_data.GetInvertedImage(), estimate_op, save_pose);

    /*Cleanup (the estimate slots' tail: delete GPU model, free scratch).*/
    delete gpu_mod;
    free(host_image);
    delete[] orientation;

    /*Degradation (AE4): an estimate-op failure surfaces a typed message
     * and leaves the display + seed untouched (the storage was not
     * written; the plain-optimize path is unaffected).*/
    if (outcome.status != jta::MlEstimateStatus::Ok) {
        emit messageRequested(
            QStringLiteral("Error!"),
            QStringLiteral("Pose estimation failed."));
        setStatus(QStringLiteral("Pose estimation failed."));
        return;
    }

    /*The estimate seeds the optimizer (R8): the orchestrator saved the
     * pose into the session storage; the bridge maps the returned seed
     * onto the scene (viewport re-render at the estimated pose) and the
     * OptimizerBridge one-shot seed (applied by the next run(), winning
     * over scene drift). The stale guards for a frame/model change between
     * estimate and run live in the run controller (U5 takeSeedForRun).*/
    scene_->setModelPose(primary, outcome.pose);
    optimizer_bridge_->setSeedPose(
        outcome.seed.pose.x,
        outcome.seed.pose.y,
        outcome.seed.pose.z,
        outcome.seed.pose.xa,
        outcome.seed.pose.ya,
        outcome.seed.pose.za);
    has_estimate_ = true;
    estimate_text_ = FormatPose(outcome.pose);
    emit estimateChanged();
    emit poseEstimated(primary);
    setStatus(QStringLiteral("Estimated pose for frame %1 (model %2).")
                  .arg(frame)
                  .arg(primary));
}

void MlBridge::clearEstimate() {
    if (!has_estimate_ && estimate_text_.isEmpty()) {
        return;
    }
    has_estimate_ = false;
    estimate_text_.clear();
    optimizer_bridge_->clearSeedPose();
    emit estimateChanged();
}

/*---- Private helpers ----*/

QString MlBridge::activeSegmentModelPath() const {
    /*Kind-preferred (Femur -> segmentFemPt, Tibia -> segmentTibPt), falling
     * back to the other picker when the preferred one is unset (one
     * Segment button — the widgets has two actions; the fallback keeps the
     * button usable with a single .pt).*/
    if (implant_kind_ == 1 && !segment_tib_pt_.isEmpty()) {
        return segment_tib_pt_;
    }
    if (implant_kind_ == 0 && !segment_fem_pt_.isEmpty()) {
        return segment_fem_pt_;
    }
    if (!segment_fem_pt_.isEmpty()) {
        return segment_fem_pt_;
    }
    return segment_tib_pt_;
}

int MlBridge::primaryModelRow() const {
    /*The v1 single-model rule: pose ops are primary-model-only (the first
     * selected row — DelegateSelection ascending rule).*/
    return study_bridge_->primaryModelIndex();
}

bool MlBridge::guardStudyReady() {
    /*U6 locking belt-and-braces: the shell disables the ML buttons during
     * a run; the bridge guards anyway (QML bindings can race).*/
    if (optimizer_bridge_->running()) {
        emit messageRequested(
            QStringLiteral("Error!"),
            QStringLiteral("Stop the optimizer run first."));
        return false;
    }
    if (session_->loaded_frames.empty()) {
        emit messageRequested(
            QStringLiteral("Error!"), QStringLiteral("Load images first!"));
        setStatus(QStringLiteral("No frames loaded."));
        return false;
    }
    if (study_bridge_->currentFrame() < 0) {
        emit messageRequested(
            QStringLiteral("Error!"), QStringLiteral("Select a frame first."));
        setStatus(QStringLiteral("No current frame."));
        return false;
    }
    return true;
}

bool MlBridge::runSegmentOnCurrentFrame() {
    const int frame = study_bridge_->currentFrame();
    const QString pt_path = activeSegmentModelPath();

    /*torch::jit::load (widgets segmentHelperFunction parity — a load
     * failure surfaces the widgets' typed message).*/
    torch::jit::Module module;
    try {
        module = torch::jit::load(pt_path.toStdString(), torch::kCUDA);
    } catch (const c10::Error&) {
        emit messageRequested(
            QStringLiteral("Error!"),
            QStringLiteral("Cannot load PyTorch Torch Script model at: ") +
                pt_path);
        setStatus(QStringLiteral("Segmentation model failed to load."));
        return false;
    }
    torch::jit::Module* model = &module;

    /*Per-frame segment (plan 006 U8 / R12 part): the shared orchestrator
     * owns the segment op -> inverted copy -> post-processing chain
     * (segmentHelperFunction parity — edge/dilated/distance/curvature);
     * the bridge keeps the .pt load + the dilation sourcing.*/
    Frame& frame_data = session_->loaded_frames[static_cast<size_t>(frame)];
    /*Dilation from the active trunk cost function (the widgets
     * segmentHelperFunction reads ui/trunk_manager_ the same way).*/
    int dilation_val = 0;
    settings_bridge_->trunkManager()
        ->getActiveCostFunctionClass()
        ->getIntParameterValue("Dilation", dilation_val);
    const auto segment_op = [this, model](const cv::Mat& original) {
        return segmentation_controller_->SegmentFrame(
            original, black_sil_used_, model, kInputWidth, kInputHeight);
    };
    const jta::MlSegmentStatus status = ml_orchestrator_.SegmentFrame(
        frame_data,
        frame_data.GetAperture(),
        frame_data.GetLowThreshold(),
        frame_data.GetHighThreshold(),
        dilation_val,
        /*full_postprocessing=*/true,
        segment_op);
    if (status != jta::MlSegmentStatus::Ok) {
        /*Degradation (AE4): a segment failure surfaces a typed message and
         * leaves the frame + viewport untouched.*/
        emit messageRequested(
            QStringLiteral("Error!"), QStringLiteral("Segmentation failed."));
        setStatus(QStringLiteral("Segmentation failed."));
        return false;
    }

    /*The segmented view: the frame's inverted image now holds the
     * silhouette; the viewport re-renders (the mode flip + the content
     * change both flow through sceneBackgroundChanged).*/
    setBackgroundMode(1);  // BackgroundMode::Inverted
    emit sceneBackgroundChanged();
    setStatus(QStringLiteral("Segmented frame %1.").arg(frame));
    return true;
}

void MlBridge::setStatus(const QString& text) {
    if (status_text_ == text) {
        return;
    }
    status_text_ = text;
    emit statusChanged();
}
