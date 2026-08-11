// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

// 005 U4: StudyBridge implementation — see the header for the contract.
// Everything here is orchestration order around the seams; the widgets
// precedents it mirrors are called out per site.

#include "StudyBridge.h"

// Qt
#include <QUrl>
#include <QVector>

// OpenCV (scene background buffers)
#include <opencv2/core.hpp>

// The seams + the app-owned dataset + the direct-compiled list models.
#include "AppBridge.h"
#include "DelegateSelection.h"
#include "ExperimentalScene.h"
#include "ExperimentalSession.h"
#include "compute/frame.h"
#include "domain/data_structures_6D.h"
#include "services/model.h"
#include "services/session_controller.h"
#include "view/frame_list_model.h"
#include "view/model_list_model.h"

#include "coordinator/session_state_controller.h"

#include <cmath>
#include <vector>

namespace {

/*QML FileDialog yields file:// URLs; the seams want local paths (the widgets
 * QFileDialog returned plain paths). QUrl handles the scheme stripping +
 * percent-decoding; plain paths pass through untouched (the headless tests
 * call with plain paths).*/
QString LocalPath(const QString& path) {
    if (path.startsWith(QStringLiteral("file://"))) {
        return QUrl(path).toLocalFile();
    }
    return path;
}

QStringList LocalPaths(const QStringList& paths) {
    QStringList local;
    local.reserve(paths.size());
    for (const QString& path : paths) {
        local.push_back(LocalPath(path));
    }
    return local;
}

/*Widgets CalculateViewingAngle(A) replica (mainscreen.cpp:134): the view
 * angle that frames the image height at the principal distance (camera A —
 * v1 is monoplane, R17).*/
double ViewingAngleForFrame(const Calibration& cal, int width, int height) {
    Q_UNUSED(width);
    static constexpr double kPi = 3.14159265358979323846;
    const double y = height * cal.camera_A_principal_.pixel_pitch_ / 2.0 +
                     std::abs(cal.camera_A_principal_.principal_y_);
    return 180.0 / kPi * 2.0 *
           std::atan2(y, cal.camera_A_principal_.principal_distance_);
}

}  // namespace

StudyBridge::StudyBridge(AppBridge* hub, ExperimentalSession* session,
                         ExperimentalScene* scene,
                         SessionStateController* session_state_controller,
                         QObject* parent)
    : QObject(parent),
      hub_(hub),
      session_(session),
      scene_(scene),
      controller_(new jta::SessionController),
      /*Plan 006 U7: the shared study-load controller wraps controller_ and
       * consults the session-state controller's M7 run-in-flight probe at
       * each load (L17 — the QML app has no load-time guard today; the
       * shared check is defense-in-depth, rejection is silent). The lambda
       * is invoked only at load time, never during construction.*/
      study_load_controller_(
          controller_,
          [this] { return session_state_controller_->runInFlight(); }),
      session_state_controller_(session_state_controller),
      selection_(new DelegateSelection),
      frame_list_model_(new FrameListModel),
      model_list_model_(new ModelListModel) {}

StudyBridge::~StudyBridge() {
    /*The engine (and with it the QML bindings) is destroyed before the hub
     * in main.cpp, so the current instances are unreferenced here. Retired
     * instances self-delete via deleteLater().*/
    delete frame_list_model_;
    delete model_list_model_;
    delete selection_;
    delete controller_;
}

/*---- Load actions ----*/

void StudyBridge::loadCalibration(const QString& file_path) {
    /*One shared load path (plan 006 U7 / R11): the one-use-per-session
     * rejection + parse + the caller-owned container writes (calibration +
     * flags + the shared active-camera mirror) relocated into
     * StudyLoadController.*/
    const jta::StudyCalibrationLoadResult result =
        study_load_controller_.LoadCalibration(
            LocalPath(file_path),
            session_->calibration_file,
            session_->calibrated_for_monoplane_viewport,
            session_->calibrated_for_biplane_viewport);
    /*Policy rejections are silent (widgets parity: the load button disables
     * after a successful load — one-use per session; the run-in-flight
     * guard is defense-in-depth).*/
    if (result.status == jta::StudyLoadStatus::RunInFlight ||
        result.status == jta::StudyLoadStatus::CalibrationAlreadyLoaded) {
        return;
    }
    /*Typed error mapping (widgets precedent): PixelSizeZero / InvalidCode
     * show the box; FileOpenFailed is silent and changes nothing.*/
    if (result.parse.error == jta::CalibrationParseResult::Error::PixelSizeZero) {
        emit messageRequested(
            QStringLiteral("Error!"),
            QStringLiteral("Pixel size (the last number in the calibration "
                           "file) is specified as 0! This is impossible."));
        return;
    }
    if (result.parse.error == jta::CalibrationParseResult::Error::InvalidCode) {
        emit messageRequested(QStringLiteral("Error!"),
                              QStringLiteral("Invalid Configuration File!"));
        return;
    }
    if (!result.parse.ok) {
        return;  // FileOpenFailed: silent, state unchanged
    }
    /*The controller wrote the calibration + flags + the active-camera
     * mirror; the scene focal is a view mapping (the widgets sets it inside
     * Viewer::setup_camera_calibration).*/
    scene_->setFocalLengthPx(result.parse.calibration.camera_A_principal_.principal_distance_);
    syncSessionState();
    emit datasetChanged();
    emit sceneCameraChanged();
}

void StudyBridge::loadImages(const QStringList& paths) {
    /*Calibration-required guard (widgets "Load Calibration First!" box).*/
    if (!hasCalibration()) {
        emit messageRequested(QStringLiteral("Error!"),
                              QStringLiteral("Load Calibration First!"));
        return;
    }
    /*Edge params: the widgets reads its live controls; the QML app has no
     * edge controls yet (a later unit) — the session_controller_test
     * defaults ({3, 40, 120, 0}) stand in. They only feed the edge/dilation
     * images; the viewport background is the ORIGINAL frame image.*/
    /*One shared load path (plan 006 U7 / R11): the parse + populate +
     * counts relocated into StudyLoadController. The run-in-flight
     * rejection is silent (defense-in-depth — the shared check lands
     * without new user-visible behavior).*/
    const jta::StudyImageLoadResult result = study_load_controller_.LoadImages(
        LocalPaths(paths), jta::ImageLoadParams{3, 40, 120, 0},
        session_->loaded_frames, session_->model_locations);
    if (result.status == jta::StudyLoadStatus::RunInFlight) {
        return;
    }
    for (const QString& name : result.frame_names) {
        frame_list_model_->AppendFrame(name);
    }
    if (result.status == jta::StudyLoadStatus::SizeMismatchAborted) {
        /*goto-stop semantics: the frames appended so far persist; the box is
         * shown after the call, exactly like the widgets slot.*/
        emit messageRequested(
            QStringLiteral("Error!"),
            QStringLiteral("Images Loaded Must Be The Same Size!"));
    }
    /*Default-select the first frame (widgets parity: "If No Loaded Frames,
     * Default Select First").*/
    if (!session_->loaded_frames.empty() && selection_->GetCurrentFrame() < 0) {
        selection_->SetCurrentFrame(0);
    }
    updateSceneBackground();
    updateSceneCamera();
    syncSessionState();
    syncHubCounts();
    emit selectionChanged();
    emit datasetChanged();
    emit sceneBackgroundChanged();
    emit sceneCameraChanged();
}

void StudyBridge::loadModels(const QStringList& paths) {
    /*Models-before-calibration guard (review fix): PopulateModels needs a
     * calibration — defer + prompt (widgets "Load Calibration First!" box).*/
    if (!hasCalibration()) {
        emit messageRequested(QStringLiteral("Error!"),
                              QStringLiteral("Load Calibration First!"));
        return;
    }
    /*One shared load path (plan 006 U7 / R11): parse -> dedup -> populate ->
     * counts relocated into StudyLoadController; the two-pass dedup runs
     * through this bridge's ModelListModel::AppendModels seam (the
     * ModelListBuilder mutated-name-rescan quirk preserved — the returned
     * unique names drive the Model names AND the renderer binding). The
     * run-in-flight rejection is silent (defense-in-depth).*/
    const jta::StudyModelLoadResult result = study_load_controller_.LoadModels(
        LocalPaths(paths), session_->calibration_file,
        session_->loaded_models, session_->model_locations,
        [this](const QVector<QString>& base_names) {
            return model_list_model_->AppendModels(base_names);
        });
    if (result.status == jta::StudyLoadStatus::RunInFlight) {
        return;
    }
    /*STL-parse warnings: the widgets checks per-file
     * (vw->are_models_loaded_incorrectly); here the Model carries
     * initialized_correctly_ and the renderer logs failed loads
     * (QmlVtkRenderer::RebuildModels) — the v1 surface has no per-file
     * warning slot.*/
    updateSceneModels();
    syncSessionState();
    syncHubCounts();
    emit datasetChanged();
    emit sceneModelsChanged();
}

void StudyBridge::clearDataset() {
    session_->ClearDataset();
    /*Cross-dataset state (plan 006 U6, H5/M10b): the shared controller
     * resets the previous-selection mirrors (save-last-pose must never
     * name the wiped dataset), drops the optimizer's pending seed (via the
     * seed-clear wired in AppBridge), and emits its datasetChanged. The
     * current values were already wiped by ClearDataset above — the
     * subsequent syncSessionState diffs to no change.*/
    session_state_controller_->ResetForDatasetClear();
    selection_->SetCurrentFrame(-1);
    selection_->ClearModelSelection();
    /*Fresh list models: the models are write-once with no reset API. The
     * retired instances are released with deleteLater() — the QML bindings
     * re-point at the fresh instances on the datasetChanged emission.*/
    frame_list_model_->deleteLater();
    model_list_model_->deleteLater();
    frame_list_model_ = new FrameListModel;
    model_list_model_ = new ModelListModel;
    scene_->setBackgroundImage(cv::Mat());
    scene_->setModels({});
    syncSessionState();
    syncHubCounts();
    emit selectionChanged();
    emit datasetChanged();
    emit sceneBackgroundChanged();
    emit sceneModelsChanged();
}

/*---- Delegate selection contract ----*/

void StudyBridge::setCurrentFrame(int index) {
    selection_->SetCurrentFrame(index);
    updateSceneBackground();
    syncSessionState();
    emit selectionChanged();
    emit sceneBackgroundChanged();
}

void StudyBridge::toggleModelSelected(int row) {
    if (row < 0 || row >= model_list_model_->rowCount()) {
        return;
    }
    selection_->ToggleModel(row);
    syncSessionState();
    emit selectionChanged();
}

void StudyBridge::clearModelSelection() {
    if (!selection_->HasModelSelection()) {
        return;
    }
    selection_->ClearModelSelection();
    syncSessionState();
    emit selectionChanged();
}

bool StudyBridge::isModelSelected(int row) const {
    return selection_->IsModelSelected(row);
}

void StudyBridge::applyViewerPose(int sceneModelIndex, double x, double y,
                                  double z, double xa, double ya, double za) {
    // Plan-005 feedback #2: the model-centric drag ended — sync the visually
    // arranged pose into LocationStorage (the optimizer's starting point:
    // OptimizerBridge::run passes the storage by value into Initialize) and
    // into the scene. Pure orchestration around the seams.
    if (!scene_ || !session_ || sceneModelIndex < 0) {
        return;
    }
    const std::vector<SceneModel> scene_models = scene_->models();
    if (sceneModelIndex >= static_cast<int>(scene_models.size())) {
        return;
    }
    const int frame = currentFrame();
    if (frame < 0) {
        return;
    }
    // Name-match the scene model to loaded_models (both are built from the
    // same parse in order; the match is defensive) — the primary may not be
    // scene index 0 once multi-model selection lands.
    int model_index = sceneModelIndex;
    const std::string& scene_name = scene_models[static_cast<size_t>(sceneModelIndex)].name;
    for (int i = 0; i < static_cast<int>(session_->loaded_models.size()); ++i) {
        if (session_->loaded_models[static_cast<size_t>(i)].model_name_ ==
            scene_name) {
            model_index = i;
            break;
        }
    }
    const Point6D pose(x, y, z, xa, ya, za);
    session_->model_locations.SavePose(frame, model_index, pose);
    scene_->setModelPose(sceneModelIndex, pose);
    emit viewerPoseApplied(sceneModelIndex);
}

/*---- Reads ----*/

bool StudyBridge::hasCalibration() const {
    return session_->calibrated_for_monoplane_viewport ||
           session_->calibrated_for_biplane_viewport;
}

bool StudyBridge::calibratedForMonoplane() const {
    return session_->calibrated_for_monoplane_viewport;
}

bool StudyBridge::calibratedForBiplane() const {
    return session_->calibrated_for_biplane_viewport;
}

bool StudyBridge::hasDataset() const {
    return !session_->loaded_frames.empty() || !session_->loaded_models.empty();
}

int StudyBridge::frameCount() const {
    return frame_list_model_->rowCount();
}

int StudyBridge::modelCount() const {
    return model_list_model_->rowCount();
}

int StudyBridge::currentFrame() const {
    return selection_->GetCurrentFrame();
}

int StudyBridge::primaryModelIndex() const {
    return selection_->GetPrimaryModelIndex();
}

int StudyBridge::selectedModelCount() const {
    return selection_->GetSelectedModelCount();
}

QVariantList StudyBridge::selectedModels() const {
    QVariantList rows;
    for (int row : selection_->GetSelectedModelRows()) {
        rows.push_back(row);
    }
    return rows;
}

FrameListModel* StudyBridge::frameListModel() {
    return frame_list_model_;
}

ModelListModel* StudyBridge::modelListModel() {
    return model_list_model_;
}

/*---- Private mirrors ----*/

/*The widgets SyncSessionState() tail (mainscreen.cpp:96) via the shared
 * session controller (plan 006 U6): the controller diffs the plain facts
 * and emits datasetChanged/selectionChanged only on actual change (M9);
 * the previous mirrors advance to the now-current selection (H2 steady
 * state) through CommitSelection. The QML side has no save-last-pose
 * between the sync and the mirrors (its save-last-pose mirror lives in
 * OptimizerBridge's run request), so the two calls are adjacent.*/
void StudyBridge::syncSessionState() {
    session_state_controller_->UpdateSession(
        frame_list_model_->rowCount(),
        model_list_model_->rowCount(),
        selection_->GetCurrentFrame(),
        selection_->GetSelectedModelRows());
    session_state_controller_->CommitSelection();
}

/*The hub's headline counts (the QML "Frames (n) / Models (n)" labels).*/
void StudyBridge::syncHubCounts() {
    hub_->setFrameCount(frame_list_model_->rowCount());
    hub_->setModelCount(model_list_model_->rowCount());
}

/*Scene = the current frame's ORIGINAL image (the display mode is applied at
 * render time, widgets update_display_background_to_* semantics). Empty when
 * no frame is current — the renderer keeps the last background and the QML
 * placeholder overlay covers the viewport (well-defined empty state).*/
void StudyBridge::updateSceneBackground() {
    const int frame = selection_->GetCurrentFrame();
    if (frame >= 0 &&
        frame < static_cast<int>(session_->loaded_frames.size())) {
        scene_->setBackgroundImage(
            session_->loaded_frames[static_cast<size_t>(frame)]
                .GetOriginalImage());
    } else {
        scene_->setBackgroundImage(cv::Mat());
    }
}

/*Scene models at their stored poses (LocationStorage::GetPose — the default
 * pose for the current frame, or the no-image pose vector when no frame is
 * current).*/
void StudyBridge::updateSceneModels() {
    std::vector<SceneModel> scene_models;
    scene_models.reserve(session_->loaded_models.size());
    const int frame = selection_->GetCurrentFrame();
    for (size_t i = 0; i < session_->loaded_models.size(); ++i) {
        const Model& model = session_->loaded_models[i];
        scene_models.push_back(SceneModel{
            model.file_location_, model.model_name_,
            session_->model_locations.GetPose(frame, static_cast<int>(i))});
    }
    scene_->setModels(scene_models);
}

/*Scene camera from the calibration: focal = principal distance (set at
 * calibration load); view angle from the first frame's dims (set at image
 * load — the widgets sets the angle at frame selection).*/
void StudyBridge::updateSceneCamera() {
    if (!hasCalibration() || session_->loaded_frames.empty()) {
        return;
    }
    const Calibration& cal = session_->calibration_file;
    const cv::Mat& image = session_->loaded_frames.front().GetOriginalImage();
    scene_->setFocalLengthPx(cal.camera_A_principal_.principal_distance_);
    scene_->setCameraViewAngle(
        ViewingAngleForFrame(cal, image.cols, image.rows));
}
