// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

/*StudyLoadController implementation (plan 006 U7 / R11 + R13). Every body
 * here is relocated verbatim from the widgets load slots (src/view/
 * mainscreen.cpp) — the per-cut R13 gate: the parse -> populate -> dedup ->
 * counts orchestration moved byte-identical over the plan-004 U6
 * SessionController primitives; only the widget calls (dialogs, error
 * boxes, setDisabled/setChecked, interactor.h global writes, VTK/scene
 * setup) stayed in the views. The calibration one-use-per-session and
 * dataset-replace (calibration kept) semantics are the spec, preserved
 * exactly — they derive from the caller-owned containers, so a dataset
 * replace (which keeps the calibration) keeps the one-use rejection across
 * studies.*/

#include "services/study_load_controller.h"

#include <utility>  // std::move

namespace jta {

StudyLoadController::StudyLoadController(
    SessionController* session_controller,
    std::function<bool()> run_in_flight)
    : session_controller_(session_controller),
      run_in_flight_(std::move(run_in_flight)) {}

StudyCalibrationLoadResult StudyLoadController::LoadCalibration(
    const QString& file_path,
    Calibration& calibration,
    bool& calibrated_for_monoplane_viewport,
    bool& calibrated_for_biplane_viewport) {
    StudyCalibrationLoadResult result;

    /*Run-in-flight guard (L17): the views wire the probe from their
     * SessionStateController::runInFlight() (M7). Rejection is silent — the
     * widgets' DisableAll covers the load buttons during a run (the shared
     * check is defense-in-depth); the QML app has no load-time guard today
     * (no new user-visible behavior).*/
    if (RunInFlight()) {
        result.status = StudyLoadStatus::RunInFlight;
        return result;
    }

    /*Calibration one-use per session (widgets parity: the load-calibration
     * button disables after a successful load). The rule derives from the
     * caller-owned calibrated flags, so a dataset replace (calibration
     * kept) keeps the rejection across studies. Silent in both views.*/
    if (calibrated_for_monoplane_viewport ||
        calibrated_for_biplane_viewport) {
        result.status = StudyLoadStatus::CalibrationAlreadyLoaded;
        return result;
    }

    /*Parse the calibration file (the plan-004 U6 / R6 SessionController
     * primitive; the inline QTextStream + QRegularExpression parsing moved
     * there verbatim before this unit).*/
    result.parse = session_controller_->ParseCalibration(file_path);

    /*Error-path writes relocated verbatim from the widgets slot: the
     * PixelSizeZero / InvalidCode boxes set both calibrated flags false
     * (reachable only while nothing is calibrated, so the writes are
     * no-ops in practice — the widgets' literal behavior, preserved);
     * FileOpenFailed changes nothing (the slot's silent open guard).*/
    if (result.parse.error == CalibrationParseResult::Error::PixelSizeZero ||
        result.parse.error == CalibrationParseResult::Error::InvalidCode) {
        calibrated_for_monoplane_viewport = false;
        calibrated_for_biplane_viewport = false;
        result.status = StudyLoadStatus::CalibrationParseError;
        return result;
    }
    if (!result.parse.ok) {
        result.status = StudyLoadStatus::CalibrationParseError;
        return result;
    }

    /*Initialize Calibration (the caller-owned containers).*/
    calibration = result.parse.calibration;
    calibrated_for_monoplane_viewport =
        result.parse.calibrated_for_monoplane_viewport;
    calibrated_for_biplane_viewport =
        result.parse.calibrated_for_biplane_viewport;
    /*The camera A radio is checked after any valid calibration (active-
     * camera mirror, R10) — on the SHARED SessionController instance the
     * view's camera slots also write (the mirrors must agree).*/
    session_controller_->SetActiveCamera(ActiveCamera::CameraA);
    result.status = StudyLoadStatus::Ok;
    return result;
}

StudyImageLoadResult StudyLoadController::LoadImages(
    const QStringList& paths,
    const ImageLoadParams& params,
    std::vector<Frame>& frames,
    LocationStorage& locations) {
    StudyImageLoadResult result;

    /*Run-in-flight guard (L17): see LoadCalibration. The load is rejected
     * before any parse/populate — the dataset is never mutated mid-run.*/
    if (RunInFlight()) {
        result.status = StudyLoadStatus::RunInFlight;
        return result;
    }

    /*Parse + populate (the plan-004 U6 / R6 primitive): the Frame
     * construction, all-same-size check, append, and LoadNewFrame sizing
     * (goto stop semantics: on a size mismatch the frames appended so far
     * persist; the view shows the box after appending the returned names,
     * same as the in-loop box).*/
    const ImageLoadResult load =
        session_controller_->ParseImages(paths, params, frames, locations);
    result.frame_names = load.frame_names;
    if (load.status == ImageLoadStatus::SizeMismatchAborted) {
        result.status = StudyLoadStatus::SizeMismatchAborted;
    }
    /*Counts (the load slots' sync-tail facts): the shared SessionController
     * mirrors, post-load. The views forward the equivalent facts to their
     * U6 SessionStateController through their own sync tails (which read
     * their list models after appending the returned names — identical
     * values; see the header for the least-coupled-shape rationale).*/
    result.frame_count = session_controller_->GetFrameCount();
    result.model_count = session_controller_->GetModelCount();
    return result;
}

StudyImageLoadResult StudyLoadController::LoadBiplaneImages(
    const QStringList& paths_a,
    const QStringList& paths_b,
    const ImageLoadParams& params,
    std::vector<Frame>& frames_a,
    std::vector<Frame>& frames_b,
    LocationStorage& locations) {
    StudyImageLoadResult result;

    if (RunInFlight()) {
        result.status = StudyLoadStatus::RunInFlight;
        return result;
    }

    /*Parse + populate (the plan-004 U6 / R6 primitive): the A/B same-count
     * gate, per-list all-same-size checks, append, and LoadNewFrame sizing
     * (goto stop_biplane semantics: frames appended so far persist; a
     * count mismatch appends nothing — the view shows the same-count box
     * and returns).*/
    const ImageLoadResult load = session_controller_->ParseBiplaneImages(
        paths_a, paths_b, params, frames_a, frames_b, locations);
    result.frame_names = load.frame_names;
    if (load.status == ImageLoadStatus::CameraCountMismatch) {
        result.status = StudyLoadStatus::CameraCountMismatch;
    } else if (load.status == ImageLoadStatus::SizeMismatchAborted) {
        result.status = StudyLoadStatus::SizeMismatchAborted;
    }
    result.frame_count = session_controller_->GetFrameCount();
    result.model_count = session_controller_->GetModelCount();
    return result;
}

StudyModelLoadResult StudyLoadController::LoadModels(
    const QStringList& paths,
    const Calibration& calibration,
    std::vector<Model>& models,
    LocationStorage& locations,
    const std::function<QVector<QString>(const QVector<QString>&)>&
        append_names) {
    StudyModelLoadResult result;

    if (RunInFlight()) {
        result.status = StudyLoadStatus::RunInFlight;
        return result;
    }

    /*For Each Cad File Extension Create Model Name — relocated verbatim
     * from the widgets load-model slot (R13): the base-name computation
     * (path parsing) via the plan-004 U6 ParseModels primitive; the two-pass
     * dedup through the injected append_names seam (the view-model OWNS the
     * loaded display names; the ModelListBuilder mutated-name-rescan quirk
     * is preserved inside it). The returned unique names drive the Model
     * names AND the view's renderer binding.*/
    const std::vector<ParsedModel> parsed_models =
        session_controller_->ParseModels(paths);
    QVector<QString> base_names;
    base_names.reserve(static_cast<int>(parsed_models.size()));
    for (const auto& parsed_model : parsed_models) {
        base_names.push_back(parsed_model.base_name);
    }
    const QVector<QString> unique_names = append_names(base_names);

    QStringList model_names;
    model_names.reserve(static_cast<int>(unique_names.size()));
    for (const auto& n : unique_names) {
        model_names.push_back(n);
    }
    /*The view's renderer-binding inputs (the widgets' vw->load_models
     * arguments — the raw file paths + the deduped display names).*/
    for (const auto& parsed_model : parsed_models) {
        result.file_paths.push_back(parsed_model.file_path);
    }
    result.unique_names = model_names;

    /*Dataset population (the plan-004 U6 / R6 primitive): Model
     * construction (the STL parse) + LocationStorage sizing via
     * LoadNewModel, moved verbatim.*/
    session_controller_->PopulateModels(
        parsed_models, model_names, calibration, models, locations);

    result.frame_count = session_controller_->GetFrameCount();
    result.model_count = session_controller_->GetModelCount();
    result.status = StudyLoadStatus::Ok;
    return result;
}

}  // namespace jta
