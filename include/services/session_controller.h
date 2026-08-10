/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

/*SessionController (plan 004 U6 / R6 + R10): headless owner of the load path
 * (calibration / image / model file parsing + dataset population) and the
 * camera A/B switching state, previously inline in the MainScreen load and
 * camera slots.
 *
 * The controller is widget-free: it never touches QWidgets, VTK, or the
 * interactor.h file-scope globals (interactor_calibration /
 * interactor_camera_B stay one-TU view writes; no second TU may include
 * interactor.h). It operates on the view's dataset by reference -- the view
 * keeps ownership of loaded_frames / loaded_models / model_locations_ /
 * calibration_file_ (the pose slots, DRR, and VTK bindings read them
 * directly) -- and the view orchestrates: dialogs -> controller parse /
 * populate -> view-model insertion (AppendFrame / AppendModels dedup, R5) ->
 * VTK.
 *
 * The controller mirrors the dataset counts (the load slots'
 * SyncSessionState() tail, reproduced headlessly; the view keeps its
 * widget-read SyncSessionState() calls) and the active-camera enum (a mirror
 * of the camera radios -- the RADIO remains the runtime source of truth;
 * isChecked() reads stay in the slots).*/

#ifndef SESSION_CONTROLLER_H
#define SESSION_CONTROLLER_H

/*QtCore (no widgets)*/
#include <QString>
#include <QStringList>

/*Standard*/
#include <vector>

/*Dataset + storage types (the controller operates on the view's instances)*/
#include "compute/frame.h"
#include "services/calibration.h"
#include "services/location_storage.h"
#include "services/model.h"

namespace jta {

/*Active camera mirror (R10). The radios remain the runtime source of truth;
 * the slots call SetActiveCamera next to their interactor_camera_B writes.*/
enum class ActiveCamera { CameraA, CameraB };

/*Camera radio enable/disable actions decided by the controller (R10). The
 * view applies them with setEnabled(); the setChecked(true) calls (radio
 * checks, not enable/disable) stay in the view.*/
struct CameraRadioActions {
    bool enable_camera_a = false;
    bool enable_camera_b = false;
};

/*The camera-radio decision context (R10): what the radios should be after a
 * calibration load, or after the user switches camera. Pure function of the
 * calibrated viewport flags -- see DecideCameraRadios.*/
enum class CameraRadioEvent { CalibrationLoaded, SwitchToCameraA, SwitchToCameraB };

/*Calibration parse outcome (R6). ok == a valid calibration was parsed and
 * the view should proceed with the VTK setup; the error kind selects the
 * view's error box (FileOpenFailed shows none, exactly like the slot's
 * silent open guard). kind distinguishes the three valid formats because the
 * view's interactor.h global writes are branch-dependent: Monoplane writes
 * interactor_calibration AND interactor_camera_B = false; Biplane writes
 * only interactor_calibration; Denver writes neither (preserved verbatim).*/
struct CalibrationParseResult {
    enum class Error { None, FileOpenFailed, PixelSizeZero, InvalidCode };
    enum class Kind { None, Monoplane, Biplane, Denver };
    bool ok = false;
    Error error = Error::None;
    Kind kind = Kind::None;
    Calibration calibration;
    bool calibrated_for_monoplane_viewport = false;
    bool calibrated_for_biplane_viewport = false;
};

/*Aperture/low/high/dilation values the view passes in with each image load
 * (the view reads the widgets + the active cost function's Dilation param).*/
struct ImageLoadParams {
    int aperture = 0;
    int low_threshold = 0;
    int high_threshold = 0;
    int dilation = 0;
};

enum class ImageLoadStatus {
    Completed,
    SizeMismatchAborted, /* goto stop / stop_biplane: frames appended so far persist */
    CameraCountMismatch, /* biplane A/B list length mismatch: nothing appended */
};

struct ImageLoadResult {
    ImageLoadStatus status = ImageLoadStatus::Completed;
    /*Display names of the frames appended, in order (the view's
     * FrameListModel::AppendFrame inputs; biplane names are the
     * "A: <base>\nB: <base>" form). Fewer than the input paths when a size
     * mismatch aborted the list.*/
    QStringList frame_names;
};

/*Parsed model metadata (R6): the file path + its pre-dedup display base
 * name (QFileInfo::baseName), exactly as the load-model slot computed them.*/
struct ParsedModel {
    QString file_path;
    QString base_name;
};

class SessionController {
public:
    /*---- Calibration parsing (R6) ----*/
    static CalibrationParseResult ParseCalibration(const QString& file_path);

    /*---- Image parsing + dataset population (R6) ----*/
    /*Monoplane: construct the Frames (imread + edge pipeline), enforce the
     * all-same-size rule against the frames appended so far, append, and
     * size the LocationStorage via LoadNewFrame(). A size mismatch aborts
     * mid-list (goto stop semantics -- frames appended so far persist); the
     * caller shows the error box. The dataset vectors/storage belong to the
     * view and are mutated in place; the returned frame_names are the
     * parsed display names for the view's AppendFrame calls.*/
    ImageLoadResult ParseImages(const QStringList& paths,
                                const ImageLoadParams& params,
                                std::vector<Frame>& frames,
                                LocationStorage& locations);
    /*Biplane twin: the A/B lists must be the same length (CameraCountMismatch
     * -- nothing is appended), each A frame is checked against the A list and
     * each B frame against the B list before either is appended (goto
     * stop_biplane semantics).*/
    ImageLoadResult ParseBiplaneImages(const QStringList& paths_a,
                                       const QStringList& paths_b,
                                       const ImageLoadParams& params,
                                       std::vector<Frame>& frames_a,
                                       std::vector<Frame>& frames_b,
                                       LocationStorage& locations);

    /*---- Model parsing + dataset population (R6) ----*/
    static std::vector<ParsedModel> ParseModels(const QStringList& paths);
    /*Construct the Model objects (the STL parse the load slot's
     * loaded_models.push_back(Model(...)) did) and size the LocationStorage
     * via LoadNewModel(calibration). unique_names are the deduped display
     * names from the view's ModelListModel::AppendModels (R5) -- loading the
     * same file list twice behaves exactly like today (duplicate model names
     * go through the ModelListBuilder dedup).*/
    void PopulateModels(const std::vector<ParsedModel>& parsed_models,
                        const QStringList& unique_names,
                        const Calibration& calibration,
                        std::vector<Model>& models,
                        LocationStorage& locations);

    /*---- Camera A/B state (R10) ----*/
    void SetActiveCamera(ActiveCamera camera);
    ActiveCamera GetActiveCamera() const;

    /*The camera radio enable/disable decision, a pure function of the
     * calibrated viewport flags:
     *  - CalibrationLoaded: monoplane -> both radios disabled (A stays
     *    checked); biplane -> both enabled (A stays checked); neither flag
     *    -> both disabled (not applied: the view returns after the error
     *    box).
     *  - SwitchToCameraA: biplane -> A disabled, B enabled; monoplane -> no
     *    action (the camera-A slot's biplane guard, preserved -- the caller
     *    applies the decision only when the guard holds).
     *  - SwitchToCameraB: -> B disabled, A enabled (unconditional, exactly
     *    like the camera-B slot today).*/
    static CameraRadioActions DecideCameraRadios(CameraRadioEvent event,
                                                 bool calibrated_for_monoplane_viewport,
                                                 bool calibrated_for_biplane_viewport);

    /*---- Session mirror (the load slots' SyncSessionState() tail) ----*/
    /*Dataset counts the controller refreshes at the end of each load/populate
     * op -- the headless reproduction of the slots' SyncSessionState() tail
     * (the view keeps its widget-read SyncSessionState() calls).*/
    int GetFrameCount() const;
    int GetModelCount() const;

private:
    void SyncSessionState(const std::vector<Frame>& frames,
                          LocationStorage& locations);
    void SyncSessionState(const std::vector<Model>& models);

    ActiveCamera active_camera_ = ActiveCamera::CameraA;
    int frame_count_ = 0;
    int model_count_ = 0;
};

}  // namespace jta

#endif  // SESSION_CONTROLLER_H
