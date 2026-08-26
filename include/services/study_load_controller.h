// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

/*StudyLoadController (plan 006 U7 / R11 + R13): the ONE shared study-load
 * path for BOTH front-ends, over SessionController parsing + caller-owned
 * containers. The widgets load slots (mainscreen.cpp
 * on_load_calibration_button_clicked / on_load_image_button_clicked /
 * on_load_model_button_clicked) and StudyBridge thin onto it; the scene/
 * background/VTK update orchestration STAYS view-side — each view maps the
 * load result onto its own renderer.
 *
 * Every body here is relocated verbatim from the widgets load slots (R13:
 * pure move — parse -> populate -> dedup -> counts -> sync tail); only the
 * widget calls (dialogs, error boxes, setDisabled/setChecked, interactor.h
 * global writes, VTK/scene setup) stayed in the views. The calibration
 * one-use-per-session and dataset-replace (calibration kept) semantics are
 * load POLICY and live here, preserved exactly.
 *
 * Layering:
 *  - The controller WRAPS the view's SessionController (non-owning
 *    pointer): the parsing seam + the shared active-camera / dataset-count
 *    mirrors stay on the ONE instance the widgets camera slots also use
 *    (the calibration load's SetActiveCamera(CameraA) must agree with the
 *    camera slots' writes).
 *  - The run-in-flight guard (L17) is an injected std::function<bool()>
 *    probe (default: no run in flight). The views wire it from their
 *    SessionStateController::runInFlight() (M7). Rejection is SILENT in
 *    both views: the widgets' DisableAll covers the load buttons during a
 *    run (unreachable — the shared check is defense-in-depth); the QML app
 *    has no load-time guard today, so the shared check lands without new
 *    user-visible behavior. Services never reference coordinator (layering).
 *  - The calibration-required guard ("Load Calibration First!") stays in
 *    the VIEW slots: the widgets checks BEFORE its file dialog, so it
 *    cannot move into the shared load path without changing dialog order.
 *  - The sync tail: each result carries the post-load counts as FACTS
 *    (the wrapped SessionController's mirrors). The views forward the
 *    equivalent facts to their U6 SessionStateController through their own
 *    sync tails (widgets SyncSessionState; StudyBridge syncSessionState +
 *    syncHubCounts), which read their list models AFTER appending the
 *    returned names — the values are identical to the facts, and a
 *    controller-fired callback would necessarily precede those appends
 *    (stale list-model counts), so returning facts is the least-coupled
 *    shape.
 *
 * View-agnostic (R15): caller-owned containers + plain paths + the injected
 * probe; no view/scene/renderer pointers anywhere; drivable headless under
 * QCoreApplication.
 */

#ifndef STUDY_LOAD_CONTROLLER_H
#define STUDY_LOAD_CONTROLLER_H

/*QtCore (no widgets)*/
#include <QString>
#include <QStringList>
#include <QVector>

/*Standard*/
#include <functional>
#include <vector>

/*Dataset + storage types (the controller operates on the view's instances)*/
#include "compute/frame.h"
#include "services/calibration.h"
#include "services/location_storage.h"
#include "services/model.h"
#include "services/session_controller.h"

namespace jta {

/*Load outcome (policy + parse status). The view maps each status onto its
 * own error surface; CalibrationParseError carries the full parse details
 * in the result's parse member (the views' box logic keys on the parse
 * error kind, preserved verbatim).*/
enum class StudyLoadStatus {
    Ok,
    /*Calibration one-use per session: a calibration is already loaded (the
     * caller-owned flags say so — the rule survives a dataset replace,
     * which keeps the calibration). Silent in both views (the widgets
     * disables the button; the QML guard returns).*/
    CalibrationAlreadyLoaded,
    /*The calibration parse failed; the result's parse member carries the
     * error kind (PixelSizeZero / InvalidCode boxes, FileOpenFailed silent
     * — the widgets' exact error mapping). The PixelSizeZero / InvalidCode
     * paths also wrote both calibrated flags false (the slots' error-path
     * writes, relocated). No dataset writes happened.*/
    CalibrationParseError,
    /*Image load aborted mid-list on an all-same-size violation: the frames
     * appended so far persist (goto stop / stop_biplane semantics — the
     * caller shows the size-mismatch box after appending the returned
     * names).*/
    SizeMismatchAborted,
    /*Biplane A/B list-length mismatch: nothing was appended (the caller
     * shows the same-count box and returns).*/
    CameraCountMismatch,
    /*The injected run-in-flight probe (L17) reported a run in progress:
     * the load was rejected before any parse/populate, nothing changed.*/
    RunInFlight,
};

/*Calibration load outcome: status + the full parse result (error kind,
 * branch kind, calibration, viewport flags — the view's interactor.h
 * global writes and radio decisions are branch-dependent, preserved).*/
struct StudyCalibrationLoadResult {
    StudyLoadStatus status = StudyLoadStatus::Ok;
    CalibrationParseResult parse;
};

/*Image load outcome: the parsed display names (the view's AppendFrame
 * inputs; fewer than the input paths when a size mismatch aborted the
 * list) + the post-load counts (the sync-tail facts).*/
struct StudyImageLoadResult {
    StudyLoadStatus status = StudyLoadStatus::Ok;
    QStringList frame_names;
    int frame_count = 0;
    int model_count = 0;
};

/*Model load outcome: the parsed file paths + the deduped unique display
 * names (the view's renderer-binding inputs — the same names that were
 * appended to its list model through the dedup seam) + the post-load
 * counts.*/
struct StudyModelLoadResult {
    StudyLoadStatus status = StudyLoadStatus::Ok;
    QStringList file_paths;
    QStringList unique_names;
    int frame_count = 0;
    int model_count = 0;
};

class StudyLoadController {
public:
    /*Wraps the view's SessionController (non-owning; must outlive the load
     * controller). The optional run-in-flight probe (L17) is wired by the
     * composition roots (the views' SessionStateController::runInFlight,
     * M7); it is consulted at each load and never during construction.*/
    explicit StudyLoadController(
        SessionController* session_controller,
        std::function<bool()> run_in_flight = {});

    /*Calibration load: one-use-per-session rejection (from the caller-owned
     * calibrated flags), then ParseCalibration + the caller-owned container
     * writes (calibration + both flags; the PixelSizeZero / InvalidCode
     * error paths write both flags false exactly like the widgets slot,
     * FileOpenFailed changes nothing), then the shared active-camera mirror
     * (CameraA — the radio is checked after any valid calibration).*/
    StudyCalibrationLoadResult LoadCalibration(
        const QString& file_path,
        Calibration& calibration,
        bool& calibrated_for_monoplane_viewport,
        bool& calibrated_for_biplane_viewport);

    /*Monoplane image load: SessionController::ParseImages into the
     * caller-owned frames + locations (goto stop partial-load semantics —
     * the frames appended so far persist on a size mismatch; the returned
     * names are exactly the appended frames' display names).*/
    StudyImageLoadResult LoadImages(
        const QStringList& paths,
        const ImageLoadParams& params,
        std::vector<Frame>& frames,
        LocationStorage& locations);

    /*Biplane twin: ParseBiplaneImages (same-count gate, per-list size
     * checks, goto stop_biplane semantics).*/
    StudyImageLoadResult LoadBiplaneImages(
        const QStringList& paths_a,
        const QStringList& paths_b,
        const ImageLoadParams& params,
        std::vector<Frame>& frames_a,
        std::vector<Frame>& frames_b,
        LocationStorage& locations);

    /*Model load: ParseModels -> two-pass dedup through the injected
     * append_names seam -> PopulateModels into the caller-owned models +
     * locations. append_names is the view's ModelListModel::AppendModels
     * (the view-model OWNS the loaded display names; the returned unique
     * names drive the Model names AND the view's renderer binding — the
     * ModelListBuilder mutated-name-rescan quirk is preserved inside it).*/
    StudyModelLoadResult LoadModels(
        const QStringList& paths,
        const Calibration& calibration,
        std::vector<Model>& models,
        LocationStorage& locations,
        const std::function<QVector<QString>(const QVector<QString>&)>&
            append_names);

private:
    bool RunInFlight() const {
        return run_in_flight_ ? run_in_flight_() : false;
    }

    SessionController* session_controller_ = nullptr;
    std::function<bool()> run_in_flight_;
};

}  // namespace jta

#endif  // STUDY_LOAD_CONTROLLER_H
