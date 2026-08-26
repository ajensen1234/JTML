/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

/*SessionController implementation (plan 004 U6 / R6 + R10). Every body here
 * is relocated verbatim from the MainScreen load/camera slots (src/view/
 * mainscreen.cpp) -- the per-cut R13 gate: the parsing, dataset population,
 * partial-load (goto stop / stop_biplane) semantics, and the camera radio
 * decision moved byte-identical; only the widget calls (dialogs, error
 * boxes, setChecked/setEnabled, interactor.h global writes, VTK) stayed in
 * the view.*/

#include "services/session_controller.h"

/*QtCore file/text parsing (no widgets)*/
#include <QFile>
#include <QFileInfo>
#include <QRegularExpression>
#include <QTextStream>

/*All-same-size image gate*/
#include "domain/model_list_builder.h"

namespace jta {

/*---- Calibration parsing (R6) -------------------------------------------*/

CalibrationParseResult SessionController::ParseCalibration(
    const QString& file_path) {
    CalibrationParseResult result;
    QFile inputFile(file_path);
    if (inputFile.open(QIODevice::ReadOnly)) {
        QTextStream in(&inputFile);
        QStringList InputList = in.readAll().split(
            QRegularExpression("[\\r\\n]|,|\\t| "), Qt::SkipEmptyParts);

        /*Valid Code for Monoplane*/
        if (InputList[0] == "JT_INTCALIB" || InputList[0] == "JTA_INTCALIB") {
            /*Error Check*/
            if (InputList[4].toDouble() == 0) {
                result.error = CalibrationParseResult::Error::PixelSizeZero;
                inputFile.close();
                return result;
            }

            /*Initialize Calibration*/
            result.calibrated_for_monoplane_viewport = true;
            result.calibrated_for_biplane_viewport = false;
            CameraCalibration principal_calibration_file(
                InputList[1].toDouble(),
                -1 * InputList[2].toDouble(),
                // Negative For Offsets to make consistent with JointTrack
                -1 * InputList[3].toDouble(),
                InputList[4].toDouble());
            float* prin_dist_ = &principal_calibration_file.principal_distance_;
            result.calibration = Calibration(principal_calibration_file);
            Calibration* cal_pointer_ = &result.calibration;
            result.kind = CalibrationParseResult::Kind::Monoplane;
            result.ok = true;
        }
        /*Valid Code for Biplane*/
        /*NOT WORKING, BUT GOOD STARTING PLACE*/
        else if (InputList[0] == "JTA_INTCALIB_BIPLANE") {
            /*Convert and Do PIX MM Error CHECK*/
            /*Error Check*/
            if (InputList[4].toDouble() == 0 || InputList[8].toDouble() == 0) {
                result.error = CalibrationParseResult::Error::PixelSizeZero;
                inputFile.close();
                return result;
            }
            /*Initialize Calibrations*/
            result.calibrated_for_monoplane_viewport = false;
            result.calibrated_for_biplane_viewport = true;
            /*Calibrate for Main View (A) and alternate view (B).
            Read in (x,y,z) displacement vector from origin (where A is) to
            origin of camera B. Read in othroogonal axis matrix for camera B
            (A is taken to be standard basis vectors)*/
            CameraCalibration principal_calibration_file_A(
                InputList[1].toDouble(),
                -1 * InputList[2].toDouble(),
                -1 * InputList[3].toDouble(),
                InputList[4].toDouble());
            // Negatives to make consistent with JT
            CameraCalibration principal_calibration_file_B(
                InputList[5].toDouble(),
                -1 * InputList[6].toDouble(),
                -1 * InputList[7].toDouble(),
                InputList[8].toDouble());
            Vect_3 origin_B(
                InputList[9].toDouble(),
                InputList[10].toDouble(),
                InputList[11].toDouble());
            Matrix_3_3 orthogonal_axes_B(
                InputList[12].toDouble(),
                InputList[13].toDouble(),
                InputList[14].toDouble(),
                InputList[15].toDouble(),
                InputList[16].toDouble(),
                InputList[17].toDouble(),
                InputList[18].toDouble(),
                InputList[19].toDouble(),
                InputList[20].toDouble());
            result.calibration = Calibration(
                principal_calibration_file_A,
                principal_calibration_file_B,
                origin_B,
                orthogonal_axes_B);
            result.kind = CalibrationParseResult::Kind::Biplane;
            result.ok = true;
        } else if (InputList[0] == "image") {  // Would need a way to
                                               // distinguish Denver single
                                               // plane from biplane
            CameraCalibration denver_calibration_A(
                InputList[6].toDouble(),
                InputList[7].toDouble(),
                InputList[8].toDouble(),
                InputList[10].toDouble(),
                InputList[11].toDouble());

            result.calibrated_for_monoplane_viewport = true;
            result.calibrated_for_biplane_viewport = false;
            result.calibration = Calibration(denver_calibration_A, "Denver");
            result.kind = CalibrationParseResult::Kind::Denver;
            result.ok = true;
        }
        /*Invalid Code*/
        else {
            result.error = CalibrationParseResult::Error::InvalidCode;
            inputFile.close();
            return result;
        }
        inputFile.close();
    } else {
        /*Open failure: silent in the view (the slot's open guard showed no
         * box and changed nothing) -- the error kind documents it.*/
        result.error = CalibrationParseResult::Error::FileOpenFailed;
    }
    return result;
}

/*---- Image parsing + dataset population (R6) -----------------------------*/

ImageLoadResult SessionController::ParseImages(
    const QStringList& paths,
    const ImageLoadParams& params,
    std::vector<Frame>& frames,
    LocationStorage& locations) {
    ImageLoadResult result;
    for (int i = 0; i < paths.size(); i++) {
        auto new_frame = Frame(
            paths[i].toStdString(),
            params.aperture,
            params.low_threshold,
            params.high_threshold,
            params.dilation);
        /*Check That All Frames Are The Same Size and Not Empty*/
        int width = new_frame.GetEdgeImage().cols;
        int height = new_frame.GetEdgeImage().rows;
        std::vector<std::pair<int, int>> loaded_sizes;
        loaded_sizes.reserve(frames.size());
        for (auto& f : frames) {
            loaded_sizes.emplace_back(
                f.GetEdgeImage().cols, f.GetEdgeImage().rows);
        }
        if (!jta::ModelListBuilder::AllSameSize(width, height, loaded_sizes)) {
            /*goto stop: abort mid-list; the frames appended so far persist
             * (the caller shows the size-mismatch box).*/
            result.status = ImageLoadStatus::SizeMismatchAborted;
            goto stop;
        }
        // Add to Loaded Frames
        frames.push_back(new_frame);
        // Populate Frame List Widget (parsed display name for the view's
        // AppendFrame)
        result.frame_names.push_back(QFileInfo(paths[i]).baseName());
        /*Add Blank Model Locations for Loaded Models*/
        locations.LoadNewFrame();
    }
    /*Exit Label*/
stop:;
    /*The load slots' SyncSessionState() tail, reproduced headlessly (the
     * view keeps its widget-read SyncSessionState() calls).*/
    SyncSessionState(frames, locations);
    return result;
}

ImageLoadResult SessionController::ParseBiplaneImages(
    const QStringList& paths_a,
    const QStringList& paths_b,
    const ImageLoadParams& params,
    std::vector<Frame>& frames_a,
    std::vector<Frame>& frames_b,
    LocationStorage& locations) {
    ImageLoadResult result;
    /*Check Same Amount of Loaded Images*/
    if (paths_a.size() != paths_b.size()) {
        result.status = ImageLoadStatus::CameraCountMismatch;
        return result;
    }

    for (int i = 0; i < paths_a.size(); i++) {
        auto new_frame_A = Frame(
            paths_a[i].toStdString(),
            params.aperture,
            params.low_threshold,
            params.high_threshold,
            params.dilation);
        auto new_frame_B = Frame(
            paths_b[i].toStdString(),
            params.aperture,
            params.low_threshold,
            params.high_threshold,
            params.dilation);
        /*Check That All Camera-A Frames Are The Same Size and Not Empty*/
        int widthA = new_frame_A.GetEdgeImage().cols;
        int heightA = new_frame_A.GetEdgeImage().rows;
        std::vector<std::pair<int, int>> sizesA;
        sizesA.reserve(frames_a.size());
        for (auto& f : frames_a) {
            sizesA.emplace_back(f.GetEdgeImage().cols, f.GetEdgeImage().rows);
        }
        if (!jta::ModelListBuilder::AllSameSize(widthA, heightA, sizesA)) {
            /*goto stop_biplane: abort mid-list; the frames appended so far
             * persist (the caller shows the size-mismatch box).*/
            result.status = ImageLoadStatus::SizeMismatchAborted;
            goto stop_biplane;
        }
        /*Camera-B frames must also match the loaded B list.*/
        int widthB = new_frame_B.GetEdgeImage().cols;
        int heightB = new_frame_B.GetEdgeImage().rows;
        std::vector<std::pair<int, int>> sizesB;
        sizesB.reserve(frames_b.size());
        for (auto& f : frames_b) {
            sizesB.emplace_back(f.GetEdgeImage().cols, f.GetEdgeImage().rows);
        }
        if (!jta::ModelListBuilder::AllSameSize(widthB, heightB, sizesB)) {
            result.status = ImageLoadStatus::SizeMismatchAborted;
            goto stop_biplane;
        }

        // Add to Loaded Frames
        frames_a.push_back(new_frame_A);
        frames_b.push_back(new_frame_B);
        // Populate Frame List Widget (parsed display name for the view's
        // AppendFrame)
        result.frame_names.push_back(
            "A: " + QFileInfo(paths_a[i]).baseName() +
            "\nB: " + QFileInfo(paths_b[i]).baseName());
        /*Add Blank Model Locations for Loaded Models*/
        locations.LoadNewFrame();
    }
    /*Exit Label*/
stop_biplane:;
    /*The load slots' SyncSessionState() tail, reproduced headlessly (the
     * view keeps its widget-read SyncSessionState() calls).*/
    SyncSessionState(frames_a, locations);
    return result;
}

/*---- Model parsing + dataset population (R6) -----------------------------*/

std::vector<ParsedModel> SessionController::ParseModels(
    const QStringList& paths) {
    std::vector<ParsedModel> parsed;
    parsed.reserve(static_cast<size_t>(paths.size()));
    for (int i = 0; i < paths.size(); i++) {
        parsed.push_back(ParsedModel{paths[i], QFileInfo(paths[i]).baseName()});
    }
    return parsed;
}

void SessionController::PopulateModels(
    const std::vector<ParsedModel>& parsed_models,
    const QStringList& unique_names,
    const Calibration& calibration,
    std::vector<Model>& models,
    LocationStorage& locations) {
    for (int i = 0; i < static_cast<int>(parsed_models.size()); i++) {
        models.push_back(Model(
            parsed_models[i].file_path.toStdString(),
            unique_names[i].toStdString(),
            "BLANK"));
    }
    /*Load Blank Poses for Available Frames (and Default Blank Poses even if
     * no frames for viewing without frame)*/
    for (int i = 0; i < static_cast<int>(parsed_models.size()); i++) {
        locations.LoadNewModel(calibration);
    }
    /*The load slots' SyncSessionState() tail, reproduced headlessly (the
     * view keeps its widget-read SyncSessionState() calls).*/
    SyncSessionState(models);
}

/*---- Camera A/B state (R10) ----------------------------------------------*/

void SessionController::SetActiveCamera(ActiveCamera camera) {
    active_camera_ = camera;
}

ActiveCamera SessionController::GetActiveCamera() const {
    return active_camera_;
}

CameraRadioActions SessionController::DecideCameraRadios(
    CameraRadioEvent event,
    bool calibrated_for_monoplane_viewport,
    bool calibrated_for_biplane_viewport) {
    CameraRadioActions actions;
    switch (event) {
    case CameraRadioEvent::CalibrationLoaded:
        /*Monoplane disables both radios (A stays checked); biplane enables
         * both (A stays checked); neither flag (invalid load) is not applied
         * by the view (it returns after the error box).*/
        if (calibrated_for_monoplane_viewport) {
            actions.enable_camera_a = false;
            actions.enable_camera_b = false;
        } else if (calibrated_for_biplane_viewport) {
            actions.enable_camera_a = true;
            actions.enable_camera_b = true;
        }
        break;
    case CameraRadioEvent::SwitchToCameraA:
        /*The camera-A slot applies this only under its biplane guard; the
         * monoplane no-op is preserved (the caller does not apply it).*/
        if (calibrated_for_biplane_viewport) {
            actions.enable_camera_a = false;
            actions.enable_camera_b = true;
        }
        break;
    case CameraRadioEvent::SwitchToCameraB:
        /*Unconditional, exactly like the camera-B slot today.*/
        actions.enable_camera_a = true;
        actions.enable_camera_b = false;
        break;
    }
    return actions;
}

/*---- Session mirror (the load slots' SyncSessionState() tail) ------------*/

void SessionController::SyncSessionState(
    const std::vector<Frame>& frames,
    LocationStorage& locations) {
    frame_count_ = static_cast<int>(frames.size());
    model_count_ = locations.GetModelCount();
}

void SessionController::SyncSessionState(const std::vector<Model>& models) {
    model_count_ = static_cast<int>(models.size());
}

int SessionController::GetFrameCount() const {
    return frame_count_;
}

int SessionController::GetModelCount() const {
    return model_count_;
}

}  // namespace jta
